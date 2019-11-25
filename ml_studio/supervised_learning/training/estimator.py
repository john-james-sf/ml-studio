# =========================================================================== #
#                             ESTIMATOR CLASS                                 #
# =========================================================================== #
"""Gradient Descent base class, from which regression and classification inherit."""
from abc import ABC, abstractmethod, ABCMeta
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
import warnings

from ml_studio.utils.data_manager import batch_iterator, data_split, shuffle_data
from ml_studio.supervised_learning.training.callbacks import CallbackList, Callback
from ml_studio.supervised_learning.training.monitor import History, Progress, summary
from ml_studio.supervised_learning.training.learning_rate_schedules import LearningRateSchedule
from ml_studio.supervised_learning.training.early_stop import EarlyStop
from ml_studio.supervised_learning.training.early_stop import EarlyStopImprovement
# --------------------------------------------------------------------------- #

class Estimator(ABC, BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """Base class gradient descent estimator."""

    DEFAULT_METRIC = 'mean_squared_error'

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 early_stop=False, val_size=0.3, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, seed=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.cost = cost
        self.metric = metric
        self.early_stop = early_stop
        self.val_size = val_size
        self.patience = patience
        self.precision = precision
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.name = name
        self.seed = seed
        # Instance variables
        self.epoch = 0
        self.batch = 0
        self.converged = False
        self.theta = None
        self.eta = None                
        self.cbks = None
        self.X = self.y = self.X_val = self.y_val = None
        self.regularizer = lambda x: 0
        self.regularizer.gradient = lambda x: 0
        self.algorithm = None
        # Functions
        self.scorer = None
        self.cost_function = None
        self.convergence_monitor = None
        # Attributes
        self.coef_ = None
        self.intercept_ = None
        self.epochs_ = 0

    def _set_algorithm_name(self):
        """Sets the name of the algorithm for plotting purposes."""
        if self.batch_size is None:
            self.algorithm = 'Batch Gradient Descent'
        elif self.batch_size == 1:
            self.algorithm = 'Stochastic Gradient Descent'
        else:
            self.algorithm = 'Minibatch Gradient Descent'

    @abstractmethod
    def _set_name(self):
        pass

    def set_params(self, **kwargs):
        """Sets parameters to **kwargs and validates."""
        super().set_params(**kwargs)
        self._validate_params()
        return self

    def _validate_params(self):
        """Validate parameters."""
        if not isinstance(self.learning_rate, (int, float, LearningRateSchedule)):
            raise TypeError("learning_rate must provide an int, float or a LearningRateSchedule object.")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, int):
                raise TypeError("batch_size must provide an integer.")            
        if self.theta_init is not None:
            if not isinstance(self.theta_init, (list, pd.core.series.Series, np.ndarray)):
                raise TypeError("theta must be an array like object.")            
        if not isinstance(self.epochs, int):
            raise TypeError("epochs must be an integer.")        
        if self.early_stop:
            if not isinstance(self.early_stop, (bool,EarlyStop)):
                raise TypeError("early stop is not a valid EarlyStop callable.")
        if self.early_stop:
            if not isinstance(self.val_size, float) or self.val_size < 0 or self.val_size >= 1:
                raise ValueError("val_size must be a float between 0 and 1.")
        if not isinstance(self.patience, int):
            raise ValueError("patience must be an integer")
        if not isinstance(self.precision, float) or self.precision < 0 or self.precision >= 1:
            raise ValueError("precision must be a float between 0 and 1.")            
        if self.metric is not None:
            if not isinstance(self.metric, str):
                raise TypeError("metric must be string containing name of metric for scoring")                
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be either True or False")
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, int):
                raise TypeError(
                    "checkpoint must be a positive integer or None.")
            elif self.checkpoint < 0:
                raise ValueError(
                    "checkpoint must be a positive integer or None.")
        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise TypeError("seed must be a positive integer.")

    def _validate_data(self, X, y=None):
        """Validates and reports n_features."""
        if not isinstance(X, (np.ndarray)):
            raise TypeError("X must be of type np.ndarray")
        if y is not None:
            if not isinstance(y, (np.ndarray)):
                raise TypeError("y must be of type np.ndarray")            
            if len(y.shape) > 1:
                raise ValueError("y should be of shape (m,), not %s" % str(y.shape))
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y have incompatible lengths")        

    def _prepare_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        self.X = self.X_val = self.y = self.y_val = None
        # Add a column of ones to train the intercept term
        self.X = np.insert(X, 0, 1, axis=1)  
        self.y = y
        # Set aside val_size training observations for validation set 
        if self.val_size:
            self.X, self.X_val, self.y, self.y_val = \
                data_split(self.X, self.y, 
                test_size=self.val_size, seed=self.seed)

    def _evaluate_epoch(self, log=None):
        """Computes training (and validation) costs and scores."""
        log = log or {}
        # Compute costs 
        y_pred = self._predict(self.X)
        log['train_cost'] = self.cost_function(y=self.y, y_pred=y_pred)
        if self.val_size > 0 and self.early_stop:
            y_pred_val = self._predict(self.X_val)
            log['val_cost'] = self.cost_function(y=self.y_val, y_pred=y_pred_val)        
        # Compute scores 
        if self.metric is not None:            
            log['train_score'] = self.score(self.X, self.y)
            if self.val_size > 0 and self.early_stop:
                log['val_score'] = self.score(self.X_val, self.y_val)        

        return log


    @abstractmethod
    def _get_cost_function(self):
        """Obtains the cost function for the cost parameter."""
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod        
    def _get_scorer(self):
        """Obtains the scoring function for the metric parameter."""
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    def _get_convergence_monitor(self):
        if isinstance(self.early_stop, EarlyStop):
            convergence_monitor = self.early_stop
        else:
            if self.early_stop:
                if self.metric:
                    convergence_monitor = EarlyStopImprovement(monitor='val_score',
                                                              precision=self.precision,
                                                              patience=self.patience)
                else:
                    convergence_monitor = EarlyStopImprovement(monitor='val_cost',
                                                              precision=self.precision,
                                                              patience=self.patience)
            else:
                if self.metric:
                    convergence_monitor = EarlyStopImprovement(monitor='train_score',
                                                              precision=self.precision,
                                                              patience=self.patience)
                else:
                    convergence_monitor = EarlyStopImprovement(monitor='train_cost',
                                                              precision=self.precision,
                                                              patience=self.patience)            
        return convergence_monitor

    def _compile(self):
        """Obtains external objects and add key functions to the log."""
        self.cost_function = self._get_cost_function()
        self.scorer = self._get_scorer()        
        self.convergence_monitor = self._get_convergence_monitor()

    def _init_callbacks(self):
        # Initialize callback list
        self.cbks = CallbackList()        
        # Instantiate monitor callbacks and add to list
        self.history = History()
        self.cbks.append(self.history)
        self.progress = Progress()        
        self.cbks.append(self.progress)
        # Add additional callbacks if available
        if isinstance(self.learning_rate, Callback):
            self.cbks.append(self.learning_rate)
        if isinstance(self.convergence_monitor, Callback):
            self.cbks.append(self.convergence_monitor)
        # Initialize all callbacks.
        self.cbks.set_params(self.get_params())
        self.cbks.set_model(self)

    
    def _init_weights(self):
        """Initializes weights"""        
        if self.theta_init is not None:
            if self.theta_init.shape[0] != self.X.shape[1]:
                raise ValueError("theta_init shape mispatch. Expected shape %s,"
                                 " but theta_init.shape = %s." % ((self.X.shape[1],1),
                                 self.theta_init.shape))
            else:
                self.theta = np.atleast_2d(self.theta_init).reshape(-1,1)
        else:
            n_features = self.X.shape[1]
            np.random.seed(seed=self.seed)
            self.theta = np.random.normal(size=n_features).reshape(-1,1)

    def _set_learning_rate(self):
        if isinstance(self.learning_rate,float):
            self.eta = self.learning_rate
        else:
            self.eta = self.learning_rate.learning_rate

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self.converged = False
        self._validate_params()
        self._validate_data(log.get('X'), log.get('y'))        
        self._prepare_data(log.get('X'), log.get('y'))
        self._set_name()
        self._init_weights()   
        self._compile()
        self._init_callbacks()
        self.cbks.on_train_begin(log)
        self._set_learning_rate()
        

    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self.cbks.on_train_end()
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        self.epochs_ = self.epoch

    def _begin_epoch(self):
        """Increment the epoch count and shuffle the data."""
        self.epoch += 1
        self.X, self.y = shuffle_data(self.X, self.y, seed=self.seed)
        if self.seed:
            self.seed += 1
        self.cbks.on_epoch_begin(self.epoch)

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}
        # Update log with current learning rate and parameters theta
        log['epoch'] = self.epoch
        log['learning_rate'] = self.eta
        log['theta'] = self.theta.copy()     
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch(log)
        # Call 'on_epoch_end' methods on callbacks.
        self.cbks.on_epoch_end(self.epoch, log)

    def _begin_batch(self, log=None):
        self.batch += 1
        self.cbks.on_batch_begin(self.batch)

    def _end_batch(self, log=None):
        self.cbks.on_batch_end(self.batch, log)

    def fit(self, X, y):
        """Trains model until stop condition is met.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        y : numpy array, shape (n_samples,)
            Target values 

        Returns
        -------
        self : returns instance of self.
        """
        train_log = {'X': X, 'y': y}
        self._begin_training(train_log)
        
        while (self.epoch < self.epochs and not self.converged):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X, self.y, batch_size=self.batch_size):

                self._begin_batch()
                # Compute prediction
                y_pred = self._predict(X_batch)
                # Compute costs
                J = self.cost_function(
                    y=y_batch, y_pred=y_pred) + self.regularizer(self.theta)
                # Update batch log with weights and cost
                batch_log = {'batch': self.batch, 'batch_size': X_batch.shape[0],
                             'theta': self.theta.copy(), 'train_cost': J}
                # Compute gradient 
                gradient = self.cost_function.gradient(
                    X_batch, y_batch, y_pred) - self.regularizer.gradient(self.theta)
                # Update parameters              
                self.theta -= self.eta * gradient
                # Update batch log
                self._end_batch(batch_log)

            # Wrap up epoch
            self._end_epoch()

        self._end_training()
        return self
    
    def _linear_prediction(self, X):
        """Computes prediction as linear combination of inputs and thetas."""
        if X.shape[1] == self.theta.shape[0]:
            y_pred = X.dot(self.theta)
        else:
            if not hasattr(self, 'coef_') or self.coef_ is None:
                raise Exception("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})              
            y_pred = self.intercept_ + X.dot(self.coef_)
        return y_pred            

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass

    def summary(self):
        summary(self.history)
