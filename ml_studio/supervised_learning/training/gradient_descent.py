# =========================================================================== #
#                          GRADIENT DESCENT CLASS                             #
# =========================================================================== #
"""Regression classes."""
from abc import ABC, abstractmethod, ABCMeta
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils import shuffle
import warnings

from ml_studio.utils.data_manager import batch_iterator, data_split
from ml_studio.supervised_learning.training.callbacks import CallbackList, Callback
from ml_studio.supervised_learning.training.monitor import History, Progress
from ml_studio.supervised_learning.training.learning_rate_schedules import LearningRateSchedule
from ml_studio.supervised_learning.training.early_stop import EarlyStop

from ml_studio.supervised_learning.training import reports

# --------------------------------------------------------------------------- #

class GradientDescent(ABC, BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """Base class gradient descent estimator.
    
    Gradient Descent is a first-order iterative optimization algorithm for 
    finding the minimum of a real-valued, differentiable objective function. 
    Parameterized by :math:`\\theta \\in \\mathbb{R}^n`, Gradient Descent 
    iteratively updates the parameters in the direction opposite to the 
    gradient of the objective function :math:`\\nabla_\\theta J(\\theta)`.

    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, optional (default='quadratic')
        The string name for a cost function

    metric : str, optional (default='mean_squared_error')
        The metric to use when computing the score:

        'r2': 
            The coefficient of determination.
        'var_explained': 
            Proportion of variance explained by model.
        'mean_absolute_error':
            Mean absolute error.
        'mean_squared_error': 
            Mean squared error.
        'neg_mean_squared_error':
            Negative mean squared error.
        'root_mean_squared_error': 
            Root mean squared error.
        'neg_root_mean_squared_error': 
            Negative root mean squared error.
        'mean_squared_log_error':
            Log of mean squared.
        'root_mean_squared_log_error': 
            Log of root mean squared error.
        'median_absolute_error': 
            Mean absolute error.
        'accuracy': 
            Accuracy

    early_stop : None or EarlyStop subclass, optional (default=None)
        The early stopping algorithm to use during training.

    verbose : bool, optional (default=False)
        If true, performance during training is summarized to sysout.

    checkpoint : None or int, optional (default=100)
        If verbose, report performance each 'checkpoint' epochs

    name : None or str, optional (default=None)
        The name of the model used for plotting

    seed : None or int, optional (default=None)
        Random state seed        

    Attributes
    ----------
    coef_ : array-like shape (n_features,1) or (n_features, n_classes)
        Coefficient of the features in X. 'coef_' is of shape (n_features,1)
        for binary problems. For multi-class problems, 'coef_' corresponds
        to outcome 1 (True) and '-coef_' corresponds to outcome 0 (False).

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 
        'intercept_' is of shape (1,) for binary problems. For multi-class
        problems, `intercept_` corresponds to outcome 1 (True) and 
        `-intercept_` corresponds to outcome 0 (False).

    epochs_ : int
        Total number of epochs executed.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    regression.LinearRegression : Linear Regression Class
    regression.LassoRegression : Lasso Regression Class
    regression.RidgeRegression : Ridge Regression Class
    regression.ElasticNetRegression : ElasticNet Regression Class
    classification.Classification : Binary and Multinomial Classification
    classification.LassoClassification : Lasso Classification Class
    classification.RidgeClassification : Ridge Classification Class
    classification.ElasticNetClassification : ElasticNet Classification Class
    """

    DEFAULT_METRIC = 'mean_squared_error'

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 early_stop=None, verbose=False, 
                 checkpoint=100, name=None, seed=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.cost = cost
        self.metric = metric
        self.early_stop = early_stop
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
        self.cost_function = None
        self.cbks = None
        self.X = self.y = self.X_val = self.y_val = None
        self.regularizer = lambda x: 0
        self.regularizer.gradient = lambda x: 0
        self.algorithm = None
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
            if not isinstance(self.early_stop, EarlyStop):
                raise TypeError("early stop is not a valid EarlyStop callable.")
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
            elif self.checkpoint > self.epochs:
                warnings.warn(UserWarning(
                    "checkpoint must not be greater than the number of epochs."))
        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise TypeError("seed must be a positive integer.")

    def _validate_data(self, X, y=None):
        """Confirms data are numpy arrays."""
        if not isinstance(X, (np.ndarray)):
            raise TypeError("X must be of type np.ndarray")
        if y is not None:
            if not isinstance(y, (np.ndarray)):
                raise TypeError("y must be of type np.ndarray")            
            if len(y.shape) > 1:
                raise ValueError("y should be of shape (m,) or (m,n_classes), not %s" % str(y.shape))
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y have incompatible lengths")

    def _prepare_data(self, X, y):
        """Prepares training (and validation) data."""
        self.X = self.X_val = self.y = self.y_val = None
        # Add a column of ones to train the intercept term
        self.X = np.insert(X, 0, 1, axis=1)  
        self.y = y
        # Set aside val_size training observations for validation set 
        if self.early_stop:
            if self.early_stop.val_size:
                self.X, self.X_val, self.y, self.y_val = \
                    data_split(self.X, self.y, 
                    test_size=self.early_stop.val_size, random_state=self.seed)

    def _evaluate_epoch(self, log=None):
        """Computes training (and validation) costs and scores."""
        log = log or {}
        y_pred = self._predict(self.X)
        # Compute epoch training cost
        log['epoch'] = self.epoch
        log['learning_rate'] = self.eta
        log['theta'] = self.theta.copy()        
        log['train_cost'] = self.cost_function(y=self.y, y_pred=y_pred)
        # Compute epoch training score
        if self.metric is not None:
            log['train_score'] = self.scorer(y=self.y, y_pred=y_pred)        
        # Compute epoch validation cost and scores        
        if self.early_stop:
            if self.early_stop.val_size:
                y_pred = self._predict(self.X_val)
                log['val_cost'] = self.cost_function(y=self.y_val, y_pred=y_pred)
                if self.metric:
                    log['val_score'] = self.scorer(y=self.y_val, y_pred=y_pred)
        return log


    @abstractmethod
    def _get_cost_function(self):
        """Obtains the cost function for the cost parameter."""
        pass

    @abstractmethod        
    def _get_scorer(self):
        """Obtains the scoring function for the metric parameter."""
        pass

    def _compile(self):
        """Obtains external objects and add key functions to the log."""
        self.cost_function = self._get_cost_function()
        self.scorer = self._get_scorer()        

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
        if isinstance(self.early_stop, Callback):
            self.cbks.append(self.early_stop)
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
        """Initializes eta."""
        if isinstance(self.learning_rate, LearningRateSchedule):
            self.eta = self.learning_rate.learning_rate
        else:
            self.eta = self.learning_rate

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self.converged = False
        self._validate_params()
        self._validate_data(log.get('X'), log.get('y'))        
        self._prepare_data(log.get('X'), log.get('y'))
        self._init_weights()   
        self._compile()
        self._init_callbacks()
        self.cbks.on_train_begin(log)
        self._set_learning_rate()
        self._set_algorithm_name()
        self._set_name()

    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self.cbks.on_train_end()
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]        
        self.epochs_ = self.epoch

    def _begin_epoch(self):
        """Increment the epoch count and shuffle the data."""
        self.epoch += 1
        self.X, self.y = shuffle(self.X, self.y, random_state=self.seed)
        if self.seed:
            self.seed += 1
        self.cbks.on_epoch_begin(self.epoch)

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch()
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
                # Compute gradient and update weights
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
    
    def _decision(self, X):
        """Computes decision based upon data."""
        if X.shape[1] == self.theta.shape[0]:
            d = X.dot(self.theta)
        else:
            if not hasattr(self, 'coef_') or self.coef_ is None:
                raise Exception("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})              
            d = self.intercept_ + X.dot(self.coef_)  
        return d            

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
        reports.summary(self.history)
