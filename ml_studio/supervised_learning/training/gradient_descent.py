# =========================================================================== #
#                          GRADIENT DESCENT CLASS                             #
# =========================================================================== #
"""Regression classes."""
from abc import ABC, abstractmethod
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings

from ml_studio.utils.data_manager import batch_iterator

from ml_studio.supervised_learning.training import callbacks as cbks
from ml_studio.supervised_learning.training.early_stop import EarlyStop, EarlyStopPlateau
from ml_studio.supervised_learning.training.metrics import Metric, Scorer
from ml_studio.supervised_learning.training.regularizers import Regularizer, L1, L2, ElasticNet
from ml_studio.supervised_learning.training.cost import Cost, CostFunctions
from ml_studio.supervised_learning.training.learning_rate_schedules import LearningRateSchedule

from ml_studio.supervised_learning.training import reports
from ml_studio.utils.data_manager import make_polynomial_features

# --------------------------------------------------------------------------- #

class GradientDescent(ABC, BaseEstimator, RegressorMixin):
    """Base class gradient descent estimator.
    
    Gradient Descent is a first-order iterative optimization algorithm for 
    finding the minimum of a real-valued, differentiable objective function. 
    Parameterized by :math:`\\theta \\in \\mathbb{R}^n`, Gradient Descent 
    iteratively updates the parameters in the direction opposite to the 
    gradient of the objective function :math:`\\nabla_\\theta J(\\theta)`.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    """

    DEFAULT_METRIC = 'mean_squared_error'

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 early_stop=None, verbose=False, checkpoint=100, name=None, 
                 seed=None):
        """Instantiates instance of class.

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

        cost : str, optional (default='quadratic)
            The string containing the name of the cost function:

            'quadratic':
                The least mean squares cost function for regression tasks
            'binary_crossentropy':
                Cost function for binary (sigmoid) classification
            'categorical_crossentropy':
                Cost function for multinomial logistic regression with softmax

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

        """
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
        # Various state variables
        self.epoch = 0
        self.batch = 0
        self.converged = False
        self.theta = None
        self.eta = None
        self.cost_function = None
        self.X = self.y = self.X_val = self.y_val = None
        self.regularizer = lambda x: 0
        self.regularizer.gradient = lambda x: 0
        self.coef = None
        self.intercept = None
        # Set name, task, and algorithm name for reporting purposes
        if self.batch_size is None:
            self.algorithm = 'Batch Gradient Descent'
        elif self.batch_size == 1:
            self.algorithm = 'Stochastic Gradient Descent'
        else:
            self.algorithm = 'Minibatch Gradient Descent'


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
        if self.cost:
            if not CostFunctions()(cost=self.cost):
                msg = self.cost + ' is not a supported cost function.'
                raise ValueError(msg)        
        else:
            raise ValueError("cost must be a string containing the name of the cost function.")

        if self.early_stop:
            if not isinstance(self.early_stop, EarlyStop):
                raise TypeError("early stop is not a valid EarlyStop callable.")
        if self.metric is not None:
            if not isinstance(self.metric, str):
                raise TypeError("metric must be string containing name of metric for scoring")            
            if not Scorer()(metric=self.metric):            
                msg = self.metric + ' is not a supported metric.'
                raise ValueError(msg)        
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
            if X.shape[0] != len(y):
                raise ValueError("X and y have incompatible shapes")

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
                    train_test_split(self.X, self.y, 
                    test_size=self.early_stop.val_size, random_state=self.seed)

    def _compile(self):
        self.cost_function = CostFunctions()(cost=self.cost)        
        if self.metric:
            self.scorer = Scorer()(metric=self.metric)
        # Initialize callbacks
        self.history = cbks.History()
        self.history.set_params(self.get_params())
        self.progress = cbks.Progress()
        self.progress.set_params(self.get_params())
        
    def _init_weights(self):
        """Initializes weights"""        
        if self.theta_init is None:
            # Initialize weights using random normal distributiohn
            np.random.seed(self.seed)
            self.theta = np.random.normal(size=self.X.shape[1])
        else:
            # Confirm theta_init is right shape, then set thetas.
            if len(self.theta_init) != self.X.shape[1]:
                raise ValueError("length of theta_init does not match X.shape[1]")
            else:
                self.theta = self.theta_init.copy()

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self.converged = False
        self._validate_params()
        self._validate_data(log.get('X'), log.get('y'))        
        self._prepare_data(log.get('X'), log.get('y'))
        self._compile()
        self._init_weights()   
        self.history.on_train_begin()      
        # Initialize eta, the learning rate used in calculations
        if isinstance(self.learning_rate, LearningRateSchedule):
            self.eta = self.learning_rate.learning_rate
        else:
            self.eta = self.learning_rate
        # Initialize early stopping callback
        if self.early_stop:
            early_stop_log = {}
            early_stop_log['metric'] = self.metric
            self.early_stop.on_train_begin(early_stop_log)


    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self.history.on_train_end()
        self.intercept = self.theta[0]
        self.coef = self.theta[1:]        

    def _begin_epoch(self):
        """Increment the epoch count and shuffle the data."""
        self.epoch += 1
        self.X, self.y = shuffle(self.X, self.y, random_state=self.seed)
        if self.seed:
            self.seed += 1

    def _end_epoch(self, log=None):
        """Performs end-of-epoch evaluation and scoring."""
        # Compute final epoch training prediction
        log = log or {}
        y_pred = self._predict(self.X)
        # Compute final epoch training cost (and scores)
        log['epoch'] = self.epoch
        log['learning_rate'] = self.eta
        log['theta'] = self.theta.copy()        
        log['train_cost'] = self.cost_function(y=self.y, y_pred=y_pred)
        if self.metric is not None:
            log['train_score'] = self.scorer(y=self.y, y_pred=y_pred)        

        # Compute final epoch validation cost (and scores)        
        if self.early_stop:
            if self.early_stop.val_size:
                y_pred = self._predict(self.X_val)
                log['val_cost'] = self.cost_function(y=self.y_val, y_pred=y_pred)
                if self.metric:
                    log['val_score'] = self.scorer(y=self.y_val, y_pred=y_pred)

        # Update history callback
        self.history.on_epoch_end(self.epoch, log)
        # Report progress if verbose
        if self.verbose and self.epoch % self.checkpoint == 0:
            self.progress.on_epoch_end(self.epoch, log)
        # Update learning rate of learning_rate is a callable
        if isinstance(self.learning_rate, LearningRateSchedule):
            self.eta = self.learning_rate(log)
        # Evaluate early stopping criteria if early stop is callable
        if self.early_stop:
            self.early_stop.on_epoch_end(self.epoch, log)
            self.converged = self.early_stop.converged

    def _begin_batch(self, log=None):
        """Placeholder for batch initialization functionality."""
        self.batch += 1

    def _end_batch(self, log=None):
        """Updates batch history."""
        self.history.on_batch_end(self.batch, log)

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
        if X.shape[1] == len(self.theta):
            d = X.dot(self.theta)
        else:
            if not hasattr(self, 'coef') or self.coef is None:
                raise Exception("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})              
            d = self.intercept + X.dot(self.coef)  
        return d            

    def _predict(self, X):
        """Private predict method that computes predictions with current weights."""
        assert X.shape[1] == len(self.theta), "Shape of X is incompatible with shape of theta"
        y_pred = self._decision(X)
        return y_pred

    def predict(self, X):
        """Predicts output as a linear function of inputs and final parameters.
        
        This method produces predictions using the linear regression hypothesis
        function. It is overridden by classification subclasses.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        Returns
        -------
        array, shape(n_samples,)
            Returns the linear regression prediction.        
        """
        if self.coef is None:
            raise Exception("Unable to predict. Model has not been fit.")        
        self._validate_data(X)
        if X.shape[1] != len(self.coef):
            raise ValueError("Shape of X is incompatible with shape of theta")        
        y_pred = self._decision(X)
        return y_pred

    def score(self, X, y):
        """Computes a score for the current model, given inputs X and output y.

        The score uses the class associated the metric parameter from class
        instantiation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        y : numpy array, shape (n_samples,)
            Target values             

        Returns
        -------
        float
            Returns the score for the designated metric.
        """
        if self.coef is None:
            raise Exception("Unable to compute score. Model has not been fit.")
        self._validate_data(X, y)
        if X.shape[1] != len(self.coef):
            raise ValueError("Shape of X is incompatible with shape of theta")        
        y_pred = self.predict(X)
        if self.metric:
            score = self.scorer(y=y, y_pred=y_pred)    
        else:
            score = Scorer()(metric=self.DEFAULT_METRIC)(y=y, y_pred=y_pred)        
        return score

    def summary(self):
        reports.summary(self.history)
