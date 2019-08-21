# =========================================================================== #
#                          REGRESSION CLASSES                                 #
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

from ml_studio.utils.data import batch_iterator

from ml_studio.operations import callbacks as cbks
from ml_studio.operations.metrics import Metric, Scorer
from ml_studio.operations.regularizers import Regularizer, L1, L2, ElasticNet
from ml_studio.operations.cost import Cost, CostFunctions

from ml_studio.utils import reports
from ml_studio.utils.data import make_polynomial_features
from ml_studio.visual import plots

import warnings

# --------------------------------------------------------------------------- #
#                             REGRESSION CLASS                                #
# --------------------------------------------------------------------------- #
class Regression(ABC, BaseEstimator, RegressorMixin):
    """Class defines base behavior for regression classes."""
    
    def __init__(self, learning_rate=0.01, theta_init=None, epochs=1000, 
                 fit_intercept=True, cost='quadratic', monitor='val_score',  
                 metric='root_mean_squared_error', val_size=0.3, 
                 verbose = False, checkpoint=100, name=None, seed=None):
        
        self.learning_rate = learning_rate
        self.theta_init = theta_init
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.cost = cost
        self.monitor = monitor
        self.metric = metric                
        self.val_size = val_size
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.name = name
        self.seed = seed   
        # Various state variables
        self.epoch = 0
        self.batch = 0
        self.batch_size=None
        self.theta = None
        self.scorer = lambda x: None
        self.cost_function = None
        self.X = self.y = self.X_val = self.y_val = None
        self.regularizer = lambda x: 0
        self.regularizer.gradient = lambda x: 0    
        self.regularizer.name = None        
        self.final_coef = None
        self.final_intercept = None
        self.best_coef = None
        self.best_intercept = None

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'theta_init': self.theta_init,
                "epochs": self.epochs,
                "fit_intercept": self.fit_intercept,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _validate_params(self):
        """Validate parameters."""
        if not isinstance(self.learning_rate, (int,float)):
            raise ValueError("learning_rate must provide a float.")
        if self.theta_init is not None:
            if not isinstance(self.theta_init,(list,pd.core.series.Series,np.ndarray)):
                raise ValueError("theta must be an array like object.")
        if not isinstance(self.epochs, int):
            raise ValueError("epochs must be an integer.")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, int):
                raise ValueError("batch size must be an integer.") 
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be True or False")
        if self.monitor is not None:
            if self.monitor not in ('train_cost', 'val_cost', 
                                    'train_score', 'val_score'):
                raise ValueError("monitor must be 'train_cost', 'train_score', "
                                " 'val_cost' or 'val_score'.")        
        if self.metric is not None:
            if self.metric not in ('mean_squared_error',
                                   'neg_mean_squared_error',
                                   'root_mean_squared_error',
                                   'neg_log_root_mean_squared_error',
                                   'neg_root_mean_squared_error',
                                   'binary_accuracy',
                                   'categorical_accuracy'):
                raise ValueError("Metric %s is not support. " % self.metric)
        if 'score' in self.monitor:
            if self.metric is None:
                raise ValueError("If monitoring scores, a valid metric must be provided.")
        if not (0.0 <= self.val_size < 1):
            raise ValueError("val_size must be in [0, 1]")
        if 'val' in self.monitor:
            if self.val_size == 0:
                raise ValueError("If val_size = 0, then monitor must be 'train_cost', or 'train_score'")
            elif self.val_size < 0 or self.val_size >= 1:
                raise ValueError("val_size must be greater than 0 and less than 1.")
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be either True or False")
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, int):
                raise ValueError("checkpoint must be a positive integer or None.")
            elif self.checkpoint < 0:
                raise ValueError("checkpoint must be a positive integer or None.")
        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise ValueError("seed must be a positive integer.")    

    def set_name(self, name=None):
        """Sets name for model."""
        raise NotImplementedError()


    def _compile(self):
        self.cost_function = CostFunctions()(cost=self.cost)
        if self.metric:
            self.scorer = Scorer()(metric=self.metric)

    def _format_data(self, X, y=None):
        """Reformats dataframes into numpy arrays."""
        X = X.values if isinstance(X, pd.core.frame.DataFrame) else X
        if y is not None:
            y = y.values.flatten() if isinstance(y, pd.core.frame.DataFrame) else y
        return X, y

    def _prepare_data(self, X,y):
        """Prepares training (and validation) data."""        
        # Reformat dataframes into numpy arrays
        X, y = self._format_data(X, y)

        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)  # Insert 1s as bias term            

        if self.val_size > 0.0:
            X, X_val, y, y_val = train_test_split(X,y, test_size=self.val_size,
                                                random_state=self.seed)                                                                                    

            self.X = X
            self.y = y
            self.X_val = X_val
            self.y_val = y_val                  
        else:            
            self.X = X
            self.y = y

    def _init_weights(self):
        """Initializes weights (thetas) to random normal distribution."""
        if self.theta_init is None:
            np.random.seed(self.seed)
            self.theta = np.random.normal(size=self.X.shape[1])
        else:
            self.theta = self.theta_init.copy()

    def _evaluate(self):
        """Computes cost and score for validation set."""               
        y_pred = self._predict(self.X_val)
        cost = self.cost_function(y=self.y_val, y_pred=y_pred)
        score = self.scorer(y=self.y_val, y_pred=y_pred)
        return cost, score

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""        
        self._validate_params()
        self._compile()          
        self._prepare_data(log.get('X'),log.get('y'))
        self._init_weights()        
        self.set_name()

        self.history = cbks.History(verbose=self.verbose)
        self.history.set_params(self.get_params())
        self.history.on_train_begin()
        
        self.benchmark = cbks.Benchmark(verbose=self.verbose)
        self.benchmark.set_params(self.get_params())
        self.benchmark.on_train_begin()
    
    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self.history.on_train_end()
        if self.fit_intercept:
            self.final_coef = self.theta[1:]        
            self.final_intercept = self.theta[0]
            self.best_coef = self.benchmark.best_model['theta'][1:]
            self.best_intercept = self.benchmark.best_model['theta'][0]
        else:
            self.final_coef = self.theta                    
            self.best_coef = self.benchmark.best_model['theta']            

    def _begin_epoch(self, log=None):
        """Increment the epoch count and shuffle the data."""
        self.epoch += 1
        self.X, self.y = shuffle(self.X, self.y, random_state=self.seed)
        if self.seed:
            self.seed += 1            
    
    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        # Compute final epoch training cost and score  
        log = {}
        y_pred = self._predict(self.X)   
        # Compute final epoch training cost (and scores)
        log['epoch'] = self.epoch
        log['train_cost'] = log['train_score'] = None
        log['train_cost'] = self.cost_function(y=self.y, y_pred=y_pred) 
        if self.metric is not None and self.val_size > 0:             
            log['train_score'] = self.scorer(y=self.y, y_pred=y_pred)       
        
        # Compute final epoch validation cost (and scores)
        log['val_cost'] = log['val_score'] = None
        if self.metric is not None and self.val_size > 0:
            log['val_cost'], log['val_score'] = self._evaluate()     

        # Save epoch final weights in log
        log['theta'] = self.theta

        # Update history and benchmark callbacks
        self.history.on_epoch_end(self.epoch, log)
        self.benchmark.on_epoch_end(self.epoch, log)

    def _begin_batch(self, log=None):
        """Placeholder for batch initialization functionality."""
        self.batch += 1

    def _end_batch(self, log=None):
        """Updates batch history."""
        self.history.on_batch_end(self.batch, log)

    def fit(self, X, y):
        """Trains model until stop condition is met."""
        train_log = {'X':X, 'y':y}
        self._begin_training(train_log)
        
        while (self.epoch < self.epochs):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X, self.y, batch_size=self.batch_size):

                self._begin_batch()
                # Compute prediction
                y_pred = self._predict(X_batch)   
                # Compute costs
                J = self.cost_function(y=y_batch, y_pred = y_pred) + self.regularizer(self.theta[1:])              
                # Update batch log with weights and cost
                batch_log = {'batch': self.batch, 'batch_size': X_batch.shape[0], 
                             'theta': self.theta.copy(), 'train_cost': J.copy()}      
                # Compute gradient and update weights                    
                gradient = self.cost_function.gradient(X_batch, y_batch, y_pred) - self.regularizer.gradient(self.theta[1:])
                self.theta  -= self.learning_rate * gradient
                # Update batch log
                self._end_batch(batch_log)

            # Wrap up epoch
            self._end_epoch() 

        self._end_training()
        return self
    def _predict(self, X):
        """Private predict method that computes predictions in current weights."""
        y_pred = X.dot(self.theta)
        return y_pred

    def predict(self, X):
        """Public predict method that computes predictions on 'best' weights."""
        X, _, = self._format_data(X)
        y_pred = self.best_intercept + X.dot(self.best_coef)
        return y_pred

    def score(self, X, y):
        if self.best_coef is None:
            raise Exception("Unable to compute score. Model has not been fit.")
        X, y, = self._format_data(X, y)
        if self.fit_intercept:
            y_pred = self.best_intercept + X.dot(self.best_coef)
        else:
            y_pred = X.dot(self.best_coef)
        score = self.scorer(y=y, y_pred=y_pred)
        return score   

    def summary(self):
        reports.summary(self.history, self.benchmark)

    def learning_curves(self, which='optimization'):
        if which.startswith('o'):
            plots.LearningCurves().optimization(history=self.history)
        else:
            plots.LearningCurves().performance(history=self.history)



# --------------------------------------------------------------------------- #
#                         LINEAR REGRESSION CLASS                             #
# --------------------------------------------------------------------------- #
class LinearRegression(Regression):
    """Performs linear regression with gradient descent."""

    def __init__(self, learning_rate=0.01, theta_init=None, epochs=1000, 
                 fit_intercept=True, cost='quadratic', monitor='val_score',  
                 metric='root_mean_squared_error',  val_size=0.3, 
                 verbose = False, checkpoint=100, name=None, seed=None):
        super(LinearRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, epochs=epochs, fit_intercept=fit_intercept, 
              cost=cost, monitor=monitor, metric=metric, val_size=val_size, 
              verbose=verbose, checkpoint=checkpoint, name=name, seed=seed)

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                self.name = "Linear Regression with Batch Gradient Descent"
        else:
            self.name = name

# --------------------------------------------------------------------------- #
#                         LASSO REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class LassoRegression(Regression):
    """Performs lasso regression with gradient descent."""

    def __init__(self, learning_rate=0.01, theta_init=None, alpha=1.0, epochs=1000, 
                 fit_intercept=True, cost='quadratic', monitor='val_score',  
                 metric='root_mean_squared_error',  val_size=0.3, 
                 verbose = False, checkpoint=100, name=None, seed=None):
        super(LassoRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, epochs=epochs,  
              fit_intercept=fit_intercept, cost=cost, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L1(alpha=alpha)

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                self.name = "Lasso Regression with Batch Gradient Descent"
        else:
            self.name = name

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}          
    

# --------------------------------------------------------------------------- #
#                         RIDGE REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class RidgeRegression(Regression):
    """Performs ridge regression with gradient descent."""

    def __init__(self, learning_rate=0.01, theta_init=None, alpha=1.0, epochs=1000, 
                 fit_intercept=True, cost='quadratic', monitor='val_score',  
                 metric='root_mean_squared_error',  val_size=0.3, 
                 verbose = False, checkpoint=100, name=None, seed=None):
        super(RidgeRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, epochs=epochs, 
              fit_intercept=fit_intercept, cost=cost, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L2(alpha=alpha)

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                self.name = "Ridge Regression with Batch Gradient Descent"
        else:
            self.name = name        

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}        

# --------------------------------------------------------------------------- #
#                        ELASTICNET REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #
class ElasticNetRegression(Regression):
    """Performs elastic net regression with gradient descent."""

    def __init__(self, learning_rate=0.01, theta_init=None, alpha=1.0, ratio=0.5,
                 epochs=1000,  fit_intercept=True, cost='quadratic',
                 monitor='val_score',  metric='root_mean_squared_error',
                 val_size=0.3, verbose = False, checkpoint=100, 
                 name=None, seed=None):
        super(ElasticNetRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, epochs=epochs, 
              fit_intercept=fit_intercept, cost=cost, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.ratio = ratio
        self.regularizer = ElasticNet(alpha=alpha, ratio=ratio)

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                self.name = "Elastic Net Regression with Batch Gradient Descent"
        else:
            self.name = name    

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                'ratio': self.ratio,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}          

# --------------------------------------------------------------------------- #
#                        POLYNOMIAL REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #
class PolynomialRegression(Regression):
    """The relationship between x and y is modelled as an nth degree polynomial."""

    def __init__(self, degree, learning_rate=0.01, theta_init=None, 
                 epochs=1000,  fit_intercept=True, cost='quadratic',
                 monitor='val_score',  metric='root_mean_squared_error', 
                 val_size=0.3, verbose = False, checkpoint=100, 
                 name=None, seed=None):
        super(PolynomialRegression, self).__init__(learning_rate=learning_rate,
              theta_init=theta_init, epochs=epochs,  fit_intercept=fit_intercept, 
              cost=cost, monitor=monitor, metric=metric, 
              val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)
        self.degree = degree        
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        self.regularizer.name = None         


    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                self.name = "Polynomial Regression with Batch Gradient Descent"
        else:
            self.name = name            

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'degree': self.degree,
                'theta_init': self.theta_init,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}          


    def fit(self, X, y):
        X = make_polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X,y)

    def predict(self, X, y):
        X = make_polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).predict(X)

# --------------------------------------------------------------------------- #
#                           SGD REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class SGDRegression(LinearRegression):
    """The relationship between x and y is modelled as an nth degree polynomial."""

    def __init__(self, learning_rate=0.01, batch_size=1, theta_init=None, epochs=1000, 
                 fit_intercept=True, cost='quadratic', monitor='val_score',  
                 metric='root_mean_squared_error',  val_size=0.3, 
                 verbose = False, checkpoint=100, name=None, seed=None):
        super(SGDRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, epochs=epochs, 
              fit_intercept=fit_intercept, cost=cost, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)        
        self.batch_size = batch_size


    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                if self.batch_size == 1:
                    self.name = "Linear Regression with Stochastic Gradient Descent"
                else:
                    self.name = "Linear Regression with Minibatch Gradient Descent"
        else:
            self.name = name                

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                'theta_init': self.theta_init,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}                    

# --------------------------------------------------------------------------- #
#                         SGD LASSO REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #
class SGDLassoRegression(LassoRegression):
    """The relationship between x and y is modelled as an nth degree polynomial."""

    def __init__(self, learning_rate=0.01, batch_size=1, theta_init=None, alpha=1.0,
                 epochs=1000,  fit_intercept=True, monitor='val_score',  
                 metric='root_mean_squared_error',  val_size=0.3, 
                 verbose = False, checkpoint=100, name=None, seed=None):
        super(SGDLassoRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, alpha=alpha, epochs=epochs, 
              fit_intercept=fit_intercept, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)                       
        self.batch_size = batch_size

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                if self.batch_size == 1:
                    self.name = "Lasso Regression with Stochastic Gradient Descent"
                else:
                    self.name = "Lasso Regression with Minibatch Gradient Descent"
        else:
            self.name = name    

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}          

# --------------------------------------------------------------------------- #
#                         SGD RIDGE REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #
class SGDRidgeRegression(RidgeRegression):
    """Performs ridge regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=1, theta_init=None, alpha=1.0, 
                epochs=1000, fit_intercept=True, cost='quadratic', 
                monitor='val_score',  metric='root_mean_squared_error',  
                val_size=0.3, verbose = False, checkpoint=100, name=None, 
                seed=None):
        super(SGDRidgeRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, alpha=alpha, epochs=epochs, 
              fit_intercept=fit_intercept, cost=cost, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                if self.batch_size == 1:
                    self.name = "Ridge Regression with Stochastic Gradient Descent"
                else:
                    self.name = "Ridge Regression with Minibatch Gradient Descent"
        else:
            self.name = name        

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}

# --------------------------------------------------------------------------- #
#                       SGD ELASTICNET REGRESSION CLASS                       #
# --------------------------------------------------------------------------- #
class SGDElasticNetRegression(ElasticNetRegression):
    """Performs elastic net regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=1, theta_init=None, alpha=1.0, 
                 ratio=0.5, epochs=1000,  fit_intercept=True, cost='quadratic',
                 monitor='val_score',  metric='root_mean_squared_error',
                 val_size=0.3, verbose = False, checkpoint=100, 
                 name=None, seed=None):
        super(SGDElasticNetRegression, self).__init__(learning_rate=learning_rate, 
              theta_init=theta_init, alpha=alpha, ratio=ratio, epochs=epochs, 
              fit_intercept=fit_intercept, cost=cost, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                if self.batch_size == 1:
                    self.name = "Elastic Net Regression with Stochastic Gradient Descent"
                else:
                    self.name = "Elastic Net Regression with Minibatch Gradient Descent"
        else:
            self.name = name        

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'batch_size': self.batch_size,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                'ratio': self.ratio,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}       

# --------------------------------------------------------------------------- #
#                        SGD POLYNOMIAL REGRESSION CLASS                      #
# --------------------------------------------------------------------------- #
class SGDPolynomialRegression(PolynomialRegression):
    """The relationship between x and y is modelled as an nth degree polynomial."""

    def __init__(self, degree, learning_rate=0.01, batch_size=1, theta_init=None, 
                 epochs=1000,  fit_intercept=True, cost='quadratic',
                 monitor='val_score',  metric='root_mean_squared_error', 
                 val_size=0.3, verbose = False, checkpoint=100, 
                 name=None, seed=None):
        super(SGDPolynomialRegression, self).__init__(learning_rate=learning_rate,
              degree = degree, theta_init=theta_init, epochs=epochs,  
              fit_intercept=fit_intercept,  cost=cost, monitor=monitor, 
              metric=metric, val_size=val_size, verbose=verbose, 
              checkpoint=checkpoint, name=name, seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if name is None:
            if self.name is None:
                if self.batch_size == 1:
                    self.name = "Polynomial Regression with Stochastic Gradient Descent"
                else:
                    self.name = "Polynomial Regression with Minibatch Gradient Descent"
        else:
            self.name = name            
        

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'degree': self.degree,
                'batch_size': self.batch_size,                
                'theta_init': self.theta_init,
                "epochs": self.epochs,
                "cost": self.cost,
                "monitor": self.monitor,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}                     