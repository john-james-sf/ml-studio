# =========================================================================== #
#                          REGRESSION CLASSES                                 #
# =========================================================================== #
"""Regression, Linear Regression, L1, L2 and ElasticNet Regression classes."""
from abc import abstractmethod
import numpy as np

from ml_studio.supervised_learning.training.regularizers import L1, L2, ElasticNet
from ml_studio.supervised_learning.training.estimator import Estimator
from ml_studio.supervised_learning.training.metrics import RegressionMetric
from ml_studio.supervised_learning.training.metrics import RegressionMetricFactory
from ml_studio.supervised_learning.training.cost import RegressionCostFunction
from ml_studio.supervised_learning.training.cost import RegressionCostFactory

import warnings

# --------------------------------------------------------------------------- #
#                          REGRESSION CLASS                                   #
# --------------------------------------------------------------------------- #

class Regression(Estimator):
    """Base class for all regression classes."""

    DEFAULT_METRIC = 'mean_squared_error'
    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 early_stop=False, val_size=0.3, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(Regression, self).__init__(learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, cost=cost, 
                                         metric=metric, 
                                         early_stop=early_stop, 
                                         val_size=val_size, 
                                         patience=patience, 
                                         precision=precision,
                                         verbose=verbose, checkpoint=checkpoint, 
                                         name=name, seed=seed)     
 
    @abstractmethod
    def _set_name(self):
        pass    

    def _get_cost_function(self):
        """Obtains the cost function associated with the cost parameter."""
        cost_function = RegressionCostFactory()(cost=self.cost)
        if not isinstance(cost_function, RegressionCostFunction):
            msg = str(self.cost) + ' is not a supported regression cost function.'
            raise ValueError(msg)
        else:
            return cost_function

    def _get_scorer(self):
        """Obtains the scoring function associated with the metric parameter."""
        if self.metric is not None:
            scorer = RegressionMetricFactory()(metric=self.metric)
            if not isinstance(scorer, RegressionMetric):
                msg = str(self.metric) + ' is not a supported regression metric.'
                raise ValueError(msg)
            else:
                return scorer
        
    def _predict(self, X):
        """Computes predictions during training with current weights."""
        self._validate_data(X)
        y_pred = self._linear_prediction(X)
        return y_pred.ravel()

    def predict(self, X):
        """Predicts output as a linear function of inputs and final parameters.

        The method computes predictions based upon final parameters; therefore,
        the model must have been trained.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        Returns
        -------
        array, shape(n_samples,)
            Returns the linear regression prediction.        
        """
        return self._predict(X)

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
        self._validate_data(X, y)
        y_pred = self.predict(X)
        if self.metric:
            score = self.scorer(y=y, y_pred=y_pred)    
        else:
            score = RegressionMetricFactory()(metric=self.DEFAULT_METRIC)(y=y, y_pred=y_pred)        
        return score

# --------------------------------------------------------------------------- #
#                         LINEAR REGRESSION CLASS                             #
# --------------------------------------------------------------------------- #


class LinearRegression(Regression):
    """Trains linear regression models using Gradient Descent.
    
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

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mean_squared_error')
        Metrics used to evaluate classification scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mean_absolute_error':
            Mean absolute error
        'mean_squared_error':
            Mean squared error
        'neg_mean_squared_error':
            Negative mean squared error
        'root_mean_squared_error':
            Root mean squared error
        'neg_root_mean_squared_error':
            Negative root mean squared error
        'mean_squared_log_error':
            Mean squared log error
        'root_mean_squared_log_error':
            Root mean squared log error
        'median_absolute_error':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.3
        The proportion of the training set to allocate to the validation set.
        Must be between 0 and 1. Only used when early_stop is not False.

    patience : int, default=5
        The number of consecutive iterations with no improvement to wait before
        early stopping.

    precision : float, default=0.01
        The stopping criteria. The precision with which improvement in training
        cost or validation score is measured e.g. training cost at time k+1
        has improved if it has dropped training cost (k) * precision.

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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

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
    regression.LassoRegression : Lasso Regression
    regression.RidgeRegression : Ridge Regression
    regression.ElasticNetRegression : ElasticNet Regression
    """    

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 early_stop=False, val_size=0.3, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(LinearRegression, self).__init__(learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, cost=cost, 
                                         metric=metric, 
                                         early_stop=early_stop, 
                                         val_size=val_size, 
                                         patience=patience, 
                                         precision=precision,
                                         verbose=verbose, checkpoint=checkpoint, 
                                         name=name, seed=seed)   

    def _set_name(self):
        self._set_algorithm_name()
        self.task = "Linear Regression"
        self.name = self.name or self.task + ' with ' + self.algorithm


# --------------------------------------------------------------------------- #
#                         LASSO REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #


class LassoRegression(Regression):
    """Trains lasso regression models using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    alpha : Float, default=0.001
        Constant that multiplies the regularization term.

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mean_squared_error')
        Metrics used to evaluate classification scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mean_absolute_error':
            Mean absolute error
        'mean_squared_error':
            Mean squared error
        'neg_mean_squared_error':
            Negative mean squared error
        'root_mean_squared_error':
            Root mean squared error
        'neg_root_mean_squared_error':
            Negative root mean squared error
        'mean_squared_log_error':
            Mean squared log error
        'root_mean_squared_log_error':
            Root mean squared log error
        'median_absolute_error':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.3
        The proportion of the training set to allocate to the validation set.
        Must be between 0 and 1. Only used when early_stop is not False.

    patience : int, default=5
        The number of consecutive iterations with no improvement to wait before
        early stopping.

    precision : float, default=0.01
        The stopping criteria. The precision with which improvement in training
        cost or validation score is measured e.g. training cost at time k+1
        has improved if it has dropped training cost (k) * precision.

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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

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
    regression.LinearRegression : Lasso Regression
    regression.RidgeRegression : Ridge Regression
    regression.ElasticNetRegression : ElasticNet Regression
    """    

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=0.0001, epochs=1000, cost='quadratic', 
                 metric='mean_squared_error',  early_stop=False, 
                 val_size=0.3, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(LassoRegression, self).__init__(learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, cost=cost, 
                                         metric=metric, 
                                         early_stop=early_stop, 
                                         val_size=val_size, 
                                         patience=patience, 
                                         precision=precision,
                                         verbose=verbose, checkpoint=checkpoint, 
                                         name=name, seed=seed)    
        self.alpha = alpha
        self.regularizer = L1(alpha=alpha)

    def _set_name(self):
        self._set_algorithm_name()
        self.task = "Lasso Regression"
        self.name = self.name or self.task + ' with ' + self.algorithm

# --------------------------------------------------------------------------- #
#                         RIDGE REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class RidgeRegression(Regression):
    """Trains ridge regression models using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    alpha : Float, default=0.0001
        Constant that multiplies the regularization term.

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mean_squared_error')
        Metrics used to evaluate classification scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mean_absolute_error':
            Mean absolute error
        'mean_squared_error':
            Mean squared error
        'neg_mean_squared_error':
            Negative mean squared error
        'root_mean_squared_error':
            Root mean squared error
        'neg_root_mean_squared_error':
            Negative root mean squared error
        'mean_squared_log_error':
            Mean squared log error
        'root_mean_squared_log_error':
            Root mean squared log error
        'median_absolute_error':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.3
        The proportion of the training set to allocate to the validation set.
        Must be between 0 and 1. Only used when early_stop is not False.

    patience : int, default=5
        The number of consecutive iterations with no improvement to wait before
        early stopping.

    precision : float, default=0.01
        The stopping criteria. The precision with which improvement in training
        cost or validation score is measured e.g. training cost at time k+1
        has improved if it has dropped training cost (k) * precision.

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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

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
    regression.LinearRegression : Linear Regression
    regression.LassoRegression : Lasso Regression
    regression.ElasticNetRegression : ElasticNet Regression
    """    

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=0.0001, epochs=1000, cost='quadratic', 
                 metric='mean_squared_error',  early_stop=False, 
                 val_size=0.3, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(RidgeRegression, self).__init__(learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, cost=cost, 
                                         metric=metric, 
                                         early_stop=early_stop, 
                                         val_size=val_size, 
                                         patience=patience, 
                                         precision=precision,
                                         verbose=verbose, checkpoint=checkpoint, 
                                         name=name, seed=seed)     
        self.alpha = alpha
        self.regularizer = L2(alpha=alpha)

    def _set_name(self):
        """Sets name of model for plotting purposes."""
        self._set_algorithm_name()
        self.task = "Ridge Regression"
        self.name = self.name or self.task + ' with ' + self.algorithm

# --------------------------------------------------------------------------- #
#                        ELASTICNET REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #


class ElasticNetRegression(Regression):
    """Trains lasso regression models using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    alpha : Float, default=0.0001
        Constant that multiplies the regularization term.

    ratio : Float, default=0.5
        The L1 ratio with 0 <= ratio <= 1. For ratio = 0 the penalty is an L2 
        penalty. For ratio = 1 it is an L1 penalty. For 0 < ratio < 1, the 
        penalty is a combination of L1 and L2. 

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mean_squared_error')
        Metrics used to evaluate classification scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mean_absolute_error':
            Mean absolute error
        'mean_squared_error':
            Mean squared error
        'neg_mean_squared_error':
            Negative mean squared error
        'root_mean_squared_error':
            Root mean squared error
        'neg_root_mean_squared_error':
            Negative root mean squared error
        'mean_squared_log_error':
            Mean squared log error
        'root_mean_squared_log_error':
            Root mean squared log error
        'median_absolute_error':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.3
        The proportion of the training set to allocate to the validation set.
        Must be between 0 and 1. Only used when early_stop is not False.

    patience : int, default=5
        The number of consecutive iterations with no improvement to wait before
        early stopping.

    precision : float, default=0.01
        The stopping criteria. The precision with which improvement in training
        cost or validation score is measured e.g. training cost at time k+1
        has improved if it has dropped training cost (k) * precision.

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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

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
    regression.LinearRegression : Linear Regression
    regression.RidgeRegression : Ridge Regression
    regression.LassoRegression : Lasso Regression
    """    

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=0.0001, ratio=0.15, epochs=1000, cost='quadratic', 
                 metric='mean_squared_error',  early_stop=False, 
                 val_size=0.3, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(ElasticNetRegression, self).__init__(learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, cost=cost, 
                                         metric=metric, 
                                         early_stop=early_stop, 
                                         val_size=val_size, 
                                         patience=patience, 
                                         precision=precision,
                                         verbose=verbose, checkpoint=checkpoint, 
                                         name=name, seed=seed)    
        self.alpha = alpha
        self.ratio = ratio
        self.regularizer = ElasticNet(alpha=alpha, ratio=ratio)    
        
    def _set_name(self):
        """Sets name of model for plotting purposes."""
        self._set_algorithm_name()
        self.task = "ElasticNet Regression"
        self.name = self.name or self.task + ' with ' + self.algorithm
