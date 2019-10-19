# =========================================================================== #
#                          REGRESSION CLASSES                                 #
# =========================================================================== #
"""Regression classes."""
from ml_studio.supervised_learning.training.regularizers import L1, L2, ElasticNet
from ml_studio.supervised_learning.training.gradient_descent import GradientDescent
from ml_studio.utils.data_manager import make_polynomial_features

import warnings

# --------------------------------------------------------------------------- #
#                         LINEAR REGRESSION CLASS                             #
# --------------------------------------------------------------------------- #


class LinearRegression(GradientDescent):
    """Performs linear regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mean_squared_error', 
                 early_stop=None, verbose=False, checkpoint=100, 
                 name=None, seed=None):
        super(LinearRegression, self).__init__(learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost, metric=metric, 
                                              early_stop=early_stop,
                                              verbose=verbose,
                                              checkpoint=checkpoint, 
                                              name=name, seed=seed)    
        self.task = "Linear Regression"
        self.name = name or self.task + ' with ' + self.algorithm

# --------------------------------------------------------------------------- #
#                         LASSO REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #


class LassoRegression(GradientDescent):
    """Performs lasso regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=1.0, epochs=1000, cost='quadratic',
                 metric='mean_squared_error', early_stop=None, 
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(LassoRegression, self).__init__(learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost,
                                              metric=metric, early_stop=early_stop,
                                              verbose=verbose,
                                              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L1(alpha=alpha)
        self.task = "Lasso Regression"
        self.name = name or self.task + ' with ' + self.algorithm

# --------------------------------------------------------------------------- #
#                         RIDGE REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class RidgeRegression(GradientDescent):
    """Performs ridge regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=1.0, epochs=1000, cost='quadratic',
                 metric='mean_squared_error',  early_stop=None,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(RidgeRegression, self).__init__(learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost,
                                              metric=metric, early_stop=early_stop,
                                              verbose=verbose,
                                              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L2(alpha=alpha)
        self.task = "Ridge Regression"
        self.name = name or self.task + ' with ' + self.algorithm

# --------------------------------------------------------------------------- #
#                        ELASTICNET REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #


class ElasticNetRegression(GradientDescent):
    """Performs elastic net regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=1.0, ratio=0.5, epochs=1000,  cost='quadratic', 
                 metric='mean_squared_error', early_stop=None,
                 verbose=False, checkpoint=100,
                 name=None, seed=None):
        super(ElasticNetRegression, self).__init__(learning_rate=learning_rate,
                                                   batch_size=batch_size,
                                                   theta_init=theta_init, 
                                                   epochs=epochs, cost=cost,
                                                   metric=metric, 
                                                   early_stop=early_stop,
                                                   verbose=verbose,
                                                   checkpoint=checkpoint, 
                                                   name=name, seed=seed)
        self.alpha = alpha
        self.ratio = ratio
        self.regularizer = ElasticNet(alpha=alpha, ratio=ratio)
        self.task = "ElasticNet Regression"
        self.name = name or self.task + ' with ' + self.algorithm
