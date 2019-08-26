# =========================================================================== #
#                          REGRESSION CLASSES                                 #
# =========================================================================== #
"""Regression classes."""
from ml_studio.operations.regularizers import Regularizer, L1, L2, ElasticNet
from ml_studio.supervised_learning.estimator import GradientDescent
from ml_studio.utils.data import make_polynomial_features

import warnings

# --------------------------------------------------------------------------- #
#                         LINEAR REGRESSION CLASS                             #
# --------------------------------------------------------------------------- #


class LinearRegression(GradientDescent):
    """Performs linear regression with gradient descent."""

    def set_name(self, name=None):
        self.task = "Linear Regression"
        self.name = name or self.task + ' with ' + self.algorithm

# --------------------------------------------------------------------------- #
#                         LASSO REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #


class LassoRegression(GradientDescent):
    """Performs lasso regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=1.0, epochs=1000, cost='quadratic',
                 metric='root_mean_squared_error',  val_size=0.3,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(LassoRegression, self).__init__(learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost,
                                              metric=metric, val_size=val_size, verbose=verbose,
                                              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L1(alpha=alpha)

    def set_name(self, name=None):
        self.task = "Lasso Regression"
        self.name = name or self.task + ' with ' + self.algorithm

# --------------------------------------------------------------------------- #
#                         RIDGE REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class RidgeRegression(GradientDescent):
    """Performs ridge regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=1.0, epochs=1000, cost='quadratic',
                 metric='root_mean_squared_error',  val_size=0.3,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(RidgeRegression, self).__init__(learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost,
                                              metric=metric, val_size=val_size, verbose=verbose,
                                              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L2(alpha=alpha)
        self.task = "Ridge Regression"

# --------------------------------------------------------------------------- #
#                        ELASTICNET REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #


class ElasticNetRegression(GradientDescent):
    """Performs elastic net regression with gradient descent."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 alpha=1.0, ratio=0.5, epochs=1000,  cost='quadratic', 
                 metric='root_mean_squared_error',
                 val_size=0.3, verbose=False, checkpoint=100,
                 name=None, seed=None):
        super(ElasticNetRegression, self).__init__(learning_rate=learning_rate,
                                                   batch_size=batch_size,
                                                   theta_init=theta_init, 
                                                   epochs=epochs, cost=cost,
                                                   metric=metric, 
                                                   val_size=val_size, 
                                                   verbose=verbose,
                                                   checkpoint=checkpoint, 
                                                   name=name, seed=seed)
        self.alpha = alpha
        self.ratio = ratio
        self.regularizer = ElasticNet(alpha=alpha, ratio=ratio)
        self.task = "ElasticNet Regression"
