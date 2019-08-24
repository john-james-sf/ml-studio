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

    def __init__(self, learning_rate=0.01, theta_init=None, epochs=1000,
                 cost='quadratic',  metric='root_mean_squared_error',
                 val_size=0.3, verbose=False, checkpoint=100, name=None,
                 seed=None):
        super(LinearRegression, self).__init__(learning_rate=learning_rate,
                                               theta_init=theta_init,
                                               epochs=epochs, cost=cost,
                                               metric=metric, val_size=val_size,
                                               verbose=verbose,
                                               checkpoint=checkpoint, name=name,
                                               seed=seed)

    def set_name(self, name=None):
        self.name = name or 'Linear Regression with Batch Gradient Descent'

# --------------------------------------------------------------------------- #
#                         LASSO REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #


class LassoRegression(GradientDescent):
    """Performs lasso regression with gradient descent."""

    def __init__(self, learning_rate=0.01, theta_init=None, alpha=1.0, epochs=1000,
                 cost='quadratic',
                 metric='root_mean_squared_error',  val_size=0.3,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(LassoRegression, self).__init__(learning_rate=learning_rate,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost,
                                              metric=metric, val_size=val_size, verbose=verbose,
                                              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L1(alpha=alpha)

    def set_name(self, name=None):
        self.name = name or "Lasso Regression with Batch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}


# --------------------------------------------------------------------------- #
#                         RIDGE REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class RidgeRegression(GradientDescent):
    """Performs ridge regression with gradient descent."""

    def __init__(self, learning_rate=0.01, theta_init=None, alpha=1.0, epochs=1000,
                 cost='quadratic',
                 metric='root_mean_squared_error',  val_size=0.3,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(RidgeRegression, self).__init__(learning_rate=learning_rate,
                                              theta_init=theta_init, epochs=epochs,
                                              cost=cost,
                                              metric=metric, val_size=val_size, verbose=verbose,
                                              checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.regularizer = L2(alpha=alpha)

    def set_name(self, name=None):
        self.name = name or "Ridge Regression with Batch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}

# --------------------------------------------------------------------------- #
#                        ELASTICNET REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #


class ElasticNetRegression(GradientDescent):
    """Performs elastic net regression with gradient descent."""

    def __init__(self, learning_rate=0.01, theta_init=None, alpha=1.0, ratio=0.5,
                 epochs=1000,  cost='quadratic',
                 metric='root_mean_squared_error',
                 val_size=0.3, verbose=False, checkpoint=100,
                 name=None, seed=None):
        super(ElasticNetRegression, self).__init__(learning_rate=learning_rate,
                                                   theta_init=theta_init, epochs=epochs,
                                                   cost=cost,
                                                   metric=metric, val_size=val_size, verbose=verbose,
                                                   checkpoint=checkpoint, name=name, seed=seed)
        self.alpha = alpha
        self.ratio = ratio
        self.regularizer = ElasticNet(alpha=alpha, ratio=ratio)

    def set_name(self, name=None):
        self.name = name or "Elastic Net Regression with Batch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                'ratio': self.ratio,
                "epochs": self.epochs,
                "cost": self.cost,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}

# --------------------------------------------------------------------------- #
#                        POLYNOMIAL REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #


class PolynomialRegression(GradientDescent):
    """The relationship between x and y is modelled as an nth degree polynomial."""

    def __init__(self, degree, learning_rate=0.01, theta_init=None,
                 epochs=1000,  cost='quadratic', metric='root_mean_squared_error',
                 val_size=0.3, verbose=False, checkpoint=100,
                 name=None, seed=None):
        super(PolynomialRegression, self).__init__(learning_rate=learning_rate,
                                                   theta_init=theta_init, epochs=epochs,
                                                   cost=cost, metric=metric,
                                                   val_size=val_size, verbose=verbose,
                                                   checkpoint=checkpoint, name=name, seed=seed)
        self.degree = degree
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        self.regularizer.name = None

    def set_name(self, name=None):
        self.name = name or "Polynomial Regression with Batch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'degree': self.degree,
                'theta_init': self.theta_init,
                "epochs": self.epochs,
                "cost": self.cost,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}

    def fit(self, X, y):
        X = make_polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X, y):
        X = make_polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).predict(X)

# --------------------------------------------------------------------------- #
#                           SGD REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #


class SGDRegression(LinearRegression):
    """The relationship between x and y is modelled as an nth degree polynomial."""

    def __init__(self, learning_rate=0.01, batch_size=1, theta_init=None, epochs=1000,
                 cost='quadratic', metric='root_mean_squared_error',  val_size=0.3,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(SGDRegression, self).__init__(learning_rate=learning_rate,
                                            theta_init=theta_init, epochs=epochs,
                                            cost=cost, metric=metric, 
                                            val_size=val_size, verbose=verbose,
                                            checkpoint=checkpoint, name=name, 
                                            seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if self.batch_size == 1:
            self.name = name or "Linear Regression with Stochastic Gradient Descent"
        else:
            self.name = name or "Linear Regression with Minibatch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                'theta_init': self.theta_init,
                "epochs": self.epochs,
                "cost": self.cost,
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
                 epochs=1000, metric='root_mean_squared_error',  val_size=0.3,
                 verbose=False, checkpoint=100, name=None, seed=None):
        super(SGDLassoRegression, self).__init__(learning_rate=learning_rate,
                                                 theta_init=theta_init, alpha=alpha, 
                                                 epochs=epochs, metric=metric, 
                                                 val_size=val_size, verbose=verbose,
                                                 checkpoint=checkpoint, name=name, 
                                                 seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if self.batch_size == 1:
            self.name = name or "Lasso Regression with Stochastic Gradient Descent"
        else:
            self.name = name or "Lasso Regression with Minibatch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
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
                 epochs=1000, cost='quadratic', metric='root_mean_squared_error',
                 val_size=0.3, verbose=False, checkpoint=100, name=None,
                 seed=None):
        super(SGDRidgeRegression, self).__init__(learning_rate=learning_rate,
                                                 theta_init=theta_init, alpha=alpha, 
                                                 epochs=epochs,
                                                 cost=cost, metric=metric, 
                                                 val_size=val_size, verbose=verbose,
                                                 checkpoint=checkpoint, name=name, 
                                                 seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if self.batch_size == 1:
            self.name = name or "Ridge Regression with Stochastic Gradient Descent"
        else:
            self.name = name or "Ridge Regression with Minibatch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                "epochs": self.epochs,
                "cost": self.cost,
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
                 ratio=0.5, epochs=1000,  cost='quadratic',
                 metric='root_mean_squared_error',
                 val_size=0.3, verbose=False, checkpoint=100,
                 name=None, seed=None):
        super(SGDElasticNetRegression, self).__init__(learning_rate=learning_rate,
                                                      theta_init=theta_init, 
                                                      alpha=alpha, ratio=ratio, 
                                                      epochs=epochs, cost=cost,
                                                      metric=metric, 
                                                      val_size=val_size,
                                                      verbose=verbose,
                                                      checkpoint=checkpoint,
                                                      name=name, seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if self.batch_size == 1:
            self.name = name or "ElasticNet Regression with Stochastic Gradient Descent"
        else:
            self.name = name or "ElasticNet Regression with Minibatch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'batch_size': self.batch_size,
                'theta_init': self.theta_init,
                'alpha': self.alpha,
                'ratio': self.ratio,
                "epochs": self.epochs,
                "cost": self.cost,
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
                 epochs=1000,  cost='quadratic', metric='root_mean_squared_error',
                 val_size=0.3, verbose=False, checkpoint=100,
                 name=None, seed=None):
        super(SGDPolynomialRegression, self).__init__(learning_rate=learning_rate,
                                                      degree=degree, 
                                                      theta_init=theta_init, epochs=epochs,
                                                      cost=cost, metric=metric, 
                                                      val_size=val_size, 
                                                      verbose=verbose,
                                                      checkpoint=checkpoint, 
                                                      name=name, seed=seed)
        self.batch_size = batch_size

    def set_name(self, name=None):
        if self.batch_size == 1:
            self.name = name or "Polynomial Regression with Stochastic Gradient Descent"
        else:
            self.name = name or "Polynomial Regression with Minibatch Gradient Descent"

    def get_params(self, deep=True):
        """Returns the parameters for the estimator."""

        return {"learning_rate": self.learning_rate,
                'degree': self.degree,
                'batch_size': self.batch_size,
                'theta_init': self.theta_init,
                "epochs": self.epochs,
                "cost": self.cost,
                "metric": self.metric,
                "val_size": self.val_size,
                'verbose': self.verbose,
                'checkpoint': self.checkpoint,
                "name": self.name,
                "seed": self.seed}
