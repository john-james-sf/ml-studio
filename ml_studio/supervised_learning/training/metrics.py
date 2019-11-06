# =========================================================================== #
#                                  METRICS MODULE                             #
# =========================================================================== #
from abc import ABC
import math
import numpy as np
import sklearn.metrics

from ml_studio.utils.data_manager import decode

class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self):
        raise NotImplementedError("Metric is an abstract base class."
                                  "Must instantiate the class associated "
                                  "with the required.")

    def __call__(self, y, y_pred):
        raise NotImplementedError("Metric is an abstract base class."
                                  "Must instantiate the class associated "
                                  "with the required.")

# --------------------------------------------------------------------------- #
#                           REGRESSION METRICS                                #
# --------------------------------------------------------------------------- #
class SSE(Metric):
    """Computes sum squared error given data and parameters"""

    def __init__(self):
        self.mode = 'min'
        self.name = 'sum_squared_error'
        self.label = "Sum Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1

    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.sum(e**2)  

class SST(Metric):
    """Computes total sum squared error given data and parameters"""

    def __init__(self):
        self.mode = 'min'
        self.name = 'total_sum_squared_error'
        self.label = "Total Sum Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1

    
    def __call__(self, y, y_pred):
        y_avg = np.mean(y)
        e = y-y_avg                
        return np.sum(e**2)

class R2(Metric):
    """Computes coefficient of determination."""

    def __init__(self):
        self.mode = 'max'        
        self.name = 'R2'
        self.label = "Coefficient of Determination (R2)"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        self._sse = SSE()
        self._sst = SST()
        r2 = 1 - (self._sse(y, y_pred)/self._sst(y, y_pred))        
        return r2

class VarExplained(Metric):
    """Computes proportion of variance explained."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'percent_variance_explained'
        self.label = "Percent Variance Explained"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        var_explained = 1 - (np.var(y-y_pred) / np.var(y))
        return var_explained                   

class MAE(Metric):
    """Computes mean absolute error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_absolute_error'
        self.label = "Mean Absolute Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = abs(y-y_pred)
        return np.mean(e)


class MSE(Metric):
    """Computes mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_squared_error'
        self.label = "Mean Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.mean(e**2)

class NMSE(Metric):
    """Computes negative mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'neg_mean_squared_error'
        self.label = "Negative Mean Squared Error"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return -np.mean(e**2)

class RMSE(Metric):
    """Computes root mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'root_mean_squared_error'
        self.label = "Root Mean Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.sqrt(np.mean(e**2)) 

class NRMSE(Metric):
    """Computes negative root mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'neg_root_mean_squared_error'
        self.label = "Negative Root Mean Squared Error"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return -np.sqrt(np.mean(e**2))

class MSLE(Metric):
    """Computes mean squared log error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_squared_log_error'
        self.label = "Mean Squared Log Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = np.log(y+1)-np.log(y_pred+1)
        return np.mean(e**2)  

class RMSLE(Metric):
    """Computes root mean squared log error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'root_mean_squared_log_error'
        self.label = "Root Mean Squared Log Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = np.log(y+1)-np.log(y_pred+1)
        return np.sqrt(np.mean(e**2))

class MEDAE(Metric):
    """Computes median absolute error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'median_absolute_error'
        self.label = "Median Absolute Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):        
        return np.median(np.abs(y_pred-y))


class RegressionMetrics:
    """Returns the requested score class."""

    def __call__(self, metric='neg_mean_squared_error'):

        dispatcher = {'r2': R2(),
                      'var_explained': VarExplained(),
                      'mean_absolute_error': MAE(),
                      'mean_squared_error': MSE(),                      
                      'neg_mean_squared_error': NMSE(),
                      'root_mean_squared_error': RMSE(),
                      'neg_root_mean_squared_error': NRMSE(),
                      'mean_squared_log_error': MSLE(),
                      'root_mean_squared_log_error': RMSLE(),
                      'median_absolute_error': MEDAE()}
        return(dispatcher.get(metric,False))

# --------------------------------------------------------------------------- #
#                       CLASSIFICATION METRICS                                #
# --------------------------------------------------------------------------- #
class Accuracy(Metric):
    """Computes accuracy."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'accuracy'
        self.label = "Accuracy"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1
    
    def __call__(self, y, y_pred):
        # If y and y_pred are probability distributions, convert to class prediction
        if len(y.shape) > 1:
            y = decode(y)
        
        if len(y_pred.shape) > 1:
            y_pred = decode(y_pred)

        y = np.atleast_2d(y).reshape(-1,1)
        y_pred = np.atleast_2d(y_pred).reshape(-1,1)

        return np.mean(np.equal(y[0], y_pred[0]), axis=-1) 

class ClassificationMetrics:
    """Returns the requested score class."""

    def __call__(self, metric='accuracy'):

        dispatcher = {'accuracy': Accuracy()}
        return(dispatcher.get(metric,False))
