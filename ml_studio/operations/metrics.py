# =========================================================================== #
#                                  METRICS MODULE                             #
# =========================================================================== #
from abc import ABC
import math
import numpy as np
import sklearn.metrics
# TODO: Add F1 and other 'stateful' scores
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

    
    def __call__(self, y, y_pred):
        y_avg = np.mean(y)
        e = (y-y_avg)**2                
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

    
    def __call__(self, y, y_pred):
        self._sse = SSE()
        self._sst = SST()
        r2 = 1 - self._sse(y, y_pred)/self._sst(y, y_pred)        
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

    
    def __call__(self, y, y_pred):
        var_explained = 1 - np.var(y-y_pred) / np.var(y)
        return var_explained                   

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

    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return -np.sqrt(np.mean(e**2))

class NLRMSE(Metric):
    """Computes negative log root mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'neg_log_root_mean_squared_error'
        self.label = "Negative Log Root Mean Squared Error"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
    
    def __call__(self, y, y_pred):
        e = np.log(y+1)-np.log(y_pred)
        return -np.sqrt(np.mean(e**2))  

class BinaryAccuracy(Metric):
    """Computes binary accuracy."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'binary_accuracy'
        self.label = "Binary Accuracy"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
    
    def __call__(self, y, y_pred):        
        return np.mean(np.equal(y, np.round(y_pred)), axis=-1)  

class CategoricalAccuracy(Metric):
    """Computes categorical accuracy."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'categorical_accuracy'
        self.label = "Categorical Accuracy"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
    
    def __call__(self, y, y_pred):        
        return np.equal(np.argmax(y, axis=-1),
                        np.argmax(y_pred, axis=-1))

class Scorer:
    """Returns the requested score class."""

    def __call__(self, metric='neg_mean_squared_error'):

        dispatcher = {'mean_squared_error': MSE(),                      
                      'neg_mean_squared_error': NMSE(),
                      'root_mean_squared_error': RMSE(),
                      'neg_log_root_mean_squared_error': NLRMSE(),
                      'neg_root_mean_squared_error': NRMSE(),
                      'binary_accuracy': BinaryAccuracy(),
                      'categorical_accuracy': CategoricalAccuracy()}
        return(dispatcher[metric])
