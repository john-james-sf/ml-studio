# %%
# =========================================================================== #
#                             EARLY STOP CLASSES                              #
# =========================================================================== #

from collections import deque
import numpy as np

from ml_studio.operations.callbacks import Callback
from ml_studio.operations.metrics import Scorer

# --------------------------------------------------------------------------- #
#                          EARLY STOP PERFORMANCE                             #
# --------------------------------------------------------------------------- #

class EarlyStop(Callback):
    """Abstact base class for all early stopping callbacks."""
    def __init__(self, val_size=0.2):
        self.val_size = val_size
        self.converged = False
        self.best_weights = None

    def _validate(self):
        if not isinstance(self.val_size, (int,float)):
            raise TypeError('val_size must be an integer or float')

    def on_train_begin(self, logs=None):        
        self.converged = False


class EarlyStopPlateau(EarlyStop):
    """Stops training if performance hasn't improved."""

    def __init__(self, val_size=0.2, precision=0.01, patience=5):
        super(EarlyStopPlateau, self).__init__(val_size=val_size)
        self.precision = precision
        self.patience = patience
        self._iter_no_improvement = 0
        self._best_performance = None
        self.better = None    
        self.metric = None  

    def _validate(self):
        super(EarlyStopPlateau, self)._validate()
        if self.metric:
            if not isinstance(self.metric, str):
                raise TypeError("metric must be None or a valid string.")
            if self.metric not in ('r2',
                                    'var_explained',
                                    'mean_absolute_error',
                                    'mean_squared_error',
                                    'neg_mean_squared_error',
                                    'root_mean_squared_error',
                                    'neg_root_mean_squared_error',
                                    'mean_squared_log_error',
                                    'root_mean_squared_log_error',
                                    'median_absolute_error',
                                    'binary_accuracy',
                                    'categorical_accuracy'):
                raise ValueError("Metric %s is not support. " % self.metric)
        if not isinstance(self.precision, float):
            raise TypeError("precision must be a float between -1 and 1")
        if abs(self.precision) >= 1:
            raise ValueError("precision must have an absolute value less than 1")
        if not isinstance(self.patience, (int)):
            raise TypeError("patience must be an integer.")
        if self.val_size is not None:
            if not isinstance(self.val_size, (int,float)):
                raise ValueError("val_size must be an int or a float.")

    def on_train_begin(self, logs=None):        
        """Initializes performance and improvement function."""
        super(EarlyStopPlateau, self).on_train_begin()
        logs = logs or {}
        self.metric = logs.get('metric')
        self._validate()
        # We evaluate improvement against the prior metric plus or minus a
        # margin given by precision * the metric. Whether we add or subtract the margin
        # is based upon the metric. For metrics that increase as they improve
        # we add the margin, otherwise we subtract the margin.  Each metric
        # has a bit called a precision factor that is -1 if we subtract the 
        # margin and 1 if we add it. The following logic extracts the precision
        # factor for the metric and multiplies it by the precision for the 
        # improvement calculation.
        if self.metric:
            scorer = Scorer()(metric=self.metric)
            self.better = scorer.better
            self.best_performance = scorer.worst
            self.precision *= scorer.precision_factor
        else:
            self.better = np.less
            self.best_performance = np.Inf
            self.precision *= -1 # Bit always -1 since it improves negatively

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Obtain current cost or score
        if self.metric is None:
            if self.val_size > 0:
                current = logs.get('val_cost') 
            else:
                current = logs.get('train_cost') 
        else:
            if self.val_size > 0:
                current = logs.get('val_score') 
            else:
                current = logs.get('train_score')  
        # Handle the first iteration
        if self.best_performance in [np.Inf,-np.Inf]:
            self._iter_no_improvement = 0
            self.best_performance = current
            self.best_weights = logs.get('theta')
            self.converged = False
        # Evaluate performance
        elif self.better(current, 
                             (self.best_performance+self.best_performance \
                                 *self.precision)):            
            self._iter_no_improvement = 0
            self.best_performance = current
            self.best_weights = logs.get('theta')
            self.converged=False
        else:
            self._iter_no_improvement += 1
            if self._iter_no_improvement == self.patience:
                self.converged = True                                 

# --------------------------------------------------------------------------- #
#                      EARLY STOP GENERALIZATION LOSS                         #
# --------------------------------------------------------------------------- #
class EarlyStopGeneralizationLoss(EarlyStop):
    """Early stopping criteria based upon generalization cost."""

    def __init__(self, val_size=0.2, threshold=2):
        super(EarlyStopGeneralizationLoss,self).__init__(val_size=val_size)        
        self.threshold=threshold
        self.best_val_cost = np.Inf    

    def _validate(self):
        if self.val_size is not None:
            if not isinstance(self.val_size, float):
                raise ValueError("val_size must be a float.")
            elif self.val_size < 0 or self.val_size > 1:
                raise ValueError("val_size must be None or between 0 and 1.")        

    def _generalization_loss(self, logs):        
        val_cost = logs.get('val_cost')
        if val_cost < self.best_val_cost:
            self.best_val_cost = val_cost
            self.best_weights = logs.get('theta')
        gl = 100 * ((val_cost/self.best_val_cost)-1)        
        return gl

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._validate()
        gl = self._generalization_loss(logs)
        self.converged = (gl > self.threshold)

# --------------------------------------------------------------------------- #
#                      EARLY STOP PROGRESS                                    #
# --------------------------------------------------------------------------- #
class EarlyStopProgress(EarlyStop):
    """Early stopping criteria based upon progress of training."""

    def __init__(self, val_size=0.2, threshold=2, strip_size=5):
        super(EarlyStopProgress,self).__init__(val_size=val_size)        
        self.threshold = threshold
        self.strip_size = strip_size
        self.best_val_cost = np.Inf
        self.strip = deque([], strip_size)

    def _validate(self):

        if self.val_size is not None:
            if not isinstance(self.val_size, float):
                raise ValueError("val_size must be a float.")
            elif self.val_size < 0 or self.val_size > 1:
                raise ValueError("val_size must be None or between 0 and 1.") 
        if not isinstance(self.threshold, (int, float)):
            raise ValueError("threshold must be an integer or float.")                    
        if not isinstance(self.strip_size, int):
            raise ValueError("strip_size must be an integer.")                                

    def _generalization_loss(self, logs):        
        val_cost = logs.get('val_cost')
        if val_cost < self.best_val_cost:
            self.best_val_cost = val_cost
            self.best_weights = logs.get('theta')
        gl = 100 * ((val_cost/self.best_val_cost)-1)        
        return gl      

    def _progress(self, logs):
        logs = logs or {}
        train_cost = logs.get('train_cost')
        self.strip.append(train_cost)
        progress = None
        if len(self.strip) == self.strip_size:
            progress = 1000 * ((sum(self.strip)/ \
                                (self.strip_size * min(self.strip)))-1)
        return progress
   
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}        
        self._validate()
        progress = self._progress(logs)
        if progress:
            gl = self._generalization_loss(logs)
            self.converged = ((progress/gl) > self.threshold)


