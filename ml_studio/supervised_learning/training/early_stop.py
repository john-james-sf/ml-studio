# %%
# =========================================================================== #
#                             EARLY STOP CLASSES                              #
# =========================================================================== #

from collections import deque
import numpy as np

from ml_studio.supervised_learning.training.callbacks import Callback
from ml_studio.supervised_learning.training.metrics import RegressionMetrics

# --------------------------------------------------------------------------- #
#                          EARLY STOP PERFORMANCE                             #
# --------------------------------------------------------------------------- #

class EarlyStop(Callback):
    """Abstact base class for all early stopping callbacks."""

    def __init__(self):
        super(EarlyStop, self).__init__()        
        self.epochs_ = 0
        self.converged = False
        self.best_weights_ = None

class EarlyStopPlateau(EarlyStop):
    """Stops training if performance hasn't improved.
    
    Stops training if performance hasn't improved. Improvement is measured 
    with a 'tolerance', so that performance must improve by a factor greater
    than the tolerance, to be considered improved. A 'patience' parameter
    indicates how long non-performance has to occur, in epochs, to stop
    training.

    Parameters
    ----------
    metric : str, optional (default='score')
        Specifies which metric to use when evaluating performance

        'train_cost': Training set costs
        'train_score': Training set scores
        'val_cost': Validation set costs
        'val_score': Validation set scores

    val_size : None, float, optional (default=0.2)
        The proportion of the data to allocate to a validation set. If
        metric is 'score', val_size must be in range (0,1)

    precision : float, optional (default=0.01)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, metric='val_score', val_size=0.2, precision=0.01, patience=5):
        super(EarlyStopPlateau, self).__init__()
        self.metric = metric
        self.val_size = val_size        
        self.precision = precision
        self.patience = patience
        # Instance variables
        self._iter_no_improvement = 0
        self._better = None    
        # Attributes
        self.best_performance_ = None
        

    def _validate(self):
        if not isinstance(self.val_size, (int,float)):
            raise TypeError("val_size must be an int=0, or a float.")         
        elif self.metric not in ['train_cost', 'train_score', 'val_cost', 'val_score']:
            raise ValueError("metric must in ['train_cost', 'train_score', 'val_cost', 'val_score']")
        elif 'val' in self.metric and (self.val_size == 0 or self.val_size is None):
            raise ValueError("val_size must be in range (0,0.5] if metric is 'score'.")
        elif not isinstance(self.precision, float):
            raise TypeError("precision must be a float.")
        elif self.precision < 0 or self.precision > 1:
            raise ValueError("precision must be between 0 and 1. A good default is 0.01 or 1%.")
        elif not isinstance(self.patience, (int)):
            raise TypeError("patience must be an integer.")
        elif 'score' in self.metric and self.model.metric is None:
            raise ValueError("'score' has been selected for evaluation; however"
                             " no scoring metric has been provided for the model. "
                             "Either change the metric in the EarlyStop class to "
                             "'cost', or add a metric to the model.")


    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(EarlyStopPlateau, self).on_train_begin()
        logs = logs or {}
        self._validate()
        # We evaluate improvement against the prior metric plus or minus a
        # margin given by precision * the metric. Whether we add or subtract the margin
        # is based upon the metric. For metrics that increase as they improve
        # we add the margin, otherwise we subtract the margin.  Each metric
        # has a bit called a precision factor that is -1 if we subtract the 
        # margin and 1 if we add it. The following logic extracts the precision
        # factor for the metric and multiplies it by the precision for the 
        # improvement calculation.
        if 'score' in self.metric:
            scorer = self.model.scorer
            self._better = scorer.better
            self.best_performance_ = scorer.worst
            self.precision *= scorer.precision_factor
        else:
            self._better = np.less
            self.best_performance_ = np.Inf
            self.precision *= -1 # Bit always -1 since it improves negatively

    def on_epoch_end(self, epoch, logs=None):
        """Determines whether convergence has been achieved.

        Parameters
        ----------
        epoch : int
            The current epoch number

        logs : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  

        Returns
        -------
        Bool if True convergence has been achieved. 

        """
        logs = logs or {}
        # Obtain current cost or score
        current = logs.get(self.metric)

        # Handle the first iteration
        if self.best_performance_ in [np.Inf,-np.Inf]:
            self._iter_no_improvement = 0
            self.best_performance_ = current
            self.best_weights_ = logs.get('theta')
            self.converged = False
        # Evaluate performance
        elif self._better(current, 
                             (self.best_performance_+self.best_performance_ \
                                 *self.precision)):            
            self._iter_no_improvement = 0
            self.best_performance_ = current
            self.best_weights_ = logs.get('theta')
            self.converged=False
        else:
            self._iter_no_improvement += 1
            if self._iter_no_improvement == self.patience:
                self.converged = True            
        self.model.converged = self.converged                     

# --------------------------------------------------------------------------- #
#                      EARLY STOP GENERALIZATION LOSS                         #
# --------------------------------------------------------------------------- #
class EarlyStopGeneralizationLoss(EarlyStop):
    """Early stopping criteria based upon generalization cost.

    This technique is proposed by Lutz Prechelt in his paper 'Early Stopping
    - but when?' Training stops when generalization loss exceeds a certain
    threshold.

    Parameters
    ----------
    val_size : float, optional (default=0.2)
        The proportion of the data to allocate to a validation set. Must
        be in range (0,1)

    threshold : int, optional (default=2)
        The threshold of generalization loss, above which training stops.
    
    """

    def __init__(self, val_size=0.2, threshold=2):
        super(EarlyStopGeneralizationLoss,self).__init__()        
        self.val_size = val_size
        self.threshold=threshold
        self.best_val_cost = np.Inf    

    def _validate(self):
        if not isinstance(self.val_size, float):
            raise TypeError("val_size must be a float.")
        if not isinstance(self.threshold,(int, float)):
            raise TypeError("threshold must be an integer or float.")

    def on_train_begin(self, logs=None):
        super(EarlyStopGeneralizationLoss, self).on_train_begin()
        logs = logs or {}        
        self._validate()
       
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_cost = logs.get('val_cost')
        gl = 100 * ((val_cost/self.best_val_cost)-1)        
        if val_cost < self.best_val_cost:
            self.best_val_cost = val_cost
            self.best_weights = logs.get('theta')
        self.converged = (gl > self.threshold)
        self.model.converged = self.converged

# --------------------------------------------------------------------------- #
#                      EARLY STOP PROGRESS                                    #
# --------------------------------------------------------------------------- #
class EarlyStopProgress(EarlyStop):
    """Early stopping criteria based upon progress of training."""

    def __init__(self, val_size=0.2, threshold=0.25, strip_size=5):
        super(EarlyStopProgress,self).__init__()        
        self.val_size = val_size
        self.threshold = threshold
        self.strip_size = strip_size
        self.best_val_cost = np.Inf
        self.strip = deque([], strip_size)
        self.progress_threshold = 0.1

    def _validate(self):

        if not isinstance(self.val_size, (int,float)):
            raise TypeError("val_size must be an int or a float.")
        if not isinstance(self.threshold, (int, float)):
            raise TypeError("threshold must be an integer or float.")                    
        if not isinstance(self.strip_size, int):
            raise TypeError("strip_size must be an integer.")     


    def _generalization_loss(self, logs):        
        val_cost = logs.get('val_cost')
        gl = 100 * ((val_cost/self.best_val_cost)-1)
        if val_cost < self.best_val_cost:
            self.best_val_cost = val_cost
            self.best_weights = logs.get('theta')                
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

    def on_train_begin(self, logs=None):
        logs = logs or {}        
        self._validate()
   
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}                
        progress = self._progress(logs)
        if progress:            
            if progress < self.progress_threshold:
                self.converged = True
            else:
                gl = self._generalization_loss(logs)
                self.converged = ((gl/progress) > self.threshold)
        self.model.converged = self.converged

# --------------------------------------------------------------------------- #
#                         EARLY STOP STRIP                                    #
# --------------------------------------------------------------------------- #
class EarlyStopStrips(EarlyStop):
    """Stop when validation error has not improved over 'patience' successive strips."""

    def __init__(self, val_size=0.2, strip_size=5, patience=5):
        super(EarlyStopStrips,self).__init__()        
        self.val_size = val_size
        self.strip_size = strip_size
        self.strip = deque([], strip_size)
        self.patience = patience
        self._strips_no_improvement = 0

    def _validate(self):

        if not isinstance(self.val_size, (int,float)):
            raise TypeError("val_size must be an int or a float.")                   
        if not isinstance(self.strip_size, int):
            raise TypeError("strip_size must be an integer.")      
        if not isinstance(self.patience, int):
            raise TypeError("patience must be an integer")
        if self.patience == 0:
            raise ValueError("patience must be an integer > 0")

    def on_train_begin(self, logs=None):
        logs = logs or {}        
        self._validate()
   
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_cost = logs.get('val_cost')
        self.strip.append(val_cost)
        if len(self.strip) == self.strip_size:
            if self.strip[0] < self.strip[-1]:
                self._strips_no_improvement += 1
            else:
                self._strips_no_improvement = 0
            if self._strips_no_improvement == self.patience:
                self.converged = True
        self.model.converged = self.converged
   





# %%
