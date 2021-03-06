# %%
# =========================================================================== #
#                             EARLY STOP CLASSES                              #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \early_stop.py                                                        #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday September 24th 2019, 3:16:03 am                        #
# Last Modified: Saturday November 30th 2019, 10:36:20 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

from abc import ABC, abstractmethod, ABCMeta
from collections import deque
import numpy as np

from ml_studio.supervised_learning.training.callbacks import Callback

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
    
    @abstractmethod
    def on_train_begin(self, logs=None):        
        pass

    @abstractmethod
    def on_epoch_end(self, logs=None):        
        pass

class EarlyStopImprovement(EarlyStop):
    """Stops training if performance hasn't improved.
    
    Stops training if performance hasn't improved. Improvement is measured 
    with a 'tolerance', so that performance must improve by a factor greater
    than the tolerance, to be considered improved. A 'patience' parameter
    indicates how long non-performance has to occur, in epochs, to stop
    training.

    Parameters
    ----------
    monitor : str, optional (default='val_score')
        Specifies which statistic to monitor for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter

    precision : float, optional (default=0.01)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, monitor='val_score', precision=0.01, patience=10):
        super(EarlyStopImprovement, self).__init__()
        self.monitor = monitor
        self.precision = precision
        self.patience = patience
        # Instance variables
        self._iter_no_improvement = 0
        self._better = None    
        # Attributes
        self.best_performance_ = None
        

    def _validate(self):
        if self.monitor not in ['train_cost', 'train_score', 'val_cost', 'val_score']:
            raise ValueError("monitor must be in ['train_cost', 'train_score', 'val_cost', 'val_score']")
        elif not isinstance(self.precision, float):
            raise TypeError("precision must be a float.")
        elif self.precision < 0 or self.precision > 1:
            raise ValueError("precision must be between 0 and 1. A good default is 0.01 or 1%.")
        elif not isinstance(self.patience, (int)):
            raise TypeError("patience must be an integer.")
        elif 'score' in self.monitor and self.model.metric is None:
            raise ValueError("'score' has been selected for evaluation; however"
                             " no scoring metric has been provided for the model. "
                             "Either change the metric in the EarlyStop class to "
                             "'train_cost' or 'val_cost', or add a metric to the model.")


    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(EarlyStopImprovement, self).on_train_begin()
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
        if 'score' in self.monitor:
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
        current = logs.get(self.monitor)

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
    threshold : int, optional (default=2)
        The threshold of generalization loss, above which training stops.
    
    """

    def __init__(self, threshold=2):
        super(EarlyStopGeneralizationLoss,self).__init__()                
        self.threshold=threshold
        self.best_val_cost = np.Inf    

    def _validate(self):
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

    def __init__(self, threshold=0.25, strip_size=5):
        super(EarlyStopProgress,self).__init__()                
        self.threshold = threshold
        self.strip_size = strip_size
        self.best_val_cost = np.Inf
        self.strip = deque([], strip_size)
        self.progress_threshold = 0.1

    def _validate(self):

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

    def __init__(self, strip_size=5, patience=10):
        super(EarlyStopStrips,self).__init__()
        self.strip_size = strip_size
        self.strip = deque([], strip_size)
        self.patience = patience
        self._strips_no_improvement = 0

    def _validate(self):

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
