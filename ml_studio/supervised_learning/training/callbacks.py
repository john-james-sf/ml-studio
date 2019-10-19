# %%
# =========================================================================== #
#                                  CALLBACKS                                  #
# =========================================================================== #
"""Module containing functionality called during the training process.

Note: The CallbackList and Callback abstract base classes were inspired by
the Keras implementation.  
"""
import abc as ABC
from collections import deque
import datetime
import numpy as np
import types

from ml_studio.supervised_learning.training.metrics import Scorer

# --------------------------------------------------------------------------- #
#                             CALLBACK CLASS                                  #
# --------------------------------------------------------------------------- #
class Callback():
    """Abstract base class used to build new callbacks.

    Properties
    ----------
        params: dict. Training parameters
            (eg. batch size, number of epochs...).
    """

    def __init__(self):
        self.params = None

    def set_params(self, params):
        self.params = params


# --------------------------------------------------------------------------- #
#                             HISTORY CLASS                                   #
# --------------------------------------------------------------------------- #
class History(Callback):
    """Records history and metrics for training by epoch.
    
    Arguments
    ---------
        The callback is automatically attached to all model objects. The 
        Log is returned by the 'fit' method of models.        
    """
    def on_train_begin(self, logs=None):
        self.total_epochs = 0
        self.total_batches = 0
        self.start = datetime.datetime.now() 
        self.epoch_log = {}
        self.batch_log = {}

    def on_train_end(self, logs=None):        
        self.end = datetime.datetime.now()
        self.duration = (self.end-self.start).total_seconds() 

    def on_batch_end(self, batch, logs=None):
        self.total_batches = batch
        for k,v in logs.items():
            self.batch_log.setdefault(k,[]).append(v)        

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.total_epochs = epoch
        for k,v in logs.items():
            self.epoch_log.setdefault(k,[]).append(v)

# --------------------------------------------------------------------------- #
#                            PROGRESS CLASS                                   #
# --------------------------------------------------------------------------- #              
class Progress(Callback):

    def on_epoch_end(self, epoch, logs=None):
        items_to_report = ('epoch', 'train', 'val')
        logs = {k:v for k,v in logs.items() if k.startswith(items_to_report)}
        progress = "".join(str(key) + ': ' + str(round(value,4)) + ' ' \
            for key, value in logs.items())
        print(progress)

