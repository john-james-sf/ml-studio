# %%
# =========================================================================== #
#                                  CALLBACKS                                  #
# =========================================================================== #
"""Module containing callbacks used to monitor training performance."""
import datetime
from ml_studio.supervised_learning.training.callbacks import Callback

# --------------------------------------------------------------------------- #
#                             HISTORY CLASS                                   #
# --------------------------------------------------------------------------- #
class History(Callback):
    """Records history and metrics for training by epoch."""
    def on_train_begin(self, logs=None):
        """Sets instance variables at the beginning of training.""" 
        self.total_epochs = 0
        self.total_batches = 0
        self.start = datetime.datetime.now() 
        self.epoch_log = {}
        self.batch_log = {}

    def on_train_end(self, logs=None):        
        """Sets instance variables at end of training."""
        self.end = datetime.datetime.now()
        self.duration = (self.end-self.start).total_seconds() 

    def on_batch_end(self, batch, logs=None):
        """Updates data and statistics relevant to the training batch.
        
        Parameters
        ----------
        batch : int
            The current training batch
        
        logs : dict
            Dictionary containing batch statistics, such as batch size, current
            weights and training cost.
        """
        self.total_batches = batch
        for k,v in logs.items():
            self.batch_log.setdefault(k,[]).append(v)        

    def on_epoch_end(self, epoch, logs=None):
        """Updates data and statistics relevant to the training epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch
        
        logs : dict
            Dictionary containing data and statistics for the current epoch,
            such as weights, costs, and optional validation set statistics
            beginning with 'val_'.
        """
        logs = logs or {}
        self.total_epochs = epoch
        for k,v in logs.items():
            self.epoch_log.setdefault(k,[]).append(v)

# --------------------------------------------------------------------------- #
#                            PROGRESS CLASS                                   #
# --------------------------------------------------------------------------- #              
class Progress(Callback):
    """Class that reports progress at designated points during training."""
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.model.checkpoint == 0:
            items_to_report = ('epoch', 'train', 'val')
            logs = {k:v for k,v in logs.items() if k.startswith(items_to_report)}
            progress = "".join(str(key) + ': ' + str(round(value,4)) + ' ' \
                for key, value in logs.items())
            print(progress)

