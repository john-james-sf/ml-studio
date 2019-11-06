# =========================================================================== #
#                          LEARNING RATE SCHEDULES                            #
# =========================================================================== #
"""Learning rate schedules, including constant, time, step, exponential. """
from abc import ABC
import math
import numpy as np
from ml_studio.supervised_learning.training.callbacks import Callback

class LearningRateSchedule(Callback):
    """Abstract base class used to build new learning rate schedules.

    Properties
    ----------
        params: dict. Training parameters
            (eg. batch size, number of epochs...).
    """

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate        

class TimeDecay(LearningRateSchedule):
    """Method for time (epoch) based learning rate schedule."""

    def __init__(self, learning_rate=0.1, decay_steps=1.0, decay_rate=0.5,
                 staircase=False):
        super(TimeDecay, self).__init__(learning_rate=learning_rate)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def on_epoch_end(self, epoch, logs=None):
        if self.staircase:
            learning_rate = self.learning_rate \
                / (1 + self.decay_rate * math.floor(epoch \
                    / self.decay_steps))
        else:
            learning_rate = self.learning_rate \
                / (1 + self.decay_rate * epoch / self.decay_steps)
        self.model.eta = learning_rate        

class StepDecay(LearningRateSchedule):
    """Method for step (epoch) based learning rate schedule."""

    def __init__(self, learning_rate=0.1, decay_steps=10, decay_rate=0.5):
        super(StepDecay, self).__init__(learning_rate=learning_rate)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def on_epoch_end(self, epoch, logs=None):
        learning_rate = self.learning_rate * self.decay_rate ** \
            math.floor((1+epoch)/self.decay_steps)
        self.model.eta = learning_rate

class NaturalExponentialDecay(LearningRateSchedule):
    """Exponential decay based learning rate schedule."""

    def __init__(self, learning_rate=0.1, decay_steps=5, decay_rate=0.5,
                 staircase=False):
        super(NaturalExponentialDecay, self).__init__(learning_rate=learning_rate)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def on_epoch_end(self, epoch, logs=None):
        if self.staircase:
            learning_rate = self.learning_rate \
                * math.exp(-self.decay_rate * math.floor(epoch \
                    / self.decay_steps))
        else:
            learning_rate = self.learning_rate * \
                math.exp(-self.decay_rate * \
                    (epoch / self.decay_steps))

        self.model.eta = learning_rate


class ExponentialDecay(LearningRateSchedule):
    """Exponential decay based learning rate schedule based upon TensorFlow"""

    def __init__(self, learning_rate=0.1, decay_steps=5, decay_rate=0.96,
                 staircase=False):
        super(ExponentialDecay, self).__init__(learning_rate=learning_rate)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def on_epoch_end(self, epoch, logs=None):
        if self.staircase:
            learning_rate = self.learning_rate \
                * self.decay_rate ** \
                    math.floor(epoch/self.decay_steps)
        else:
            learning_rate = self.learning_rate \
                * self.decay_rate ** (epoch/self.decay_steps)
        self.model.eta = learning_rate


class InverseScaling(LearningRateSchedule):
    """Inverse scaling learning rate implemented in sklearn's SGD optimizers"""
    def __init__(self, learning_rate=0.1, power=0.5):
        super(InverseScaling, self).__init__(learning_rate=learning_rate)        
        self.power = power

    def on_epoch_end(self, epoch, logs=None):
        learning_rate = self.learning_rate \
            / epoch ** (self.power)
        self.model.eta = learning_rate


class PolynomialDecay(LearningRateSchedule):
    """Polynomial decay based learning rate schedule based upon TensorFlow"""

    def __init__(self, learning_rate=0.1, decay_steps=100, power=0.5,
                 end_learning_rate=0.0001, cycle=False):
        super(PolynomialDecay, self).__init__(learning_rate=learning_rate)
        self.decay_steps = decay_steps
        self.power = power
        self.end_learning_rate = end_learning_rate
        self.cycle = cycle

    def on_epoch_end(self, epoch, logs=None):
        if self.cycle:
            decay_steps = self.decay_steps * \
                math.ceil(epoch / self.decay_steps)
            learning_rate = (self.learning_rate \
                - self.end_learning_rate) \
                * (1 - epoch / decay_steps) ** (self.power) \
                + self.end_learning_rate
        else:
            iteration = min(epoch, self.decay_steps)
            learning_rate = (self.learning_rate - self.end_learning_rate) \
                * (1 - iteration / self.decay_steps) ** (self.power) \
                + self.end_learning_rate
        self.model.eta = learning_rate

class Adaptive(LearningRateSchedule):
    """Decays learning rate based upon improvement in training cost"""

    def __init__(self, learning_rate=0.1, decay_rate=0.2, precision=0.1, patience=5):
        super(Adaptive, self).__init__(learning_rate=learning_rate)
        self.decay_rate = decay_rate
        self.precision = precision
        self.patience = patience
        self._iter_no_improvement = 0
        self._best_metric = np.Inf

    def _improvement(self, logs):

        # Handle first iteration
        if self._best_metric == np.Inf:            
            self._best_metric = logs.get('train_cost')
            self._iter_no_improvement = 0
            return True
        # Handle other iterations
        elif logs.get('train_cost') < \
            self._best_metric - (self.precision * self._best_metric):
            self._best_metric = logs.get('train_cost')
            self._iter_no_improvement = 0
            return True
        else:
            self._iter_no_improvement += 1
            return False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        learning_rate = self.model.eta                
        if not self._improvement(logs):
            if self._iter_no_improvement == self.patience:
                self._iter_no_improvement = 0
                learning_rate *= self.decay_rate
        self.model.eta = learning_rate
