# =========================================================================== #
#                          LEARNING RATE SCHEDULES                            #
# =========================================================================== #
"""Learning rate schedules, including constant, time, step, exponential. """
from abc import ABC
import math
import numpy as np
from ml_studio.operations.metrics import Scorer

class LearningRateSchedule():
    """Abstract base class used to build new learning rate schedules.

    Properties
    ----------
        params: dict. Training parameters
            (eg. batch size, number of epochs...).
    """

    def __init__(self):
        self.params = None

    def set_params(self, params):
        self.params = params


class TimeDecay(LearningRateSchedule):
    """Method for time (logs.get('epoch')) based learning rate schedule."""

    def __init__(self, decay_steps=1.0, decay_rate=0.5,
                 staircase=False):
        super(TimeDecay, self).__init__()
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, logs):
        if self.staircase:
            learning_rate = self.params.get('learning_rate') \
                / (1 + self.decay_rate * math.floor(logs.get('epoch') \
                    / self.decay_steps))
        else:
            learning_rate = self.params.get('learning_rate') \
                / (1 + self.decay_rate * logs.get('epoch') / self.decay_steps)
        return learning_rate


class NaturalExponentialDecay(LearningRateSchedule):
    """Exponential decay based learning rate schedule."""

    def __init__(self, decay_steps=1.0, decay_rate=0.5,
                 staircase=False):
        super(NaturalExponentialDecay, self).__init__()
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, logs):
        if self.staircase:
            learning_rate = self.params.get('learning_rate') \
                * math.exp(-self.decay_rate * math.floor(logs.get('epoch') \
                    / self.decay_steps))
        else:
            learning_rate = self.params.get('learning_rate') * \
                math.exp(-self.decay_rate * \
                    (logs.get('epoch') / self.decay_steps))

        return learning_rate


class ExponentialDecay(LearningRateSchedule):
    """Exponential decay based learning rate schedule based upon TensorFlow"""

    def __init__(self, decay_steps=1.0, decay_rate=0.5,
                 staircase=False):
        super(ExponentialDecay, self).__init__()
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, logs):
        if self.staircase:
            learning_rate = self.params.get('learning_rate') \
                * self.decay_rate ** \
                    math.floor(logs.get('epoch')/self.decay_steps)
        else:
            learning_rate = self.params.get('learning_rate') \
                * self.decay_rate ** (logs.get('epoch')/self.decay_steps)
        return learning_rate


class InverseScaling(LearningRateSchedule):
    """Inverse scaling learning rate implemented in sklearn's SGD optimizers"""
    def __init__(self, power=0.5):
        super(InverseScaling, self).__init__()        
        self.power = power

    def __call__(self, logs):
        learning_rate = self.params.get('learning_rate') \
            / logs.get('epoch') ** (self.power)
        return learning_rate


class PolynomialDecay(LearningRateSchedule):
    """Polynomial decay based learning rate schedule based upon TensorFlow"""

    def __init__(self, decay_steps=1.0, power=0.5,
                 end_learning_rate=0.0001, cycle=False):
        super(PolynomialDecay, self).__init__()
        self.decay_steps = decay_steps
        self.power = power
        self.end_learning_rate = end_learning_rate
        self.cycle = cycle

    def __call__(self, logs):
        if self.cycle:
            decay_steps = self.decay_steps * \
                math.ceil(logs.get('epoch') / self.decay_steps)
            learning_rate = (self.params.get('learning_rate') \
                - self.end_learning_rate) \
                * (1 - logs.get('epoch') / decay_steps) ** (self.power) \
                + self.end_learning_rate
        else:
            iteration = min(logs.get('epoch'), self.decay_steps)
            learning_rate = (self.params.get('learning_rate') - self.end_learning_rate) \
                * (1 - iteration / self.decay_steps) ** (self.power) \
                + self.end_learning_rate
        return learning_rate

class Adaptive(LearningRateSchedule):
    """Decays learning rate based upon improvement in training cost"""

    def __init__(self, decay_rate=0.2, precision=0.01, patience=5):
        super(Adaptive, self).__init__()
        self.decay_rate = decay_rate
        self.precision = precision
        self.patience = patience
        self._iter_no_improvement = 0
        self._best_metric = None

    def _improvement(self, logs):
        if self._best_metric is None:
            self._best_metric = logs.get('train_cost')
            return True
        else:
            if logs.get('train_cost') < \
                self._best_metric - (self.precision * self._best_metric):
                self._best_metric = logs.get('train_cost')
                self._iter_no_improvement = 0
                return True
            else:
                self._iter_no_improvement += 1
                return False

    def __call__(self, logs):
        if not self._improvement(logs):
            if self._iter_no_improvement == self.patience:
                return logs.get('learning_rate') * self.decay_rate
        return logs.get('learning_rate')
