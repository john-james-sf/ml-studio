import matplotlib.pyplot as plt
import numpy as np

from ml_studio.operations.learning_rate_schedules import TimeDecay
from ml_studio.operations.learning_rate_schedules import NaturalExponentialDecay
from ml_studio.operations.learning_rate_schedules import ExponentialDecay
from ml_studio.operations.learning_rate_schedules import InverseScaling
from ml_studio.operations.learning_rate_schedules import PolynomialDecay

def plot_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()