# =========================================================================== #
#                         LEARNING RATE SCHEDULES                             #
# =========================================================================== #
#%%
import matplotlib.pyplot as plt
import numpy as np

from ml_studio.operations.learning_rate_schedules import TimeDecay
from ml_studio.operations.learning_rate_schedules import StepDecay
from ml_studio.operations.learning_rate_schedules import NaturalExponentialDecay
from ml_studio.operations.learning_rate_schedules import ExponentialDecay
from ml_studio.operations.learning_rate_schedules import InverseScaling
from ml_studio.operations.learning_rate_schedules import PolynomialDecay

def plot_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    log = {}
    lrs = []
    for i in iterations:
        log['epoch'] = i
        lrs.append(schedule_fn(log))
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()

# Time decay
lrs = TimeDecay(learning_rate=1, decay_rate=0.2)
plot_schedule(lrs)

# Step decay
lrs = StepDecay(learning_rate=1, decay_steps=250)
plot_schedule(lrs)

# Natural exponential decay
lrs = NaturalExponentialDecay(learning_rate=1)
plot_schedule(lrs)
#%%
# Exponential decay
lrs = ExponentialDecay(learning_rate=1, decay_rate=0.2)
plot_schedule(lrs)

# Inverse scaling decay
lrs = InverseScaling(learning_rate=1)
plot_schedule(lrs)

#%%
# Polynomial decay
lrs = PolynomialDecay(learning_rate=1, power=0.2, decay_steps=1500)
plot_schedule(lrs)



#%%
