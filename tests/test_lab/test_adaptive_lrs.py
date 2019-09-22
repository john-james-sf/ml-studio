# --------------------------------------------------------------------------- #
#                          TEST GRADIENT DESCENT                              #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.operations.learning_rate_schedules import Adaptive

# Get data
X, y = datasets.load_boston(return_X_y=True)
scaler = StandardScaler()    
X = scaler.fit_transform(X)
# Obtain learning rate schedule
lrs = Adaptive(learning_rate=0.5, precision=0.1, patience=2)
# Instantiate and train 
lr = LinearRegression(learning_rate = lrs, epochs=500)
lr.fit(X,y)
print("cost: ", lr.history.epoch_log['train_cost'], 'learning rate: ', str(lr.history.epoch_log['learning_rate']))
