# --------------------------------------------------------------------------- #
#                          TEST EARLY STOP PLATEAU                            #
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
from ml_studio.supervised_learning.training.early_stop import EarlyStopPlateau

# Get data
X, y = datasets.load_boston(return_X_y=True)
scaler = StandardScaler()    
X = scaler.fit_transform(X)
# Obtain learning rate schedule
es = EarlyStopPlateau(precision=0.2, patience=2)
# Instantiate and train 
lr = LinearRegression(learning_rate = 0.3, epochs=500, early_stop=es)
lr.fit(X,y)
print("Total epochs : ", str(lr.history.total_epochs))
print("train_score: ", lr.history.epoch_log['train_score'])
print("val_score: ", lr.history.epoch_log['val_score'])
