# =========================================================================== #
#                             TEST ANIMATIONS                                 #
# =========================================================================== #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visual.animations import MultiModelFit2D, MultiModelSearch3D
# --------------------------------------------------------------------------- #
# Constants
directory = "./tests/test_visuals/test_figures/"
# --------------------------------------------------------------------------- #
# Get data
#%%
X, y = datasets.load_boston(return_X_y=True)
scaler = StandardScaler()    
X = scaler.fit_transform(X)
X = X[:,5].reshape(-1,1)
# Train data
# --------------------------------------------------------------------------- #
plot = MultiModelSearch3D()
models = {}
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.8]
names = ['Very Low Learning Rate', 'Low Learning Rate', 'Good Learning Rate', 
         'High Learning Rate', 'Very High Learning Rate']

for i in range(len(learning_rates)):
    bgd = LinearRegression(epochs=200, learning_rate=learning_rates[i],
                           theta_init=[0,0], name=names[i])
    models[names[i]] = bgd.fit(X,y)

plot.search(models, directory=directory, filename='search_by_learning_rate.gif')
#%%
