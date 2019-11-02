# --------------------------------------------------------------------------- #
#                           TEST CLASSIFICATION                               #
# --------------------------------------------------------------------------- #
#%%
# Imports
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark
from sklearn.datasets import load_wine

from ml_studio.supervised_learning.classification import MultinomialLogisticRegression
from ml_studio.utils.data_manager import StandardScaler
# --------------------------------------------------------------------------- #
#%%
# Multinomial Logistic Regression
X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
clf = MultinomialLogisticRegression(epochs=1000)        
clf.fit(X,y)
y_pred = clf.predict(X)
score = clf.score(X,y)
print(score)



# %%
