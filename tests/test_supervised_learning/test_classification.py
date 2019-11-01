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

from ml_studio.supervised_learning.classification import LogisticRegression
from ml_studio.supervised_learning.classification import MulticlassClassification
from ml_studio.supervised_learning.training.cost import BinaryCrossEntropy
from ml_studio.supervised_learning.training.cost import CategoricalCrossEntropy
from ml_studio.supervised_learning.training.metrics import Metric
# --------------------------------------------------------------------------- #
#%%
# Logistic Regression
class LogisticRegressionTests:

    @mark.logistic_regression
    def test_logistic_regression_validation(self, get_binary_classification_data):
        X, _ = get_binary_classification_data
        clf = LogisticRegression(epochs=50)        
        with pytest.raises(Exception):
            clf._predict(X)

    @mark.logistic_regression
    def test_logistic_regression_predict(self, get_binary_classification_data):
        X, y = get_binary_classification_data
        clf = LogisticRegression(epochs=1000)        
        clf.fit(X,y)
        y_pred = clf.predict(X)
        y = np.multiply(y,1)
        score = clf.score(X,y)
        assert all((y_pred==0)|(y_pred==1)), "Prediction values not 1 or 0."
        assert score > 0.9, "Predictions are not close."
