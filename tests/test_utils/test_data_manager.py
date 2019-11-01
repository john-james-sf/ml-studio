# --------------------------------------------------------------------------- #
#                              TEST DATA MANAGER                              #
# --------------------------------------------------------------------------- #
#%%
import numpy as np
import pytest
from pytest import mark
from sklearn import preprocessing

from ml_studio.utils import data_manager

class StandardScalerTests:

    @mark.transformer
    def test_standard_scaler(self, get_regression_data):
        X, y = get_regression_data
        ml_scaler = data_manager.StandardScaler()        
        sk_scaler = preprocessing.StandardScaler()
        
        ml_scaler.fit(X)
        X_ml = ml_scaler.transform(X)
        
        sk_scaler.fit(X)
        X_sk = sk_scaler.transform(X)
        
        assert np.isclose(X_ml.any(),X_sk.any(),rtol=1e-2), "Scaler's not close"
