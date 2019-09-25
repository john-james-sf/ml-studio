# =========================================================================== #
#                         TEST TRAINING PLOTS                                 #
# =========================================================================== #
#%%
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import mark
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visual.animations import SingleModelSearch3D, SingleModelFit2D
from ml_studio.visual.animations import MultiModelSearch3D, MultiModelFit2D
directory = "./tests/test_visuals/test_figures/"

class AnimationTests:
    
    @mark.animations
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_single_model_search(self, get_regression_data):
        X, y = get_regression_data
        X = X[:,5]
        X = np.reshape(X, (-1,1))
        model = LinearRegression(learning_rate=0.1, epochs=200, seed=50)
        model.fit(X,y)
        ani = SingleModelSearch3D()
        ani.search(model, directory)

    @mark.animations
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_single_model_fit(self, get_regression_data):
        X, y = get_regression_data
        X = X[:,5]
        X = np.reshape(X, (-1,1))
        model = LinearRegression(learning_rate=0.1, epochs=200, seed=50)
        model.fit(X,y)
        ani = SingleModelFit2D()
        ani.fit(model, directory) 

    @mark.animations
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_multi_model_search(self, fit_multiple_models):
        models = fit_multiple_models
        ani = MultiModelSearch3D()
        ani.search(models=models, directory=directory)               

    @mark.animations
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_multi_model_fit(self, fit_multiple_models):
        models = fit_multiple_models
        ani = MultiModelFit2D()
        ani.fit(models=models, directory=directory)         
