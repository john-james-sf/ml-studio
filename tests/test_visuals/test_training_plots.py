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

from ml_studio.supervised_learning.regression import SGDRegression
from ml_studio.supervised_learning.regression import SGDLassoRegression
from ml_studio.supervised_learning.regression import SGDRidgeRegression
from ml_studio.visual.plots import plot_loss, plot_score
directory = "./tests/test_visuals/test_figures/"

class TrainingPlotTests:
    
    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_loss_only(self, get_regression_data):
        X, y = get_regression_data
        model = SGDRegression(learning_rate=0.01, epochs=50, metric=None, 
                              val_size=0, seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        with pytest.raises(UserWarning):
            plot_score(model=model, directory=directory)

    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_val_loss_only(self, get_regression_data):
        X, y = get_regression_data
        model = SGDLassoRegression(learning_rate=0.01, epochs=50, verbose=True,
                                   alpha=0.01, metric=None, val_size=0.33, seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        with pytest.raises(UserWarning):
            plot_score(model=model, directory=directory)        

    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_loss_scores_only(self, get_regression_data):
        X, y = get_regression_data
        model = SGDRidgeRegression(learning_rate=0.01, epochs=50, verbose=True, 
                                   alpha=0.01, metric='root_mean_squared_error',
                                   val_size=0.0, seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        plot_score(model=model, directory=directory)

    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_val_loss_and_scores(self, get_regression_data):
        X, y = get_regression_data
        model = SGDRegression(learning_rate=0.01, epochs=50, verbose=True, 
                              batch_size=32, metric='root_mean_squared_error',
                              val_size=0.33, seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        plot_score(model=model, directory=directory)

