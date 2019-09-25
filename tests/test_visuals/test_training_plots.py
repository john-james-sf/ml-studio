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

from ml_studio.operations.early_stop import EarlyStopGeneralizationLoss
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.supervised_learning.regression import RidgeRegression
from ml_studio.visual.plots import plot_loss, plot_score, plot_learning_curves
directory = "./tests/test_visuals/test_figures/"

class TrainingPlotTests:
    
    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_loss_only(self, get_regression_data):
        X, y = get_regression_data
        model = LinearRegression(learning_rate=0.01, epochs=50, metric=None, 
                              seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        with pytest.raises(UserWarning):
            plot_score(model=model, directory=directory)

    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_val_loss_only(self, get_regression_data):
        X, y = get_regression_data
        es = EarlyStopGeneralizationLoss()
        model = LassoRegression(learning_rate=0.01, epochs=50, verbose=True,
                                   alpha=0.01, early_stop=es, seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        with pytest.raises(UserWarning):
            plot_score(model=model, directory=directory)        

    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_loss_scores_only(self, get_regression_data):
        X, y = get_regression_data
        model = RidgeRegression(learning_rate=0.01, epochs=50, verbose=True, 
                                   alpha=0.01, metric='root_mean_squared_error',
                                   seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        plot_score(model=model, directory=directory)

    @mark.plots
    @pytest.mark.skip(reason="working: avoid pop-ups")
    def test_training_plots_train_val_loss_and_scores(self, get_regression_data):
        X, y = get_regression_data
        es = EarlyStopGeneralizationLoss()
        model = LinearRegression(learning_rate=0.01, epochs=50, verbose=True, 
                              batch_size=32, metric='root_mean_squared_error',
                              early_stop=es, seed=50)
        model.fit(X,y)
        # Render Plots
        plot_loss(model=model, directory=directory)
        plot_score(model=model, directory=directory)

    @mark.plots
    @mark.learning_curves
    def test_learning_curves(self, get_regression_data):
        X, y = get_regression_data
        models = []
        bgd = LinearRegression(epochs=200, seed=50)
        sgd = LinearRegression(epochs=200, seed=50, batch_size=1)
        mgd = LinearRegression(epochs=200, seed=50, batch_size=32)
        models.append(bgd.fit(X,y))
        models.append(sgd.fit(X,y))
        models.append(mgd.fit(X,y))
        plot_learning_curves(models=models, directory=directory)


