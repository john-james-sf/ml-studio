# --------------------------------------------------------------------------- #
#                               TEST PLOTS                                    #
# --------------------------------------------------------------------------- #
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visual.plots import TrainingPlots
from ml_studio.operations.metrics import Scorer
from ml_studio.utils.filemanager import save_csv


class TrainingPlotTests:

    @mark.plot
    def test_training_plots_validation(self,
                                       get_figure_path,
                                       train_algorithm_w_validation):

        algorithm = train_algorithm_w_validation
        directory = get_figure_path
        plots = TrainingPlots()

        with pytest.raises(TypeError):
            plots.cost_plot(algorithm='4')
        with pytest.raises(TypeError):
            plots.cost_plot(algorithm=algorithm, directory=5)
        with pytest.raises(TypeError):
            plots.cost_plot(algorithm=algorithm, filename=5)
        with pytest.raises(TypeError):
            plots.cost_plot(algorithm=algorithm, directory=directory,
                            filename=9)
        with pytest.raises(TypeError):
            plots.cost_plot(algorithm=algorithm, xlim='x')
        with pytest.raises(TypeError):
            plots.cost_plot(algorithm=algorithm, ylim='x')
        with pytest.raises(TypeError):
            plots.cost_plot(algorithm=algorithm, figsize='x')
        with pytest.raises(TypeError):
            plots.cost_plot(algorithm=algorithm, show='x')

    @mark.plot
    def test_cost_plot_no_validation(self,
                                     get_figure_path,
                                     train_algorithm_wo_validation):

        algorithm = train_algorithm_wo_validation
        directory = get_figure_path
        plots = TrainingPlots()
        plot = plots.cost_plot(algorithm, directory=directory)
        assert isinstance(plot, dict), "cost_plot did not return a dictionary"
        assert isinstance(plot['fig'], plt.Figure), "no figure returned"
        # Check plot file exists
        path = os.path.join(plot, directory, plot.filename)
        assert os.path.exists(path), "Plot was not saved to file"

    def test_cost_plot_validation(self,
                                  get_figure_path,
                                  train_algorithm_w_validation):

        algorithm = train_algorithm_w_validation
        directory = get_figure_path
        plots = TrainingPlots()
        plot = plots.cost_plot(algorithm, directory=directory)
        assert isinstance(plot, dict), "cost_plot did not return a dictionary"
        assert isinstance(plot['fig'], plt.Figure), "no figure returned"
        # Check plot file exists
        path = os.path.join(plot, directory, plot.filename)
        assert os.path.exists(path), "Plot was not saved to file"

    def test_score_plot_validation(self,
                                   get_figure_path,
                                   train_algorithm_w_validation):

        algorithm = train_algorithm_w_validation
        directory = get_figure_path
        plots = TrainingPlots()
        plot = plots.score_plot(algorithm, directory=directory)
        assert isinstance(plot, dict), "score_plot did not return a dictionary"
        assert isinstance(plot['fig'], plt.Figure), "no figure returned"
        # Check plot file exists
        path = os.path.join(plot, directory, plot.filename)
        assert os.path.exists(path), "Plot was not saved to file"
