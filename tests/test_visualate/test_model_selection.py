#!/usr/bin/env python3
# =========================================================================== #
#                                OPTIMIZATION                                 #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_optimization.py                                                 #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Sunday December 1st 2019, 9:22:07 pm                           #
# Last Modified: Sunday December 1st 2019, 9:22:27 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test data influence observations"""
import numpy as np
import pytest
from pytest import mark
from sklearn.datasets import make_regression
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visualate.model_evaluation.optimization import TrainingCurve
from ml_studio.visualate.model_evaluation.optimization import LearningCurve
from ml_studio.visualate.model_evaluation.optimization import ScalabilityCurve
from ml_studio.visualate.model_evaluation.optimization import ProductivityCurve
from ml_studio.visualate.model_selection.search import CVLinePlot
# --------------------------------------------------------------------------- #
#                            RESIDUAL PLOT                                    #
# --------------------------------------------------------------------------- #
class ModelSelectionPlotTests:

    @mark.model_selection
    @mark.cv_lineplot_validation
    def test_cv_lineplot_validation(self, get_regression_data):
        X, y = get_regression_data
        model = LinearRegression()
        learning_rate = np.logspace(1e-4,1)
        param_grid = {'learning_rates': learning_rate}
        with pytest.raises(ValueError):
            v = CVLinePlot(model=model, param_grid=param_grid)
            v.fit(X,y)

    @mark.model_selection
    @mark.cv_lineplot
    def test_cv_lineplot(self, get_regression_data):
        X, y = get_regression_data
        model = LinearRegression()
        learning_rate = np.logspace(1e-4,1)
        early_stop = [True, False]
        verbose = [True, False]
        nominal = ['hat', 'shirt', 'jacket', 'candy']
        param_grid = {'learning_rate': learning_rate,
                      'early_stop' : early_stop,
                      'nominal': nominal,
                      'verbose': verbose}
        s = CVLinePlot(model, param_grid)
        s.fit(X, y)
        s.show()
            