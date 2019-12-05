#!/usr/bin/env python3
# =========================================================================== #
#                                  BASE                                       #
# =========================================================================== #
# =========================================================================== #
# Project: Visualate                                                          #
# Version: 0.1.0                                                              #
# File: \test_validity.py                                                     #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Thursday November 28th 2019, 6:34:05 am                        #
# Last Modified: Thursday November 28th 2019, 6:34:18 am                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test validity visualizations"""
from pytest import mark
import numpy as np

from ml_studio.supervised_learning.regression import LinearRegression

from ml_studio.visualate.regression.validity import Residuals
from ml_studio.visualate.regression.validity import StandardizedResiduals
from ml_studio.visualate.regression.validity import StudentizedResiduals
from ml_studio.visualate.regression.validity import ScaleLocation
from ml_studio.visualate.regression.validity import QQPlot
# --------------------------------------------------------------------------- #
#                            RESIDUAL PLOT                                    #
# --------------------------------------------------------------------------- #
class ResidualPlotTests:

    @mark.validity
    @mark.residuals
    def test_residual_plot(self, split_regression_data):
        X_train, X_test, y_train, y_test = split_regression_data
        model = LinearRegression(epochs=1000, metric='mape')                       
        v = Residuals(model=model)
        v.fit(X_train, y_train)
        v.score(X_test, y_test)
        v.show()

    @mark.validity
    @mark.standardized_residuals
    def test_standardized_residual_plot(self, split_regression_data):
        X_train, X_test, y_train, y_test = split_regression_data
        model = LinearRegression(epochs=1000, metric='r2')                       
        v = StandardizedResiduals(model=model)
        v.fit(X_train, y_train)
        v.score(X_test, y_test)
        v.show()        

    @mark.validity
    @mark.studentized_residuals
    def test_studentized_residual_plot(self, split_regression_data):
        X_train, X_test, y_train, y_test = split_regression_data
        model = LinearRegression(epochs=1000, metric='mae')                       
        v = StudentizedResiduals(model=model)
        v.fit(X_train, y_train)
        v.score(X_test, y_test)
        v.show()         

    @mark.validity
    @mark.scale_location
    def test_scale_location_plot(self, split_regression_data):
        X_train, X_test, y_train, y_test = split_regression_data
        model = LinearRegression(epochs=1000, metric='mae')                       
        v = ScaleLocation(model=model)
        v.fit(X_train, y_train)
        v.score(X_test, y_test)
        v.show()          

    @mark.validity
    @mark.qq_plot
    def test_qq_plot_plot(self, split_regression_data):
        X_train, _, y_train, _, = split_regression_data
        model = LinearRegression(epochs=1000, metric='mae')                       
        v = QQPlot(model=model)
        v.fit(X_train, y_train)
        v.show()               