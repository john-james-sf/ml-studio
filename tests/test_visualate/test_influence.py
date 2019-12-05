#!/usr/bin/env python3
# =========================================================================== #
#                                  BASE                                       #
# =========================================================================== #
# =========================================================================== #
# Project: Visualate                                                          #
# Version: 0.1.0                                                              #
# File: \test_influence.py                                                    #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Thursday November 28th 2019, 9:01:49 pm                        #
# Last Modified: Thursday November 28th 2019, 9:02:09 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test data influence observations"""
from pytest import mark
import numpy as np

from ml_studio.supervised_learning.regression import LinearRegression

from ml_studio.visualate.regression.influence import ResidualsLeverage
from ml_studio.visualate.regression.influence import CooksDistance
# --------------------------------------------------------------------------- #
#                            RESIDUAL PLOT                                    #
# --------------------------------------------------------------------------- #
class InfluencePlotTests:

    @mark.influence
    @mark.residuals_leverage
    def test_residuals_leverage_plot(self, split_regression_data):
        X_train, X_test, y_train, y_test = split_regression_data
        model = LinearRegression(epochs=1000, metric='mape')                       
        v = ResidualsLeverage(model=model)
        v.fit(X_train, y_train)
        v.score(X_test, y_test)
        v.show()

    @mark.influence
    @mark.cooks_distance
    def test_cooks_distance_plot(self, split_regression_data):
        X_train, X_test, y_train, y_test = split_regression_data
        model = LinearRegression(epochs=1000, metric='mape')                       
        v = CooksDistance(model=model)
        v.fit(X_train, y_train)
        v.score(X_test, y_test)
        v.show()
