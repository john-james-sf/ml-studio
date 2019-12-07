#!/usr/bin/env python3
# =========================================================================== #
#                            DATA PREPARATION                                 #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_data_preparation.py                                             #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 6th 2019, 10:48:21 pm                          #
# Last Modified: Friday December 6th 2019, 10:48:41 pm                        #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test data preparation"""
from pytest import mark
import numpy as np

from ml_studio.visualate.data_preparation.data_explorer import generate_data
# --------------------------------------------------------------------------- #
class DataPreparationTests:

    @mark.data_preparation
    @mark.data_explorer
    def test_data_explorer(self):
        datasets = ['california_housing', 'msd', 'online_news', 'speed_dating', 'regression']
        X_shape_0 = [20640, 515345 ,39644, 8378, 1000]
        X_shape_1 = [8, 90, 60, 194, 50]
        n_samples = 1000
        n_features = 50
        seed = 5

        for i, d in enumerate(datasets):
            X, y = generate_data(d, n_samples, n_features, seed)            
            assert X.shape[0] == X_shape_0[i], "Expected %d samples. Observed %d samples" % (X_shape_0[i], X.shape[0])
            assert X.shape[1] == X_shape_1[i], "Expected %d features. Observed %d features" % (X_shape_1[i], X.shape[1])

