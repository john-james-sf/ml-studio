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
from pytest import mark
import numpy as np
from sklearn.datasets import make_regression
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visualate.model_evaluation.optimization import TrainingCurve
from ml_studio.visualate.model_evaluation.optimization import LearningCurve
from ml_studio.visualate.model_evaluation.optimization import ScalabilityCurve
from ml_studio.visualate.model_evaluation.optimization import ProductivityCurve
from ml_studio.model_evaluation.optimization import KFoldCV
# --------------------------------------------------------------------------- #
#                            RESIDUAL PLOT                                    #
# --------------------------------------------------------------------------- #
class OptimizationPlotTests:

    @mark.optimization
    @mark.training_curve
    def test_training_curve(self, get_regression_data):
        X_train, y_train = get_regression_data
        model = LinearRegression(epochs=1000, metric='mape')                       
        v = TrainingCurve(model=model)
        v.fit(X_train, y_train)
        #v.show(dataset='score')            


    @mark.optimization
    @mark.kfold
    def test_kfold_cv(self, get_generated_medium_regression_data):
        X_train, y_train = get_generated_medium_regression_data
        model = LinearRegression(epochs=500, batch_size=32, metric='r2', 
                                 verbose=False, val_size=0, early_stop=False)                       
        sizes = np.arange(start=100,stop=1100,step=100, dtype=np.int32)        
        k = 5
        est = KFoldCV(model=model, sizes=sizes, k=k)
        est.fit(X_train, y_train)


    @mark.optimization
    @mark.learning_curve
    def test_learning_curve(self, get_generated_medium_regression_data):
        X_train, y_train = get_generated_medium_regression_data
        model = LinearRegression(epochs=500, batch_size=32, metric='r2', 
                                 verbose=False, val_size=0, early_stop=False)                       
        sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]         
        cv = 5
        est = LearningCurve(model=model,sizes=sizes, cv=cv)
        est.fit(X_train, y_train)
        #est.show()

    @mark.optimization
    @mark.scalability_curve
    def test_scalability_curve(self, get_generated_medium_regression_data):
        X_train, y_train = get_generated_medium_regression_data
        model = LinearRegression(epochs=500, batch_size=32, metric='r2', 
                                 verbose=False, val_size=0, early_stop=False)                       
        sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]         
        cv = 5
        est = ScalabilityCurve(model=model,sizes=sizes, cv=cv)
        est.fit(X_train, y_train)
        #est.show()        
        
    @mark.optimization
    @mark.productivity_curve
    def test_productivity_curve(self, get_generated_medium_regression_data):
        X_train, y_train = get_generated_medium_regression_data
        model = LinearRegression(epochs=500, batch_size=32, metric='r2', 
                                 verbose=False, val_size=0, early_stop=False)                       
        sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]         
        cv = 5
        est = ProductivityCurve(model=model,sizes=sizes, cv=cv)
        est.fit(X_train, y_train)
        #est.show()             