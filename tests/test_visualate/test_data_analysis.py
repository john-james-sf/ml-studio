#!/usr/bin/env python3
# =========================================================================== #
#                            TEST DATA ANALYSIS                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_data_analysis.py                                                #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 7th 2019, 1:51:34 pm                         #
# Last Modified: Saturday December 7th 2019, 1:52:38 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test data analysis"""
#%%
import os
import shutil
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.data_analysis.univariate import Histogram, histogram
# --------------------------------------------------------------------------- #
#%%

class DataAnalysisTests:

    def cleanup(self, dirpath):
        if os.path.exists(dirpath):
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                try:
                    shutil.rmtree(filepath)
                except NotADirectoryError:
                    if filename.endswith(".html"):
                        os.remove(os.path.join(dirpath, filename))      

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram0
    def test_histogram_validation(self):
        X = 'a'
        y = 5
        with pytest.raises(TypeError):
            v = Histogram(dataset_name='Boston Housing Prices')
            v.fit(X, y)
        X = [1,2]
        y = [1,2,3]
        with pytest.raises(ValueError):
            v = Histogram(dataset_name='Boston Housing Prices')
            v.fit(X, y)                                         
        

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram1
    def test_histogram_with_array_no_title_with_directory(self, get_regression_data):
        directory = "./tests/test_visualate/test_figures/histogram/test_1/"        
        self.cleanup(directory)
        X, _ = get_regression_data
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 14, "Expected 14 reports. Got %d" % reports


    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram2
    def test_histogram_with_array_no_title_with_dirpath(self, get_regression_data):
        directory = "./tests/test_visualate/test_figures/histogram/test_2/"
        self.cleanup(directory)
        X, _ = get_regression_data
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 14, "Expected 14 reports. Got %d" % reports

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram3
    def test_histogram_with_array_with_title_with_dirpath(self, get_regression_data):
        kwargs = {'template': 'none'}
        directory = "./tests/test_visualate/test_figures/histogram/test_3/"
        self.cleanup(directory)
        X, _ = get_regression_data
        v = Histogram(title="Histogrammy3", dataset_name='Boston Housing Prices', **kwargs)
        v.fit(X)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 13, "Expected 13 reports. Got %d" % reports

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram4
    def test_histogram_with_array_with_title_with_directory(self, get_regression_data):
        directory = "./tests/test_visualate/test_figures/histogram/test_4/"
        self.cleanup(directory)
        X, _ = get_regression_data
        v = Histogram(dataset_name='Boston Housing Prices', title="Histogrammy4")
        v.fit(X)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 13, "Expected 13 reports. Got %d" % reports

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram5
    def test_histogram_with_1darray_with_title_with_directory(self, get_regression_data):
        directory = "./tests/test_visualate/test_figures/histogram/test_5/"
        self.cleanup(directory)
        X, _ = get_regression_data
        v = Histogram(dataset_name='Boston Housing Prices', title="Histogrammy5")
        v.fit(X[:,1])
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports    

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram6
    def test_histogram_with_1darray_wo_title_with_directory(self, get_regression_data):
        directory = "./tests/test_visualate/test_figures/histogram/test_6/"
        self.cleanup(directory)
        X, _ = get_regression_data
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X[:,1])
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports    

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram7
    def test_histogram_with_1darray_wo_title_with_dir(self, get_regression_data):
        directory = "./tests/test_visualate/test_figures/histogram/test_7/"
        self.cleanup(directory)
        X, _ = get_regression_data
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X[:,1])        
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports     

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram8
    def test_histogram_with_X_df_y_df(self, get_regression_data_df_plus):
        directory = "./tests/test_visualate/test_figures/histogram/test_8/"
        self.cleanup(directory)
        X, y, _, _ = get_regression_data_df_plus
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X,y)        
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 14, "Expected 14 reports. Got %d" % reports           

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram9
    def test_histogram_with_X_df_y_series(self, get_regression_data_df_plus):
        directory = "./tests/test_visualate/test_figures/histogram/test_9/"
        self.cleanup(directory)
        X, _, y, _ = get_regression_data_df_plus
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X,y)        
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 14, "Expected 14 reports. Got %d" % reports    

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram10
    def test_histogram_with_X_df_y_np(self, get_regression_data_df_plus):
        directory = "./tests/test_visualate/test_figures/histogram/test_10/"
        self.cleanup(directory)
        X, _, _, y = get_regression_data_df_plus
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X,y)        
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 14, "Expected 14 reports. Got %d" % reports                   

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram11
    def test_histogram_with_X_df_y_list(self, get_regression_data_df_plus):
        directory = "./tests/test_visualate/test_figures/histogram/test_11/"
        self.cleanup(directory)
        X, _, _, y = get_regression_data_df_plus
        v = Histogram(dataset_name='Boston Housing Prices')
        v.fit(X,list(y))        
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 14, "Expected 14 reports. Got %d" % reports             

    @mark.data_analysis
    @mark.univariate
    @mark.histogram
    @mark.histogram12
    def test_histogram_function(self, get_regression_data):
        directory = "./tests/test_visualate/test_figures/histogram/test_12/"
        self.cleanup(directory)
        X, y = get_regression_data
        histogram(X,y=y, dataset_name='Boston Housing Prices', directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 14, "Expected 14 reports. Got %d" % reports     

