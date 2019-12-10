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

from ml_studio.visualate.data_analysis.distribution import Histogram, histogram
from ml_studio.visualate.data_analysis.distribution import DensityPlot
# --------------------------------------------------------------------------- #
#%%

class HistogramTests:

    def cleanup(self, dirpath):
        if os.path.exists(dirpath):
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                try:
                    shutil.rmtree(filepath)
                except NotADirectoryError:
                    if filename.endswith(".html"):
                        os.remove(os.path.join(dirpath, filename))      

    @mark.histogram
    @mark.histogram0
    def test_histogram(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_0/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram')
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports

    @mark.histogram
    @mark.histogram1
    def test_histogram_cumulative(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_1/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram', cumulative=True)
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 2, "Expected 2 reports. Got %d" % reports        

    @mark.histogram
    @mark.histogram2
    def test_histogram_orientation(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_2/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram', cumulative=True, orientation='h')
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 2, "Expected 2 reports. Got %d" % reports        

    @mark.histogram
    @mark.histogram3
    def test_histogram_marginal(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_3/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram', cumulative=True, marginal='box')
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports 

    @mark.histogram
    @mark.histogram4
    def test_histogram_function(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_4/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        histogram(x=x, y=y, z=z, dataset=df, orientation=None, directory=directory,
              marginal='box', cumulative=False, nbins=100,
              title='Histogram', name='histogram')
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports 

    @mark.histogram
    @mark.density5
    def test_density_plot_1_group(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_5/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        v = DensityPlot(name='Density Plot')
        v.fit(x=x,dataset=df)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports                         

    @mark.histogram
    @mark.density6
    def test_density_plot_3_groups(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_6/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        v = DensityPlot(name='Density Plot', bin_size=0.1)
        # Randomly select three columns 
        features = df.columns[:-1]
        x = np.random.choice(features, size=3, replace=False)
        v.fit(x=x,dataset=df)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports                                 

    @mark.histogram
    @mark.density7
    def test_density_plot_4_groups(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_7/"        
        self.cleanup(directory)
        x, y, z, df = get_regression_hastie
        v = DensityPlot(name='Density Plot')
        features = df.columns[:-1]
        x = np.random.choice(features, size=4, replace=False)
        v.fit(x=x,dataset=df)
        v.show(directory=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports              