#!/usr/bin/env python3
# =========================================================================== #
#                            TEST HISTOGRAM                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_histogram.py                                                    #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 7th 2019, 1:51:34 pm                         #
# Last Modified: Wednesday December 11th 2019, 10:18:39 pm                    #
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

from ml_studio.visualate.data_analysis.normality import Histogram, histogram
from ml_studio.visualate.data_analysis.normality import DensityPlot, density_plot
from ml_studio.visualate.data_analysis.normality import NormalProbability
from ml_studio.utils.file_manager import cleanup
# --------------------------------------------------------------------------- #
#%%

class VisualateTests:

    @mark.normality
    @mark.histogram
    @mark.histogram0
    def test_histogram(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_0/"        
        cleanup(directory, ext=".html")
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram')
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports

    @mark.normality
    @mark.histogram
    @mark.histogram1
    def test_histogram_cumulative(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_1/"        
        cleanup(directory, ext=".html")
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram', cumulative=True)
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 2, "Expected 2 reports. Got %d" % reports        

    @mark.normality
    @mark.histogram
    @mark.histogram2
    def test_histogram_orientation(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_2/"        
        cleanup(directory, ext=".html")
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram', cumulative=True, orientation='h')
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 2, "Expected 2 reports. Got %d" % reports        

    @mark.normality
    @mark.histogram
    @mark.histogram3
    def test_histogram_marginal(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_3/"        
        cleanup(directory, ext=".html")
        x, y, z, df = get_regression_hastie
        v = Histogram(name='Histogram', cumulative=True, marginal='box')
        v.fit(x=x, y=y, z=z, dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports 

    @mark.normality
    @mark.histogram
    @mark.histogram4
    def test_histogram_function(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/histogram/test_4/"        
        cleanup(directory, ext=".html")
        x, y, z, df = get_regression_hastie
        histogram(x=x, y=y, z=z, dataset=df, orientation=None, directory=directory,
              marginal='box', cumulative=False, nbins=100,
              title='Histogram', name='histogram')
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports 

class DensityPlotTests:

    @mark.normality
    @mark.density
    @mark.density0
    def test_density_plot_1_group(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/density_plots/test_0/"        
        cleanup(directory, ext=".html")
        x, _, _, df = get_regression_hastie
        v = DensityPlot(name='Density Plot')
        v.fit(x=x,dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports                         

    @mark.normality
    @mark.density
    @mark.density1
    def test_density_plot_3_groups(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/density_plots/test_1/"        
        cleanup(directory, ext=".html")
        x, _, _, df = get_regression_hastie
        v = DensityPlot(name='Density Plot', bin_size=0.1)
        # Randomly select three columns 
        features = df.columns[:-1]
        x = np.random.choice(features, size=3, replace=False)
        v.fit(x=x,dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports                                 

    @mark.normality
    @mark.density
    @mark.density2
    def test_density_plot_4_groups(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/density_plots/test_2/"              
        cleanup(directory, ext=".html")
        x, _, _, df = get_regression_hastie
        v = DensityPlot(name='Density Plot')
        features = df.columns[:-1]
        x = np.random.choice(features, size=4, replace=False)
        v.fit(x=x,dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports    

class NormalProbabilityPlotTests:

    @mark.normality
    @mark.normal_probability
    @mark.normal_probability0
    def test_density_plot_1_group(self, get_regression_hastie):
        directory = "./tests/test_visualate/test_figures/normal_probability_plots/test_0/"        
        cleanup(directory, ext=".html")
        x, _, _, df = get_regression_hastie
        v = NormalProbability()
        v.fit(x=x,dataset=df)
        v.show(path=directory)
        reports = len(os.listdir(directory))
        assert reports == 1, "Expected 1 reports. Got %d" % reports   
                     