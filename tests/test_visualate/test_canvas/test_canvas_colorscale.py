#!/usr/bin/env python3
# =========================================================================== #
#                         TEST CANVAS COLOR SCALES                            # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_colorscale.py                                            #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 8:08:35 pm                         #
# Last Modified: Tuesday December 17th 2019, 8:38:39 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasColorScale"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorScale

# --------------------------------------------------------------------------- #
#                               CanvasColorScale                        #
# --------------------------------------------------------------------------- #
class CanvasColorScaleTests:

    @mark.canvas
    @mark.canvas_colorscale
    @mark.canvas_colorscale_defaults
    def test_canvas_colorscale_defaults(self):
        DEFAULTS = {
            'colorscale_sequential' : [[0, 'rgb(220,220,220)'], 
                                    [0.2, 'rgb(245,195,157)'], 
                                    [0.4, 'rgb(245,160,105)'], 
                                    [1, 'rgb(178,10,28)'], ],
            'colorscale_sequentialminus' : [[0, 'rgb(5,10,172)'], 
                                            [0.35, 'rgb(40,60,190)'], 
                                            [0.5, 'rgb(70,100,245)'], 
                                            [0.6, 'rgb(90,120,245)'], 
                                            [0.7, 'rgb(106,137,247)'], 
                                            [1, 'rgb(220,220,220)'], ],
            'colorscale_diverging' : [[0, 'rgb(5,10,172)'], 
                                    [0.35, 'rgb(106,137,247)'], 
                                    [0.5, 'rgb(190,190,190)'], 
                                    [0.6, 'rgb(220,170,132)'], 
                                    [0.7, 'rgb(230,145,90)'], 
                                    [1, 'rgb(178,10,28)'], ],
            'colorway' : ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }        
        canvas = CanvasColorScale()
        expected_set = set(map(tuple, DEFAULTS['colorscale_sequential']))
        actual_set = set(map(tuple, canvas.colorscale_sequential))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorscale_sequential not initialized."

        expected_set = set(map(tuple, DEFAULTS['colorscale_sequentialminus']))
        actual_set = set(map(tuple, canvas.colorscale_sequentialminus))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorscale_sequentialminus not initialized."

        expected_set = set(map(tuple, DEFAULTS['colorscale_diverging']))
        actual_set = set(map(tuple, canvas.colorscale_diverging))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorscale_diverging not initialized."

        expected_set = set(map(tuple, DEFAULTS['colorway']))
        actual_set = set(map(tuple, canvas.colorway))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorway not initialized."

    @mark.canvas
    @mark.canvas_colorscale
    @mark.canvas_colorscale_sequential_update
    def test_canvas_colorscale_sequential_update(self):
        colorscales = [[0, 'blue'], [0, 'green'], [0, 'red']]
        colorway = ['blue', 'green', 'yellow', 'orange']
        canvas = CanvasColorScale()
        canvas.colorscale_sequential = colorscales
        canvas.colorscale_sequentialminus = colorscales
        canvas.colorscale_diverging = colorscales
        canvas.colorway = colorway

        expected_set = set(map(tuple, colorscales))
        actual_set = set(map(tuple, canvas.colorscale_sequential))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorscale_sequential not updated."

        actual_set = set(map(tuple, canvas.colorscale_sequentialminus))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorscale_sequentialminus not updated."        

        actual_set = set(map(tuple, canvas.colorscale_diverging))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorscale_diverging not updated." 

        expected_set = set(map(tuple, colorway))
        actual_set = set(map(tuple, canvas.colorway))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.colorway not updated."               
