#!/usr/bin/env python3
# =========================================================================== #
#                      TEST CANVAS COLOR AXIS SCALES                          # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_scales.py                                          #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 8:39:24 pm                         #
# Last Modified: Tuesday December 17th 2019, 8:39:37 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Test CanvasColorAxisScales"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisScales

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisScales                             #
# --------------------------------------------------------------------------- #
class CanvasColorAxisScalesTests:

    @mark.canvas
    @mark.canvas_color_axis_scales
    @mark.canvas_color_axis_scales_defaults
    def test_canvas_color_axis_scales_defaults(self):
        colorscale = [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]
        canvas = CanvasColorAxisScales()

        expected_set = set(map(tuple, colorscale))
        actual_set = set(map(tuple, canvas.coloraxis_colorscale))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.coloraxis_colorscale not initialized."
        assert canvas.coloraxis_autoscale == True, "canvas.coloraxis_autoscale not initialized"
        assert canvas.coloraxis_reversescale == True, "canvas.coloraxis_reversescale not initialized"
        assert canvas.coloraxis_showscale == True, "canvas.coloraxis_showscale not initialized"        

    @mark.canvas
    @mark.canvas_color_axis_scales
    @mark.canvas_color_axis_scales_validation
    def test_canvas_color_axis_scales_validation(self):
        canvas = CanvasColorAxisScales()
        with pytest.raises(TypeError):
            canvas.coloraxis_colorscale = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_autoscale = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_reversescale = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_showscale = 'x'

    @mark.canvas
    @mark.canvas_color_axis_scales
    @mark.canvas_color_axis_scales_update
    def test_canvas_color_axis_scales_update(self):
        colorscale = [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]
        canvas = CanvasColorAxisScales()
        canvas.coloraxis_colorscale = colorscale
        canvas.coloraxis_autoscale = False
        canvas.coloraxis_reversescale = False
        canvas.coloraxis_showscale = False

        expected_set = set(map(tuple, colorscale))
        actual_set = set(map(tuple, canvas.coloraxis_colorscale))
        diff = expected_set.symmetric_difference(actual_set)
        assert len(diff) == 0, "canvas.coloraxis_colorscale not updated."        

        assert canvas.coloraxis_autoscale == False, "canvas.coloraxis_autoscale not updated"
        assert canvas.coloraxis_reversescale == False, "canvas.coloraxis_reversescale not updated"
        assert canvas.coloraxis_showscale == False, "canvas.coloraxis_showscale not updated"        
