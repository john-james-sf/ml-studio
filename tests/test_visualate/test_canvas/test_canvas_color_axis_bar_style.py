#!/usr/bin/env python3
# =========================================================================== #
#                    TEST CANVAS COLOR AXIS BAR STYLE                         # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_bar_style.py                                  #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 9:03:31 pm                         #
# Last Modified: Tuesday December 17th 2019, 9:03:41 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasColorAxisBarStyle"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarStyle

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarStyle                             #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarStyleTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_style
    @mark.canvas_color_axis_bar_style_defaults
    def test_canvas_color_axis_bar_style_defaults(self):
        canvas = CanvasColorAxisBarStyle()        
        assert canvas.coloraxis_colorbar_thicknessmode == 'pixels', \
            "canvas.coloraxis_colorbar_thicknessmode not initialized"
        assert canvas.coloraxis_colorbar_thickness == 30, \
            "canvas.coloraxis_colorbar_thickness not initialized"
        assert canvas.coloraxis_colorbar_lenmode == 'fraction', \
            "canvas.coloraxis_colorbar_lenmode not initialized"
        assert canvas.coloraxis_colorbar_len == 1, \
            "canvas.coloraxis_colorbar_len not initialized"
        assert canvas.coloraxis_colorbar_bgcolor == 'rgba(0000)', \
            "canvas.coloraxis_colorbar_bgcolor not initialized"            

    @mark.canvas
    @mark.canvas_color_axis_bar_style
    @mark.canvas_color_axis_bar_style_validation
    def test_canvas_color_axis_bar_style_validation(self):
        canvas = CanvasColorAxisBarStyle()
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_thicknessmode = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_thickness = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_lenmode = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_len = -2
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_bgcolor = 'red'            

    @mark.canvas
    @mark.canvas_color_axis_bar_style
    @mark.canvas_color_axis_bar_style_update
    def test_canvas_color_axis_bar_style_update(self):        
        canvas = CanvasColorAxisBarStyle()
        canvas.coloraxis_colorbar_thicknessmode = 'fraction'
        canvas.coloraxis_colorbar_thickness = 22
        canvas.coloraxis_colorbar_lenmode = 'fraction'        
        canvas.coloraxis_colorbar_len = 22
        canvas.coloraxis_colorbar_bgcolor = 'green'        

        assert canvas.coloraxis_colorbar_thicknessmode == 'fraction'," canvas.coloraxis_colorbar_thicknessmode not updated."
        assert canvas.coloraxis_colorbar_thickness == 22," canvas.coloraxis_colorbar_thickness not updated."
        assert canvas.coloraxis_colorbar_lenmode == 'fraction'," canvas.coloraxis_colorbar_lenmode not updated."
        assert canvas.coloraxis_colorbar_len == 22," canvas.coloraxis_colorbar_len not updated."
        assert canvas.coloraxis_colorbar_bgcolor == 'green'," canvas.coloraxis_colorbar_bgcolor not updated."
    
