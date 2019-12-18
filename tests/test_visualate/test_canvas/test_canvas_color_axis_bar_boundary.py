#!/usr/bin/env python3
# =========================================================================== #
#                   TEST CANVAS COLOR AXIS BAR BOUNDARY                       # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_bar_boundary.py                               #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 9:34:49 pm                         #
# Last Modified: Tuesday December 17th 2019, 9:34:58 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasColorAxisBarBoundary"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarBoundary

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarBoundary                        #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarBoundaryTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_boundary
    @mark.canvas_color_axis_bar_boundary_defaults
    def test_canvas_color_axis_bar_boundary_defaults(self):
        canvas = CanvasColorAxisBarBoundary()        
        assert canvas.coloraxis_colorbar_outlinecolor == '#444', \
            "canvas.coloraxis_colorbar_outlinecolor not initialized"
        assert canvas.coloraxis_colorbar_outlinewidth == 1, \
            "canvas.coloraxis_colorbar_outlinewidth not initialized"
        assert canvas.coloraxis_colorbar_bordercolor == '#444', \
            "canvas.coloraxis_colorbar_bordercolor not initialized"            
        assert canvas.coloraxis_colorbar_borderwidth == 0, \
            "canvas.coloraxis_colorbar_borderwidth not initialized"            

    @mark.canvas
    @mark.canvas_color_axis_bar_boundary
    @mark.canvas_color_axis_bar_boundary_validation
    def test_canvas_color_axis_bar_boundary_validation(self):
        canvas = CanvasColorAxisBarBoundary()
        with pytest.raises(TypeError):
            canvas.coloraxis_colorbar_outlinecolor = 22
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_outlinewidth = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_colorbar_bordercolor = 22
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_borderwidth = 'x'

    @mark.canvas
    @mark.canvas_color_axis_bar_boundary
    @mark.canvas_color_axis_bar_boundary_update
    def test_canvas_color_axis_bar_boundary_update(self):        
        canvas = CanvasColorAxisBarBoundary()
        canvas.coloraxis_colorbar_outlinecolor = 'blue'
        canvas.coloraxis_colorbar_outlinewidth = 2
        canvas.coloraxis_colorbar_bordercolor = 'green'        
        canvas.coloraxis_colorbar_borderwidth = 2

        assert canvas.coloraxis_colorbar_outlinecolor == 'blue', "canvas.coloraxis_colorbar_outlinecolor not updated."
        assert canvas.coloraxis_colorbar_outlinewidth == 2, "canvas.coloraxis_colorbar_outlinewidth not updated."
        assert canvas.coloraxis_colorbar_bordercolor == 'green', "canvas.coloraxis_colorbar_bordercolor not updated."
        assert canvas.coloraxis_colorbar_borderwidth == 2, "canvas.coloraxis_colorbar_borderwidth not updated."



    
