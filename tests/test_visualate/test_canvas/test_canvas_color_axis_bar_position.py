#!/usr/bin/env python3
# =========================================================================== #
#                   TEST CANVAS COLOR AXIS BAR POSITION                       # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_bar_position.py                               #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 9:21:42 pm                         #
# Last Modified: Tuesday December 17th 2019, 9:22:29 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasColorAxisBarPosition"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarPosition

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarPosition                             #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarPositionTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_position
    @mark.canvas_color_axis_bar_position_defaults
    def test_canvas_color_axis_bar_position_defaults(self):
        canvas = CanvasColorAxisBarPosition()        
        assert canvas.coloraxis_colorbar_x == 1.02, \
            "canvas.coloraxis_colorbar_x not initialized"
        assert canvas.coloraxis_colorbar_y == 0.5, \
            "canvas.coloraxis_colorbar_y not initialized"
        assert canvas.coloraxis_colorbar_xanchor == 'left', \
            "canvas.coloraxis_colorbar_xanchor not initialized"
        assert canvas.coloraxis_colorbar_yanchor == 'middle', \
            "canvas.coloraxis_colorbar_yanchor not initialized"
        assert canvas.coloraxis_colorbar_xpad == 10, \
            "canvas.coloraxis_colorbar_xpad not initialized"
        assert canvas.coloraxis_colorbar_ypad == 10, \
            "canvas.coloraxis_colorbar_ypad not initialized"            

    @mark.canvas
    @mark.canvas_color_axis_bar_position
    @mark.canvas_color_axis_bar_position_validation
    def test_canvas_color_axis_bar_position_validation(self):
        canvas = CanvasColorAxisBarPosition()
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_x = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_y = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_xanchor = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_yanchor = -2
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_xpad = 'red'            
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_ypad = 'red'                        

    @mark.canvas
    @mark.canvas_color_axis_bar_position
    @mark.canvas_color_axis_bar_position_update
    def test_canvas_color_axis_bar_position_update(self):        
        canvas = CanvasColorAxisBarPosition()
        canvas.coloraxis_colorbar_x = 2
        canvas.coloraxis_colorbar_y = 2
        canvas.coloraxis_colorbar_xanchor = 'right'        
        canvas.coloraxis_colorbar_yanchor = 'middle'
        canvas.coloraxis_colorbar_xpad = 2
        canvas.coloraxis_colorbar_ypad = 2

        assert canvas.coloraxis_colorbar_x == 2,"canvas.coloraxis_colorbar_x not updated."
        assert canvas.coloraxis_colorbar_y == 2,"canvas.coloraxis_colorbar_y not updated."
        assert canvas.coloraxis_colorbar_xanchor == 'right',"canvas.coloraxis_colorbar_xanchor not updated."
        assert canvas.coloraxis_colorbar_yanchor == 'middle',"canvas.coloraxis_colorbar_yanchor not updated."
        assert canvas.coloraxis_colorbar_xpad == 2,"canvas.coloraxis_colorbar_xpad not updated."
        assert canvas.coloraxis_colorbar_ypad == 2,"canvas.coloraxis_colorbar_ypad not updated."


    
