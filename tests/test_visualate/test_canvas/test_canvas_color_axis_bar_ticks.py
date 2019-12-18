#!/usr/bin/env python3
# =========================================================================== #
#                   TEST CANVAS COLOR AXIS BAR TICKS                          # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_bar_ticks.py                                  #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 9:47:18 pm                         #
# Last Modified: Tuesday December 17th 2019, 9:47:32 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasColorAxisBarTicks"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarTicks

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarTicks                           #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarTicksTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_ticks
    @mark.canvas_color_axis_bar_ticks_defaults
    def test_canvas_color_axis_bar_ticks_defaults(self):
        canvas = CanvasColorAxisBarTicks()        
        assert canvas.coloraxis_colorbar_tickmode == 'array', \
            "canvas.coloraxis_colorbar_tickmode not initialized"
        assert canvas.coloraxis_colorbar_nticks == 0, \
            "canvas.coloraxis_colorbar_nticks not initialized"
        assert canvas.coloraxis_colorbar_tick0 == None, \
            "canvas.coloraxis_colorbar_tick0 not initialized"            
        assert canvas.coloraxis_colorbar_dtick == None, \
            "canvas.coloraxis_colorbar_dtick not initialized"            
        assert canvas.coloraxis_colorbar_tickvals == None, \
            "canvas.coloraxis_colorbar_tickvals not initialized"                        
        assert canvas.coloraxis_colorbar_ticktext == "", \
            "canvas.coloraxis_colorbar_ticktext not initialized"                                    
        assert canvas.coloraxis_colorbar_ticks == None, \
            "canvas.coloraxis_colorbar_tickts not initialized"                                                

    @mark.canvas
    @mark.canvas_color_axis_bar_ticks
    @mark.canvas_color_axis_bar_ticks_validation
    def test_canvas_color_axis_bar_ticks_validation(self):
        canvas = CanvasColorAxisBarTicks()
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_tickmode = 22
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_nticks = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_ticks = 22

    @mark.canvas
    @mark.canvas_color_axis_bar_ticks
    @mark.canvas_color_axis_bar_ticks_update
    def test_canvas_color_axis_bar_ticks_update(self):        
        canvas = CanvasColorAxisBarTicks()
        canvas.coloraxis_colorbar_tickmode = 'linear'
        canvas.coloraxis_colorbar_nticks = 2
        canvas.coloraxis_colorbar_tick0 = 2        
        canvas.coloraxis_colorbar_dtick = 2
        canvas.coloraxis_colorbar_tickvals = 2
        canvas.coloraxis_colorbar_ticktext = 2
        canvas.coloraxis_colorbar_ticks = 'inside'

        assert canvas.coloraxis_colorbar_tickmode == 'linear', "canvas.coloraxis_colorbar_tickmode not updated."
        assert canvas.coloraxis_colorbar_nticks == 2, "canvas.coloraxis_colorbar_nticks not updated."
        assert canvas.coloraxis_colorbar_tick0 == 2, "canvas.coloraxis_colorbar_tick0 not updated."
        assert canvas.coloraxis_colorbar_dtick == 2, "canvas.coloraxis_colorbar_dtick not updated."
        assert canvas.coloraxis_colorbar_tickvals == 2, "canvas.coloraxis_colorbar_tickvals not updated."
        assert canvas.coloraxis_colorbar_ticktext == 2, "canvas.coloraxis_colorbar_ticktext not updated."
        assert canvas.coloraxis_colorbar_ticks == 'inside', "canvas.coloraxis_colorbar_ticks not updated."





    
