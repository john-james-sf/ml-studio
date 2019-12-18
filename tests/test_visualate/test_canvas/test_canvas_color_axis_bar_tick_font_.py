#!/usr/bin/env python3
# =========================================================================== #
#                   TEST CANVAS COLOR AXIS BAR TICKS                          # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_bar_tick_font_.py                            #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 10:01:59 pm                        #
# Last Modified: Tuesday December 17th 2019, 10:03:34 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasColorAxisBarTickFont"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarTickFont

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarTickFont                       #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarTickFontTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_tick
    @mark.canvas_color_axis_bar_tick_font
    @mark.canvas_color_axis_bar_tick_font_defaults
    def test_canvas_color_axis_bar_tick_font_defaults(self):
        canvas = CanvasColorAxisBarTickFont()        
        assert canvas.coloraxis_colorbar_tickfont_family == None, \
            "canvas.coloraxis_colorbar_tickfont_family not initialized"
        assert canvas.coloraxis_colorbar_tickfont_size == 1, \
            "canvas.coloraxis_colorbar_tickfont_size not initialized"
        assert canvas.coloraxis_colorbar_tickfont_color == None, \
            "canvas.coloraxis_colorbar_tickfont_color not initialized"            

    @mark.canvas
    @mark.canvas_color_axis_bar_tick_font
    @mark.canvas_color_axis_bar_tick_font_validation
    def test_canvas_color_axis_bar_tick_font_validation(self):
        canvas = CanvasColorAxisBarTickFont()
        with pytest.raises(TypeError):
            canvas.coloraxis_colorbar_tickfont_family = 2
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_tickfont_size = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_colorbar_tickfont_color = 22                                 

    @mark.canvas
    @mark.canvas_color_axis_bar_tick_font
    @mark.canvas_color_axis_bar_tick_font_update
    def test_canvas_color_axis_bar_tick_font_update(self):        
        canvas = CanvasColorAxisBarTickFont()
        canvas.coloraxis_colorbar_tickfont_family = 'Times'
        canvas.coloraxis_colorbar_tickfont_size = 10
        canvas.coloraxis_colorbar_tickfont_color = "blue"

        assert canvas.coloraxis_colorbar_tickfont_family == 'Times', "canvas.coloraxis_colorbar_tickfont_family not updated."
        assert canvas.coloraxis_colorbar_tickfont_size == 10, "canvas.coloraxis_colorbar_tickfont_size not updated."
        assert canvas.coloraxis_colorbar_tickfont_color == "blue", "canvas.coloraxis_colorbar_tickfont_color not updated."












    
