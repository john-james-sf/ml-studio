#!/usr/bin/env python3
# =========================================================================== #
#                   TEST CANVAS COLOR AXIS BAR TICKS                          # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_bar_tick_style_.py                            #
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
"""Test CanvasColorAxisBarTickStyle"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarTickStyle

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarTickStyle                       #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarTickStyleTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_tick
    @mark.canvas_color_axis_bar_tick_style
    @mark.canvas_color_axis_bar_tick_style_defaults
    def test_canvas_color_axis_bar_tick_style_defaults(self):
        canvas = CanvasColorAxisBarTickStyle()        
        assert canvas.coloraxis_colorbar_ticklen == 5, \
            "canvas.coloraxis_colorbar_ticklen not initialized"
        assert canvas.coloraxis_colorbar_tickwidth == 1, \
            "canvas.coloraxis_colorbar_tickwidth not initialized"
        assert canvas.coloraxis_colorbar_tickcolor == '#444', \
            "canvas.coloraxis_colorbar_tickcolor not initialized"            
        assert canvas.coloraxis_colorbar_showticklabels == True, \
            "canvas.coloraxis_colorbar_showticklabels not initialized"            
        assert canvas.coloraxis_colorbar_tickangle == "", \
            "canvas.coloraxis_colorbar_tickangle not initialized"                        
        assert canvas.coloraxis_colorbar_tickprefix == "", \
            "canvas.coloraxis_colorbar_tickprefix not initialized"                                    
        assert canvas.coloraxis_colorbar_showtickprefix == "all", \
            "canvas.coloraxis_colorbar_showtickprefix not initialized"                                                
        assert canvas.coloraxis_colorbar_ticksuffix == "", \
            "canvas.coloraxis_colorbar_ticksuffix not initialized"                                    
        assert canvas.coloraxis_colorbar_showticksuffix == "all", \
            "canvas.coloraxis_colorbar_showticksuffix not initialized"                                                            

    @mark.canvas
    @mark.canvas_color_axis_bar_tick_style
    @mark.canvas_color_axis_bar_tick_style_validation
    def test_canvas_color_axis_bar_tick_style_validation(self):
        canvas = CanvasColorAxisBarTickStyle()
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_ticklen = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_tickwidth = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_colorbar_showticklabels = 22
        with pytest.raises(TypeError):
            canvas.coloraxis_colorbar_tickangle = 'x'    
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_showtickprefix = 'x'                        
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_showticksuffix = 'x'                                    

    @mark.canvas
    @mark.canvas_color_axis_bar_tick_style
    @mark.canvas_color_axis_bar_tick_style_update
    def test_canvas_color_axis_bar_tick_style_update(self):        
        canvas = CanvasColorAxisBarTickStyle()
        canvas.coloraxis_colorbar_ticklen = 10
        canvas.coloraxis_colorbar_tickwidth = 10
        canvas.coloraxis_colorbar_tickcolor = "blue"
        canvas.coloraxis_colorbar_showticklabels = False
        canvas.coloraxis_colorbar_tickangle = 90
        canvas.coloraxis_colorbar_tickprefix = "pre"
        canvas.coloraxis_colorbar_showtickprefix = "first"
        canvas.coloraxis_colorbar_ticksuffix = "suf"
        canvas.coloraxis_colorbar_showticksuffix = "last"

        assert canvas.coloraxis_colorbar_ticklen == 10, "coloraxis_colorbar_ticklen not updated."
        assert canvas.coloraxis_colorbar_tickwidth == 10, "coloraxis_colorbar_tickwidth not updated."
        assert canvas.coloraxis_colorbar_tickcolor == "blue", "coloraxis_colorbar_tickcolor not updated."
        assert canvas.coloraxis_colorbar_showticklabels == False, "coloraxis_colorbar_showticklabels not updated."
        assert canvas.coloraxis_colorbar_tickangle == 90, "coloraxis_colorbar_tickangle not updated."
        assert canvas.coloraxis_colorbar_tickprefix == "pre", "coloraxis_colorbar_tickprefix not updated."
        assert canvas.coloraxis_colorbar_showtickprefix == "first", "coloraxis_colorbar_showtickprefix not updated."
        assert canvas.coloraxis_colorbar_ticksuffix == "suf", "coloraxis_colorbar_ticksuffix not updated."
        assert canvas.coloraxis_colorbar_showticksuffix == "last", "coloraxis_colorbar_showticksuffix not updated."










    
