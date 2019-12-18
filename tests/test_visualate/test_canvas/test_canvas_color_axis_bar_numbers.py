#!/usr/bin/env python3
# =========================================================================== #
#                   TEST CANVAS COLOR AXIS BAR NUMBERSS                       # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_bar_numbers.py                                #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 10:33:01 pm                        #
# Last Modified: Tuesday December 17th 2019, 10:34:06 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasColorAxisBarNumbers"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarNumbers

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarNumbers                       #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarNumbersTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_numbers
    @mark.canvas_color_axis_bar_numbers_defaults
    def test_canvas_color_axis_bar_numbers_font_defaults(self):
        canvas = CanvasColorAxisBarNumbers()        
        assert canvas.coloraxis_colorbar_separatethousands == True, \
            "canvas.coloraxis_colorbar_separatethousands not initialized"
        assert canvas.coloraxis_colorbar_exponentformat == 'B', \
            "canvas.coloraxis_colorbar_exponentformat not initialized"
        assert canvas.coloraxis_colorbar_showexponent == 'all', \
            "canvas.coloraxis_colorbar_showexponent not initialized"            

    @mark.canvas
    @mark.canvas_color_axis_bar_numbers
    @mark.canvas_color_axis_bar_numbers_validation
    def test_canvas_color_axis_bar_numbers_validation(self):
        canvas = CanvasColorAxisBarNumbers()
        with pytest.raises(TypeError):
            canvas.coloraxis_colorbar_separatethousands = 2
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_exponentformat = 'x'
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_showexponent = 22                                 

    @mark.canvas
    @mark.canvas_color_axis_bar_numbers_font
    @mark.canvas_color_axis_bar_numbers_font_update
    def test_canvas_color_axis_bar_numbers_font_update(self):        
        canvas = CanvasColorAxisBarNumbers()
        canvas.coloraxis_colorbar_separatethousands = False
        canvas.coloraxis_colorbar_exponentformat = 'SI'
        canvas.coloraxis_colorbar_showexponent = "first"

        assert canvas.coloraxis_colorbar_separatethousands == False, "canvas.coloraxis_colorbar_separatethousands not updated."
        assert canvas.coloraxis_colorbar_exponentformat == "SI", "canvas.coloraxis_colorbar_exponentformat not updated."
        assert canvas.coloraxis_colorbar_showexponent == "first", "canvas.coloraxis_colorbar_showexponent not updated."












    
