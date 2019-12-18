#!/usr/bin/env python3
# =========================================================================== #
#                   TEST CANVAS COLOR AXIS BAR NUMBERSS                       # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_axis_title.py                                #
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
"""Test CanvasColorAxisBarTitle"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisBarTitle

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisBarTitle                       #
# --------------------------------------------------------------------------- #
class CanvasColorAxisBarTitleTests:

    @mark.canvas
    @mark.canvas_color_axis
    @mark.canvas_color_axis_bar_title
    @mark.canvas_color_axis_bar_title_defaults
    def test_canvas_color_axis_bar_title_defaults(self):
        canvas = CanvasColorAxisBarTitle()        
        assert canvas.coloraxis_colorbar_title_text == "", \
            "canvas.coloraxis_colorbar_title_text not initialized"
        assert canvas.coloraxis_colorbar_title_font_family == None, \
            "canvas.coloraxis_colorbar_title_font_family not initialized"
        assert canvas.coloraxis_colorbar_title_font_size == 1, \
            "canvas.coloraxis_colorbar_title_font_size not initialized"
        assert canvas.coloraxis_colorbar_title_font_color == None, \
            "canvas.coloraxis_colorbar_title_font_color not initialized"            
        assert canvas.coloraxis_colorbar_title_side == "top", \
            "canvas.coloraxis_colorbar_title_side not initialized"            


    @mark.canvas
    @mark.canvas_color_axis_bar_title
    @mark.canvas_color_axis_bar_title_validation
    def test_canvas_color_axis_title_validation(self):
        canvas = CanvasColorAxisBarTitle()
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_title_font_size = 0
        with pytest.raises(ValueError):
            canvas.coloraxis_colorbar_title_side = 'x'

    @mark.canvas
    @mark.canvas_color_axis_title_font
    @mark.canvas_color_axis_title_font_update
    def test_canvas_color_axis_title_font_update(self):        
        canvas = CanvasColorAxisBarTitle()
        canvas.coloraxis_colorbar_title_text = "Some title"
        canvas.coloraxis_colorbar_title_font_family = "Open Sans"
        canvas.coloraxis_colorbar_title_font_size = 10
        canvas.coloraxis_colorbar_title_font_color = "green"
        canvas.coloraxis_colorbar_title_side = "right"

        assert canvas.coloraxis_colorbar_title_text == "Some title", "canvas.coloraxis_colorbar_title_text not updated."
        assert canvas.coloraxis_colorbar_title_font_family == "Open Sans", "canvas.coloraxis_colorbar_title_font_family not updated."
        assert canvas.coloraxis_colorbar_title_font_size == 10, "canvas.coloraxis_colorbar_title_font_size not updated."
        assert canvas.coloraxis_colorbar_title_font_color == "green", "canvas.coloraxis_colorbar_title_font_color not updated."
        assert canvas.coloraxis_colorbar_title_side == "right", "canvas.coloraxis_colorbar_title_side not updated."

        
