#!/usr/bin/env python3
# =========================================================================== #
#                              TEST CANVAS FONT                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_font.py                                                  #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 7:50:41 pm                         #
# Last Modified: Tuesday December 17th 2019, 7:50:52 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasFont"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasFont

# --------------------------------------------------------------------------- #
#                               CanvasFont                                    #
# --------------------------------------------------------------------------- #
class CanvasFontTests:

    @mark.canvas
    @mark.canvas_font
    @mark.canvas_font_defaults
    def test_canvas_font_defaults(self):
        canvas = CanvasFont()
        assert canvas.font_family == None,"canvas.font_family not initialized."
        assert canvas.font_size == 12,"canvas.font_size not initialized."
        assert canvas.font_color == '#444',"canvas.font_color not initialized."
        assert canvas.font_separators == '.,',"canvas.font_separators not initialized."


    @mark.canvas
    @mark.canvas_font
    @mark.canvas_font_validation
    def test_canvas_font_validation(self):
        canvas = CanvasFont()
        with pytest.raises(ValueError):
            canvas.font_size = 'x'
            

    def test_canvas_font_update(self):
        canvas = CanvasFont()
        canvas.font_family = "Open Sans"
        canvas.font_size = 18
        canvas.font_color = "blue"
        canvas.font_separators = "."

        assert canvas.font_family == "Open Sans", "canvas.font_family not updated."
        assert canvas.font_size == 18, "canvas.font_size not updated."
        assert canvas.font_color == "blue", "canvas.font_color not updated."
        assert canvas.font_separators == ".", "canvas.font_separators not updated."

