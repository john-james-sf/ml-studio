#!/usr/bin/env python3
# =========================================================================== #
#                              TEST CANVAS SIZE                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_size.py                                                  #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 7:46:33 pm                         #
# Last Modified: Tuesday December 17th 2019, 7:46:48 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Test CanvasSize"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasSize

# --------------------------------------------------------------------------- #
#                               CanvasSize                                   #
# --------------------------------------------------------------------------- #
class CanvasSizeTests:

    @mark.canvas
    @mark.canvas_size
    @mark.canvas_size_defaults
    def test_canvas_size_defaults(self):
        canvas = CanvasSize()
        assert canvas.size_autosize == True,"canvas.size_left not initialized."
        assert canvas.size_width == 700,"canvas.size_top not initialized."
        assert canvas.size_height == 450,"canvas.size_bottom not initialized."


    @mark.canvas
    @mark.canvas_size
    @mark.canvas_size_validation
    def test_canvas_size_validation(self):
        canvas = CanvasSize()
        with pytest.raises(ValueError):
            canvas.size_autosize = 'x'
        with pytest.raises(ValueError):
            canvas.size_width = 'x'            
        with pytest.raises(ValueError):
            canvas.size_height = 'x'
            

    def test_canvas_size_update(self):
        canvas = CanvasSize()
        canvas.size_autosize = False
        canvas.size_width =1000
        canvas.size_height = 500

        assert canvas.size_autosize == False , 'canvas.size_autosize not updated.'
        assert canvas.size_width == 100 , 'canvas.size_width not updated.'
        assert canvas.size_height == 500 , 'canvas.size_height not updated.'
