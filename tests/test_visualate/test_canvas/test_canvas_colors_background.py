#!/usr/bin/env python3
# =========================================================================== #
#                       TEST CANVAS BACKGROUND COLOR                         #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_background.py                                     #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 7:58:54 pm                         #
# Last Modified: Tuesday December 17th 2019, 7:59:01 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Test CanvasColorBackground"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorBackground

# --------------------------------------------------------------------------- #
#                               CanvasColorBackground                        #
# --------------------------------------------------------------------------- #
class CanvasColorBackgroundTests:

    @mark.canvas
    @mark.canvas_color_background
    @mark.canvas_color_background_defaults
    def test_canvas_color_background_defaults(self):
        canvas = CanvasColorBackground()
        assert canvas.paper_bgcolor == '#fff',"canvas.font_family not initialized."
        assert canvas.plot_bgcolor == '#fff',"canvas.font_size not initialized."

    @mark.canvas
    @mark.canvas_color_background
    @mark.canvas_color_background_update
    def test_canvas_color_background_update(self):
        canvas = CanvasColorBackground()
        canvas.paper_bgcolor = 'green'
        canvas.plot_bgcolor = 'green'

        assert canvas.paper_bgcolor == "green", "canvas.paper_bgcolor not updated."
        assert canvas.plot_bgcolor == "green", "canvas.plot_bgcolor not updated."

