#!/usr/bin/env python3
# =========================================================================== #
#                              TEST CANVAS MARGINS                            #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_margins.py                                                #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 7:02:08 pm                         #
# Last Modified: Tuesday December 17th 2019, 7:02:16 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Test CanvasMargins"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasMargins

# --------------------------------------------------------------------------- #
#                               CanvasMargins                                   #
# --------------------------------------------------------------------------- #
class CanvasMarginsTests:

    @mark.canvas
    @mark.canvas_margins
    @mark.canvas_margins_defaults
    def test_canvas_margins_defaults(self):
        canvas = CanvasMargins()
        assert canvas.margins_left == 80,"canvas.margins_left not initialized."
        assert canvas.margins_top == 100,"canvas.margins_top not initialized."
        assert canvas.margins_bottom == 80,"canvas.margins_bottom not initialized."
        assert canvas.margins_pad == 0,"canvas.margins_pad not initialized."


    @mark.canvas
    @mark.canvas_margins
    @mark.canvas_margins_validation
    def test_canvas_margins_validation(self):
        canvas = CanvasMargins()
        with pytest.raises(ValueError):
            canvas.margins_left = 'x'
        with pytest.raises(ValueError):
            canvas.margins_top = 'x'            
        with pytest.raises(ValueError):
            canvas.margins_bottom = 'x'
            

    def test_canvas_margins_update(self):
        canvas = CanvasMargins()
        canvas.margins_bgcolor = 'green'
        canvas.margins_bordercolor = 'green'
        canvas.margins_borderwidth = 2
        canvas.margins_font_family = 'Open Sans'
        canvas.margins_font_size = 4
        canvas.margins_font_color = 'blue'
        canvas.margins_orientation = 'h'
        canvas.margins_itemsizing = 'constant'
        canvas.margins_itemclick = 'toggleothers'
        canvas.margins_x = 1.1
        canvas.margins_y = 0.5
        canvas.margins_xanchor = 'center'
        canvas.margins_yanchor = 'top'
        canvas.margins_valign = 'top'

        assert canvas.margins_bgcolor == 'green' , 'canvas.margins_bgcolor not initialized.'
        assert canvas.margins_bordercolor == 'green', 'canvas.margins_bordercolor not initialized.'
        assert canvas.margins_borderwidth == 2, 'canvas.margins_borderwidth not initialized.'
        assert canvas.margins_font_family == 'Open Sans', 'canvas.margins_font_family not initialized.'
        assert canvas.margins_font_size == 4, 'canvas.margins_font_size not initialized.'
        assert canvas.margins_font_color == 'blue', 'canvas.margins_font_color not initialized.'
        assert canvas.margins_orientation == 'h', 'canvas.margins_orientation not initialized.'
        assert canvas.margins_itemsizing == 'constant', 'canvas.margins_itemsizing not initialized.'
        assert canvas.margins_itemclick == 'toggleothers', 'canvas.margins_itemclick not initialized.'
        assert canvas.margins_x == 1.1, 'canvas.margins_x not initialized.'
        assert canvas.margins_y == 0.5, 'canvas.margins_y not initialized.'
        assert canvas.margins_xanchor == 'center', 'canvas.margins_xanchor not initialized.'
        assert canvas.margins_yanchor == 'top', 'canvas.margins_yanchor not initialized.'
        assert canvas.margins_valign == 'top', 'canvas.margins_valign not initialized.'
        