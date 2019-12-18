#!/usr/bin/env python3
# =========================================================================== #
#                              TEST CANVAS LEGEND                             #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_legend.py                                                #
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

"""Test CanvasLegend"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasLegend

# --------------------------------------------------------------------------- #
#                               CanvasLegend                                   #
# --------------------------------------------------------------------------- #
class CanvasLegendTests:

    @mark.canvas
    @mark.canvas_legend
    @mark.canvas_legend_defaults
    def test_canvas_legend_defaults(self):
        canvas = CanvasLegend()
        assert canvas.legend_bgcolor == None , 'canvas.legend_bgcolor not initialized.'
        assert canvas.legend_bordercolor == '#444', 'canvas.legend_bordercolor not initialized.'
        assert canvas.legend_borderwidth == 0, 'canvas.legend_borderwidth not initialized.'
        assert canvas.legend_font_family == None, 'canvas.legend_font_family not initialized.'
        assert canvas.legend_font_size == None, 'canvas.legend_font_size not initialized.'
        assert canvas.legend_font_color == None, 'canvas.legend_font_color not initialized.'
        assert canvas.legend_orientation == 'v', 'canvas.legend_orientation not initialized.'
        assert canvas.legend_itemsizing == 'trace', 'canvas.legend_itemsizing not initialized.'
        assert canvas.legend_itemclick == 'toggle', 'canvas.legend_itemclick not initialized.'
        assert canvas.legend_x == 1.02, 'canvas.legend_x not initialized.'
        assert canvas.legend_y == 1, 'canvas.legend_y not initialized.'
        assert canvas.legend_xanchor == 'left', 'canvas.legend_xanchor not initialized.'
        assert canvas.legend_yanchor == 'auto', 'canvas.legend_yanchor not initialized.'
        assert canvas.legend_valign == 'middle', 'canvas.legend_valign not initialized.'

    @mark.canvas
    @mark.canvas_legend
    @mark.canvas_legend_validation
    def test_canvas_legend_validation(self):
        canvas = CanvasLegend()
        with pytest.raises(TypeError):
            canvas.legend_bgcolor = 99
        with pytest.raises(TypeError):
            canvas.legend_borderwidth = 'x'
        with pytest.raises(ValueError):
            canvas.legend_borderwidth = -2
        with pytest.raises(TypeError):
            canvas.legend_font_size = 'x'
        with pytest.raises(ValueError):
            canvas.legend_font_size = 0   
        with pytest.raises(ValueError):
            canvas.legend_orientation = 0            
        with pytest.raises(ValueError):
            canvas.legend_itemsizing = 0            
        with pytest.raises(ValueError):
            canvas.legend_itemclick = 'x'         
        with pytest.raises(TypeError):
            canvas.legend_x = 'str'                           
        with pytest.raises(ValueError):
            canvas.legend_x = 55
        with pytest.raises(TypeError):
            canvas.legend_y = 'str'                           
        with pytest.raises(ValueError):
            canvas.legend_y = 55
        with pytest.raises(ValueError):
            canvas.legend_xanchor = 'str'                           
        with pytest.raises(ValueError):
            canvas.legend_yanchor = 'str'                           
        with pytest.raises(ValueError):
            canvas.legend_yanchor = 'str'                 

    def test_canvas_legend_update(self):
        canvas = CanvasLegend()
        canvas.legend_bgcolor = 'green'
        canvas.legend_bordercolor = 'green'
        canvas.legend_borderwidth = 2
        canvas.legend_font_family = 'Open Sans'
        canvas.legend_font_size = 4
        canvas.legend_font_color = 'blue'
        canvas.legend_orientation = 'h'
        canvas.legend_itemsizing = 'constant'
        canvas.legend_itemclick = 'toggleothers'
        canvas.legend_x = 1.1
        canvas.legend_y = 0.5
        canvas.legend_xanchor = 'center'
        canvas.legend_yanchor = 'top'
        canvas.legend_valign = 'top'

        assert canvas.legend_bgcolor == 'green' , 'canvas.legend_bgcolor not initialized.'
        assert canvas.legend_bordercolor == 'green', 'canvas.legend_bordercolor not initialized.'
        assert canvas.legend_borderwidth == 2, 'canvas.legend_borderwidth not initialized.'
        assert canvas.legend_font_family == 'Open Sans', 'canvas.legend_font_family not initialized.'
        assert canvas.legend_font_size == 4, 'canvas.legend_font_size not initialized.'
        assert canvas.legend_font_color == 'blue', 'canvas.legend_font_color not initialized.'
        assert canvas.legend_orientation == 'h', 'canvas.legend_orientation not initialized.'
        assert canvas.legend_itemsizing == 'constant', 'canvas.legend_itemsizing not initialized.'
        assert canvas.legend_itemclick == 'toggleothers', 'canvas.legend_itemclick not initialized.'
        assert canvas.legend_x == 1.1, 'canvas.legend_x not initialized.'
        assert canvas.legend_y == 0.5, 'canvas.legend_y not initialized.'
        assert canvas.legend_xanchor == 'center', 'canvas.legend_xanchor not initialized.'
        assert canvas.legend_yanchor == 'top', 'canvas.legend_yanchor not initialized.'
        assert canvas.legend_valign == 'top', 'canvas.legend_valign not initialized.'
        