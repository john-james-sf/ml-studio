#!/usr/bin/env python3
# =========================================================================== #
#                              TEST CANVAS TITLE                              #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_title.py                                                 #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 5:58:51 pm                         #
# Last Modified: Tuesday December 17th 2019, 6:57:15 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test CanvasTitle"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasTitle

# --------------------------------------------------------------------------- #
#                               CanvasTitle                                   #
# --------------------------------------------------------------------------- #
class CanvasTitleTests:

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_defaults
    def test_canvas_title_defaults(self):
        canvas = CanvasTitle()
        assert canvas.title_text == "", "title_text default not set "
        assert canvas.title_font_family is None, "title_font_family default not set "
        assert canvas.title_font_size is None, "title_font_size default not set "
        assert canvas.title_font_color is None, "title_font_color default not set "
        assert canvas.title_xref is 'container', "title_font_xref default not set "
        assert canvas.title_yref is 'container', "title_font_yref default not set "
        assert canvas.title_x == 0.5, "title_x default not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_text
    def test_canvas_title_text(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        assert canvas.title_text == "Some title", "title_text not set "
        assert canvas.title_font_family is None, "title_font_family default not set "
        assert canvas.title_font_size is None, "title_font_size default not set "
        assert canvas.title_font_color is None, "title_font_color default not set "
        assert canvas.title_xref is 'container', "title_font_xref default not set "
        assert canvas.title_yref is 'container', "title_font_yref default not set "
        assert canvas.title_x == 0.5, "title_x default not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_font_family
    def test_canvas_title_font_family(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        assert canvas.title_text == "Some title", "title_text not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family not set "
        assert canvas.title_font_size is None, "title_font_size default not set "
        assert canvas.title_font_color is None, "title_font_color default not set "
        assert canvas.title_xref is 'container', "title_font_xref default not set "
        assert canvas.title_yref is 'container', "title_font_yref default not set "
        assert canvas.title_x == 0.5, "title_x default not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_font_size
    def test_canvas_title_font_size(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        with pytest.raises(ValueError):
            canvas.title_font_size = -2
        canvas.title_font_size = 12
        assert canvas.title_text == "Some title", "title_text not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family not set "
        assert canvas.title_font_size == 12, "title_font_size not set "
        assert canvas.title_font_color is None, "title_font_color default not set "
        assert canvas.title_xref is 'container', "title_font_xref default not set "
        assert canvas.title_yref is 'container', "title_font_yref default not set "
        assert canvas.title_x == 0.5, "title_x default not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "            
      

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_font_color
    def test_canvas_title_font_color(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'container', "title_font_xref default not set "
        assert canvas.title_yref is 'container', "title_font_yref default not set "
        assert canvas.title_x == 0.5, "title_x default not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "            

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_xref
    def test_canvas_title_xref(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        with pytest.raises(ValueError):
            canvas.title_xref = "hat"
        canvas.title_xref = "paper"
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'paper', "title_font_xref not set "
        assert canvas.title_yref is 'container', "title_font_yref default not set "
        assert canvas.title_x == 0.5, "title_x default not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "  

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_yref
    def test_canvas_title_yref(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        with pytest.raises(ValueError):
            canvas.title_yref = "hat"
        canvas.title_xref = "paper" 
        canvas.title_yref = "paper"
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'paper', "title_font_xref not set "
        assert canvas.title_yref is 'paper', "title_font_yref not set "
        assert canvas.title_x == 0.5, "title_x default not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "          

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_x
    def test_canvas_title_x(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        with pytest.raises(ValueError):
            canvas.title_yref = "hat"
        canvas.title_xref = "paper" 
        canvas.title_yref = "paper"
        with pytest.raises(ValueError):
            canvas.title_x = 5
        canvas.title_x = 0.4
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'paper', "title_font_xref not set "
        assert canvas.title_yref is 'paper', "title_font_yref not set "
        assert canvas.title_x == 0.4, "title_x not set "
        assert canvas.title_y == 'auto', "title_y default not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "                  

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_y
    def test_canvas_title_y(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        with pytest.raises(ValueError):
            canvas.title_yref = "hat"
        canvas.title_xref = "paper" 
        canvas.title_yref = "paper"
        canvas.title_x = 0.4        
        with pytest.raises(ValueError):
            canvas.title_y = 5
        with pytest.raises(ValueError):
            canvas.title_y = 'dig'            
        canvas.title_y = 0.4
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'paper', "title_font_xref not set "
        assert canvas.title_yref is 'paper', "title_font_yref not set "
        assert canvas.title_x == 0.4, "title_x not set "
        assert canvas.title_y == 0.4, "title_y not set "
        assert canvas.title_xanchor == 'auto', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "         

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_xanchor
    def test_canvas_title_xanchor(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        canvas.title_xref = "paper" 
        canvas.title_yref = "paper"
        canvas.title_x = 0.4                 
        canvas.title_y = 0.4
        with pytest.raises(ValueError):
            canvas.title_xanchor = 'best'
        canvas.title_xanchor = 'left'
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'paper', "title_font_xref not set "
        assert canvas.title_yref is 'paper', "title_font_yref not set "
        assert canvas.title_x == 0.4, "title_x not set "
        assert canvas.title_y == 0.4, "title_y not set "
        assert canvas.title_xanchor == 'left', "title_xanchor default not set "
        assert canvas.title_yanchor == 'auto', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "          

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_yanchor
    def test_canvas_title_yanchor(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        canvas.title_xref = "paper" 
        canvas.title_yref = "paper"
        canvas.title_x = 0.4                 
        canvas.title_y = 0.4
        canvas.title_xanchor = 'left'
        with pytest.raises(ValueError):
            canvas.title_yanchor = 'best'
        canvas.title_yanchor = 'top'
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'paper', "title_font_xref not set "
        assert canvas.title_yref is 'paper', "title_font_yref not set "
        assert canvas.title_x == 0.4, "title_x not set "
        assert canvas.title_y == 0.4, "title_y not set "
        assert canvas.title_xanchor == 'left', "title_xanchor default not set "
        assert canvas.title_yanchor == 'top', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 0, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 0, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 0, "title_pad_l default not set "           

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_pad
    def test_canvas_title_pad(self):
        canvas = CanvasTitle()
        canvas.title_text = "Some title"
        canvas.title_font_family = 'Open Sans'
        canvas.title_font_size = 12
        canvas.title_font_color = 'blue'
        canvas.title_xref = "paper" 
        canvas.title_yref = "paper"
        canvas.title_x = 0.4                 
        canvas.title_y = 0.4
        canvas.title_xanchor = 'left'
        canvas.title_yanchor = 'top' 
        with pytest.raises(TypeError):
            canvas.title_pad = 'best'
        with pytest.raises(KeyError):                    
            canvas.title_pad = {'x': 5}
        with pytest.raises(TypeError):                    
            canvas.title_pad = {'b': 'stop'}            
        canvas.title_pad = {'t': 5,'b':5, 'l':5}
        assert canvas.title_text == "Some title", "title_text default not set "
        assert canvas.title_font_family == 'Open Sans', "title_font_family default not set "
        assert canvas.title_font_size == 12 , "title_font_size default not set "
        assert canvas.title_font_color == 'blue', "title_font_color not set "
        assert canvas.title_xref is 'paper', "title_font_xref not set "
        assert canvas.title_yref is 'paper', "title_font_yref not set "
        assert canvas.title_x == 0.4, "title_x not set "
        assert canvas.title_y == 0.4, "title_y not set "
        assert canvas.title_xanchor == 'left', "title_xanchor default not set "
        assert canvas.title_yanchor == 'top', "title_yanchor default not set "
        assert canvas.title_pad['t'] == 5, "title_pad_t default not set "
        assert canvas.title_pad['b'] == 5, "title_pad_b default not set "
        assert canvas.title_pad['l'] == 5, "title_pad_l default not set "                