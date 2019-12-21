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
from ml_studio.services.validation import StringValidator


# --------------------------------------------------------------------------- #
#                               CanvasTitle                                   #
# --------------------------------------------------------------------------- #
class CanvasTitleTests:

    @mark.canvas
    @mark.canvas_title
    @mark.canvas_title_defaults
    def test_canvas_title_defaults(self):
        canvas = CanvasTitle()
        value = np.arange(1,20)
        var_type = "str"
        validator = StringValidator(instance=canvas, category="title", 
                                    attribute="title_text", 
                                    var_type=var_type) 
        validated = validator.validate_coerce(value)
        assert [v for v in validated if isinstance(v, str)], "data converted to strings."
        
