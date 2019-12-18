#!/usr/bin/env python3
# =========================================================================== #
#                         TEST CANVAS COLOR DOMAIN                            # 
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_canvas_color_domain.py                                          #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 8:39:24 pm                         #
# Last Modified: Tuesday December 17th 2019, 8:39:37 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Test CanvasColorAxisDomain"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.visualate.canvas import CanvasColorAxisDomain

# --------------------------------------------------------------------------- #
#                           CanvasColorAxisDomain                             #
# --------------------------------------------------------------------------- #
class CanvasColorAxisDomainTests:

    @mark.canvas
    @mark.canvas_color_domain
    @mark.canvas_color_domain_defaults
    def test_canvas_color_domain_defaults(self):
        canvas = CanvasColorAxisDomain()
        assert canvas.coloraxis_cauto == True, "canvas.coloraxis_cauto not initialized"
        assert canvas.coloraxis_cmin == None, "canvas.coloraxis_cmin not initialized"
        assert canvas.coloraxis_cmax == None, "canvas.coloraxis_cmax not initialized"
        assert canvas.coloraxis_cmid == None, "canvas.coloraxis_cmid not initialized"

    @mark.canvas
    @mark.canvas_color_domain
    @mark.canvas_color_domain_validation
    def test_canvas_color_domain_validation(self):
        canvas = CanvasColorAxisDomain()
        with pytest.raises(TypeError):
            canvas.coloraxis_cauto = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_cmin = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_cmax = 'x'
        with pytest.raises(TypeError):
            canvas.coloraxis_cmid = 'x'

    @mark.canvas
    @mark.canvas_color_domain
    @mark.canvas_color_domain_update
    def test_canvas_color_domain_update(self):
        canvas = CanvasColorAxisDomain()
        canvas.coloraxis_cauto = False
        canvas.coloraxis_cmin = 2
        canvas.coloraxis_cmax = 2
        canvas.coloraxis_cmid = 2

        assert canvas.coloraxis_cauto == False, "canvas.coloraxis_cauto not updated"
        assert canvas.coloraxis_cmin == 2, "canvas.coloraxis_cmin not updated"
        assert canvas.coloraxis_cmax == 2, "canvas.coloraxis_cmax not updated"
        assert canvas.coloraxis_cmid == 2, "canvas.coloraxis_cmid not updated"
