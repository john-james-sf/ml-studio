#!/usr/bin/env python3
# =========================================================================== #
#                             TEST LAYOUT                                     #  
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_layout.py                                                       #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Wednesday December 18th 2019, 12:02:54 am                      #
# Last Modified: Wednesday December 18th 2019, 12:03:17 am                    #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test LayoutColorAxisBarBoundary"""
import numpy as np
import plotly.graph_objs as go
import plotly.offline as po
import pytest
from pytest import mark

# --------------------------------------------------------------------------- #
#                               LayoutTests                                   #
# --------------------------------------------------------------------------- #
class LayoutTests:

    @mark.layout
    @mark.layout_title
    def test_layout_title(self, canvas_layouts):
        canvas, layout = canvas_layouts
        fig = go.Figure()
        updated_fig = layout.update_layout(canvas, fig)
        assert updated_fig, "Figure layout not updated"
