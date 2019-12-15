#!/usr/bin/env python3
# =========================================================================== #
#                                 NORMALITY                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \normality.py                                                         #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 10th 2019, 12:33:53 am                        #
# Last Modified: Tuesday December 10th 2019, 12:34:21 am                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module for statistical calculations pertaining to the normal distribution."""
#%%
import os

import math
import numpy as np
# --------------------------------------------------------------------------- #
#                   Normal Probability Order Statistics                       #
# --------------------------------------------------------------------------- #
def uniform_order_stat(x):
    """Estimates uniform order statistics medians for the normal distribution."""
    positions = np.arange(1, len(x)+1)
    n = len(positions)
    u_i = (positions-0.375)/(n+0.25)
    return u_i
# --------------------------------------------------------------------------- #
#                                  Z-Score                                    #
# --------------------------------------------------------------------------- #
def z_score(x):
    """Computes z-scores for a series of values."""
    mu = np.mean(x)
    std = np.std(x)
    z = (x-mu)/std
    return z

# --------------------------------------------------------------------------- #
#                             Theoretical Quantiles                           #
# --------------------------------------------------------------------------- #    
def theoretical_quantiles(x):
    """Computes the theoretical quantiles for a vector x."""
    u_i =  uniform_order_stat(x)
    q = z_score(u_i)
    return q

def sample_quantiles(x):
    """Computes the sample quantiles for a vector x."""
    x_sorted = np.sort(x)
    q = z_score(x_sorted)
    return q


# %%
