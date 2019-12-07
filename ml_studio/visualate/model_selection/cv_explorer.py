# =========================================================================== #
#                              OPTIMIZATION                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \cv_explorer.py                                                       #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Thursday December 5th 2019, 3:09:25 pm                         #
# Last Modified: Friday December 6th 2019, 5:09:03 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #


"""Model selection plots.""" 
#%%
import os
import time
from textwrap import dedent

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston, make_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import RidgeRegression
from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.supervised_learning.regression import ElasticNetRegression

from ml_studio.visualate.base import ModelVisualator
from ml_studio.utils.model import get_model_name
from ml_studio.utils.data_manager import sampler, data_split, StandardScaler      
from ml_studio.utils.misc import proper

# --------------------------------------------------------------------------- #
#                             CV EXPLORER                                     #
# --------------------------------------------------------------------------- #
app = dash.Dash(__name__)
server = app.server