# =========================================================================== #
#                             DATA EXPLORER                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \data_explorer.py                                                     #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 6th 2019, 9:12:28 pm                           #
# Last Modified: Friday December 6th 2019, 9:12:35 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Data Explorer - A dash powered web app for analyzing and preparing data.

    This module provides a dashboard application that supports:
        - Data Audit : Missing values and outliers
        - Data Analysis : Exploration of data vis-a-vis statistical 
            assumptions of independence, linearity, normality, 
            and homoscedasticity
        - Data Preparation : Missing values, and outliers
        - Feature Selection : Identifying the features that most 
            influence the dependent variable
        - Features Engineering : Feature transformation, Binning
            One-Hot Encoding, Features Split and Scaling
        - Dimensionality Reduction : PCA,  
            t-Distributed Stochastic Neighbor Embedding (t-SNE)
            see https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/

Note: This module was highly inspired by the plotly dash-svm 
        at https://github.com/plotly/dash-svm.
""" 
#%%
import os
import time
from textwrap import dedent
import warnings

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ml_studio.utils.model import get_model_name
from ml_studio.utils.data_manager import sampler, data_split      
from ml_studio.utils.misc import proper
import ml_studio.utils.visual as drc
# --------------------------------------------------------------------------- #
app = dash.Dash(__name__)
server = app.server

def generate_data(dataset, n_samples=None, n_features=None, seed=None):

    if dataset == 'california_housing':
        return(fetch_california_housing(return_X_y=True))
    elif dataset == 'msd':
        data = pd.read_csv("ml_studio/data_gathering/msd/year_prediction.csv")
        y = data[['label']]
        X = data.drop(columns=['label'], inplace=False)
        msd = (X, y)
        return msd
    elif dataset == 'online_news':        
        data = pd.read_csv("ml_studio/data_gathering/online_news_popularity/OnlineNewsPopularity.csv")        
        data.columns = data.columns.str.replace(r'\s+', '')
        y = data[['shares']]
        X = data.drop(columns=['shares'], inplace=False)
        online_news = (X, y)
        return online_news
    elif dataset == 'speed_dating':
        data = pd.read_csv("ml_studio/data_gathering/speed_dating/Speed Dating Data.csv",
                            encoding = 'unicode_escape')
        y = data[['match']]
        X = data.drop(columns=['match'], inplace=False)
        speed_dating = (X, y)
        return speed_dating               
    elif dataset == 'regression':
        if n_samples is None:
            warnings.warn("n_samples is None, defaulting to 10,000")
            n_samples = 10000
        if n_features is None:
            warnings.warn("n_features is None, defaulting to 100")
            n_features = 100

        X, y = make_regression(n_samples, n_features,
                               n_informative=100,
                               bias=400,
                               effective_rank=50,
                               noise=100,
                               random_state=seed)

        regression = (X, y)
        return regression
    else:
        raise ValueError(
            'Data type incorrectly specified. Please choose an existing '
            'dataset.')


app = dash.Dash(__name__)
server = app.server

def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab1",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Source-data-tab",
                        label="Source Data",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Analysis-tab",
                        label="Data Analysis",
                        value="tab3",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Cleaning-tab",
                        label="Data Cleaning",
                        value="tab4",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Feature-selection-tab",
                        label="Feature Selection",
                        value="tab5",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Features-engineering-tab",
                        label="Feature Engineering",
                        value="tab6",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Dimension-reduction-tab",
                        label="Dimension Reduction",
                        value="tab7",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),                                                                                
                ],
            )
        ],
    )

app.layout = html.Div(children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", children=[
        # Change App Name here
        html.Div(className='container scalable', children=[
            # Change App Name here
            html.H2(html.A(
                'ML Studio Data Explorer',
                href='https://github.com/decisionscients/ml-studio',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit'
                }
            )),

            html.A(
                # TODO: Create logo
                html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
                href='https://plot.ly/products/dash/'
            )
        ]),
    ]),

    html.Div(id='body', className='container scalable', children=[
        html.Div(className='row', children=[
            html.Div(
                id='div-graphs',
                children=dcc.Graph(
                    id='graph-sklearn-svm',
                    style={'display': 'none'}
                )
            ),

            html.Div(
                className='three columns',
                style={
                    'min-width': '24.5%',
                    'max-height': 'calc(100vh - 85px)',
                    'overflow-y': 'auto',
                    'overflow-x': 'hidden',
                },
                children=[
                    drc.Card([
                        drc.NamedDropdown(
                            name='Select Dataset',
                            id='dropdown-select-dataset',
                            options=[
                                {'label': 'Moons', 'value': 'moons'},
                                {'label': 'Linearly Separable',
                                 'value': 'linear'},
                                {'label': 'Circles', 'value': 'circles'}
                            ],
                            clearable=False,
                            searchable=False,
                            value='moons'
                        ),

                        drc.NamedSlider(
                            name='Sample Size',
                            id='slider-dataset-sample-size',
                            min=100,
                            max=500,
                            step=100,
                            marks={i: i for i in [100, 200, 300, 400, 500]},
                            value=300
                        ),

                        drc.NamedSlider(
                            name='Noise Level',
                            id='slider-dataset-noise-level',
                            min=0,
                            max=1,
                            marks={i / 10: str(i / 10) for i in
                                   range(0, 11, 2)},
                            step=0.1,
                            value=0.2,
                        ),
                    ]),

                    drc.Card([
                        drc.NamedSlider(
                            name='Threshold',
                            id='slider-threshold',
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01
                        ),

                        html.Button(
                            'Reset Threshold',
                            id='button-zero-threshold'
                        ),
                    ]),

                    drc.Card([
                        drc.NamedDropdown(
                            name='Kernel',
                            id='dropdown-svm-parameter-kernel',
                            options=[
                                {'label': 'Radial basis function (RBF)',
                                 'value': 'rbf'},
                                {'label': 'Linear', 'value': 'linear'},
                                {'label': 'Polynomial', 'value': 'poly'},
                                {'label': 'Sigmoid', 'value': 'sigmoid'}
                            ],
                            value='rbf',
                            clearable=False,
                            searchable=False
                        ),

                        drc.NamedSlider(
                            name='Cost (C)',
                            id='slider-svm-parameter-C-power',
                            min=-2,
                            max=4,
                            value=0,
                            marks={i: '{}'.format(10 ** i) for i in
                                   range(-2, 5)}
                        ),

                        drc.FormattedSlider(
                            style={'padding': '5px 10px 25px'},
                            id='slider-svm-parameter-C-coef',
                            min=1,
                            max=9,
                            value=1
                        ),

                        drc.NamedSlider(
                            name='Degree',
                            id='slider-svm-parameter-degree',
                            min=2,
                            max=10,
                            value=3,
                            step=1,
                            marks={i: i for i in range(2, 11, 2)},
                        ),

                        drc.NamedSlider(
                            name='Gamma',
                            id='slider-svm-parameter-gamma-power',
                            min=-5,
                            max=0,
                            value=-1,
                            marks={i: '{}'.format(10 ** i) for i in
                                   range(-5, 1)}
                        ),

                        drc.FormattedSlider(
                            style={'padding': '5px 10px 25px'},
                            id='slider-svm-parameter-gamma-coef',
                            min=1,
                            max=9,
                            value=5
                        ),

                        drc.NamedRadioItems(
                            name='Shrinking',
                            id='radio-svm-parameter-shrinking',
                            labelStyle={
                                'margin-right': '7px',
                                'display': 'inline-block'
                            },
                            options=[
                                {'label': ' Enabled', 'value': True},
                                {'label': ' Disabled', 'value': False},
                            ],
                            value=True,
                        ),
                    ]),

                    html.Div(
                        dcc.Markdown(dedent("""
                        [Click here](https://github.com/plotly/dash-svm) to visit the project repo, and learn about how to use the app.
                        """)),
                        style={'margin': '20px 0px', 'text-align': 'center'}
                    ),
                ]
            ),
        ]),
    ])
])





external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)