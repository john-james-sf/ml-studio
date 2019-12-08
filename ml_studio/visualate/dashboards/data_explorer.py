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
import sys
sys.path.append('ml_studio')
sys.path.append('ml_studio/utils/visual')
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
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

from ml_studio.visualate.classification.figures import serve_prediction_plot, serve_roc_curve, \
    serve_pie_confusion_matrix

import ml_studio
from ml_studio.utils.model import get_model_name
from ml_studio.utils.data_manager import sampler, data_split, StandardScaler      
from ml_studio.utils.misc import proper
import ml_studio.utils.visual as drc
# --------------------------------------------------------------------------- #
external_scripts = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

app = dash.Dash(__name__, 
        external_scripts=external_scripts)
app.scripts.config.serve_locally = False
server = app.server
# --------------------------------------------------------------------------- #
#                            Generate Data                                    #
# --------------------------------------------------------------------------- #
def generate_data(dataset, n_samples=None, n_features=None, noise=100, 
                  seed=None):

    if dataset == 'california':
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

    elif dataset == 'binary':
        X, y = make_classification(
            n_samples=100,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1
        )

        linearly_separable = (X, y)

        return linearly_separable        
    else:
        raise ValueError(
            'Data type incorrectly specified. Please choose an existing '
            'dataset.')

# --------------------------------------------------------------------------- #
#                            Define Tabs                                      #
# --------------------------------------------------------------------------- #            
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'border': '1px solid #282b38',    
    'borderBottom': '1px solid #282b38',    
    'backgroundColor': '#282b38',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'border': '1px solid #282b38',
    'borderBottom': '1px solid #31459E',
    'backgroundColor': '#282b38',
    'color': 'white',
    'padding': '6px'
}
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
                        id="Analysis-tab",
                        label="Data Analysis",
                        value="tab3",
                        style=tab_style,
                        selected_style=tab_selected_style,                        
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Cleaning-tab",
                        label="Data Cleaning",
                        value="tab4",
                        style=tab_style,
                        selected_style=tab_selected_style,                        
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Feature-selection-tab",
                        label="Feature Selection",
                        value="tab5",
                        style=tab_style,
                        selected_style=tab_selected_style,                        
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Features-engineering-tab",
                        label="Feature Engineering",
                        value="tab6",
                        style=tab_style,
                        selected_style=tab_selected_style,                        
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Dimension-reduction-tab",
                        label="Dimension Reduction",
                        value="tab7",
                        style=tab_style,
                        selected_style=tab_selected_style,                        
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),                                                                                
                ],
            )
        ],
    )

def build_analysis_tab():
    pass

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
        html.Div(
            id="app-container",
            children=[
                build_tabs()
            ],
        ),        
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
                            name='Select Data Type',
                            id='dropdown-select-datatype',
                            options=[
                                {'label': 'Regression', 'value': 'regression'},
                                {'label': 'Binary Classification','value': 'binary'},
                                {'label': 'Multiclass Classification','value': 'multiclass'}
                            ],
                            clearable=False,
                            searchable=False,
                            value='regression'
                        ),                        
                        drc.NamedDropdown(
                            name='Select Dataset',
                            id='dropdown-select-dataset',
                            options=[
                                {'label': 'California Housing', 'value': 'california'},
                                {'label': 'Million Song Dataset','value': 'msd'},
                                {'label': 'Online News Popularity','value': 'online_news'},
                                {'label': 'Speed Dating', 'value': 'speed_dating'},
                                {'label': 'Regression', 'value': 'regression'},
                                {'label': 'Binary', 'value': 'binary'}
                            ],
                            clearable=False,
                            searchable=False,
                            value='california'
                        ),
                    ]),              
              
                    html.Div(
                        dcc.Markdown(dedent("""
                        [Click here](https://github.com/decisionscients/ml-studio) to visit the project repo, and learn about how to use the app.
                        """)),
                        style={'margin': '20px 0px', 'text-align': 'center'}
                    ),
                ]
            ),
        ]),
    ])
])




# @app.callback(Output('div-graphs', 'children'),
#                Input('dropdown-select-dataset', 'value'),
#                Input('slider-threshold', 'value')
# def update_svm_graph(kernel,
#                      degree,
#                      C_coef,
#                      C_power,
#                      gamma_coef,
#                      gamma_power,
#                      dataset,
#                      noise,
#                      shrinking,
#                      threshold,
#                      sample_size):
#     t_start = time.time()
#     h = .3  # step size in the mesh

#     # Data Pre-processing
#     X, y = generate_data(dataset=dataset)
#     StandardScaler().fit(X)
#     X = StandardScaler().transform(X)
#     X_train, X_test, y_train, y_test = \
#         data_split(X, y, test_size=.4, seed=42)

#     x_min = X[:, 0].min() - .5
#     x_max = X[:, 0].max() + .5
#     y_min = X[:, 1].min() - .5
#     y_max = X[:, 1].max() + .5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))

#     C = C_coef * 10 ** C_power
#     gamma = gamma_coef * 10 ** gamma_power

#     # Train SVM
#     clf = SVC(
#         C=C,
#         kernel=kernel,
#         degree=degree,
#         gamma=gamma,
#         shrinking=shrinking
#     )
#     clf.fit(X_train, y_train)

#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     if hasattr(clf, "decision_function"):
#         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     else:
#         Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

#     prediction_figure = serve_prediction_plot(
#         model=clf,
#         X_train=X_train,
#         X_test=X_test,
#         y_train=y_train,
#         y_test=y_test,
#         Z=Z,
#         xx=xx,
#         yy=yy,
#         mesh_step=h,
#         threshold=threshold
#     )

#     roc_figure = serve_roc_curve(
#         model=clf,
#         X_test=X_test,
#         y_test=y_test
#     )

#     confusion_figure = serve_pie_confusion_matrix(
#         model=clf,
#         X_test=X_test,
#         y_test=y_test,
#         Z=Z,
#         threshold=threshold
#     )

#     print(
#         f"Total Time Taken: {time.time() - t_start:.3f} sec")

    # return [
    #     html.Div(
    #         className='three columns',
    #         style={
    #             'min-width': '24.5%',
    #             'height': 'calc(100vh - 90px)',
    #             'margin-top': '5px',

    #             # Remove possibility to select the text for better UX
    #             'user-select': 'none',
    #             '-moz-user-select': 'none',
    #             '-webkit-user-select': 'none',
    #             '-ms-user-select': 'none'
    #         },
    #         children=[
    #             dcc.Graph(
    #                 id='graph-line-roc-curve',
    #                 style={'height': '40%'},
    #                 figure=roc_figure
    #             ),

    #             dcc.Graph(
    #                 id='graph-pie-confusion-matrix',
    #                 figure=confusion_figure,
    #                 style={'height': '60%'}
    #             )
    #         ]),

    #     html.Div(
    #         className='six columns',
    #         style={'margin-top': '5px'},
    #         children=[
    #             dcc.Graph(
    #                 id='graph-sklearn-svm',
    #                 figure=prediction_figure,
    #                 style={'height': 'calc(100vh - 90px)'}
    #             )
    #         ])
    # ]


# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)

# %%
