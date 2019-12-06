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
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as py
from plotly.subplots import make_subplots
from progress.bar import ChargingBar
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import RidgeRegression
from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.supervised_learning.regression import ElasticNetRegression

from ml_studio.visualate.base import ModelVisualator
from ml_studio.utils.model import get_model_name
from ml_studio.utils.data_manager import sampler      
from ml_studio.utils.misc import proper

# --------------------------------------------------------------------------- #
#                             CV EXPLORER                                     #
# --------------------------------------------------------------------------- #
class CVLinePlot(ModelVisualator):
    """Model selection line plot for GridSearchCV and RandomizedSearchCV.

    Leveraging Scikit-learn's GridSearchCV and RandomSearchCV classes, this
    class provides a line plot showing training scores, test scores and 
    training time for a single continuous parameter and an arbitrary number 
    of categorical parameters. 

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator. 

    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of parameter 
        settings to try as values. 

    grid_search : Bool. Default = True
        If True, GridSearchCV is performed. Otherwise, RandomizedSearchCV
        is performed 

    title : str. Defaults to Model Selection + Estimator Name
        Title for plot. 
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    COLORS = ['#0272a2','#015b82','#014663','#5a852b','#78a24a']

    def __init__(self, model, param_grid, grid_search=True, title=None, 
                 **kwargs):    
        super(CVLinePlot, self).__init__(model, **kwargs)                        
        self.param_grid = param_grid
        self.grid_search = grid_search
        self.params = {}
        self.results = None

    def fit(self, X, y, **kwargs):
        """ Fits the visualator to the data.
        
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Keyword arguments passed to the scikit-learn API. 
            See visualizer specific details for how to use
            the kwargs to modify the visualization or fitting process.    

        Returns
        -------
        self : visualator
        """
        super(CVLinePlot, self).fit(X,y)        
        
        # Format plot title        
        if self.title is None:
            if self.grid_search:
                search = 'GridSearchCV'
            else:
                search = 'RandomizedSearchCV'            
            self.title = search + " : " + self.model.name     

        # Perform search
        if self.grid_search:
            cv = GridSearchCV(self.model, self.param_grid)
        else:
            cv = RandomizedSearchCV(self.model, self.param_grid)
        # cv.fit(X,y)
        # self.results = pd.DataFrame.from_dict(cv.cv_results_)                

    def _display_controls(self, boolean_params, nominal_params):            
        width_bool_switches = '10%'
        width_dropdowns = '25%'            
        divs = []            
        for k, v in nominal_params.items():
            label = proper(k)
            options = []
            for j in v:
                option_label = proper(j)
                d = {'label': option_label, 'value': j}
                options.append(d)
            divs.append(                    
                html.Div(
                    html.Label([label,
                        dcc.Dropdown(
                            options=options,
                            value=v[0]
                        )],
                        id=k,
                        style={'width': width_dropdowns, 'display': 'inline-block'}
                ))
            )
        for i, k in enumerate(boolean_params.keys()):
            label = proper(k)                            
            divs.append(
                html.Div(daq.BooleanSwitch(
                id=k,
                on=False,
                label=label,
                labelPosition="top",
                color=self.COLORS[i]
            ), style={'width': width_bool_switches, 'display': 'inline-block'})                    
            )                
        return divs

    def show(self, path=None, **kwargs):        
        """Renders the visualization.

        Contains the Plotly code that renders the visualization
        in a notebook or in a pop-up GUI. If the  path variable
        is not None, the visualization will be saved to disk.
        Subclasses will override with visualization specific logic.
        
        Parameters
        ----------
        path : str
            The relative directory and file name to which the visualization
            will be saved.

        kwargs : dict
            Various keyword arguments.

        """
        self.path = path

        # Get the categorical and continuous parameters from the parameter grid. 
        self.categorical_params = dict((key, value) for key, value in self.param_grid.items() \
            if isinstance(value[0], (str, bool)))
        self.continuous_params = dict((key, value) for key, value in self.param_grid.items() \
            if key not in self.categorical_params.keys())

        # Create app
        self.app = dash.Dash()
        # Suppress exceptions since we are dynamically adding components
        self.app.config.suppress_callback_exceptions = True 

        # Extract boolean and nominal parameters
        boolean_params = dict((key, value) for key, value in self.param_grid.items() \
            if isinstance(value[0], bool))
        nominal_params = dict((key, value) for key, value in self.param_grid.items() \
            if isinstance(value[0], str))        

        # Dynamically create boolean switch and dropdown controls
        self.app.layout = html.Div([
            html.Div([
                html.H1("Model Selection"),                
            ]),
            html.Div(id='controls', 
            children=self._display_controls(boolean_params, nominal_params),
            style={'width': '50%', 'display': 'inline-block'}                        
            ),
            html.Div(id='selections'),
            html.Div([
                dcc.Graph(id='graph')
            ])            
            ])
        # Create callback that creates the line plot based upon control inputs
                
        @self.app.callback(
            Output('selections', 'children'),
            [Input(k, 'value') for k in self.param_grid.keys()])
        def update_selection(*input_values):                            
            print(input_values)            
            return 'you have selected "{}"'.format(str(v) for v in input_values)

        self.app.run_server(debug=True)
        

    def _plot_1_param(self):

        # Group data by categorical parameters and iteratively build traces        
        groups = self.results.groupby(self.categorical_params) 
        for name, group in groups:
            train_scores = np.array(group['mean_train_scores'])
            train_scores_std = np.array(group['std_train_scores'])
            train_scores_lower_ci = train_scores - 1.96 * (train_scores_std/np.sqrt(self.nobs))
            train_scores_upper_ci = train_scores + 1.96 * (train_scores_std/np.sqrt(self.nobs))        

            test_scores = np.array(group['mean_test_scores'])
            test_scores_std = np.array(group['std_test_scores'])
            test_scores_lower_ci = test_scores - 1.96 * (test_scores_std/np.sqrt(self.nobs))
            test_scores_upper_ci = test_scores + 1.96 * (test_scores_std/np.sqrt(self.nobs))             

            self.fig = make_subplots(specs=[[{"secondary_y": True}]])        

            # Training set train score upper ci
            self.fig.add_trace(go.Scatter(x=group[self.continuous_params],
                        y=train_scores_upper_ci,
                        name="Train Scores Upper Bound",
                        mode="lines",
                        line=dict(width=0.5),
                        marker=dict(color=self.train_color),
                        showlegend=False,
                        opacity=self.train_alpha
                        ),
                        secondary_y=False
            )
            # Training set train score lower ci
            self.fig.add_trace(go.Scatter(x=group[self.continuous_params],
                        y=train_scores_lower_ci,
                        name="Train Score 95% CI",
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0.5),
                        marker=dict(color=self.train_color),
                        showlegend=True,
                        opacity=self.train_alpha
                        ),
                        secondary_y=False
            )
            # Training Set Score Line
            self.fig.add_trace(go.Scatter(x=group[self.continuous_params],
                        y=train_scores,
                        name="Train Scores",
                        mode="lines+markers",
                        marker=dict(color=self.train_color),
                        showlegend=True,
                        opacity=self.train_alpha
                        ),
                        secondary_y=False
            )     

            # Test set test score upper ci
            self.fig.add_trace(go.Scatter(x=group[self.continuous_params],
                        y=test_scores_upper_ci,
                        name="Test Scores Upper Bound",
                        mode="lines",
                        line=dict(width=0.5),
                        marker=dict(color=self.test_color),
                        showlegend=False,
                        opacity=self.test_alpha
                        ),
                        secondary_y=False
            )
            # Test set test score lower ci
            self.fig.add_trace(go.Scatter(x=group[self.continuous_params],
                        y=test_scores_lower_ci,
                        name="Test Score 95% CI",
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0.5),
                        marker=dict(color=self.test_color),
                        showlegend=True,
                        opacity=self.test_alpha
                        ),
                        secondary_y=False
            )
            # Test Set Score Line
            self.fig.add_trace(go.Scatter(x=group[self.continuous_params],
                        y=test_scores,
                        name="Test Scores",
                        mode="lines+markers",
                        marker=dict(color=self.test_color),
                        showlegend=True,
                        opacity=self.test_alpha
                        ),
                        secondary_y=False
            )

            # Fit times
            self.fig.add_trace(go.Scatter(x=group[self.continuous_params],
                        y=group['mean_fit_time'],
                        name="Mean Fit Time",
                        mode="lines+markers",
                        marker=dict(color=self.test_color),
                        showlegend=True,
                        opacity=self.test_alpha
                        ),
                        secondary_y=True
            )            

# %%
