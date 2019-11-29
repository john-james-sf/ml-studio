# =========================================================================== #
#                                VALIDATION                                   #
# =========================================================================== #
# =========================================================================== #
# Project: Visualate                                                          #
# Version: 0.1.0                                                              #
# File: \validity.py                                                          #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Wednesday November 27th 2019, 10:07:13 am                      #
# Last Modified: Thursday November 28th 2019, 2:19:31 am                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Visualators used to analyze outliers and observations with influence.""" 
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.model_evaluation.model_validation import standardized_residuals
from ml_studio.model_evaluation.model_validation import studentized_residuals
from ml_studio.model_diagnostics.influence import leverage, cooks_distance

from ..base import ModelVisualator
from ...utils.model import get_model_name      

# --------------------------------------------------------------------------- #
#                        RESIDUALS VS LEVERAGE                                #
# --------------------------------------------------------------------------- #
class ResidualsLeverage(ModelVisualator):
    """Residuals vs Leverage Plot.

    The Residuals vs Leverage plot helps to illuminate data points that may
    be influential. Observations that lie in the upper and lower right 
    corners of the plot may be outside of the Cook's distance line, and 
    would require further investigation.

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator.

    hist : bool, 'density', 'frequency', default: True
        Draw a histogram showing the distribution of the residuals on the 
        right side of the figure. If set to 'density', the probability
        density will be plotted. If set to 'frequency', the frequency will
        be plotted.

    train_color : color, default: 'darkblue'
        Residuals from the training set are plotted with this color. Can
        be any matplotlib color.

    test_color : color, default: 'green'
        Residuals from the test set are plotted with this color. Can
        be any matplotlib color.

    line_color : color, default: dark grey
        Defines the color of the zero error line, can be any matplotlib color.
    
    train_alpha : float, default: 0.75
        Specify a transparency for training data, where 1 is completely opaque
        and 0 is completely transparent. This property makes densely clustered
        points more visible.

    test_alpha : float, default: 0.75
        Specify a transparency for test data, where 1 is completely opaque
        and 0 is completely transparent. This property makes densely clustered
        points more visible.    
    
    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

        ===========  ==========================================================
        Property     Description
        -----------  ----------------------------------------------------------
        height       specify the height in pixels for the figure
        width        specify the width in pixels of the figure
        template     specify the theme template 
        title        specify the title for the visualization
        ===========  ==========================================================

    """

    def __init__(self, model, hist=False, train_color='#0272a2', 
                 test_color='#9fc377', line_color='darkgray', train_alpha=0.75, 
                 test_alpha=0.75, **kwargs):    
        super(ResidualsLeverage, self).__init__(model, **kwargs)                
        self.hist = hist
        self.train_color = train_color
        self.test_color = test_color
        self.line_color = line_color
        self.train_alpha = train_alpha
        self.test_alpha = test_alpha    

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
        super(ResidualsLeverage, self).fit(X,y)        
        # Format plot title
        if self.title is None:
            self.title = "Residual vs. Leverage Plot : " + self.model.name        
        
        # Compute predictions and standardized residuals        
        self.train_residuals, self.y_train_pred = standardized_residuals(self.model, X, y)
        
        # Compute leverage and cooks distances
        self.leverage = leverage(X)
        self.cooks_d = cooks_distance(self.model, X, y)

        return self           

    def _cooks_contour(self, distance=0.5, sign=1):
        """Computes the leverage v standardized residuals, given Cooks Distance."""        
        # Compute F-distribution degrees of freedom
        n = self.X_train.shape[0]
        p = self.X_train.shape[1]
        df = (1/p) * (n/(n-p))
        # Designate the leverage range being plotted
        min_leverage = np.min(self.leverage)
        max_leverage = np.max(self.leverage)
        leverage = np.linspace(start=min_leverage, stop=max_leverage)
        # Compute standard residuals
        std_resid_squared = distance/df * ((1-leverage)/leverage)
        std_resid = sign * np.sqrt(std_resid_squared)
        return (leverage, std_resid)
        

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
            Various keyword arguments

        """
        self.path = path
        # Create lowess smoothing line
        z1 = lowess(self.train_residuals, self.leverage, frac=1./3, it=0, 
                    is_sorted=False, return_sorted=True)    

        # Grab data for Cooks Distance lines
        cooks_line_1_upper = self._cooks_contour(distance=0.05)
        cooks_line_1_lower = self._cooks_contour(distance=0.05, sign=-1)
        cooks_line_2_upper = self._cooks_contour(distance=1)
        cooks_line_2_lower = self._cooks_contour(distance=1, sign=-1)
        
        # Create scatterplot traces
        data = [
            go.Scattergl(x=self.leverage, y=self.train_residuals,
                        mode='markers',
                        marker=dict(color=self.train_color),
                        name="Residual vs Leverage",
                        showlegend=True,
                        opacity=self.train_alpha),
            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        opacity=self.train_alpha,
                        showlegend=False),
            go.Scatter(x=cooks_line_1_upper[0],
                       y=cooks_line_1_upper[1],
                       mode='lines',
                       line=dict(dash='dash'),
                       marker=dict(color='red'),
                       name="Cooks Distance = 0.5",
                       opacity=self.train_alpha,
                       showlegend=True),
            go.Scatter(x=cooks_line_1_lower[0],
                       y=cooks_line_1_lower[1],
                       mode='lines',
                       line=dict(dash='dash'),
                       marker=dict(color='red'),
                       name="Cooks Distance = 0.5 (Lower)",
                       opacity=self.train_alpha,
                       showlegend=False),
            go.Scatter(x=cooks_line_2_upper[0],
                       y=cooks_line_2_upper[1],
                       mode='lines',
                       line=dict(dash='dot'),
                       marker=dict(color='red'),
                       name="Cooks Distance = 1",
                       opacity=self.train_alpha,
                       showlegend=True),
            go.Scatter(x=cooks_line_2_lower[0],
                       y=cooks_line_2_lower[1],
                       mode='lines',
                       line=dict(dash='dot'),
                       marker=dict(color='red'),
                       name="Cooks Distance = 0.5 (Lower)",
                       opacity=self.train_alpha,
                       showlegend=False)                                                                                                                                   
       ]

        # Compute x and y axis limits based 110% of the data range
        xmin = np.min(self.leverage) + (0.1 * np.min(self.leverage))
        xmax = np.max(self.leverage) + (0.1 * np.max(self.leverage)) 
        ymin = np.min(self.train_residuals) + (0.1 * np.min(self.train_residuals))
        ymax = np.max(self.train_residuals) + (0.1 * np.max(self.train_residuals))

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Leverage",
                        yaxis_title="Standardized Residuals",
                        xaxis=dict(domain=[0,0.85],  zeroline=False, 
                                    range=[xmin, xmax]),
                        yaxis=dict(domain=[0,0.85],  zeroline=False, 
                                    range=[ymin, ymax]),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=True,
                        legend=dict(x=0, bgcolor='white'),
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)  

        # Specify existence and type of histogram 
        if self.hist is True:
            self.hist = ""
        if self.hist in ["density", ""]:
            self.fig.add_trace(go.Histogram(y=self.train_residuals,
                                name="y density train",
                                showlegend=False,
                                xaxis="x2",
                                orientation="h",
                                opacity=self.train_alpha,
                                marker_color=self.train_color,
                                histnorm=self.hist))
                                
        # Render plot and save if path is provided
        if self.path:
            py.plot(self.fig, filename=self.path, auto_open=True, include_mathjax='cdn')
        else:
            py.plot(self.fig, auto_open=True, include_mathjax='cdn')            

