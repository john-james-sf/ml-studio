# =========================================================================== #
#                                INFLUENCE                                    #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \influence.py                                                         #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Thursday November 28th 2019, 8:46:35 pm                        #
# Last Modified: Saturday November 30th 2019, 10:32:04 am                     #
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
from ml_studio.model_evaluation.validity import standardized_residuals
from ml_studio.model_evaluation.validity import studentized_residuals
from ml_studio.model_evaluation.influence import leverage, cooks_distance

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

    refit : Bool. Default is False
        Refit if True. Otherwise, only fit of model has not been trained        

    hist : bool, 'density', 'frequency', default: True
        Draw a histogram showing the distribution of the residuals on the 
        right side of the figure. If set to 'density', the probability
        density will be plotted. If set to 'frequency', the frequency will
        be plotted.  
    
    kwargs : see PlotKwargs class documentation

    """

    def __init__(self, model, refit=False, hist=False, **kwargs):    
        super(ResidualsLeverage, self).__init__(model=model, 
                                                refit=refit, 
                                                **kwargs)
        self.hist = hist

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
        self.train_residuals, self.y_train_pred =\
            standardized_residuals(self.model, X, y, return_predictions=True)
        
        # Compute leverage and cooks distances
        self.leverage = leverage(X)
        self.cooks_d = cooks_distance(self.model, X, y)

        return self           

    def _cooks_contour(self, distance=0.5, sign=1):
        """Computes the leverage v standardized residuals, given Cooks Distance."""        
        # Compute F-distribution degrees of freedom
        n = self.X_train.shape[0]
        p = self.X_train.shape[1]

        # Designate the leverage range being plotted
        min_leverage = np.min(self.leverage)
        max_leverage = np.max(self.leverage)
        leverage = np.linspace(start=min_leverage, stop=max_leverage, num=n)

        # Compute standard residuals
        std_resid = sign * np.sqrt(distance * ((1-leverage)/leverage) * (p+1))        

        # Establish evaluator
        better = {1:np.greater, -1:np.less}

        # Find outliers better than threshold
        threshold = sign * np.sqrt(distance * ((1-self.leverage)/self.leverage) * (p+1))        
        outliers = np.argwhere(better[sign](self.train_residuals,threshold))
        return leverage, std_resid, outliers

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

        # Grab the outliers 
        outliers = []
        l1u_leverage, l1u_residual, outlier = \
            self._cooks_contour(distance=0.5, sign=1)
        if len(outlier) > 0:
            outliers.append(outlier)
        
        l1l_leverage, l1l_residual, outlier = \
            self._cooks_contour(distance=0.5, sign=-1)
        if len(outlier) > 0:
            outliers.append(outlier)

        l2u_leverage, l2u_residual, outlier = \
            self._cooks_contour(distance=1, sign=1)
        if len(outlier) > 0:
            outliers.append(outlier)

        l2l_leverage, l2l_residual, outlier = \
            self._cooks_contour(distance=1, sign=-1)
        if len(outlier) > 0:
            outliers.append(outlier)

        # Format the annotations
        annotations = np.arange(len(self.train_residuals))
        annotations = annotations.astype(object)
        notoutlier = np.array([i for i in np.arange(len(self.train_residuals)) if i not in outliers])
        annotations[notoutlier] = " "
        
        # Create scatterplot traces
        data = [
            go.Scatter(x=self.leverage, y=self.train_residuals,
                        mode='markers+text',
                        marker=dict(color=self.train_color),
                        name="Residual vs Leverage",
                        text=annotations,
                        textposition="top center",
                        showlegend=False,
                        opacity=self.train_alpha),
            go.Scattergl(x=l1u_leverage[outliers], 
                        y=l1u_residual[outliers], 
                        mode='markers',
                        marker=dict(color='red'),
                        name="Bad Points",
                        opacity=self.train_alpha,
                        showlegend=True),                        
            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        opacity=self.train_alpha,
                        showlegend=False),
            go.Scatter(x=l1u_leverage,
                       y=l1u_residual,
                       mode='lines',
                       line=dict(dash='dash'),
                       marker=dict(color='red'),
                       name="Cooks' D = 0.5",
                       opacity=self.train_alpha,
                       showlegend=True),
            go.Scatter(x=l1l_leverage,
                       y=l1l_residual,
                       mode='lines',
                       line=dict(dash='dash'),
                       marker=dict(color='red'),
                       name="Cooks' D = 0.5 (Lower)",
                       opacity=self.train_alpha,
                       showlegend=False),
            go.Scatter(x=l2u_leverage,
                       y=l2u_residual,
                       mode='lines',
                       line=dict(dash='dot'),
                       marker=dict(color='red'),
                       name="Cooks' D = 1.0",
                       opacity=self.train_alpha,
                       showlegend=True),
            go.Scatter(x=l2l_leverage,
                       y=l2l_residual,
                       mode='lines',
                       line=dict(dash='dot'),
                       marker=dict(color='red'),
                       name="Cooks' D = 1 (Lower)",
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
                        margin=self.margin,
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

# --------------------------------------------------------------------------- #
#                              COOKS DISTANCE                                 #
# --------------------------------------------------------------------------- #
class CooksDistance(ModelVisualator):
    """Cooks Distance.

    Cook's distance" is a measure of the influence of each observation on the 
    regression coefficients. The Cook's distance statistic is a measure, 
    for each observation in turn, of the extent of change in model 
    estimates when that particular observation is omitted. Any observation 
    for which the Cook's distance is close to 0.5 or more, or that is 
    substantially larger than other Cook's distances 
    (highly influential data points), requires investigation.

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator.

    refit : Bool. Default is False
        Refit if True. Otherwise, only fit of model has not been trained           
    
    kwargs : see PlotKwargs class documentation

    """

    def __init__(self, model, refit=False, hist=False, **kwargs):    
        super(CooksDistance, self).__init__(model=model, 
                                            refit=refit, 
                                            **kwargs)               
        self.hist = hist

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
        super(CooksDistance, self).fit(X,y)        
        # Format plot title
        if self.title is None:
            self.title = "Cooks Distance : " + self.model.name        
        
        # Compute Cooks Distance
        self.cooks_d = cooks_distance(self.model, X, y)

        return self                   

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
        # Print format options                
        
        # Create scatterplot traces
        data = [
            go.Scattergl(x=np.arange(len(self.cooks_d)), y=[round(x,4) for x in self.cooks_d],
                        mode='lines',
                        marker=dict(color=self.train_color),
                        name="Residual vs Leverage",                                                
                        showlegend=False,
                        opacity=self.train_alpha)                                                                                                                             
       ]
        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Observation",
                        yaxis_title="Cooks Distance",
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)  

        # Add horizontal line if any appoach Cooks Distance of 0.5
        if len(self.cooks_d[self.cooks_d > 0.5]) > 0:
           self.fig.add_shape(
               go.layout.Shape(
                   type="line",
                   x0=0,
                   y0=0.5,
                   x1=len(self.cooks_d),
                   y1=0.5,
                   line=dict(
                       color="darkgrey",
                       width=2
                   )
               )
           )
                                
        # Render plot and save if path is provided
        if self.path:
            py.plot(self.fig, filename=self.path, auto_open=True, include_mathjax='cdn')
        else:
            py.plot(self.fig, auto_open=True, include_mathjax='cdn')            
