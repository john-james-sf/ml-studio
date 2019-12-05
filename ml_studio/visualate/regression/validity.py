# =========================================================================== #
#                                VALIDITY                                     #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \validity.py                                                          #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Thursday November 28th 2019, 8:53:35 am                        #
# Last Modified: Saturday November 30th 2019, 10:31:24 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #


"""Visualators used to assess the validity of regression models.""" 
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.model_evaluation.validity import standardized_residuals
from ml_studio.model_evaluation.validity import studentized_residuals
from ml_studio.model_evaluation.validity import quantile

from ..base import ModelVisualator
from ...utils.model import get_model_name
# --------------------------------------------------------------------------- #
#                           RESIDUAL VISUALATOR                               #
# --------------------------------------------------------------------------- #
class Residuals(ModelVisualator):
    """Residual plot.

    Renders a residual plot showing the residuals on the vertical axis and 
    the predicted values on the horizontal access.

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

    def __init__(self, model, refit=False, title=None, hist=True, **kwargs):    
        super(Residuals, self).__init__(model=model, 
                                        refit=refit, 
                                        title=title,
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
        super(Residuals, self).fit(X,y)
        # Format plot title
        if self.title is None:
            self.title = "Residuals Plot: " + self.model.name        
        # Compute predictions and residuals for training set
        self.y_train_pred = self.model.predict(X)        
        self.train_residuals = y - self.y_train_pred        
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
        # Create lowess smoothing line
        z1 = lowess(self.train_residuals, self.y_train_pred, frac=1./3, it=0, is_sorted=False)

        # Create scatterplot traces
        data = [
            go.Scattergl(x=self.y_train_pred, y=self.train_residuals,
                        mode='markers',
                        marker=dict(color=self.train_color),
                        name="Residual Plot",
                        showlegend=False,
                        opacity=self.train_alpha),
            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        opacity=self.train_alpha,
                        showlegend=False)                                                
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted Values",
                        yaxis_title="Residuals",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create and add shapes
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(self.y_train_pred),
                y0=0,
                x1=np.max(self.y_train_pred),
                y1=0,
                line=dict(
                    color=self.line_color
                )
            )
        )
        self.fig.update_shapes(dict(xref='x', yref='y'))

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
#                     STANDARDIZED RESIDUAL VISUALATOR                        #
# --------------------------------------------------------------------------- #
class StandardizedResiduals(ModelVisualator):
    """Standardized Residual plot.

    Renders a standardized residual plot showing the residuals on the vertical axis and 
    the predicted values on the horizontal access.

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

    def __init__(self, model, refit=False,  title=None, hist=True, **kwargs):    
        super(StandardizedResiduals, self).__init__(model=model, 
                                                    refit=refit, 
                                                    title=title,
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
        super(StandardizedResiduals, self).fit(X,y)        
        # Format plot title
        if self.title is None:
            self.title = "Standardized Residuals Plot: " + self.model.name        
        # Compute predictions and standardized residuals                
        self.train_residuals, self.y_train_pred = \
            standardized_residuals(self.model, X,y, return_predictions=True)        
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
        # Create lowess smoothing line
        z1 = lowess(self.train_residuals, self.y_train_pred, frac=1./3, it=0, is_sorted=False)        
                
        # Create scatterplot traces
        data = [
            go.Scattergl(x=self.y_train_pred, y=self.train_residuals,
                        mode='markers',
                        marker=dict(color=self.train_color),
                        name="Standardized Residuals",
                        showlegend=False,
                        opacity=self.train_alpha),
            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        opacity=self.train_alpha,
                        showlegend=False)                             
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted Values",
                        yaxis_title="Standardized Residuals",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        legend=dict(x=0, bgcolor='white'),
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create and add shapes
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(self.y_train_pred),
                y0=0,
                x1=np.max(self.y_train_pred),
                y1=0,
                line=dict(
                    color=self.line_color
                )
            )
        )
        self.fig.update_shapes(dict(xref='x', yref='y'))

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
#                     STUDENTIZED RESIDUAL VISUALATOR                         #
# --------------------------------------------------------------------------- #
class StudentizedResiduals(ModelVisualator):
    """Studentized Residual plot.

    Renders a studentized residual plot showing the residuals on the vertical axis and 
    the predicted values on the horizontal access.

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

    def __init__(self, model, refit=False, title=None, hist=True, **kwargs):    
        super(StudentizedResiduals, self).__init__(model=model, 
                                                   refit=refit, 
                                                   title=title,
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
        super(StudentizedResiduals, self).fit(X,y)        
        # Format plot title
        if self.title is None:
            self.title = "Studentized Residuals Plot: " + self.model.name        
        # Compute predictions and studentized residuals                
        self.train_residuals, self.y_train_pred = \
            studentized_residuals(self.model,X,y, return_predictions=True)        
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

        # Create lowess smoothing line
        z1 = lowess(self.train_residuals, self.y_train_pred, frac=1./3, it=0, is_sorted=False)  
                
        # Create scatterplot traces
        data = [
            go.Scattergl(x=self.y_train_pred, y=self.train_residuals,
                        mode='markers',
                        marker=dict(color=self.train_color),
                        name="Studentized Residuals",
                        showlegend=False,
                        opacity=self.train_alpha),
            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        opacity=self.train_alpha,
                        showlegend=False)                        
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted Values",
                        yaxis_title="Studentized Residuals",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        legend=dict(x=0, bgcolor='white'),
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create and add shapes
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(self.y_train_pred),
                y0=0,
                x1=np.max(self.y_train_pred),
                y1=0,
                line=dict(
                    color=self.line_color
                )
            )
        )
        self.fig.update_shapes(dict(xref='x', yref='y'))

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
#                        SCALE LOCATION VISUALATOR                            #
# --------------------------------------------------------------------------- #
class ScaleLocation(ModelVisualator):
    """Studentized Residual plot.

    Renders a scale location plot that shows the square root of the 
    standardized residuals against fitted values.  This plot is used to 
    evaluate homoscedasticity.

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

    def __init__(self, model, refit=False, title=None, hist=True, **kwargs):    
        super(ScaleLocation, self).__init__(model=model, 
                                            refit=refit, 
                                            title=title,
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
        super(ScaleLocation, self).fit(X,y)        
        # Format plot title
        if self.title is None:
            self.title = "Scale Location: : " + self.model.name        
        # Compute predictions and standardized residuals        
        self.train_residuals, self.y_train_pred = \
            standardized_residuals(self.model, X,y, return_predictions=True) 
        
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
        # Create lowess smoothing line
        z1 = lowess(np.sqrt(self.train_residuals), self.y_train_pred, frac=1./3, it=0, is_sorted=False)
        
        # Create scatterplot traces
        data = [
            go.Scattergl(x=self.y_train_pred, y=np.sqrt(self.train_residuals),
                        mode='markers',
                        marker=dict(color=self.train_color),
                        name="Scale Location",
                        showlegend=False,
                        opacity=self.train_alpha),
            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        opacity=self.train_alpha,
                        showlegend=False)                  
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted Values",
                        yaxis_title=r"$\sqrt{\text{Standardized Residuals}}$",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        legend=dict(x=0, bgcolor='white'),
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create and add shapes
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(self.y_train_pred),
                y0=0,
                x1=np.max(self.y_train_pred),
                y1=0,
                line=dict(
                    color=self.line_color
                )
            )
        )
        self.fig.update_shapes(dict(xref='x', yref='y'))

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
#                                 QQ NORM                                     #
# --------------------------------------------------------------------------- #
class QQPlot(ModelVisualator):
    """QQ Plot.

    The quantile-quantile (QQ) plot is used to show if two data sets come from 
    the same distribution. Concretely, one data set's quantiles are plotted 
    along the x-axis and the quantiles for the second distribution are plotted 
    on the y-axis. Typically, data sets are compared to the normal distribution.
    Considered the reference distribution, it is plotted along the X-axis as the
    "Theoretical Quantiles" while the sample is plotted along the Y-axis as
    the "Sample Quantiles".

    Parameters
    ----------    
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator.

    refit : Bool. Default is False
        Refit if True. Otherwise, only fit of model has not been trained        

    dist : A scipy.stats or statsmodels distribution
        Compare x against dist. The default is
        scipy.stats.distributions.norm (a standard normal).

    distargs : tuple
        A tuple of arguments passed to dist to specify it fully
        so dist.ppf may be called. distargs must not contain loc
        or scale. These values must be passed using the loc or
        scale inputs.

    a : float
        Offset for the plotting position of an expected order
        statistic, for example. The plotting positions are given
        by (i - a)/(nobs - 2*a + 1) for i in range(0,nobs+1)

    loc : float
        Location parameter for dist

    scale : float
        Scale parameter for dist

    fit : bool
        If fit is false, loc, scale, and distargs are passed to the
        distribution. If fit is True then the parameters for dist
        are fit automatically using dist.fit. The quantiles are formed
        from the standardized data, after subtracting the fitted loc
        and dividing by the fitted scale. 

    kwargs : see PlotKwargs class documentation

    """

    def __init__(self, model, refit=False,  a=0, loc=0, scale=1, 
                 title=None, **kwargs):   
        super(QQPlot, self).__init__(model=model, 
                                     refit=refit, 
                                     title=title,
                                     **kwargs)
        self.a = a
        self.loc = loc
        self.scale = scale 

    def fit(self, X, y=None):
        """ Fits the visualator to the data.
        
        Parameters
        ----------
        X : array-like
            1D Array

        y : ndarray or Series of length n
            Not used

        kwargs: dict
            Keyword arguments passed to the scikit-learn API. 
            See visualizer specific details for how to use
            the kwargs to modify the visualization or fitting process.    

        Returns
        -------
        self : visualator
        """
        super(QQPlot, self).fit(X, y)        
        # Format plot title
        if self.title is None:
            self.title = "QQ Plot: : " + self.model.name        

        # Compute residuals
        self.residuals = (y - self.model.predict(X)).flatten()
        # Obtain inverse CDF
        self.theoretical_quantiles = []
        for i in np.arange(len(self.residuals)):
            result = quantile(self.residuals[i])
            self.theoretical_quantiles.append(result)
        print(self.theoretical_quantiles[0:10])
        print(self.residuals[0:10])

        def plot_ppf(p):
            # Compute plotting position
            self.positions = []
            nobs = len(self.residuals)
            for i in np.arange(len(self.residuals)+1):
                p = (i-self.a)*(nobs-2*self.a + 1)
                self.positions.append(p)
            return self.positions           
        self.theoretical_points = plot_ppf(self.theoretical_quantiles)
        self.sample_points = plot_ppf(self.residuals)

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
        
        # Create scatterplot traces
        data = [
            go.Scattergl(x=np.arange(len(self.theoretical_points)), 
                         y=self.theoretical_points,
                         mode='markers',
                         marker=dict(color=self.DEFAULT_PARAMETERS['train_color'],
                                     size=3),
                         name="Theoretical Quantiles",
                         showlegend=True),
            go.Scattergl(x=np.arange(len(self.sample_points)), 
                         y=self.sample_points,
                         mode='markers',
                         marker=dict(color=self.DEFAULT_PARAMETERS['test_color'],
                                     size=3),
                         name="Sample Quantiles",                        
                         showlegend=True)
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles",
                        showlegend=True,                        
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create Diagonal Line
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(self.theoretical_quantiles[0]),
                y0=0,
                x1=np.max(self.theoretical_quantiles[0]),
                y1=np.max(self.theoretical_quantiles[1]),
                line=dict(
                    color=self.line_color
                )
            )
        )
        # Render plot and save if path is provided
        if self.path:
            py.plot(self.fig, filename=self.path, auto_open=True, include_mathjax='cdn')
        else:
            py.plot(self.fig, auto_open=True, include_mathjax='cdn')            
