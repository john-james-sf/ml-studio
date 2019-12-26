# =========================================================================== #
#                              NORMALITY                                      #
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
# Create Date: Saturday December 7th 2019, 12:37:03 pm                        #
# Last Modified: Wednesday December 11th 2019, 10:19:48 pm                    #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Plots used to assess the normality of the data.""" 
#%%
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as py
from plotly.subplots import make_subplots

from ml_studio.data_analysis.normality import sample_quantiles
from ml_studio.data_analysis.normality import theoretical_quantiles
from ml_studio.visualate.base import DataVisualator
from ml_studio.utils.misc import proper, snake
# --------------------------------------------------------------------------- #
#                              HISTOGRAM                                      #
# --------------------------------------------------------------------------- #
class Histogram(DataVisualator):
    """Histogram.

    Histograms provide a sense of the distribution and concentration of the
    data. It reveals potential outliers, gaps in the data and unusual values
    as well as an estimated probability distribution.

    The histogram splits the data into equally sized bins, which are plotted
    along the x-axis. The y-axis contains the frequencies of datapoints
    that fall within each bin.

    The density plot is supported via the 'density' parameter. If 'density'
    is set to True, the density plot is rendered which estimates an 
    unobserved probability density function based upon the observed data.
    The probability density for a bin is the probability that a point
    will fall into that bin.  This is a smoothed, continuous version of the 
    histogram which uses kernel density estimation. Kernel density estimation 
    draws a continuous curve or kernel at each data point, then adds them 
    together to make a smooth continuous density estimation. The most common 
    kernel is the Gaussian which produces a Gaussian bell curve at each point. 

    For the density function, the x-axis is the value of the variable, just as
    with the histogram. The y-axis is the probability density function for 
    the kernel density estimation. Unlike a 'probability', the probability 
    'density' is the probability per unit on the x-axis. To actual probability
    is the area under the curve for a specific interval on the x-axis. The 
    y-axis can take values greater than one. The only requirement of the 
    density plot is that the total area under the curve integrates to one.

    Parameters
    ----------
    dataframe : DataFrame or array-like
        A pandas DataFrame or array-like with column names. If missing, a 
        DataFrame is constructed using the other parameters.

    x : str, int, Series or array-like
        Either a column in the dataframe, or a pandas Series or array-like
        object. Values from this argument are used to position marks along
        the x-axis.

    y : str, int, Series or array-like
        Either a column in the dataframe, or a pandas Series or array-like
        object. Values from this argument are used to position marks along
        the y-axis.        

    color : str or int or Series or array-like)
        Either a name of a column in data_frame, or a pandas Series or 
        array_like object. Values from this column or array_like are used 
        to assign color to marks.

    labels : dict with str keys and str values (default {})
        By default, column names are used in the figure for axis titles, 
        legend entries and hovers. This parameter allows this to be overridden. 
        The keys of this dict should correspond to column names, and the 
        values should correspond to the desired label to be displayed.    

    nbins : int. Defaults to 0.
        Specifies the maximum number of desired bins. This value will be used 
        in an algorithm that will decide the optimal bin size such that the 
        histogram best visualizes the distribution of the data. 

    cumulative : bool. Default is False
        If True, A cumulative histogram is rendered, which maps the cumulative 
        number of observations in all of the bins up to the specified bin. 

    marginal : str. Default is None
        String indicating the type of plot to be added to the margin of the
        plot. Choices are None, 'box', 'violin', and 'rug'.

    orientation : str, Default = 'v'
        Either 'h' for horizontal or 'v' for vertical orientation

    xrange : list of two numbers
        If provided, indicates the limits of the x-axis

    yrange : list of two numbers
          If provided, indicates the limits of the y-axis

    template : plotly layout Template instance
        The figure template

    width : int, Default None
        The width of figure in pixels

    height : int, Default 600
        The height of the figure in pixels.

    title : str. Defaults to Histogram + Variable name (if known) 
        The title for the plot. 

    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, nbins=0, cumulative=False, marginal=None, orientation='v', 
                 template="none", width=None, height=600,
                 title=None,  **kwargs):    
        self.color = color
        self.labels = labels
        self.nbins = nbins
        self.cumulative = cumulative
        self.marginal = marginal
        self.orientation = orientation
        self.xrange = xrange
        self.yrange = yrange
        self.template = template
        self.width = width
        self.height = height
        self.title = title
        
    def fit(self, x, y, dataframe=None, color=None, labels=None, nbins=0,
                 cumulative=False, marginal=None, orientation='v', xrange=None,
                 yrange=None, template="none", width=None, height=600,
                 title=None,  **kwargs):
        """ Fits the visualator to the data.
        
        Parameters
        ----------
        x : str, ndarray, pd.Series, pd.DataFrame of shape n x 1
            If a str, this is a column name of the dataset object. Otherwise
            it is an array-like containing values to be plotted

        y : str, ndarray, pd.Series, pd.DataFrame of shape n x 1
            If a str, this is a column name of the dataset object. Otherwise
            it is an array-like containing values to be plotted

        dataset : pd.DataFrame DataSet, or array-like
            Contains the data to be plotted. If not None, x and optionally
            y must be strings indicating column names. 

        Returns
        -------
        self : visualator
        """
        super(Histogram, self).fit(x=x, y=y, dataset=dataset)  

        self.title = self.title or "Histogram"

        #TODO: Remove the following once the DataStage class is done.
        self.dataframe = dataset

    def show(self, path=None,  **kwargs):        
        if self.z:
            barmode = 'overlay'        
        else:
            barmode = None

        self.fig = px.histogram(data_frame=self.dataframe, x=self.x, y=self.y,         
                                color=self.z, opacity=self.train_alpha, 
                                orientation=self.orientation, marginal=self.marginal,
                                cumulative=self.cumulative, nbins=self.nbins,
                                title=self.title, template=self.template,
                                barmode=barmode, width=self.width, height=self.height)          

        if path is not None:
            self.save_fig(path, element_name=self.x)

        self.fig.show()

        return self

def histogram(x, y=None, z=None, dataset=None, orientation=None, 
              marginal=None, cumulative=None, nbins=None, directory=None,
              title=None, name=None):
    """Functional interface to Histogram visualization."""
    v = Histogram(name=name, orientation=orientation, marginal=marginal,
                  cumulative=cumulative, nbins=nbins, title=title)
    v.fit(x, y, z, dataset)                  
    v.show(path=directory)



# --------------------------------------------------------------------------- #
#                              HISTOGRAM                                      #
# --------------------------------------------------------------------------- #
class DensityPlot(DataVisualator):
    """DensityPlot.

    The density plot is supported via the 'density' parameter. If 'density'
    is set to True, the density plot is rendered which estimates an 
    unobserved probability density function based upon the observed data.
    The probability density for a bin is the probability that a point
    will fall into that bin.  This is a smoothed, continuous version of the 
    histogram which uses kernel density estimation. Kernel density estimation 
    draws a continuous curve or kernel at each data point, then adds them 
    together to make a smooth continuous density estimation. The most common 
    kernel is the Gaussian which produces a Gaussian bell curve at each point. 

    For the density function, the x-axis is the value of the variable, just as
    with the histogram. The y-axis is the probability density function for 
    the kernel density estimation. Unlike a 'probability', the probability 
    'density' is the probability per unit on the x-axis. To actual probability
    is the area under the curve for a specific interval on the x-axis. The 
    y-axis can take values greater than one. The only requirement of the 
    density plot is that the total area under the curve integrates to one.

    Parameters
    ----------
    name : str
        The lower snake case name of the Histogram plot object, containing
        alphanumeric characters and underscores for separation.

    bin_size : float. Defaults to 1.0.
        Specifies the bin size. 

    group_labels : array-like
        A list of group labels, the length of which should match the 
        number of groups in the data.

    group_by : str, Defaults to None
        A string containing the grouping variable in the data set by which
        data will be grouped to produce separate histograms, one for each
        group.
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, name=None, bin_size=1.0, group_labels=[], title=None, 
                 **kwargs):    
        super(DensityPlot, self).__init__(name=name, 
                                          title=title,
                                          **kwargs)
        self.bin_size = bin_size
        self.group_labels = group_labels
        
    def fit(self, x, y=None, z=None, dataset=None):
        super(DensityPlot, self).fit(x, y, z, dataset)  
        
        self.title = self.title or "Density Plot"

        #TODO: Remove the following once the DataStage class is done.
        self.dataframe = dataset

    def show(self, path=None,  **kwargs):        
        
        # Convert hist_data to list of lists
        df = self.dataframe[self.x]
        if isinstance(df, pd.Series):
            self.dataframe = [df]
            self.group_labels = [self.x]
        else:
            self.dataframe = [df.iloc[i].tolist() for i in range(len(df.columns))]
            self.group_labels = list(self.x)


        self.fig = ff.create_distplot(hist_data=self.dataframe,
                                group_labels=self.group_labels, bin_size=self.bin_size,
                                curve_type='kde', histnorm='probability density',         
                                show_hist=True, show_curve=True, show_rug=False)    

        self.fig.update_layout(title_text=self.title, template=self.template,
                               height=self.height, width=self.width)     

        # Save figure if path is provided
        if path is not None:
            self.save_fig(path=path, element_name=self.x)

        self.fig.show()

        return self

def density_plot(hist_data=None, x=None, group_labels=None, bin_size=1.0, 
                 directory=None, title=None):
    """Functional interface to Histogram visualization."""
    v = Histogram(group_labels=group_labels, bin_size=bin_size, directory=directory,
                  title=title)
    v.fit(x=x, dataset=[hist_data])                  
    v.show(path=directory)

# --------------------------------------------------------------------------- #
#                           NORMAL PROBABILITY PLOT                           #
# --------------------------------------------------------------------------- #
class NormalProbability(DataVisualator):
    """Normal Probability Plot

    The validity of many hypothesis tests and the ability to make inferences 
    based upon test results rests on assumptions about the shape and distribution
    of the data being analyzed. The validity of machine learning models, in 
    particular, linear regression models, stems from the normality assumption.
    That is, parametric tests and linear regression require that the data 
    under examination approximate a normal distribution.

    The normal probability plot is one such visualization for assessing whether
    or not a data set is approximately normally distributed. The data are
    sorted in ascending order and plotted against a theoretical normal 
    distribution such that the points form an approximate straight 
    (diagonal) line. Departures from this straight line indiicate departures
    from normality.

    Parameters
    ----------
    title : str
        The title for the plot  

    kwargs : see PlotKwargs class documentation

    """

    def __init__(self, name=None, title=None, **kwargs):   
        super(NormalProbability, self).__init__(
                                     name=name,
                                     title=title,
                                     **kwargs)
        self.theoretical_quantiles = None
        self.x_sorted = None


    def fit(self, x, y=None, z=None, dataset=None):        
        super(NormalProbability, self).fit(x,y,z,dataset)        
        # Format plot title
        self.title = self.title or "Normal Probability Plot"

        # TODO: Delete once the base class functionality that reformats the 
        # data into the dataframe
        self.dataframe = dataset        

        # Compute theoretical quantiles
        self.theoretical_quantiles = theoretical_quantiles(self.dataframe[x])


    def show(self, path=None, **kwargs):
        self.path = path
        print(np.array(self.dataframe[self.x].sort_values()))
        # Create scatterplot traces
        data = [
            go.Scattergl(x=self.theoretical_quantiles, 
                         y=np.array(self.dataframe[self.x].sort_values()),
                         mode='markers',
                         marker=dict(color=self.train_color,
                                     size=2),
                         opacity=self.train_alpha,
                         name="Normal Probabilities",
                         showlegend=False)
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title=self.x,
                        showlegend=True,                        
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create Diagonal Line
        # self.fig.add_shape(
        #     go.layout.Shape(
        #         type="line",
        #         x0=np.min(self.theoretical_quantiles[0]),
        #         y0=0,
        #         x1=np.max(self.theoretical_quantiles[0]),
        #         y1=np.max(self.theoretical_quantiles[1]),
        #         line=dict(
        #             color=self.line_color
        #         )
        #     )
        # )
        # Render plot and save if path is provided
        if self.path:
            self.save_fig(path=path, element_name = self.x)
        
        py.plot(self.fig, auto_open=True, include_mathjax='cdn')            


