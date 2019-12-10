# =========================================================================== #
#                              DISTRIBUTION                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \distribution.py                                                      #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 7th 2019, 12:37:03 pm                        #
# Last Modified: Monday December 9th 2019, 6:06:06 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Plots that reveal the distribution of the data.""" 
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

from ml_studio.visualate.base import DataVisualator
from ml_studio.utils.misc import proper, snake
from ml_studio.utils.file_manager import save_plotly

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
    name : str
        The lower snake case name of the Histogram plot object, containing
        alphanumeric characters and underscores for separation.

    nbins : int. Defaults to 0.
        Specifies the maximum number of desired bins. This value will be used 
        in an algorithm that will decide the optimal bin size such that the 
        histogram best visualizes the distribution of the data. 

    cumulative : bool. Default is False
        If True, A cumulative histogram is rendered, which maps the cumulative 
        number of observations in all of the bins up to the specified bin. 

    group_by : str, Defaults to None
        A string containing the grouping variable in the data set by which
        data will be grouped to produce separate histograms, one for each
        group.

    marginal : str. Default is None
        String indicating the type of plot to be added to the margin of the
        plot. Choices are None, 'box', 'violin', and 'rug'.

    horizontal : bool. Default is False
        If True, counts will be plotted on the x-axis.

    title : str. Defaults to Histogram + Variable name (if known) 
        The title for the plot. 
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, name=None, nbins=0, cumulative=False, 
                 marginal=None, orientation='v', title=None, **kwargs):    
        super(Histogram, self).__init__(name=name, title=title,**kwargs)
        self.nbins = nbins
        self.cumulative = cumulative
        self.marginal = marginal
        self.orientation = orientation
        
    def fit(self, x, y=None, z=None, dataset=None):
        """ Fits the visualator to the data.

        For ModelVisulator classes, the fit method fits the data to an underlying
        model. 
        
        Parameters
        ----------
        x : str or ndarray or DataFrame of shape n x m or 1D array
            If a str, this is a column name of the dataset object.
            A matrix of n instances with m features

        y : str or ndarray or Series or 1D array
            If a str, this is a column name of the dataset object.
            A matrix of n instances with m features

        dataset : pd.DataFrame DataSet, or array-like
            A DataSet object or an array-like with shap (m,n), where m is
            the number of observations and n is the number of variables,
            including the target.            

        Returns
        -------
        self : visualator
        """
        super(Histogram, self).fit(x, y, z, dataset)  

        if self.title is None:
            self.title = "Histogram " + proper(self.x) 

        #TODO: Remove the following once the DataStage class is done.
        self.dataframe = dataset

    def show(self, directory=None,  **kwargs):        
        """Renders the visualization.

        Contains the Plotly code that renders the visualization
        in a notebook or in a pop-up GUI. If the  path variable
        is not None, the visualization will be saved to disk.
        Subclasses will override with visualization specific logic.
        
        Parameters
        ----------
        directory : str
            The relative directory or directory to which the visualization 
            will be saved. 

        kwargs : dict
            Various keyword arguments.

        """
        x = self.x
        y = self.y
        z = self.z

        if z:
            barmode = 'overlay'        
        else:
            barmode = None

        self.fig = px.histogram(data_frame=self.dataframe, x=x, y=y,         
                                color=z, opacity=self.train_alpha, 
                                orientation=self.orientation, marginal=self.marginal,
                                cumulative=self.cumulative, nbins=self.nbins,
                                title=self.title, template=self.template,
                                barmode=barmode, width=self.width, height=self.height)          

        filename = self._get_filename(object_name=self.name, element_name=self.x)                                     

        if directory is not None:
            save_plotly(self.fig, directory=directory, filename=filename)                                

        self.fig.show()

        return self

def histogram(x, y=None, z=None, dataset=None, orientation=None, 
              marginal=None, cumulative=None, nbins=None, directory=None,
              title=None, name=None):
    """Functional interface to Histogram visualization."""
    v = Histogram(name=name, orientation=orientation, marginal=marginal,
                  cumulative=cumulative, nbins=nbins, title=title)
    v.fit(x, y, z, dataset)                  
    v.show(directory=directory)



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
        """ Fits the visualator to the data.

        For ModelVisulator classes, the fit method fits the data to an underlying
        model. 
        
        Parameters
        ----------
        x : str or ndarray or DataFrame of shape n x m or 1D array
            If a str, this is a column name of the dataset object.
            A matrix of n instances with m features

        y : Not used

        z: Not used

        dataset : pd.DataFrame DataSet, or array-like
            A DataSet object or an array-like with shap (m,n), where m is
            the number of observations and n is the number of variables,
            including the target.   

        Returns
        -------
        self : visualator
        """
        super(DensityPlot, self).fit(x, y, z, dataset)  

        if self.title is None:
            if len(self.x) == 1:
                self.title = "Density Plot " + proper(self.x) 
            else:
                self.title = "Density Plot"

        #TODO: Remove the following once the DataStage class is done.
        self.dataframe = dataset

    def show(self, directory=None,  **kwargs):        
        """Renders the visualization.

        Contains the Plotly code that renders the visualization
        in a notebook or in a pop-up GUI. If the  path variable
        is not None, the visualization will be saved to disk.
        Subclasses will override with visualization specific logic.
        
        Parameters
        ----------
        directory : str
            The relative directory or directory to which the visualization 
            will be saved. 

        kwargs : dict
            Various keyword arguments.

        """
        x = self.x
        y = self.y
        z = self.z

        
        # Convert hist_data to list of lists
        df = self.dataframe[x]
        if isinstance(df, pd.Series):
            self.dataframe = [df]
            self.group_labels = [x]
        else:
            self.dataframe = [df.iloc[i].tolist() for i in range(len(df.columns))]
            self.group_labels = list(x)


        self.fig = ff.create_distplot(hist_data=self.dataframe,
                                group_labels=self.group_labels, bin_size=self.bin_size,
                                curve_type='kde', histnorm='probability density',         
                                show_hist=True, show_curve=True, show_rug=False)    

        self.fig.update_layout(title_text=self.title, template=self.template,
                               height=self.height, width=self.width)     

        filename = self._get_filename(object_name=self.name, element_name=self.x[0])                                     

        if directory is not None:
            save_plotly(self.fig, directory=directory, filename=filename)                                

        self.fig.show()

        return self

def density_plot(hist_data=None, x=None, group_labels=None, bin_size=1.0, 
                 directory=None, title=None):
    """Functional interface to Histogram visualization."""
    v = Histogram(group_labels=group_labels, bin_size=bin_size, directory=directory,
                  title=title)
    v.fit(x=x, dataset=[hist_data])                  
    v.show(directory=directory)


