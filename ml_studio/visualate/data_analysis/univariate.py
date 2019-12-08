# =========================================================================== #
#                          UNIVARIATE ANALYSIS                                #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \univariate.py                                                        #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 7th 2019, 12:37:03 pm                        #
# Last Modified: Saturday December 7th 2019, 12:37:44 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Univariate analysis of quantitative and qualitative data.""" 
#%%
import os

import numpy as np
import pandas as pd
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

    Histogram reveals the shape and distribution of quantitative variables.
    Each bar in a histogram represents the tabulated frequency at each 
    interval/bin.
    
    Histograms provide a sense of the distribution and concentration of the
    data. It reveals potential outliers, gaps in the data and unusual values
    as well as an estimated probability distribution.

    Parameters
    ----------
    title : str. Defaults to Histogram + Variable name (if known) 
        The title for the plot. 
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, dataset_name=None, title=None, density=False, **kwargs):    
        super(Histogram, self).__init__(dataset_name=dataset_name, 
                                        title=title,**kwargs)        
        self.density = density

    def fit(self, X, y=None):
        """ Fits the visualator to the data.

        If X is a matrix, a histogram is produced for 
        
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m or 1D array
            A matrix of n instances with m features

        y : ndarray or Series of length n
            Not used

        Returns
        -------
        self : visualator
        """
        super(Histogram, self).fit(X,y)  
    

    def _histogram(self, variable):
        """Prints and optionally saves individual histogram."""

        # Designate whether the probability density should be rendered
        histnorm = None
        if self.density:
            histnorm='probability'

        # Obtain the histogram object
        data=[go.Histogram(x=self.df[variable], 
                         histnorm=histnorm,
                         nbinsx=100,
                         autobinx=True)]

        # Specify the layout.
        layout = go.Layout(
            height=self.height,
            width=self.width,
            template=self.template
        )

        self.fig = go.Figure(data=data, layout=layout)

        # Format the title and update trace and layout
        if self.title is None:
            title = "Histogram : " + proper(variable) 
        else:
            title = self.title
            text = "Histogram : " + proper(variable)
            x = 0.5
            y = 1.10
            self.fig.update_layout(annotations = [dict(text=text, x=x, y=y,
                                                  xref='paper', yref='paper',
                                                  showarrow=False)])

        # Update trace and layout
        self.fig.update_layout(title=title)

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

        self.figures = []
        for variable in self.numeric_variable_names:            
            self._histogram(variable)
            if directory:
                filename = self._get_filename(object_name=self.dataset_name, element_name=variable)
                self.save(self.fig, directory, filename)            

        return self

def histogram(X, y=None, dataset_name=None, density=False, title=None,
              directory=None):
    """Functional interface to Histogram visualization."""
    v = Histogram(dataset_name=dataset_name, title=title, density=density)
    v.fit(X, y)
    v.show(directory=directory)


