#!/usr/bin/env python3
# =========================================================================== #
#                                  BASE                                       #
# =========================================================================== #
# =========================================================================== #
# Project: Visualate                                                          #
# Version: 0.1.0                                                              #
# File: \base.py                                                              #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Wednesday November 27th 2019, 10:28:47 am                      #
# Last Modified: Wednesday November 27th 2019, 12:51:57 pm                    #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Base class and interface for all Visualators"""
import os
import time

from abc import ABC, abstractmethod, ABCMeta
from itertools import chain
import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as po
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator

from ..supervised_learning.training.estimator import Estimator
from ..supervised_learning.regression import LinearRegression
from ..utils.model import get_model_name
from ..utils.misc import snake
from ..utils.file_manager import save_plotly, get_filename

# --------------------------------------------------------------------------- #
#                            BASE VISUALATOR                                  #
# --------------------------------------------------------------------------- #

class BaseVisualator(ABC, BaseEstimator, metaclass=ABCMeta):
    """Abstact base class at the top of the visualator object hierarchy.

    Class defines the interface for creating, storing and rendering 
    visualizations using Plotly.

    Parameters
    ----------
    canvas : Canvas object
        Object containing configurable plotting options.

    Notes
    -----
    There are four types of visualization subclasses: DataVisualator, 
    ModelVisualator, GroupVisualator, and CrossVisualator. The DataVisualator
    is used to analyze data prior to model building and selection.  
    The ModelVisualator renders visualizations for a single model. 
    GroupVisualator accepts a list of Visualator objects and delivers 
    visualizations using subplots.  The CrossVisualator wraps a 
    Scikit-Learn GridSearchCV or RandomizedSearchCV  object and presents 
    model selection visualizations.  Those inherit directly from this class.
    """

    def __init__(self, canvas=None, *args, **kwargs):  
        self.canvas = canvas

    def _init_layout(self):
        """Initializes the plotly layout object with canvas settings."""
        

    @abstractmethod
    def fit(self, dataframe=None, x=None, y=None, z=None):
        """ Fits the visualator to the data.

        For DataVisualator classes, this method fits the data to the visualator.
        For ModelVisulator classes, the fit method fits the data to an underlying
        model. GroupVisulators iteratively fit several models to the data. The 
        CrossVisulators call the fit methods on GridSearchCV and RandomizedSearchCV 
        objects.

        Parameters
        ----------
        dataset : pandas DataFrame or ML Studio DataSet object.             
            Tidy (“long-form”) dataframe where each column is a variable and 
            each row is an observation or a DataSet object containing a 
            DataFrame object.

        x : str or array-like of shape (m,), where m is the number 
            of observations. if x is a str, it is a column in the dataframe 
            argument.

        y : str or array-like of shape (m,), where m is the number 
            of observations. if y is a str, it is a column in the dataframe 
            argument.

        z : str or array-like of shape (m,), where m is the number 
            of observations. if x is a str, it is a column in the dataframe 
            argument. Whether z is an array-like or a str column name,
            it typically represents categorical or grouping factors for
            x, and optionally y.              

        Returns
        -------
        self : visualator
        """
        pass
    @abstractmethod
    def show(self):
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
        pass

    def save_fig(self, directory, filename=None):
        """Saves a figure to file.
        
        If filename is not provided, a filename will be generated using the 
        following format:

            'user_classname_dataframe_name_xvar_yvar_zvar_time_stamp.html'        

        Variables 'dataframe name', 'xvar', 'yvar', and 'zvar', if provided, 
        will be obtained or derived from the parameters of the fit method.

        Parameters
        ----------
        directory : str
            Relative or absolute directory path

        filename : str
            Name of the file to which the plot is being saved. It should
            end with an '.html' extension. If missing, it will be added to
            the filename
        
        """

        if filename:
            if os.path.splitext(filename)[1] == "":
                # The .html file extension must be added to the filename.
                filename += '.html'
        else:
            filename = get_filename(self, ext='.html')

        save_plotly(self.fig, directory=directory, filename=filename)    




# --------------------------------------------------------------------------- #
#                            DATA VISUALATOR                                  #
# --------------------------------------------------------------------------- #
class DataVisualator(BaseVisualator):
    """Abstact base class for data visualators.

    Class defines the interface for creating Plotly visualizations of data 
    prior to model building and selection. 

    Parameters
    ----------
    title : str
        The title for the plot. It defaults to the plot name and optionally
        the model name.

    kwargs : see docstring for PlotKwargs class.  

    """

    def __init__(self, title=None, **kwargs):    
        """Instantiate the object and specify data input requirements."""
        super(DataVisualator, self).__init__(title=title,
                                             **kwargs)        


    def fit(self, x, y=None, z=None, dataset=None):
        """ Fits the visualator to the data.

        For ModelVisulator classes, the fit method fits the data to an underlying
        model. 
        
        Parameters
        ----------
        x : str, ndarray, pd.Series, pd.DataFrame of shape n x 1
            If a str, this is a column name of the dataset object. Otherwise
            it is an array-like containing values to be plotted

        y : str, ndarray, pd.Series, pd.DataFrame of shape n x 1
            If a str, this is a column name of the dataset object. Otherwise
            it is an array-like containing values to be plotted

        z : str, ndarray, pd.Series, pd.DataFrame of shape n x 1
            If a str, this is a column name of the dataset object. Otherwise
            it is an array-like containing values to be plotted

        dataset : pd.DataFrame DataSet, or array-like
            Contains the data to be plotted. If not None, x and optionally
            y and z must be strings indicating column names. 

        Returns
        -------
        self : visualator
        """
        # Store X, y as class instance variables
        self.x = x
        self.y = y
        self.z = z
        self.dataset = dataset

        #TODO
        # Validate the data
        # Instantiate the Validatrix, and invoke the validate the instantiation
        # the parameters and the data. 
        # v = Validatrix()
        # v.validate(self)

        #TODO
        # Ratify the data
        # Convert the data to pd.DataFrame format using the DataStage class.
        # d = DataStage()
        # d.fit(x, y, z, dataset)
        # self.x = d.x
        # self.y = d.y
        # self.dataframe = d.dataframe

        return self


# --------------------------------------------------------------------------- #
#                            MODEL VISUALATOR                                 #
# --------------------------------------------------------------------------- #
class ModelVisualator(BaseVisualator):
    """Abstact base class for model visualators.

    Class defines the interface for creating Plotly visualizations of 
    individual models. 

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn estimator which wraps functionality

    refit : Bool. Default is False
        Refit if True. Otherwise, only fit of model has not been trained        

    kwargs : see PlotKwargs class documentation

    """

    def __init__(self, model, refit=False, title=None, **kwargs):    
        super(ModelVisualator, self).__init__(title=title,
                                              **kwargs)
        self.model = model
        self.refit = refit
        self.nobs = 0

    def transform(self, X, y=None):
        """Transforms the data.

        Method exposed for subclasses that perform transformations on 
        X and optionally y.

        Parameters
        ---------- 
        X : array-like, shape (n_samples, n_features)
            Feature dataset to be transformed.

        y : array-like, shape (n_samples,)
            Dependent target data associated with X.        

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Returns X un-transformed.

        y : array-like, shape (n_samples,) (optional)
            Not used in this base class.

        """
        pass
    
    def fit(self, X, y, **kwargs):
        """ Fits the visualator to the data.
        
        For ModelVisulator classes, the fit method fits the data to an underlying
        model. 

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
        self.X_train = X
        self.y_train = y
        self.nobs = X.shape[0]
        self.n_features = X.shape[1]
        if self.refit or not self.model.fitted:        
            self.model.fit(X,y)
        return self        
    
    def score(self, X, y, **kwargs):
        """Computes a score for a Model.

        Score invokes the estimator's score method and makes predictions based
        upon X and scores them relative to y.

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
        metric : str
            String indicating the name of the metric

        train_score : float or array-like
            Returns the training score of the underlying model, the metric 
            is specified in the model's hyperparameters.

        test_score : float or array-like
            Returns the test score of the underlying model, the metric 
            is specified in the model's hyperparameters.            
        """
        self.X_test = X
        self.y_test = y

        # Scikit Learn estimators compute r2 scores. ML Studio estimator metric
        # must be obtained from the model
        if isinstance(self.model, Estimator):
            self.metric = self.model.metric_name
        else:
            self.metric = r"$R^2"
        # Compute training and test scores
        self.train_score = self.model.score(self.X_train, self.y_train)        
        self.test_score = self.model.score(X, y) 

        return self

    @abstractmethod
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
        pass        

# --------------------------------------------------------------------------- #
#                             VISUALATORS                                     #
# --------------------------------------------------------------------------- #
class Visualators(BaseVisualator):
    """Abstact base class for collections of Visualators.

    Class defines the interface for rendering multiple Plotly visualizations. 

    Parameters
    ----------
    visualators : A list of instantiated visualators

    nrows: integer, default: None
        The number of rows desired, if you would like a fixed number of rows.
        Specify only one of nrows and ncols, the other should be None. If you
        specify nrows, there will be enough columns created to fit all the
        visualators specified in the visualators list.

    ncols: integer, default: None
        The number of columns desired, if you would like a fixed number of columns.
        Specify only one of nrows and ncols, the other should be None. If you
        specify ncols, there will be enough rows created to fit all the
        visualators specified in the visualators list.

    kwargs : dict, default: None
        Additional keyword arguments that define the layout for visualators 
        subclasses.   

    """
    DEFAULT_PARAMETERS = {'height': 250, 'width': 400, 'template': "plotly_white"}

    def __init__(self, title=None, visualators=[], nrows=None, ncols=None, **kwargs):    
        super(Visualators, self).__init__(title=title, **kwargs)
        self.visualators = visualators
        self.nrows = nrows
        self.ncols = ncols

    def fit(self, X, y, **kwargs):
        """ Fits the visualators.

        Iteratively fits the visualators.

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
        self : visualators object
        """        

        # Set plot rows and columns
        plotcount = len(self.visualators)
        if self.nrows is None and self.ncols is None:
            self.ncols = 1
            self.nrows = plotcount
        elif self.ncols is None:            
            self.ncols = int(math.ceil(plotcount/self.nrows))
        elif self.nrows is None:
            self.nrows=int(math.ceil(plotcount/self.ncols))
        else:
            raise ValueError(
        "Either nrows or ncols must be None. The parameter = 'None' will be "
        "calculated based upon the number of visualators in the list."
        )

        for viz in self.visualators:
            viz.fit(X,y, **kwargs)

        return self        

    def score(self, X, y, **kwargs):
        """Score the visualators.

        Iteratively invoke the score methods on the visualators.

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
        self : visualators object
            Returns the score of the underlying model, the metric is specified
            in the model's hyperparameters.
        """

        for viz in self.visualators:
            viz.score(X, y)                          

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
        # Create subplots
        self.fig = make_subplots(rows=self.nrows, cols=self.ncols)

        # Import traces as a list of dictionaries
        idx = 0
        for row in range(self.nrows):
            for col in range(self.ncols):
                try:
                    data = self.visualators[idx].fig.data
                except IndexError:
                    print("index on visualators object out of range.")
                for trace in data:
                    try:
                        self.fig.add_trace(trace, row=row, col=col)
                    except IndexError:
                        print("row or column index on trace object out of range.")
                idx += 1

        self.fig.update_layout(title={"text": self.title},
                           height=self.height,
                           width=self.width,
                           template=self.template)
        

# --------------------------------------------------------------------------- #
#                            MODEL VISUALATOR                                 #
# --------------------------------------------------------------------------- #
class CrossVisualator(BaseVisualator):
    """Abstact base class for cross validation visualators.

    Class defines the interface for creating Plotly visualizations of 
    GridSearchCV and RandomizedSearchCV models. 

    Parameters
    ----------
    cv : a Scikit-Learn GridSearchCv or RandomizedSearchCV object
        Contains results of k-fold cross validations

    kwargs : see PlotKwargs class documentation.

    Attributes
    ----------
    cv : GridSearchCV or RandomizedSearchCV object
        Scikit Learn cross-validation objects

    best_estimator : BaseEstimator
        The best performing Scikit Learn or ML Studio estimator

    best_params : dict
        Contains the hyperparameters for the best performing model

    score : float
        The score for the best performing model.

    refit_time : int
        The number of seconds it took to train the best model on the entire 
        dataset  

    n_splits : int
        The number of k-folds used in cross-validation

    metric : str
        The metric used to score performance

    cv_results_df : pd.DataFrame
        Further cross validation statistics such as: 
            mean_train_score
            std_train_score
            mean_fit_time
            std_fit_time
            etc... 

    """

    def __init__(self, cv, title=None, **kwargs):    
        super(CrossVisualator, self).__init__(title=title,
                                              **kwargs)        
        self.cv = cv
        self.name = get_model_name(cv)
    
    def fit(self, X, y, **kwargs):
        """ Fits the cv object to the data.
        
        Fits the CV object to the data.

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
        self.cv.fit(X, y)
        return self
    
    def score(self, X, y, **kwargs):
        """Computes a score for the best model obtained from cross-validation.

        Score invokes the estimator's score method on the best model and makes 
        predictions based upon X and scores them relative to y.

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
        score : float or array-like
            Returns the score of the underlying model, the metric is specified
            in the model's hyperparameters.
        """
        # Obtain cross validation results
        self.best_estimator = self.cv.cv_results_['best_estimator_']
        self.best_params = self.cv.cv_results_['best_params_']
        self.best_score = self.cv.cv_results_['best_score_']
        self.refit_time = self.cv.cv_results_['refit_time_']
        self.n_splits = self.cv.cv_results_['n_splits_']
        self.metric = [value for key, value in self.cv.cv_results_.items() \
                            if '_<scorer_name>' in key]
        self.cv_results_df = pd.DataFrame.from_dict(self.cv.cv_results_)
        
        # Compute score on X and y
        self.test_score = self.best_estimator.score(X,y)