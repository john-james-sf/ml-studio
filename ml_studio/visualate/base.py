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
from ..utils.file_manager import save_plotly

# --------------------------------------------------------------------------- #
#                            PlotKwargs                                       #
# --------------------------------------------------------------------------- #
class PlotKwargs():
    """
    Keyword Parameters
    ------------------
    kwargs : dict
        Keyword arguments that define the plot layout for visualization subclasses.   

        ===========  ========== ===============================================
        Property     Format     Description
        -----------  ---------- -----------------------------------------------
        height       int        the height in pixels for the figure
        line_color   str        default line color
        margin       dict       margin in pixels. dict keys are l, t, and b        
        template     str        the theme template 
        test_alpha   float      opacity for objects associated with test data
        test_color   str        color for objects associated with test data                
        train_alpha  float      opacity for objects associated with training data
        train_color  str        color for objects associated with training data
        width        int        the width in pixels of the figure
        ===========  ========== ===============================================    
    """

# --------------------------------------------------------------------------- #
#                            BASE VISUALATOR                                  #
# --------------------------------------------------------------------------- #

class BaseVisualator(ABC, BaseEstimator, metaclass=ABCMeta):
    """Abstact base class at the top of the visualator object hierarchy.

    Class defines the interface for creating, storing and rendering 
    visualizations using Plotly.

    Parameters
    ----------
    title : str
        The title for the plot. It defaults to the plot name and optionally
        the model name.

    kwargs : see docstring for PlotKwargs class.        

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
    DEFAULT_PARAMETERS = {'height': 450, 'line_color': 'darkgrey', 
                          'margin': {'l':80, 't':100, 'b':80}, 'template': "none",
                          'test_alpha': 0.75, 'test_color': '#9fc377',
                          'train_alpha': 0.75, 'train_color': '#0272a2',
                          'width': 700}

    ARRAY_LIKE = (np.ndarray, np.generic, list, dict, pd.Series, \
                pd.DataFrame, tuple)

    def __init__(self, title=None, **kwargs):     
        self.title = title
        self.height = kwargs.get('height', self.DEFAULT_PARAMETERS['height'])
        self.width = kwargs.get('width', self.DEFAULT_PARAMETERS['width'])
        self.line_color = kwargs.get('line_color', \
            self.DEFAULT_PARAMETERS['line_color'])
        self.train_color = kwargs.get('train_color', \
            self.DEFAULT_PARAMETERS['train_color'])
        self.test_color = kwargs.get('test_color', \
            self.DEFAULT_PARAMETERS['test_color'])
        self.train_alpha = kwargs.get('train_alpha', \
            self.DEFAULT_PARAMETERS['train_alpha'])
        self.test_alpha = kwargs.get('test_alph', \
            self.DEFAULT_PARAMETERS['test_alpha'])
        self.template = kwargs.get('template', \
            self.DEFAULT_PARAMETERS['template'])
        self.margin = kwargs.get('margin', \
            self.DEFAULT_PARAMETERS['margin'])
        self.filetype = ".html"


    @abstractmethod
    def fit(self, X, y, **kwargs):
        """ Fits the visualator to the data.

        For DataVisualator classes, this method fits the data to the visualator.
        For ModelVisulator classes, the fit method fits the data to an underlying
        model. GroupVisulators iteratively fit several models to the data. The 
        CrossVisulators call the fit methods on GridSearchCV and RandomizedSearchCV 
        objects.

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
        pass
    
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

    def _get_filename(self, object_name=None, element_name=None):
        """Creates a standard format filename for saving plots."""    

        # Obtain user id, class name and date time        
        userhome = os.path.expanduser('~')          
        username = os.path.split(userhome)[-1] 
        object_name = object_name or ""       
        clsname = self.__class__.__name__
        element_name = element_name or ""
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # Snake case format filename
        filename = username + '_' + object_name + '_' + clsname + '_' + \
        element_name + '_' + timestr + self.filetype
        filename = snake(filename)        
        return filename


    def save(self, fig, directory, filename):
        """Saves a plot to file.
        
        Parameters
        ----------
        fig : plotly figure object
            The figure to be saved

        directory : str
            The name of the directory to which the file is to be saved

        filename : str
            The name of the file to be saved.
        
        """

        save_plotly(fig, directory=directory, filename=filename)    




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

    Attributes
    ----------
    n_features : int
        The number of features in the dataset X

    n_numeric_features : int
        The number of numeric features in the dataset X

    n_categorical_features : int
        The number of categorical features in the dataset X

    feature_names : list
        List of the names of the features in the dataset X

    numeric_feature_names : list
        List of the names of the numeric features in the dataset X

    categorical_feature_names : list
        List of the names of the categorical features in the dataset X        

    figures : list
        List of figure objects 

    """

    def __init__(self, dataset_name=None, title=None, **kwargs):    
        super(DataVisualator, self).__init__(title=title,
                                             **kwargs)
                                
        self.dataset_name = dataset_name
        self.n_features = None
        self.n_numeric_features = None
        self.n_categorical_features = None
        self.feature_names = None
        self.numeric_feature_names = None
        self.categorical_feature_names = None
        self._numeric_data_types = ['int16', 'int32', 'int64', 'float16', \
                                    'float32', 'float64']
        self._categorical_data_types = ['category', 'object']                                    
        


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

    def _validate(self, X, y):
        """Validates type of X and y."""

        # Validate types
        if not isinstance(X,  self.ARRAY_LIKE):
            raise TypeError("X must be an np.ndarray, pd.DataFrame, "
                            "pd.Series or list")
        if y is None:
            pass
        elif not isinstance(y,  self.ARRAY_LIKE):
            raise TypeError("y must be an np.ndarray, pd.DataFrame, "
                                "pd.Series or list")

        # Validate compatible lengths
        if isinstance(y,  self.ARRAY_LIKE):
            try:
                X_len = X.shape[0]
            except AttributeError:
                X_len = len(X)
            if X_len != len(y):
                raise ValueError("X and y have incompatible lengths. "
                                "X has a length of %d, y's length is %d" 
                                % (X_len, len(y)))

    def _get_object_names(self, x):
        """Gets (col) names from pandas or numpy object or returns None."""        
        if isinstance(x, pd.DataFrame):            
            names = x.columns
        elif isinstance(x, pd.Series):            
            names = x.name
        elif isinstance(x, (np.generic, np.ndarray)):            
            names = x.dtype.names
        else:
            names = None
        return names

    def _generate_variable_names(self, x, target=False):
        """Generates a list of variable names based upon shape of x."""
        if target:
            var_names = ['target']
        elif len(x.shape) == 1:
            var_names = ['var_0']            
        elif x.shape[1] == 1:
            var_names = ['var_0']
        else:
            var_names = ["var_" + str(i) for i in range(x.shape[1])] 
        return var_names        

    def _get_variable_names(self, x, target=False, **kwargs):
        """Gets variable names from object or generate a dummy name."""
        # Obtain variable names from kwargs if available
        var_names = None
        if target:
            var_names = kwargs.get('target_name', None)
        else:
            var_names = kwargs.get('feature_names', None)
        if isinstance(var_names, self.ARRAY_LIKE):
            return var_names
        
        # Ok, try extracting variable names from the objects themselves
        var_names = self._get_object_names(x)
        if isinstance(var_names, self.ARRAY_LIKE):
            return var_names
        
        # Alright, let's create dummy variable names since none are available.
        var_names = self._generate_variable_names(x, target) 
        return var_names


    def _reformat(self, x, target=False, **kwargs):
        """Reformats data into a dataframe."""
        var_names = self._get_variable_names(x, target, **kwargs)
        if isinstance(x, pd.DataFrame):
            return x
        else:
            return pd.DataFrame(data=x, columns=var_names)  

    def fit(self, X, y, **kwargs):
        """ Fits the visualator to the data.

        Prepares the data for plotting. Numpy arrays are converted to pandas
        DataFrames, a format more conducive to exploratory data analysis.
        Feature names, types and counts are computed.  

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Additional keyword arguments used for presenting visualizations,
            including:

            feature_names : array-like
                Names of features in X

            target_name : str
                The name for the target variable in the dataset.    

        Returns
        -------
        self : visualator
        """
        # Validate data
        self._validate(X,y)

        # Store X, y as class instance variables
        self.X = X
        self.y = y

        # Reformat data into pd.DataFrames 
        X_df = self._reformat(X, target=False, **kwargs)
        
        # Obtain feature names, types and counts
        self.feature_names = X_df.columns
        self.n_features = len(self.feature_names)               
        self.numeric_feature_names = \
            X_df.select_dtypes(include=self._numeric_data_types).columns        
        self.n_numeric_features = len(self.numeric_feature_names)
        self.categorical_feature_names = \
            X_df.select_dtypes(include=self._categorical_data_types).columns
        self.n_categorical_features = len(self.categorical_feature_names)

        # If y, reformat into a pd.DataFrame and concatenate with X to produce
        # the dataframe that subclasses will analyze.            
        if isinstance(y,  self.ARRAY_LIKE):
            y_df = self._reformat(y, target=True, **kwargs)
            self.df = pd.concat([X_df, y_df], axis=1)
        else:
            self.df = X_df


        # Extract variable names, types and counts
        self.numeric_variable_names = \
            self.df.select_dtypes(include=self._numeric_data_types).columns
        self.n_numeric_variables = len(self.numeric_variable_names)        
        self.categorical_variable_names = \
        self.df.select_dtypes(include=self._categorical_data_types).columns
        self.n_categorical_variables = len(self.categorical_variable_names)

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
        self.name = get_model_name(model)
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