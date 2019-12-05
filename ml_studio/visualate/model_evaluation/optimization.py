# =========================================================================== #
#                              OPTIMIZATION                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \optimization.py                                                      #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Sunday December 1st 2019, 4:53:02 pm                           #
# Last Modified: Sunday December 1st 2019, 5:02:32 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Model tuning and selection.""" 
#%%
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as py
from plotly.subplots import make_subplots
from progress.bar import ChargingBar
from sklearn.model_selection import KFold

from ml_studio.model_evaluation.optimization import KFoldCV
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import RidgeRegression
from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.supervised_learning.regression import ElasticNetRegression

from ml_studio.visualate.base import ModelVisualator
from ml_studio.utils.model import get_model_name
from ml_studio.utils.data_manager import sampler      

# --------------------------------------------------------------------------- #
#                            TRAINING CURVE                                   #
# --------------------------------------------------------------------------- #
class TrainingCurve(ModelVisualator):
    """Training Curve.

    This visualization reveals performance over the training process. Cost
    and scores are obtained for the training set and optionally, the validation 
    set. Monitoring training performance on both training and validation
    sets will illuminate overfitting and loss of generalization performance. 

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator. 
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, model, refit=False, title=None, hist=False, 
                 train_color='#0272a2', test_color='#9fc377', 
                 line_color='darkgray', train_alpha=0.75, 
                 test_alpha=0.75, **kwargs):    
        super(TrainingCurve, self).__init__(model, **kwargs)                        
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
        super(TrainingCurve, self).fit(X,y)        
        # Format plot title
        if self.title is None:
            self.title = "Training Curve : " + self.model.name                
        

    def show(self, report='loss', path=None, **kwargs):        
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


        # Obtain the plotting data 

        if report == 'loss':

            stub = "Cost"
            metric = "Evaluation Metric: " + self.model.cost_function.name

            # Obtain data        
            train_score = self.model.history.epoch_log['train_cost']
            val_score = self.model.history.epoch_log['val_cost']          

        else:

            stub = "Score"
            metric = "Evaluation Metric: " + self.model.metric.name

            train_score = self.model.history.epoch_log['train_score']
            val_score = self.model.history.epoch_log['val_score']
        
        total_epochs = self.model.history.total_epochs

            
#%%     # Create main traces
        trace1 = go.Scatter(x=np.arange(len(train_score)),
                       y=train_score,
                       name="Train " + stub,
                       mode="lines",
                       marker=dict(color=self.train_color),
                       showlegend=True,
                       opacity=self.train_alpha
                       )
        trace2 = go.Scatter(x=[np.argmin(train_score)],
                       y=[min(train_score)],
                       name="Minimum Train " + stub,
                       mode="markers",
                       marker=dict(color=self.train_color),
                       showlegend=False,
                       opacity=self.train_alpha
                       )
        data = [trace1, trace2]

        if self.model.val_size > 0:

            trace3= go.Scatter(x=np.arange(len(val_score)),
                        y=val_score,
                        name="Validation " + stub,
                        mode="lines",
                        marker=dict(color=self.test_color),
                        showlegend=True,
                        opacity=self.test_alpha
                        )   
            trace4 = go.Scatter(x=[np.argmin(val_score)],
                        y=[min(val_score)],
                        name="Minimum Validation " + stub,
                        mode="markers",
                        marker=dict(color=self.test_color),
                        showlegend=False,
                        opacity=self.test_alpha
                        )           
            data.append(trace3)
            data.append(trace4)


        # Designate Layout
        layout = go.Layout(title=dict(text=self.title, x=0.5, y=0.9, 
                        xanchor='center', yanchor='top'),
                        height=self.height,
                        width=self.width,
                        xaxis_title="Epoch",
                        yaxis_title=stub,
                        showlegend=True,
                        font=dict(family='Open Sans'),
                        legend=dict(x=.25, y=1.1, orientation='h'),
                        template=self.template)        

        
        fig = go.Figure(data=data, layout=layout)                        


        # Metrics Annotation
        fig.add_annotation(
                go.layout.Annotation(
                    name="Metric",
                    text=metric,                        
                    font=dict(color="black", size=14),
                    x=total_epochs/2,
                    y=np.max(train_score)*1.12,
                    showarrow=False
                )
        )         

        fig.show()

  

# %%
# --------------------------------------------------------------------------- #
#                            LEARNING CURVE                                   #
# --------------------------------------------------------------------------- #
class LearningCurve(ModelVisualator):
    """Learning Curve.

    All models are wrong, but some are useful. And the degree to which they 
    are wrong can be decomposed into two components: a bias component and a
    variance component. The bias term is the squared difference between
    the true mean and the expected estimate. The expectation averages the
    randomness in the data. The variance component is the average of a
    variance. Machine learning practitioners are usually in a state of
    compromise between bias and variance.  If you have high bias, you 
    are wrong.  If you have high variance, well...a broken clock is 
    right twice a day. Variance captures the degree to which the results
    from the model vary with new data.

    The learning curve is a tool for dealing with the bias and variance
    trade-off in machine learning.  If you plot your model with increasing
    complexity or increasing amounts of data on the horizontal axis and
    plot prediction accuracy or error on the vertical axis, we will notice
    that as the model increases in complexity or one acquires more data
    variance tends to increase and the bias tends to decrease. The opposite
    behavior occurs when we decrease model complexity (or reduce the data).

    This plot produces a learning curve vis-a-vis data. Evaluating model
    performance on a separate validation set reveals generalization loss
    and the degree to which the bias variance trade-off is out of balance.
    Provide the number and sizes of the data, and the application will
    evaluate performance using GridSearchCV.

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator.

    sizes : int or array-like
        The number of observations to include in our training set for each
        trial. If it is an integer, i, then the training set for the first
        trial will have i observations. Each subsequent trial will 
        increase the dataset by this amount   

    splits : int. Default = 5
        The number of splits for k-fold cross-validation. 
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, model, sizes, splits=5,  
                 title=None, **kwargs):    
        super(LearningCurve, self).__init__(model=model, 
                                            refit=False, 
                                            **kwargs)
        
        self.sizes= sizes
        self.splits = splits        
        self.cv = None


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
        super(LearningCurve, self).fit(X,y)        

        # Format plot title
        if self.title is None:
            self.title = "Learning Curve : " + self.model.name                

        self.cv = KFoldCV(model=self.model, sizes=self.sizes, k=self.splits)
        self.cv.fit(X, y)

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

        # Extract mean scores and standard deviations from the list of cv_results                 
        train_scores = self.cv.cv_results_['mean_train_scores']
        train_scores_std = self.cv.cv_results_['std_train_scores']
        train_scores_lower_ci = train_scores - 1.96 * (train_scores_std/np.sqrt(self.sizes))
        train_scores_upper_ci = train_scores + 1.96 * (train_scores_std/np.sqrt(self.sizes))        

        test_scores = self.cv.cv_results_['mean_test_scores']
        test_scores_std = self.cv.cv_results_['std_test_scores']
        test_scores_lower_ci = test_scores - 1.96 * (test_scores_std/np.sqrt(self.sizes))
        test_scores_upper_ci = test_scores + 1.96 * (test_scores_std/np.sqrt(self.sizes))       

        # Training set train score upper ci
        trace1 = go.Scatter(x=self.sizes,
                       y=train_scores_upper_ci,
                       name="Train Scores Upper Bound",
                       mode="lines",
                       line=dict(width=0.5),
                       marker=dict(color=self.train_color),
                       showlegend=False,
                       opacity=self.train_alpha
                       )
        # Training set train score lower ci
        trace2 = go.Scatter(x=self.sizes,
                       y=train_scores_lower_ci,
                       name="Train Score 95% CI",
                       mode="lines",
                       fill="tonexty",
                       line=dict(width=0.5),
                       marker=dict(color=self.train_color),
                       showlegend=True,
                       opacity=self.train_alpha
                       )
        # Training Set Score Line
        trace3 = go.Scatter(x=self.sizes,
                       y=train_scores,
                       name="Train Scores",
                       mode="lines+markers",
                       marker=dict(color=self.train_color),
                       showlegend=True,
                       opacity=self.train_alpha
                       )     

        # Test set test score upper ci
        trace4 = go.Scatter(x=self.sizes,
                       y=test_scores_upper_ci,
                       name="Test Scores Upper Bound",
                       mode="lines",
                       line=dict(width=0.5),
                       marker=dict(color=self.test_color),
                       showlegend=False,
                       opacity=self.test_alpha
                       )
        # Test set test score lower ci
        trace5 = go.Scatter(x=self.sizes,
                       y=test_scores_lower_ci,
                       name="Test Score 95% CI",
                       mode="lines",
                       fill="tonexty",
                       line=dict(width=0.5),
                       marker=dict(color=self.test_color),
                       showlegend=True,
                       opacity=self.test_alpha
                       )
        # Test Set Score Line
        trace6 = go.Scatter(x=self.sizes,
                       y=test_scores,
                       name="Test Scores",
                       mode="lines+markers",
                       marker=dict(color=self.test_color),
                       showlegend=True,
                       opacity=self.test_alpha
                       )                               

        data = [trace1, trace2, trace3, trace4, trace5, trace6]
        
        # Designate Layout
        layout = go.Layout(
                        title=dict(text=self.title), 
                        height=self.height,
                        width=self.width,
                        xaxis_title="Training Set Size",
                        yaxis_title=self.model.scorer.label,
                        showlegend=True,
                        legend=dict(x=.1, y=1.15, orientation='h'),
                        template=self.template)
        
        # Create and show figure
        self.fig = go.Figure(data=data, layout=layout)                        
        self.fig.show()

# --------------------------------------------------------------------------- #
#                            SCALABILITY CURVE                                #
# --------------------------------------------------------------------------- #
class ScalabilityCurve(ModelVisualator):
    """Scalability Curve.

    Scalability curve shows the fit times vis-a-vis training set sizes. For
    each training set, a K-fold cross validation is performed to obtain 
    average fit times and scores. Since the model stops training when 
    performance has not improved in a predesignated number of epochs, the 
    epochs for each training set will vary. Hence, two fit measures are 
    reported against the number of observations in the training set: 
    total fit time and normalized fit time. Total fit time is the mean fit 
    of the K-Fold cross validations for the training set.  The normalized
    fit time is the fit time normalized by the number of epochs.  

    Normalized fit times are expected to monotonically increase with 
    increasing training set sizes.  Total fit time, given empircal 
    results, may diminish with large training set sizes. This is likely
    a consequence of the improvement in the gradient estimates that 
    occur with larger datasets.

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator.

    sizes : int or array-like
        The number of observations to include in our training set for each
        trial. If it is an integer, i, then the training set for the first
        trial will have i observations. Each subsequent trial will 
        increase the dataset by this amount   

    splits : int. Default = 5
        The number of splits for k-fold cross-validation. 
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, model, sizes, splits=5,  
                 title=None, **kwargs):    
        super(ScalabilityCurve, self).__init__(model=model, 
                                            refit=False, 
                                            **kwargs)
        
        self.sizes= sizes
        self.splits = splits        
        self.cv = None


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
        super(ScalabilityCurve, self).fit(X,y)        

        # Format plot title
        if self.title is None:
            self.title = "Scalability Curve : " + self.model.name                

        self.cv = KFoldCV(model=self.model, sizes=self.sizes, k=self.splits)
        self.cv.fit(X, y)

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

        # Extract mean scores and standard deviations from the list of cv_results
        mean_epochs = self.cv.cv_results_['mean_epochs']                 
        fit_times = self.cv.cv_results_['mean_fit_times']
        fit_times_std = self.cv.cv_results_['std_fit_times']
        fit_times_lower_ci = fit_times - 1.96 * (fit_times_std/np.sqrt(mean_epochs))
        fit_times_upper_ci = fit_times + 1.96 * (fit_times_std/np.sqrt(mean_epochs))        

        fit_times_norm = self.cv.cv_results_['mean_fit_times_norm']
        fit_times_norm_std = self.cv.cv_results_['std_fit_times_norm']
        fit_times_norm_lower_ci = fit_times_norm - 1.96 * (fit_times_norm_std/np.sqrt(mean_epochs))
        fit_times_norm_upper_ci = fit_times_norm + 1.96 * (fit_times_norm_std/np.sqrt(mean_epochs))   

        # Instantiate figure object
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        # --------------------  NORMALIZED FIT VALUES ----------------------- #
        # Training set fit times upper ci
        self.fig.add_trace(go.Scatter(x=self.sizes,
                       y=fit_times_norm_upper_ci,
                       name="Fit Times Norm Upper Bound",
                       mode="lines",
                       line=dict(width=0.5),
                       marker=dict(color=self.train_color),
                       showlegend=False,
                       opacity=self.train_alpha
                       ),
                       secondary_y=False
        )
        # Training set fit times lower ci
        self.fig.add_trace(go.Scatter(x=self.sizes,
                       y=fit_times_norm_lower_ci,
                       name="Normalized Fit Times 95% CI",
                       mode="lines",
                       fill="tonexty",
                       line=dict(width=0.5),
                       marker=dict(color=self.train_color),
                       showlegend=True,
                       opacity=self.train_alpha
                       ),
                       secondary_y=False
        )
        # Training Set Fit Times
        self.fig.add_trace(go.Scatter(x=self.sizes,
                       y=fit_times_norm,
                       name="Normalized Fit Times",
                       mode="lines+markers",
                       marker=dict(color=self.train_color),
                       showlegend=True,
                       opacity=self.train_alpha
                       ),
                       secondary_y=False
        )     

        # --------------------  TOTAL FIT VALUES ----------------------- #
        # Training set fit times upper ci
        self.fig.add_trace(go.Scatter(x=self.sizes,
                       y=fit_times_upper_ci,
                       name="Total Fit Times Upper Bound",
                       mode="lines",
                       line=dict(width=0.5),
                       marker=dict(color=self.test_color),
                       showlegend=False,
                       opacity=self.test_alpha
                       ),
                       secondary_y=True
        )
        # Training set fit times lower ci
        self.fig.add_trace(go.Scatter(x=self.sizes,
                       y=fit_times_lower_ci,
                       name="Total Fit Times 95% CI",
                       mode="lines",
                       fill="tonexty",
                       line=dict(width=0.5),
                       marker=dict(color=self.test_color),
                       showlegend=True,
                       opacity=self.test_alpha
                       ),
                       secondary_y=True
        )
        # Training Set Fit Times
        self.fig.add_trace(go.Scatter(x=self.sizes,
                       y=fit_times,
                       name="Total Fit Times",
                       mode="lines+markers",
                       marker=dict(color=self.test_color),
                       showlegend=True,
                       opacity=self.test_alpha
                       ),
                       secondary_y=True
        )     
       
        # Designate Layout
        self.fig.update_layout(go.Layout(title=dict(text=self.title), 
                        height=self.height,
                        width=self.width,
                        showlegend=True,
                        legend=dict(x=.1, y=1.1, orientation='h'),
                        template=self.template)
        )

        # Set x-axis title
        self.fig.update_xaxes(title_text="Training Set Size")

        # Set y-axes titles
        self.fig.update_yaxes(title_text="<b>Normalized</b> Fit Times", secondary_y=False)
        self.fig.update_yaxes(title_text="<b>Total</b> Fit Times", secondary_y=True)        
        
        self.fig.show()


# --------------------------------------------------------------------------- #
#                            PRODUCTIVITY CURVE                               #
# --------------------------------------------------------------------------- #
class ProductivityCurve(ModelVisualator):
    """Productivity curve.

    The productivity curve evaluates performance against fit times. Concretely
    fit times are plotted on the horizontal axis. Scores are plotted on 
    the vertical axis.

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator.

    sizes : int or array-like
        The number of observations to include in our training set for each
        trial. If it is an integer, i, then the training set for the first
        trial will have i observations. Each subsequent trial will 
        increase the dataset by this amount   

    splits : int. Default = 5
        The number of splits for k-fold cross-validation. 
    
    kwargs : dict
        see documentation for PlotKwargs class.

    """

    def __init__(self, model, sizes, splits=5,  
                 title=None, **kwargs):    
        super(ProductivityCurve, self).__init__(model=model, 
                                            refit=False, 
                                            **kwargs)
        
        self.sizes= sizes
        self.splits = splits        
        self.cv = None


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
        super(ProductivityCurve, self).fit(X,y)        

        # Format plot title
        if self.title is None:
            self.title = "Productivity Curve : " + self.model.name                

        self.cv = KFoldCV(model=self.model, sizes=self.sizes, k=self.splits)
        self.cv.fit(X, y)

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

        # Extract mean scores and standard deviations from the list of cv_results        
        fit_times = np.array(self.cv.cv_results_['mean_fit_times'])

        train_scores = np.array(self.cv.cv_results_['mean_train_scores'])
        train_scores_std = np.array(self.cv.cv_results_['std_train_scores'])
        train_scores_lower_ci = train_scores - 1.96 * (train_scores_std/np.sqrt(self.sizes))
        train_scores_upper_ci = train_scores + 1.96 * (train_scores_std/np.sqrt(self.sizes))        

        test_scores = np.array(self.cv.cv_results_['mean_test_scores'])
        test_scores_std = np.array(self.cv.cv_results_['std_test_scores'])
        test_scores_lower_ci = test_scores - 1.96 * (test_scores_std/np.sqrt(self.sizes))
        test_scores_upper_ci = test_scores + 1.96 * (test_scores_std/np.sqrt(self.sizes))    

        # Get sort order
        idx = np.array(np.argsort(fit_times))

        # Instantiate figure object
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])        

        # Training set train score upper ci
        self.fig.add_trace(go.Scatter(x=fit_times[idx],
                       y=train_scores_upper_ci[idx],
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
        self.fig.add_trace(go.Scatter(x=fit_times[idx],
                       y=train_scores_lower_ci[idx],
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
        self.fig.add_trace(go.Scatter(x=fit_times[idx],
                       y=train_scores[idx],
                       name="Train Scores",
                       mode="lines+markers",
                       marker=dict(color=self.train_color),
                       showlegend=True,
                       opacity=self.train_alpha
                       ),
                       secondary_y=False
        )     

        # Test set test score upper ci
        self.fig.add_trace(go.Scatter(x=fit_times[idx],
                       y=test_scores_upper_ci[idx],
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
        self.fig.add_trace(go.Scatter(x=fit_times[idx],
                       y=test_scores_lower_ci[idx],
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
        self.fig.add_trace(go.Scatter(x=fit_times[idx],
                       y=test_scores[idx],
                       name="Test Scores",
                       mode="lines+markers",
                       marker=dict(color=self.test_color),
                       showlegend=True,
                       opacity=self.test_alpha
                       ),
                       secondary_y=False
        )

        # Data set size
        self.fig.add_trace(go.Scatter(
                       x=fit_times[idx],
                       y=np.array(self.sizes)[idx],
                       name="Training Set Observations",
                       mode="lines+markers",
                       marker=dict(color='#CE2700'),
                       showlegend=True,
                       opacity=self.test_alpha
                       ),
                       secondary_y=True
        )                                      
        
        # Designate Layout
        self.fig.update_layout(go.Layout(title=dict(text=self.title), 
                        height=self.height,
                        width=self.width,
                        showlegend=True,
                        template=self.template,                        
                        legend=dict(x=.2, y=1.19, orientation='h'))
        )
        # Set x-axis title
        self.fig.update_xaxes(title_text="Fit Time")

        # Set y-axes titles
        self.fig.update_yaxes(title_text=self.model.scorer.label, secondary_y=False)
        self.fig.update_yaxes(title_text="Epochs", secondary_y=True)  
        
        self.fig.show()