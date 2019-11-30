# =========================================================================== #
#                             REGRESSION                                      #
# =========================================================================== #
"""Plots for analyzing the performance of regression models."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings

from ml_studio.supervised_learning.training.estimator import Estimator
from ml_studio.supervised_learning.training.metrics import RegressionMetricFactory
from ml_studio.utils.misc import proper
from ml_studio.utils.file_manager import save_fig

# --------------------------------------------------------------------------- #
#                            RESIDUALS PLOT                                   #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#    Title: Yellowbrick source code
#   Author:   Rebecca Bilbro
#   Author:   Benjamin Bengfort
#    Date: 2017
#    Code version: 1.0.1
#    Availability: https://www.scikit-yb.org/en/latest/
#
# --------------------------------------------------------------------------- #
def histogram_ax(ax):
    """
    Histogram axis
    """
    divider = make_axes_locatable(ax)
    hax = divider.append_axes("right", size=1, pad=0.0, sharey=ax)
    hax.yaxis.tick_right()
    hax.grid(False, axis="x")
    return hax


def residuals(model, X, y, type='standardized', hist=True, title=None, 
              figsize=(12,6), directory=None, filename=None):
    """Plots residuals versus actual."""
    # Validate request
    if not isinstance(model, Estimator):
        raise ValueError("Model is not a valid Estimator or subclass object.")
    if not isinstance(figsize, tuple):
        raise TypeError("figsize is not a valid tuple.")  
    
    # Format title
    if type == ''
    if title is None:
        title = model.history.params.get('name') + "\n" + \
            "Residuals Plot"      

    # Compute training predictions, residuals, and R2
    X_train, y_train = model.X, model.y    
    y_pred = model.predict(X_train)
    residuals = y_train - y_pred
    
    # Compute R2
    r2 = RegressionMetricFactory()(metric='r2')(y_train, y_pred)
    
    # Compute Leverage
    leverage = (X_train * np.linalg.pinv(X_train).T).sum(1)
    
    # Compute degrees of freedom and MSE
    rank = np.linalg.matrix_rank(X_train)
    df = X.shape[0] - rank
    mse = np.dot(residuals, residuals) / df

    # Calculate standardized and studentized residuals
    standardized_residuals = residuals / np.sqrt(mse*(1-leverage))
    studentized_residuals = residuals / np.sqrt(mse)/ np.sqrt(1-leverage)
    
    # Initialize figure and axes with appropriate figsize and title
    fig, ax = _init_image(x='$\\hat{y}$', y='Studentized Residuals', figsize=figsize,
                                    title=title)            
    # Set labels
    ax.set_xlabel('$\\hat{y}$')
    ax.set_ylabel('Studentized Residuals')                                    

    # Render scatterplot of residuals vs predicted
    label = "Train $R^2 = {:0.3f}$".format(r2)    
    ax = sns.residplot(y_pred, studentized_residuals, lowess=True,
                       line_kws={'color': 'red', 'lw': 1, 'alpha':0.8})
    #ax.scatter(y_pred, studentized_residuals, label=label)
    ax.legend()
        
    # Add residuals historgram
    if hist:
        hax = histogram_ax(ax)
        hax.hist(studentized_residuals, bins=50, orientation='horizontal')
    
    # Save figure if directory is not None
    if directory is not None:
        title = title.replace('\n', ' ') + '.png'
        save_plot(fig, directory, filename, title)

    # Show plot
    fig.tight_layout()
    plt.show()                                             
        
# --------------------------------------------------------------------------- #
#                           PREDICTION ERROR PLOT                             #
# --------------------------------------------------------------------------- #
def prediction_error(model, X, y, shared_limits=True, title=None, 
                     bestfit=True, identity=True, figsize=(8,8), 
                     directory=None, filename=None):
    """Plots residuals versus actual."""
    # Validate request
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have incompatible shapes. X.shape = %s "
                         " y.shape = %s" %(str(X.shape), str(y.shape)))
    if not isinstance(model, Estimator):
        raise ValueError("Model is not a valid Estimator or subclass object.")
    if not isinstance(figsize, tuple):
        raise TypeError("figsize is not a valid tuple.")  
    # Format title
    if title is None:
        title = model.history.params.get('name') + "\n" + \
            "Prediction Error Plot"   

    # Compute predictions and and R2
    y_pred = model.predict(X)
    r2 = RegressionMetricFactory()(metric='r2')(y, y_pred)

    # Get Datapoints
    y_min = min(min(y), min(y_pred))
    y_max = max(max(y), max(y_pred))

    # Initialize figure and axes with appropriate figsize and title
    xlim = (y_min, y_max)
    ylim = (y_min, y_max)
    fig, ax = _init_image(x='$y$', y='$\\hat{y}',figsize=figsize, xlim=xlim, ylim=ylim, 
                          title=title)            
    ax.set_xlabel('y')
    ax.set_ylabel('$\\hat{y}$')    

    # Render prediction versus actual
    ax = sns.scatterplot(x=y, y=y_pred, ax=ax)

    # Render identity line
    if identity:
        x_i = [y_min,y_max]
        y_i = [y_min,y_max]
        ax = sns.lineplot(x=x_i,y=y_i, dashes=[(2,2)], legend='full', label='Identity Line')
    
    # Render best fit line
    if bestfit:        
        X = np.array(y).reshape(-1,1)
        Y = y_pred.reshape(-1,1)
        lr = LinearRegression()
        lr.fit(X,Y)        
        bias = lr.intercept_
        coef = lr.coef_
        def f(x):
            y = bias + coef * x
            return y.flatten()
        x = np.linspace(y_min, y_max,100)
        y = f(x)
        ax = sns.lineplot(x=x,y=y, dashes=True, ax=ax, legend='full', label='Best Fit Line')

    # Add R2 Score to plot
    r2_text = "Coefficient of Determination (R2): " + str(round(r2,4))
    ax.text(0.3, 0.96, r2_text, fontsize=12, transform=ax.transAxes)

    # Fix axis limits
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)
        
    # Save figure if directory is not None
    if directory is not None:
        title = title.replace('\n', ' ') + '.png'
        save_plot(fig, directory, filename, title)

    # Show plot
    fig.tight_layout()
    plt.show()                                             
        
# --------------------------------------------------------------------------- #
#                            VARIOUS ROUTINES                                 #
# --------------------------------------------------------------------------- #
def save_plot(fig, directory, filename=None, title=None):
    """Save plot with title to designated directory and filename."""
    if filename is None and title is None:
        raise ValueError("Must provide filename or title to save plot.")
    if filename is None:
        filename = title.replace('\n', ' ') + " .png"
    save_fig(fig, directory, filename)
    return directory, filename

def _init_image(x, y, figsize=(12, 4), xlim=None, ylim=None, title=None, log=False):
    """Creates and sets the axis aesthetics, labels, scale, and limits."""

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize)
    # Set aesthetics
    sns.set(style="whitegrid", font_scale=1)
    ax.set_facecolor('w')
    ax.tick_params(colors='k')
    ax.xaxis.label.set_color('k')
    ax.yaxis.label.set_color('k')
    ax.set_title(title, color='k', fontsize=16)
    # Set labels
    ax.set_xlabel(proper(x))
    ax.set_ylabel(proper(y))
    # Change to log scale if requested
    if log:
        ax.set_xscale('log')
    # Set x and y axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return fig, ax

        
         
