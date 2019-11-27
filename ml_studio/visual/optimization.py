# =========================================================================== #
#                              OPTIMIZATION                                   #
# =========================================================================== #
"""Plots for analyzing the optimization process."""
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from ml_studio.supervised_learning.training.estimator import Estimator
from ml_studio.supervised_learning.training.metrics import RegressionMetricFactory
from ml_studio.utils.misc import proper
from ml_studio.utils.file_manager import save_fig

# --------------------------------------------------------------------------- #
#                          TRAINING LOSS PLOTS                                #
# --------------------------------------------------------------------------- #
def _plot_train_loss(model, title=None, figsize=(12,4)):
    """Plots training loss."""
    # Format title
    if title is None:
        title = model.history.params.get('name') + "\n" + \
            "Training Loss"     
    # Extract training loss                    
    d = {'Epoch': model.history.epoch_log['epoch'],
            'Cost': model.history.epoch_log['train_cost']}
    df = pd.DataFrame(data=d)
    # Extract row with minimum cost for scatterplot    
    min_cost = df.loc[df.Cost.idxmin()]
    min_cost = pd.DataFrame({"Epoch": [min_cost['Epoch']],
                             "Cost": [min_cost['Cost']]})
    # Initialize figure and axes with appropriate figsize and title
    fig, ax = _init_image(x='Epoch', y='Cost', figsize=figsize,
                                    title=title)
    # Render cost lineplot
    ax = sns.lineplot(x='Epoch', y='Cost', data=df, ax=ax)
    # Render scatterplot showing minimum cost points
    ax = sns.scatterplot(x=min_cost['Epoch'], y=min_cost['Cost'], ax=ax)
    return fig, ax, title    


def _plot_train_val_loss(model, title=None, figsize=(12,4)):
    """Plots training and validation loss on single plot."""
    # Format title
    if title is None:
        title = model.history.params.get('name') + "\n" + \
            "Training and Validation Loss" 
    # Extract training and validation loss                    
    d = {'Epoch': model.history.epoch_log['epoch'],
            'Training': model.history.epoch_log['train_cost'],
            'Validation': model.history.epoch_log.get('val_cost')}
    df = pd.DataFrame(data=d)
    df = pd.melt(df, id_vars='Epoch', value_vars=['Training',
                                                'Validation'],
                var_name=['Dataset'], value_name='Cost')  
    # Extract row with minimum cost by dataset for scatterplot
    min_cost = df.loc[df.groupby('Dataset').Cost.idxmin()]
    # Initialize figure and axes with appropriate figsize and title
    fig, ax = _init_image(x='Epoch', y='Cost', figsize=figsize,
                                    title=title)
    # Render cost lineplot
    ax = sns.lineplot(x='Epoch', y='Cost', hue='Dataset', data=df, 
                        legend='full', ax=ax)
    # Render scatterplot showing minimum cost points
    ax = sns.scatterplot(x='Epoch', y='Cost', hue='Dataset', 
                            data=min_cost, legend=False, ax=ax)
                                          
    return fig, ax, title
    

def plot_loss(model, title=None, figsize=(12,4), directory=None, filename=None):
    """Plots training loss (and optionally validation loss) by epoch."""
    # Validate request
    if not isinstance(model, Estimator):
        raise ValueError("Model is not a valid Estimator or subclass object.")
    if not isinstance(figsize, tuple):
        raise TypeError("figsize is not a valid tuple.")    

    # If val loss is on the log, plot both training and validation loss
    if 'val_cost' in model.history.epoch_log:
        fig, _, title = _plot_train_val_loss(model, title=title, 
                                            figsize=figsize)
    else:
        fig, _, title = _plot_train_loss(model, title=title, 
                                        figsize=figsize)

    # Save figure if directory is not None
    if directory is not None:
        title = title.replace('\n', ' ') + '.png'
        save_plot(fig, directory, filename, title)

    # Show plot
    fig.tight_layout()
    plt.show()                                             
        



# --------------------------------------------------------------------------- #
#                          TRAINING SCORE PLOTS                               #
# --------------------------------------------------------------------------- #
def _plot_train_score(model, title=None, figsize=(12,4)):
    """Plots training score."""
    if title is None:
        title = model.history.params.get('name') + "\n" + \
            "Training Scores" +\
            '\n' + proper(model.metric)    
    # Extract training score                    
    d = {'Epoch': model.history.epoch_log['epoch'],
            'Score': model.history.epoch_log['train_score']}
    df = pd.DataFrame(data=d)
    # Extract row with best score for scatterplot    
    if RegressionMetricFactory()(model.metric).mode == 'max': 
        best_score = df.loc[df.Score.idxmax()]
    else:
        best_score = df.loc[df.Score.idxmin()]    
    best_score = pd.DataFrame({"Epoch": [best_score['Epoch']],
                             "Score": [best_score['Score']]})        
    # Initialize figure and axes with appropriate figsize and title
    fig, ax = _init_image(x='Epoch', y='Score', figsize=figsize,
                                    title=title)
    # Render score lineplot
    ax = sns.lineplot(x='Epoch', y='Score', data=df, ax=ax)
    # Render scatterplot showing minimum score points
    ax = sns.scatterplot(x='Epoch', y='Score', data=best_score, ax=ax)
    return fig, ax, title    


def _plot_train_val_score(model, title=None, figsize=(12,4)):
    """Plots training and validation score on single plot."""
    # Format plot title
    if title is None:
        title = model.history.params.get('name') + "\n" + \
            "Training and Validation Scores" +\
            '\n' + proper(model.metric)    
    # Extract training and validation score                    
    d = {'Epoch': model.history.epoch_log['epoch'],
            'Training': model.history.epoch_log['train_score'],
            'Validation': model.history.epoch_log.get('val_score')}
    df = pd.DataFrame(data=d)
    df = pd.melt(df, id_vars='Epoch', value_vars=['Training',
                                                'Validation'],
                var_name=['Dataset'], value_name='Score')  
    # Extract row with best score by dataset for scatterplot
    if RegressionMetricFactory()(model.metric).mode == 'max': 
        best_score = df.loc[df.groupby('Dataset').Score.idxmax()]
    else:
        best_score = df.loc[df.groupby('Dataset').Score.idxmin()]    
    # Initialize figure and axes with appropriate figsize and title
    fig, ax = _init_image(x='Epoch', y='Score', figsize=figsize,
                                    title=title)
    # Render score lineplot
    ax = sns.lineplot(x='Epoch', y='Score', hue='Dataset', data=df, 
                        legend='full', ax=ax)
    # Render scatterplot showing minimum score points
    ax = sns.scatterplot(x='Epoch', y='Score', hue='Dataset', 
                            data=best_score, legend=False, ax=ax)
    return fig, ax, title
    

def plot_score(model, title=None, figsize=(12,4), directory=None, filename=None):
    """Plots training score (and optionally validation score) by epoch."""

    # Validate request
    if not model.metric:
        raise Exception("No metric designated for score.")
    if not isinstance(model, Estimator):
        raise ValueError("Model is not a valid Estimator or subclass object.")
    if not isinstance(figsize, tuple):
        raise TypeError("figsize is not a valid tuple.")
    
    # If val score is on the log, plot both training and validation score
    if 'val_score' in model.history.epoch_log:
        fig, _, title = _plot_train_val_score(model, title=title, 
                                            figsize=figsize)
    else:
        fig, _, title = _plot_train_score(model, title=title, 
                                        figsize=figsize)

    # Save figure if directory is not None
    if directory is not None:        
        title = title.replace('\n', ' ')
        save_plot(fig, directory, filename, title)

    # Show plot
    fig.tight_layout()
    plt.show()                                                     


# --------------------------------------------------------------------------- #
#                            GRIDSEARCHCV PLOTS                               #
# --------------------------------------------------------------------------- #    
def gscv_line_plot(x, y, gscv, title=None, directory=None, filename=None):
    """Creates line plot from dictionary of GridSearchCV objects."""
    # Extract data from dictionary of GridSearchCV objects
    data = pd.DataFrame()
    for name, gs in gscv.items():      
        results = pd.DataFrame(data=gs.cv_results_)
        x_data = results.filter(like=x, axis=1).values.flatten()
        y_data = results.filter(like=y, axis=1).values.flatten()
        df = pd.DataFrame({proper(x):x_data, proper(y): y_data, 'Model':name})
        data = pd.concat([data,df], axis=0)
    if title is None:
        title = 'Gridsearch Cross-validation Plot'
    # Initialize and render plot
    fig, ax = _init_image(x, y)
    ax = sns.lineplot(x=proper(x), y=proper(y), hue='Model', data=data, ax=ax)
    # Save figure if requested
    if directory is not None:
        title = title.replace('\n', ' ')
        save_plot(fig, directory=directory, filename=filename, title=title)    
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
    ax.set_title(title, color='k')
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
        

        
         
