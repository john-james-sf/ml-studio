# =========================================================================== #
#                                PLOTS                                        #
# =========================================================================== #
from ml_studio.utils.misc import randomString, proper
from ml_studio.supervised_learning.regression import GradientDescent
from ml_studio.operations.callbacks import History
from ml_studio.utils.filemanager import save_fig
import seaborn as sns
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# TODO: Add Classification class once finished

# --------------------------------------------------------------------------- #
#                              Plot Class                                     #
# --------------------------------------------------------------------------- #


class Plots():
    """Plot class"""

    def __init__(self):
        self.history = {}

    def _log_plot(self, algorithm, title, fig, directory, filename):
        """Stores the plot and algorithm to history."""
        # Create unique 5 character string plot id and timestamp
        pid = randomString(stringLength=5)
        timestamp = datetime.datetime.now
        # Format the plot dictionary and add to history
        plot = {'title': title, 'algorithm': algorithm, 'fig': fig,
                'directory': directory, 'filename': filename,
                'timestamp': timestamp}
        self.history[pid] = plot
        return self.history[pid]

    def list_plots(self):
        """Returns a dataframe of plot metadata"""
        plots = []
        for pid, plot in self.history.items():
            plots.append([pid, plot.title, plot.directory,
                          plot.filename, plot.timestamp])
        df = pd.DataFrame(
            plots, columns=['ID', 'Title', 'Directory', 'Filename', 'TimeStamp'])
        return df

    def show_plot(self, x):
        """Renders a plot given the plot id, or figure object"""
        if isinstance(x, str):
            try:
                fig = self.history[x].fig
            except KeyError:
                print("x is not a valid plot id")
        else:
            fig = x

        try:
            fig.tight_layout()
            plt.show()
        except ValueError:
            print("x is not a valid figure object")

    def save_plot(self, fig, directory=None, filename=None, title=None):
        """Save plot with title to designated directory and filename."""
        if directory is not None:
            if filename is None:
                filename = title.strip('\n') + '.png'
            save_fig(fig, directory, filename)
        return directory, filename

    def _distplot(self, data, x, y, z=None, title=None,
                  log=False, xlim=None, ylim=None, figsize=(12, 4)):
        """Renders a univariate distribution of observations."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.distplot(a=data[x])

        return fig, ax

    def _scatterplot(self, data, x, y, z=None, title=None,
                     log=False, xlim=None, ylim=None, figsize=(12, 4)):
        """Renders and optionally saves a scatterplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.scatterplot(x=x, y=y, hue=z, data=data, ax=ax, legend='full')

        return fig, ax

    def _barplot(self, data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None, figsize=(12, 4)):
        """Renders and optionally saves a barplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.barplot(x=x, y=y, hue=z, data=data, ax=ax)

        return fig, ax

    def _boxplot(self, data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None, figsize=(12, 4)):
        """Renders and optionally saves a boxplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.boxplot(x=x, y=y, hue=z, data=data, ax=ax)

        return fig, ax

    def _lineplot(self, data, x, y, z=None, title=None,
                  log=False, xlim=None, ylim=None, figsize=(12, 4)):
        """Renders and optionally saves a lineplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.lineplot(x=x, y=y, hue=z, data=data, legend='full', ax=ax)

        return fig, ax

    def _catplot(self, data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None, directory=None,
                 filename=None, figsize=(12, 4)):
        """Renders and optionally saves a catplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.catplot(x=x, y=y, hue=z, kind='bar', data=data, ax=ax)

        return fig, ax

    def _init_image(self, x, y, figsize=(12, 4), xlim=None, ylim=None, title=None, log=False):
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

# --------------------------------------------------------------------------- #
#                            TRAINING PLOTS                                   #
# --------------------------------------------------------------------------- #


class TrainingPlots(Plots):

    def _validate(self, algorithm, directory, filename,
                  xlim, ylim, figsize, show):
        """Validates parameters for learning rate plots."""
        if not isinstance(algorithm, (GradientDescent)):
            raise TypeError("model is not a valid model object")
        if directory is not None:
            if not isinstance(directory, str):
                raise TypeError("directory must be a string object")
        if filename is not None:
            if not isinstance(filename, str):
                raise TypeError("filename must be a string object")
        if directory is None and filename is not None:
            print(filename)
            raise ValueError("directory not provided")
        if xlim is not None:
            if not isinstance(xlim, (int, tuple)):
                raise TypeError(
                    "xlim must be an integer or a tuple of integers")
        if ylim is not None:
            if not isinstance(ylim, (int, tuple)):
                raise TypeError(
                    "ylim must be an integer or a tuple of integers")
        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple")
        if not isinstance(show, bool):
            raise TypeError("show must be a boolean object")

    def cost_plot(self, algorithm, directory=None, filename=None,
                  xlim=None, ylim=None, figsize=(12, 4), show=True):
        """Renders optimization learning curves for training and validation"""
        # Validate parameters
        self._validate(algorithm, directory, filename,
                       xlim, ylim, figsize, show)

        # Extract data to be plotted in wide format
        d = {'Epoch': algorithm.history.epoch_log['epoch'],
             'Training Cost': algorithm.history.epoch_log['train_cost'],
             'Validation Cost': algorithm.history.epoch_log.get('val_cost')}
        # Create data in dataframe, then convert wide to long format.
        df = pd.DataFrame(data=d)
        df = pd.melt(df, id_vars='Epoch', value_vars=['Training Cost',
                                                      'Validation Cost'],
                     var_name=['Dataset'], value_name='Cost')
        # Format title
        title = algorithm.history.params.get('name') + "\n" + "Cost Plot" + \
            '\n' + proper(algorithm.history.params.get('cost')) + " Cost"

        # Render (and save plot if necessary)
        fig, _ = self._lineplot(data=df, x=df.columns[0], y='Cost',
                                z='Dataset', title=title, log=False,
                                xlim=xlim, ylim=ylim, figsize=figsize)
        # Show plot
        fig.tight_layout()
        if show:
            plt.show()

        # Save plot if directory is not null
        directory, filename = self.save_plot(fig=fig, directory=directory,
                                             filename=filename, title=title)

        # Log the plot
        plot=self._log_plot(algorithm, title, fig, directory, filename)

        return plot

    def score_plot(self, algorithm, directory = None, filename = None,
                   xlim = None, ylim = None, figsize = (12, 4), show = True):
        """Renders optimization learning curves for training and validation"""
        # Validate parameters
        self._validate(algorithm, directory, filename,
                       xlim, ylim, figsize, show)

        # Extract data to be plotted in wide format
        d={'Epoch': algorithm.history.epoch_log['epoch'],
             'Training Score': algorithm.history.epoch_log['train_score'],
             'Validation Score': algorithm.history.epoch_log.get('val_score')}
        # Create data in dataframe, then convert wide to long format.
        df = pd.DataFrame(data=d)
        df = pd.melt(df, id_vars='Epoch', value_vars=['Training Score',
                                                      'Validation Score'],
                     var_name=['Dataset'], value_name='Score')
        # Format title
        title = algorithm.history.params.get('name') + "\n" + "Performance Plot" + \
            '\n' + proper(algorithm.history.params.get('metric')) + " Score"

        # Render (and save plot if necessary)
        fig, _ = self._lineplot(data=df, x=df.columns[0], y='Score',
                                z='Dataset', title=title, log=False,
                                xlim=xlim, ylim=ylim, figsize=figsize)
        # Show plot
        fig.tight_layout()
        if show:
            plt.show()
            
        # Save plot if directory is not null
        directory, filename = self.save_plot(fig=fig, directory=directory,
                                             filename=filename, title=title)        
        # Log the plot
        plot = self._log_plot(algorithm, title, fig, directory, filename)

        return plot
