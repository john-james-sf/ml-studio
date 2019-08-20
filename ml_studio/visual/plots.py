# =========================================================================== #
#                                PLOTS                                        #
# =========================================================================== #
from ml_studio.utils.filemanager import save_fig
import seaborn as sns
import pandas as pd
import numpy as np
"""Class containing basic plotting functions"""
# --------------------------------------------------------------------------- #
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# --------------------------------------------------------------------------- #
#                            Learning Curves                                  #
# --------------------------------------------------------------------------- #


class LearningCurves():

    def __init__(self):
        pass

    def optimization(self, history, directory=None, filename=None,
                     xlim=None, ylim=None, figsize=(12, 4), show=True):
        """Renders optimization learning curves for training and validation"""

        # Extract parameters and data
        d = {'Epoch': history.epochs,
             'Training Cost': history.epoch_log.get('train_cost'),
             'Validation Cost': history.epoch_log.get('val_cost')}
        df = pd.DataFrame(data=d)
        df = pd.melt(df, id_vars='Epoch', value_vars=['Training Cost', 'Validation Cost'],
                     var_name=['Dataset'], value_name='Cost')
        # Format title
        title = history.params.get('name') + "\n" + "Optimization Learning Curves" + \
            '\n' + proper(history.params.get('cost')) + " Cost"

        # Render and save plot (if necessary)
        plot = BasicPlots()
        fig, _ = plot.lineplot(data=df, x=df.columns[0], y='Cost',
                               z='Dataset', title=title, log=False, xlim=xlim, ylim=ylim,
                               directory=directory, filename=filename, figsize=figsize)
        # Show plot
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def performance(self, history, directory=None, filename=None,
                    xlim=None, ylim=None, figsize=(12, 4), show=True):
        """Renders performance learning curves for training and validation"""

        # Extract parameters and data
        d = {'Epoch': history.epochs,
             'Training Scores': history.epoch_log.get('train_score'),
             'Validation Scores': history.epoch_log.get('val_score')}
        df = pd.DataFrame(data=d)
        df = pd.melt(df, id_vars='Epoch', value_vars=['Training Scores', 'Validation Scores'],
                     var_name=['Dataset'], value_name=proper(history.params.get('metric')))
        # Format title
        title = history.params.get('name') + "\n" + "Performance Learning Curves" + \
            '\n' + proper(history.params.get('metric')) + "Score"

        # Render and save plot (if necessary)
        plot = BasicPlots()
        fig, _ = plot.lineplot(data=df, x=df.columns[0], y=history.params.get('metric'),
                               z='Dataset', title=title, log=False, xlim=xlim, ylim=ylim,
                               directory=directory, filename=filename, figsize=figsize)
        # Show plot
        fig.tight_layout()
        if show:
            plt.show()
        return fig


# --------------------------------------------------------------------------- #
#                         Formatting Functions                                #
# --------------------------------------------------------------------------- #
def proper(s):
    s = s.replace("-", " ").capitalize()
    s = s.replace("_", " ").capitalize()
    return s
# --------------------------------------------------------------------------- #
#                             Basic Plots                                     #
# --------------------------------------------------------------------------- #


class BasicPlots:
    """Standard interfaces for basic plotting methods"""

    def __init__(self):
        pass

    def distplot(self, data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None, directory=None,
                 filename=None, figsize=(12, 4)):
        """Renders a univariate distribution of observations."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.distplot(a=data[x])
        # Save image if path information is provided
        self._save_plot(fig=fig, directory=directory,
                        filename=filename, title=title)
        return fig, ax

    def scatterplot(self, data, x, y, z=None, title=None,
                    log=False, xlim=None, ylim=None, directory=None,
                    filename=None, figsize=(12, 4)):
        """Renders and optionally saves a scatterplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.scatterplot(x=x, y=y, hue=z, data=data, ax=ax, legend='full')
        # Save image if path information is provided
        self._save_plot(fig=fig, directory=directory,
                        filename=filename, title=title)
        return fig, ax

    def barplot(self, data, x, y, z=None, title=None,
                log=False, xlim=None, ylim=None, directory=None,
                filename=None, figsize=(12, 4)):
        """Renders and optionally saves a barplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.barplot(x=x, y=y, hue=z, data=data, ax=ax)
        # Save image if path information is provided
        self._save_plot(fig=fig, directory=directory,
                        filename=filename, title=title)
        return fig, ax

    def boxplot(self, data, x, y, z=None, title=None,
                log=False, xlim=None, ylim=None, directory=None,
                filename=None, figsize=(12, 4)):
        """Renders and optionally saves a boxplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.boxplot(x=x, y=y, hue=z, data=data, ax=ax)
        # Save image if path information is provided
        self._save_plot(fig=fig, directory=directory,
                        filename=filename, title=title)
        return fig, ax

    def lineplot(self, data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None, directory=None,
                 filename=None, figsize=(12, 4)):
        """Renders and optionally saves a lineplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.lineplot(x=x, y=y, hue=z, data=data, legend='full', ax=ax)
        # Save image if path information is provided
        self._save_plot(fig=fig, directory=directory,
                        filename=filename, title=title)
        return fig, ax

    def catplot(self, data, x, y, z=None, title=None,
                log=False, xlim=None, ylim=None, directory=None,
                filename=None, figsize=(12, 4)):
        """Renders and optionally saves a catplot."""

        # Initialize axis aesthetics, labels, title, scale and limits
        fig, ax = self._init_image(x, y, figsize, xlim, ylim, log)
        # Call seaborn method for plot
        ax = sns.catplot(x=x, y=y, hue=z, kind='bar', data=data, ax=ax)
        # Save image if path information is provided
        self._save_plot(fig=fig, directory=directory,
                        filename=filename, title=title)
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
            if len(xlim) == 1:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim(left=xlim[0], right=xlim[1])
        if ylim:
            if len(ylim) == 1:
                ax.set_ylim(ylim)
            else:
                ax.set_xlim(bottom=ylim[0], top=ylim[1])
        return fig, ax

    def _save_plot(self, fig, directory=None, filename=None, title=None):
        """Save plot with title to designated directory and filename."""
        if directory is not None:
            if filename is None:
                filename = title.strip('\n') + '.png'
            save_fig(fig, directory, filename)
