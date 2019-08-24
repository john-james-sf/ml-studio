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
        self.log = pd.DataFrame()
        self.inventory = {}

    def show_plot(self, pid=None, fig=None):
        """Renders a plot given the plot id, or figure object"""
        if isinstance(pid, str):
            try:
                fig = self.log[pid].fig
            except KeyError:
                print("pid is not a valid plot id")
        try:
            fig.tight_layout()
            plt.show()
        except ValueError:
            print("Unable to render plot")

    def get_plot(self, pid):
        """Returns plot object associated with pid."""
        return self.inventory[pid]

    def show_log(self):
        """Prints the log including plot metadata to stdout"""
        print(self.log)

    def _log_plot(self, pid, model, title, fig):
        """Stores a plot object along with its metadata and model."""
        # Create plot object w/ metadata, model and fig, then add to inventory 
        metadata = {'pid': pid, 'title': title.replace('\n', " "),
                    'directory': "", 'filename': "", 
                    'datetime': datetime.datetime.now()}
        plot = {'metadata': metadata, 'model': model, 'fig': fig}
        self.inventory[pid] = plot
        # Store metadata in log dataframe and add to log
        df = pd.DataFrame(data = metadata, index=['0'])                
        self.log = pd.concat([self.log, df], axis=0)
        return plot

    def save_plot(self, plot, directory, filename=None):
        """Save plot with title to designated directory and filename."""
        title = plot.metadata.get('title')
        fig = plot.get('fig')
        if filename is None:
            filename = title + '.png'
        save_fig(fig, directory, filename)
        return directory, filename

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
    def _plot_train_loss(self, pid, model, title=None, figsize=(12,4)):
        """Plots training loss."""
        # Extract training loss                    
        d = {'Epoch': model.history.epoch_log['epoch'],
             'Training': model.history.epoch_log['train_cost']}
        df = pd.DataFrame(data=d)
        # Extract row with minimum cost by dataset for scatterplot
        min_cost = df.loc[df.Cost.idxmin()]
        # Extract learning rate data for plotting along secondary y-axis
        lr = {'Epoch': model.history.epoch_log['epoch'],
              'Learning Rate': model.history.epoch_log['learning_rate']}
        lr = pd.DataFrame(data=lr)
        # Initialize figure and axes with appropriate figsize and title
        fig, ax = self._init_image(x='Epoch', y='Cost', figsize=figsize,
                                       title=title)
        ax2 = ax.twinx()
        # Render cost lineplot
        ax = sns.lineplot(x='Epoch', y='Cost', data=df, ax=ax)
        # Render scatterplot showing minimum cost points
        ax = sns.scatterplot(x='Epoch', y='Cost', data=min_cost, ax=ax)
        # Show learning rate along secondary y-axis
        ax2 = sns.lineplot(x='Epoch', y='Learning Rate', data=lr, ax=ax2) 
        # Add pid to footnote of plot
        ax.text(0.8, 0.1, pid)
        # Show plot
        fig.tight_layout()
        plt.show()        
        return fig, ax    

    def _plot_train_val_loss(self, pid, model, title=None, figsize=(12,4)):
        """Plots training and validation loss on single plot."""
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
        # Extract learning rate data for plotting along secondary y-axis
        lr = {'Epoch': model.history.epoch_log['epoch'],
              'Learning Rate': model.history.epoch_log['learning_rate']}
        lr = pd.DataFrame(data=lr)
        # Initialize figure and axes with appropriate figsize and title
        fig, ax = self._init_image(x='Epoch', y='Cost', figsize=figsize,
                                       title=title)
        ax2 = ax.twinx()
        # Render cost lineplot
        ax = sns.lineplot(x='Epoch', y='Cost', hue='Dataset', data=df, 
                          legend='full', ax=ax)
        # Render scatterplot showing minimum cost points
        ax = sns.scatterplot(x='Epoch', y='Cost', hue='Dataset', 
                                data=min_cost, legend=False, ax=ax)
        # Show learning rate along secondary y-axis
        ax2 = sns.lineplot(x='Epoch', y='Learning Rate', data=lr, ax=ax2)                                                
        # Show plot
        fig.tight_layout()
        plt.show()        
        return fig, ax
        

    def plot_loss(self, model, figsize=(12,4)):
        """Plots training loss (and optionally validation loss) by epoch."""
        # Create plot id
        pid = randomString(5)
        # Format plot title
        title = model.history.params.get('name') + "\n" + \
            "Training Plot with Learning Rate" +\
            '\n' + proper(model.history.params.get('cost')) + " Cost"
        # If val loss is on the log, plot both training and validation loss
        if 'val_cost' in model.history.epoch_log:
            fig, _ = self._plot_train_val_loss(pid, model, 
                                                title=title, figsize=figsize)
        else:
            fig, _ = self._plot_train_loss(pid, model, title=title, 
                                            figsize=figsize)
        # Create and log the plot object into inventory
        plot = self._log_plot(pid, model, title, fig)
        return plot
        

        

        
         
