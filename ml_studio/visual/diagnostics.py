# --------------------------------------------------------------------------- #
#                                  Diagnostics                                #   
# --------------------------------------------------------------------------- #
"""Diagnostic Plots for single gradient descent optimizations. """
import datetime
from IPython.display import HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .basic_plots import BasicPlots
from ..utils.filemanager import save_fig, save_csv, save_gif

class Diagnostics():

    def learning_rates(self, models, directory=None, filename=None, 
                       xlim=None, ylim=None,show=True):
        """Prints learning rates by epoch for one or more models"""

        results = []
        for _, v in models.items():            
            result = pd.DataFrame({"Name": v.name, 
                                   "Iteration": np.arange(1,len(v.blackbox.learning_rates)+1),
                                   "Learning Rate": v.blackbox.learning_rates})
            results.append(result)
        results = pd.concat(results, axis=0)

        # Render Plot
        fig, ax = plt.subplots(figsize=(12,4))               
        plot = BasicPlots()
        title = "Learning Rate(s)"
        ax = plot.lineplot(x='Iteration', y='Learning Rate', z='Name', data=results, title=title, ax=ax)
        # Set x and y limits
        if xlim is not None:
            ax.set_xlim(left = xlim[0], right=xlim[1])
        if ylim is not None:
            ax.set_xlim(bottom=ylim[0], top=ylim[1])
        # Finalize, show and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = title + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)    

   
    def validation_curve(self, model, directory=None, filename=None, 
                         xlim=None, ylim=None,show=True):
        """Renders validation curve e.g. training and validation error"""

        # Extract parameters and data
        params = model.get_params()

        d = {'Iteration': np.arange(1,model.epochs+1), 
             'Learning Rates': model.learning_rates,           
             'Training Set': model.train_scores,
             'Validation Set': model.val_scores}
        df = pd.DataFrame(data=d)
        df = pd.melt(df, id_vars=['Iteration', 'Learning Rates'], var_name='Dataset', value_name='Scores')
        # Format title
        title = model.algorithm + "\n" + "Validation Curve" 

        # Initialize plot and set aesthetics        
        fig, ax = plt.subplots(figsize=(12,4))               
        sns.set(style="whitegrid", font_scale=1)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rates')
        ax.set_title(title, color='k')            

        # Plot Learning Rates
        ax = sns.lineplot(x='Iteration', y='Learning Rates', color='g', data=df, ax=ax)   

        # Plot scores
        ax2 = ax.twinx()
        ax2 = sns.lineplot(x='Iteration', y='Scores', hue='Dataset', data=df, ax=ax2)
        ax2.set_ylabel('Scores')
        # Set x and y limits
        if xlim is not None:
            ax.set_xlim(left = xlim[0], right=xlim[1])
        if ylim is not None:
            ax.set_xlim(bottom=ylim[0], top=ylim[1])

        # Show plot
        fig.tight_layout()
        if show:
            plt.show()
        # Save plot if instructed to do so
        if directory is not None:
            if filename is None:
                filename = model.algorithm + ' Validation Curve.png '
            save_fig(fig, directory, filename)       
        return fig 