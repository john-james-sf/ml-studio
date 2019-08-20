# =========================================================================== #
#                                PLOTS                                        #
# =========================================================================== #
"""Class containing basic plotting functions"""
# --------------------------------------------------------------------------- #
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import seaborn as sns
from ml_studio.utils.filemanager import save_fig

# --------------------------------------------------------------------------- #
#                            Learning Curves                                  #
# --------------------------------------------------------------------------- #
class LearningCurves():

    def __init__(self):
            pass  

    def optimization(self, history, directory=None, filename=None, 
                         xlim=None, ylim=None,show=True):
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


        # Initialize plot and set aesthetics        
        fig, ax = plt.subplots(figsize=(12,4))               
        sns.set(style="whitegrid", font_scale=1)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(proper(history.params.get('cost')) + "Cost")
        ax.set_title(title, color='k')            

        # Plot Cost
        ax = sns.lineplot(x='Epoch', y='Cost', hue='Dataset', data=df, ax=ax)   
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
                filename = title.strip('\n') + '.png'
            save_fig(fig, directory, filename)       
        return fig     

    def performance(self, history, directory=None, filename=None, 
                         xlim=None, ylim=None,show=True):
        """Renders performance learning curves for training and validation"""

        # Extract parameters and data
        d = {'Epoch': history.epochs,              
             'Training Scores': history.epoch_log.get('train_score'),
             'Validation Scores': history.epoch_log.get('val_score')}
        df = pd.DataFrame(data=d)
        df = pd.melt(df, id_vars='Epoch', value_vars=['Training Scores', 'Validation Scores'],
                     var_name=['Dataset'], value_name='Scores')
        # Format title
        title = history.params.get('name') + "\n" + "Performance Learning Curves" + \
            '\n' + proper(history.params.get('metric')) 

        # Initialize plot and set aesthetics        
        fig, ax = plt.subplots(figsize=(12,4))               
        sns.set(style="whitegrid", font_scale=1)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(proper(history.params.get('metric')) + "Score")
        ax.set_title(title, color='k')            

        # Plot Scores
        ax = sns.lineplot(x='Epoch', y='Scores', hue='Dataset', data=df, ax=ax)   
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
                filename = title.strip('\n') + '.png'
            save_fig(fig, directory, filename)       
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

    def distplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)        

        # Call seaborn method
        ax = sns.distplot(a=data[x])
        # Set aesthetics
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_title(title, color='k')
        # Set labels
        ax.set_xlabel(proper(x))
        ax.set_ylabel(proper(y))
        if z is not None:
            l = ax.legend()
            l.texts[0].set_text(proper(z))

        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)             
    
    def scatterplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None,  ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)        

        # # Call seaborn method
        ax = sns.scatterplot(x=x, y=y, hue=z, data=data, ax=ax, legend='full')
        # Set aesthetics
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_title(title, color='k')
        # Set labels
        ax.set_xlabel(proper(x))
        ax.set_ylabel(proper(y))
        if z is not None:
            l = ax.legend()
            l.texts[0].set_text(proper(z))
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)         

    def barplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)

        # # Call seaborn method 
        ax = sns.barplot(x=x, y=y, hue=z, data=data, ax=ax)
        # Set aesthetics
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_title(title, color='k')
        # Set labels
        ax.set_xlabel(proper(x))
        ax.set_ylabel(proper(y))
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)     

    def boxplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)  

        # # Call seaborn method 
        ax = sns.boxplot(x=x, y=y, hue=z, data=data, ax=ax)
        # Set aesthetics
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_title(title, color='k')
        # Set labels
        ax.set_xlabel(proper(x))
        ax.set_ylabel(proper(y))
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax) 

    def lineplot(self, ax, data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)

        # # Call seaborn method 
        ax = sns.lineplot(x=x, y=y, hue=z, data=data, legend='full', ax=ax)
        # Set aesthetics
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_title(title, color='k')
        # Set labels
        ax.set_xlabel(proper(x))
        ax.set_ylabel(proper(y))
        if z is not None:
            l = ax.legend()
            l.texts[0].set_text(proper(z))
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)     

    def catplot(self, ax,  data, x, y, z=None, title=None,
                 log=False, xlim=None, ylim=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)

        # Call seaborn method 
        ax = sns.catplot(x=x, y=y, hue=z, kind='bar', data=data, ax=ax)
        # Set labels
        ax.set_xlabels(proper(x))
        ax.set_ylabels(proper(y))
        # Change to log scale and impose axis limits if requested
        if log: ax.set_xscale('log')
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)            
        return(ax)           