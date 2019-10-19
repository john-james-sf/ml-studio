# =========================================================================== #
#                                GRIDSEARCH                                   #
# =========================================================================== #
"""Plots results of sklearn gridsearches"""
# --------------------------------------------------------------------------- #
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import seaborn as sns

from ml_studio.utils.file_manager import save_fig
# --------------------------------------------------------------------------- #
def _format_string(s):
    if "__" in s:
        s = s.split("__")[1]
    return s.replace("_", " ").capitalize()
    
class VisualCV:

    def _plot(self, pipelines, x, y, z=None, kind='line', 
              title='GridSearchCV Scores', height=1, width=1,  
              log=False, directory=None, filename=None, show=True):

        # Obtain class containing standard plotting functions
        plot = BasicPlots()

        # Extract results from pipeline and concatenate
        results = []
        for _, v in pipelines.items():
            result = pd.DataFrame.from_dict(v.cv_results_)
            results.append(result)
        results = pd.concat(results, axis=0)

        # Remove unwanted prefixes from results column names
        results.columns = results.columns.str.split('__').str[-1]
        try:
            x_param = results.filter(like=x).columns[0]            
            if z is not None:
                z_param = results.filter(like=z).columns[0]    
                # Convert z_param to categorical since it is used in seaborn 
                # line plot hue parameter
                results[z_param] = pd.Categorical(results[z_param])                
            else:
                z_param = None     
        except:
            print("Invalid parameter name")

        # Obtain and initialize matplotlib figure
        fig_width = math.floor(12*width)
        fig_height = math.floor(4*height)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
        
        # Set Title
        title_plot = title + '\n' + 'By ' + _format_string(x_param) 
        
        # Render Plot
        if kind == 'line':
            ax = plot.lineplot(x=x_param, y=y, z=z_param, data=results, ax=ax, title=title_plot)
        else:
            ax = plot.barplot(x=x_param, y=y, z=z_param, data=results, ax=ax, title=title_plot)

        # Change to log scale if requested
        if log: ax.set_xscale('log')
        
        # Finalize, show and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = title + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)             
    
    def scores(self, pipelines, x, y='train', z=None, kind='line', height=1, width=1,  
               log=False, directory=None, filename=None, show=True):
        """Plots training scores for one or more GridSearchCV Objects"""        
        y = 'mean_train_score' if y == 'train' else "mean_test_score"
        title = 'GridSearchCV Training Scores' if y == 'train' else 'GridSearchCV Validation Scores'
        self._plot(pipelines, x, y, z, kind, title,
                        height, width, log, directory, filename, show)                  

    def times(self, pipelines, x, y=None, z=None, kind='line', height=1, width=1,  
                log=False, directory=None, filename=None, show=True):
        """Plots fit times for one or more GridSearchCV Objects"""                
        y = 'mean_fit_time'
        title = 'GridSearchCV Fit Times'
        self._plot(pipelines, x, y, z, kind, title,
                        height, width, log, directory, filename, show)   

class Selection():                                 

    def validation_curves(self, models, log=False, directory=None, filename=None, 
                       xlim=None, ylim=None, subtitle=None, show=True):
        """Plots validation scores for multiple models."""

        # Extract data
        results = []
        for name, model in models.items():
            d = {'Epoch': np.arange(1,model.epochs+1),
                 'Model': name,
                 'Validation Score': model.val_scores}
            result = pd.DataFrame(data=d)
            results.append(result)
        results = pd.concat(results, axis=0)

        # Format title
        title = "Validation Scores"
        if subtitle is not None:
            title = title + "\n" + subtitle

        # Render Plot
        fig, ax = plt.subplots(figsize=(12,4))               
        plot = BasicPlots()
        ax = plot.lineplot(x='Epoch', y='Validation Score', z='Model', data=results, title=title, ax=ax)
        # Set x and y limits
        if xlim is not None:
            ax.set_xlim(left = xlim[0], right=xlim[1])
        if ylim is not None:
            ax.set_xlim(bottom=ylim[0], top=ylim[1])
        # Change to log scale if requested
        if log: ax.set_xscale('log')        
        # Finalize, show and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = title + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)                                                     


    def learning_curves(self, models, z="name", log=False, directory=None, filename=None, 
                       xlim=None, ylim=None,show=True):
        """For fixed learning rates, plots learning curves for multiple models."""

        # Extract data
        results = []
        for _, v in models.items():
            params = v.get_params()
            result = pd.DataFrame({z: params[z], "Epoch": np.arange(1,v.epochs+1), 
                                   "Cost": v.train_costs})
            results.append(result)
        results = pd.concat(results, axis=0)

        # Render Plot
        fig, ax = plt.subplots(figsize=(12,4))               
        plot = BasicPlots()
        title = "Learning Curve(s)"
        ax = plot.lineplot(x='Epoch', y='Cost', z=z, data=results, title=title, ax=ax)
        # Set x and y limits
        if xlim is not None:
            ax.set_xlim(left = xlim[0], right=xlim[1])
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        # Change to log scale if requested
        if log: ax.set_xscale('log')        
        # Finalize, show and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = title + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)
