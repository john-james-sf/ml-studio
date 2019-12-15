# %%
# =========================================================================== #
#                                FILE MANAGER                                 #
# =========================================================================== #
import os
import time
import shutil

from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.offline as py

from ml_studio.utils.misc import snake

def save_fig(fig, directory, filename):
    if os.path.exists(directory):
        path = os.path.join(os.path.abspath(directory), filename)
        fig.savefig(path, facecolor='w', bbox_inches=None)
    else:
        os.makedirs(directory)
        path = os.path.join(os.path.abspath(directory),filename)
        fig.savefig(path, facecolor='w', bbox_inches=None)

def save_gif(ani, directory, filename, fps):
    face_edge_colors = {'facecolor': 'w', 'edgecolor': 'w'}
    path = os.path.join(os.path.abspath(directory), filename)
    if os.path.exists(directory):
        ani.save(path, writer='imagemagick', fps=fps, savefig_kwargs = face_edge_colors)
    else:
        os.makedirs(directory)                
        ani.save(path, writer='imagemagick', fps=fps, savefig_kwargs = face_edge_colors)

def save_csv(df, directory, filename):
    path = os.path.join(os.path.abspath(directory), filename)
    if os.path.exists(directory):
        df.to_csv(path, index=False)
    else:
        os.makedirs(directory)                
        df.to_csv(path, index=False)

def save_numpy(a, directory, filename):
    path = os.path.join(os.path.abspath(directory), filename)
    if os.path.exists(directory):
        np.save(file=path, arr=a)
    else:
        os.makedirs(directory)                
        np.save(file=path, arr=a)

def save_plotly(a, directory, filename):
    path = os.path.join(os.path.abspath(directory), filename)
    if os.path.exists(directory):
        py.plot(a, filename=path, auto_open=False, include_mathjax='cdn')
    else:
        os.makedirs(directory)                
        py.plot(a, filename=path, auto_open=False, include_mathjax='cdn')

def cleanup(directory, ext=".html"):
    """Removes a directory or files within directory ending with 'ext'.""" 
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            try:
                shutil.rmtree(filepath)
            except NotADirectoryError:
                if filename.endswith(ext):
                    os.remove(os.path.join(directory, filename))           
        
def get_filename(instance, ext, name=None):
    """Creates a standard format filename for saving files.

    Parameters
    ----------
    instance : DataPackage, DataSet, Model, WorkFlow, Project, Visualator
        An instance of any class within the ML Studio object model

    ext :  str
        String containing the file extension

    name : str
        A string containing the name of a sub object. For instance, a 
        Histogram object may have a sub_object of 'price' which indicates
        the specific variable for which the histogram is being plotted.    
    
    """        

    # Obtain user id, class name and date time        
    userhome = os.path.expanduser('~')          
    username = os.path.split(userhome)[-1]     
    clsname = instance.__class__.__name__
    object_name = ""
    if hasattr(instance, 'name'):
        if instance.name is not None:
            object_name = instance.name + '_' or ""
    element_name = name or ""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Snake case format filename
    filename = username + '_' + clsname + '_' + object_name  + \
    element_name + '_' + timestr + ext
    filename = snake(filename)        
    return filename

