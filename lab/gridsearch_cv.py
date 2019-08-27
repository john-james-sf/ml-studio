# =========================================================================== #
#              LINEAR REGRESSION CROSS VALIDATION: LEARNING RATES             #
# =========================================================================== #
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.visual.plots import plot_loss, plot_score, gscv_line_plot

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
directory = "./demo/demo_figures/"
#%%
# Obtain and split data
X, y = datasets.load_boston(return_X_y=True)
y = np.log(y)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=50)
# Initialize regressors
bgd = LinearRegression(seed=50, epochs=200)
sgd = LinearRegression(seed=50, batch_size=1, epochs=200)
mgd = LinearRegression(seed=50, batch_size=32, epochs=200)

# Build the pipelines
pipe1 = Pipeline([('std', StandardScaler()),
                  ('bgd', bgd)])
pipe2 = Pipeline([('std', StandardScaler()),
                  ('sgd', sgd)])
pipe3 = Pipeline([('std', StandardScaler()),
                  ('mgd', mgd)])        

# Set up parameter grids
param_grid1 = [{'bgd__learning_rate': np.linspace(0.001,0.07,20)}]
param_grid2 = [{'sgd__learning_rate': np.linspace(0.001,0.07,20)}]
param_grid3 = [{'mgd__learning_rate': np.linspace(0.001,0.07,20)}]

# Set up a GridSearchCV 
gridcvs = {}
for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3),
                            (pipe1, pipe2, pipe3),
                            ('Batch', 'Stochastic', 'Minibatch')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='neg_mean_squared_error',
                       return_train_score=True,
                       n_jobs=1,
                       cv=5,
                       verbose=0,
                       refit=True)
    gcv.fit(X_train, y_train)
    gridcvs[name] = gcv
#%%
# Plot
x = 'learning_rate'
y = 'mean_test_score'
gscv_line_plot(x=x,y=y,gscv=gridcvs)