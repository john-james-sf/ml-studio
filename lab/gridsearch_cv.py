# =========================================================================== #
#              LINEAR REGRESSION CROSS VALIDATION: LEARNING RATES             #
# =========================================================================== #
#%%
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import SGDElasticNetRegression
from ml_studio.visual.plots import plot_loss, plot_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
directory = "./demo/demo_figures/"

# Obtain and split data
X, y = datasets.load_boston(return_X_y=True)
y = np.log(y)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=50)
# Initialize regressors
bgd = LinearRegression(seed=50, epochs=100)
sgd = SGDRegression(seed=50, batch_size=1, epochs=100)
mgd = SGDRegression(seed=50, batch_size=32, epochs=100)

# Build the pipelines
pipe1 = Pipeline([('std', StandardScaler()),
                  ('bgd', bgd)])
pipe2 = Pipeline([('std', StandardScaler()),
                  ('sgd', sgd)])
pipe3 = Pipeline([('std', StandardScaler()),
                  ('mgd', mgd)])        

# Set up parameter grids
param_grid1 = [{'bgd__learning_rate': np.linspace(0.001,0.1,20)}]
param_grid2 = [{'sgd__learning_rate': np.linspace(0.001,0.1,20)}]
param_grid3 = [{'mgd__learning_rate': np.linspace(0.001,0.1,20)}]

# Designate inner and outer loops
inner_cv = KFold(n_splits=5, shuffle=True, random_state=50)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=50)

# Set up a GridSearchCV inner loop object for each algorithm
gridcvs = {}
for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3),
                            (pipe1, pipe2, pipe3),
                            ('Batch', 'Stochastic', 'Minibatch')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='neg_mean_squared_error',
                       n_jobs=1,
                       cv=inner_cv,
                       verbose=0,
                       refit=True)
    gridcvs[name] = gcv

# Set up cross-validation outer loop for model selection
for name, gs_est in sorted(gridcvs.items()):
    nested_score = cross_val_score(gs_est, 
                                   X=X_train, 
                                   y=y_train, 
                                   cv=outer_cv,
                                   n_jobs=1)
    print('%s | outer NMSE %.2f%% +/- %.2f' % 
          (name, nested_score.mean() * 100, nested_score.std() * 100))

#%%
# Fit best model to the whole training set
best_algo = gridcvs['Stochastic']

best_algo.fit(X_train, y_train)
train_acc = mean_squared_error(y_true=y_train, y_pred=best_algo.predict(X_train))
test_acc = mean_squared_error(y_true=y_test, y_pred=best_algo.predict(X_test))

print('NMSE %.2f%% (average over CV test folds)' %
      (best_algo.best_score_))
print('Best Parameters: %s' % gridcvs['Stochastic'].best_params_)
print('Training NMSE: %.2f%%' % (100 * train_acc))
print('Test NMSE: %.2f%%' % (100 * test_acc))

#%%
