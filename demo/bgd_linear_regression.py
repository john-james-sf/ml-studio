# =========================================================================== #
#                         BGD LINEAR REGRESSION DEMO                          #
# =========================================================================== #
#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import SGDRegression
from ml_studio.visual.plots import TrainingPlots
#%%
# Obtain, standardize and transform data
X, y = datasets.load_boston(return_X_y=True)
scaler = StandardScaler()    
X = scaler.fit_transform(X)
y = np.log1p(y) 

# Train model
model = SGDRegression(learning_rate=0.01, epochs=100, verbose=True, 
                         metric=None, val_size=0.33, seed=50)
model.fit(X,y)

# Render Training Plot
p = TrainingPlots()
x = p.plot_loss(model=model)

