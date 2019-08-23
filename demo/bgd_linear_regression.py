# =========================================================================== #
#                         BGD LINEAR REGRESSION DEMO                          #
# =========================================================================== #
#%%
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ml_studio.supervised_learning.regression import LinearRegression
# Obtain, standardize and transform data
X, y = datasets.load_boston(return_X_y=True)
scaler = StandardScaler()    
X = scaler.fit_transform(X)
y = np.log1p(y) 
# Train model
model = LinearRegression(learning_rate=0.01, epochs=100, verbose=True, 
                         monitor='train_cost', metric=None, checkpoint=10, val_size=0)
model.fit(X,y)
model.summary()