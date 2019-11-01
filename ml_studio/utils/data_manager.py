# =========================================================================== #
#                                 DATA                                        #
# =========================================================================== #
#%%
"""Data manipulation functions."""

from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

# --------------------------------------------------------------------------- #
#                               Transformers                                  #
# --------------------------------------------------------------------------- #
class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mu=0
        self.s=1

    def fit(self, X):
        if self.with_mean:
            self.mu = np.mean(X,axis=0)
        if self.with_std:
            self.s = np.std(X,axis=0)

    def transform(self, X):
        z = (X-self.mu)/self.s
        return z

    def inverse_transform(self, X):
        X = X * self.s
        X = X + self.mu
        return X


def batch_iterator(X, y=None, batch_size=None):
    """ Batch generator """
    n_samples = X.shape[0]
    if batch_size is None:
        batch_size = n_samples    
    for i in np.arange(0, n_samples, batch_size):
        if y is not None:
            yield X[i:i+batch_size], y[i:i+batch_size]
        else:
            yield X[i:i+batch_size]

def one_hot(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def todf(x, stub):
    """Converts nested array to dataframe."""
    n = len(x[0])
    df = pd.DataFrame()
    for i in range(n):
        colname = stub + str(i)
        vec = [item[i] for item in x]
        df_vec = pd.DataFrame(vec, columns=[colname])
        df = pd.concat([df, df_vec], axis=1)
    return(df)             

def make_polynomial_features(X, degree):
    n_samples, n_features = X.shape
    
    combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
    combinations = [item for sublist in combs for item in sublist]
    
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, combs], axis=1)

    return X_new    

#%%%
