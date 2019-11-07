# =========================================================================== #
#                                 DATA                                        #
# =========================================================================== #
#%%
"""Data manipulation functions."""

from itertools import combinations_with_replacement
from math import ceil
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

# --------------------------------------------------------------------------- #
#                               TRANSFORMERS                                  #
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
# --------------------------------------------------------------------------- #
#                               FUNCTIONS                                     #
# --------------------------------------------------------------------------- #
def shuffle_data(X, y=None, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def data_split(X, y, test_size=0.3, shuffle=True, stratify=False, seed=None):
    """ Split the data into train and test sets 
    
    Splits inputs X, and y into training and test sets of proportions
    1-test_size, and test_size respectively.

    Parameters
    ----------
    X : array_like of shape (m, n_features)
        Input data

    y : array_like of shape (m,)
        Target data

    test_size : float, optional (default=0.3)
        The proportion of X, and y to be designated to the test set.

    shuffle : bool, optional (default=True)
        Bool indicating whether the data should be shuffled prior to split.

    stratify : bool, optional (default=False)
        If True, stratified sampling is performed. 

    seed : int, optional (default=None)
        Random state variable

    Returns
    -------
    X_train : array-like
        Training data 

    X_test : array-like
        Test data

    y_train : array-like
        Targets for X_train

    y_test : array_like
        Targets for X_test 
    """
    if not stratify:
        if shuffle:
            X, y = shuffle_data(X, y, seed)
        split_i = len(y) - int(len(y) // (1 / test_size))
        X_train, X_test = X[:split_i], X[split_i:]
        y_train, y_test = y[:split_i], y[split_i:]
    else:
        train_idx = []
        test_idx = []
        classes, group_indices = np.unique(y, return_inverse=True)
        for k in classes:
            idx_k = np.array(np.where(group_indices == k)).reshape(-1,1)  
            n_samples_k = idx_k.shape[0]
            n_train_samples_k = ceil(n_samples_k * (1-test_size))
            n_test_samples_k = n_samples_k - n_train_samples_k
            if shuffle:
                if seed:
                    np.random.seed(seed)
                np.random.shuffle(idx_k)
            train_idx_k = idx_k[0:n_train_samples_k]
            test_idx_k = idx_k[n_train_samples_k:n_train_samples_k+n_test_samples_k]
            train_idx.append(train_idx_k)
            test_idx.append(test_idx_k)
        train_idx = np.concatenate(train_idx).ravel()
        test_idx = np.concatenate(test_idx).ravel()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def batch_iterator(X, y=None, batch_size=None):
    """Batch generator.
    
    Creates an iterable of batches of the designated batch size.

    Parameters
    ----------
    X : array-like
        A features matrix of shape (m, n_features), where m is the number of 
        examples in X.
    
    y : array-like, optional (default=None)
        The target vector of shape (m,)

    batch_size : None or int, optional (default=None)
        The number of observations to be included in each batch. 

    Returns
    -------
    array-like
        Returns inputs in batches of batch_size. If batch_size
        is None, a single batch containing all data is generated.
    
    """
    n_samples = X.shape[0]
    if batch_size is None:
        batch_size = n_samples    
    for i in np.arange(0, n_samples, batch_size):
        if y is not None:
            yield X[i:i+batch_size], y[i:i+batch_size]
        else:
            yield X[i:i+batch_size]

def one_hot(x, n_classes=None, dtype='float32'):
    """Converts a vector of integers to one-hot encoding. 
    
    Creates a one-hot matrix for multi-class classification
    and categorical cross_entropy.

    Parameters
    ----------
    x : array-like of shape(n_observations,)
        Vector of integers (from 0 to num_classes-1) to be converted to one-hot matrix.
    n_classes : int
        Number of classes
    dtype : Data type to be expected, as a string
        ('float32', 'float64', 'int32', ...)

    Returns
    -------
    A binary one-hot matrix representation of the input. The classes axis
    is placed last.
    """
    x = np.array(x, dtype='int')
    if not n_classes:
        n_classes = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_classes))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def decode(x, axis=1):
    """Convert probability distributions to integers."""     
    return np.argmax(x, axis=axis)

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

#%%%
