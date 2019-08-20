# =========================================================================== #
#                           REGRESSION CLASSES                                #
# =========================================================================== #
"""Model for training and evaluating a neural network."""
import numpy as np
from .gradient_descent import GradientDescent
from ml_studio.operations.loss import BinaryCrossEntropy, SoftmaxCrossEntropy
from ml_studio.deep_learning.neural_network.activations import Sigmoid, Softmax

# --------------------------------------------------------------------------- #
#                          LOGISTIC CLASSIFICATION                            #
# --------------------------------------------------------------------------- #            
class LogisticRegression(GradientDescent):
    """Logistic regression class for binary classification problems."""

    def __init__(self, *args, **kwargs):
        super(LogisticRegression,self).__init__(*args, **kwargs)
        self._loss = BinaryCrossEntropy()
        self._sigmoid = Sigmoid()

    def _predict(self, X):
        """Computes sigmoid prediction."""        
        Z = X.dot(self._weights) 
        y_pred = self._sigmoid(Z)
        return y_pred

    def predict(self, X):
        """Computes binary prediction."""
        # Add intercept term if required
        n_features = self._weights.shape[0]
        if n_features == X.shape[1] + 1:
            X = np.insert(X, 0, 1, axis=1)
        Z = X.dot(self._weights) 
        S = self._sigmoid(Z)
        y_pred = np.round(S).astype(int)        
        return y_pred

# --------------------------------------------------------------------------- #
#                           MULTICLASS REGRESSION                             #
# --------------------------------------------------------------------------- #
class MulticlassRegression(GradientDescent):
    """Classification for n>2 classes."""
   
    def __init__(self, *args, **kwargs):
        super(MulticlassRegression,self).__init__(*args, **kwargs)
        self._loss = SoftmaxCrossEntropy()
        self._softmax = Softmax()

    def _init_weights(self, X, y):
        n_features = X.shape[1]
        n_outputs = len(np.unique(y))
        limit = 1 / np.sqrt(n_features)
        self._weights = np.random.uniform(-limit, limit, (n_features, n_outputs))         

    def _predict(self, X):        
        Z = X.dot(self._weights)
        y_pred = self._softmax(Z)        
        return y_pred

    def predict(self, X):        
        # Add intercept term if required
        n_features = self._weights.shape[0]
        if n_features == X.shape[1] + 1:
            X = np.insert(X, 0, 1, axis=1)            
        Z = X.dot(self._weights)
        S = self._softmax(Z)        
        y_pred = np.argmax(S, axis=1)        
        return y_pred

            