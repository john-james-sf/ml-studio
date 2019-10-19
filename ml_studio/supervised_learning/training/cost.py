# =========================================================================== #
#                                    COST                                     #
# =========================================================================== #
"""Cost functions and gradient computations."""
from abc import ABC, abstractmethod
import numpy as np

class Cost(ABC):

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

class Quadratic(Cost):
    """Computes cost."""

    def __init__(self):        
        self.name = "Quadratic Loss Function"

    def __call__(self, y, y_pred):
        """Computes quadratic costs e.g. squared error cost"""
        e = y_pred - y 
        J = 1/2 * np.mean(e**2)
        return(J)

    def gradient(self, X, y, y_pred):
        """Computes quadratic costs gradient with respect to weights"""
        n_samples = y.shape[0]
        dW = 1/n_samples * (y_pred-y).dot(X)
        return(dW)

class BinaryCrossEntropy(Cost):
    """Computes cost and gradient w.r.t. weights and bias."""
    
    def __init__(self):        
        self.name = "Binary Cross Entropy Loss Function"

    def __call__(self, y, y_pred):
        """Computes binary cross entropy (w/sigmoid) costs"""
        n_samples = y.shape[0]
        # Prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_pred)) + np.multiply(1-y, np.log(1-y_pred)))
        return J

    def gradient(self, X, y, y_pred):
        """Computes binary cross entropy (w/sigmoid) gradient w.r.t. weights."""
        n_samples = y.shape[0]
        dW = 1/n_samples * X.T.dot(y_pred-y)
        return(dW)

class CategoricalCrossEntropy(Cost):
    """Computes softmax cross entropy (w/softmax) cost and gradient w.r.t. parameters."""
    
    def __init__(self):        
        self.name = "Categorical Cross Entropy Loss Function"

    def __call__(self, y, y_pred):
        """Computes cross entropy (w/softmax) costs"""
        n_samples = y.shape[0]
        # Convert y to integer if one-hot encoded
        if isinstance(y, np.ndarray):
            if y.shape[1] > 1:
                y = y.argmax(axis=1)
        # Prevent division by zero. Note y is NOT one-hot encoded
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        log_likelihood = -np.log(y_pred[np.arange(n_samples),y])
        J = np.sum(log_likelihood) / n_samples
        return(J)

    def gradient(self, X, y, y_pred):
        """Computes cross entropy (w/softmax) gradient w.r.t. parameters."""       
        n_samples = y.shape[0] 
        # Prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        y_pred[np.arange(n_samples), y] -= 1
        dy_pred = y_pred/n_samples
        dW = X.T.dot(dy_pred)        
        return dW        

class CostFunctions():
    """Returns the requested cost class."""

    def __call__(self,cost='quadratic'):

        dispatcher = {'quadratic': Quadratic(),
                      'binary_crossentropy': BinaryCrossEntropy(),
                      'categorical_crossentropy': CategoricalCrossEntropy()}
        return(dispatcher.get(cost))