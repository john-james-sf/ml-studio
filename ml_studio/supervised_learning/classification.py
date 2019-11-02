# =========================================================================== #
#                         CLASSIFICATION CLASSES                              #
# =========================================================================== #
"""Model for training and evaluating a neural network."""
import numpy as np


from ml_studio.supervised_learning.training.cost import BinaryCrossEntropy
from ml_studio.supervised_learning.training.cost import CategoricalCrossEntropy
from ml_studio.supervised_learning.regression import GradientDescent
# --------------------------------------------------------------------------- #
#                            SIGMOID & SOFTMAX                                #
# --------------------------------------------------------------------------- #
def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(z, axis=None):
    return np.exp(z)/np.sum(np.exp(z))

# --------------------------------------------------------------------------- #
#                          LOGISTIC CLASSIFICATION                            #
# --------------------------------------------------------------------------- #            
class LogisticRegression(GradientDescent):
    """Logistic regression class for binary classification problems."""

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='binary_crossentropy', 
                 metric='accuracy', 
                 early_stop=None, verbose=False, checkpoint=100, 
                 name=None, seed=None):
        super(LogisticRegression,self).__init__(learning_rate=learning_rate, 
                 batch_size=batch_size, theta_init=theta_init, 
                 epochs=epochs, cost=cost, 
                 metric=metric,  early_stop=early_stop, verbose=verbose, 
                 checkpoint=checkpoint, 
                 name=name, seed=seed)

    def _predict(self, X):
        """Computes sigmoid prediction during training."""        
        z = self.decision(X) 
        y_pred = sigmoid(z)
        return y_pred

    def predict(self, X):
        """Computes binary prediction. Public class used on unseen data."""        
        z = self.decision(X) 
        s = sigmoid(z)
        y_pred = np.round(s).astype(int)        
        return y_pred

# --------------------------------------------------------------------------- #
#                    MULTINOMIAL LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #
class MultinomialLogisticRegression(GradientDescent):
    """Classification for n>2 classes."""
   
    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='categorical_crossentropy', 
                 metric='accuracy', 
                 early_stop=None, verbose=False, checkpoint=100, 
                 name=None, seed=None):
        super(MultinomialLogisticRegression,self).__init__(learning_rate=learning_rate, 
                 batch_size=batch_size, theta_init=theta_init, 
                 epochs=epochs, cost=cost, 
                 metric=metric,  early_stop=early_stop, verbose=verbose, 
                 checkpoint=checkpoint, 
                 name=name, seed=seed)

    def _init_weights(self):
        n_features = self.X.shape[1]
        n_outputs = len(np.unique(self.y))
        limit = 1 / np.sqrt(n_features)
        self.theta = np.random.uniform(-limit, limit, (n_features, n_outputs))         

    def _predict(self, X):        
        z = self.decision(X)
        y_pred = softmax(z)        
        return y_pred

    def predict(self, X):        
        z = self.decision(X)
        s = softmax(z)        
        y_pred = np.argmax(s, axis=1)        
        return y_pred

            