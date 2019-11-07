# =========================================================================== #
#                         CLASSIFICATION MODULE                               #
# =========================================================================== #
#%%
"""Classes for binary and multi-class classification."""
import numpy as np

from ml_studio.supervised_learning.training.cost import Cost
from ml_studio.supervised_learning.training.cost import BinaryClassificationCostFunctions
from ml_studio.supervised_learning.training.cost import MultiClassificationCostFunctions
from ml_studio.supervised_learning.training.metrics import ClassificationMetric
from ml_studio.supervised_learning.training.metrics import ClassificationMetrics
from ml_studio.supervised_learning.training.gradient_descent import GradientDescent
from ml_studio.utils.data_manager import data_split, one_hot

# --------------------------------------------------------------------------- #
#                              CLASSIFICATION                                 #
# --------------------------------------------------------------------------- #                
class Classification(GradientDescent):
    """Abstract base class for classification classes."""

    DEFAULT_METRIC = 'accuracy'

    def score(self, X, y):
        """Computes a score for the current model, given inputs X and output y.

        The score uses the class associated the metric parameter from class
        instantiation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        y : numpy array, shape (n_samples,)
            Target values             

        Returns
        -------
        float
            Returns the score for the designated metric.
        """
        self._validate_data(X, y)
        y_pred = self.predict(X)
        if self.metric:
            score = self.scorer(y=y, y_pred=y_pred)    
        else:
            score = ClassificationMetrics()(metric=self.DEFAULT_METRIC)(y=y, y_pred=y_pred)        
        return score    

    def _get_scorer(self):
        """Obtains the scoring function associated with the metric parameter."""
        scorer = ClassificationMetrics()(metric=self.metric)
        if not isinstance(scorer, ClassificationMetric):
            msg = str(self.metric) + ' is not a supported classification metric.'
            raise ValueError(msg)
        else:
            return scorer        

# --------------------------------------------------------------------------- #
#                          LOGISTIC CLASSIFICATION                            #
# --------------------------------------------------------------------------- #            
class LogisticRegression(Classification):
    """Trains models for binary classification using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, optional (default='binary_cross_entropy')
        The string name for the cost function

        'binary_cross_entropy':
            Computes binary cross entropy 
        'hinge':
            Computes Hinge Loss
        'squared_hinge':
            Computes Squared Hinge Loss

    metric : str, optional (default='accuracy')
        Metrics used to evaluate classification scores:

        'accuracy': 
            Accuracy - Total Accurate Predictions / Total Predictions
        'auc': 
            Compute Area Under the Curve (AUC)
        'confusion_matrix':
            Compute confusion matrix to evaluate accuracy of a classification
        'f1':
            Compute F1 score.
        'precision':
            Compute the precision
        'recall':
            Compute the recall
        'roc':
            Compute Reciever Operating Characteristics (ROC)

    early_stop : None or EarlyStop subclass, optional (default=None)
        The early stopping algorithm to use during training.

    verbose : bool, optional (default=False)
        If true, performance during training is summarized to sysout.

    checkpoint : None or int, optional (default=100)
        If verbose, report performance each 'checkpoint' epochs

    name : None or str, optional (default=None)
        The name of the model used for plotting

    seed : None or int, optional (default=None)
        Random state seed        

    Attributes
    ----------
    coef_ : array-like shape (n_features,1) or (n_features, n_classes)
        Coefficient of the features in X. 'coef_' is of shape (n_features,1)
        for binary problems. For multi-class problems, 'coef_' corresponds
        to outcome 1 (True) and '-coef_' corresponds to outcome 0 (False).

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 
        'intercept_' is of shape (1,) for binary problems. For multi-class
        problems, `intercept_` corresponds to outcome 1 (True) and 
        `-intercept_` corresponds to outcome 0 (False).

    epochs_ : int
        Total number of epochs executed.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    classification.MultinomialLogisticRegression : Multinomial Classification
    """    

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None,
                 epochs=1000, cost='binary_cross_entropy',
                 metric='accuracy',  early_stop=None, verbose=False, 
                 checkpoint=100, name=None, seed=None):
        super(LogisticRegression,self).__init__(learning_rate=learning_rate, 
                 batch_size=batch_size, theta_init=theta_init, 
                 epochs=epochs, cost=cost, metric=metric,  
                 early_stop=early_stop, verbose=verbose, 
                 checkpoint=checkpoint, name=name, seed=seed)                 

    def _sigmoid(self, z):
        """Computes the sigmoid for a scalar or vector z."""
        s = 1/(1+np.exp(-z))
        return s                 

    def _set_name(self):
        """Set name of model for plotting purposes."""
        self.task = "Logistic Regression"
        self.name = self.name or self.task + ' with ' + self.algorithm

    def _get_cost_function(self):
        """Obtains the cost function associated with the cost parameter."""
        cost_function = BinaryClassificationCostFunctions()(cost=self.cost)
        if not isinstance(cost_function, Cost):
            msg = str(self.cost) + ' is not a supported binary classification cost function.'
            raise ValueError(msg)
        else:
            return cost_function

    def _predict(self, X):
        """Predicts sigmoid probabilities."""        
        z = self._linear_prediction(X) 
        y_pred = self._sigmoid(z).astype('float64').flatten()
        return y_pred

    def predict(self, X):
        """Predicts class label.
        
        Parameters
        ----------
        X : array-like of shape (m, n_features)

        Returns
        -------
        Vector of class label predictions
        """        
        prob = self._predict(X)      
        y_pred = np.round(prob).astype(int).flatten()
        return y_pred

# --------------------------------------------------------------------------- #
#                  MULTINOMIAL LOGISTIC REGRESSION                            #
# --------------------------------------------------------------------------- #            
class MultinomialLogisticRegression(Classification):
    """Trains models for binary classification using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, optional (default='binary_cross_entropy')
        The string name for the cost function

        'binary_cross_entropy':
            Computes binary cross entropy 
        'hinge':
            Computes Hinge Loss
        'squared_hinge':
            Computes Squared Hinge Loss

    metric : str, optional (default='accuracy')
        Metrics used to evaluate classification scores:

        'accuracy': 
            Accuracy - Total Accurate Predictions / Total Predictions
        'auc': 
            Compute Area Under the Curve (AUC)
        'confusion_matrix':
            Compute confusion matrix to evaluate accuracy of a classification
        'f1':
            Compute F1 score.
        'precision':
            Compute the precision
        'recall':
            Compute the recall
        'roc':
            Compute Reciever Operating Characteristics (ROC)

    early_stop : None or EarlyStop subclass, optional (default=None)
        The early stopping algorithm to use during training.

    verbose : bool, optional (default=False)
        If true, performance during training is summarized to sysout.

    checkpoint : None or int, optional (default=100)
        If verbose, report performance each 'checkpoint' epochs

    name : None or str, optional (default=None)
        The name of the model used for plotting

    seed : None or int, optional (default=None)
        Random state seed        

    Attributes
    ----------
    coef_ : array-like shape (n_features,1) or (n_features, n_classes)
        Coefficient of the features in X. 'coef_' is of shape (n_features,1)
        for binary problems. For multi-class problems, 'coef_' corresponds
        to outcome 1 (True) and '-coef_' corresponds to outcome 0 (False).

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 
        'intercept_' is of shape (1,) for binary problems. For multi-class
        problems, `intercept_` corresponds to outcome 1 (True) and 
        `-intercept_` corresponds to outcome 0 (False).

    classes_ : array-like, shape (n_classes_,)
        Array containing all class labels

    n_classes_ : int
        The number of classes 

    epochs_ : int
        Total number of epochs executed.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    classification.LogisticRegression : Binary Classification
    """    

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='categorical_cross_entropy',
                 metric='accuracy',  early_stop=None, verbose=False, 
                 checkpoint=100, name=None, seed=None):
        super(MultinomialLogisticRegression,self).__init__(learning_rate=learning_rate, 
                 batch_size=batch_size, theta_init=theta_init, 
                 epochs=epochs, cost=cost, metric=metric,  
                 early_stop=early_stop, verbose=verbose, 
                 checkpoint=checkpoint, name=name, seed=seed)

        self.n_classes_ = 0
        self.classes_ = None

    def _set_name(self):
        """Set name of model for plotting purposes."""
        self.task = "Multinomial Logistic Regression"
        self.name = self.name or self.task + ' with ' + self.algorithm

    def _softmax(self, z, axis=None):
        """Computes softmax probabilities."""
        return np.exp(z)/np.sum(np.exp(z), axis=axis)


    def _validate_params(self):
        """Adds confirmation that metric is a valid regression metric."""
        super(MultinomialLogisticRegression,self)._validate_params()
        if self.metric is not None:
            if not ClassificationMetrics()(metric=self.metric):            
                msg = str(self.metric) + ' is not a supported classification metric.'
                raise ValueError(msg)    
        if not MultiClassificationCostFunctions()(cost=self.cost):
            msg = str(self.cost) + ' is not a supported multinomial classification cost function.'
            raise ValueError(msg)   

    def _prepare_data(self, X, y):
        """Prepares data and reports classes and n_classes."""
        super(MultinomialLogisticRegression, self)._prepare_data(X,y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
    
    def _init_weights(self):
        """Initializes weights according to the shapes of X and y."""
        # Perform random uniform initialization of parameters                              
        limit = 1 / np.sqrt(self.n_features_)
        np.random.seed(self.seed)
        self.theta = np.random.uniform(-limit, limit, (self.n_features_, self.n_classes_))  

    def _get_cost_function(self):
        """Obtains the cost function associated with the cost parameter."""
        cost_function = MultiClassificationCostFunctions()(cost=self.cost)
        if not isinstance(cost_function, Cost):
            msg = str(self.cost) + ' is not a supported multi class classification cost function.'
            raise ValueError(msg)
        else:
            return cost_function

    def _predict(self, X):
        """Predicts softmax probabilities."""        
        z = self._linear_prediction(X) 
        y_pred = self._softmax(z).astype('float64')
        return y_pred

    def predict(self, X):
        """Predicts class labels.
        
        Parameters
        ----------
        X : array-like of shape (m, n_features)

        Returns
        -------
        Vector of class label predictions
        """        
        prob = self._predict(X)
        y_pred = np.argmax(prob, axis=1).flatten()
        return y_pred



# %%
