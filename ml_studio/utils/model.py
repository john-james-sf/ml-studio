# =========================================================================== #
#                        MODEL RELATED UTILITIES                              #
# =========================================================================== #
# =========================================================================== #
# Project: Visualate                                                          #
# Version: 0.1.0                                                              #
# File: \model.py                                                             #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Wednesday November 27th 2019, 6:30:46 pm                       #
# Last Modified: Wednesday November 27th 2019, 7:22:53 pm                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

import numpy as np
import sklearn
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ml_studio.supervised_learning.training.estimator import Estimator

def get_model_name(model):
    """Obtains the model name for a Scikit-Learn, ML Studio estimator

    Obtains the model name for Scikit-Learn estimators, ML Studio estimators, 
    and GridSearchCV and RandomSearchCV objects
    
    Parameters
    ----------
    model : BaseEstimator, GridSearchCV, RandomSearchCV
        The object for which the name is being obtained

    Returns
    -------
    model_name : str
        The name of the model object
    """
    if isinstance(model, Estimator):
        # ML Studio estimator
        return model.name
    elif isinstance(model, (BaseEstimator, BaseSearchCV)):
        # Scikit-Learn estimator
        return model.__class__.__name__
    else:
        raise TypeError("Cannot detect the model name for an object "
                        "of type: %s" % str(type(model)))



def is_fitted(estimator, X=None, y=None):
    """Determines if an estimator has been fitted.

    Determines whether the Scikit Learn or ML Studio estimator has been fitted.
    For Scikit Learn estimators, the function calls the 'predict' method
    on the estimator and returns 'False' if it raises a 
    `sklearn.exceptions.NotFittedError` and `True` otherwise.  
    
    For ML Studio estimators, the function accesses the value of the 
    fitted attribute on the estimator. If the fitted attribute is 'False', 
    the function returns 'False'. If the X and y have been provided and the
    fitted attribute is 'True', the function will compare the shape of the 
    X data from the estimator to the shape of the X parameter. If the shapes 
    are equal, it is assumed that not only was the model fitted, but it 
    was fitted on the same data and the function returns 'True'. 
    Otherwise, it returns 'False'.
    """
    if isinstance(estimator, Estimator):
        if estimator.fitted is False:
            return False
        elif X:
            if X.shape == estimator.X.shape:
                return True
            else:
                return False
        else:
            return True
    else:        
        try:
            estimator.predict(np.zeros((10,2)))
        except sklearn.exceptions.NotFittedError:
            return False
        try:
        # The following was adapted from:
        # Title: yellowbrick.utils.helpers
        # Date: November 27, 2019
        # Version: 1.0.1
        # Availability: https://github.com/DistrictDataLabs/yellowbrick/blob/dd795b492a77da33f00f06f2f098c6b3324b36d0/yellowbrick/utils/helpers.py#L118
            check_is_fitted(
                estimator,
                [
                    "coef_",
                    "estimator_",
                    "labels_",
                    "n_clusters_",
                    "children_",
                    "components_",
                    "n_components_",
                    "n_iter_",
                    "n_batch_iter_",
                    "explained_variance_",
                    "singular_values_",
                    "mean_",
                ],
                all_or_any=any,
            )
            return True
        except sklearn.exceptions.NotFittedError:
            return False    