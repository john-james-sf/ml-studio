# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \optimization.py                                                      #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 3rd 2019, 5:00:16 pm                          #
# Last Modified: Tuesday December 3rd 2019, 5:10:27 pm                        #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Optimization related functionality."""
import numpy as np
from sklearn.model_selection import KFold

from ml_studio.utils.data_manager import sampler
# --------------------------------------------------------------------------- #
class KFoldCV():
    """Performs KFold cross validation on a single estimator. 

    This is to analyze performance vis-a-vis training set sizes on a single
    estimator.

    Parameters
    ----------
    model : a Scikit-Learn or an ML Studio estimator
        A Scikit-Learn or ML Studio estimator.

    sizes : array-like
        List or nd.array containing the training set sizes to evaluate.        

    k : int Default = 5
        The number of folds.


    Attributes
    ----------
    cv_results_ : dict
        dictionary contains:
            mean_train_scores : nd.array. 
            mean_test_scores : nd.array 
            std_train_scores : nd.array 
            std_test_scores : nd.array 
            mean_fit_time : nd.array 
            std_fit_time : nd.array 
            sets = list of dictionaries. One element per dataset size
            train_scores : nd.array 
            test_scores : nd.array 
            fit_times : nd.array 

    """
    def __init__(self, model, sizes, k=5):
        self.model = model
        self.sizes = sizes
        self.k = k
        self.cv_results_ = {}

    def fit(self, X, y):
        """Performs the cross-validation over varying training set sizes.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        -------
        self : KFoldLearningCurve

        """
        # Validate parameters
        if not hasattr(self.sizes, "__len__"):
            raise TypeError("sizes must be a list or a numpy array")    

        if not isinstance(self.k, int):
            raise TypeError("k must be an integer greater than 1.")

        k_fold = KFold(n_splits=self.k)
        # Perform cross-validation over training set sizes  
        mean_epochs = []      
        mean_train_scores = []
        mean_test_scores = []
        mean_fit_times = []
        mean_fit_times_norm = []
        std_epochs = []
        std_train_scores = []
        std_test_scores = []
        std_fit_times = []
        std_fit_times_norm = []
        training_sets = []
        for s in self.sizes:
            training_set = {} 
            total_epochs = []
            train_scores = []
            test_scores = []
            fit_times = []
            fit_times_norm = []

            X_train, y_train = sampler(X,y, size=s, seed=50)            
            for train, test in k_fold.split(X_train, y_train):                
                training_set = {}                
                self.model.fit(X_train[train], y_train[train])                

                epochs = self.model.history.total_epochs
                train_score = self.model.score(X_train[train], y_train[train])
                test_score = self.model.score(X_train[test], y_train[test])
                fit_time = self.model.history.duration
                fit_time_norm = fit_time / epochs

                total_epochs.append(epochs)
                train_scores.append(train_score)
                test_scores.append(test_score)
                fit_times.append(fit_time)
                fit_times_norm.append(fit_time_norm)

            mean_total_epochs = np.mean(total_epochs)
            mean_epochs.append(mean_total_epochs)
            
            mean_train_score = np.mean(train_scores)
            mean_train_scores.append(mean_train_score)

            mean_test_score = np.mean(test_scores)
            mean_test_scores.append(mean_test_score)

            mean_fit_time = np.mean(fit_times)
            mean_fit_times.append(mean_fit_time)

            mean_fit_time_norm = np.mean(fit_times_norm)
            mean_fit_times_norm.append(mean_fit_time_norm)       

            std_total_epochs = np.std(total_epochs)
            std_epochs.append(std_total_epochs)     

            std_train_score = np.std(train_scores)
            std_train_scores.append(std_train_score)

            std_test_score = np.std(test_scores)
            std_test_scores.append(std_test_score)            

            std_fit_time = np.std(fit_times)
            std_fit_times.append(std_fit_time)

            std_fit_time_norm = np.std(fit_times_norm)
            std_fit_times_norm.append(std_fit_time_norm)

            # Format attribute
            training_set['size'] = s
            training_set['epochs'] = total_epochs
            training_set['train_scores'] = train_scores
            training_set['test_scores'] = test_scores
            training_set['fit_times'] = fit_times
            training_set['fit_times_norm'] = fit_times_norm

            training_sets.append(training_set)

        self.cv_results_['mean_epochs'] = mean_epochs
        self.cv_results_['mean_train_scores'] = mean_train_scores
        self.cv_results_['mean_test_scores'] = mean_test_scores
        self.cv_results_['mean_fit_times'] = mean_fit_times        
        self.cv_results_['mean_fit_times_norm'] = mean_fit_times_norm

        self.cv_results_['std_epochs'] = std_epochs
        self.cv_results_['std_train_scores'] = std_train_scores
        self.cv_results_['std_test_scores'] = std_test_scores
        self.cv_results_['std_fit_times'] = std_fit_times
        self.cv_results_['std_fit_times_norm'] = std_fit_times_norm

        self.cv_results_['training_sets'] = training_sets





