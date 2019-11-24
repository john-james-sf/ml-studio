# --------------------------------------------------------------------------- #
#                           TEST CLASSIFICATION                               #
# --------------------------------------------------------------------------- #
#%%
# Imports
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.classification import LogisticRegression
from ml_studio.supervised_learning.classification import MultinomialLogisticRegression
from ml_studio.supervised_learning.training.early_stop import EarlyStopImprovement
from ml_studio.supervised_learning.training.early_stop import EarlyStopStrips
from ml_studio.supervised_learning.training.metrics import Metric
# --------------------------------------------------------------------------- #
#%%
# --------------------------------------------------------------------------- #
#                         LOGISTIC REGRESSION                                 #
# --------------------------------------------------------------------------- #
class LogisticRegressionTests:

    @mark.logistic_regression
    @mark.logistic_regression_name
    def test_logistic_regression_name(self, get_binary_classification_data):
        X, y = get_binary_classification_data
        clf = LogisticRegression(epochs=50)        
        clf.fit(X,y)
        assert clf.name == 'Logistic Regression with Batch Gradient Descent'
        clf = LogisticRegression(epochs=50, batch_size=1)        
        clf.fit(X,y)
        assert clf.name == 'Logistic Regression with Stochastic Gradient Descent'
        clf = LogisticRegression(epochs=50, batch_size=32)        
        clf.fit(X,y)
        assert clf.name == 'Logistic Regression with Minibatch Gradient Descent'

    @mark.logistic_regression
    @mark.logistic_regression_val
    def test_logistic_regression_validation(self, get_binary_classification_data):
        X, y = get_binary_classification_data
        clf = LogisticRegression(epochs=50, metric='mean')                
        with pytest.raises(ValueError):
            clf.fit(X,y)
        clf = LogisticRegression(epochs=50, cost='quadratic')                
        with pytest.raises(ValueError):
            clf.fit(X,y)            

    @mark.logistic_regression
    @mark.logistic_regression_predict
    def test_logistic_regression_predict(self, get_binary_classification_data):
        X, y = get_binary_classification_data
        clf = LogisticRegression(epochs=100, learning_rate=0.01, checkpoint=10)
        clf.fit(X,y)
        y_pred = clf._predict(X)
        assert y_pred.shape == (y.shape[0],), "y_pred has wrong shape for binary problem"                
        y_pred = clf.predict(X)        
        score = clf.score(X,y)
        assert y_pred.shape == (y.shape[0],), "y_pred has wrong shape for binary problem"
        assert score > 0.3, "Accuracy below 0.3"
        assert score < 1, "Accuracy is greater than or equal to 1"
        
    @mark.logistic_regression
    @mark.logistic_regression_history
    def test_logistic_regression_history_w_early_stop(self, get_binary_classification_data):        
        X, y = get_binary_classification_data
        es = EarlyStopImprovement()
        clf = LogisticRegression(epochs=10, early_stop=es)
        clf.fit(X, y)        
        # Test epoch history
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('val_cost')), "number of val costs in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('train_score')), "number of train score in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('val_score')), "number of val score in log doesn't match epochs"        
        assert all(np.equal(clf.theta, clf.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert clf.history.epoch_log.get('train_cost')[0] > clf.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        #assert clf.history.epoch_log.get('train_score')[0] > clf.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert clf.history.epoch_log.get('val_cost')[0] > clf.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        #assert clf.history.epoch_log.get('val_score')[0] > clf.history.epoch_log.get('val_score')[-1], "val_score does not decrease"        
        # Test batch history
        assert clf.history.total_batches == len(clf.history.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert clf.history.total_batches == len(clf.history.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert clf.history.total_batches == len(clf.history.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert clf.history.total_batches == len(clf.history.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"                


    @mark.logistic_regression
    @mark.logistic_regression_learning_rate_schedules
    def test_logistic_regression_learning_rate_schedules(self, learning_rate_schedules, get_binary_classification_data):        
        X, y = get_binary_classification_data        
        clf = LogisticRegression(epochs=200, checkpoint=10, learning_rate=learning_rate_schedules, patience=40)
        clf.fit(X, y)       
        # Confirm learning rates decreased
        assert clf.history.epoch_log.get('learning_rate')[0] > clf.history.epoch_log.get('learning_rate')[-1], "Learning rate didn't decrease"
        assert clf.history.epoch_log.get('learning_rate')[0] != clf.eta, "Learning rate didn't change"        

    @mark.logistic_regression
    def test_logistic_regression_early_stop_from_estimator_train_cost(self, get_binary_classification_data): 
        X, y = get_binary_classification_data        
        clf = LogisticRegression(epochs=5000, early_stop=False, val_size=0.3, metric=None)
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'train_cost', "Estimator is not sending correct metric"

    @mark.logistic_regression
    def test_logistic_regression_early_stop_from_estimator_train_score(self, get_binary_classification_data): 
        X, y = get_binary_classification_data        
        clf = LogisticRegression(epochs=5000, early_stop=False, val_size=0.3, metric='accuracy')
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'train_score', "Estimator is not sending correct metric"

    @mark.logistic_regression
    def test_logistic_regression_early_stop_from_estimator_val_cost(self, get_binary_classification_data): 
        X, y = get_binary_classification_data        
        clf = LogisticRegression(epochs=5000, early_stop=True, val_size=0.3, metric=None)
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'val_cost', "Estimator is not sending correct metric"        

    @mark.logistic_regression    
    def test_logistic_regression_early_stop_from_estimator_val_score(self, get_binary_classification_data): 
        X, y = get_binary_classification_data        
        clf = LogisticRegression(epochs=5000, early_stop=True, val_size=0.3, metric='accuracy')
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'val_score', "Estimator is not sending correct metric"                     

# --------------------------------------------------------------------------- #
#                    MULTINOMIAL LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #
class MultinomialLogisticRegressionTests:

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    @mark.multinomial_logistic_regression_name
    def test_multinomial_logistic_regression_name(self, get_multinomial_classification_data):
        X, y = get_multinomial_classification_data
        clf = MultinomialLogisticRegression(epochs=50, cost='categorical_cross_entropy')        
        clf.fit(X,y)
        assert clf.name == 'Multinomial Logistic Regression with Batch Gradient Descent'
        clf = MultinomialLogisticRegression(epochs=50, batch_size=1,cost='categorical_cross_entropy')        
        clf.fit(X,y)
        assert clf.name == 'Multinomial Logistic Regression with Stochastic Gradient Descent'
        clf = MultinomialLogisticRegression(epochs=50, batch_size=32, cost='categorical_cross_entropy')        
        clf.fit(X,y)
        assert clf.name == 'Multinomial Logistic Regression with Minibatch Gradient Descent'

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    @mark.multinomial_logistic_regression_val
    def test_multinomial_logistic_regression_validation(self, get_multinomial_classification_data):
        X, y = get_multinomial_classification_data
        clf = MultinomialLogisticRegression(epochs=50, metric='mean_squared_error')                
        with pytest.raises(ValueError):
            clf.fit(X,y)
        clf = MultinomialLogisticRegression(epochs=50, cost='binary_cross_entropy')                
        with pytest.raises(ValueError):
            clf.fit(X,y)

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    @mark.multinomial_logistic_regression_prep_data
    def test_multinomial_logistic_regression_prep_data(self, get_multinomial_classification_data):
        X, y = get_multinomial_classification_data
        clf = MultinomialLogisticRegression(epochs=50, cost='categorical_cross_entropy',
                                            val_size=0, early_stop=False)                
        clf.fit(X,y)
        assert X.shape[0] == clf.X.shape[0], "X.shape[0] incorrect in prep data"
        assert X.shape[1]+1 == clf.X.shape[1], "X.shape[1] incorrect in prep data"

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    @mark.multinomial_logistic_regression_init_weights
    def test_multinomial_logistic_regression_init_weights(self, get_multinomial_classification_data):
        X, y = get_multinomial_classification_data        
        n_features = X.shape[1]+1
        n_classes = len(np.unique(y))
        clf = MultinomialLogisticRegression(epochs=50)                
        clf.fit(X,y)
        assert clf.theta.shape == (n_features,n_classes), "theta shape incorrect for multi classification"
        
    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    @mark.multinomial_logistic_regression_predict
    def test_multinomial_logistic_regression_predict(self, get_multinomial_classification_data):
        X, y = get_multinomial_classification_data
        clf = MultinomialLogisticRegression(epochs=100, cost='categorical_cross_entropy')
        clf.fit(X,y)
        y_pred = clf._predict(X)
        assert y_pred.shape == (y.shape[0],3), "Shape of prediction is not correct."
        y_pred = clf.predict(X)
        score = clf.score(X,y)
        assert y_pred.shape == (y.shape[0],), "Shape of prediction is not correct."
        assert clf.history.epoch_log.get('train_cost')[0] > clf.history.epoch_log.get('train_cost')[-1], "Training costs didn't decrease"
        assert clf.history.epoch_log.get('train_score')[0] < clf.history.epoch_log.get('train_score')[-1], "Training score didn't increase"
        assert score >= 0.5, "Accuracy below 0.5"
        assert score < 1, "Accuracy is greater than or equal to 1"

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    @mark.multinomial_logistic_regression_early_stop
    def test_multinomial_logistic_regression_early_stop(self, get_multinomial_classification_data):        
        X, y = get_multinomial_classification_data
        es = EarlyStopImprovement(precision=0.001, patience=5)
        clf = MultinomialLogisticRegression(epochs=100, early_stop=es, checkpoint=10)
        clf.fit(X, y)       
        # Confirm early stop happened
        assert clf.history.total_epochs < clf.epochs, "Early stop didn't happen."
        # Test epoch history
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('val_cost')), "number of val costs in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('train_score')), "number of train score in log doesn't match epochs"        
        assert clf.history.total_epochs == len(clf.history.epoch_log.get('val_score')), "number of val score in log doesn't match epochs"        
        assert np.array_equal(clf.theta, clf.history.epoch_log.get('theta')[-1]) == True, "Last theta in log doesn't equal final theta."
        # Test Performance Trends
        assert clf.history.epoch_log.get('train_cost')[0] > clf.history.epoch_log.get('train_cost')[-1], "Training costs didn't decrease"
        #assert clf.history.epoch_log.get('train_score')[0] < clf.history.epoch_log.get('train_score')[-1], "Training score didn't increase"
        assert clf.history.epoch_log.get('val_cost')[0] > clf.history.epoch_log.get('val_cost')[-1], "Validation costs didn't decrease"
        #assert clf.history.epoch_log.get('val_score')[0] < clf.history.epoch_log.get('val_score')[-1], "Validation score didn't increase"
        # Test batch history
        assert clf.history.total_batches == len(clf.history.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert clf.history.total_batches == len(clf.history.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert clf.history.total_batches == len(clf.history.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert clf.history.total_batches == len(clf.history.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"                  
                   
    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    @mark.multinomial_logistic_regression_learning_rate_schedules
    def test_multinomial_logistic_regression_learning_rate_schedules(self, learning_rate_schedules, get_multinomial_classification_data):        
        X, y = get_multinomial_classification_data        
        clf = MultinomialLogisticRegression(epochs=50, checkpoint=10, learning_rate=learning_rate_schedules)
        clf.fit(X, y)       
        # Confirm learning rates decreased
        assert clf.history.epoch_log.get('learning_rate')[0] > clf.history.epoch_log.get('learning_rate')[-1], "Learning rate didn't decrease"
        assert clf.history.epoch_log.get('learning_rate')[0] != clf.eta, "Learning rate didn't change"

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    def test_multinomial_logistic_regression_early_stop_from_estimator_train_cost(self, get_multinomial_classification_data): 
        X, y = get_multinomial_classification_data        
        clf = MultinomialLogisticRegression(epochs=5000, early_stop=False, val_size=0.3, metric=None)
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'train_cost', "Estimator is not sending correct metric"

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    def test_multinomial_logistic_regression_early_stop_from_estimator_train_score(self, get_multinomial_classification_data): 
        X, y = get_multinomial_classification_data        
        clf = MultinomialLogisticRegression(epochs=5000, early_stop=False, val_size=0.3, metric='accuracy')
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'train_score', "Estimator is not sending correct metric"

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    def test_multinomial_logistic_regression_early_stop_from_estimator_val_cost(self, get_multinomial_classification_data): 
        X, y = get_multinomial_classification_data        
        clf = MultinomialLogisticRegression(epochs=5000, early_stop=True, val_size=0.3, metric=None)
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'val_cost', "Estimator is not sending correct metric"        

    @mark.logistic_regression
    @mark.multinomial_logistic_regression
    def test_multinomial_logistic_regression_early_stop_from_estimator_val_score(self, get_multinomial_classification_data): 
        X, y = get_multinomial_classification_data        
        clf = MultinomialLogisticRegression(epochs=5000, early_stop=True, val_size=0.3, metric='accuracy')
        clf.fit(X, y)
        assert clf.convergence_monitor.metric == 'val_score', "Estimator is not sending correct metric"                