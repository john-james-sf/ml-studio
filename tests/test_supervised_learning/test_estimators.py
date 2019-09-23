# --------------------------------------------------------------------------- #
#                          TEST GRADIENT DESCENT                              #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.supervised_learning.regression import RidgeRegression
from ml_studio.supervised_learning.regression import ElasticNetRegression


from ml_studio.operations.callbacks import Callback
from ml_studio.operations.cost import Cost, Quadratic, BinaryCrossEntropy
from ml_studio.operations.cost import CategoricalCrossEntropy
from ml_studio.operations.metrics import Metric
from ml_studio.operations.early_stop import EarlyStopPlateau

class EstimatorTests:

    @mark.estimator
    @mark.estimator_val
    def test_estimator_validation(self, estimator, get_regression_data):

        X, y = get_regression_data
        with pytest.raises(TypeError):
            est = estimator(learning_rate="x")
            est.fit(X, y)        
        with pytest.raises(TypeError):
            est = estimator(batch_size='k')            
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = estimator(theta_init='k')            
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = estimator(epochs='k')           
            est.fit(X, y)
        with pytest.raises(ValueError):
            est = estimator(cost='x')                                
            est.fit(X, y) 
        with pytest.raises(ValueError):
            est = estimator(cost=None)                                
            est.fit(X, y)                  
        with pytest.raises(TypeError):
            est = estimator(early_stop='x')                                
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = estimator(metric=0)                                
            est.fit(X, y)             
        with pytest.raises(ValueError):
            est = estimator(metric='x')                                
            est.fit(X, y) 
        with pytest.raises(TypeError):
            est = estimator(epochs=10,verbose=None)                                                                                      
            est.fit(X, y)
        with pytest.raises(ValueError):
            est = estimator(epochs=10,checkpoint=-1)
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = estimator(epochs=10,checkpoint='x')
            est.fit(X, y)  
        with pytest.warns(UserWarning):
            est = estimator(epochs=10,checkpoint=100)
            est.fit(X, y)                        
        with pytest.raises(TypeError):
            est = estimator(epochs=10,seed='k')                                
            est.fit(X, y)
                       

    @mark.estimator
    @mark.estimator_get_params
    def test_estimator_get_params(self, estimator):
        est = estimator(learning_rate=0.01, theta_init=np.array([2,2,2]),
                             epochs=10, cost='quadratic', 
                             verbose=False, checkpoint=100, 
                             name=None, seed=50)
        params = est.get_params()
        assert params['learning_rate'] == 0.01, "learning rate is invalid" 
        assert all(np.equal(params['theta_init'], np.array([2,2,2]))) , "theta_init is invalid"
        assert params['epochs'] == 10, "epochs is invalid"
        assert params['cost'] == 'quadratic', "cost is invalid"
        assert params['verbose'] == False, "verbose is invalid"
        assert params['checkpoint'] == 100, "checkpoint is invalid"
        assert params['seed'] == 50, "seed is invalid"

    @mark.estimator
    @mark.estimator_validate_data
    def test_estimator_validate_data(self, estimator, get_regression_data):
        est = estimator(epochs=10)
        X, y = get_regression_data        
        with pytest.raises(TypeError):
            est.fit([1,2,3], y)
        with pytest.raises(TypeError):
            est.fit(X, [1,2,3])            
        with pytest.raises(ValueError):
            est.fit(X, y[0:5])            

    @mark.estimator
    @mark.estimator_prepare_data
    def test_estimator_prepare_data_no_val_set(self, estimator, get_regression_data):
        est = estimator(epochs=10)
        X, y = get_regression_data                
        est.fit(X,y)
        assert X.shape[1] == est.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] == est.X.shape[0], "X.shape[0] changed in prepare data"
        assert y.shape == est.y.shape, "y shape changed in prepare data" 

    @mark.estimator
    @mark.estimator_prepare_data
    def test_estimator_prepare_data_w_val_set(self, estimator, get_regression_data):                
        stop = EarlyStopPlateau(val_size=0.2)
        est = estimator(epochs=10, early_stop=stop)
        X, y = get_regression_data                
        est.fit(X,y)
        assert X.shape[1] == est.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] != est.X.shape[0], "X.shape[0] not changed in prepare data validation split"
        assert y.shape != est.y.shape, "y shape not changed in prepare data validation split" 
        assert est.X_val is not None, "X_val is None, not created in prepare data."
        assert est.y_val is not None, "y_val is None, not created in prepare data."
        assert est.X.shape[0] + est.X_val.shape[0] == X.shape[0], "X.shape[0] plus X_val.shape[0] doesn't match input shape"
        assert est.y.shape[0] + est.y_val.shape[0] == y.shape[0], "y.shape[0] plus y_val.shape[0] doesn't match output shape"

    @mark.estimator
    @mark.estimator_compile
    def test_estimator_compile(self, estimator, get_regression_data):
        X, y = get_regression_data        
        est = estimator(epochs=10)
        est.fit(X,y)        
        assert isinstance(est.cost_function, Quadratic), "cost function is not an instance of valid Cost subclass."
        assert isinstance(est.scorer, Metric), "scorer function is not an instance of a valid Metric subclass."
        assert isinstance(est.history, Callback), "history function is not an instance of a valid Callback subclass."
        assert isinstance(est.progress, Callback), "progress function is not an instance of a valid Callback subclass."

    @mark.estimator
    @mark.estimator_init_weights
    def test_estimator_init_weights_shape_match(self, estimator, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1]+1)        
        est = estimator(epochs=10, theta_init=theta_init)
        est.fit(X,y)
        assert not all(np.equal(est.theta, est.theta_init)), "final and initial thetas are equal" 

    @mark.estimator
    @mark.estimator_init_weights
    def test_estimator_init_weights_shape_mismatch(self, estimator, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1])        
        est = estimator(epochs=10, theta_init=theta_init)
        with pytest.raises(ValueError):
            est.fit(X,y)

    @mark.estimator
    @mark.estimator_learning_rate
    def test_estimator_fit_learning_rate_constant(self, estimator, get_regression_data):
        X, y = get_regression_data        
        est = estimator(learning_rate = 0.1, epochs=10)
        est.fit(X,y)
        assert est.learning_rate == 0.1, "learning rate not initialized correctly"
        assert est.history.epoch_log['learning_rate'][0]==\
            est.history.epoch_log['learning_rate'][-1], "learning rate not constant in history"

    @mark.estimator
    @mark.estimator_learning_rate
    def test_estimator_fit_learning_rate_sched(self, estimator, get_regression_data,
                                                      learning_rate_schedules):
        X, y = get_regression_data        
        lrs = learning_rate_schedules
        est = estimator(learning_rate = lrs, epochs=100)
        est.fit(X,y)
        assert est.history.epoch_log['learning_rate'][0] !=\
            est.history.epoch_log['learning_rate'][-1], "learning rate not changed in history"

    @mark.estimator
    @mark.estimator_batch_size
    def test_estimator_fit_batch_size(self, estimator, get_regression_data):
        X, y = get_regression_data         
        X = X[0:33]
        y = y[0:33]       
        est = estimator(batch_size=32, epochs=10)
        est.fit(X,y)                
        assert est.history.total_epochs == 10, "total epochs in history not correct"
        assert est.history.total_batches == 20, "total batches in history not correct"
        assert est.history.total_epochs != est.history.total_batches, "batches and epochs are equal"
        assert est.history.batch_log['batch_size'][0]==32, "batch size not correct in history"
        assert est.history.batch_log['batch_size'][1]!=32, "batch size not correct in history"
        assert len(est.history.batch_log['batch_size']) ==20, "length of batch log incorrect"
        assert len(est.history.epoch_log['learning_rate'])==10, "length of epoch log incorrect"


    @mark.estimator
    @mark.estimator_epochs
    def test_estimator_fit_epochs(self, estimator, get_regression_data):
        X, y = get_regression_data                
        est = estimator(epochs=10)
        est.fit(X,y)
        assert est.epochs == 10, "estimator epochs invalid"
        assert est.history.total_epochs == 10, "total epochs in history not valid"
        assert len(est.history.epoch_log['learning_rate']) == 10, "epoch log not equal to epochs"

    @mark.estimator
    @mark.estimator_early_stop
    def test_estimator_fit_early_stop(self, estimator, get_regression_data,
                                                     early_stop):
        stop = early_stop
        X, y = get_regression_data                
        est = estimator(learning_rate=0.5, epochs=5000, early_stop=stop)
        est.fit(X,y)
        assert est.history.total_epochs < 5000, "didn't stop early"
        assert len(est.history.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"
    
    @mark.estimator
    @mark.estimator_predict
    def test_estimator_predict(self, estimator, get_regression_data):
        X, y = get_regression_data                
        est = estimator(learning_rate = 0.1, epochs=1000)
        with pytest.raises(Exception): # Tests predict w/o first fitting model
            y_pred = est.predict(X)
        est.fit(X, y)
        with pytest.raises(TypeError):
            y_pred = est.predict([1,2,3])
        with pytest.raises(ValueError):            
            y_pred = est.predict(np.reshape(X, (-1,1)))        
        y_pred = est.predict(X)
        assert all(np.equal(y.shape, y_pred.shape)), "y and y_pred have different shapes"  

    @mark.estimator
    @mark.estimator_score
    def test_estimator_score(self, estimator, get_regression_data_w_validation, 
                                    regression_metric):
        X, X_test, y, y_test = get_regression_data_w_validation                
        est = estimator(learning_rate = 0.1, epochs=1000, metric=regression_metric)
        with pytest.raises(Exception):
            score = est.score(X, y)
        est.fit(X, y)
        with pytest.raises(TypeError):
            score = est.score("X", y)
        with pytest.raises(TypeError):
            score = est.score(X, [1,2,3])        
        with pytest.raises(ValueError):
            score = est.score(X, np.array([1,2,3]))        
        with pytest.raises(ValueError):
            score = est.score(np.reshape(X, (-1,1)), y)    
        # Model evaluation 
        score = est.score(X_test, y_test)
        assert isinstance(score, (int,float)), "score is not an int nor a float"   

    @mark.estimator
    @mark.estimator_history
    def test_estimator_history_no_val_data_no_metric(self, estimator, get_regression_data):        
        X, y = get_regression_data        
        est = estimator(epochs=10, metric=None)
        est.fit(X, y)        
        # Test epoch history
        assert est.history.total_epochs == len(est.history.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert all(np.equal(est.theta, est.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert est.history.epoch_log.get('train_cost')[0] > est.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history.epoch_log.get("train_score", None) is None, "train score without metric is not None"
        assert est.history.epoch_log.get("val_cost", None) is None, "val cost without early stopping is not None"
        assert est.history.epoch_log.get("val_score", None) is None, "val score without early stopping is not None"
        # Test batch history
        assert est.history.total_batches == len(est.history.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.estimator
    @mark.estimator_history
    def test_estimator_history_w_val_data_no_metric(self, estimator, get_regression_data):        
        X, y = get_regression_data     
        stop = EarlyStopPlateau()   
        est = estimator(epochs=10, metric=None, early_stop=stop)
        est.fit(X, y)        
        # Test epoch history
        assert est.history.total_epochs == len(est.history.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('val_cost')), "number of val costs in log doesn't match epochs"        
        assert all(np.equal(est.theta, est.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert est.history.epoch_log.get('train_cost')[0] > est.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history.epoch_log.get("train_score", None) is None, "train score without metric is not None"
        assert est.history.epoch_log.get("val_score", None) is None, "val score without early stopping is not None"
        # Test batch history
        assert est.history.total_batches == len(est.history.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.estimator
    @mark.estimator_history
    def test_estimator_history_no_val_data_w_metric(self, estimator, get_regression_data):        
        X, y = get_regression_data        
        est = estimator(epochs=10, metric='mean_squared_error')
        est.fit(X, y)        
        # Test epoch history
        assert est.history.total_epochs == len(est.history.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('train_score')), "number of train scores in log doesn't match epochs"        
        assert all(np.equal(est.theta, est.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert est.history.epoch_log.get('train_cost')[0] > est.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history.epoch_log.get('train_score')[0] > est.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert est.history.epoch_log.get("val_cost", None) is None, "val cost without early stopping is not None"
        assert est.history.epoch_log.get("val_score", None) is None, "val score without early stopping is not None"
        # Test batch history
        assert est.history.total_batches == len(est.history.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.estimator
    @mark.estimator_history
    def test_estimator_history_w_val_data_w_metric(self, estimator, get_regression_data):        
        X, y = get_regression_data     
        stop = EarlyStopPlateau()   
        est = estimator(epochs=10, metric='mean_squared_error', early_stop=stop)
        est.fit(X, y)        
        # Test epoch history
        assert est.history.total_epochs == len(est.history.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('val_cost')), "number of val costs in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('train_score')), "number of train score in log doesn't match epochs"        
        assert est.history.total_epochs == len(est.history.epoch_log.get('val_score')), "number of val score in log doesn't match epochs"        
        assert all(np.equal(est.theta, est.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert est.history.epoch_log.get('train_cost')[0] > est.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history.epoch_log.get('train_score')[0] > est.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert est.history.epoch_log.get('val_cost')[0] > est.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        assert est.history.epoch_log.get('val_score')[0] > est.history.epoch_log.get('val_score')[-1], "val_score does not decrease"        
        # Test batch history
        assert est.history.total_batches == len(est.history.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history.total_batches == len(est.history.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.estimator
    @mark.linear_regression
    def test_linear_regression_name(self):        
        est = LinearRegression()
        assert est.name == "Linear Regression with Batch Gradient Descent", "incorrect name"

    @mark.estimator
    @mark.lasso_regression
    def test_lasso_regression_name(self):        
        est = LassoRegression()
        assert est.name == "Lasso Regression with Batch Gradient Descent", "incorrect name"

    @mark.estimator
    @mark.ridge_regression
    def test_ridge_regression_name(self):        
        est = RidgeRegression()
        assert est.name == "Ridge Regression with Batch Gradient Descent", "incorrect name"        

    @mark.estimator
    @mark.elasticnet_regression
    def test_elasticnet_regression_name(self):        
        est = ElasticNetRegression()
        assert est.name == "ElasticNet Regression with Batch Gradient Descent", "incorrect name"        