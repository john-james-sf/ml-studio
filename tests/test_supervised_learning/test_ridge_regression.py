# --------------------------------------------------------------------------- #
#                          TEST GRADIENT DESCENT                              #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.regression import RidgeRegression
from ml_studio.operations.callbacks import Callback
from ml_studio.operations.cost import Cost, Quadratic, BinaryCrossEntropy
from ml_studio.operations.cost import CategoricalCrossEntropy
from ml_studio.operations.metrics import Metric
from ml_studio.operations.early_stop import EarlyStopPlateau

class RidgeRegressionTests:

    @mark.ridge_regression
    @mark.ridge_regression_val
    def test_ridge_regression_validation(self, get_regression_data):

        X, y = get_regression_data
        with pytest.raises(TypeError):
            rr = RidgeRegression(learning_rate="x")
            rr.fit(X, y)        
        with pytest.raises(TypeError):
            rr = RidgeRegression(batch_size='k')            
            rr.fit(X, y)
        with pytest.raises(TypeError):
            rr = RidgeRegression(theta_init='k')            
            rr.fit(X, y)
        with pytest.raises(TypeError):
            rr = RidgeRegression(epochs='k')           
            rr.fit(X, y)
        with pytest.raises(ValueError):
            rr = RidgeRegression(cost='x')                                
            rr.fit(X, y) 
        with pytest.raises(ValueError):
            rr = RidgeRegression(cost=None)                                
            rr.fit(X, y)                  
        with pytest.raises(TypeError):
            rr = RidgeRegression(early_stop='x')                                
            rr.fit(X, y)
        with pytest.raises(TypeError):
            rr = RidgeRegression(metric=0)                                
            rr.fit(X, y)             
        with pytest.raises(ValueError):
            rr = RidgeRegression(metric='x')                                
            rr.fit(X, y) 
        with pytest.raises(ValueError):
            rr = RidgeRegression(early_stop=EarlyStopPlateau(), metric=None)                                
            rr.fit(X, y)                                                             
        with pytest.raises(TypeError):
            rr = RidgeRegression(epochs=10,verbose=None)                                                                                      
            rr.fit(X, y)
        with pytest.raises(ValueError):
            rr = RidgeRegression(epochs=10,checkpoint=-1)
            rr.fit(X, y)
        with pytest.raises(TypeError):
            rr = RidgeRegression(epochs=10,checkpoint='x')
            rr.fit(X, y)  
        with pytest.warns(UserWarning):
            rr = RidgeRegression(epochs=10,checkpoint=100)
            rr.fit(X, y)                        
        with pytest.raises(TypeError):
            rr = RidgeRegression(epochs=10,seed='k')                                
            rr.fit(X, y)
                       

    @mark.ridge_regression
    @mark.ridge_regression_get_params
    def test_ridge_regression_get_params(self):
        rr = RidgeRegression(learning_rate=0.01, theta_init=np.array([2,2,2]),
                             epochs=10, cost='quadratic', 
                             verbose=False, checkpoint=100, 
                             name=None, seed=50)
        params = rr.get_params()
        assert params['learning_rate'] == 0.01, "learning rate is invalid" 
        assert all(np.equal(params['theta_init'], np.array([2,2,2]))) , "theta_init is invalid"
        assert params['epochs'] == 10, "epochs is invalid"
        assert params['cost'] == 'quadratic', "cost is invalid"
        assert params['verbose'] == False, "verbose is invalid"
        assert params['checkpoint'] == 100, "checkpoint is invalid"
        assert params['name'] == "Ridge Regression with Batch Gradient Descent", "name is invalid"        
        assert params['seed'] == 50, "seed is invalid"

    @mark.ridge_regression
    @mark.ridge_regression_validate_data
    def test_ridge_regression_validate_data(self, get_regression_data):
        rr = RidgeRegression(epochs=10)
        X, y = get_regression_data        
        with pytest.raises(TypeError):
            rr.fit([1,2,3], y)
        with pytest.raises(TypeError):
            rr.fit(X, [1,2,3])            
        with pytest.raises(ValueError):
            rr.fit(X, y[0:5])            

    @mark.ridge_regression
    @mark.ridge_regression_prepare_data
    def test_ridge_regression_prepare_data_no_val_set(self, get_regression_data):
        rr = RidgeRegression(epochs=10)
        X, y = get_regression_data                
        rr.fit(X,y)
        assert X.shape[1] == rr.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] == rr.X.shape[0], "X.shape[0] changed in prepare data"
        assert y.shape == rr.y.shape, "y shape changed in prepare data" 

    @mark.ridge_regression
    @mark.ridge_regression_prepare_data
    def test_ridge_regression_prepare_data_w_val_set(self, get_regression_data):                
        es = EarlyStopPlateau(val_size=0.2)
        rr = RidgeRegression(epochs=10, early_stop=es)
        X, y = get_regression_data                
        rr.fit(X,y)
        assert X.shape[1] == rr.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] != rr.X.shape[0], "X.shape[0] not changed in prepare data validation split"
        assert y.shape != rr.y.shape, "y shape not changed in prepare data validation split" 
        assert rr.X_val is not None, "X_val is None, not created in prepare data."
        assert rr.y_val is not None, "y_val is None, not created in prepare data."
        assert rr.X.shape[0] + rr.X_val.shape[0] == X.shape[0], "X.shape[0] plus X_val.shape[0] doesn't match input shape"
        assert rr.y.shape[0] + rr.y_val.shape[0] == y.shape[0], "y.shape[0] plus y_val.shape[0] doesn't match output shape"

    @mark.ridge_regression
    @mark.ridge_regression_compile
    def test_ridge_regression_compile(self, get_regression_data):
        X, y = get_regression_data        
        rr = RidgeRegression(epochs=10)
        rr.fit(X,y)        
        assert isinstance(rr.cost_function, Quadratic), "cost function is not an instance of valid Cost subclass."
        assert isinstance(rr.scorer, Metric), "scorer function is not an instance of a valid Metric subclass."
        assert isinstance(rr.history, Callback), "history function is not an instance of a valid Callback subclass."
        assert isinstance(rr.progress, Callback), "progress function is not an instance of a valid Callback subclass."

    @mark.ridge_regression
    @mark.ridge_regression_init_weights
    def test_ridge_regression_init_weights_shape_match(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1]+1)        
        rr = RidgeRegression(epochs=10, theta_init=theta_init)
        rr.fit(X,y)
        assert not all(np.equal(rr.theta, rr.theta_init)), "final and initial thetas are equal" 

    @mark.ridge_regression
    @mark.ridge_regression_init_weights
    def test_ridge_regression_init_weights_shape_mismatch(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1])        
        rr = RidgeRegression(epochs=10, theta_init=theta_init)
        with pytest.raises(ValueError):
            rr.fit(X,y)

    @mark.ridge_regression
    @mark.ridge_regression_learning_rate
    def test_ridge_regression_fit_learning_rate_constant(self, get_regression_data):
        X, y = get_regression_data        
        rr = RidgeRegression(learning_rate = 0.1, epochs=10)
        rr.fit(X,y)
        assert rr.learning_rate == 0.1, "learning rate not initialized correctly"
        assert rr.history.epoch_log['learning_rate'][0]==\
            rr.history.epoch_log['learning_rate'][-1], "learning rate not constant in history"

    @mark.ridge_regression
    @mark.ridge_regression_learning_rate
    def test_ridge_regression_fit_learning_rate_sched(self, get_regression_data,
                                                      learning_rate_schedules):
        X, y = get_regression_data        
        rrs = learning_rate_schedules
        rr = RidgeRegression(learning_rate = rrs, epochs=500)
        rr.fit(X,y)
        assert rr.history.epoch_log['learning_rate'][0] !=\
            rr.history.epoch_log['learning_rate'][-1], "learning rate not changed in history"

    @mark.ridge_regression
    @mark.ridge_regression_batch_size
    def test_ridge_regression_fit_batch_size(self, get_regression_data):
        X, y = get_regression_data         
        X = X[0:33]
        y = y[0:33]       
        rr = RidgeRegression(batch_size=32, epochs=10)
        rr.fit(X,y)                
        assert rr.history.total_epochs == 10, "total epochs in history not correct"
        assert rr.history.total_batches == 20, "total batches in history not correct"
        assert rr.history.total_epochs != rr.history.total_batches, "batches and epochs are equal"
        assert rr.history.batch_log['batch_size'][0]==32, "batch size not correct in history"
        assert len(rr.history.batch_log['batch_size']) ==20, "length of batch log incorrect"
        assert len(rr.history.epoch_log['learning_rate'])==10, "length of epoch log incorrect"


    @mark.ridge_regression
    @mark.ridge_regression_epochs
    def test_ridge_regression_fit_epochs(self, get_regression_data):
        X, y = get_regression_data                
        rr = RidgeRegression(epochs=10)
        rr.fit(X,y)
        assert rr.epochs == 10, "estimator epochs invalid"
        assert rr.history.total_epochs == 10, "total epochs in history not valid"
        assert len(rr.history.epoch_log['learning_rate']) == 10, "epoch log not equal to epochs"

    @mark.ridge_regression
    @mark.ridge_regression_early_stop
    def test_ridge_regression_fit_early_stop(self, get_regression_data,
                                                     early_stop):
        es = early_stop
        X, y = get_regression_data                
        rr = RidgeRegression(learning_rate=0.5, epochs=5000, early_stop=es)
        rr.fit(X,y)
        assert rr.history.total_epochs < 5000, "didn't stop early"
        assert len(rr.history.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"
    
    @mark.ridge_regression
    @mark.ridge_regression_predict
    def test_ridge_regression_predict(self, get_regression_data):
        X, y = get_regression_data                
        rr = RidgeRegression(learning_rate = 0.1, epochs=1000)
        with pytest.raises(Exception): # Tests predict w/o first fitting model
            y_pred = rr.predict(X)
        rr.fit(X, y)
        with pytest.raises(TypeError):
            y_pred = rr.predict([1,2,3])
        with pytest.raises(ValueError):            
            y_pred = rr.predict(np.reshape(X, (-1,1)))        
        y_pred = rr.predict(X)
        assert all(np.equal(y.shape, y_pred.shape)), "y and y_pred have different shapes"  

    @mark.ridge_regression
    @mark.ridge_regression_score
    def test_ridge_regression_score(self, get_regression_data_w_validation, 
                                    regression_metric):
        X, X_test, y, y_test = get_regression_data_w_validation                
        rr = RidgeRegression(learning_rate = 0.1, epochs=1000, metric=regression_metric)
        with pytest.raises(Exception):
            score = rr.score(X, y)
        rr.fit(X, y)
        with pytest.raises(TypeError):
            score = rr.score("X", y)
        with pytest.raises(TypeError):
            score = rr.score(X, [1,2,3])        
        with pytest.raises(ValueError):
            score = rr.score(X, np.array([1,2,3]))        
        with pytest.raises(ValueError):
            score = rr.score(np.reshape(X, (-1,1)), y)    
        score = rr.score(X_test, y_test)
        assert isinstance(score, (int,float)), "score is not an int nor a float"   

    @mark.ridge_regression
    @mark.ridge_regression_history
    def test_ridge_regression_history_no_val_data(self, get_regression_data):        
        X, y = get_regression_data        
        rr = RidgeRegression(epochs=10)
        rr.fit(X, y)        
        # Test epoch history
        assert rr.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(rr.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert (len(rr.history.epoch_log.get("learning_rate")) == 10), "length of learning rate doesn't match epochs"
        assert len(rr.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(rr.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert rr.history.epoch_log.get('train_cost')[0] > rr.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert rr.history.epoch_log.get('train_score')[0] > rr.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert all(np.equal(rr.theta, rr.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."


    @mark.ridge_regression
    @mark.ridge_regression_history
    def test_ridge_regression_history_w_val_data(self, get_regression_data):        
        X, y = get_regression_data
        es = EarlyStopPlateau(precision=0.1)        
        rr = RidgeRegression(epochs=100, metric='mean_squared_error', early_stop=es)
        rr.fit(X, y)        
        # Test epoch history
        assert rr.history.total_epochs < 100, "total_epochs from history doesn't match epochs"
        assert len(rr.history.epoch_log.get('epoch')) < 100, "number of epochs in log doesn't match epochs"
        assert (len(rr.history.epoch_log.get("learning_rate")) < 100), "length of learning rate doesn't match epochs"
        assert len(rr.history.epoch_log.get('theta')) < 100, "number of thetas in log doesn't match epochs"
        assert all(np.equal(rr.theta, rr.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert len(rr.history.epoch_log.get('train_cost')) < 100, "length of train_cost doesn't match epochs"
        assert rr.history.epoch_log.get('train_cost')[0] > rr.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert rr.history.epoch_log.get('train_score')[0] > rr.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert len(rr.history.epoch_log.get('val_cost')) < 100, "length of val_cost doesn't match epochs"
        assert rr.history.epoch_log.get('val_cost')[0] > rr.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        assert rr.history.epoch_log.get('val_score')[0] > rr.history.epoch_log.get('val_score')[-1], "val_score does not decrease"
