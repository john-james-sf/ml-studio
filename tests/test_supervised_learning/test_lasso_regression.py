# --------------------------------------------------------------------------- #
#                          TEST GRADIENT DESCENT                              #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.operations.callbacks import Callback
from ml_studio.operations.cost import Cost, Quadratic, BinaryCrossEntropy
from ml_studio.operations.cost import CategoricalCrossEntropy
from ml_studio.operations.metrics import Metric
from ml_studio.operations.early_stop import EarlyStopPlateau

class LassoRegressionTests:

    @mark.lasso_regression
    @mark.lasso_regression_val
    def test_lasso_regression_validation(self, get_regression_data):

        X, y = get_regression_data
        with pytest.raises(TypeError):
            lr = LassoRegression(learning_rate="x")
            lr.fit(X, y)        
        with pytest.raises(TypeError):
            lr = LassoRegression(batch_size='k')            
            lr.fit(X, y)
        with pytest.raises(TypeError):
            lr = LassoRegression(theta_init='k')            
            lr.fit(X, y)
        with pytest.raises(TypeError):
            lr = LassoRegression(epochs='k')           
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LassoRegression(cost='x')                                
            lr.fit(X, y) 
        with pytest.raises(ValueError):
            lr = LassoRegression(cost=None)                                
            lr.fit(X, y)                  
        with pytest.raises(TypeError):
            lr = LassoRegression(early_stop='x')                                
            lr.fit(X, y)
        with pytest.raises(TypeError):
            lr = LassoRegression(metric=0)                                
            lr.fit(X, y)             
        with pytest.raises(ValueError):
            lr = LassoRegression(metric='x')                                
            lr.fit(X, y) 
        with pytest.raises(ValueError):
            lr = LassoRegression(early_stop=EarlyStopPlateau(), metric=None)                                
            lr.fit(X, y)                                                             
        with pytest.raises(TypeError):
            lr = LassoRegression(epochs=10,verbose=None)                                                                                      
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LassoRegression(epochs=10,checkpoint=-1)
            lr.fit(X, y)
        with pytest.raises(TypeError):
            lr = LassoRegression(epochs=10,checkpoint='x')
            lr.fit(X, y)  
        with pytest.warns(UserWarning):
            lr = LassoRegression(epochs=10,checkpoint=100)
            lr.fit(X, y)                        
        with pytest.raises(TypeError):
            lr = LassoRegression(epochs=10,seed='k')                                
            lr.fit(X, y)
                       

    @mark.lasso_regression
    @mark.lasso_regression_get_params
    def test_lasso_regression_get_params(self):
        lr = LassoRegression(learning_rate=0.01, theta_init=np.array([2,2,2]),
                             epochs=10, cost='quadratic', 
                             verbose=False, checkpoint=100, 
                             name=None, seed=50)
        params = lr.get_params()
        assert params['learning_rate'] == 0.01, "learning rate is invalid" 
        assert all(np.equal(params['theta_init'], np.array([2,2,2]))) , "theta_init is invalid"
        assert params['epochs'] == 10, "epochs is invalid"
        assert params['cost'] == 'quadratic', "cost is invalid"
        assert params['verbose'] == False, "verbose is invalid"
        assert params['checkpoint'] == 100, "checkpoint is invalid"
        assert params['name'] == "Lasso Regression with Batch Gradient Descent", "name is invalid"        
        assert params['seed'] == 50, "seed is invalid"

    @mark.lasso_regression
    @mark.lasso_regression_validate_data
    def test_lasso_regression_validate_data(self, get_regression_data):
        lr = LassoRegression(epochs=10)
        X, y = get_regression_data        
        with pytest.raises(TypeError):
            lr.fit([1,2,3], y)
        with pytest.raises(TypeError):
            lr.fit(X, [1,2,3])            
        with pytest.raises(ValueError):
            lr.fit(X, y[0:5])            

    @mark.lasso_regression
    @mark.lasso_regression_prepare_data
    def test_lasso_regression_prepare_data_no_val_set(self, get_regression_data):
        lr = LassoRegression(epochs=10)
        X, y = get_regression_data                
        lr.fit(X,y)
        assert X.shape[1] == lr.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] == lr.X.shape[0], "X.shape[0] changed in prepare data"
        assert y.shape == lr.y.shape, "y shape changed in prepare data" 

    @mark.lasso_regression
    @mark.lasso_regression_prepare_data
    def test_lasso_regression_prepare_data_w_val_set(self, get_regression_data):                
        es = EarlyStopPlateau(val_size=0.2)
        lr = LassoRegression(epochs=10, early_stop=es)
        X, y = get_regression_data                
        lr.fit(X,y)
        assert X.shape[1] == lr.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] != lr.X.shape[0], "X.shape[0] not changed in prepare data validation split"
        assert y.shape != lr.y.shape, "y shape not changed in prepare data validation split" 
        assert lr.X_val is not None, "X_val is None, not created in prepare data."
        assert lr.y_val is not None, "y_val is None, not created in prepare data."
        assert lr.X.shape[0] + lr.X_val.shape[0] == X.shape[0], "X.shape[0] plus X_val.shape[0] doesn't match input shape"
        assert lr.y.shape[0] + lr.y_val.shape[0] == y.shape[0], "y.shape[0] plus y_val.shape[0] doesn't match output shape"

    @mark.lasso_regression
    @mark.lasso_regression_compile
    def test_lasso_regression_compile(self, get_regression_data):
        X, y = get_regression_data        
        lr = LassoRegression(epochs=10)
        lr.fit(X,y)        
        assert isinstance(lr.cost_function, Quadratic), "cost function is not an instance of valid Cost subclass."
        assert isinstance(lr.scorer, Metric), "scorer function is not an instance of a valid Metric subclass."
        assert isinstance(lr.history, Callback), "history function is not an instance of a valid Callback subclass."
        assert isinstance(lr.progress, Callback), "progress function is not an instance of a valid Callback subclass."

    @mark.lasso_regression
    @mark.lasso_regression_init_weights
    def test_lasso_regression_init_weights_shape_match(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1]+1)        
        lr = LassoRegression(epochs=10, theta_init=theta_init)
        lr.fit(X,y)
        assert not all(np.equal(lr.theta, lr.theta_init)), "final and initial thetas are equal" 

    @mark.lasso_regression
    @mark.lasso_regression_init_weights
    def test_lasso_regression_init_weights_shape_mismatch(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1])        
        lr = LassoRegression(epochs=10, theta_init=theta_init)
        with pytest.raises(ValueError):
            lr.fit(X,y)

    @mark.lasso_regression
    @mark.lasso_regression_learning_rate
    def test_lasso_regression_fit_learning_rate_constant(self, get_regression_data):
        X, y = get_regression_data        
        lr = LassoRegression(learning_rate = 0.1, epochs=10)
        lr.fit(X,y)
        assert lr.learning_rate == 0.1, "learning rate not initialized correctly"
        assert lr.history.epoch_log['learning_rate'][0]==\
            lr.history.epoch_log['learning_rate'][-1], "learning rate not constant in history"

    @mark.lasso_regression
    @mark.lasso_regression_learning_rate
    def test_lasso_regression_fit_learning_rate_sched(self, get_regression_data,
                                                      learning_rate_schedules):
        X, y = get_regression_data        
        lrs = learning_rate_schedules
        lr = LassoRegression(learning_rate = lrs, epochs=500)
        lr.fit(X,y)
        assert lr.history.epoch_log['learning_rate'][0] !=\
            lr.history.epoch_log['learning_rate'][-1], "learning rate not changed in history"

    @mark.lasso_regression
    @mark.lasso_regression_batch_size
    def test_lasso_regression_fit_batch_size(self, get_regression_data):
        X, y = get_regression_data         
        X = X[0:33]
        y = y[0:33]       
        lr = LassoRegression(batch_size=32, epochs=10)
        lr.fit(X,y)                
        assert lr.history.total_epochs == 10, "total epochs in history not correct"
        assert lr.history.total_batches == 20, "total batches in history not correct"
        assert lr.history.total_epochs != lr.history.total_batches, "batches and epochs are equal"
        assert lr.history.batch_log['batch_size'][0]==32, "batch size not correct in history"
        assert len(lr.history.batch_log['batch_size']) ==20, "length of batch log incorrect"
        assert len(lr.history.epoch_log['learning_rate'])==10, "length of epoch log incorrect"


    @mark.lasso_regression
    @mark.lasso_regression_epochs
    def test_lasso_regression_fit_epochs(self, get_regression_data):
        X, y = get_regression_data                
        lr = LassoRegression(epochs=10)
        lr.fit(X,y)
        assert lr.epochs == 10, "estimator epochs invalid"
        assert lr.history.total_epochs == 10, "total epochs in history not valid"
        assert len(lr.history.epoch_log['learning_rate']) == 10, "epoch log not equal to epochs"

    @mark.lasso_regression
    @mark.lasso_regression_early_stop
    def test_lasso_regression_fit_early_stop(self, get_regression_data,
                                                     early_stop):
        es = early_stop
        X, y = get_regression_data                
        lr = LassoRegression(learning_rate=0.5, epochs=5000, early_stop=es)
        lr.fit(X,y)
        assert lr.history.total_epochs < 5000, "didn't stop early"
        assert len(lr.history.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"
    
    @mark.lasso_regression
    @mark.lasso_regression_predict
    def test_lasso_regression_predict(self, get_regression_data):
        X, y = get_regression_data                
        lr = LassoRegression(learning_rate = 0.1, epochs=1000)
        with pytest.raises(Exception): # Tests predict w/o first fitting model
            y_pred = lr.predict(X)
        lr.fit(X, y)
        with pytest.raises(TypeError):
            y_pred = lr.predict([1,2,3])
        with pytest.raises(ValueError):            
            y_pred = lr.predict(np.reshape(X, (-1,1)))        
        y_pred = lr.predict(X)
        assert all(np.equal(y.shape, y_pred.shape)), "y and y_pred have different shapes"  

    @mark.lasso_regression
    @mark.lasso_regression_score
    def test_lasso_regression_score(self, get_regression_data_w_validation, 
                                    regression_metric):
        X, X_test, y, y_test = get_regression_data_w_validation                
        lr = LassoRegression(learning_rate = 0.1, epochs=1000, metric=regression_metric)
        with pytest.raises(Exception):
            score = lr.score(X, y)
        lr.fit(X, y)
        with pytest.raises(TypeError):
            score = lr.score("X", y)
        with pytest.raises(TypeError):
            score = lr.score(X, [1,2,3])        
        with pytest.raises(ValueError):
            score = lr.score(X, np.array([1,2,3]))        
        with pytest.raises(ValueError):
            score = lr.score(np.reshape(X, (-1,1)), y)    
        score = lr.score(X_test, y_test)
        assert isinstance(score, (int,float)), "score is not an int nor a float"   

    @mark.lasso_regression
    @mark.lasso_regression_history
    def test_lasso_regression_history_no_val_data(self, get_regression_data):        
        X, y = get_regression_data        
        lr = LassoRegression(epochs=10)
        lr.fit(X, y)        
        # Test epoch history
        assert lr.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(lr.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert (len(lr.history.epoch_log.get("learning_rate")) == 10), "length of learning rate doesn't match epochs"
        assert len(lr.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(lr.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert lr.history.epoch_log.get('train_cost')[0] > lr.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert lr.history.epoch_log.get('train_score')[0] > lr.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert all(np.equal(lr.theta, lr.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."


    @mark.lasso_regression
    @mark.lasso_regression_history
    def test_lasso_regression_history_w_val_data(self, get_regression_data):        
        X, y = get_regression_data
        es = EarlyStopPlateau(precision=0.1)        
        lr = LassoRegression(epochs=100, metric='mean_squared_error', early_stop=es)
        lr.fit(X, y)        
        # Test epoch history
        assert lr.history.total_epochs < 100, "total_epochs from history doesn't match epochs"
        assert len(lr.history.epoch_log.get('epoch')) < 100, "number of epochs in log doesn't match epochs"
        assert (len(lr.history.epoch_log.get("learning_rate")) < 100), "length of learning rate doesn't match epochs"
        assert len(lr.history.epoch_log.get('theta')) < 100, "number of thetas in log doesn't match epochs"
        assert all(np.equal(lr.theta, lr.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert len(lr.history.epoch_log.get('train_cost')) < 100, "length of train_cost doesn't match epochs"
        assert lr.history.epoch_log.get('train_cost')[0] > lr.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert lr.history.epoch_log.get('train_score')[0] > lr.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert len(lr.history.epoch_log.get('val_cost')) < 100, "length of val_cost doesn't match epochs"
        assert lr.history.epoch_log.get('val_cost')[0] > lr.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        assert lr.history.epoch_log.get('val_score')[0] > lr.history.epoch_log.get('val_score')[-1], "val_score does not decrease"
