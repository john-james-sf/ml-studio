# --------------------------------------------------------------------------- #
#                          TEST GRADIENT DESCENT                              #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.regression import ElasticNetRegression
from ml_studio.operations.callbacks import Callback
from ml_studio.operations.cost import Cost, Quadratic, BinaryCrossEntropy
from ml_studio.operations.cost import CategoricalCrossEntropy
from ml_studio.operations.metrics import Metric
from ml_studio.operations.early_stop import EarlyStopPlateau

class ElasticNetRegressionTests:

    @mark.elasticnet_regression
    @mark.elasticnet_regression_val
    def test_elasticnet_regression_validation(self, get_regression_data):

        X, y = get_regression_data
        with pytest.raises(TypeError):
            en = ElasticNetRegression(learning_rate="x")
            en.fit(X, y)        
        with pytest.raises(TypeError):
            en = ElasticNetRegression(batch_size='k')            
            en.fit(X, y)
        with pytest.raises(TypeError):
            en = ElasticNetRegression(theta_init='k')            
            en.fit(X, y)
        with pytest.raises(TypeError):
            en = ElasticNetRegression(epochs='k')           
            en.fit(X, y)
        with pytest.raises(ValueError):
            en = ElasticNetRegression(cost='x')                                
            en.fit(X, y) 
        with pytest.raises(ValueError):
            en = ElasticNetRegression(cost=None)                                
            en.fit(X, y)                  
        with pytest.raises(TypeError):
            en = ElasticNetRegression(early_stop='x')                                
            en.fit(X, y)
        with pytest.raises(TypeError):
            en = ElasticNetRegression(metric=0)                                
            en.fit(X, y)             
        with pytest.raises(ValueError):
            en = ElasticNetRegression(metric='x')                                
            en.fit(X, y) 
        with pytest.raises(ValueError):
            en = ElasticNetRegression(early_stop=EarlyStopPlateau(), metric=None)                                
            en.fit(X, y)                                                             
        with pytest.raises(TypeError):
            en = ElasticNetRegression(epochs=10,verbose=None)                                                                                      
            en.fit(X, y)
        with pytest.raises(ValueError):
            en = ElasticNetRegression(epochs=10,checkpoint=-1)
            en.fit(X, y)
        with pytest.raises(TypeError):
            en = ElasticNetRegression(epochs=10,checkpoint='x')
            en.fit(X, y)  
        with pytest.warns(UserWarning):
            en = ElasticNetRegression(epochs=10,checkpoint=100)
            en.fit(X, y)                        
        with pytest.raises(TypeError):
            en = ElasticNetRegression(epochs=10,seed='k')                                
            en.fit(X, y)
                       

    @mark.elasticnet_regression
    @mark.elasticnet_regression_get_params
    def test_elasticnet_regression_get_params(self):
        en = ElasticNetRegression(learning_rate=0.01, theta_init=np.array([2,2,2]),
                             epochs=10, cost='quadratic', 
                             verbose=False, checkpoint=100, 
                             name=None, seed=50)
        params = en.get_params()
        assert params['learning_rate'] == 0.01, "learning rate is invalid" 
        assert all(np.equal(params['theta_init'], np.array([2,2,2]))) , "theta_init is invalid"
        assert params['epochs'] == 10, "epochs is invalid"
        assert params['cost'] == 'quadratic', "cost is invalid"
        assert params['verbose'] == False, "verbose is invalid"
        assert params['checkpoint'] == 100, "checkpoint is invalid"
        assert params['name'] == "ElasticNet Regression with Batch Gradient Descent", "name is invalid"        
        assert params['seed'] == 50, "seed is invalid"

    @mark.elasticnet_regression
    @mark.elasticnet_regression_validate_data
    def test_elasticnet_regression_validate_data(self, get_regression_data):
        en = ElasticNetRegression(epochs=10)
        X, y = get_regression_data        
        with pytest.raises(TypeError):
            en.fit([1,2,3], y)
        with pytest.raises(TypeError):
            en.fit(X, [1,2,3])            
        with pytest.raises(ValueError):
            en.fit(X, y[0:5])            

    @mark.elasticnet_regression
    @mark.elasticnet_regression_prepare_data
    def test_elasticnet_regression_prepare_data_no_val_set(self, get_regression_data):
        en = ElasticNetRegression(epochs=10)
        X, y = get_regression_data                
        en.fit(X,y)
        assert X.shape[1] == en.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] == en.X.shape[0], "X.shape[0] changed in prepare data"
        assert y.shape == en.y.shape, "y shape changed in prepare data" 

    @mark.elasticnet_regression
    @mark.elasticnet_regression_prepare_data
    def test_elasticnet_regression_prepare_data_w_val_set(self, get_regression_data):                
        es = EarlyStopPlateau(val_size=0.2)
        en = ElasticNetRegression(epochs=10, early_stop=es)
        X, y = get_regression_data                
        en.fit(X,y)
        assert X.shape[1] == en.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] != en.X.shape[0], "X.shape[0] not changed in prepare data validation split"
        assert y.shape != en.y.shape, "y shape not changed in prepare data validation split" 
        assert en.X_val is not None, "X_val is None, not created in prepare data."
        assert en.y_val is not None, "y_val is None, not created in prepare data."
        assert en.X.shape[0] + en.X_val.shape[0] == X.shape[0], "X.shape[0] plus X_val.shape[0] doesn't match input shape"
        assert en.y.shape[0] + en.y_val.shape[0] == y.shape[0], "y.shape[0] plus y_val.shape[0] doesn't match output shape"

    @mark.elasticnet_regression
    @mark.elasticnet_regression_compile
    def test_elasticnet_regression_compile(self, get_regression_data):
        X, y = get_regression_data        
        en = ElasticNetRegression(epochs=10)
        en.fit(X,y)        
        assert isinstance(en.cost_function, Quadratic), "cost function is not an instance of valid Cost subclass."
        assert isinstance(en.scorer, Metric), "scorer function is not an instance of a valid Metric subclass."
        assert isinstance(en.history, Callback), "history function is not an instance of a valid Callback subclass."
        assert isinstance(en.progress, Callback), "progress function is not an instance of a valid Callback subclass."

    @mark.elasticnet_regression
    @mark.elasticnet_regression_init_weights
    def test_elasticnet_regression_init_weights_shape_match(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1]+1)        
        en = ElasticNetRegression(epochs=10, theta_init=theta_init)
        en.fit(X,y)
        assert not all(np.equal(en.theta, en.theta_init)), "final and initial thetas are equal" 

    @mark.elasticnet_regression
    @mark.elasticnet_regression_init_weights
    def test_elasticnet_regression_init_weights_shape_mismatch(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1])        
        en = ElasticNetRegression(epochs=10, theta_init=theta_init)
        with pytest.raises(ValueError):
            en.fit(X,y)

    @mark.elasticnet_regression
    @mark.elasticnet_regression_learning_rate
    def test_elasticnet_regression_fit_learning_rate_constant(self, get_regression_data):
        X, y = get_regression_data        
        en = ElasticNetRegression(learning_rate = 0.1, epochs=10)
        en.fit(X,y)
        assert en.learning_rate == 0.1, "learning rate not initialized coen.ctly"
        assert en.history.epoch_log['learning_rate'][0]==\
            en.history.epoch_log['learning_rate'][-1], "learning rate not constant in history"

    @mark.elasticnet_regression
    @mark.elasticnet_regression_learning_rate
    def test_elasticnet_regression_fit_learning_rate_sched(self, get_regression_data,
                                                      learning_rate_schedules):
        X, y = get_regression_data        
        lrs = learning_rate_schedules
        en = ElasticNetRegression(learning_rate = lrs, epochs=500)
        en.fit(X,y)
        assert en.history.epoch_log['learning_rate'][0] !=\
            en.history.epoch_log['learning_rate'][-1], "learning rate not changed in history"

    @mark.elasticnet_regression
    @mark.elasticnet_regression_batch_size
    def test_elasticnet_regression_fit_batch_size(self, get_regression_data):
        X, y = get_regression_data         
        X = X[0:33]
        y = y[0:33]       
        en = ElasticNetRegression(batch_size=32, epochs=10)
        en.fit(X,y)                
        assert en.history.total_epochs == 10, "total epochs in history not coen.ct"
        assert en.history.total_batches == 20, "total batches in history not coen.ct"
        assert en.history.total_epochs != en.history.total_batches, "batches and epochs are equal"
        assert en.history.batch_log['batch_size'][0]==32, "batch size not coen.ct in history"
        assert len(en.history.batch_log['batch_size']) ==20, "length of batch log incoen.ct"
        assert len(en.history.epoch_log['learning_rate'])==10, "length of epoch log incoen.ct"


    @mark.elasticnet_regression
    @mark.elasticnet_regression_epochs
    def test_elasticnet_regression_fit_epochs(self, get_regression_data):
        X, y = get_regression_data                
        en = ElasticNetRegression(epochs=10)
        en.fit(X,y)
        assert en.epochs == 10, "estimator epochs invalid"
        assert en.history.total_epochs == 10, "total epochs in history not valid"
        assert len(en.history.epoch_log['learning_rate']) == 10, "epoch log not equal to epochs"

    @mark.elasticnet_regression
    @mark.elasticnet_regression_early_stop
    def test_elasticnet_regression_fit_early_stop(self, get_regression_data,
                                                     early_stop):
        es = early_stop
        X, y = get_regression_data                
        en = ElasticNetRegression(learning_rate=0.5, epochs=5000, early_stop=es)
        en.fit(X,y)
        assert en.history.total_epochs < 5000, "didn't stop early"
        assert len(en.history.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"
    
    @mark.elasticnet_regression
    @mark.elasticnet_regression_predict
    def test_elasticnet_regression_predict(self, get_regression_data):
        X, y = get_regression_data                
        en = ElasticNetRegression(learning_rate = 0.1, epochs=1000)
        with pytest.raises(Exception): # Tests predict w/o first fitting model
            y_pred = en.predict(X)
        en.fit(X, y)
        with pytest.raises(TypeError):
            y_pred = en.predict([1,2,3])
        with pytest.raises(ValueError):            
            y_pred = en.predict(np.reshape(X, (-1,1)))        
        y_pred = en.predict(X)
        assert all(np.equal(y.shape, y_pred.shape)), "y and y_pred have different shapes"  

    @mark.elasticnet_regression
    @mark.elasticnet_regression_score
    def test_elasticnet_regression_score(self, get_regression_data_w_validation, 
                                    regression_metric):
        X, X_test, y, y_test = get_regression_data_w_validation                
        en = ElasticNetRegression(learning_rate = 0.1, epochs=1000, metric=regression_metric)
        with pytest.raises(Exception):
            score = en.score(X, y)
        en.fit(X, y)
        with pytest.raises(TypeError):
            score = en.score("X", y)
        with pytest.raises(TypeError):
            score = en.score(X, [1,2,3])        
        with pytest.raises(ValueError):
            score = en.score(X, np.array([1,2,3]))        
        with pytest.raises(ValueError):
            score = en.score(np.reshape(X, (-1,1)), y)    
        score = en.score(X_test, y_test)
        assert isinstance(score, (int,float)), "score is not an int nor a float"   

    @mark.elasticnet_regression
    @mark.elasticnet_regression_history
    def test_elasticnet_regression_history_no_val_data(self, get_regression_data):        
        X, y = get_regression_data        
        en = ElasticNetRegression(epochs=10)
        en.fit(X, y)        
        # Test epoch history
        assert en.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(en.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert (len(en.history.epoch_log.get("learning_rate")) == 10), "length of learning rate doesn't match epochs"
        assert len(en.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(en.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert en.history.epoch_log.get('train_cost')[0] > en.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert en.history.epoch_log.get('train_score')[0] > en.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert all(np.equal(en.theta, en.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."


    @mark.elasticnet_regression
    @mark.elasticnet_regression_history
    def test_elasticnet_regression_history_w_val_data(self, get_regression_data):        
        X, y = get_regression_data
        es = EarlyStopPlateau(precision=0.1)        
        en = ElasticNetRegression(epochs=100, metric='mean_squared_error', early_stop=es)
        en.fit(X, y)        
        # Test epoch history
        assert en.history.total_epochs < 100, "total_epochs from history doesn't match epochs"
        assert len(en.history.epoch_log.get('epoch')) < 100, "number of epochs in log doesn't match epochs"
        assert (len(en.history.epoch_log.get("learning_rate")) < 100), "length of learning rate doesn't match epochs"
        assert len(en.history.epoch_log.get('theta')) < 100, "number of thetas in log doesn't match epochs"
        assert all(np.equal(en.theta, en.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert len(en.history.epoch_log.get('train_cost')) < 100, "length of train_cost doesn't match epochs"
        assert en.history.epoch_log.get('train_cost')[0] > en.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert en.history.epoch_log.get('train_score')[0] > en.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert len(en.history.epoch_log.get('val_cost')) < 100, "length of val_cost doesn't match epochs"
        assert en.history.epoch_log.get('val_cost')[0] > en.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        assert en.history.epoch_log.get('val_score')[0] > en.history.epoch_log.get('val_score')[-1], "val_score does not decrease"
