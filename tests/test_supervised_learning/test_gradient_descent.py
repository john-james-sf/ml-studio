# --------------------------------------------------------------------------- #
#                          TEST GRADIENT DESCENT                              #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.estimator import GradientDescent
from ml_studio.operations.callbacks import Callback
from ml_studio.operations.cost import Cost, Quadratic, BinaryCrossEntropy
from ml_studio.operations.cost import CategoricalCrossEntropy
from ml_studio.operations.metrics import Metric
from ml_studio.operations.early_stop import EarlyStopPlateau

class GradientDescentTests:

    @mark.gradient_descent
    @mark.gradient_descent_val
    def test_gradient_descent_validation(self, get_regression_data):

        X, y = get_regression_data
        with pytest.raises(TypeError):
            gd = GradientDescent(learning_rate="x")
            gd.fit(X, y)        
        with pytest.raises(TypeError):
            gd = GradientDescent(batch_size='k')            
            gd.fit(X, y)
        with pytest.raises(TypeError):
            gd = GradientDescent(theta_init='k')            
            gd.fit(X, y)
        with pytest.raises(TypeError):
            gd = GradientDescent(epochs='k')           
            gd.fit(X, y)
        with pytest.raises(ValueError):
            gd = GradientDescent(cost='x')                                
            gd.fit(X, y) 
        with pytest.raises(ValueError):
            gd = GradientDescent(cost=None)                                
            gd.fit(X, y)                  
        with pytest.raises(TypeError):
            gd = GradientDescent(early_stop='x')                                
            gd.fit(X, y)
        with pytest.raises(TypeError):
            gd = GradientDescent(metric=0)                                
            gd.fit(X, y)             
        with pytest.raises(ValueError):
            gd = GradientDescent(metric='x')                                
            gd.fit(X, y) 
        with pytest.raises(ValueError):
            gd = GradientDescent(early_stop=EarlyStopPlateau(), metric=None)                                
            gd.fit(X, y)                                                             
        with pytest.raises(TypeError):
            gd = GradientDescent(epochs=10,verbose=None)                                                                                      
            gd.fit(X, y)
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs=10,checkpoint=-1)
            gd.fit(X, y)
        with pytest.raises(TypeError):
            gd = GradientDescent(epochs=10,checkpoint='x')
            gd.fit(X, y)  
        with pytest.warns(UserWarning):
            gd = GradientDescent(epochs=10,checkpoint=100)
            gd.fit(X, y)                        
        with pytest.raises(TypeError):
            gd = GradientDescent(epochs=10,seed='k')                                
            gd.fit(X, y)
                       

    @mark.gradient_descent
    @mark.gradient_descent_get_params
    def test_gradient_descent_get_params(self):
        gd = GradientDescent(learning_rate=0.01, theta_init=np.array([2,2,2]),
                             epochs=10, cost='quadratic', 
                             verbose=False, checkpoint=100, 
                             name=None, seed=50)
        params = gd.get_params()
        assert params['learning_rate'] == 0.01, "learning rate is invalid" 
        assert all(np.equal(params['theta_init'], np.array([2,2,2]))) , "theta_init is invalid"
        assert params['epochs'] == 10, "epochs is invalid"
        assert params['cost'] == 'quadratic', "cost is invalid"
        assert params['verbose'] == False, "verbose is invalid"
        assert params['checkpoint'] == 100, "checkpoint is invalid"
        assert params['seed'] == 50, "seed is invalid"

    @mark.gradient_descent
    @mark.gradient_descent_validate_data
    def test_gradient_descent_validate_data(self, get_regression_data):
        gd = GradientDescent(epochs=10)
        X, y = get_regression_data        
        with pytest.raises(TypeError):
            gd.fit([1,2,3], y)
        with pytest.raises(TypeError):
            gd.fit(X, [1,2,3])            
        with pytest.raises(ValueError):
            gd.fit(X, y[0:5])            

    @mark.gradient_descent
    @mark.gradient_descent_prepare_data
    def test_gradient_descent_prepare_data_no_val_set(self, get_regression_data):
        gd = GradientDescent(epochs=10)
        X, y = get_regression_data                
        gd.fit(X,y)
        assert X.shape[1] == gd.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] == gd.X.shape[0], "X.shape[0] changed in prepare data"
        assert y.shape == gd.y.shape, "y shape changed in prepare data" 

    @mark.gradient_descent
    @mark.gradient_descent_prepare_data
    def test_gradient_descent_prepare_data_w_val_set(self, get_regression_data):                
        es = EarlyStopPlateau(val_size=0.2)
        gd = GradientDescent(epochs=10, early_stop=es)
        X, y = get_regression_data                
        gd.fit(X,y)
        assert X.shape[1] == gd.X.shape[1] - 1, "intercept column not added to X in prepare data"
        assert X.shape[0] != gd.X.shape[0], "X.shape[0] not changed in prepare data validation split"
        assert y.shape != gd.y.shape, "y shape not changed in prepare data validation split" 
        assert gd.X_val is not None, "X_val is None, not created in prepare data."
        assert gd.y_val is not None, "y_val is None, not created in prepare data."
        assert gd.X.shape[0] + gd.X_val.shape[0] == X.shape[0], "X.shape[0] plus X_val.shape[0] doesn't match input shape"
        assert gd.y.shape[0] + gd.y_val.shape[0] == y.shape[0], "y.shape[0] plus y_val.shape[0] doesn't match output shape"

    @mark.gradient_descent
    @mark.gradient_descent_compile
    def test_gradient_descent_compile(self, get_regression_data):
        X, y = get_regression_data        
        gd = GradientDescent(epochs=10)
        gd.fit(X,y)        
        assert isinstance(gd.cost_function, Quadratic), "cost function is not an instance of valid Cost subclass."
        assert isinstance(gd.scorer, Metric), "scorer function is not an instance of a valid Metric subclass."
        assert isinstance(gd.history, Callback), "history function is not an instance of a valid Callback subclass."
        assert isinstance(gd.progress, Callback), "progress function is not an instance of a valid Callback subclass."

    @mark.gradient_descent
    @mark.gradient_descent_init_weights
    def test_gradient_descent_init_weights_shape_match(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1]+1)        
        gd = GradientDescent(epochs=10, theta_init=theta_init)
        gd.fit(X,y)
        assert not all(np.equal(gd.theta, gd.theta_init)), "final and initial thetas are equal" 

    @mark.gradient_descent
    @mark.gradient_descent_init_weights
    def test_gradient_descent_init_weights_shape_mismatch(self, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1])        
        gd = GradientDescent(epochs=10, theta_init=theta_init)
        with pytest.raises(ValueError):
            gd.fit(X,y)

    @mark.gradient_descent
    @mark.gradient_descent_learning_rate
    def test_gradient_descent_fit_learning_rate_constant(self, get_regression_data):
        X, y = get_regression_data        
        gd = GradientDescent(learning_rate = 0.1, epochs=10)
        gd.fit(X,y)
        assert gd.learning_rate == 0.1, "learning rate not initialized correctly"
        assert gd.history.epoch_log['learning_rate'][0]==\
            gd.history.epoch_log['learning_rate'][-1], "learning rate not constant in history"

    @mark.gradient_descent
    @mark.gradient_descent_learning_rate
    def test_gradient_descent_fit_learning_rate_sched(self, get_regression_data,
                                                      learning_rate_schedules):
        X, y = get_regression_data        
        lrs = learning_rate_schedules
        gd = GradientDescent(learning_rate = lrs, epochs=100)
        gd.fit(X,y)
        assert gd.history.epoch_log['learning_rate'][0] !=\
            gd.history.epoch_log['learning_rate'][-1], "learning rate not changed in history"

    @mark.gradient_descent
    @mark.gradient_descent_batch_size
    def test_gradient_descent_fit_batch_size(self, get_regression_data):
        X, y = get_regression_data         
        X = X[0:33]
        y = y[0:33]       
        gd = GradientDescent(batch_size=32, epochs=10)
        gd.fit(X,y)                
        assert gd.history.total_epochs == 10, "total epochs in history not correct"
        assert gd.history.total_batches == 20, "total batches in history not correct"
        assert gd.history.total_epochs != gd.history.total_batches, "batches and epochs are equal"
        assert gd.history.batch_log['batch_size'][0]==32, "batch size not correct in history"
        assert len(gd.history.batch_log['batch_size']) ==20, "length of batch log incorrect"
        assert len(gd.history.epoch_log['learning_rate'])==10, "length of epoch log incorrect"


    @mark.gradient_descent
    @mark.gradient_descent_epochs
    def test_gradient_descent_fit_epochs(self, get_regression_data):
        X, y = get_regression_data                
        gd = GradientDescent(epochs=10)
        gd.fit(X,y)
        assert gd.epochs == 10, "estimator epochs invalid"
        assert gd.history.total_epochs == 10, "total epochs in history not valid"
        assert len(gd.history.epoch_log['learning_rate']) == 10, "epoch log not equal to epochs"

    @mark.gradient_descent
    @mark.gradient_descent_early_stop
    def test_gradient_descent_fit_early_stop(self, get_regression_data,
                                                     early_stop):
        es = early_stop
        X, y = get_regression_data                
        gd = GradientDescent(learning_rate=0.5, epochs=5000, early_stop=es)
        gd.fit(X,y)
        assert gd.history.total_epochs < 5000, "didn't stop early"
        assert len(gd.history.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"
    
    @mark.gradient_descent
    @mark.gradient_descent_predict
    def test_gradient_descent_predict(self, get_regression_data):
        X, y = get_regression_data                
        gd = GradientDescent(learning_rate = 0.1, epochs=1000)
        with pytest.raises(Exception): # Tests predict w/o first fitting model
            y_pred = gd.predict(X)
        gd.fit(X, y)
        with pytest.raises(TypeError):
            y_pred = gd.predict([1,2,3])
        with pytest.raises(ValueError):            
            y_pred = gd.predict(np.reshape(X, (-1,1)))        
        y_pred = gd.predict(X)
        assert all(np.equal(y.shape, y_pred.shape)), "y and y_pred have different shapes"  

    @mark.gradient_descent
    @mark.gradient_descent_score
    def test_gradient_descent_score(self, get_regression_data_w_validation, 
                                    regression_metric):
        X, X_test, y, y_test = get_regression_data_w_validation                
        gd = GradientDescent(learning_rate = 0.1, epochs=1000, metric=regression_metric)
        with pytest.raises(Exception):
            score = gd.score(X, y)
        gd.fit(X, y)
        with pytest.raises(TypeError):
            score = gd.score("X", y)
        with pytest.raises(TypeError):
            score = gd.score(X, [1,2,3])        
        with pytest.raises(ValueError):
            score = gd.score(X, np.array([1,2,3]))        
        with pytest.raises(ValueError):
            score = gd.score(np.reshape(X, (-1,1)), y)    
        # Model evaluation 
        score = gd.score(X_test, y_test)
        assert isinstance(score, (int,float)), "score is not an int nor a float"   
        if regression_metric == 'r2':
            assert score >= 0.6, "R2 score below 0.6"
        elif regression_metric == 'var_explained':
            assert score >= 0.6, "Var explained below 0.6"
        elif regression_metric == 'mean_absolute_error':
            assert score < 5, "Mean absolute error greater > 5"            
        elif regression_metric == 'mean_squared_error':
            assert score < 35, "mean_squared_error > 35"            
        elif regression_metric == 'neg_mean_squared_error':
            assert score > -35, "neg_mean_squared_error < -35"            
        elif regression_metric == 'root_mean_squared_error':
            assert score < 10, "root_mean_squared_error > 10"                        
        elif regression_metric == 'neg_root_mean_squared_error':
            assert score > -10, "neg_root_mean_squared_error < -10"                                    
        elif regression_metric == 'median_absolute_error':
            assert score < 5, "median_absolute_error > 5"            

    @mark.gradient_descent
    @mark.gradient_descent_history
    def test_gradient_descent_history_no_val_data(self, get_regression_data):        
        X, y = get_regression_data        
        gd = GradientDescent(epochs=10)
        gd.fit(X, y)        
        # Test epoch history
        assert gd.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(gd.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert (len(gd.history.epoch_log.get("learning_rate")) == 10), "length of learning rate doesn't match epochs"
        assert len(gd.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert gd.history.epoch_log.get('train_cost')[0] > gd.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert gd.history.epoch_log.get('train_score')[0] > gd.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert all(np.equal(gd.theta, gd.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."


    @mark.gradient_descent
    @mark.gradient_descent_history
    def test_gradient_descent_history_w_val_data(self, get_regression_data):        
        X, y = get_regression_data
        es = EarlyStopPlateau(precision=0.1)        
        gd = GradientDescent(epochs=100, metric='mean_squared_error', early_stop=es)
        gd.fit(X, y)        
        # Test epoch history
        assert gd.history.total_epochs < 100, "total_epochs from history doesn't match epochs"
        assert len(gd.history.epoch_log.get('epoch')) < 100, "number of epochs in log doesn't match epochs"
        assert (len(gd.history.epoch_log.get("learning_rate")) < 100), "length of learning rate doesn't match epochs"
        assert len(gd.history.epoch_log.get('theta')) < 100, "number of thetas in log doesn't match epochs"
        assert all(np.equal(gd.theta, gd.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        assert len(gd.history.epoch_log.get('train_cost')) < 100, "length of train_cost doesn't match epochs"
        assert gd.history.epoch_log.get('train_cost')[0] > gd.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert gd.history.epoch_log.get('train_score')[0] > gd.history.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert len(gd.history.epoch_log.get('val_cost')) < 100, "length of val_cost doesn't match epochs"
        assert gd.history.epoch_log.get('val_cost')[0] > gd.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        assert gd.history.epoch_log.get('val_score')[0] > gd.history.epoch_log.get('val_score')[-1], "val_score does not decrease"
