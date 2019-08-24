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
from ml_studio.operations.metrics import Scorer

class GradientDescentTests:

    @mark.gradient_descent
    @mark.gradient_descent_validation
    def test_gradient_descent_validation(self, get_regression_data):

        X, y = get_regression_data
        with pytest.raises(ValueError):
            gd = GradientDescent(learning_rate="x")
            gd.fit(X, y)        
        with pytest.raises(ValueError):
            gd = GradientDescent(theta_init='k')            
            gd.fit(X, y)
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs='k')           
            gd.fit(X, y)
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs=10,metric='k')                                
            gd.fit(X, y)
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs=10,val_size=1)                                
            gd.fit(X, y)                                       
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs=10,val_size=None)                                
            gd.fit(X, y)                                                    
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs=10,verbose=None)                                                                                      
            gd.fit(X, y)
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs=10,checkpoint=-1)
            gd.fit(X, y)
        with pytest.raises(ValueError):
            gd = GradientDescent(epochs=10,seed='k')                                
            gd.fit(X, y)

    @mark.gradient_descent
    @mark.gradient_descent_get_params
    def test_gradient_descent_get_params(self):
        gd = GradientDescent(learning_rate=0.01, theta_init=np.array([2,2,2]),
                             epochs=10, cost='quadratic', 
                             metric='root_mean_squared_error',
                             val_size=0.3, verbose=False, checkpoint=100, 
                             name=None, seed=50)
        params = gd.get_params()
        assert params['learning_rate'] == 0.01, "learning rate is invalid" 
        assert all(np.equal(params['theta_init'], np.array([2,2,2]))) , "theta_init is invalid"
        assert params['epochs'] == 10, "epochs is invalid"
        assert params['cost'] == 'quadratic', "cost is invalid"
        assert params['metric'] == 'root_mean_squared_error', "metric is invalid"
        assert params['val_size'] == 0.3, "val_size is invalid"
        assert params['verbose'] == False, "verbose is invalid"
        assert params['checkpoint'] == 100, "checkpoint is invalid"
        assert params['name'] is None, "name is invalid"        
        assert params['seed'] == 50, "seed is invalid"


    @mark.gradient_descent
    @mark.gradient_descent_set_name
    def test_gradient_descent_set_name(self):
        gd = GradientDescent()
        gd.set_name("Alex")
        assert gd.name == 'Alex', "name not successfully set"

    @mark.gradient_descent
    @mark.gradient_descent_fit
    def test_gradient_descent_fit_w_val_and_metric(self, get_regression_data, 
                                  regression_metric):
        """Baseline test of fit, including testing of private methods"""  
        if regression_metric == 'root_mean_squared_log_error':
            learning_rate = 0.0001
        else:
            learning_rate = 0.1
        X, y = get_regression_data        
        gd = GradientDescent(learning_rate=learning_rate, epochs=10, val_size=0.3, metric=regression_metric)
        gd.fit(X, y)        
        # Obtain better function for specified metric
        better = Scorer()(gd.history.params.get('metric')).better
        # Test prepare data
        assert X.shape[0] == gd.X.shape[0] + gd.X_val.shape[0], "Sum of shape[0] of X and X_val doesn't match shape of original input"
        assert gd.X.shape[0] > gd.X_val.shape[0], "X shape not greater than X_val shape for split 0.3"
        assert gd.X.shape[1] == X.shape[1] + 1, "X.shape is not correct"
        # Test epoch history
        assert gd.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(gd.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert len(gd.history.epoch_log.get('train_score')) == 10, "length of train_score doesn't match epochs"
        assert len(gd.history.epoch_log.get('val_cost')) == 10, "length of val_cost doesn't match epochs"
        assert len(gd.history.epoch_log.get('val_score')) == 10, "length of val_score doesn't match epochs"
        assert gd.history.epoch_log.get('train_cost')[0] > gd.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert gd.history.epoch_log.get('val_cost')[0] > gd.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        print(gd.history.epoch_log.get('train_score'))
        assert better(gd.history.epoch_log.get('train_score')[-1], 
                      gd.history.epoch_log.get('train_score')[0]), "train_score did not improve"
        assert better(gd.history.epoch_log.get('val_score')[-1], 
                      gd.history.epoch_log.get('val_score')[0]), "val_score did not improve"
        assert all(np.equal(gd.theta, gd.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        
    @mark.gradient_descent
    @mark.gradient_descent_fit
    def test_gradient_descent_fit_wo_val_w_metric(self, get_regression_data, 
                                  regression_metric):
        """Baseline test of fit, including testing of private methods"""  
        X, y = get_regression_data        
        gd = GradientDescent(epochs=10, val_size=0, metric=regression_metric)
        gd.fit(X, y)        
        # Obtain better function for specified metric
        better = Scorer()(gd.history.params.get('metric')).better
        # Test prepare data
        assert X.shape[0] == gd.X.shape[0], "X.shape[0] is not correct "
        assert gd.X.shape[1] == X.shape[1] + 1, "X.shape[1] is not correct"
        # Test epoch history
        assert gd.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(gd.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert len(gd.history.epoch_log.get('train_score')) == 10, "length of train_score doesn't match epochs"
        assert gd.history.epoch_log.get('val_cost') is None, "Validation cost should be none"
        assert gd.history.epoch_log.get('val_score') is None, "Validation score should be none"
        assert gd.history.epoch_log.get('train_cost')[0] > gd.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert better(gd.history.epoch_log.get('train_score')[-1], 
                      gd.history.epoch_log.get('train_score')[0]), "train_score did not improve"
        assert all(np.equal(gd.theta, gd.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."
        
    @mark.gradient_descent
    @mark.gradient_descent_fit
    def test_gradient_descent_fit_w_val_wo_metric(self, get_regression_data):
        """Baseline test of fit, including testing of private methods"""  
        X, y = get_regression_data        
        gd = GradientDescent(epochs=10, val_size=0.3, metric=None)
        gd.fit(X, y)        
        # Test prepare data
        assert X.shape[0] == gd.X.shape[0] + gd.X_val.shape[0], "Sum of shape[0] of X and X_val doesn't match shape of original input"
        assert gd.X.shape[0] > gd.X_val.shape[0], "X shape not greater than X_val shape for split 0.3"
        assert gd.X.shape[1] == X.shape[1] + 1, "X.shape[1] is not correct"
        # Test epoch history
        assert gd.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(gd.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert gd.history.epoch_log.get('train_score') is None, "Train score should be none"
        assert len(gd.history.epoch_log.get('val_cost')) == 10, "length of val_cost doesn't match epochs"
        assert gd.history.epoch_log.get('val_score') is None, "Validation score should be none"
        assert gd.history.epoch_log.get('train_cost')[0] > gd.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"        
        assert gd.history.epoch_log.get('val_cost')[0] > gd.history.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        assert all(np.equal(gd.theta, gd.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."

    @mark.gradient_descent
    @mark.gradient_descent_fit
    def test_gradient_descent_fit_wo_val_wo_metric(self, get_regression_data):
        """Baseline test of fit, including testing of private methods"""  
        X, y = get_regression_data        
        gd = GradientDescent(epochs=10, val_size=0, metric=None)
        gd.fit(X, y)        
        # Test prepare data
        assert X.shape[0] == gd.X.shape[0], "X.shape[0] is incorrect"
        assert gd.X.shape[1] == X.shape[1] + 1, "X.shape[1] is not correct"
        # Test epoch history
        assert gd.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(gd.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert gd.history.epoch_log.get('train_score') is None, "Train score should be none"
        assert gd.history.epoch_log.get('val_cost') is None, "Val cost should be none"
        assert gd.history.epoch_log.get('val_score') is None, "Validation score should be none"
        assert gd.history.epoch_log.get('train_cost')[0] > gd.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert all(np.equal(gd.theta, gd.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."

    @mark.gradient_descent
    @mark.gradient_descent_fit
    def test_gradient_descent_fit_wo_val_wo_metric_init_weights(self, get_regression_data):
        """Baseline test of fit, including testing of private methods"""  
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1]+1)
        gd = GradientDescent(epochs=10, val_size=0, metric=None, theta_init=theta_init)
        gd.fit(X, y)        
        # Test prepare data
        assert X.shape[0] == gd.X.shape[0], "X.shape[0] is incorrect"
        assert gd.X.shape[1] == X.shape[1] + 1, "X.shape[1] is not correct"
        # Test epoch history
        assert gd.history.total_epochs == 10, "total_epochs from history doesn't match epochs"
        assert len(gd.history.epoch_log.get('epoch')) == 10, "number of epochs in log doesn't match epochs"
        assert len(gd.history.epoch_log.get('theta')) == 10, "number of thetas in log doesn't match epochs"
        assert all(np.isclose(gd.history.epoch_log.get('theta')[0], theta_init, rtol=1e-1)), "Theta[0] not equal to theta_init"
        assert len(gd.history.epoch_log.get('train_cost')) == 10, "length of train_cost doesn't match epochs"
        assert gd.history.epoch_log.get('train_score') is None, "Train score should be none"
        assert gd.history.epoch_log.get('val_cost') is None, "Val cost should be none"
        assert gd.history.epoch_log.get('val_score') is None, "Validation score should be none"
        assert gd.history.epoch_log.get('train_cost')[0] > gd.history.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert all(np.equal(gd.theta, gd.history.epoch_log.get('theta')[-1])), "Last theta in log doesn't equal final theta."

    @mark.gradient_descent
    @mark.gradient_descent_predict
    def test_gradient_descent_predict(self, get_regression_data):
        X, y = get_regression_data                
        gd = GradientDescent(epochs=10)
        with pytest.raises(Exception):
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
    def test_gradient_descent_score(self, get_regression_data, regression_metric):
        X, y = get_regression_data                
        gd = GradientDescent(epochs=10, metric=regression_metric)
        with pytest.raises(Exception):
            s = gd.score(X, y)
        gd.fit(X, y)
        with pytest.raises(TypeError):
            s = gd.score("X", y)
        with pytest.raises(TypeError):
            s = gd.score(X, [1,2,3])        
        with pytest.raises(ValueError):
            s = gd.score(X, np.array([1,2,3]))        
        with pytest.raises(ValueError):
            s = gd.score(np.reshape(X, (-1,1)), y)    
        s = gd.score(X, y)
        assert isinstance(s, float), "score is not a float"    


    @mark.gradient_descent
    @mark.gradient_descent_fit_solution
    def test_gradient_descent_fit_solution(self, get_regression_data, analytical_solution):
        X, y = get_regression_data        
        gd = GradientDescent(epochs=2000, val_size=0, seed=50)
        gd.fit(X, y)                       
        y_pred = gd.predict(X)        
        assert all(np.isclose(gd.theta, analytical_solution, rtol=1e1)), "Solution is not close to analytical solution."
        assert all(np.isclose(y, y_pred, rtol=1e1)), "Predictions are not close to analytical solution."  