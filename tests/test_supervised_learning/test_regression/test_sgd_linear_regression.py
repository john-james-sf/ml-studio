# --------------------------------------------------------------------------- #
#                          TEST LINEAR REGRESSION                             #
# --------------------------------------------------------------------------- #
#%%
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import SGDRegression
from ml_studio.operations.callbacks import Callback
from ml_studio.operations.metrics import Scorer
from ml_studio.utils.filemanager import save_csv

@mark.sgd
@mark.sgdregression
class SGDRegressionTests:

    def test_sgd_validation(self, get_ames_data, get_optimizer):

        X, y = get_ames_data
        optimizer = get_optimizer
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer="x")
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,theta_init='k')            
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs='k')           
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,fit_intercept='k')                                            
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,monitor='k')                                
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,metric='k')                                
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,val_size=1)                                
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,monitor='val_cost',
                                          val_size=0)
            lr.fit(X, y)                                          
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,verbose='k')                                                                                      
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,checkpoint='k')
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,checkpoint=-1)
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5,seed='k')                                
            lr.fit(X, y)
        with pytest.raises(ValueError):
            lr = LinearRegression(optimizer=optimizer, epochs=5, metric=None,
                                  monitor='val_score')                                
            lr.fit(X, y)            

    def test_sgd_prepare_data_w_fit_intercept_n_validation(self, get_ames_data, get_optimizer):

        optimizer = get_optimizer
        X, y = get_ames_data
        # Test with fit intercept and validation set
        lr = LinearRegression(optimizer=optimizer, epochs=5)
        lr.fit(X, y)
        assert lr.X.shape[0] > 700, "X data not available when fitting intercept"
        assert lr.X_val.shape[0] > 200, "X_val data not available when fitting intercept"
        assert lr.X.shape[1] == X.shape[1] + 1, "X shape does not include bias term"
        assert lr.X_val.shape[1] == X.shape[1] + 1, "X_val shape does not include bias term"
        assert isinstance(lr.X, (np.ndarray, np.generic)), "X is not a numpy array"
        assert isinstance(lr.y, (np.ndarray, np.generic)), "y is not a numpy array"
        assert isinstance(lr.X_val, (np.ndarray, np.generic)), "X_val is not a numpy array"
        assert isinstance(lr.y_val, (np.ndarray, np.generic)), "y_val is not a numpy array"

    def test_sgd_prepare_data_wo_fit_intercept_w_validation(self, get_ames_data, get_optimizer):
        optimizer = get_optimizer
        X, y = get_ames_data
        # Test with validation set and without fitting intercept
        lr = LinearRegression(optimizer=optimizer, epochs=5, fit_intercept=False)
        lr.fit(X, y)        
        assert lr.X.shape[0] > 700, "X data not available when not fitting intercept"
        assert lr.X_val.shape[0] > 200, "X_val data not available when not fitting intercept"
        assert lr.X.shape[1] == X.shape[1], "X shape not equal to input shape."
        assert lr.X_val.shape[1] == X.shape[1], "X_val shape not equal to input shape."
        assert isinstance(lr.X, (np.ndarray, np.generic)), "X is not a numpy array"
        assert isinstance(lr.y, (np.ndarray, np.generic)), "y is not a numpy array"
        assert isinstance(lr.X_val, (np.ndarray, np.generic)), "X_val is not a numpy array"
        assert isinstance(lr.y_val, (np.ndarray, np.generic)), "y_val is not a numpy array"        

    def test_sgd_prepare_data_w_fit_intercept_wo_validation(self, get_ames_data, get_optimizer):
        optimizer = get_optimizer
        X, y = get_ames_data        
        # Test with fit intercept, no validation set
        lr = LinearRegression(optimizer=optimizer, epochs=5, val_size=0.0, monitor='train_cost')
        lr.fit(X, y)        
        assert lr.X.shape[0] > 1000, "X data not available when not fitting intercept"
        assert lr.X.shape[1] == X.shape[1]+1, "X doesn't equal input shape plus bias."
        assert lr.X_val is None, "X_val does not equal None."
        assert lr.y_val is None, "y_val does not equal None."
        assert isinstance(lr.X, (np.ndarray, np.generic)), "X is not a numpy array"
        assert isinstance(lr.y, (np.ndarray, np.generic)), "y is not a numpy array"

    def test_sgd_prepare_data_wo_fit_intercept_wo_validation(self, get_ames_data, get_optimizer):
        optimizer = get_optimizer
        X, y = get_ames_data        
        # Test with no validation set and no fitting intercept
        lr = LinearRegression(optimizer=optimizer, epochs=5, fit_intercept=False, 
                              monitor='train_cost', val_size=0)
        lr.fit(X, y)        
        assert lr.X.shape[0] > 1000, "X data not available when not fitting intercept"
        assert lr.X.shape[1] == X.shape[1], "X shape not equal to input shape."
        assert isinstance(lr.X, (np.ndarray, np.generic)), "X is not a numpy array"
        assert isinstance(lr.y, (np.ndarray, np.generic)), "y is not a numpy array"
        assert lr.X_val is None, "X_val is not None"
        assert lr.y_val is None, "y_val is not None"

    def test_sgd_init_weights(self, get_ames_data, get_optimizer):
        optimizer = get_optimizer
        X, y = get_ames_data
        # With initialization and bias term
        theta_init = np.ones(X.shape[1]+1)
        lr = LinearRegression(optimizer=optimizer, epochs=5, theta_init=theta_init)
        lr.fit(X, y)
        assert np.array_equal(lr.history.batch_log.get('theta')[0],theta_init), "Theta failed to initiate to initial weights."        
        assert len(lr.history.batch_log.get('theta')[0]) == X.shape[1] + 1, "Theta shape doesn't equal input shape plus bias."
        # With initialization and no bias term
        theta_init = np.ones(X.shape[1])
        lr = LinearRegression(optimizer=optimizer, epochs=5, theta_init=theta_init, fit_intercept=False)
        lr.fit(X, y)
        assert np.array_equal(lr.history.batch_log.get('theta')[0],theta_init), "Theta failed to initiate to initial weights."        
        assert len(lr.history.epoch_log.get('theta')[0]) == X.shape[1], "Theta shape doesn't equal input shape."         
        # With random initialization
        lr = LinearRegression(optimizer=optimizer)
        lr.fit(X, y)
        assert sum(lr.history.batch_log['theta'][0]) < 10, "Random theta initialization failed"

    @mark.fit
    @mark.fit_linear_regression
    def test_sgd_fit_linear_regression_w_validation(self, get_ames_data, get_optimizer):        
        # Run model
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, epochs=5)
        lr.fit(X, y)
        # Obtain parameters and function that obtains best value by metric
        params = lr.get_params()
        better = Scorer()(metric=params['metric']).better
        best = Scorer()(metric=params['metric']).best
        # Obtain output from fit
        total_epochs = lr.history.total_epochs
        total_batches = lr.history.total_batches
        train_costs = lr.history.epoch_log.get("train_cost")    
        train_scores = lr.history.epoch_log.get("train_score")        
        val_costs = lr.history.epoch_log.get("val_cost")     
        val_scores = lr.history.epoch_log.get("val_score")     
        performance = {'train_cost': train_costs, 'train_score': train_scores,
                       'val_cost': val_costs, 'val_score': val_scores}

        weights = lr.history.epoch_log.get("theta") 
        best_performance = lr.benchmark.best_model.get("performance")
        best_weights = lr.benchmark.best_model.get('theta')
        best_idx = performance[params['monitor']].index(best(performance[params['monitor']]))
        # Check callbacks
        assert isinstance(lr.history, Callback), "Benchmark callback is not callable."
        assert isinstance(lr.benchmark, Callback), "History callback is not callable."
        # Check number of epochs run
        assert total_epochs == params.get("epochs"), "Training did not complete. Total epochs < epochs"
        assert total_epochs == total_batches, "Total epochs doesn't equal total batches"
        # Confirm final and best weights are presented
        assert lr.final_coef is not None, "Final coefficient should not be None."
        assert lr.final_intercept is not None, "Final intercept should not be None."
        assert lr.best_coef is not None, "Best coefficient should not be None."
        assert lr.best_intercept is not None, "Best intercept should not be None."
        # Confirm final weights values
        assert np.array_equal(lr.final_coef,weights[-1][1:]), "Final coefficient <> last log value."
        assert np.array_equal(lr.final_intercept, weights[-1][0]), "Final coefficient <> last log value."
        # Confirm best weights value        
        assert np.array_equal(lr.best_coef, weights[best_idx][1:]), "Best coefficient not matching best "
        assert np.array_equal(lr.best_intercept, weights[best_idx][0]), "Best intercept not matching best "
        assert np.array_equal(weights[best_idx], best_weights), "Best weights not associated with minimum performance"
        # Confirm cost and scores improving
        assert train_costs[0] > train_costs[-1], "Final training costs not less than initial cost."
        assert val_costs[0] > val_costs[-1], "Final training costs not less than initial cost."
        assert better(train_scores[-1], train_scores[0]), "Training scores did not improve."
        assert better(val_scores[-1], val_scores[0]), "Validation scores did not improve."
        # Check best performance
        assert best_performance == performance[params['monitor']][best_idx], "Best performance doesn't match"
        # Check shapes
        assert len(best_weights) == X.shape[1] + 1, "Weights and features dimension mismatch."

    def test_sgd_fit_linear_regression_wo_validation(self, get_ames_data, get_optimizer):        
        # Run model
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, monitor='train_cost', val_size=0, epochs=5)
        lr.fit(X, y)        
        # Obtain parameters and function that obtains best value by metric
        params = lr.get_params()
        better = Scorer()(metric=params['metric']).better
        best = Scorer()(metric=params['metric']).best
        # Obtain output from fit
        total_epochs = lr.history.total_epochs
        total_batches = lr.history.total_batches
        train_costs = lr.history.epoch_log.get("train_cost")    
        train_scores = lr.history.epoch_log.get("train_score")        
        val_costs = lr.history.epoch_log.get("val_cost")     
        val_scores = lr.history.epoch_log.get("val_score")     
        performance = {'train_cost': train_costs, 'train_score': train_scores,
                       'val_cost': val_costs, 'val_score': val_scores}

        weights = lr.history.epoch_log.get("theta") 
        best_performance = lr.benchmark.best_model.get("performance")
        best_weights = lr.benchmark.best_model.get('theta')
        best_idx = performance[params['monitor']].index(best(performance[params['monitor']]))
        # Check callbacks
        assert isinstance(lr.history, Callback), "Benchmark callback is not callable."
        assert isinstance(lr.benchmark, Callback), "History callback is not callable."
        # Check number of epochs run
        assert total_epochs == params.get("epochs"), "Training did not complete. Total epochs < epochs"
        assert total_epochs == total_batches, "Total epochs doesn't equal total batches"
        # Confirm final and best weights are presented
        assert lr.final_coef is not None, "Final coefficient should not be None."
        assert lr.final_intercept is not None, "Final intercept should not be None."
        assert lr.best_coef is not None, "Best coefficient should not be None."
        assert lr.best_intercept is not None, "Best intercept should not be None."
        # Confirm final weights values
        assert np.array_equal(lr.final_coef,weights[-1][1:]), "Final coefficient <> last log value."
        assert np.array_equal(lr.final_intercept, weights[-1][0]), "Final coefficient <> last log value."
        # Confirm best weights value        
        assert np.array_equal(lr.best_coef, weights[best_idx][1:]), "Best coefficient not matching best "
        assert np.array_equal(lr.best_intercept, weights[best_idx][0]), "Best intercept not matching best "
        assert np.array_equal(weights[best_idx], best_weights), "Best weights not associated with minimum performance"
        # Confirm cost and scores improving
        assert train_costs[0] > train_costs[-1], "Final training costs not less than initial cost."
        assert None in val_costs, "Validation costs are not None when training w/o validation."
        assert None in val_scores, "Validation scores are not None when training w/o validation."
        assert better(train_scores[-1], train_scores[0]), "Training scores did not improve."
        # Check best performance
        assert best_performance == performance[params['monitor']][best_idx], "Best performance doesn't match"
        # Check shapes
        assert len(best_weights) == X.shape[1] + 1, "Weights and features dimension mismatch."        

    def test_sgd_fit_linear_regression_wo_fit_intercept_wo_validation(self, get_ames_data, get_optimizer):        
        # Run model
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, monitor='train_cost', val_size=0, epochs=5,
                              fit_intercept=False)
        lr.fit(X, y)        
        # Obtain parameters and function that obtains best value by metric
        params = lr.get_params()
        better = Scorer()(metric=params['metric']).better
        best = Scorer()(metric=params['metric']).best
        # Obtain output from fit
        total_epochs = lr.history.total_epochs
        total_batches = lr.history.total_batches
        train_costs = lr.history.epoch_log.get("train_cost")    
        train_scores = lr.history.epoch_log.get("train_score")        
        val_costs = lr.history.epoch_log.get("val_cost")     
        val_scores = lr.history.epoch_log.get("val_score")     
        performance = {'train_cost': train_costs, 'train_score': train_scores,
                       'val_cost': val_costs, 'val_score': val_scores}

        weights = lr.history.epoch_log.get("theta") 
        best_performance = lr.benchmark.best_model.get("performance")
        best_weights = lr.benchmark.best_model.get('theta')
        best_idx = performance[params['monitor']].index(best(performance[params['monitor']]))
        # Check callbacks
        assert isinstance(lr.history, Callback), "Benchmark callback is not callable."
        assert isinstance(lr.benchmark, Callback), "History callback is not callable."
        # Check number of epochs run
        assert total_epochs == params.get("epochs"), "Training did not complete. Total epochs < epochs"
        assert total_epochs == total_batches, "Total epochs doesn't equal total batches"
        # Confirm final and best weights are presented
        assert lr.final_coef is not None, "Final coefficient should not be None."
        assert lr.final_intercept is None, "Final intercept should not be None."
        assert lr.best_coef is not None, "Best coefficient should not be None."
        assert lr.best_intercept is None, "Best intercept should not be None."
        # Confirm final weights values
        assert np.array_equal(lr.final_coef,weights[-1]), "Final coefficient <> last log value."
        # Confirm best weights value        
        assert np.array_equal(lr.best_coef, weights[best_idx]), "Best coefficient not matching best "
        assert np.array_equal(weights[best_idx], best_weights), "Best weights not associated with minimum performance"
        # Confirm cost and scores improving
        assert train_costs[0] > train_costs[-1], "Final training costs not less than initial cost."
        assert None in val_costs, "Validation costs are not None when training w/o validation."
        assert None in val_scores, "Validation scores are not None when training w/o validation."
        assert better(train_scores[-1], train_scores[0]), "Training scores did not improve."
        # Check best performance
        assert best_performance == performance[params['monitor']][best_idx], "Best performance doesn't match"
        # Check shapes
        assert len(best_weights) == X.shape[1], "Weights and features dimension mismatch."    

    def test_sgd_fit_linear_regression_monitor_train_cost(self, get_ames_data, get_optimizer):        
        # Run model
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, monitor='train_cost', epochs=5)
        lr.fit(X, y)        
        # Check benchmark
        assert lr.benchmark.best_model['monitor'] == "train_cost", "Invalid monitor value in benchmark"
       
    def test_sgd_fit_linear_regression_monitor_train_score(self, get_ames_data, get_optimizer):        
        # Run model
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, monitor='train_score', epochs=5)
        lr.fit(X, y)        
        # Check benchmark
        assert lr.benchmark.best_model['monitor'] == "train_score", "Invalid monitor value in benchmark"

    def test_sgd_fit_linear_regression_monitor_val_cost(self, get_ames_data, get_optimizer):        
        # Run model
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, monitor='val_cost', epochs=5)
        lr.fit(X, y)        
        # Check benchmark
        assert lr.benchmark.best_model['monitor'] == "val_cost", "Invalid monitor value in benchmark"

    def test_sgd_fit_linear_regression_monitor_val_score(self, get_ames_data, get_optimizer):        
        # Run model
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, monitor='val_score', epochs=5)
        lr.fit(X, y)        
        # Check benchmark
        assert lr.benchmark.best_model['monitor'] == "val_score", "Invalid monitor value in benchmark"        

    @mark.predict_linear_regression
    def test_sgd_predict_linear_regression(self, get_ames_data, get_optimizer):
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, epochs=5)
        lr.fit(X, y)
        y_pred = lr.predict(X)
        assert isinstance(y_pred, (np.ndarray, np.generic)), "The predict method did not return an np.ndarray."
        assert y_pred.shape[0] == X.shape[0], "Prediction shape[0] not equal to X.shape[0]"

    @mark.score_linear_regression
    def test_sgd_score_linear_regression(self, get_ames_data, get_optimizer):
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = LinearRegression(optimizer=optimizer, epochs=5)
        lr.fit(X, y)
        score = lr.score(X, y)
        assert isinstance(score, float), "The score method did not return a float."
    
    @mark.sgd 
    def test_sgd_linear_regression_sgd_batch_size_1(self, get_ames_data, get_optimizer):
        optimizer = get_optimizer
        X, y = get_ames_data
        lr = SGDRegression(optimizer=optimizer, epochs=5)
        lr.fit(X, y)
        total_epochs = lr.history.total_epochs
        total_batches = lr.history.total_batches
        batch_size = lr.history.batch_log.get("batch_size")[0]
        assert total_epochs < total_batches, "Total batches are not greater than total epochs."
        assert batch_size == 1, "Batch size does not equal 1."
        




