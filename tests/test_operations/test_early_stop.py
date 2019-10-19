# =========================================================================== #
#                            TEST EARLY STOP                                  #
# =========================================================================== #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.supervised_learning.training.early_stop import EarlyStopPlateau
from ml_studio.supervised_learning.training.early_stop import EarlyStopGeneralizationLoss
from ml_studio.supervised_learning.training.early_stop import EarlyStopProgress
from ml_studio.supervised_learning.training.early_stop import EarlyStopStrips
from ml_studio.supervised_learning.training.metrics import Scorer
from ml_studio.supervised_learning.regression import LinearRegression
# --------------------------------------------------------------------------- #
#                        TEST EARLY STOP PLATEAU                              #
# --------------------------------------------------------------------------- #

class EarlyStopPlateauTests:

    @mark.early_stop
    @mark.early_stop_plateau
    def test_early_stop_plateau_init(self):
        stop = EarlyStopPlateau()
        assert stop.val_size == 0.2, "val_size not correct"
        assert stop.precision == 0.01, "precision not correct"
        assert stop.better is None, "better is not None on instantiation"
        assert stop.metric is None, "metric is not None on instantiation"
        assert stop.converged is False, "converged is not False on instantiation"
        assert stop.best_weights is None, "best weights is not None on instantiation"

    @mark.early_stop
    @mark.early_stop_plateau
    def test_early_stop_plateau_validation(self):
        with pytest.raises(TypeError):
            stop = EarlyStopPlateau(val_size='x')
            stop.on_train_begin({'metric': None})
        with pytest.raises(TypeError):
            stop = EarlyStopPlateau()
            stop.on_train_begin({"metric": 9})
        with pytest.raises(ValueError):
            stop = EarlyStopPlateau()
            stop.on_train_begin({"metric": 'x'})
        with pytest.raises(TypeError):
            stop = EarlyStopPlateau(precision='x')
            stop.on_train_begin()              
        with pytest.raises(TypeError):
            stop = EarlyStopPlateau(precision=5)
            stop.on_train_begin()
        with pytest.raises(TypeError):
            stop = EarlyStopPlateau(patience='x')
            stop.on_train_begin()         

    @mark.early_stop
    @mark.early_stop_plateau
    @mark.early_stop_plateau_on_train_begin
    def test_early_stop_plateau_on_train_begin(self, regression_metric):        
        # Test with metric
        stop=EarlyStopPlateau()
        stop.on_train_begin({'metric': regression_metric})
        assert stop.metric == regression_metric, "metric not set correctly"        
        scorer = Scorer()(metric=regression_metric)
        assert stop.better == scorer.better, "metric better not set correctly"
        assert stop.best_performance == scorer.worst, "metric best_performance not set correctly"
        assert stop.precision == abs(stop.precision) * scorer.precision_factor, "precision not set correctly"
        # Test w/o metric
        stop=EarlyStopPlateau()
        stop.on_train_begin()
        assert stop.metric == None, "metric not set correctly when using cost as metric"
        assert stop.better == np.less, "better not set correctly when using cost as metric"
        assert stop.best_performance == np.Inf, "best performance not set correctly when using cost as metric"
        assert stop.precision < 0, "precision not set correctly when using cost as metric"

    @mark.early_stop
    @mark.early_stop_plateau
    @mark.early_stop_plateau_on_epoch_end
    def test_early_stop_plateau_on_epoch_end_no_metric_no_val(self):
        stop=EarlyStopPlateau(val_size=0, precision=0.1, patience=2)
        stop.on_train_begin()                
        logs = [{'train_cost': 100}, {'train_cost': 99},{'train_cost': 80},
               {'train_cost': 78},{'train_cost': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged == converged[i], "not converging correctly" 

    @mark.early_stop
    @mark.early_stop_plateau
    @mark.early_stop_plateau_on_epoch_end
    def test_early_stop_plateau_on_epoch_end_no_metric_w_val(self):
        stop=EarlyStopPlateau(precision=0.1, patience=2)
        stop.on_train_begin()                
        logs = [{'val_cost': 100}, {'val_cost': 99},{'val_cost': 80},
               {'val_cost': 78},{'val_cost': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged == converged[i], "not converging correctly"

    @mark.early_stop
    @mark.early_stop_plateau
    @mark.early_stop_plateau_on_epoch_end
    def test_early_stop_plateau_on_epoch_end_metric_no_val_lower_is_better(self, 
                            regression_metric_lower_is_better):
        stop=EarlyStopPlateau(val_size=0, precision=0.1, patience=2)
        stop.on_train_begin({'metric': regression_metric_lower_is_better})                
        logs = [{'train_score': 100}, {'train_score': 99},{'train_score': 80},
               {'train_score': 78},{'train_score': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged == converged[i], "not converging correctly"             
 
    @mark.early_stop
    @mark.early_stop_plateau
    @mark.early_stop_plateau_on_epoch_end
    def test_early_stop_plateau_on_epoch_end_metric_no_val_greater_is_better(self, 
                            regression_metric_greater_is_better):
        stop=EarlyStopPlateau(val_size=0, precision=0.1, patience=2)
        stop.on_train_begin({'metric': regression_metric_greater_is_better})                
        logs = [{'train_score': 100}, {'train_score': 101},{'train_score': 120},
               {'train_score': 122},{'train_score': 123}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged == converged[i], "not converging correctly"    

    @mark.early_stop
    @mark.early_stop_plateau
    @mark.early_stop_plateau_on_epoch_end
    def test_early_stop_plateau_on_epoch_end_metric_w_val_lower_is_better(self, 
                            regression_metric_lower_is_better):
        stop=EarlyStopPlateau(precision=0.1, patience=2)
        stop.on_train_begin({'metric': regression_metric_lower_is_better})                
        logs = [{'val_score': 100}, {'val_score': 99},{'val_score': 80},
               {'val_score': 78},{'val_score': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged == converged[i], "not converging correctly"             
 
    @mark.early_stop
    @mark.early_stop_plateau
    @mark.early_stop_plateau_on_epoch_end
    def test_early_stop_plateau_on_epoch_end_metric_w_val_greater_is_better(self, 
                            regression_metric_greater_is_better):
        stop=EarlyStopPlateau(precision=0.1, patience=2)
        stop.on_train_begin({'metric': regression_metric_greater_is_better})                
        logs = [{'val_score': 100}, {'val_score': 101},{'val_score': 120},
               {'val_score': 122},{'val_score': 123}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged == converged[i], "not converging correctly"                      

# --------------------------------------------------------------------------- #
#                  TEST EARLY STOP GENERALIZATION LOSS                        #
# --------------------------------------------------------------------------- #

class EarlyStopGeneralizationLossTests:

    @mark.early_stop
    @mark.early_stop_generalization_loss
    def test_early_stop_generalization_loss_init(self):    
        stop = EarlyStopGeneralizationLoss()
        assert stop.val_size == 0.2, "val_size not set correctly"
        assert stop.threshold == 2, "threshold not set correctly"
        assert stop.best_val_cost == np.Inf, "best_val_cost not set correctly"

    @mark.early_stop
    @mark.early_stop_generalization_loss
    def test_early_stop_generalization_loss_validation(self):    
        with pytest.raises(TypeError):
            stop = EarlyStopGeneralizationLoss(val_size='x')
            stop.on_train_begin()
        with pytest.raises(ValueError):
            stop = EarlyStopGeneralizationLoss(val_size=0)
            stop.on_train_begin()

    @mark.early_stop
    @mark.early_stop_generalization_loss
    def test_early_stop_generalization_loss_on_epoch_end(self):
        stop = EarlyStopGeneralizationLoss()
        logs = [{'val_cost': 100,'theta': np.random.rand(4)}, 
                {'val_cost': 101,'theta': np.random.rand(4)},
                {'val_cost': 120,'theta': np.random.rand(4)}]
        converged = [False,False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged == converged[i], "not converging correctly"                                  
        assert isinstance(stop.best_weights, (np.ndarray, np.generic)), "best_weights not np.array"

# --------------------------------------------------------------------------- #
#                         TEST EARLY STOP PROGRESS                            #
# --------------------------------------------------------------------------- #

class EarlyStopProgressTests:

    @mark.early_stop
    @mark.early_stop_progress
    def test_early_stop_progress_init(self):    
        stop = EarlyStopProgress(threshold=0.25)
        assert stop.val_size == 0.2, "val_size not set correctly"
        assert stop.threshold == 0.25, "threshold not set correctly"
        assert stop.best_val_cost == np.Inf, "best_val_cost not set correctly"

    @mark.early_stop
    @mark.early_stop_progress
    def test_early_stop_progress_validation(self):    
        with pytest.raises(TypeError):
            stop = EarlyStopProgress(val_size='x')
            stop.on_train_begin()
        with pytest.raises(TypeError):
            stop = EarlyStopProgress(threshold='x')
            stop.on_train_begin()
        with pytest.raises(TypeError):
            stop = EarlyStopProgress(strip_size='x')
            stop.on_train_begin()            

    @mark.early_stop
    @mark.early_stop_progress
    def test_early_stop_progress_on_epoch_end(self):
        # Obtain train and validation costs
        filename = "tests/test_operations/test_early_stop.xlsx"
        df = pd.read_excel(io=filename, sheet_name='progress_data')
        train_costs = df['train_cost']
        val_costs = df['val_cost']
        logs = []
        for i in range(len(train_costs)):
            log = {'train_cost': train_costs[i], 'val_cost': val_costs[i]}
            logs.append(log)
        # Instantiate and test early stop 
        stop = EarlyStopProgress()
        stop.on_train_begin()
        for i in range(len(train_costs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            if i < len(train_costs)-1:
                assert stop.converged == False, "not converging at the appropriate time"
            else:
                assert stop.converged == True, "not converging at the appropriate time"    

# --------------------------------------------------------------------------- #
#                         TEST EARLY STOP STRIPS                              #
# --------------------------------------------------------------------------- #

class EarlyStopStripsTests:

    @mark.early_stop
    @mark.early_stop_strips
    def test_early_stop_strips_init(self):    
        stop = EarlyStopStrips(patience=3)
        assert stop.val_size == 0.2, "val_size not set correctly"
        assert stop.strip_size == 5, "strip size not set correctly"
        assert stop.patience == 3, "patience not set correctly"

    @mark.early_stop
    @mark.early_stop_strips
    def test_early_stop_strips_validation(self):    
        with pytest.raises(TypeError):
            stop = EarlyStopStrips(val_size='x')
            stop.on_train_begin()
        with pytest.raises(TypeError):
            stop = EarlyStopStrips(patience='x')
            stop.on_train_begin()
        with pytest.raises(TypeError):
            stop = EarlyStopStrips(strip_size='x')
            stop.on_train_begin()            

    @mark.early_stop
    @mark.early_stop_strips
    def test_early_stop_strips_on_epoch_end(self):
        # Obtain train and validation costs
        filename = "tests/test_operations/test_early_stop.xlsx"
        df = pd.read_excel(io=filename, sheet_name='strips_data')
        val_costs = df['val_cost']
        logs = []
        for i in range(len(val_costs)):
            log = {'val_cost': val_costs[i]}
            logs.append(log)
        # Instantiate and test early stop 
        stop = EarlyStopStrips(patience=3)
        stop.on_train_begin()
        for i in range(len(val_costs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            if i < len(val_costs)-1:
                assert stop.converged == False, "not converging at the appropriate time"
            else:
                assert stop.converged == True, "not converging at the appropriate time"                
