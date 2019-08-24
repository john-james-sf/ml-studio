# --------------------------------------------------------------------------- #
#                              TEST METRICS                                   #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pytest
from pytest import mark

from ml_studio.operations import metrics 

class MetricsTests:

    @mark.metrics
    def test_r2(self, predict_y):
        y, y_pred = predict_y
        x = metrics.R2()(y, y_pred)            
        assert x<=1, "R2 is not less than 1"

    @mark.metrics
    def test_var_explained(self, predict_y):
        y, y_pred = predict_y
        x = metrics.VarExplained()(y, y_pred)        
        assert x<=1, "Variance explained not between 0 and 1"        

    @mark.metrics
    def test_mae(self, predict_y):
        y, y_pred = predict_y
        x = metrics.MAE()(y, y_pred)        
        assert x>0, "MAE is not positive"        

    @mark.metrics
    def test_mse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.MSE()(y, y_pred)        
        assert isinstance(x, float), "MSE is not a float"        
        assert x > 0, "MSE is not positive"

    @mark.metrics
    def test_nmse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.NMSE()(y, y_pred)        
        assert isinstance(x, float), "NMSE is not a float"                
        assert x < 0, "NMSE is not negative"

    @mark.metrics
    def test_rmse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.RMSE()(y, y_pred)        
        assert isinstance(x, float), "RMSE is not a float"                
        assert x > 0, "RMSE is not positive"        

    @mark.metrics
    def test_nrmse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.NRMSE()(y, y_pred)        
        assert isinstance(x, float), "NRMSE is not a float"                
        assert x < 0, "NRMSE is not negative"         

    # Not testing. Negative predictions.
    # @mark.metrics
    # def test_msle(self, predict_y):
    #     y, y_pred = predict_y
    #     x = metrics.MSLE()(y, y_pred)    
    #     if any(y < 0): print ("Y IS NEGATIVE")    
    #     if any(y_pred < 0): print ("Y_PRED IS NEGATIVE")    
    #     assert isinstance(x, float), "MSLE is not a float"                
    #     assert x > 0, "MSLE is not  positive"                        

    # @mark.metrics
    # def test_rmsle(self, predict_y):
    #     y, y_pred = predict_y
    #     x = metrics.RMSLE()(y, y_pred)        
    #     assert isinstance(x, float), "RMSLE is not a float"                
    #     assert x > 0, "RMSLE is not  positive"           

    @mark.metrics
    def test_medae(self, predict_y):
        y, y_pred = predict_y
        x = metrics.MEDAE()(y, y_pred)        
        assert isinstance(x, float), "MEDAE is not a float"                
        assert x > 0, "MEDAE is not  positive"          
    