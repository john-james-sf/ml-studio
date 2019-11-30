# --------------------------------------------------------------------------- #
#                     TEST LEARNING RATE SCHEDULES                            #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pytest
from pytest import mark

from ml_studio.supervised_learning.training.learning_rate_schedules import TimeDecay
from ml_studio.supervised_learning.training.learning_rate_schedules import StepDecay
from ml_studio.supervised_learning.training.learning_rate_schedules import NaturalExponentialDecay
from ml_studio.supervised_learning.training.learning_rate_schedules import ExponentialDecay
from ml_studio.supervised_learning.training.learning_rate_schedules import InverseScaling
from ml_studio.supervised_learning.training.learning_rate_schedules import PolynomialDecay
from ml_studio.supervised_learning.training.learning_rate_schedules import Adaptive
from ml_studio.supervised_learning.regression import LinearRegression

class LearningRateScheduleTests:

    # ----------------------------------------------------------------------- #
    #                             Time Decay                                  #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    @mark.learning_rate_schedules_time
    def test_time_decay_learning_rate_schedule_wo_staircase(self, get_regression_data):
        exp_result = [0.0909090909,	0.0833333333,	0.0769230769,	0.0714285714,	0.0666666667]
        act_result = []        
        lrs = TimeDecay(learning_rate=0.1, decay_rate=0.5, decay_steps=5)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:  
            lrs.on_epoch_end(i)          
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Time decay not working"

    @mark.learning_rate_schedules
    @mark.learning_rate_schedules_time_staircase
    def test_time_decay_learning_rate_schedule_w_staircase(self, get_regression_data):
        exp_result = [0.1000000000,	0.1000000000,	0.1000000000,	0.1000000000,	0.0666666667]
        act_result = []        
        lrs = TimeDecay(learning_rate=0.1, decay_steps=5, decay_rate=0.5, staircase=True)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Time decay with step not working"

    # ----------------------------------------------------------------------- #
    #                             Step Decay                                  #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    @mark.learning_rate_schedules_step
    def test_step_decay_learning_rate_schedule(self, get_regression_data):
        exp_result = [0.1000000000,	0.1000000000,	0.1000000000,	0.0500000000,	0.0500000000]
        act_result = []        
        lrs = StepDecay(learning_rate=0.1, decay_rate=0.5, decay_steps=5)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Step decay not working"

    # ----------------------------------------------------------------------- #
    #                     Natural Exponential Decay                           #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    @mark.learning_rate_schedules_nat_exp_no_staircase
    def test_nat_exp_decay_learning_rate_schedule_wo_staircase(self, get_regression_data):
        exp_result = [0.0904837418,0.0818730753,0.0740818221,0.0670320046,0.0606530660]
        act_result = []        
        lrs = NaturalExponentialDecay(learning_rate=0.1, decay_rate=0.5, decay_steps=5)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Natural exponential decay not working"    

    @mark.learning_rate_schedules
    @mark.learning_rate_schedules_nat_exp_staircase
    def test_nat_exp_decay_learning_rate_schedule_w_staircase(self, get_regression_data):
        exp_result = [0.1000000000,	0.1000000000,	0.1000000000,	0.1000000000,	0.0606530660]
        act_result = []        
        lrs = NaturalExponentialDecay(learning_rate=0.1, decay_steps=5, decay_rate=0.5,
                                      staircase=True)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Natural exponential decay with steps not working"

    # ----------------------------------------------------------------------- #
    #                           Exponential Decay                             #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_exp_decay_learning_rate_schedule_wo_staircase(self, get_regression_data):
        exp_result = [0.0870550563,	0.0757858283,	0.0659753955,	0.0574349177,	0.0500000000]
        act_result = []        
        lrs = ExponentialDecay(learning_rate=0.1, decay_rate=0.5, decay_steps=5)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Exponential decay not working"

    @mark.learning_rate_schedules
    def test_exp_decay_learning_rate_schedule_w_staircase(self, get_regression_data):
        exp_result = [0.1,0.1,0.1,0.1,0.05]
        act_result = []        
        lrs = ExponentialDecay(learning_rate=0.1, decay_rate=0.5, decay_steps=5, staircase=True)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Exponential decay with steps and staircase not working"       
    
    # ----------------------------------------------------------------------- #
    #                           Inverse Scaling                               #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_inv_scaling_learning_rate_schedule(self, get_regression_data):
        exp_result = [0.1,0.070710678,0.057735027,0.05,0.04472136]
        act_result = []        
        lrs = InverseScaling(learning_rate=0.1, power=0.5)    
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Inverse scaling not working"

    # ----------------------------------------------------------------------- #
    #                           Polynomial Decay                              #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_polynomial_decay_learning_rate_schedule_wo_cycle(self, get_regression_data):
        exp_result = [0.0895,0.0775,0.0633,0.0448,0.0001]
        act_result = []        
        lrs = PolynomialDecay(learning_rate=0.1, decay_steps=5, power=0.5,
                              end_learning_rate=0.0001)
        lrs.model = LinearRegression()
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Polynomial decay not working"
        

    @mark.learning_rate_schedules
    def test_polynomial_decay_learning_rate_schedule_w_cycle(self, get_regression_data):        
        exp_result = [0.0895,0.0775,0.0633,0.0448,0.0001]
        act_result = []        
        lrs = PolynomialDecay(learning_rate=0.1, decay_steps=5, power=0.5,
                              end_learning_rate=0.0001, cycle=True)
        lrs.model = LinearRegression()                              
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            lrs.on_epoch_end(i)
            act_result.append(lrs.model.eta)
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Polynomial decay with cycle not working"   

    # ----------------------------------------------------------------------- #
    #                              Adaptive                                   #
    # ----------------------------------------------------------------------- #        

    @mark.learning_rate_schedules
    @mark.learning_rate_schedules_adaptive
    def test_adaptive_learning_rate_schedule(self, get_regression_data):
        logs = {}
        exp_result = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05]
        act_result = []        
        lrs = Adaptive(learning_rate=0.1, decay_rate=0.5, precision=0.01, patience=5)
        lrs.model = LinearRegression()
        lrs.model.eta = 0.1             
        logs['learning_rate'] = 0.1
        cost = [5,5,5,5,4,4,4,4,4,4,4, 3]        
        iterations =  [i+1 for i in range(12)]
        for i in iterations:            
            logs['train_cost'] = cost[i-1]
            lrs.on_epoch_end(i, logs)
            act_result.append(lrs.model.eta)            
            logs['learning_rate'] = lrs.model.eta
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Adaptive decay with cycle not working"             
