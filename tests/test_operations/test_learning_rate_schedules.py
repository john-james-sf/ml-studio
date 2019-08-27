# --------------------------------------------------------------------------- #
#                     TEST LEARNING RATE SCHEDULES                            #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pytest
from pytest import mark

from ml_studio.operations.learning_rate_schedules import TimeDecay
from ml_studio.operations.learning_rate_schedules import NaturalExponentialDecay
from ml_studio.operations.learning_rate_schedules import ExponentialDecay
from ml_studio.operations.learning_rate_schedules import InverseScaling
from ml_studio.operations.learning_rate_schedules import PolynomialDecay
from ml_studio.operations.learning_rate_schedules import Adaptive

class LearningRateScheduleTests:

    # ----------------------------------------------------------------------- #
    #                             Time Decay                                  #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_time_decay_learning_rate_schedule_wo_steps_wo_staircase(self):
        logs = params = {}
        exp_result = [0.066666667,0.05,0.04,0.033333333,0.028571429]
        act_result = []        
        lrs = TimeDecay(decay_rate=0.5)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Time decay not working"

    @mark.learning_rate_schedules
    def test_time_decay_learning_rate_schedule_w_steps_wo_staircase(self):
        logs = params = {}
        exp_result = [0.08,0.066666667,0.057142857,0.05,0.044444444]
        act_result = []        
        lrs = TimeDecay(decay_steps=2, decay_rate=0.5)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Time decay with step not working"

    @mark.learning_rate_schedules
    def test_time_decay_learning_rate_schedule_w_steps_w_staircase(self):
        logs = params = {}
        exp_result = [0.1,0.066666667,0.066666667,0.05,0.05]
        act_result = []        
        lrs = TimeDecay(decay_steps=2, decay_rate=0.5, staircase=True)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Time decay with step and staircase not working"

    # ----------------------------------------------------------------------- #
    #                     Natural Exponential Decay                           #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_nat_exp_decay_learning_rate_schedule_wo_steps_wo_staircase(self):
        logs = params = {}
        exp_result = [0.060653066,0.036787944,0.022313016,0.013533528,0.0082085]
        act_result = []        
        lrs = NaturalExponentialDecay(decay_rate=0.5)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Natural exponential decay not working"

    @mark.learning_rate_schedules
    def test_nat_exp_decay_learning_rate_schedule_w_steps_wo_staircase(self):
        logs = params = {}
        exp_result = [0.0778800783071,0.0606530659713,0.0472366552741,0.0367879441171,0.0286504796860]
        act_result = []        
        lrs = NaturalExponentialDecay(decay_steps=2, decay_rate=0.5)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Natural exponential decay with steps not working"

    @mark.learning_rate_schedules
    def test_nat_exp_decay_learning_rate_schedule_w_steps_w_staircase(self):
        logs = params = {}
        exp_result = [0.1000000000000,0.0606530659713,0.0606530659713,0.0367879441171,0.0367879441171]
        act_result = []        
        lrs = NaturalExponentialDecay(decay_steps=2, decay_rate=0.5, staircase=True)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Natural exponential decay with steps and staircase not working"        
    
    # ----------------------------------------------------------------------- #
    #                           Exponential Decay                             #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_exp_decay_learning_rate_schedule_wo_steps_wo_staircase(self):
        logs = params = {}
        exp_result = [0.05,0.025,0.0125,0.00625,0.003125]
        act_result = []        
        lrs = ExponentialDecay(decay_rate=0.5)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Exponential decay not working"

    @mark.learning_rate_schedules
    def test_exp_decay_learning_rate_schedule_w_steps_wo_staircase(self):
        logs = params = {}
        exp_result = [0.070710678,0.05,0.035355339,0.025,0.01767767]
        act_result = []        
        lrs = ExponentialDecay(decay_rate=0.5, decay_steps=2)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Exponential decay with steps not working"

    @mark.learning_rate_schedules
    def test_exp_decay_learning_rate_schedule_w_steps_w_staircase(self):
        logs = params = {}
        exp_result = [0.1,0.05,0.05,0.025,0.025]
        act_result = []        
        lrs = ExponentialDecay(decay_rate=0.5, decay_steps=2, staircase=True)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Exponential decay with steps and staircase not working"        
    
    # ----------------------------------------------------------------------- #
    #                           Inverse Scaling                               #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_inv_scaling_learning_rate_schedule(self):
        logs = params = {}
        exp_result = [0.1,0.070710678,0.057735027,0.05,0.04472136]
        act_result = []        
        lrs = InverseScaling(power=0.5)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Inverse scaling not working"

    # ----------------------------------------------------------------------- #
    #                           Polynomial Decay                              #
    # ----------------------------------------------------------------------- #

    @mark.learning_rate_schedules
    def test_polynomial_decay_learning_rate_schedule_wo_cycle(self):
        logs = params = {}
        exp_result = [0.0895,0.0775,0.0633,0.0448,0.0001]
        act_result = []        
        lrs = PolynomialDecay(decay_steps=5, power=0.5,
                              end_learning_rate=0.0001)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(5)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Polynomial decay not working"

    @mark.learning_rate_schedules
    def test_polynomial_decay_learning_rate_schedule_w_cycle(self):
        logs = params = {}
        exp_result = [0.0895,0.0775,0.0633,0.0448,0.0001,0.0633]
        act_result = []        
        lrs = PolynomialDecay(decay_steps=5, power=0.5,
                              end_learning_rate=0.0001, cycle=True)
        params['learning_rate'] = 0.1
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(6)]
        for i in iterations:
            logs['epoch'] = i
            act_result.append(lrs(logs))
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Polynomial decay with cycle not working"   

    # ----------------------------------------------------------------------- #
    #                              Adaptive                                   #
    # ----------------------------------------------------------------------- #        

    @mark.learning_rate_schedules
    def test_adaptive_learning_rate_schedule(self):
        logs = params = {}
        exp_result = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05]
        act_result = []        
        lrs = Adaptive(decay_rate=0.5, precision=0.01, patience=5)
        logs['learning_rate'] = 0.1
        cost = [5,5,5,5,5,4,4,4,4,4,4,3]        
        lrs.set_params(params)             
        iterations =  [i+1 for i in range(12)]
        for i in iterations:
            logs['epoch'] = i
            logs['train_cost'] = cost[i-1]
            learning_rate = lrs(logs)
            act_result.append(learning_rate)
            logs['learning_rate'] = learning_rate
        assert all(np.isclose(exp_result,act_result,rtol=1e-1)), "Adaptive decay with cycle not working"             