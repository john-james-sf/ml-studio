# --------------------------------------------------------------------------- #
#                          TEST REGULARIZERS                                  #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pytest
from pytest import mark

from ml_studio.supervised_learning.training.regularizers import L1, L2, ElasticNet

class L1Tests:

    @mark.regularization
    def test_l1_cost(self, get_alpha, get_weights, get_l1_cost):
        alpha = get_alpha
        w = get_weights
        l1_cost = get_l1_cost
        with pytest.raises(AssertionError):
            l1 = L1(alpha='f')
            l1_cost_test = l1(w)
        l1 = L1(alpha=alpha)
        l1_cost_test = l1(w)
        assert math.isclose(l1_cost, l1_cost_test, abs_tol=10**-4)
    
    @mark.regularization
    def test_l1_gradient(self, get_alpha, get_weights, get_l1_grad):
        alpha = get_alpha
        w = get_weights
        l1_grad = get_l1_grad
        with pytest.raises(AssertionError):
            l1 = L1(alpha='f')
            l1_grad_test = l1.gradient(w)
        l1 = L1(alpha=alpha)
        l1_grad_test = l1.gradient(w)
        for a,b in zip(l1_grad, l1_grad_test):
            assert math.isclose(a, b, abs_tol=10**-4)


class L2Tests:

    @mark.regularization
    def test_l2_cost(self, get_alpha, get_weights, get_l2_cost):
        alpha = get_alpha
        w = get_weights
        l2_cost = get_l2_cost
        with pytest.raises(AssertionError):
            l2 = L2(alpha='f')
            l2_cost_test = l2(w)
        l2 = L2(alpha=alpha)
        l2_cost_test = l2(w)
        assert math.isclose(l2_cost, l2_cost_test, abs_tol=10**-4)
    
    @mark.regularization
    def test_l2_gradient(self, get_alpha, get_weights, get_l2_grad):
        alpha = get_alpha
        w = get_weights
        l2_grad = get_l2_grad
        with pytest.raises(AssertionError):
            l2 = L2(alpha='f')
            l2_grad_test = l2.gradient(w)
        l2 = L2(alpha=alpha)
        l2_grad_test = l2.gradient(w)
        for a,b in zip(l2_grad, l2_grad_test):
            assert math.isclose(a, b, abs_tol=10**-4)            

class ElasticNetTests:

    @mark.regularization
    def test_elasticnet_cost(self, get_alpha, get_ratio, get_weights, get_elasticnet_cost):
        alpha = get_alpha
        ratio = get_ratio
        w = get_weights
        elasticnet_cost = get_elasticnet_cost
        with pytest.raises(AssertionError):
            elasticnet = ElasticNet(alpha='f')
            elasticnet_cost_test = elasticnet(w)
        elasticnet = ElasticNet(alpha=alpha, ratio=ratio)
        elasticnet_cost_test = elasticnet(w)
        assert math.isclose(elasticnet_cost, elasticnet_cost_test, abs_tol=10**-4)
    
    @mark.regularization
    def test_elasticnet_gradient(self, get_alpha, get_ratio, get_weights, get_elasticnet_grad):
        alpha = get_alpha
        w = get_weights
        ratio = get_ratio
        elasticnet_grad = get_elasticnet_grad
        with pytest.raises(AssertionError):
            elasticnet = ElasticNet(alpha='f')
            elasticnet_grad_test = elasticnet.gradient(w)
        elasticnet = ElasticNet(alpha=alpha, ratio=ratio)
        elasticnet_grad_test = elasticnet.gradient(w)
        for a,b in zip(elasticnet_grad, elasticnet_grad_test):
            assert math.isclose(a, b, abs_tol=10**-4)              

