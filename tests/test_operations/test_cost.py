# --------------------------------------------------------------------------- #
#                          TEST COST FUNCTIONS                                #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pytest
from pytest import mark

from ml_studio.supervised_learning.training.cost import Quadratic
from ml_studio.supervised_learning.training.cost import BinaryCrossEntropy
from ml_studio.supervised_learning.training.cost import CategoricalCrossEntropy

class QuadraticCostTests:

    @mark.cost
    def test_quadratic_cost(self, get_quadratic_y, get_quadratic_y_pred, get_quadratic_cost):
        y = get_quadratic_y
        y_pred = get_quadratic_y_pred
        J = get_quadratic_cost
        J_test = 1/2 * np.mean((y_pred-y)**2)
        assert math.isclose(J, J_test, abs_tol=1)
    
    @mark.cost
    def test_quadratic_cost_gradient(self, get_quadratic_X, get_quadratic_y, get_quadratic_y_pred, get_quadratic_gradient):
        y = get_quadratic_y
        y_pred = get_quadratic_y_pred
        X = get_quadratic_X
        grad = get_quadratic_gradient
        grad_test = 1/y.shape[0] * (y_pred- y).dot(X)        
        for a,b in zip(grad, grad_test):
            assert math.isclose(a, b, abs_tol=1.0)

class BinaryCostTests:

    @mark.cost
    def test_binary_cost(self, get_binary_cost_y, get_binary_cost_y_pred, get_binary_cost):
        y = get_binary_cost_y
        y_pred = get_binary_cost_y_pred
        J = get_binary_cost
        J_test = -1*(1/y.shape[0] * np.sum(np.multiply(y,np.log(y_pred), np.multiply((1-y),np.log(1-y_pred)))))
        assert math.isclose(J, J_test, abs_tol=10**4)

    @mark.cost
    def test_binary_cost_gradient(self, get_binary_cost_X, get_binary_cost_y, get_binary_cost_y_pred, get_binary_cost_gradient):
        X = get_binary_cost_X
        y = get_binary_cost_y
        y_pred = get_binary_cost_y_pred
        grad = get_binary_cost_gradient
        grad_test = X.T.dot(y_pred-y)        
        for a,b in zip(grad, grad_test):
            assert math.isclose(a, b, abs_tol=1.0)

class CategoricalCostTests:

    @mark.cost
    def test_categorical_cost(self, get_categorical_cost_y, get_categorical_cost_y_pred, get_categorical_cost):
        y = get_categorical_cost_y
        y_pred = get_categorical_cost_y_pred
        J = get_categorical_cost
        J_test = -1*(1/y.shape[0] * np.sum(np.multiply(y,np.log(y_pred), np.multiply((1-y),np.log(1-y_pred)))))
        assert math.isclose(J, J_test, abs_tol=10**4)

    @mark.cost
    def test_categorical_cost_gradient(self, get_categorical_cost_X, get_categorical_cost_y, get_categorical_cost_y_pred,
                                       get_categorical_cost_gradient):
        X = get_categorical_cost_X
        y = get_categorical_cost_y
        y_pred = get_categorical_cost_y_pred
        grad = get_categorical_cost_gradient
        grad_test = 1/y.shape[0] * X.T.dot(y_pred-y)        
        for array_a,array_b in zip(grad, grad_test):
            for a, b in zip(array_a, array_b):
                assert math.isclose(a, b, abs_tol=1.0)
