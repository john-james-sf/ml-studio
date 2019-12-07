# =========================================================================== #
#                                VALIDITY                                     #
# =========================================================================== #
# =========================================================================== #
# Project: Visualate                                                          #
# Version: 0.1.0                                                              #
# File: \model_validation.py                                                  #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Thursday November 28th 2019, 1:49:18 pm                        #
# Last Modified: Thursday November 28th 2019, 1:50:37 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Model validation and verification module.""" 
#%%
import math
from math import erf
import numpy as np
from scipy.stats import norm
# --------------------------------------------------------------------------- #
#                                 LEVERAGE                                    #
# --------------------------------------------------------------------------- #
def leverage(X):
    """Computes leverage.

    Leverage is a measure of how far away an independent variable values of an 
    observation are from those of other observations.
    """
    hat = X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T)) 
    hii = np.diagonal(hat)
    return hii
print(leverage.__doc__)

# --------------------------------------------------------------------------- #
#                           RESIDUAL ANALYSIS                                 #
# --------------------------------------------------------------------------- #
def standardized_residuals(model, X, y, return_predictions=False):
    """Computes standardized residuals.

    Standardized residuals (sometimes referred to as "internally studentized 
    residuals") are defined for each observation, i = 1, ..., n as an 
    ordinary residual divided by an estimate of its standard deviation:
    ..math:: r_i = \\frac{e_i}{\\sqrt{MSE(1-h_{ii})}}

    Parameters
    ----------
    model : Estimator or BaseEstimator
        ML Studio or Scikit Learn estimator

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values 
    """
    # Compute residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Compute Leverage
    hii = leverage(X)
    
    # Compute degrees of freedom and MSE
    rank = np.linalg.matrix_rank(X)
    df = X.shape[0] - rank
    mse = np.matmul(residuals, residuals) / df

    # Calculate standardized 
    standardized_residuals = residuals / np.sqrt(mse  * (1-hii))
    
    # Return standardized residuals and optionally the predictions
    if return_predictions:
        return standardized_residuals, y_pred
    else:
        return standardized_residuals

def studentized_residuals(model, X, y, return_predictions=False):
    """Computes studentized residuals.

    Studentized residuals are just a deleted residual divided by its estimated
    standard deviation. This turns out to be equivalent to the ordinary residual
    divided by a factor that includes the mean square error based on the 
    estimated model with the ith observation deleted, MSE(i), and the leverage, hii
    .. math:: r_i = \\frac{e_i}{\\sqrt{MSE_{(i)}(1-h_{ii})}}

    Parameters
    ----------
    model : Estimator or BaseEstimator
        ML Studio or Scikit Learn estimator

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values 
    """
    # Compute residuals
    y_pred = model.predict(X)
    
    # Using the calculation from 
    # https://newonlinecourses.science.psu.edu/stat462/node/247/
    n = X.shape[0]
    k = X.shape[1]
    r = standardized_residuals(model=model, X=X, y=y)

    # Calculate studentized residuals 
    studentized_residuals = r * np.sqrt((n-k-2)/(n-k-1-np.square(r)))

    # Return studentized residuals and optionally the predictions
    if return_predictions:
        return studentized_residuals, y_pred
    else:
        return studentized_residuals   

# --------------------------------------------------------------------------- #
#                  INVERSE CUMULATIVE DISTRIBUTION FUNCTION                   #
# --------------------------------------------------------------------------- #
def quantile(p):
    """Inverse Cumulative Distribution Function for Normal Distribution.

    The cumulative distribution function (CDF) of the random variable X
    has the following definition: 
    .. math:: 
                    \\mathbb{F}_X(t) = \\mathbb{P}(X \\le t)
    The notation \\mathbb{F}_X(t) means that \\mathbb{F} is the cdf for 
    the random variable \\mathbb{X} but it is a function of t. It can be 
    defined for any kind of random variable (discrete, continuous, and 
    mixed).

    Parameters
    ----------
    p : Array-like
        Sample vector of real numbers

    Note
    ----
    The original function was obtained from the google courtesy of
    Dr. John Cook and his group of consultants. The following code 
    first appeared as A literate program to compute the inverse of 
    the normal CDF. See that page for a detailed explanation of the 
    algorithm. Ultimately had to swap it out because it could only
    handle positive values.
    Source: Author  : John D. Cook
             Date   : December 1, 2019
            Title   : Inverse Normal CDF
          Website   : https://www.johndcook.com/blog/python_phi_inverse/
    
    The second algorithm was obtained from stackoverflow
    https://stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution
    """  
    #'Cumulative distribution function for the standard normal distribution'
    #i_cdf = normal_CDF_inverse(p)    
    return (1.0 + erf(p / np.sqrt(2.0))) / 2.0    
    #i_cdf = normal_CDF_inverse(p)    

def rational_approximation(t):

    # Abramowitz and Stegun formula 26.2.23.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    numerator = (c[2]*t + c[1])*t + c[0]
    denominator = ((d[2]*t + d[1])*t + d[0])*t + 1.0
    return t - numerator / denominator
    

def normal_CDF_inverse(p):

    assert p > 0.0 and p < 1

    # See article above for explanation of this section.
    if p < 0.5:
        # F^-1(p) = - G^-1(p)
        return -rational_approximation( math.sqrt(-2.0*math.log(p)) )
    else:
        # F^-1(p) = G^-1(1-p)
        return rational_approximation( math.sqrt(-2.0*math.log(1.0-p)) )        

