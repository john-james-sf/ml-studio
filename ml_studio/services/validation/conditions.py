#!/usr/bin/env python3
# =========================================================================== #
#                    SERVICES: VALIDATION: CONDITIONS                         #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \conditions.py                                                        #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 20th 2019, 10:45:25 pm                         #
# Last Modified: Friday December 20th 2019, 10:45:42 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the conditions used for validation.

The conditions module specifies the conditions used in the validation process.
Conditions differ from the Rule classes in a couple of ways.  

    * They are simple functions that return a Boolean value.
    * The operands may or may not be attributes of the object being validated.

There are syntactic and semantic conditions. Syntactic conditions relate to
the type and state of a variable or property. Semantic conditions evaluate
variable values relative to other values or objects. 

    Syntactic Conditions
    --------------------
    * isNone : Evaluates whether the argument is None.
    * isEmpty : Evaluates whether the argument is empty string or whitespace.
    * isBool : Evaluates whether the argument is a Boolean.
    * isInt : Evaluates whether the argument is an integer.
    * isFloat : Evaluates whether the argument is an float.
    * isNumber : Evaluates whether the argument is a number.
    * isString : Evaluates whether the argument is a string. 
    * isDate : Evaluates whether a string is a valid datetime format.
    

    Semantic Conditions
    -------------------
    * isEqual : Evaluates whether two arguments are equal  
    * isIn : Evaluates whether argument a is in argument b.
    * isLess : Evaluates whether argument a is less than argument b.
    * isGreater : Evaluates whether argument a is greater than argument b.
    * isBetween : Evaluates whether argument a is between min and maa. 
    * isMatch : Evaluates whether a string matches a regea pattern.

"""

from dateutil.parser import parse
import os
import math
import numbers
import re
import sys
import textwrap
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#                            Syntactic Conditions                             #  
# --------------------------------------------------------------------------- #
def isNone(a, b=None):
    if a is None:
        return True
    else:
        return False

def isAllNone(a, b=None):
    if isArray(a):
        y = [x for x in a if x is not None]
        if y:
            return False
        else:
            return True
    if a is None:
        return True
    else:
        return False


def isEmpty(a, b=None):
    if a is None:
        return True
    elif a == "":
        return True
    elif isString(a):
        if a.isspace():
            return True
        else:
            return False
    else:
        return False

def isAllEmpty(a, b=None):
    if isArray(a):
        y = [x for x in a if not isEmpty(x)]
        if y:
            return False
        else:
            return True
    else:
        return isEmpty(a)

def isBool(a, b=None):
    if isinstance(a, bool):
        return True
    else:
        return False

def isAllBool(a, b=None):
    if isArray(a):
        y = [x for x in a if not isBool(x)]
        if y:
            return False
        else:
            return True
    else:
        return isBool(a)


def isInt(a, b=None):
    if isinstance(a, int):
        return True
    else:
        return False

def isAllInt(a, b=None):
    if isArray(a):
        y = [x for x in a if not isInt(x)]
        if y:
            return False
        else:
            return True
    else:
        return isInt(a)


def isFloat(a, b=None):
    if isinstance(a, float):
        return True
    else:
        return False

def isAllFloat(a, b=None):
    if isArray(a):
        y = [x for x in a if not isFloat(x)]
        if y:
            return False
        else:
            return True
    else:
        return isFloat(a)


def isNumber(a, b=None):
    if isinstance(a, (int, float)):
        return True
    else:
        return False        

def isAllNumber(a, b=None):
    if isArray(a):
        y = [x for x in a if not isNumber(x)]
        if y:
            return False
        else:
            return True
    else:
        return isNumber(a)


def isString(a, b=None):
    if isinstance(a, str):
        return True
    else:
        return False        

def isAllString(a, b=None):
    if isArray(a):
        y = [x for x in a if not isString(x)]
        if y:
            return False
        else:
            return True
    else:
        return isString(a)


def isDate(a, b=None, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(a, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def isAllDate(a, b=None):
    if isArray(a):
        y = [x for x in a if not isDate(x)]
        if y:
            return False
        else:
            return True
    else:
        return isDate(a)

def isArray(a, b=None):
    if isinstance(a, (pd.Series, np.ndarray, list, tuple)):
        return True
    else:
        return False
#%%
# --------------------------------------------------------------------------- #
#                            Semantic Conditions                              #  
# --------------------------------------------------------------------------- #        
def isEqual(a,b):
    """Returns true if a == b

    Parameters
    ----------
    a : int,float, array-like
        Number to be evaluated against b

    b : int, float, array-like
        Number to be evaluated against a

    Returns
    -------
    True if a == b. False otherwise
    """      
    if isArray(a) and isArray(b):
        np_a = np.array(a)
        np_b = np.array(b)
        if np.equal(np_a, np_b):
            return True
        else:
            return False
    elif a == b:
        return True
    else:
        return False

def isIn(a, b):
    """Returns true if a in b

    Parameters
    ----------
    a : int,float, array-like
        Number to be evaluated against b

    b : int, float, array-like
        Number to be evaluated against a

    Returns
    -------
    True if a in b. False otherwise
    """     
    # Four scenarios:
    #   1: both a and b are array-like : element-wise evaluation
    #   2: a is not an array, b is : Return a in b
    #   3: a is array, b is not : Return false
    #   4: neither are arrays, Return False 
   
    if isArray(a) and isArray(b):
        if sum(np.isin(a,b)) == len(a):
            return True
        else:
            return False
    elif not isArray(a) and isArray(b):
        if a in b:
            return True
        else:
            return False
    else:
        return False

#%%
        
def isLess(a, b):
    """Returns true if a<b

    Parameters
    ----------
    a : int, float
        Number to be evaluated against b

    b : int, float
        Number to be evaluated against a

    Returns
    -------
    True if a<b. False otherwise
    """        
    if a < b:
        return True
    else:
        return False

def isLessEqual(a, b):
    """Returns true if a<=b

    Parameters
    ----------
    a : int, float
        Number to be evaluated against b

    b : int, float
        Number to be evaluated against a

    Returns
    -------
    True if a<=b. False otherwise
    """        
    if a <= b:
        return True
    else:
        return False

def isGreater(a,b):
    """Returns true if a>b

    Parameters
    ----------
    a : int, float
        Number to be evaluated against b

    b : int, float
        Number to be evaluated against a

    Returns
    -------
    True if a>b. False otherwise
    """        
    if a > b:
        return True
    else:
        return False

def isGreaterEqual(a,b):
    """Returns true if a>=b

    Parameters
    ----------
    a : int, float
        Number to be evaluated against b

    b : int, float
        Number to be evaluated against a

    Returns
    -------
    True if a>=b. False otherwise
    """    
    if a >= b:
        return True
    else:
        return False        

def isMatch(a,b):
    """Returns true if a matches regex pattern in b

    Parameters
    ----------
    a : str
        String to be evaluated vis-a-vis the regex parameter

    b : str
        The regex pattern

    Returns
    -------
    True if a matches the regex pattern in b. False otherwise.
    """
    return(re.search(b,a))




# %%
