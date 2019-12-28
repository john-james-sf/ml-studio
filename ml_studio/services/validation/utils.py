#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \utils.py                                                             #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 28th 2019, 7:41:40 am                        #
# Last Modified: Saturday December 28th 2019, 11:14:40 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
#                        VALIDATION : UTILITY FUNCTIONS                       #
# =========================================================================== #
import numpy as np
import pandas as pd
import re

# --------------------------------------------------------------------------- #
#                       SYNTACTIC EVALUATION FUNCTIONS                        #
# --------------------------------------------------------------------------- #

def is_array(a):
    return isinstance(a, (np.ndarray, set, list, tuple, pd.Series))

def is_homogeneous_array(a):
    return isinstance(a, (np.ndarray,pd.Series))

def is_simple_array(a):
    return isinstance(a, (list, set, tuple))

def is_none(a, b=None):
    if is_array(a):
        return all(is_none(elem) for elem in a)
    else:
        return a is None

def is_not_none(a, b=None):
    if is_array(a):
        return any(is_not_none(elem) for elem in a)
    else:
        return a is not None

def is_empty(a, b=None):
    if is_array(a):
        return all(is_empty(elem) for elem in a)
    elif a == "":
        return True
    elif isinstance(a, str):
        return a.isspace()
    else:
        return False

def is_not_empty(a, b=None):
    if is_array(a):
        return any(is_not_empty(elem) for elem in a)
    elif a == "":
        return False
    elif isinstance(a, str):
        return not a.isspace()
    else:
        return True
            
def is_bool(a, b=None):
    if is_array(a):
        return all(is_bool(elem) for elem in a)
    else:
        return isinstance(a, bool)

def is_integer(a, b=None):
    if is_array(a):
        return all(is_integer(elem) for elem in a)
    else:
        return isinstance(a, int)

def is_number(a, b=None):
    if is_array(a):
        return all(is_number(elem) for elem in a)
    else:
        return isinstance(a, (int,float))

def is_string(a, b=None):
    if is_array(a):
        return all(is_string(elem) for elem in a)
    else:
        return isinstance(a, str)          

# --------------------------------------------------------------------------- #
#                        SEMANTIC EVALUATION FUNCTIONS                        #
# --------------------------------------------------------------------------- #              

def compare(a,b, func=None):
    # Handle both arrays
    if is_array(a) and is_array(b):
        a_np = np.array(a)
        b_np = np.array(b)
        return all(func(a_np, b_np))
    
    # Handle a is array and b is basic type
    elif is_array(a):
        return all(compare(elem, b, func=func) \
                   for elem in a)

    # Handle a is basic and b is an array
    elif not is_array(a) and is_array(b):                
        return all(func(a,b))
    
    # Handle both a and b are non-arrays
    elif not is_array(a) and not is_array(b):
        if isinstance(a, (int, float)) and isinstance(b, (int,float)):
            return func(a,b)
        else:
            if not isinstance(a, (int, float)) and not isinstance(b, (int,float)):
                raise TypeError("Invalid types. The is_less function \
                    operates on numbers only. \n   a type: {atype}\n   b type: {btype}".format(
                        atype=type(a),
                        btype=type(b)
                    ))
            elif not isinstance(a, (int, float)):
                raise TypeError("Invalid types. The is_less function \
                    operates on numbers only. \n   a type: {atype}".format(
                        atype=type(a)
                    ))
            else:
                raise TypeError("Invalid types. The is_less function \
                    operates on numbers only. \n   b type: {btype}".format(
                        btype=type(b)
                    ))
   
def is_less(a,b):
    func = np.less
    if compare(a,b, func=func):
        return True
    else:
        return False
    
def is_less_equal(a,b):
    func = np.less_equal
    if compare(a,b, func=func):
        return True
    else:
        return False
        
def is_greater(a,b):
    func = np.greater
    if compare(a,b, func=func):
        return True
    else:
        return False
    
def is_greater_equal(a,b):
    func = np.greater_equal
    if compare(a,b, func=func):
        return True
    else:
        return False

def is_match(a,b):
    """Evaluates if a or elements of a are a regex match to the pattern in b."""
    if is_array(a):
        return all(is_match(a=elem,b=b) for elem in a)
    else:
        try:
            result = bool(re.search(b,a))
        except ValueError as e:
            print(e)
        else:
            if result:
                return True
            else:
                return False
