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
# Last Modified: Saturday December 28th 2019, 6:32:28 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
#                        VALIDATION : UTILITY FUNCTIONS                       #
# =========================================================================== #
import numpy as np
import operator
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
def compare_numbers(a, b, func):
    answer = func['number'](a,b).all()
    return answer    

def compare_strings(a, b, func):
    answer = func['string'](a,b).all()
    return answer
    
def compare(a,b, func=None):
    # Convert input to numpy arrays to use numpy comparison functions.
    a = np.array(a)
    b = np.array(b)
    # Attempt to compare, first as numbers, then as strings.
    answer = None
    try:
        answer = compare_numbers(a, b, func)
    except(TypeError):
        pass        
    if answer is None:
        try:
            answer = compare_strings(a, b, func)        
        except(TypeError):
            raise TypeError("Unable to compare values as strings or numbers.")

    # when we have but one boolean in the response
    if answer:
        return True
    else:
        return False

def is_equal(a,b):
    func = dict({'number':np.equal, 'string':np.char.equal})
    answer = compare(a,b, func=func)
    if answer:
        return True
    else:
        return False

def is_not_equal(a,b):
    func = dict({'number':np.equal, 'string':np.char.equal})
    answer = compare(a,b, func=func)
    if answer:
        return False
    else:
        return True

def is_less(a,b):
    func = dict({'number':np.less, 'string':np.char.less})
    answer = compare(a,b, func=func)
    if answer:
        return True
    else:
        return False

def is_less_equal(a,b):
    func = dict({'number':np.less_equal,  'string':np.char.less_equal})
    answer = compare(a,b, func=func)
    if answer:
        return True
    else:
        return False

def is_greater(a,b):
    func = dict({'number':np.greater, 'string':np.char.greater})
    answer = compare(a,b, func=func)
    if answer:
        return True
    else:
        return False

def is_greater_equal(a,b):
    func = dict({'number':np.greater_equal, 'string':np.char.greater_equal})
    answer = compare(a,b, func=func)
    if answer:
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
