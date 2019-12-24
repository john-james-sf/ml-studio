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

The conditions module specifies the conditions that may be used in the 
validation process. Conditions differ from the Rule classes in a couple of ways.  

    * Rules describe the validity of the target object and attribute 
        being evaluated.
    * Conditions describe the validity of the rule itself. 

Two types of conditions, when and not when conditions, specify whether a 
rule is to be applied. When conditions establish when a rule should be 
applied. Not when conditions indicate when a rule should NOT be applied.

There are syntactic and semantic conditions. Syntactic conditions relate to
the type and state of a variable or property. Semantic conditions evaluate
variable values relative to other values or objects. 

    Syntactic Conditions
    --------------------
    * SyntacticCondition : Abstract base class for the following syntactic conditions.
    * isNone : Evaluates whether the argument is None.
    * isNotNone : Evaluates whether the argument is not None.
    * isEmpty : Evaluates whether the argument is empty string or whitespace.
    * isNotEmpty : Evaluates whether the argument is not empty string or whitespace.
    * isBool : Evaluates whether the argument is a Boolean.
    * isInt : Evaluates whether the argument is an integer.
    * isFloat : Evaluates whether the argument is an float.
    * isNumber : Evaluates whether the argument is a number.
    * isString : Evaluates whether the argument is a string. 
    
    Semantic Conditions
    -------------------
    * SemanticCondition : Abstract base class for the following semantic conditions.
    * IsEqual : Evaluates whether two arguments are equal  
    * IsNotEqual : Evaluates whether two arguments are not equal  
    * IsIn : Evaluates whether a is in b.    
    * IsLess : Evaluates whether a < b.
    * IsLessEqual : Evaluates whether a <= b.
    * IsGreater : Evaluates whether a > b.
    * IsGreaterEqual : Evaluates whether a >= b.
    * IsBetween : Evaluates whether argument a is between min and max. 
    * IsMatch : Evaluates whether a string matches a regex pattern.

"""
from abc import ABC, abstractmethod
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
class Condition(ABC):
    """Defines conditions to be applied to validation rules.
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
            attribute to be evaluated. Alternatively, literal string or
            numeric data.
    b : dict, str, int, float. Optional
        Dictionary containing an instance of an object and the name of the
            attribute to be evaluated. Alternatively, literal string or
            numeric data.            
    """

    def __init__(self, a, b=None):
        self.a = a
        self.b = b

    def _extract_data(self):        
        if isinstance(self.a, dict):
            instance = self.a.get('instance')
            attribute_name = self.a.get('attribute_name')
            try:
                self.a = getattr(instance, attribute_name)
            except AttributeError:
                print("{classname} has no attribute {attrname}.".format(
                    classname=instance.__class__.__name__,
                    attrname=attribute_name
                ))

        if self.b:
            if isinstance(self.b, dict):
                instance = self.b.get('instance')
                attribute_name = self.b.get('attribute_name')
                try:
                    self.b = getattr(instance, attribute_name)
                except AttributeError:
                    print("{classname} has no attribute {attrname}.".format(
                        classname=instance.__class__.__name__,
                        attrname=attribute_name
                    ))    

    @abstractmethod
    def __call__(self):
        pass

class SyntacticCondition(Condition):
    """Abstract base class for syntactic conditions with a single parameters.
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
            attribute to be evaluated. Alternatively, literal string or
            numeric data.
    b : None
        Not used
    """
    def __init__(self, a, b=None):        
        super(SyntacticCondition, self).__init__(a=a)

class IsNone(SyntacticCondition):
    """Evaluates whether a variable is None."""

    def __init__(self, a):
        super(IsNone, self).__init__(a=a)

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                 results.append(IsNone(a)())            
            return all(results)
        elif self.a is None:
            return True
        else:
            return False

class IsNotNone(SyntacticCondition):
    """Evaluates whether a variable is None."""

    def __init__(self, a):
        super(IsNotNone, self).__init__(a=a)

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                 results.append(IsNotNone(a)())            
            return all(results)
        elif self.a is not None:
            return True
        else:
            return False


class IsEmpty(SyntacticCondition):

    def __init__(self, a):
        super(IsEmpty, self).__init__(a=a)    

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsEmpty(a)())
            return all(results)
        elif self.a is None:
            return True
        elif self.a == "":
            return True
        elif IsString(self.a)():
            if self.a.isspace():
                return True
            else:
                return False
        else:
            return False

class IsNotEmpty(SyntacticCondition):

    def __init__(self, a):
        super(IsNotEmpty, self).__init__(a=a)    

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsNotEmpty(a)())
            return any(results)
        elif self.a is None:
            return False
        elif self.a == "":
            return False
        elif IsString(self.a)():
            if self.a.isspace():
                return False
            else:
                return True
        elif isinstance(self.a, (int, bool, float)):
            return True
        else:
            return True            

class IsBool(SyntacticCondition):

    def __init__(self, a):
        super(IsBool, self).__init__(a=a)    

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsBool(a)())
            return all(results)
        elif isinstance(self.a, bool):
            return True
        else:
            return False

class IsInt(SyntacticCondition):

    def __init__(self, a):
        super(IsInt, self).__init__(a=a)    

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsInt(a)())
            return all(results)
        elif isinstance(self.a, bool):
            return False
        elif isinstance(self.a, int):
            return True
        else:
            return False


class IsFloat(SyntacticCondition):

    def __init__(self, a):
        super(IsFloat, self).__init__(a=a)    

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsFloat(a)())
            return all(results)
        elif isinstance(self.a, bool):
            return False
        elif isinstance(self.a, float):
            return True
        else:
            return False
    
class IsNumber(SyntacticCondition):

    def __init__(self, a):
        super(IsNumber, self).__init__(a=a)    

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsNumber(a)())
            return all(results)
        elif isinstance(self.a, bool):
            return False
        elif isinstance(self.a, (int, float)):
            return True
        else:
            return False


class IsString(SyntacticCondition):

    def __init__(self, a):
        super(IsString, self).__init__(a=a)    

    def __call__(self):
        self._extract_data()
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsString(a)())
            return all(results)
        elif isinstance(self.a, str):
            return True
        else:
            return False   


# --------------------------------------------------------------------------- #
#                            Semantic Conditions                              #  
# --------------------------------------------------------------------------- #        
class SemanticCondition(Condition):
    """Semantic conditions involve two variables.
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
            attribute to be evaluated. Alternatively, literal string or
            numeric data.
    b : dict, str, int, float. 
        Dictionary containing an instance of an object and the name of the
            attribute to be evaluated. Alternatively, literal string or
            numeric data.            

    **kwargs : dict
        Additional keyword arguments
    
    """
    def __init__(self, a, b=None, **kwargs):
        super(SemanticCondition, self).__init__(a=a)    
        self.b = b

class IsEqual(SemanticCondition):
    def __init__(self, a, b=None):
        super(IsEqual, self).__init__(a=a,b=b)    

    def __call__(self):
        self._extract_data()

        if isArray(self.a):
            self.a = np.array(self.a)
        if isArray(self.b):
            self.b = np.array(self.b)

        if isArray(self.a) and isArray(self.b):
            if self.a.shape == self.b.shape:
                if np.array_equal(self.a, self.b):
                    return True
                else:
                    return False
            else:
                return False        
        else:
            return self.a == self.b

class IsNotEqual(SemanticCondition):
    def __init__(self, a, b=None):
        super(IsNotEqual, self).__init__(a=a,b=b)    

    def __call__(self):
        self._extract_data()

        if isArray(self.a):
            self.a = np.array(self.a)
        if isArray(self.b):
            self.b = np.array(self.b)

        if isArray(self.a) and isArray(self.b):
            if self.a.shape == self.b.shape:
                if np.array_equal(self.a, self.b):
                    return False
                else:
                    return True
            else:
                return True        
        else:
            return not self.a == self.b

class IsIn(SemanticCondition):
    """Evaluates whether a is in b.
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
        attribute to be evaluated. Alternatively, literal string or
        numeric data.

    b : dict, array-like
        Dictionary containing an instance of an object and the name of the
        attribute containing an array-like of reference values. Alternatively, 
        an array-like containing the reference values for a.  

    **kwargs : dict
        Additional keyword arguments
    
    """    
    def __init__(self, a, b=None):
        super(IsIn, self).__init__(a=a,b=b)    

    def __call__(self):
        self._extract_data()
        if isinstance(self.a, str) and isinstance(self.b, str):
            return self.a in self.b
        elif isArray(self.a):
            results = []
            for a in self.a:
                 results.append(IsIn(a, self.b)())
            return all(results)
        elif isArray(self.b):
            return self.a in self.b        
        else:
            return self.a == self.b

class IsNotIn(SemanticCondition):
    """Evaluates whether a is not in b.
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
        attribute to be evaluated. Alternatively, literal string or
        numeric data.

    b : dict, array-like
        Dictionary containing an instance of an object and the name of the
        attribute containing an array-like of reference values. Alternatively, 
        an array-like containing the reference values for a.  

    **kwargs : dict
        Additional keyword arguments
    
    """    
    def __init__(self, a, b=None):
        super(IsNotIn, self).__init__(a=a,b=b)    

    def __call__(self):
        self._extract_data()
        if isinstance(self.a, str) and isinstance(self.b, str):
            return self.a not in self.b
        elif isArray(self.a):
            results = []
            for a in self.a:
                 results.append(IsNotIn(a, self.b)())
            return any(results)
        elif isArray(self.b):
            return self.a not in self.b        
        else:
            return self.a != self.b
                

class IsLess(SemanticCondition):
    """Evaluates whether a < b.
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
        attribute to be evaluated. Alternatively, literal string or
        numeric data.

    b : dict, int, float
        Dictionary containing an instance of an object and the name of the
        attribute containing an integer or a float. Alternatively, b may be a 
        literal value.  

    inclusive : bool
        If True, then this evaluates whether a <= b. Otherwise, it evaluates
        a < b.
    
    """      

    def __init__(self, a, b=None, inclusive=True):
        super(IsLess, self).__init__(a=a,b=b)    
        self._inclusive = inclusive

    def __call__(self):
        self._extract_data()          
        if isArray(self.a) and isArray(self.b):
            # Convert array-likes to numpy arrays              
            self.a = np.array(self.a)
            self.b = np.array(self.b)
            # Shapes must be compatible for element-wise comparisons
            if self.a.shape == self.b.shape:
                if self._inclusive:
                    return all(np.less_equal(self.a, self.b))
                else:
                    return all(np.less(self.a,self.b))
            else:
                raise ValueError("If a and b are arrays, they must have the same shape.")
        elif isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsLess(a,self.b))
            return all(results)

        elif isArray(self.b):
            raise ValueError("b can be an array only when a is an array.")
        else: 
            if self._inclusive:
                return self.a <= self.b
            else:
                return self.a < self.b


class IsGreater(SemanticCondition):
    """Evaluates whether a < b.
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
        attribute to be evaluated. Alternatively, literal string or
        numeric data.

    b : dict, int, float
        Dictionary containing an instance of an object and the name of the
        attribute containing an integer or a float. Alternatively, b may be a 
        literal value.  

    inclusive : bool
        If True, then this evaluates whether a >= b. Otherwise, it evaluates
        a > b.
    
    """      

    def __init__(self, a, b=None, inclusive=True):
        super(IsGreater, self).__init__(a=a,b=b)    
        self._inclusive = inclusive

    def __call__(self):
        self._extract_data()          
        if isArray(self.a) and isArray(self.b):
            # Convert array-likes to numpy arrays              
            self.a = np.array(self.a)
            self.b = np.array(self.b)
            # Shapes must be compatible for element-wise comparisons
            if self.a.shape == self.b.shape:
                if self._inclusive:
                    return all(np.greater_equal(self.a, self.b))
                else:
                    return all(np.greater(self.a,self.b))
            else:
                raise ValueError("If a and b are arrays, they must have the same shape.")
        elif isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsGreater(a,self.b))
            return all(results)

        elif isArray(self.b):
            raise ValueError("b can be an array only when a is an array.")
        else: 
            if self._inclusive:
                return self.a >= self.b
            else:
                return self.a > self.b




class IsBetween(SemanticCondition):
    """Evaluates whether a is between b['min] and b['max']
    
    Parameters
    ----------
    a : dict, str, int, float, 
        Dictionary containing an instance of an object and the name of the
        attribute to be evaluated. Alternatively, literal string or
        numeric data.

    b : array-like
        An array-like with two elements, a minimum and maximum value.

    inclusive : bool
        If True, b is considered an inclusive range, otherwise b is an 
        exclusive range.
    
    """

    def __init__(self, a, b=None, inclusive=True):
        super(IsBetween, self).__init__(a=a,b=b)    
        self._inclusive = inclusive

    def __call__(self):
        self._extract_data()   
        # Confirm b has a length of two and has integer or float componenets
        if len(self.b) != 2:
            raise ValueError("b must be an array-like containing min and max values.")
        if not isinstance(self.b[0], (int,float)) or \
            not isinstance(self.b[1], (int,float)):
            raise TypeError("the components of b must be numbers")
        
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsBetween(a,self.b)())
            return all(results)
        elif not isinstance(self.a, (int, float)):
            raise TypeError("a must be an number or an array-like of numbers.")
        else:            
            if self._inclusive:
                if self.a >= self.b[0] and self.a <= self.b[1]:
                    return True
                else:
                    return False
            else:
                if self.a > self.b[0] and self.a < self.b[1]:
                    return True
                else:
                    return False

class IsMatch(SemanticCondition):
    """Matches a against a pattern b.

    Parameters
    ----------
    a : dict, str, 
        Dictionary containing an instance of an object and the name of the
        attribute to be evaluated. Alternatively, literal string.

    b : str
        A regex pattern in string format. 

    """

    def __init__(self, a, b=None):
        super(IsMatch, self).__init__(a=a,b=b)    

    def __call__(self):
        self._extract_data()   
        if isArray(self.a):
            results = []
            for a in self.a:
                results.append(IsMatch(a, self.b)())
            return all(results)
        else:
            matches = re.search(self.b,self.a)
            if matches:
                return True
            else:
                return False

# --------------------------------------------------------------------------- #
#                               Utility Functions                             #  
# --------------------------------------------------------------------------- #
def isArray(a, b=None):
    if isinstance(a, (pd.Series, np.ndarray, list, tuple)):
        return True
    else:
        return False