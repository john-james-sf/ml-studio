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
# Create Date: Saturday December 28th 2019, 6:31:33 am                        #
# Last Modified: Saturday December 28th 2019, 8:50:48 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Classes that manage conditions which are applied to Rule objects.

This module contains the following classes:

    * BaseCondition : The component interface for all Condition classes.
    * ConditionSet : A collection of conditions.
    * Condition : Defines a unique condition.

"""
from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
import getpass
import os
import re
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from ml_studio.services.validation.utils import is_array, is_homogeneous_array
from ml_studio.services.validation.utils import is_simple_array, is_none
from ml_studio.services.validation.utils import is_not_none, is_empty
from ml_studio.services.validation.utils import is_not_empty, is_bool
from ml_studio.services.validation.utils import is_integer, is_number
from ml_studio.services.validation.utils import is_string, is_less
from ml_studio.services.validation.utils import is_less_equal, is_greater
from ml_studio.services.validation.utils import is_greater_equal, is_match
from ml_studio.services.validation.utils import is_equal, is_not_equal
# --------------------------------------------------------------------------- #
#                                 BASECONDITION                               #
# --------------------------------------------------------------------------- #
class BaseCondition(ABC):
    """Abstract base class for all Condition classes."""
    
    def __init__(self):
        # Designate unique/opaque userid and other metadata        
        self._id = uuid4()
        self._created = datetime.now()
        self._user = getpass.getuser()
        self._is_valid = True
        self._instance = None
        self._value = None

    def on(self, instance, value):
        """Specifies the instance of the class and the value being evaluated.
        
        Parameters
        ----------
        instance : Any
            The object being evaluated.

        value : Any
            The value being evaluated.
        """
        self._instance = instance
        self._value = value

    @abstractmethod
    def evaluate(self):
        pass

# --------------------------------------------------------------------------- #
#                                CONDITIONSET                                 #
# --------------------------------------------------------------------------- #    
class ConditionSet(BaseCondition):
    """Collection of Condition objects."""

    def __init__(self):
        super(ConditionSet, self).__init__()
        self._logical = 'all'
        self._conditions = set()

    @property
    def is_valid(self):
        return self._is_valid
        
    @property
    def logical(self):
        return self._logical

    @logical.setter
    def all(self):
        self._logical = "all"

    @logical.setter
    def any(self):
        self._logical = "any" 

    @logical.setter
    def none(self):
        self._logical = "none"                
    
    def evaluate(self):
        results = []
        evaluation = dict({"all": all(results),
                          "any": any(results),
                          "none": bool(not any (results))})
        for condition in self._conditions:
            results.append(condition.evaluate().is_valid)
        return evaluation.get(self._logical)

    def add_condition(self, condition):
        self._conditions.add(condition)

    def remove_condition(self, condition):
        self._conditions.discard(condition)

# --------------------------------------------------------------------------- #
#                                 CONDITION                                   #
# --------------------------------------------------------------------------- #           
class Condition(BaseCondition):
    """Class that defines a condition to apply to Rule objects."""

    def __init__(self):
        super(Condition, self).__init__()
        self._a = None
        self._b = None        
        self._attribute = None
        self._eval_function = None

    def _reset(self):
        self._a = None
        self._b = None
        self._instance = None
        self._attribute = None
        self._eval_function = None
        

    def on(self, instance, attribute):
        self._instance = instance
        self._attribute = attribute
        return self

    @property
    def when_value(self):
        try:
            self._a = getattr(self._instance, self._attribute)
        except AttributeError:
            print("{attr} is not a valid attribute for the {classname} class.".format(
                attr=self._attribute,
                classname=self._instance.__class__.__name__
            ))
        return self

    @when_value.setter
    def when_value(self):
        self._a = self._value
        return self

    @property
    def when_attribute(self):        
        return self

    @when_attribute.setter
    def when_attribute(self, attribute_name):        
        try:            
            self._a = getattr(self._instance, attribute_name)
        except AttributeError:
            print("{attrname} is not an attribute of {classname}.".format(
                attrname=attribute_name,
                classname=self._instance.__class__.__name__
            ))
        return self

    @property
    def is_none(self):
        self._eval_function = is_none
        return self

    @property
    def is_not_none(self):
        self._eval_function = is_not_none
        return self

    @property
    def is_empty(self):
        self._eval_function = is_empty        
        return self
    
    @property
    def is_not_empty(self):
        self._eval_function = is_not_empty         
        return self

    @property
    def is_bool(self):
        self._eval_function = is_bool        
        return self
    
    @property
    def is_integer(self):
        self._eval_function = is_integer          
        return self

    @property
    def is_number(self):
        self._eval_function = is_number
        return self

    @property
    def is_string(self):
        self._eval_function = is_string         
        return self

    def _get_value(self,b):
        if isinstance(b, str):
            try:
                self._b = getattr(self._instance, b)
            except AttributeError:        
                self._b = b                
        else:
            self._b = b

    def is_equal(self, b):
        self._get_value(b)
        self._eval_function = is_equal
        return self

    def is_not_equal(self, b):
        self._get_value(b)
        self._eval_function = is_not_equal        
        return self

    def is_less(self, b):
        self._get_value(b)
        self._eval_function = is_less
        return self

    def is_less_equal(self, b):
        self._get_value(b)
        self._eval_function = is_less_equal        
        return self

    def is_greater(self, b):
        self._get_value(b)
        self._eval_function = is_greater
        return self

    def is_greater_equal(self, b):
        self._get_value(b)
        self._eval_function = is_greater_equal        
        return self

    def is_match(self, b):
        self._get_value(b)
        self._eval_function = is_match
        return self

    @property
    def evaluate(self):
        result = self._eval_function(self._a, self._b)
        self._reset()
        if result:
            return True
        else:
            return False

            