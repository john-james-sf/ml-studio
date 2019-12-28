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
# Last Modified: Saturday December 28th 2019, 9:33:12 am                      #
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

    @property
    def when_value(self):
        return self._a

    @when_value.setter
    def when_value(self):
        self._a = self._value

    @property
    def when_attribute(self):        
        return self._attribute

    def when_attribute(self, attribute_name):
        try:
            self._attribute = attribute_name
            self._a = getattr(self._instance, attribute_name)
        except AttributeError:
            print("{attrname} is not an attribute of {classname}.".format(
                attrname=attribute_name,
                classname=self._instance.__class__.__name__
            ))
            

    
    