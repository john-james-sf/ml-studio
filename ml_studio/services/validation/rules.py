#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \rules.py                                                             #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 20th 2019, 5:06:57 am                          #
# Last Modified: Sunday December 29th 2019, 6:59:39 am                        #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
#                     SERVICES : VALIDATION : RULES                           #
# =========================================================================== #
"""Module contains the validation rules.

This module implements the composite design pattern with a fluent interface,
and includes:

    Rule Classes
    --------
    * BaseRule : Abstract base class defines the interface for the Rule Classes
    * RuleSet : The composite class collection of Rule objects.
    * Rule : The leaf class that contains each Rule object.

"""

from abc import ABC, abstractmethod, abstractproperty
import builtins
from collections import defaultdict
from collections.abc import Iterable
import copy
from dateutil.parser import parse
from datetime import datetime
import getpass
import math
import numbers
import operator
import os
import re
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from ml_studio.services.validation.conditions import BaseCondition
from ml_studio.services.validation.conditions import Condition, ConditionSet
from ml_studio.services.validation.utils import is_none, is_not_none, is_empty
from ml_studio.services.validation.utils import is_not_empty, is_bool
from ml_studio.services.validation.utils import is_integer, is_number
from ml_studio.services.validation.utils import is_string, is_less
from ml_studio.services.validation.utils import is_less_equal, is_greater
from ml_studio.services.validation.utils import is_greater_equal, is_match
from ml_studio.services.validation.utils import is_equal, is_not_equal

# --------------------------------------------------------------------------- #
#                                   RULE                                      #  
# --------------------------------------------------------------------------- #
class BaseRule(ABC):
    """Base class for all rules."""

    def __init__():

        # Designate unique/opaque userid and other metadata        
        self._id = str(uuid4())
        self._created = datetime.now()
        self._creator = getpass.getuser()

        # Class and attribute to which this rule applies
        self._is_valid = True
        self._invalid_messages = []
        self._evaluated_instance = None
        self._evaluated_attribute = None

    @property
    def id(self):
        return copy.deepcopy(self._id)

    @property
    def on(self):
        return copy.deepcopy(self._evaluated_instance)
        
    def on(self, value):
        self._evaluated_instance = copy.deepcopy(value)
        {v.on(copy.deepcopy(value)) for (_,v) in self._rules.items()}
        return self

    @property
    def attribute(self):
        return copy.deepcopy(self._evaluated_attribute)

    @attribute.setter
    def attribute(self, value):
        try:
            self._evaluated_attribute == getattr(self._evaluated_instance, value)
        except AttributeError:
            print("{attr} is not a valid attribute for the {classname} class.".format(
                attr=value, classname=self._evaluated_instance.__class__.__name__
            ))        
        else:
            {v.attribute=copy.deepcopy(value) for (_,v) in self._rules.items()}
        
        return self

    @property
    def is_valid(self):
        return self._is_valid

    @abstractmethod
    def evaluate(self):
        pass

# --------------------------------------------------------------------------- #
#                               RULESET                                       #  
# --------------------------------------------------------------------------- #       
class RuleSet(BaseRule):
    """A RuleSet contains Rules and a logical operator for evaluation.
    
    This is the composite class containing individual Rules and 
    RuleSets. Individual Rules will be evaluated independently and all must
    pass the evaluation in order for the RuleSet to apply. RuleSets are
    evaluated in accordance with their logical operator.
    
    """  

    def __init__(self):
        super(RuleSet, self).__init__()
        self._logical = "all"
        self._rules = {}
        self._conditions = {}

    def reset(self):
        self._rules = {}

    @property
    def when_all_rules_are_true(self):
        self._logical = "all"
        return self

    @property
    def when_any_rule_is_true(self):
        self._logical = "any"
        return self        

    @property
    def when_no_rules_are_true(self):
        self._logical = "none"
        return self      

    @property
    def evaluate(self):

        # Update the Rules' Conditions and ConditionSets by evaluation.
        ({self.add_rule(rule.evaluate) for _, rule in self._rules.items()})

        # Obtain evaluation results from each Rule or RuleSet in the set.
        is_valid = [v.is_valid for (_,v) in self._rules.items()]  

        # Execute appropriate logical
        if self._logical == "all":
            self._is_valid = all(is_valid)
        elif self._logical == "any":
            self._is_valid = any(is_valid)
        else:
            self._is_valid = not any(is_valid)

        return self              

    def get_rule(self, rule_id):
        return self._rules[rule_id]   

    def add_rule(self, rule):           
        self._rules[rule.id] = rule 
        return self

    def remove_rule(self, rule_id):
        del self._rules[rule_id]
        return self

    def get_condition(self, condition_id):
        return self._conditions[condition_id]   

    def add_condition(self, condition):           
        self._conditions[condition.id] = condition 
        return self

    def remove_condition(self, condition_id):
        del self._conditions[condition_id]
        return self

    @property
    def print_rules(self):
        print("\RuleSet: {id} evaluates to true when {logical} rules pass.".format(
            id=self._id, 
            logical=self._logical
        ))
        for _, rule in self._rules.items():
            rule.print_rule
        return self        

    @property
    def print_conditions(self):
        print("\ConditionSet: {id} evaluates to true when {logical} conditions pass.".format(
            id=self._id, 
            logical=self._logical
        ))
        for _, condition in self._conditions.items():
            condition.print_condition
        return self                

# --------------------------------------------------------------------------- #
#                                 RULE                                        #  
# --------------------------------------------------------------------------- #       
class Rule(BaseRule):
    """Class that defines a rule to apply to the validation of an object."""

    def __init__(self):
        super(Rule, self).__init__()
        self._a = None
        self._b = None
        self._a_attribute = None
        self._b_attribute = None                
        self._eval_function = None
        self._is_valid = "Not evaluated."

    def _reset(self):
        self._is_valid = "Not evaluated."        

    def when(self, value):
        self._a = value
        return self

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

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

    def is_equal(self, b):
        self._b = b
        self._eval_function = is_equal
        return self

    def is_not_equal(self, b):
        self._b = b
        self._eval_function = is_not_equal        
        return self

    def is_less(self, b):
        self._b = b
        self._eval_function = is_less
        return self

    def is_less_equal(self, b):
        self._b = b
        self._eval_function = is_less_equal        
        return self

    def is_greater(self, b):
        self._b = b
        self._eval_function = is_greater
        return self

    def is_greater_equal(self, b):
        self._b = b
        self._eval_function = is_greater_equal        
        return self

    def is_match(self, b):
        self._b = b
        self._eval_function = is_match
        return self

    def _get_value(self,a):
        if isinstance(a, str):
            try:
                value = getattr(self._evaluated_instance, a)
            except AttributeError:        
                value = a
        else:
            value = a
        return value

    def _evaluate_rules(self):
        rule_results = [v.evaluate.is_valid for (_,v) in self._rules.items()]
        return(rule_results)

    def _evaluate_conditions(self):
        condition_results = [v.evaluate.is_valid for (_,v) in self._conditions.items()]
        return(rule_results)        

    @property
    def evaluate(self):        
        self._reset()
        self._a = self._get_value(self._a)
        self._b = self._get_value(self._b)

        # Evaluate 
        self._is_valid = self._eval_function(self._a, self._b)        
        return self        

    @property
    def _print_syntactic_rule(self):

        print("\n   Rule #{id}: when {a} is {func} evaluates to {is_valid}."\
            .format(
                id=self._id, 
                a=self._a,
                func=self._eval_function, 
                is_valid=self._is_valid
            ))
    
    @property
    def _print_semantic_rule(self):
            
        print("\n   Rule #{id}: when {a} is {func} {b} evaluates to {is_valid}."\
            .format(
                id=self._id, 
                a=self._a,
                func=self._eval_function, 
                b=self._b,
                is_valid=self._is_valid
            ))
    @property
    def print_rule(self):
        """Prints rule for most recent result or current values."""

        if self._eval_function in SYNTACTIC_EVAL_FUNCTIONS:
            self._print_syntactic_rule
        else:
            self._print_semantic_rule

    @property
    def _print_syntactic_condition(self):

        print("\n   Condition #{id}: when {a} is {func} evaluates to {is_valid}."\
            .format(
                id=self._id, 
                a=self._a,
                func=self._eval_function, 
                is_valid=self._is_valid
            ))
    
    @property
    def _print_semantic_condition(self):
            
        print("\n   Condition #{id}: when {a} is {func} {b} evaluates to {is_valid}."\
            .format(
                id=self._id, 
                a=self._a,
                func=self._eval_function, 
                b=self._b,
                is_valid=self._is_valid
            ))
    @property
    def print_condition(self):
        """Prints condition for most recent result or current values."""

        if self._eval_function in SYNTACTIC_EVAL_FUNCTIONS:
            self._print_syntactic_condition
        else:
            self._print_semantic_condition            