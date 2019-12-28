#!/usr/bin/env python3
# =========================================================================== #
#                     VALIDATION : BUILDERS : RULES                           #
# =========================================================================== #
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
# Create Date: Friday December 27th 2019, 11:33:55 am                         #
# Last Modified: Friday December 27th 2019, 11:34:20 am                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module contains Rule object builders.

This module implements the following concrete builders.

    RuleBuilder Interface
    ---------------------
    * RuleBuilder : Abstract base class for concrete Rule builders.

    Syntactic Rule Builders
    -----------------------
    * BoolRuleBuilder : Builder for the BoolRule
    * EmptyRuleBuilder : Builder for the EmptyRule
    * FloatRuleBuilder : Builder for the FloatRule
    * IntegerRuleBuilder : Builder for the IntegerRule
    * NoneRuleBuilder : Builder for the NoneRuleBu
    * NotEmptyRuleBuilder : Builder for the NotEmptyRule
    * NotNoneRuleBuilder : Builder for the NotNoneRule
    * NumberRuleBuilder : Builder for the NumberRule
    * StringRuleBuilder : Builder for the StringRule


    Semantic Rules
    -----------------
    * AllowedRuleBuilder : Builder for the AllowedRule
    * BetweenRuleBuilder : Builder for the BetweenRule
    * DisAllowedRuleBuilder : Builder for the DisAllowedRule
    * EqualRuleBuilder : Builder for the EqualRule
    * GreaterRuleBuilder : Builder for the GreaterRule
    * LessRuleBuilder : Builder for the LessRule
    * NotEqualRuleBuilder : Builder for the NotEqualRule
    * RegexRuleBuilder : Builder for the RegexRule  

    RuleSet Builder
    ---------------
    * RuleSetBuilder : Builder for the RuleSet class.    

"""
#%%
from abc import ABC, abstractmethod, abstractproperty
import builtins
from collections import defaultdict
from collections.abc import Iterable
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

from ml_studio.services.validation.rules import NoneRule, NotNoneRule
from ml_studio.services.validation.rules import EmptyRule, NotEmptyRule
from ml_studio.services.validation.rules import BoolRule, FloatRule
from ml_studio.services.validation.rules import IntegerRule, NumberRule
from ml_studio.services.validation.rules import StringRule, AllowedRule
from ml_studio.services.validation.rules import BetweenRule, DisAllowedRule
from ml_studio.services.validation.rules import EqualRule, GreaterRule
from ml_studio.services.validation.rules import LessRule, NotEqualRule
from ml_studio.services.validation.rules import RegexRule

# --------------------------------------------------------------------------- #
#                            RULE BUILDER                                     #  
# --------------------------------------------------------------------------- #
class RuleBuilder(ABC):
    """Base class for all rule builders."""

    @abstractproperty
    def rule(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def with_component(self, component):
        pass

class BoolRuleBuilder(RuleBuilder):
    def __init__(self, array_ok=False):
        self._array_ok = array_ok
        self.reset()

    def reset(self):
        instance = BoolRule()
        target_name = "BoolRule"
        parent_name = "Rule"        
        self._rule = BoolRule(instance=instance, target_name=target_name, 
                              parent_name=parent_name, array_ok=self._array_ok)    

class EmptyRuleBuilder(RuleBuilder):
class FloatRuleBuilder(RuleBuilder):
class IntegerRuleBuilder(RuleBuilder):
class NoneRuleBuilder(RuleBuilder):
class NotEmptyRuleBuilder(RuleBuilder):
class NotNoneRuleBuilder(RuleBuilder):
class NumberRuleBuilder(RuleBuilder):
class StringRuleBuilder(RuleBuilder):



class AllowedRuleBuilder(RuleBuilder):
class BetweenRuleBuilder(RuleBuilder):
class DisAllowedRuleBuilder(RuleBuilder):
class EqualRuleBuilder(RuleBuilder):
class GreaterRuleBuilder(RuleBuilder):
class LessRuleBuilder(RuleBuilder):
class NotEqualRuleBuilder(RuleBuilder):
class RegexRuleBuilder(RuleBuilder):


