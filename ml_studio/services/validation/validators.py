#!/usr/bin/env python3
# =========================================================================== #
#                          SERVICES: VALIDATORS                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \validators.py                                                        #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 21st 2019, 8:22:27 am                        #
# Last Modified: Saturday December 21st 2019, 8:23:37 am                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the validator classes. 

Validation is governed by the following validator classes:

    * BaseValidator : Abstract base class that defines the validator interface.
    * _Validator : Concrete validator for the _ variable.     
    * Validatrix : Validator factory 

Validators will expose methods to add, change and remove validation rules 
and conditions. A validate method delegates to subclasses validate methods
for each rule.  
    
"""
#%%
from abc import ABC, abstractmethod
import builtins
from collections.abc import Iterable, Iterator
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
#                                  VALIDERATOR                                #
# --------------------------------------------------------------------------- #
class Validerator(Iterator):
    """An iterator class for Validators."""
    
    # Keeps track of current position
    _position: int = None

    def __init__(self, collection):
        self._collection = collection
        self._position = 0

    def __next__(self):
        """Returns the next item in the sequence. At end, raise StopIteration."""
        try:
            value = self._collection[self._position]
            self._position += 1
        except IndexError:
            raise StopIteration()
        
        return value
# --------------------------------------------------------------------------- #
#                              BASEVALIDATOR                                  #
# --------------------------------------------------------------------------- #
class BaseValidator(ABC, Iterable):
    """Abstract base class for all validator objects."""

    def __init__(self):
        self._rules = []

    def __iter__(self):
        return Validerator(self._rules)

    def add_rule(self, rule):
        self._rules.append(rule)
        return self

    def del_rule(self, rule):
        if rule in self._rules:
            self._rules.remove(rule)
        return self

    def show_rules(self):
        print()


    def add_rule(self, rule):
        self._rules.append(rule)

    def 
