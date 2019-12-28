#!/usr/bin/env python3
# =========================================================================== #
#                     VALIDATION : BUILDERS : CORE                            #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \core.py                                                              #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 27th 2019, 12:53:50 am                         #
# Last Modified: Friday December 27th 2019, 10:45:52 am                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the core validation builders. 

The core builders include:

    Core Builders
    -------------
    * ValidatorBuilder : Abstract base class for the validation Builders.
    * ClassValidatorBuilder : A collection of Builders for class validators.
    * AttributeValidatorBuilder : A collection of attribute validation Builders.
    * ValidationRuleSetBuilder : A set of validation RuleSet Builders
    * ValidationRuleBuilder : A set of validation Rule Builders
    * ValidationConditionBuilder : A set of validation condition Builders.    

    The above classes represent a tree-like Composite Pattern starting with
    the ClassValidatorBuilder at the top of the hierarchy to the 
    ValidationConditionBuilder at the bottom.    
    
"""
#%%
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd

from ml_studio.services.validation.rules import RuleSet, Rule
from ml_studio.utils.misc import format_text
from ml_studio.services.validation.validators.core import StringValidator
from ml_studio.services.validation.validators.core import NumberValidator
from ml_studio.services.validation.validators.core import IntegerValidator
from ml_studio.services.validation.validators.core import AllowedValuesValidator
from ml_studio.services.validation.validators.core import ForbiddenValuesValidator
from ml_studio.services.validation.validators.core import ArrayValidator
from ml_studio.services.validation.validators.core import BooleanValidator

# --------------------------------------------------------------------------- #
#                              ValidatorBuilder                               #
# --------------------------------------------------------------------------- #
class ValidatorBuilder(ABC):
    """Abstract base class for all Builder classes."""

    @abstractproperty
    def validator(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def with_component(self, component):
        pass


