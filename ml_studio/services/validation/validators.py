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

    Validators
    ----------
    * Validator : Abstract base class for the following Validator classes.
    * Validerator : Rule Iterator for Validator classes.
    * StringOptionalValidator : Ensures a property value is a valid string 
        when not None.
    * BooleanValidator : Ensures a property value is a valid Boolean.
    * AllowedValueValidator : Ensures a property value is among the set of
        allowed values.
    * ColorValidator : Ensures a property value is a valid CSS color or
        a regex representation.
    * NumberValidator : Ensures a property value is a valid number
    * NumberRangeValidator : Ensures a property value is a valid number
        within a designated range.
    * ColorScaleValidator : Ensures a property value is a valid 
        colorscale.
    * ArrayValidator : Ensures a property value is an array containing
        valid types.
    * AngleValidator : Ensures that a property value is a valid angle

    Builders
    --------
    * ValidatorBuilder : Abstract base class for the following Builders
    * StringOptionalValidatorBuilder : Builds the StringOptionalValidator
    * BooleanValidatorBuilder : Builds the BooleanValidator
    * AllowedValueValidatorBuilder : Builds the AllowedValueValidator
    * ColorValidatorBuilder : Builds the ColorValidator
    * NumberValidatorBuilder : Builds the NumberValidator
    * NumberRangeValidatorBuilder : Builds the NumberRangeValidator
    * ColorScaleValidatorBuilder : Builds the ColorScaleValidator
    * ArrayValidatorBuilder : Builds the ArrayValidator
    * AngleValidatorBuilder : Builds the AngleValidator



    
"""
#%%
from abc import ABC, abstractmethod, abstractproperty
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

from ml_studio.services.classes import Classes
from ml_studio.utils.misc import format_text
from ml_studio.utils.data_analysis import is_array, is_homogeneous_array 
from ml_studio.utils.data_analysis import is_simple_array
# =========================================================================== #
#                                VALIDATORS                                   #
# =========================================================================== #
# --------------------------------------------------------------------------- #
#                                VALIDERATOR                                  #
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

    @abstractmethod
    def __init__(self):
        self._target_classname = None
        self._target_attribute_name = None
        self._rules = []        
        self._results = []
        self._invalid_messages = []
        self.is_valid = True

    def __iter__(self):
        return Validerator(self._rules)    

    @property
    def for_class(self):
        self._target_classname

    @for_class.setter
    def for_class(self, value):
        classes = Classes()
        if classes.search_classnames(value):            
            self._target_classname = value
        else:
            raise ValueError("{value} class is not a valid class".format(
        classname=value))     

    @property
    def for_attribute(self):
        self._target_attribute_name

    @for_attribute.setter
    def for_attribute(self, value):
        if self._target_classname is not None:
            classes = Classes()
            attributes = classes.get_instance_attributes(classname=self._target_classname) 
            if value in attributes:
                self._target_attribute_name = value
            else:
                raise ValueError("{attrname} is not a valid property for the\
                    {classname} class.".format(
                        attrname=value,
                        classname=self._target_classname
                    ))
        else:
            raise Exception("the target_classname must be designated in the \
                for_class method before specifying target_attribute_name")         


    def get_rule(self, rulename):
        """Retrieves a Rule or a RuleSet composite object from the validator.

        Parameters
        ----------
        rulename : str
            Rule classname or partial classname

        """        
        if self._rules:
            for rule in self._rules:
                if rulename.lower() in rule.__class__.__name__.lower():
                    return rule

    def add_rule(self, rule):
        """Adds a Rule or a RuleSet composite object to the validator.

        Parameters
        ----------
        rule : Rule or RuleSet object
            Rule or Rule composite to be evaluated by the validator

        """ 
        self._rules.append(rule)

    def del_rule(self, rule):
        """Deletes a Rule or a RuleSet composite object from the validator.

        Parameters
        ----------
        rule : Rule or RuleSet object
            Rule or Rule composite to be evaluated by the validator

        """        
        self._rules.remove(rule)   

    @abstractmethod
    def validate_coerce(self, attribute_value, instance):
        if self._rules:
            self._invalid_messages = []
            self._results = []
            for rule in self._rules:
                rule.validate(instance, self._target_attribute_name, attribute_value)
                self._results.append(rule.is_valid)
                if not rule.is_valid:
                    self._invalid_messages.append(rule.invalid_messages)
            self.is_valid = all(self._results)
    
    def print_validator(self):
        text = "\n{validatorname} for {targetclassname} {targetattrname} property:".format(
                validatorname = self.__class__.__name__,
                targetclassname = self._target_classname,
                targetattrname = self._target_attribute_name
            )        
        print(text)        
        if self._rules:
            for idx, rule in enumerate(self._rules):
                rule.compile()
                rule.print_rule()
                self._rules[idx] = rule        


# --------------------------------------------------------------------------- #
#                        STRINGOPTIONALVALIDATOR                              #
# --------------------------------------------------------------------------- #                
class StringOptionalValidator(BaseValidator):
    """Validates property as a string if not empty."""

    def __init__(self):
        super(StringOptionalValidator, self).__init__()

    
    def validate_coerce(self, attribute_value, instance):
        super(StringOptionalValidator, self).validate_coerce(
                                            attribute_value=attribute_value,
                                            instance=instance)
        # Coerce logic
        self.validated_value = str(attribute_value)
        self.is_valid = True

# --------------------------------------------------------------------------- #
#                            BOOLEANVALIDATOR                                 #
# --------------------------------------------------------------------------- #                
class BooleanValidator(BaseValidator):
    """Indicates whether a property is a valid boolean value."""

    def __init__(self):
        super(BooleanValidator, self).__init__()

    def validate_coerce(self, attribute_value, instance):
        super(BooleanValidator, self).validate_coerce(
                                            attribute_value=attribute_value,
                                            instance=instance)
        # Coerce logic
        if self.is_valid == False:
            # Attempt to coerce
            if attribute_value in [True, 'True', 'true', 1, 'yes', 'Yes', 'y']:
                validated_value = True
                self.is_valid = True
            elif attribute_value in [False, 'False', 'false', 0, 'no', 'No', 'n']:
                validated_value = False
                self.is_valid = True
            else:
                # Unable to coerce, report messages
                validated_value = attribute_value
                for message in self._invalid_messages:
                    print(message)
        else:
            validated_value = attribute_value      
        return validated_value      
            
# --------------------------------------------------------------------------- #
#                            ALLOWEDVALUESVALIDATOR                           #
# --------------------------------------------------------------------------- #                
class AllowedValuesValidator(BaseValidator):
    """Indicates whether a property contains allowed values."""

    def __init__(self):
        super(AllowedValuesValidator, self).__init__()
    
    def validate_coerce(self, attribute_value, instance):
        super(AllowedValuesValidator, self).validate_coerce(
                                            attribute_value=attribute_value,
                                            instance=instance)
        # Coerce logic
        if self.is_valid == False:
            # Unable to coerce discrete values
            validated_value = attribute_value
            for message in self._invalid_messages:
                print(message)
        else:
            validated_value = attribute_value      
        return validated_value      

# --------------------------------------------------------------------------- #
#                            REGEXVALIDATOR                                   #
# --------------------------------------------------------------------------- #                
class RegexValidator(BaseValidator):
    """Indicates whether a property value matches a regex pattern."""

    def __init__(self):
        super(RegexValidator, self).__init__()
    
    def validate_coerce(self, attribute_value, instance):
        super(RegexValidator, self).validate_coerce(
                                            attribute_value=attribute_value,
                                            instance=instance)
        # Coerce logic
        if self.is_valid == False:
            # Unable to coerce regex patterns
            validated_value = attribute_value
            for message in self._invalid_messages:
                print(message)
        else:
            validated_value = attribute_value      
        return validated_value      

# --------------------------------------------------------------------------- #
#                            NUMBERVALIDATOR                                  #
# --------------------------------------------------------------------------- #                
class NumberValidator(BaseValidator):
    """Indicates whether a property value matches a regex pattern."""

    def __init__(self):
        super(NumberValidator, self).__init__()
    
    def validate_coerce(self, attribute_value, instance):
        super(NumberValidator, self).validate_coerce(
                                            attribute_value=attribute_value,
                                            instance=instance)
        # Coerce logic        
        if self.is_valid == False:
            # Unable to coerce regex patterns
            validated_value = attribute_value
            for message in self._invalid_messages:
                print(message)
        else:
            validated_value = attribute_value      
        return validated_value 
# =========================================================================== #
#                                BUILDERS                                     #
# =========================================================================== #
# --------------------------------------------------------------------------- #
#                              VALIDATORBUILDER                               #
# --------------------------------------------------------------------------- #
class ValidatorBuilder(ABC):
    """Abstract base class for all Builder classes."""

    @abstractproperty
    def validator(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def for_class(self, target_classname):
        pass

    @abstractmethod
    def for_attribute(self, target_attribute_name):
        pass

    @abstractmethod
    def add_rule(self):
        pass

    @abstractmethod
    def del_rule(self):
        pass

# --------------------------------------------------------------------------- #
#                      STRINGOPTIONALVALIDATORBUILDER                         #
# --------------------------------------------------------------------------- #                
class StringOptionalValidatorBuilder(ValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        self._validator = StringOptionalValidator()
    
    @property
    def validator(self):
        validator = self._validator
        self.reset()
        return validator
    
    def for_class(self, target_classname):
        classes = Classes()
        if classes.search_classnames(target_classname):
            self._target_classname = target_classname
            self._validator.target_classname = target_classname
        else:
            raise ValueError("{classname} class is not a valid class".format(
        classname=target_classname))

    def for_attribute(self, target_attribute_name):
        if hasattr(self._validator, 'target_classname'):
            classes = Classes()
            attributes = classes.get_instance_attributes(classname=self._target_classname) 
            if target_attribute_name in attributes:
                self._target_attribute_name = target_attribute_name
                self._validator.target_attribute_name = target_attribute_name
            else:
                raise ValueError("{attrname} is not a valid property for the\
                    {classname} class.".format(
                        attrname=target_attribute_name,
                        classname=self._target_classname
                    ))
        else:
            raise Exception("the target_classname must be designated in the \
                for_class method before specifying target_attribute_name")
    
    def add_rule(self, rule):
        self._validator.add_rule(rule)
    
    def del_rule(self, rule):
        self._validator.del_rule(rule)
   
