#!/usr/bin/env python3
# =========================================================================== #
#                        SERVICES: VALIDATION: RULES                          #
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
# Create Date: Friday December 20th 2019, 5:06:57 am                          #
# Last Modified: Friday December 20th 2019, 10:33:23 am                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module contains the builder / fluent interface classes for validation rules.

This module implements the builder design pattern with a fluent interface,
and includes:

    * Director : Director class that constructs validation rules
    * BaseRuleBuilder : Abstract base class for rule building objects

    Syntactic Builders
    ------------------
    * NoneRuleBuilder : Concrete builder for the NoneRule, which evaluates 
        whether the value of a specific property is equal to None.
    * EmptyRuleBuilder : Concrete builder for the EmptyRule, which evaluates 
        whether the value of a specific property is empty.
    * BoolRuleBuilder : Concreate builder for the BoolRule, which evaluates 
        whether the value of a specific property is Boolean.
    * IntRuleBuilder : Concrete builder for the IntRule, which evaluates 
        whether the value of a specific property is an integer, and optionally
        whether the value is in accordance with min/max requirements.
    * FloatRule : Concrete builder for the FloatRule, which evaluates 
        whether the value of a specific property is an float, and optionally
        whether the value is in accordance with min/max requirements.
    * NumberRuleBuilder : Concrete builder for the NumberRule, which evaluates 
        whether the value of a specific property is an a number, and optionally
        whether the value is in accordance with min/max requirements.
    * StringRuleBuilder : Concrete builder for the StringRule, which evaluates 
        whether the value of a specific property is a string.

    Semantic Builders
    -----------------
    * EqualRuleBuilder : Concrete builder for the EqualRule, which ensures 
        that the value of a specific property is equal to a particular value 
        or that of another instance and/or property.        
    * AllowedRuleBuilder : Concrete builder for the AllowedRule, which ensures 
        the value of a specific property is one of a discrete set of allowed values. 
    * DisAllowedRuleBuilder : Concrete builder for the EqualRule, which ensures the 
        value of a specific property is none of a discrete set of disallowed values.     
    * LessRuleBuilder : Concrete builder for the LessRule, which ensures the 
        value of a specific property is less than a particular  value or that 
        of another instance and / or property.   
    * LessEqualRuleBuilder : Concrete builder for the LessEqualRule, which ensures the 
        value of a specific property is less than or equal to a particular 
        value or that of another instance and / or property.        
    * GreaterRuleBuilder : Concrete builder for the EqualRule, which ensures the 
        value of a specific property is greater than a particulcar value or greater 
        than the value of another property.        
    * RegexRuleBuilder : Concrete builder for the EqualRule, which ensures the 
        value of a specific property matches the given regular expression(s).     

    Syntactic Array Rule Builders
    ---------------------
    ArrayRules : Abstract base class building rules that apply to arrays.
    AllBoolRuleBuilder : Concrete builder for the AllBoolRule, which ensures that
        all values of a specific property are Booleans.
    AllIntRuleBuilder : Concrete builder for the AllIntRule, which ensures that
        all values of a specific property are integers.
    AllFloatRuleBuilder : Concrete builder for the AllFloatRule, which ensures that
        all values of a specific property are Floats.
    AllNumberRuleBuilder : Concrete builder for the AllNumberRule, which ensures that
        all values of a specific property are numbers.
    AllStringRuleBuilder : Concrete builder for the AllStringRule, which ensures that
        all values of a specific property are strings.

    Semantic Array Rule Builders
    --------------------
    AllEqualRuleBuilder : Concrete builder for the AllEqualRule, which ensures 
        that all values of a specific property are equal to all elements of 
        a reference array-like or a property on another object. 
    AllAllowedRuleBuilder : Concrete builder for the AllAllowedRule, which 
        ensures that all values of a specific property are allowed.
    AnyDisAllowedRuleBuilder : Concrete builder for the AnyDisAllowedRule, 
        which ensures that all values of a specific property are allowed.

A rule is comprised of the following parts, each of which implemented via the
BaseRuleBuilder interface.

    * Metadata : The metadata associated with the rule, such as the date and
        time, created, the creator and the Rule name.
    * Target : The target of the validation, in terms of the names of the class
        and attribute to be validated.
    * Directive : The instructions executed during validation, along with its 
        parameters. For instance, a NumberRule may have min and max as
        rule parameters.
    * Conditions : Two types of conditions are implemented: 'when' conditions
        that must be passed before performing the validation and 'except when'
        conditions that specify when a validation rule should not apply.
    * Message : The message to render when the validation fails. 


"""
#%%
from abc import ABC, abstractmethod, abstractproperty
import builtins
from collections.abc import Iterable
from dateutil.parser import parse
from datetime import datetime
import getpass
import math
import numbers
import os
import re
import sys
import textwrap
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from ml_studio.services.validation.conditions import IsNone, IsEmpty, IsBool
from ml_studio.services.validation.conditions import IsInt, IsFloat
from ml_studio.services.validation.conditions import IsNumber, IsString
from ml_studio.services.validation.conditions import IsEqual, IsIn, IsLess
from ml_studio.services.validation.conditions import IsGreater, IsMatch
from ml_studio.services.validation.conditions import isArray

# --------------------------------------------------------------------------- #
#                                RULEBUILDER                                  #  
# --------------------------------------------------------------------------- #
class RuleBuilder(ABC):
    """The Builder interface specifies methods for creating the Rules objects. """

    @abstractproperty
    def rule(self):
        pass

    @abstractmethod
    def for_target(self):
        pass

    @abstractmethod
    def when(self):
        pass

    @abstractmethod
    def when_any(self):
        pass

    @abstractmethod
    def when_all(self):
        pass

# --------------------------------------------------------------------------- #
#                            NONERULEBUILDER                                  #  
# --------------------------------------------------------------------------- #
class NoneRuleBuilder(RuleBuilder):
    """Concrete Builder class for the NoneRule follows the RuleBuilder interface."""

    def __init__(self, name, description=None):
        """Instantiate a fresh Rule instance."""
        self.reset()
        self._rule.name = name
        self._rule.description = description 

    def reset(self):
        """Creates a fresh instance of the Rule object."""
        self._rule = NoneRule()

    @property
    def rule(self):
        rule = self._rule
        self.reset()
        return rule

    def when(self, condition):
        self._rule.when(condition)

    def not_when(self, condition):
        self._rule.not_when(condition)



    


#%%
# --------------------------------------------------------------------------- #
#                                   RULE                                      #  
# --------------------------------------------------------------------------- #
class Rule(ABC):
    """Base class for all rules."""

    def __init__(self, *args, **kwargs):

        # Designate unique/opaque userid and other metadata        
        self._id = uuid4()
        self._created = datetime.now()
        self._user = getpass.getuser()

        # The evaluated property
        self._evaluated_instance = None
        self._evaluated_classname = None
        self._evaluated_attribute_name = None
        self._evaluated_attribute_value = None  

        # Two lists. Each containing conditions for when the rule should (when) or
        # should not (except when) be applied. Each time the when and 
        # except when methods are called, the condition is added to 
        # the appropriate list.
        self._conditions = dict()

        # Properties that capture the results of the validation.
        self.isValid = True
        self.invalid_value = None              
        self.invalid_message = None


    def when(self, condition):
        """Adds a single condition that must be met for a rule to apply."""
        self._conditions['when'] = [] or self._conditions['when']
        self._conditions['when'].append(condition)
        return self

    def when_all(self, conditions):
        """Adds a list of rules, all of which must be met for a rule to apply."""
        self._conditions['when_all'] = [] or self._conditions['when_all']
        self._conditions['when_all'].append(condition)
        return self

    def when_any(self, conditions):
        """Adds a list of rules, all of which must be met for a rule to apply."""
        self._conditions['when_any'] = [] or self._conditions['when_any']
        self._conditions['when_any'].append(condition)
        return self

    def _evaluate_when(self):
        if self._conditions.get('when'):
            results = []
            for condition in self._conditions.get('when'):
                results.append(condition())
            return all(results)
        else:
            return True

    def _evaluate_when_all(self):
        if self._conditions.get('when_all'):
            results = []
            for condition in self._conditions.get('when_all'):
                results.append(condition())
            return all(results)
        else:
            return True

    def _evaluate_when_any(self):
        if self._conditions.get('when_any'):
            results = []
            for condition in self._conditions.get('when_any'):
                results.append(condition())
            return any(results)
        else:
            return True

    def evaluate_conditions(self):
        """Evaluates 'when', 'when_all', and 'when_any' conditions."""
        return self._evaluate_when() and self._evaluate_when_all() and \
            self._evaluate_when_any()

    @abstractmethod
    def validate(self, instance, attribute_name, attribute_value):
        self._evaluated_instance = instance
        self._evaluated_classname = instance.__class__.__name__
        self._evaluated_attribute_name = attribute_name
        self._evaluated_attribute_value = attribute_value            
        # Creates validation to ensure that the attribute value being 
        # evaluated is not an array-like.
        if isArray(attribute_value):
            raise AttributeError("{classname} is for strings, numerics, and \
                Booleans. To evaluate array-like structures, use the \
                All{classname} or Any{classname} instead.".format(
                    classname = self.__class__.__name__
                )
            )

    @abstractmethod
    def error_message(self):
        pass
    
# --------------------------------------------------------------------------- #
#                                NONERULE                                     #  
# --------------------------------------------------------------------------- #            
class NoneRule(Rule):
    """Evaluates whether the value of a specific property is None."""

    def __init__(self):
        super(NoneRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(NoneRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value is None: 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not None. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                NOTNONERULE                                  #  
# --------------------------------------------------------------------------- #            
class NotNoneRule(Rule):
    """Evaluates whether the value of a specific property is None."""

    def __init__(self):
        super(NotNoneRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(NotNoneRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value is not None: 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is None. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg


# --------------------------------------------------------------------------- #
#                                EMPTYRULE                                    #  
# --------------------------------------------------------------------------- #            
class EmptyRule(Rule):
    """Evaluates whether the value of a specific property is empty.
    
    A value is considered empty if it equals None, the empty string or 
    whitespace.
    """

    def __init__(self):
        super(EmptyRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(EmptyRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if isEmpty(attribute_value): 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not empty. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                NOTEMPTYRULE                                 #  
# --------------------------------------------------------------------------- #            
class NotEmptyRule(Rule):
    """Evaluates whether the value of a specific property is empty.
    
    A value is considered empty if it equals None, the empty string or 
    whitespace.
    """

    def __init__(self):
        super(NotEmptyRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(NotEmptyRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if not isEmpty(attribute_value): 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is empty. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                BOOLRULE                                     #  
# --------------------------------------------------------------------------- #            
class BoolRule(Rule):
    """Evaluates whether the value of a specific property is a boolean."""

    def __init__(self):
        super(BoolRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(BoolRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if isArray(attribute_value):
            raise AttributeError("BoolRule is for strings. To evaluate\
                array-like structures, use AllBoolRule.")

        if self.evaluate_when() and self.evaluate_not_when():
            if isinstance(attribute_value, bool):
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not Boolean. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                INTRULE                                      #   
# --------------------------------------------------------------------------- #            
class IntRule(Rule):
    """Evaluates whether the value of a specific property is an integer."""

    def __init__(self):
        super(IntRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(IntRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if isinstance(attribute_value, int):
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not an integer. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                FLOATRULE                                    #   
# --------------------------------------------------------------------------- #            
class FloatRule(Rule):
    """Evaluates whether the value of a specific property is an integer."""

    def __init__(self):
        super(FloatRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(FloatRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if isinstance(attribute_value, float):
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not a float. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg


# --------------------------------------------------------------------------- #
#                              NUMBERRULE                                     #   
# --------------------------------------------------------------------------- #            
class NumberRule(Rule):
    """Evaluates whether the value of a specific property is an integer."""

    def __init__(self):
        super(NumberRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(NumberRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if isinstance(attribute_value, (int,float)):
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not a number. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                              STRINGRULE                                     #   
# --------------------------------------------------------------------------- #            
class StringRule(Rule):
    """Evaluates whether the value of a specific property is an integer."""

    def __init__(self, spaces_ok=False, empty_ok=False):
        super(StringRule, self).__init__()
        self._spaces_ok = spaces_ok
        self._empty_ok = empty_ok

    def validate(self, instance, attribute_name, attribute_value):
        super(StringRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_when() and self.evaluate_not_when():
            if isinstance(attribute_value, str):
                if self._spaces_ok and self._empty_ok:
                    self.isValid = True
                elif not self._spaces_ok and (" " in attribute_value):
                    self.isValid =False
                    self.invalid_value = attribute_value
                    self.invalid_message = self.error_message()
                elif not self._empty_ok and attribute_value == "":
                    self.isValid =False
                    self.invalid_value = attribute_value
                    self.invalid_message = self.error_message()
                else:
                    self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not a string. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg                

# --------------------------------------------------------------------------- #
#                                SEMANTICRULE                                 #  
# --------------------------------------------------------------------------- #                    
class SemanticRule(Rule):
    """Abstract base class for rules requiring cross-reference to other objects.

    This class introduces the instance and attribute name for an external
    value in the constructor. Another method is exposed to obtain the value
    from the external object. 
    
    Parameters
    ----------
    val : Any
        The external value 

    instance : ML Studio object
        The instance of a class containing the external value. Required if 
            val is None.

    attribute_name : str
        The name of the attribute containing the external value. Required if
            val is None.
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(SemanticRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_classname = instance.__class__.__name__
        self._external_attribute_name = attribute_name

    def get_external_value(self):
        try:
            self._external_attribute_value = getattr(self._external_instance, \
                self._external_attribute_name)
            self._value = self._external_attribute_value
        except AttributeError:
            classname = self._external_instance.__class__.__name__
            msg = "{classname} has no attribute {attrname}.".format(
                classname = classname,
                attrname = self._external_attribute_name
            )
            print(msg)            

# --------------------------------------------------------------------------- #
#                                 EQUALRULE                                   #  
# --------------------------------------------------------------------------- #            
class EqualRule(SemanticRule):
    """Evaluates equality of a property vis-a-vis another value or property.
    
    The EqualRule applies to strings, ints, and floats. Array-like structures
    are evaluated with the AllEqualRule.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(EqualRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(EqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value == self._value: 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class is not equal \
                to {externalclass} property {externalattr} = {externalval}. \
                Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    externalclass = self._external_instance.__class__.__name__,
                    externalattr = self._external_attribute_name,
                    externalval=self._external_attribute_value,
                    value=self._evaluated_attribute_value)

        else:
            msg = "The {attribute} property of the {classname} class is not equal \
                to {val}. \
                Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    val = self._value,
                    value=self._evaluated_attribute_value)                    
        
        formatted_msg = format_text(msg)
        return formatted_msg 
# --------------------------------------------------------------------------- #
#                                 NOTEQUALRULE                                #  
# --------------------------------------------------------------------------- #            
class NotEqualRule(SemanticRule):
    """Evaluates equality of a property vis-a-vis another value or property.
    
    The EqualRule applies to strings, ints, and floats. Array-like structures
    are evaluated with the AllNotEqualRule.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(NotEqualRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(NotEqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value != self._value: 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class is equal \
                to {externalclass} property {externalattr} = {externalval}. \
                Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    externalclass = self._external_instance.__class__.__name__,
                    externalattr = self._external_attribute_name,
                    externalval = self._external_attribute_value,
                    value=self._evaluated_attribute_value)

        else:
            msg = "The {attribute} property of the {classname} class is equal \
                to {val}. \
                Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    val = self._value,
                    value=self._evaluated_attribute_value)                    
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                               ALLOWEDRULE                                   #  
# --------------------------------------------------------------------------- #            
class AllowedRule(SemanticRule):
    """Evaluates whether the value(s) of an evaluated property are allowed.
    
    The AllowedRule applies to strings, ints, and floats. Array-like structures
    are evaluated with the AllAllowedRule.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(AllowedRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

         # Obtain the value(s) to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value in self._value: 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class does not \
                equal any of the allowed values. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    equalclass = self._external_instance.__class__.__name__,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Allowed Value(s): {allowed}.".format(
                allowed=str(self._value)
            )

        else:
            msg = "The {attribute} property of the {classname} class does not \
                equal any of the allowed values. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Allowed value(s): {allowed}".format(
                allowed=str(self._value)
            )
        
        formatted_msg = format_text(msg)
        return formatted_msg 



# --------------------------------------------------------------------------- #
#                               DISALLOWEDRULE                                #  
# --------------------------------------------------------------------------- #            
class DisAllowedRule(SemanticRule):
    """Evaluates whether the value of an evaluated property are disallowed.
    
    The DisAllowedRule applies to strings, ints, and floats. Array-like 
    structures are evaluated with the AllDisAllowedRule.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(DisAllowedRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(DisAllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

         # Obtain the value(s) to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value not in self._value: 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class equals \
                a disallowed value. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,                    
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Disallowed Value(s): {disallowed}.".format(
                disallowed=str(self._value)
            )

        else:
            msg = "The {attribute} property of the {classname} class equal \
                a disallowed value. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Disallowed value(s): {disallowed}".format(
                disallowed=str(self._value)
            )
        
        formatted_msg = format_text(msg)
        return formatted_msg 


# --------------------------------------------------------------------------- #
#                                 LESSRULE                                    #  
# --------------------------------------------------------------------------- #            
class LessRule(SemanticRule):
    """Evaluates whether a value of a specific property is less than a reference.

    The reference may be provided in the val parameter or may be provided
    via an external object and attribute.
    
    The LessRule applies to strings, ints, and floats. Array-like structures
    are evaluated with the AllEqualRule.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None,
                 equal_ok=False):
        super(LessRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name
        self._equal_ok = equal_ok

    def validate(self, instance, attribute_name, attribute_value):
        super(LessRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value < self._value: 
                self.isValid = True
            elif self._equal_ok:
                if attribute_value == self._value:
                    self.isValid = True
                else:
                    self.isValid = False
                    self.invalid_value = attribute_value
                    self.invalid_message = self.error_message()
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class is not \
                less than ".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname
            )

            if self._equal_ok:
                msg = msg + "or equal to "

            msg = msg + "{externalclass} property {externalattr} = {externalval}. \
                Received value: {value}.".format(
                    externalclass=self._external_classname,
                    externalattr=self._external_attribute_name,
                    externalval = self._value,
                    value=self._evaluated_attribute_value)

        else:
            msg = "The {attribute} property of the {classname} class is not \
                less than ".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname
            )

            if self._equal_ok:
                msg = msg + "or equal to "

            msg = msg + "{val}. \
                Received value: {value}.".format(
                    val=self._value,
                    value=self._evaluated_attribute_value)            
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                               GREATERRULE                                   #  
# --------------------------------------------------------------------------- #            
class GreaterRule(SemanticRule):
    """Evaluates whether a value of a specific property is greater than a reference.

    The reference may be provided in the val parameter or may be provided
    via an external object and attribute.
    
    The GreaterRule applies to strings, ints, and floats. Array-like structures
    are evaluated with the AllEqualRule.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None,
                 equal_ok=False):
        super(GreaterRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name
        self._equal_ok = equal_ok

    def validate(self, instance, attribute_name, attribute_value):
        super(GreaterRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if isinstance(attribute_value, (pd.Series, np.ndarray, tuple, list)):
            raise ValueError("This rule applies only to strings, integers, \
            and floats. Use the AllEqualRule for iterables.")

        # Obtain the value to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            if attribute_value > self._value: 
                self.isValid = True
            elif self._equal_ok:
                if attribute_value == self._value:
                    self.isValid = True
                else:
                    self.isValid = False
                    self.invalid_value = attribute_value
                    self.invalid_message = self.error_message()
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class is not \
                greater than ".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname
            )

            if self._equal_ok:
                msg = msg + "or equal to "

            msg = msg + "{externalclass} property {externalattr} = {externalval}. \
                Received value: {value}.".format(
                    externalclass=self._external_classname,
                    externalattr=self._external_attribute_name,
                    externalval = self._value,
                    value=self._evaluated_attribute_value)

        else:
            msg = "The {attribute} property of the {classname} class is not \
                greater than ".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname
            )

            if self._equal_ok:
                msg = msg + "or equal to "

            msg = msg + "{val}. \
                Received value: {value}.".format(
                    val=self._value,
                    value=self._evaluated_attribute_value)            
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                               REGEXRULE                                     #  
# --------------------------------------------------------------------------- #            
class RegexRule(SemanticRule):
    """Evaluates whether a value of a property matches regex pattern(s).

    The reference may be provided in the val parameter or may be provided
    via an external object and attribute.
    
    The RegexRule applies to strings and can be evaluated against one or 
    several regex patterns.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None,
                 equal_ok=False):
        super(RegexRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name        

    def validate_regex(self, string):
        try:
            re.compile(string)
        except re.error as e:
            raise AssertionError(e)

    def evaluate_regex(self, string, regex):        
        return re.search(regex, string)



    def validate(self, instance, attribute_name, attribute_value):
        super(RegexRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if not isString(self._evaluated_attribute_value):
            raise ValueError("This rule applies only to strings.")

        # Obtain the value to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Validate regex pattern(s)
        if isinstance(self._value, str):
            self.validate_regex(self._value)
        elif isArray(self._value):
            not_strings = [v for v in self._value if not isinstance(v, str)]
            if not_strings:
                raise ValueError("Some or all reference values are not strings.")
            else:
                for v in self._value:
                    self.validate_regex(v)                

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            # If array, then we evaluate to True of any matches.
            if isArray(self._value):                
                regex_matches = []
                for regex in self._value:
                    regex_matches.append(\
                        self.evaluate_regex(self._evaluated_attribute_value, regex))

                if any(regex_matches):
                    self.isValid = True
                else:
                    self.isValid = False
                    self.invalid_value = attribute_value
                    self.invalid_message = self.error_message()
            elif self.evaluate_regex(self._evaluated_attribute_value, self._value):
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value           
                self.invalid_message = self.error_message()                     
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class did not \
            match (any of) the regex patterns. Received values: {value}.".format(
            attribute=self._evaluated_attribute_name,
            classname=self._evaluated_classname,
            value=self._evaluated_attribute_value)

        msg = msg + "Regex pattern(s): {patterns}".format(
            patterns=str(self._value)
        )

        formatted_msg = format_text(msg)
        return formatted_msg 


# --------------------------------------------------------------------------- #
#                                ARRAYRULES                                   #   
# --------------------------------------------------------------------------- #                            
class ArrayRule(Rule):
    """Abstract base class to evaluate array-like structures."""
    def __init__(self, value=None):
        super(ArrayRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        self._evaluated_instance = instance
        self._evaluated_classname = instance.__class__.__name__
        self._evaluated_attribute_name = attribute_name
        self._evaluated_attribute_value = attribute_value          

        # Creates validation to ensure that the attribute value being
        # evaluated is an array-like.
        if not isArray(attribute_value):
            # Find the 'sister' classname for non array-like structures
            sister_classname = self.__class__.__name__.replace('Any', "")
            sister_classname = sister_classname.replace('All', "")

            raise AttributeError("{classname} is for numpy arrays, pandas\
                Series, lists, tuples and other array-like structures. \
                To evaluate non array-like structures, use the \
                {sister_classname} instead.".format(
                    classname = self.__class__.__name__,
                    sister_classname = sister_classname
                )
            )

# --------------------------------------------------------------------------- #
#                                ALLBOOLRULE                                  #  
# --------------------------------------------------------------------------- #            
class AllBoolRule(ArrayRule):
    """Evaluates whether all elements of a property are Booleans.
    
    Parameters
    ----------
    val : None
        None since this rule doesn't require a referene value. 
    """

    def __init__(self, value=None):
        super(AllBoolRule, self).__init__(value=value)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllBoolRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_not_when():
            invalid_values = [v for v in self._evaluated_attribute_value\
                 if (not isinstance(v, bool))]
            if invalid_values:
                self.isValid = False
                self.invalid_value = invalid_values
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class contains \
            values which are not Booleans. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=str(self.invalid_value))

        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                                ALLINTRULE                                   #  
# --------------------------------------------------------------------------- #            
class AllIntRule(ArrayRule):
    """Evaluates whether all elements of a property are Booleans.
    
    Parameters
    ----------
    val : None
        None since this rule doesn't require a referene value.
    """

    def __init__(self, value=None):
        super(AllIntRule, self).__init__(value=value)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllIntRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_not_when():
            invalid_values = [v for v in self._evaluated_attribute_value\
                 if not isinstance(v, int)]
            if invalid_values:
                self.isValid = False
                self.invalid_value = invalid_values
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class contains \
            values which are not integers. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=str(self.invalid_value))

        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                              ALLFLOATSRULE                                  #  
# --------------------------------------------------------------------------- #            
class AllFloatRule(ArrayRule):
    """Evaluates whether all elements of a property are Booleans.
    
    Parameters
    ----------
    val : None
        None since this rule doesn't require a referene value.
    """

    def __init__(self, value=None):
        super(AllFloatRule, self).__init__(value=value)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllFloatRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_not_when():
            invalid_values = [v for v in self._evaluated_attribute_value\
                 if not isinstance(v, float)]
            if invalid_values:
                self.isValid = False
                self.invalid_value = invalid_values
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class contains \
            values which are not floats. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=str(self.invalid_value))

        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                             ALLNUMBERSRULE                                  #  
# --------------------------------------------------------------------------- #            
class AllNumberRule(ArrayRule):
    """Evaluates whether all elements of a property are Booleans.
    
    Parameters
    ----------
    val : None
        None since this rule doesn't require a referene value.
    """

    def __init__(self, value=None):
        super(AllNumberRule, self).__init__(value=value)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllNumberRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_not_when():
            invalid_values = [v for v in self._evaluated_attribute_value\
                 if not isinstance(v, (int, float))]
            if invalid_values:
                self.isValid = False
                self.invalid_value = invalid_values
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class contains \
            values which are not numbers. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=str(self.invalid_value))

        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                             ALLSTRINGSSRULE                                 #  
# --------------------------------------------------------------------------- #            
class AllStringRule(ArrayRule):
    """Evaluates whether all elements of a property are Booleans.
    
    Parameters
    ----------
    val : None
        None since this rule doesn't require a referene value.
    """

    def __init__(self, value=None):
        super(AllStringRule, self).__init__(value=value)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllStringRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_not_when():
            invalid_values = [v for v in self._evaluated_attribute_value\
                 if not isinstance(v, str)]
            if invalid_values:
                self.isValid = False
                self.invalid_value = invalid_values
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class contains \
            values which are not strings. \
            Received value: {value}.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=str(self.invalid_value))

        formatted_msg = format_text(msg)
        return formatted_msg         

# --------------------------------------------------------------------------- #
#                                ALLEQUALRULE                                 #  
# --------------------------------------------------------------------------- #            
class AllEqualRule(ArrayRule, SemanticRule):
    """Evaluates equality of a property vis-a-vis another value or property.
    
    The EqualRule applies to array-like structures validated against a 
    reference array-like structure. The validation evaluates to True if the
    evaluated and reference array-like structures have the same shape, and 
    all elements of the reference array-like are included in the evaluated 
    array-like structure.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(AllEqualRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AllEqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Convert the evaluated and reference structures to numpy arrays
        np_evaluated_value = np.array(self._evaluated_attribute_value)
        np_reference_value = np.array(self._value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_not_when():
            if np.array_equal(np_evaluated_value, np_reference_value): 
                self.isValid = True
            else:
                self.isValid = False
                self.invalid_value = attribute_value
                self.invalid_message = self.error_message()
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class is not equal \
                to {equalclass} property {equalattr} = {equalval}. \
                Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    equalclass = self._external_instance.__class__.__name__,
                    value=self._evaluated_attribute_value)

        else:
            msg = "The {attribute} property of the {classname} class is not equal \
                to {val}. \
                Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    val = self._value,
                    value=self._evaluated_attribute_value)                    
        
        formatted_msg = format_text(msg)
        return formatted_msg 


# --------------------------------------------------------------------------- #
#                               ALLALLOWEDRULE                                #  
# --------------------------------------------------------------------------- #            
class AllAllowedRule(ArrayRule, SemanticRule):
    """Evaluates whether the value(s) of an evaluated property are allowed.
    
    The AllAllowedRule applies to array-like objects and evaluates whether
    all elements are allowed.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(AllAllowedRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AllAllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value(s) to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Convert the evaluated and reference structures to numpy arrays
        np_evaluated_value = np.array(self._evaluated_attribute_value)
        np_reference_value = np.array(self._value)

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            # Evaluate whether any of the elements are not in the list of 
            # allowed values
            invalid_values = [e for e in np_evaluated_value if e not in np_reference_value]
            if invalid_values: 
                self.isValid = False
                self.Invalid_value = invalid_values
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class contains \
                values that are not among the allowed values. Received \
                    value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Allowed Value(s): {allowed}.".format(
                allowed=str(self._value)
            )

        else:
            msg = "The {attribute} property of the {classname} class does not \
                equal any of the allowed values. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Allowed value(s): {allowed}".format(
                allowed=str(self._value)
            )
        
        formatted_msg = format_text(msg)
        return formatted_msg 
# --------------------------------------------------------------------------- #
#                               ANYDISALLOWEDRULE                             #  
# --------------------------------------------------------------------------- #            
class AnyDisAllowedRule(ArrayRule, SemanticRule):
    """Evaluates whether the value(s) of an evaluated property are disallowed.
    
    The AnyDisAllowedRule applies to array-like objects and evaluates whether
    any elements are allowed.

    Parameters
    ----------
    val : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(AnyDisAllowedRule, self).__init__(value=value)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AnyDisAllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value(s) to compare to our evaluated attribute
        if self._value is None:
            self.get_external_value()

        # Convert the evaluated and reference structures to numpy arrays
        np_evaluated_value = np.array(self._evaluated_attribute_value)
        np_reference_value = np.array(self._value)

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_not_when():
            # Evaluate whether any of the elements are not in the list of 
            # allowed values
            invalid_values = [e for e in np_evaluated_value if e in np_reference_value]
            if invalid_values: 
                self.isValid = False
                self.Invalid_value = invalid_values
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        if self._external_instance:
            msg = "The {attribute} property of the {classname} class contains \
                values that are disallowed. Received \
                    value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Disallowed Value(s): {disallowed}.".format(
                disallowed=str(self._value)
            )

        else:
            msg = "The {attribute} property of the {classname} class contains \
                values that are disallowed. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Disallowed value(s): {disallowed}".format(
                disallowed=str(self._value)
            )
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                              UTILITY FUNCTIONS                              #  
# --------------------------------------------------------------------------- #
def format_text(x):
    x = " ".join(x.split())
    formatted = textwrap.fill(textwrap.dedent(x))
    return formatted        


# %%
