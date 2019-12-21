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
"""Module defines the rules used for validation.

The rules module categorizes rules in terms of the type of validation they 
support and the data structure. The 

    * Syntactic Rules : Rules that ensure the data is syntactically correct
    * Semantic Rules : Rules that ensure the data makes logical sense within
        a context of other data elements or values  
    * Semantic Rules (Arrays) : Rules that operate on arrays and verify 
        semantic correctness via other data elements or values
    * Syntactic Rules (Arrays) : Rules that operate on arrays and verify 
        syntactical correctness

    The classes are as follows:

    Syntactic Rules
    --------------- 
    * BaseRule : The abstract base class for rule classes.
    * NoneRule : Ensures that a specific property is None.
    * NotNoneRule : Ensures that a specific property is not None.
    * EmptyRule : Ensures that a specific property is None, empty or whitespace.
    * NotEmptyRule : Ensures that a specific property is not None, the
        empty string or whitespace.
    * BoolRule : Ensures that the value of a specific property is a Boolean.
    * IntRule : Ensures that the value of a specific property is an integer.
    * FloatRule : Ensures that the value of a specific property is a float.
    * NumberRule : Ensures that the value of a specific property is a number.
    * StringRule : Ensures that the value of a specific property is a string.    

    Semantic Rules
    --------------
    * SemanticRule : Base class for semantic rules.
    * EqualRule : Ensures that the value of a specific property is
        equal to a particular value or the value of another property.        
    * NotEqualRule : Ensures that the value of a specific property is
        not equal to a particular value or the value of another property.
    * AllowedRule :Ensures the value of a specific property is one of a 
        discrete set of allowed values. 
    * DisAllowedRule :Ensures the value of a specific property is none of a 
        discrete set of disallowed values.     
    * LessRule : Ensures the value of a specific property is less than
        a partiulcar value or less than the value of another property.
    * GreaterRule : Ensures the value of a specific property is greater
        than a particulcar value or greater than the value of another property.        
    * RegexRule : Ensures the value of a specific property matches
        the given regular expression(s). 

    Syntactic Array Rules
    ---------------------
    ArrayRules : Abstract base class for rules applying to arrays.
    AllBoolRule : Ensures that all elements of a specific property are Boolean.
    AllIntRule : Ensures that all elements of a specific property are integers.
    AllFloatRule : Ensures that all elements of a specific property are floats.
    AllNumberRule : Ensures that all elements of a specific property are numbers.
    AllStringRule : Ensures that all elements of a specific property are strings.

    Semantic Array Rules
    --------------------
    AllEqualRule : Ensures that all elements of a specific property are equal 
        to all elements of a reference array-like.
    AllAllowedRule : Ensures that all elements of a specific property are allowed.
    AnyDisAllowedRule : Evaluates whether any elements of a specific property are 
        disallowed.

"""

from abc import ABC, abstractmethod
import builtins
from collections.abc import Iterable
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

from ml_studio.services.validation.conditions import isNone, isEmpty, isBool
from ml_studio.services.validation.conditions import isDate, isInt, isFloat
from ml_studio.services.validation.conditions import isNumber, isString, isDate
from ml_studio.services.validation.conditions import isEqual, isIn, isLess
from ml_studio.services.validation.conditions import isGreater, isMatch
from ml_studio.services.validation.conditions import isArray

# --------------------------------------------------------------------------- #
#                                   RULE                                      #  
# --------------------------------------------------------------------------- #
class Rule(ABC):
    """Base class for all rules."""

    def __init__(self, val=None, *args, **kwargs):

        # The value to which the rule applies if any
        self._val = val

        # The evaluated property
        self._evaluated_instance = None
        self._evaluated_classname = None
        self._evaluated_attribute_name = None
        self._evaluated_attribute_value = None  

        # Two lists. Each containing conditions for when the rule should (when) or
        # should not (except when) be applied. Each time the when and 
        # except when methods are called, the condition is added to 
        # the appropriate list.
        self._when = []
        self._except_when = []

        # Properties that capture the results of the validation.
        self.isValid = True
        self.invalid_value = None              
        self.invalid_message = None

    def _condition(self, condition_func, **kwargs):
        """Formats a condition function and parameters for runtime execution.
        
        Parameters
        ----------
        condition_func : function
            Condition function.            

        **kwargs : dict
            Parameters for the condition function in the form of two dictionaries:
                a_dict : the first parameter for the condition function
                b_dict : the second parameter for the condition function, if required.

                a_dict, and optionally b_dict contain either:
                    instance : an instance of a class containing a property to evaluate
                    attribute_name : the name of the attribute to evaluate
                    or
                    value : the value of the parameter

        Raises
        ------
        AttributeError if an instance and attribute name are provided and 
        attribute indicated by the attribute name is not a valid attribute
        for the instance.         
        """
        # Unpack kwargs
        a_dict = kwargs.get('a_dict')
        a_instance = a_dict.get('instance')
        a_attribute_name = a_dict.get('attribute_name')
        a_value = a_dict.get('value')

        b_dict = kwargs.get('b_dict')
        b_instance = b_dict.get('instance')
        b_attribute_name = b_dict.get('attribute_name')
        b_value = b_dict.get('value')

        # If instance and attribute are provided, confirm valid match
        if a_instance is not None and a_attribute_name is not None:
            try:
                getattr(a_instance, a_attribute_name)
            except AttributeError:
                print("Class {classname} has no attribute {attrname}.".format(
                    classname=a_instance.__class__.__name__,
                    attrname=a_attribute_name
                    ))
        if b_instance is not None and b_attribute_name is not None:            
            try:
                getattr(b_instance, b_attribute_name)
            except AttributeError:
                print("Class {classname} has no attribute {attrname}.".format(
                    classname=b_instance.__class__.__name__,
                    attrname=b_attribute_name
                ))            

        # Format the condition 
        c = {}
        c['condition_func'] = condition_func
        c['a_instance'] = a_instance
        c['a_attribute_name'] = a_attribute_name
        c['a_value'] = a_value
        c['b_instance'] = b_instance
        c['b_attribute_name'] = b_attribute_name
        c['b_value'] = b_value  
        return c

    def when(self, condition_func, **kwargs):
        """Updates list of pre-conditions for a rule."""
        condition = self._condition(condition_func, **kwargs)
        self._when.append(condition)

    def except_when(self, condition_func, **kwargs):
        """Updates a list of except pre-conditions for a rule."""
        condition = self._condition(condition_func, **kwargs)
        self._except_when.append(condition)

    def evaluate_when(self):
        """Evaluates conditions and returns true if all conditions were met."""
        if self._when:
            when_valid = []
            # Iterate through 'when' conditions
            for when in self._when:
                # GET A_VALUE
                # Attempt to get the a_value direct from the dictionary 
                if when['a_value'] is not None:
                    a_value = when['a_value']
                # Otherwise, extract from the instance and attribute provided
                elif when['a_instance'] is not None:
                    instance = when['a_instance']
                    attribute_name = when['a_attribute_name']
                    try:
                        a_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))
                # Lastly, if the a_instance is None, assume the attribute is 
                # for the evaluated instance.
                else:
                    instance = self._evaluated_instance
                    attribute_name = when['a_attribute_name']
                    try:
                        a_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))                    

                # GET B_VALUE
                # Attemp to get the b_value directly from the dictionary
                if when['b_value'] is not None:
                    b_value = when['b_value']
                # Otherwise, extract from the instance and attribute provided
                elif when['b_instance'] is not None:
                    instance = when['b_instance']
                    attribute_name = when['b_attribute_name']
                    try:
                        b_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))
                # Lastly, if the b_instance is None, assume the attribute is 
                # for the evaluated instance.
                else:
                    instance = self._evaluated_instance
                    attribute_name = when['b_attribute_name']
                    try:
                        b_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))     

                # Execute the condition and append the result to the 'when_valid' list
                condition = when['condition_func']
                when_valid.append(condition(a_value, b_value))
            
            # If all valid, then True, otherwise False indicates the condition is not met.
            if all(when_valid):
                return True
            else:
                return False
        # If there is no when condition, a True value just means we proceed with validation.
        else:
            return True

    def evaluate_except_when(self):
        """Evaluates conditions and returns true if all conditions were NOT met."""
        if self._except_when:
            except_when_valid = []
            # Iterate through 'when' conditions
            for except_when in self._except_when:
                # GET A_VALUE
                # Attempt to get the a_value direct from the dictionary 
                if except_when['a_value'] is not None:
                    a_value = except_when['a_value']
                # Otherwise, extract from the instance and attribute provided
                elif except_when['a_instance'] is not None:
                    instance = except_when['a_instance']
                    attribute_name = except_when['a_attribute_name']
                    try:
                        a_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))
                # Lastly, if the a_instance is None, assume the attribute is 
                # for the evaluated instance.
                else:
                    instance = self._evaluated_instance
                    attribute_name = except_when['a_attribute_name']
                    try:
                        a_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))                    

                # GET B_VALUE
                # Attemp to get the b_value directly from the dictionary
                if except_when['b_value'] is not None:
                    b_value = except_when['b_value']
                # Otherwise, extract from the instance and attribute provided
                elif except_when['b_instance'] is not None:
                    instance = except_when['b_instance']
                    attribute_name = except_when['b_attribute_name']
                    try:
                        b_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))
                # Lastly, if the b_instance is None, assume the attribute is 
                # for the evaluated instance.
                else:
                    instance = self._evaluated_instance
                    attribute_name = except_when['b_attribute_name']
                    try:
                        b_value = getattr(instance, attribute_name)
                    except AttributeError:
                        print("Class {classname} has no attribute {attrname}.".format(
                            classname = instance.__class__.__name__,
                            attrname = attribute_name
                        ))     

                # Execute the condition and append the result to the 'except_when_valid' list
                condition = except_when['condition_func']
                except_when_valid.append(condition(a_value, b_value))
            
            # If all valid, then True, otherwise False indicates the condition is not met.
            if any(except_when_valid):
                return False
            else:
                return True
        # If there is no except_when condition, a True value just means we proceed with validation.
        else:
            return True      

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

        if self.evaluate_when() and self.evaluate_except_when():
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

        if self.evaluate_when() and self.evaluate_except_when():
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

        if self.evaluate_when() and self.evaluate_except_when():
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

        if self.evaluate_when() and self.evaluate_except_when():
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
                array-like structures, use AllBollRule.")

        if self.evaluate_when() and self.evaluate_except_when():
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

        if self.evaluate_when() and self.evaluate_except_when():
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

        if self.evaluate_when() and self.evaluate_except_when():
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

        if self.evaluate_when() and self.evaluate_except_when():
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

        if self.evaluate_when() and self.evaluate_except_when():
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(SemanticRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_classname = instance.__class__.__name__
        self._external_attribute_name = attribute_name

    def get_external_value(self):
        try:
            self._external_attribute_value = getattr(self._external_instance, \
                self._external_attribute_name)
            self._val = self._evaluated_attribute_value
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(EqualRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(EqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
            if attribute_value == self._val: 
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
                    val = self._val,
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(NotEqualRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(NotEqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
            if attribute_value != self._val: 
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
                    val = self._val,
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(AllowedRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

         # Obtain the value(s) to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
            if attribute_value in self._val: 
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
                allowed=str(self._val)
            )

        else:
            msg = "The {attribute} property of the {classname} class does not \
                equal any of the allowed values. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Allowed value(s): {allowed}".format(
                allowed=str(self._val)
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(DisAllowedRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(DisAllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

         # Obtain the value(s) to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
            if attribute_value not in self._val: 
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
                disallowed=str(self._val)
            )

        else:
            msg = "The {attribute} property of the {classname} class equal \
                a disallowed value. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Disallowed value(s): {disallowed}".format(
                disallowed=str(self._val)
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

    def __init__(self, val=None, instance=None, attribute_name=None,
                 equal_ok=False):
        super(LessRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name
        self._equal_ok = equal_ok

    def validate(self, instance, attribute_name, attribute_value):
        super(LessRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
            if attribute_value < self._val: 
                self.isValid = True
            elif self._equal_ok:
                if attribute_value == self._val:
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
                    externalval = self._val,
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
                    val=self._val,
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

    def __init__(self, val=None, instance=None, attribute_name=None,
                 equal_ok=False):
        super(GreaterRule, self).__init__(val=val)
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
        if self._val is None:
            self.get_external_value()

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
            if attribute_value > self._val: 
                self.isValid = True
            elif self._equal_ok:
                if attribute_value == self._val:
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
                    externalval = self._val,
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
                    val=self._val,
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

    def __init__(self, val=None, instance=None, attribute_name=None,
                 equal_ok=False):
        super(RegexRule, self).__init__(val=val)
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
        if self._val is None:
            self.get_external_value()

        # Validate regex pattern(s)
        if isinstance(self._val, str):
            self.validate_regex(self._val)
        elif isArray(self._val):
            not_strings = [v for v in self._val if not isinstance(v, str)]
            if not_strings:
                raise ValueError("Some or all reference values are not strings.")
            else:
                for v in self._val:
                    self.validate_regex(v)                

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
            # If array, then we evaluate to True of any matches.
            if isArray(self._val):                
                regex_matches = []
                for regex in self._val:
                    regex_matches.append(\
                        self.evaluate_regex(self._evaluated_attribute_value, regex))

                if any(regex_matches):
                    self.isValid = True
                else:
                    self.isValid = False
                    self.invalid_value = attribute_value
                    self.invalid_message = self.error_message()
            elif self.evaluate_regex(self._evaluated_attribute_value, self._val):
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
            patterns=str(self._val)
        )

        formatted_msg = format_text(msg)
        return formatted_msg 


# --------------------------------------------------------------------------- #
#                                ARRAYRULES                                   #   
# --------------------------------------------------------------------------- #                            
class ArrayRule(Rule):
    """Abstract base class to evaluate array-like structures."""
    def __init__(self):
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

    def __init__(self, val=None):
        super(AllBoolRule, self).__init__(val=val)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllBoolRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Convert the evaluated value to a numpy array
        np_evaluated_value = np.array(self._evaluated_attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_except_when():
            invalid_values = [v for v in np_evaluated_value\
                 if not isinstance(v, bool)]
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

    def __init__(self, val=None):
        super(AllIntRule, self).__init__(val=val)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllIntRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Convert the evaluated value to a numpy array
        np_evaluated_value = np.array(self._evaluated_attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_except_when():
            invalid_values = [v for v in np_evaluated_value\
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

    def __init__(self, val=None):
        super(AllFloatRule, self).__init__(val=val)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllFloatRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Convert the evaluated value to a numpy array
        np_evaluated_value = np.array(self._evaluated_attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_except_when():
            invalid_values = [v for v in np_evaluated_value\
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

    def __init__(self, val=None):
        super(AllNumberRule, self).__init__(val=val)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllNumberRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Convert the evaluated value to a numpy array
        np_evaluated_value = np.array(self._evaluated_attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_except_when():
            invalid_values = [v for v in np_evaluated_value\
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

    def __init__(self, val=None):
        super(AllStringRule, self).__init__(val=val)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllStringRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Convert the evaluated value to a numpy array
        np_evaluated_value = np.array(self._evaluated_attribute_value)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_except_when():
            invalid_values = [v for v in np_evaluated_value\
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(AllEqualRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AllEqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Convert the evaluated and reference structures to numpy arrays
        np_evaluated_value = np.array(self._evaluated_attribute_value)
        np_reference_value = np.array(self._val)

        # Evaluate if when / except when conditions are met/not met, then proceed.
        if self.evaluate_when() and self.evaluate_except_when():
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
                    val = self._val,
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(AllAllowedRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AllAllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value(s) to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Convert the evaluated and reference structures to numpy arrays
        np_evaluated_value = np.array(self._evaluated_attribute_value)
        np_reference_value = np.array(self._val)

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
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
                allowed=str(self._val)
            )

        else:
            msg = "The {attribute} property of the {classname} class does not \
                equal any of the allowed values. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Allowed value(s): {allowed}".format(
                allowed=str(self._val)
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

    def __init__(self, val=None, instance=None, attribute_name=None):
        super(AnyDisAllowedRule, self).__init__(val=val)
        self._external_instance = instance
        self._external_attribute_name = attribute_name

    def validate(self, instance, attribute_name, attribute_value):
        super(AnyDisAllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value(s) to compare to our evaluated attribute
        if self._val is None:
            self.get_external_value()

        # Convert the evaluated and reference structures to numpy arrays
        np_evaluated_value = np.array(self._evaluated_attribute_value)
        np_reference_value = np.array(self._val)

        # Evaluate if when / except when conditions are met/not met.
        if self.evaluate_when() and self.evaluate_except_when():
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
                disallowed=str(self._val)
            )

        else:
            msg = "The {attribute} property of the {classname} class contains \
                values that are disallowed. Received value: {value}.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    value=str(self._evaluated_attribute_value))
            msg = msg + "Disallowed value(s): {disallowed}".format(
                disallowed=str(self._val)
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
