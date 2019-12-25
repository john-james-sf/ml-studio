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
"""Module contains the validation rules.

This module implements the strategy design pattern with a fluent interface,
and includes:

    Syntactic Rules
    ------------------
    * NoneRule : NoneRule, which evaluates whether the value of a specific 
        property is equal to None.
    * NotNoneRule : NotNoneRule, which evaluates whether the value of a 
        specific property is not equal to None.
    * EmptyRule : EmptyRule, which evaluates whether the value of a 
        specific property is empty.
    * NotEmptyRule : NotEmptyRule, which evaluates whether the value 
        of a specific property is not empty.        
    * BoolRule : BoolRule, which evaluates whether the value of a 
        specific property is Boolean.
    * IntRule : IntRule, which evaluates whether the value of a specific 
        property is an integer.
    * FloatRule : FloatRule, which evaluates whether the value of a 
        specific property is an float.
    * NumberRule : NumberRule, which evaluates whether the value of a 
        specific property is an a number.
    * StringRule : StringRule, which evaluates whether the value of a 
        specific property is a string.

    Semantic Rules
    -----------------
    * EqualRule : EqualRule, which ensures that the value of a specific property    
        is equal to a particular value  or that of another instance 
        and/or property.  
    * NotEqualRule : NotEqualRule, which ensures that the value of a specific 
        property is not equal to a particular value or that of another instance 
        and/or property.                
    * AllowedRule : AllowedRule, which ensures the value of a specific property 
        is one of a discrete set of allowed values. 
    * DisAllowedRule : EqualRule, which ensures the value of a specific property 
        is none of a discrete set of disallowed values.     
    * LessRule : LessRule, which ensures the value of a specific property is 
        less than a particular  value or that of another instance and / or 
        property. If the inclusive parameter is True, this evaluates
        less than or equal to.
    * GreaterRule : GreaterRule, which ensures the value of a specific property 
        is greater than a particulcar value or greater than the value of 
        another property. If the inclusive parameter is True, this evaluates
        greater than or equal to.
    * BetweenRule : BetweenRule, which ensures the value of a specific property 
        is between than a particulcar value or greater than the value of 
        another property. If the inclusive parameter is True, the range is 
        evaluated as inclusive.
    * RegexRule : EqualRule, which ensures the 
        value of a specific property matches the given regular expression(s).     

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

from ml_studio.services.validation.conditions import Condition
from ml_studio.services.validation.conditions import IsNone, IsEmpty, IsBool
from ml_studio.services.validation.conditions import IsInt, IsFloat
from ml_studio.services.validation.conditions import IsNumber, IsString
from ml_studio.services.validation.conditions import IsEqual, IsIn, IsLess
from ml_studio.services.validation.conditions import IsGreater, IsMatch
from ml_studio.services.validation.conditions import isArray

# --------------------------------------------------------------------------- #
#                                   RULE                                      #  
# --------------------------------------------------------------------------- #
class Rule(ABC):
    """Base class for all rules."""

    def __init__(self, *kwargs):

        # Designate unique/opaque userid and other metadata        
        self._id = uuid4()
        self._created = datetime.now()
        self._user = getpass.getuser()
        
        # The evaluated property
        self._evaluated_instance = None
        self._evaluated_classname = None
        self._evaluated_attribute_name = None
        self._evaluated_attribute_value = None  

        # Three types of conditions: 'when', 'when_any', and 'when_all'.
        # The latter two take a list of conditions
        self._conditions = dict()

        # Properties that capture the results of the validation.
        self.isValid = True
        self.decisive_values = []              
        self.invalid_message = None

    def when(self, condition):
        """Adds a single condition that must be met for a rule to apply."""        
        if isinstance(condition, Condition):
            self._conditions['when'] = condition
        else:
            raise TypeError("condition must be of type Condition.")
        return self

    def when_all(self, conditions):
        """Adds a list of rules, all of which must be met for a rule to apply."""
        if isArray(conditions):
            self._conditions['when_all'] = conditions
        else:
            raise TypeError("conditions must be an array-like object \
                containing Condition objects.")
        return self

    def when_any(self, conditions):
        """Adds a list of rules, all of which must be met for a rule to apply."""        
        if isArray(conditions):
            self._conditions['when_any'] = conditions
        else:
            raise TypeError("conditions must be an array-like object \
                containing Condition objects.")
        return self

    def _evaluate_when(self):
        condition = self._conditions.get('when')
        if condition is not None:
            return condition()
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

    def _validate_params(self):
        # Ensure the evaluated class and attribute are a valid match          
        if not hasattr(self._evaluated_instance, 
                       self._evaluated_attribute_name):
            raise AttributeError("{attrname} is not a valid attribute for \
                {classname}.".format(
                    attrname=self._evaluated_attribute_name,
                    classname=self._evaluated_classname
                ))

    @abstractmethod
    def validate(self, instance, attribute_name, attribute_value):
        self._evaluated_instance = instance
        self._evaluated_classname = instance.__class__.__name__
        self._evaluated_attribute_name = attribute_name
        self._evaluated_attribute_value = attribute_value          
        self._validate_params()

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

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)                                       
            # Otherwise, we have a literal value we can evaluate 
            elif attribute_value is not None: 
                self.decisive_values.append(attribute_value)     

            # Validity is based upon the presence of invalid values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not None. \
            Invalid value(s): '{value}'.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                NotNoneRULE                                  #  
# --------------------------------------------------------------------------- #            
class NotNoneRule(Rule):
    """Evaluates whether the value of a specific property is not None."""

    def __init__(self):
        super(NotNoneRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(NotNoneRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)                                       
            # Otherwise, we have a literal value we can evaluate 
            elif attribute_value is None: 
                self.decisive_values.append(attribute_value)     

            # Validity is based upon the presence of invalid values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class has values \
            that are None.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname)
        
        formatted_msg = format_text(msg)
        return formatted_msg



# --------------------------------------------------------------------------- #
#                                EmptyRULE                                    #  
# --------------------------------------------------------------------------- #            
class EmptyRule(Rule):
    """Evaluates whether the value of a specific property is Empty."""

    def __init__(self):
        super(EmptyRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(EmptyRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)                                       
            # Otherwise, we have a literal value we can evaluate 
            elif not IsEmpty(attribute_value)(): 
                self.decisive_values.append(attribute_value)     

            # Validity is based upon the presence of invalid values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is not Empty. \
            Invalid value(s): '{value}'.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self._evaluated_attribute_value)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                NotEmptyRULE                                 #  
# --------------------------------------------------------------------------- #            
class NotEmptyRule(Rule):
    """Evaluates whether the value of a specific property is NotEmpty."""

    def __init__(self):
        super(NotEmptyRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(NotEmptyRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)      

            # Otherwise, we have a literal value we can evaluate 
            elif not IsEmpty(attribute_value)(): 
                self.decisive_values.append(attribute_value)     

            # Validation passes if we have any non-empty values
            if self.decisive_values:
                self.isValid = True                
            else:
                self.isValid = False
                self.invalid_message = self.error_message()
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class is \
            Empty.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                 BoolRULE                                    #  
# --------------------------------------------------------------------------- #            
class BoolRule(Rule):
    """Evaluates whether the value of a specific property is NotEmpty."""

    def __init__(self):
        super(BoolRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(BoolRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)      
                                                         
            # Otherwise, we have a literal value we can evaluate 
            elif not isinstance(attribute_value, (bool, np.bool_)): 
                self.decisive_values.append(attribute_value)     

            # Validation fails if we have Non-Boolean values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()                
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class has values \
            that are not Boolean. Invalid value(s): '{value}'".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self.decisive_values)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                 IntRULE                                     #  
# --------------------------------------------------------------------------- #            
class IntRule(Rule):
    """Evaluates whether the value of a specific property is NotEmpty."""

    def __init__(self):
        super(IntRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(IntRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)      
                                                         
            # Otherwise, we have a literal value we can evaluate 
            elif not isinstance(attribute_value, int): 
                self.decisive_values.append(attribute_value)     

            # Validation fails if we have Non int values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()                
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class has values \
            that are not integers. Invalid value(s): '{value}'".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self.decisive_values)
        
        formatted_msg = format_text(msg)
        return formatted_msg


# --------------------------------------------------------------------------- #
#                                FloatRULE                                    #  
# --------------------------------------------------------------------------- #            
class FloatRule(Rule):
    """Evaluates whether the value of a specific property is NotEmpty."""

    def __init__(self):
        super(FloatRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(FloatRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)      
                                                         
            # Otherwise, we have a literal value we can evaluate 
            elif not isinstance(attribute_value, float): 
                self.decisive_values.append(attribute_value)     

            # Validation fails if we have Non float values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()                
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class has values \
            that are not floats. Invalid value(s): '{value}'".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self.decisive_values)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                NumberRULE                                   #  
# --------------------------------------------------------------------------- #            
class NumberRule(Rule):
    """Evaluates whether the value of a specific property is NotEmpty."""

    def __init__(self):
        super(NumberRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(NumberRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)      
                                                         
            # Otherwise, we have a literal value we can evaluate 
            elif not isinstance(attribute_value, (int,float)): 
                self.decisive_values.append(attribute_value)     

            # Validation fails if we have Non number values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()                
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class has values \
            that are not numbers. Invalid value(s): '{value}'".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self.decisive_values)
        
        formatted_msg = format_text(msg)
        return formatted_msg

# --------------------------------------------------------------------------- #
#                                StringRULE                                   #  
# --------------------------------------------------------------------------- #            
class StringRule(Rule):
    """Evaluates whether the value of a specific property is NotEmpty."""

    def __init__(self):
        super(StringRule, self).__init__()

    def validate(self, instance, attribute_name, attribute_value):
        super(StringRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        if self.evaluate_conditions():
            # If array, start recursion
            if isArray(attribute_value):                
                for v in attribute_value: 
                    self.validate(self._evaluated_instance, \
                        self._evaluated_attribute_name,v)      
                                                         
            # Otherwise, we have a literal value we can evaluate 
            elif not isinstance(attribute_value, str): 
                self.decisive_values.append(attribute_value)     

            # Validation fails if we have Non string values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()                
            else:
                self.isValid = True
        # If conditions aren't met, the rule can't be applied and so we 
        # set isValid to True by convention.
        else:
            self.isValid = True
        return self

    def error_message(self):
        msg = "The {attribute} property of the {classname} class has values \
            that are not strings. Invalid value(s): '{value}'".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                value=self.decisive_values)
        
        formatted_msg = format_text(msg)
        return formatted_msg              

# --------------------------------------------------------------------------- #
#                                SEMANTICRULE                                 #  
# --------------------------------------------------------------------------- #                    
class SemanticRule(Rule):
    """Abstract base class for rules requiring cross-reference to other objects.

    This class introduces the instance and attribute name for an reference
    value in the constructor. Another method is exposed to obtain the value
    from the reference object. 
    
    Parameters
    ----------
    value : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 

    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(SemanticRule, self).__init__()
        self._reference_value = value
        self._reference_instance = instance
        self._reference_attribute_name = attribute_name
        if instance:
            self._reference_classname = instance.__class__.__name__        

    def _validate_params(self):
        super(SemanticRule, self)._validate_params()
        if self._reference_instance:
            # Validate reference instance and attribute
            if not hasattr(self._reference_instance, self._reference_attribute_name):
                raise AttributeError("{classname} has no attribute {attrname}.".format(
                        classname = self._reference_classname,
                        attrname = self._reference_attribute_name
                    ))          

    def validate(self, instance, attribute_name, attribute_value):
        super(SemanticRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Obtain the value to compare to our evaluated attribute from the reference 
        # instance if not provided in the value property.
        if self._reference_instance is not None:
            self._reference_value = getattr(self._reference_instance, \
                self._reference_attribute_name)                      

# --------------------------------------------------------------------------- #
#                                 EQUALRULE                                   #  
# --------------------------------------------------------------------------- #            
class EqualRule(SemanticRule):
    """Evaluates equality of a property vis-a-vis another value or property.
    
    The EqualRule applies to basic types as well as array-like objects. Array-
    like objects are evaluated against basic types by recursing over each 
    element of the array. When evaluating two array-like objects, equality
    is evaluated element-wise using numpy array-equal.

    Parameters
    ----------
    value : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(EqualRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)

    def validate(self, instance, attribute_name, attribute_value):
        super(EqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If both evaluated and reference values are arrays, convert to
            # numpy arrays and use numpy.array_equal
            if isArray(self._evaluated_attribute_value) and isArray(self._reference_value):
                # Convert both to numpy arrays for element wise comparisons
                attribute_value = np.array(self._evaluated_attribute_value)
                reference_value = np.array(self._reference_value)
                if not np.array_equal(attribute_value, reference_value):
                    self.decisive_values.append(attribute_value)
            # If the evaluated attribute is an array like 
            # (and the reference value isn't), Recursively evaluate equality
            elif isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # If the reference value is an array (and the evaluated value isn't)
            # They obviously are not equal
            elif isArray(self._reference_value):
                self.decisive_values.append(self._evaluated_attribute_value)

            # Lastly, we are evaluating two basic, non-array types 
            elif self._evaluated_attribute_value != self._reference_value: 
                self.decisive_values.append(self._evaluated_attribute_value)

            # Equality is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        if self._reference_instance:
            msg = "The {attribute} property of the {classname} class is not equal \
                to {referenceclass} property {referenceattr} = {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceclass = self._reference_classname,
                    referenceattr = self._reference_attribute_name,
                    referenceval=self._reference_value,
                    value=self.decisive_values)

        else:
            msg = "The {attribute} property of the {classname} class is not equal \
                to {val}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    val = self._reference_value,
                    value=self.decisive_values)                    
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                              NOTEQUALRULE                                   #  
# --------------------------------------------------------------------------- #            
class NotEqualRule(SemanticRule):
    """Evaluates equality of a property vis-a-vis another value or property.
    
    The NotEqualRule applies to basic types as well as array-like objects. Array-
    like objects are evaluated against basic types by recursing over each 
    element of the array. When evaluating two array-like objects, equality
    is evaluated element-wise using numpy array-equal.

    Parameters
    ----------
    value : Any
        The validated value must be equal to this value.

    instance : ML Studio object
        The instance of a class containing the equality value

    attribute_name : str
        The name of the attribute containing the equality value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(NotEqualRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)

    def validate(self, instance, attribute_name, attribute_value):
        super(NotEqualRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If both evaluated and reference values are arrays, convert to
            # numpy arrays and use numpy.array_equal
            if isArray(self._evaluated_attribute_value) and isArray(self._reference_value):
                # Convert both to numpy arrays for element wise comparisons
                attribute_value = np.array(self._evaluated_attribute_value)
                reference_value = np.array(self._reference_value)
                if np.array_equal(attribute_value, reference_value):
                    self.decisive_values.append(attribute_value)
            # If the evaluated attribute is an array like 
            # (and the reference value isn't), Recursively evaluate equality
            elif isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # If the reference value is an array (and the evaluated value isn't)
            # They obviously are not equal
            elif isArray(self._reference_value):
                pass

            # Lastly, we are evaluating two basic, non-array types 
            elif self._evaluated_attribute_value == self._reference_value: 
                self.decisive_values.append(self._evaluated_attribute_value)

            # NotEquality is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        if self._reference_instance:
            msg = "The {attribute} property of the {classname} class is equal \
                to {referenceclass} property {referenceattr} = {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceclass = self._reference_classname,
                    referenceattr = self._reference_attribute_name,
                    referenceval=self._reference_value,
                    value=self.decisive_values)

        else:
            msg = "The {attribute} property of the {classname} class is equal \
                to {val}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    val = self._reference_value,
                    value=self.decisive_values)                    
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                                ALLOWEDRULE                                  #  
# --------------------------------------------------------------------------- #            
class AllowedRule(SemanticRule):
    """Evaluates whether the value or values of a property is/are allowed.

    The evaluated property is confirmed to be one of a set of allowed values. 
    If the evaluated property is an array-like, its values are recursively
    evaluated against the allowed values.
    
    Parameters
    ----------
    value : Any
        The allowed values.

    instance : ML Studio object
        The instance of a class containing the allowed values.

    attribute_name : str
        The name of the attribute containing the allowed values.
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(AllowedRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)

    def validate(self, instance, attribute_name, attribute_value):
        super(AllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If the evaluated attribute is an array like 
            # (and the reference value isn't), Recursively evaluate allowedity
            if isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # If evaluated attribute is not among the allowed values, append
            # to decisive values list.     
            elif self._evaluated_attribute_value not in self._reference_value:
                self.decisive_values.append(self._evaluated_attribute_value)

            # Validity is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The value of {attribute} property of the {classname} class is \
            not allowed. Allowed value(s): '{allowed}' \
            Invalid value(s): '{value}'.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                allowed = self._reference_value,
                value=self.decisive_values)

        formatted_msg = format_text(msg)
        return formatted_msg 


# --------------------------------------------------------------------------- #
#                                DISALLOWEDRULE                               #  
# --------------------------------------------------------------------------- #            
class DisAllowedRule(SemanticRule):
    """Evaluates whether the value or values of a property is/are disallowed.

    The evaluated property is confirmed not to be one of a set of disallowed 
    values. If the evaluated property is an array-like, its values are recursively
    evaluated against the disallowed values.
    
    Parameters
    ----------
    value : Any
        The disallowed values.

    instance : ML Studio object
        The instance of a class containing the disallowed values.

    attribute_name : str
        The name of the attribute containing the disallowed values.
    """

    def __init__(self, value=None, instance=None, attribute_name=None):
        super(DisAllowedRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)

    def validate(self, instance, attribute_name, attribute_value):
        super(DisAllowedRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If the evaluated attribute is an array like 
            # (and the reference value isn't), Recursively evaluate disallowedity
            if isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # If evaluated attribute is not among the disallowed values, append
            # to decisive values list.     
            elif self._evaluated_attribute_value in self._reference_value:
                self.decisive_values.append(self._evaluated_attribute_value)

            # Validity is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The value of {attribute} property of the {classname} class is \
            not allowed. Disallowed value(s): '{disallowed}' \
            Invalid value(s): '{value}'.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                disallowed = self._reference_value,
                value=self.decisive_values)

        formatted_msg = format_text(msg)
        return formatted_msg 


# --------------------------------------------------------------------------- #
#                                 LESSRULE                                    #  
# --------------------------------------------------------------------------- #            
class LessRule(SemanticRule):
    """Evaluates whether a property is less than another value or property.
    
    The LessRule applies to basic types as well as array-like objects. Array-
    like objects are evaluated against basic types by recursing over each 
    element of the array. When evaluating two array-like objects, the LessRule
    is evaluated element-wise using numpy array-less.

    Parameters
    ----------
    value : Any
        The validated value must be less to this value.

    instance : ML Studio object
        The instance of a class containing the lessity value

    attribute_name : str
        The name of the attribute containing the lessity value 

    inclusive : bool
        If True, LessRule evaluates less than or equal. Otherwise it 
        evaluates less than exclusively.
    """

    def __init__(self, value=None, instance=None, attribute_name=None,
                 inclusive=True):
        super(LessRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)
        self._inclusive = inclusive

    def _validate_params(self):
        super(LessRule, self)._validate_params()
        if isArray(self._reference_value) and not \
            isArray(self._evaluated_attribute_value):
            raise ValueError("the 'value' parameter can be an array-like, only \
                when the evaluated attribute value is an array-like.")

    def validate(self, instance, attribute_name, attribute_value):
        super(LessRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If both evaluated and reference values are arrays, convert to
            # numpy arrays and use numpy.array_less
            if isArray(self._evaluated_attribute_value) and isArray(self._reference_value):
                # Convert both to numpy arrays for element wise comparisons
                attribute_value = np.array(self._evaluated_attribute_value)
                reference_value = np.array(self._reference_value)
                if self._inclusive:
                    if not any(np.less_equal(attribute_value, reference_value)):
                        self.decisive_values.append(attribute_value)
                else:
                    if not any(np.less(attribute_value, reference_value)):
                        self.decisive_values.append(attribute_value)

            # If the evaluated attribute is an array like 
            # (and the reference value isn't), Recursively evaluate.
            elif isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # Lastly, we are evaluating two basic, non-array types             
            elif self._inclusive and \
                self._evaluated_attribute_value > self._reference_value:
                self.decisive_values.append(self._evaluated_attribute_value)
            elif not self._inclusive and \
                (self._evaluated_attribute_value >= self._reference_value):
                self.decisive_values.append(self._evaluated_attribute_value)

            # Validity is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        if self._reference_instance:
            msg = "The {attribute} property of the {classname} class is not less \
                than {referenceclass} property {referenceattr} = {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceclass = self._reference_classname,
                    referenceattr = self._reference_attribute_name,
                    referenceval=self._reference_value,
                    value=self.decisive_values)

        else:
            msg = "The {attribute} property of the {classname} class is not less \
                to {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceval = self._reference_value,
                    value=self.decisive_values)                    
        
        formatted_msg = format_text(msg)
        return formatted_msg 

# --------------------------------------------------------------------------- #
#                               GREATERRULE                                   #  
# --------------------------------------------------------------------------- #            
class GreaterRule(SemanticRule):
    """Evaluates whether a property is greater than another value or property.
    
    The GreaterRule applies to basic types as well as array-like objects. Array-
    like objects are evaluated against basic types by recursing over each 
    element of the array. When evaluating two array-like objects, the GreaterRule
    is evaluated element-wise using numpy array-greater.

    Parameters
    ----------
    value : Any
        The validated value must be greater to this value.

    instance : ML Studio object
        The instance of a class containing the greaterity value

    attribute_name : str
        The name of the attribute containing the greaterity value 

    inclusive : bool
        If True, GreaterRule evaluates greater than or equal. Otherwise it 
        evaluates greater than exclusively.
    """

    def __init__(self, value=None, instance=None, attribute_name=None,
                 inclusive=True):
        super(GreaterRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)
        self._inclusive = inclusive

    def _validate_params(self):
        super(GreaterRule, self)._validate_params()
        if isArray(self._reference_value) and not \
            isArray(self._evaluated_attribute_value):
            raise ValueError("the 'value' parameter can be an array-like, only \
                when the evaluated attribute value is an array-like.")

    def validate(self, instance, attribute_name, attribute_value):
        super(GreaterRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If both evaluated and reference values are arrays, convert to
            # numpy arrays and use numpy.array_greater
            if isArray(self._evaluated_attribute_value) and isArray(self._reference_value):
                # Convert both to numpy arrays for element wise comparisons
                attribute_value = np.array(self._evaluated_attribute_value)
                reference_value = np.array(self._reference_value)
                if self._inclusive:
                    if not all(np.greater_equal(attribute_value, reference_value)):
                        self.decisive_values.append(attribute_value)
                else:
                    if not all(np.greater(attribute_value, reference_value)):
                        self.decisive_values.append(attribute_value)

            # If the evaluated attribute is an array like 
            # (and the reference value isn't), Recursively evaluate.
            elif isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # Lastly, we are evaluating two basic, non-array types             
            elif self._inclusive and \
                self._evaluated_attribute_value < self._reference_value:
                self.decisive_values.append(self._evaluated_attribute_value)
            elif not self._inclusive and \
                (self._evaluated_attribute_value <= self._reference_value):
                self.decisive_values.append(self._evaluated_attribute_value)

            # Validity is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        if self._reference_instance:
            msg = "The {attribute} property of the {classname} class is not greater \
                than {referenceclass} property {referenceattr} = {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceclass = self._reference_classname,
                    referenceattr = self._reference_attribute_name,
                    referenceval=self._reference_value,
                    value=self.decisive_values)

        else:
            msg = "The {attribute} property of the {classname} class is not greater \
                to {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceval = self._reference_value,
                    value=self.decisive_values)                    
        
        formatted_msg = format_text(msg)
        return formatted_msg 



# --------------------------------------------------------------------------- #
#                               REGEXRULE                                     #  
# --------------------------------------------------------------------------- #            
class RegexRule(SemanticRule):
    """Evaluates whether a property is regex than another value or property.
    
    The RegexRule validates basic types and array-like objects against a
    regex string pattern. 

    Parameters
    ----------
    value : Any
        The validated value must be regex to this value.

    instance : ML Studio object
        The instance of a class containing the regexity value

    attribute_name : str
        The name of the attribute containing the regexity value 
    """

    def __init__(self, value=None, instance=None, attribute_name=None,
                 inclusive=True):
        super(RegexRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)
        self._inclusive = inclusive

    def validate(self, instance, attribute_name, attribute_value):
        super(RegexRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Raise exception if reference value is not a valid regex string.
        re.compile(self._reference_value)

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If the evaluated attribute is an array like, Recursively evaluate.
            if isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # Lastly, we are evaluating two basic, non-array types             
            else:
                matches = re.search(self._reference_value, 
                                    self._evaluated_attribute_value)
                if not matches:                
                    self.decisive_values.append(self._evaluated_attribute_value)

            # Validity is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        msg = "The {attribute} property of the {classname} class does not  \
            match regex pattern {referenceval}. \
            Invalid value(s): '{value}'.".format(
                attribute=self._evaluated_attribute_name,
                classname=self._evaluated_classname,
                referenceval=self._reference_value,
                value=self.decisive_values)

        formatted_msg = format_text(msg)
        return formatted_msg 

class BetweenRule(SemanticRule):
    """Evaluates whether the value of a property is between a min/max.

    Evaluates whether the value of a property falls within a range specified
    by a min, max, and inclusive parameter. The inclusive parameter is a 
    Boolean and indicates whether the range should be inclusive of the min and
    max parameters.

    Parameters
    ----------
    value : array-like of int or float of length=2. Optional
        Contains min and max values

    instance : ML Studio object
        The instance of a class containing reference attribute

    attribute_name : str
        The name of the attribute containing the min and max values 

    inclusive : bool
        If True, the range is inclusive of min/max values.

    """
    def __init__(self, value=None, instance=None, attribute_name=None,
                 inclusive=True):
        super(BetweenRule, self).__init__(value=value,
                                        instance=instance,
                                        attribute_name=attribute_name)
        self._inclusive = inclusive

    def validate(self, instance, attribute_name, attribute_value):
        super(BetweenRule, self).validate(instance=instance,
                                       attribute_name=attribute_name,
                                       attribute_value=attribute_value)

        # Confirm reference values contains min and max
        if not isArray(self._reference_value):
            raise TypeError("the reference value must be an array-like \
                of length=2, containing two numbers, min and max.")
        elif len(self._reference_value) != 2:
            raise ValueError("the reference value must be an array-like \
                of length=2, containing two numbers, min and max.")
        elif not isinstance(self._reference_value[0], (int, float)) or \
            not isinstance(self._reference_value[1], (int, float)):
            raise ValueError("the reference value must be an array-like \
                of length=2, containing two numbers, min and max.")

        # Evaluate iff when conditions are met.
        if self.evaluate_conditions():
            # If the evaluated attribute is an array like, Recursively evaluate.
            if isArray(self._evaluated_attribute_value):
                for v in self._evaluated_attribute_value:
                    self.validate(self._evaluated_instance, 
                                  self._evaluated_attribute_name,
                                  v)

            # Lastly, we are evaluating two basic, non-array types             
            elif self._inclusive and \
                (self._evaluated_attribute_value < self._reference_value[0] or\
                    self._evaluated_attribute_value > self._reference_value[1]):
                self.decisive_values.append(self._evaluated_attribute_value)
            elif not self._inclusive and \
                (self._evaluated_attribute_value <= self._reference_value[0] or\
                    self._evaluated_attribute_value >= self._reference_value[1]):
                self.decisive_values.append(self._evaluated_attribute_value)

            # Validity is based upon the presence of decisive (invalid) values
            if self.decisive_values:
                self.isValid = False
                self.invalid_message = self.error_message()
            else:
                self.isValid = True
        else:
            self.isValid = True

    def error_message(self):
        if self._reference_instance:
            msg = "The {attribute} property of the {classname} class is not greater \
                than {referenceclass} property {referenceattr} = {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceclass = self._reference_classname,
                    referenceattr = self._reference_attribute_name,
                    referenceval=self._reference_value,
                    value=self.decisive_values)

        else:
            msg = "The {attribute} property of the {classname} class is not greater \
                to {referenceval}. \
                Invalid value(s): '{value}'.".format(
                    attribute=self._evaluated_attribute_name,
                    classname=self._evaluated_classname,
                    referenceval = self._reference_value,
                    value=self.decisive_values)                    
        
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
