#!/usr/bin/env python3
# =========================================================================== #
#                          SERVICES: VALIDATION                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \validation.py                                                        #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Monday December 9th 2019, 2:07:26 pm                           #
# Last Modified: Monday December 9th 2019, 2:08:13 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the classes responsible for validating entity classes.

Validation is accomplished by two types of classes, a configuration class and
a validator class. The former are:

    * ValidationAttr : contains the attribute / validator relationship     
    * ValidationRule : contains the attribute / validator / class validation rule
    * Validatrix : invokes the appropriate validator based upon the rule class.

Each attribute has a validator class with a validate method that is 
invoked by the controller. The interface for the validator classes is defined
by the Validator base class.
    
"""
#%%
from abc import ABC, abstractmethod
import builtins
from collections.abc import Iterable
import os
import math
import numbers
import re
import sys
import time

import numpy as np
import pandas as pd

from ml_studio.entities.classes import Classes

# --------------------------------------------------------------------------- #
#                              UTILITY FUNCTIONS                              #
# --------------------------------------------------------------------------- #
def isArray(x):
    if isinstance(x, (np.generic, np.array, pd.Series, list, tuple)):
        return True
    else:
        return False

def isInteger(x):
    result = True if isinstance(x, int) else False
    return result

def isFloat(x):
    result = True if isinstance(x, float) else False
    return result

def isEmpty(x):
    if x == "" or x == " " or x is None:
        return True
    else:
        return False
# --------------------------------------------------------------------------- #
#                             NOTIFY FUNCTIONS                                #
# --------------------------------------------------------------------------- #
def notify_wrong_type(classname, category, attribute, value, var_type, 
                      valid_desc):
    """Renders an invalid type error message to the console."""
    invalid_values = []
    for v in value:
        msg = str(v) + " is type: " + str(type(v)) 
        invalid_values.append(msg)
    if isArray(value): 
        raise ValueError(
            """
            Invalid element(s) encountered in {classname} for the {category} 
            property {attr}. Expected {var_type}s, but encountered the following 
            invalid elements including{invalid}:\n{valid_clr_desc}""".format(
                classname=classname,
                category=category,
                attr=attribute,
                var_type = var_type,
                invalid=invalid_values[0:5],
                valid_clr_desc=valid_desc
            )
        )
        



# --------------------------------------------------------------------------- #
#                              BASEVALIDATOR                                  #
# --------------------------------------------------------------------------- #
class BaseValidator(ABC):
    """Base class for all validators.
    
    Parameters
    ----------
    instance : ML Studio object
        The instance of the class being evaluated.

    category : str
        Corresponds to a plotly object within the Layout API

    attribute : str
        The name of the attribute being evaluated. 

    empty_ok : bool. Optional. Default = False
        Indicates whether the value may be empty or None.

    array_ok : bool. Optional.  Default = False
        Specifies whether the value may be an array-like object.

    """

    def __init__(self, instance, category, attribute, empty_ok=False, 
                 array_ok=False):
        self.__classname = instance.__class__.__name__
        self.__category = category
        self.__attribute = attribute
        self.__empty_ok = empty_ok
        self.__array_ok = array_ok
    
    @abstractmethod
    def description(self):
        """Returns a string that describes the valid values for the validator."""
        raise NotImplementedError()
    
    def raise_invalid_value(self, value):
        """ Creates informative exception when an invalid value cannot be coerced.

        Parameters
        ----------
        value : Any
            Invalid value that could not be coerced

        Raises  
        -------
        ValueError
            If value is invalid and cannot be coerced.

        """
        raise ValueError(
            """
            Invalid value of type {typ} was encountered in {classname} 
            for the {category} property {attr}. The received value: {v}
            {valid_clr_desc}""".format(
                category=self.__category,
                attr=self.__attribute,
                typ=type(value),
                v=repr(value),
                valid_clr_desc=self.description(),
            )
        )

    def raise_invalid_elements(self, values):
        """ Creates an informative exception when an array has invalid elements.

        Parameters
        ----------
        values : array-like
            An array-like of elements that were invalid and could not be coerced.

        Raises
        ------
        ValueError
            If array elements were found to be invalid and unable to be coerced.

        """
        if values:
            raise ValueError(
                """
                Invalid element(s) encountered in {classname} for the {category} 
                property {attr}. Invalid elements include: {invalid}
                {valid_clr_desc}""".format(
                    classname=self.__classname,
                    category=self.__category,
                    attr=self.__attribute,
                    invalid=values[:10],
                    valid_clr_desc=self.description(),
                )
            )

    @abstractmethod
    def validate_coerce(self, value):
        """ Validate and if incompatible, coerce if possible.

        Validate whether an input value is compatible with this property, and 
        coerce the value to be compatible of possible.

        Parameters
        ----------
        value : Any
            The input value to be validated

        Raises
        ------
        ValueError
            if `value` cannot be coerced into a compatible form

        Returns
        -------
        The coerced input if possible.

        """
        raise NotImplementedError()



# --------------------------------------------------------------------------- #
#                            StringValidator                                  #
# --------------------------------------------------------------------------- #    
class StringValidator(BaseValidator):
    """String validator.
    
    Parameters
    ----------
    instance : ML Studio object
        The instance of the class being evaluated.

    category : str
        Corresponds to a plotly object within the Layout API

    attribute : str
        The name of the attribute being evaluated. 

    empty_ok : bool. Optional. Default = False
        Indicates whether the value may be empty or None.

    array_ok : bool. Optional. Default = False
        Specifies whether the value may be an array-like object.

    spaces_ok : bool. Optional. Default = False
        Indicates whether spaces in the string are valid.

    numbers_ok : bool. Optional. Default = False
        States whether numbers are permitted in the string.
    """
    def __init__(self, instance, category, attribute, empty_ok=False, 
                 array_ok=False, spaces_ok=False, numbers_ok=False):
        super(StringValidator,self).__init__(
            classname = instance.__class__.__name__,
            category = category,
            attribute = attribute,
            empty_ok = empty_ok,
            array_ok = array_ok)
        self.__spaces_ok = spaces_ok
        self.__numbers_ok = numbers_ok

    
    @abstractmethod
    def description(self):

        desc = """The {category} property, {attribute}, of the {classname} class\
            must be """.format(
                    category=self.__category,
                    attribute=self.__attribute,
                    classname=self.__classname)

        if not self.__empty_ok:
            desc = desc + """a non-empty string"""

        if self.__array_ok:
            desc = desc + """ or tuple, list, Series, or a one-dimensional \
                numpy array"""

        if not self.__spaces_ok and not self.__numbers_ok:
            desc = desc + " with alphabetic characters and no spaces"

        desc = desc + """."""

    
    def raise_invalid_value(self, value):
        """ Creates informative exception when an invalid value cannot be coerced.

        Parameters
        ----------
        value : Any
            Invalid value that could not be coerced

        Raises  
        -------
        ValueError
            If value is invalid and cannot be coerced.

        """
        raise ValueError(
            """
            Invalid value of type {typ} received by class {classname} for the 
            {category} property {attribute}
            Received value: {v}
            {valid_clr_desc}""".format(
                classname=self.__classname,
                category=self.__category,
                attribute=self.__attribute,
                typ=type(value),
                v=repr(value),
                valid_clr_desc=self.description(),
            )
        )

    def raise_invalid_elements(self, values):
        """ Creates an informative exception when an array has invalid elements.

        Parameters
        ----------
        values : array-like
            An array-like of elements that were invalid and could not be coerced.

        Raises
        ------
        ValueError
            If array elements were found to be invalid and unable to be coerced.

        """
        if values:
            raise ValueError(
                """
                Invalid element(s) encountered in class {classname} for the 
                {category} property {attribute}
                Invalid elements include: {invalid}
                {valid_clr_desc}""".format(
                    classname=self.__classname,
                    category=self.__category,
                    attribute=self.__attribute,
                    invalid=values[:10],
                    valid_clr_desc=self.description(),
                )
            )

    @abstractmethod
    def validate_coerce(self, value, notify_errors=True):
        """ Validate and if incompatible, coerce if possible.

        Validate whether an input value is compatible with this property, and 
        coerce the value to be compatible of possible.

        Parameters
        ----------
        value : Any
            The value being evaluated

        Raises
        ------
        ValueError
            if `value` cannot be coerced into a compatible form

        Returns
        -------
        The coerced input if possible.

        """

        # Check if empty and not_empty is no ok
        if value is None and not self.__empty_ok:
            self.raise_invalid_value(value)


        #Check for numbers of numbers_ok is False
        if self.__numbers_ok is False:
            if isArray(value):                
                invalid_elements = [e for e in value if bool(re.search(r'\d',e))]
                if invalid_elements and notify_errors:
                    self.raise_invalid_elements(invalid_elements)
            else:
                if bool(re.search(r'\d', value)):
                    self.raise_invalid_elements(value)            

        # Check for spaces if spaces_ok is False
        if self.__spaces_ok is False:
            if isArray(value):            
                invalid_elements = [e for e in value if " " in e]
                if invalid_elements and notify_errors:
                    self.raise_invalid_elements(invalid_elements)
            else:
                if " " in value:
                    self.raise_invalid_elements(value)                    

        # Check if array and recursively validate
        if self.__array_ok and isinstance(value, (np.array, np.generic, \
            pd.Series, list, tuple)):
            value = [self.validate_coerce(e, notify_errors=False) for e in value]
        
        # If no errors, coerce to string
        value = str(value)

        return value        

# --------------------------------------------------------------------------- #
#                            NumberValidator                                  #
# --------------------------------------------------------------------------- #    
class NumberValidator(BaseValidator):
    """Number validator.
    
    Parameters
    ----------
    instance : ML Studio object
        The instance of the class being evaluated.

    category : str
        Corresponds to a plotly object within the Layout API

    attribute : str
        The name of the attribute being evaluated. 

    min : int, float. Optional. Default = False
        Minimum value that the attribute can take.

    max : int, float. Optional. Default = False
        Maximum value that the attribute can take.

    empty_ok : bool. Optional. Default = False
        Indicates whether the value may be empty or None.

    array_ok : bool. Optional. Default = False
        Specifies whether the value may be an array-like object.

    typ : tuple. Optional. Default = (int, float)
        Specifies the valid number type(s). Valid types can be int or float.

    """
    def __init__(self, instance, category, attribute, min=None,
                 max=None, empty_ok=False, array_ok=False, typ=float):
        super(NumberValidator,self).__init__(
            classname = instance.__class__.__name__,
            category = category,
            attribute = attribute,
            empty_ok = empty_ok,
            array_ok = array_ok)
        self.__min = min
        self.__max = max
        self.__typ = typ

    
    @abstractmethod
    def description(self):

        desc = """The {category} property, {attribute}, of the {classname} class\
            must be """.format(
                    category=self.__category,
                    attribute=self.__attribute,
                    classname=self.__classname)

        if not self.__empty_ok:
            desc = desc + """a non-empty  """

        else:
            desc = desc + """none, empty or """

        if self.__typ == float:
            desc = desc + "a float, "
        else:
            desc = desc + "an integer, "

        if self.__min:
            desc = desc + "greater than or equal to " + self.__min

        if self.__max:
            desc = desc + " and less than or equal to " + self.__max 

        if self.__array_ok:
            desc = desc + """. Tuples, lists, Series, and one-dimensional \
                numpy arrays are acceptable formats"""

        desc = desc + """."""

    
    def raise_invalid_value(self, value):
        """ Creates informative exception when an invalid value cannot be coerced.

        Parameters
        ----------
        value : Any
            Invalid value that could not be coerced

        Raises  
        -------
        ValueError
            If value is invalid and cannot be coerced.

        """
        raise ValueError(
            """
            Invalid value of type {typ} received by class {classname} for the 
            {category} property {attribute}
            Received value: {v}
            {valid_clr_desc}""".format(
                classname=self.__classname,
                category=self.__category,
                attribute=self.__attribute,
                typ=type(value),
                v=repr(value),
                valid_clr_desc=self.description(),
            )
        )

    def raise_invalid_elements(self, values):
        """ Creates an informative exception when an array has invalid elements.

        Parameters
        ----------
        values : array-like
            An array-like of elements that were invalid and could not be coerced.

        Raises
        ------
        ValueError
            If array elements were found to be invalid and unable to be coerced.

        """
        if values:
            raise ValueError(
                """
                Invalid element(s) encountered in class {classname} for the 
                {category} property {attribute}
                Invalid elements include: {invalid}
                {valid_clr_desc}""".format(
                    classname=self.__classname,
                    category=self.__category,
                    attribute=self.__attribute,
                    invalid=values[:10],
                    valid_clr_desc=self.description(),
                )
            )

    @abstractmethod
    def validate_coerce(self, value, notify_errors=True):
        """ Validate and if incompatible, coerce if possible.

        Validate whether an input value is compatible with this property, and 
        coerce the value to be compatible of possible.

        Parameters
        ----------
        value : Any

        Raises
        ------
        ValueError
            if `value` cannot be coerced into a compatible form

        Returns
        -------
        The coerced input if possible.

        """

        # Check if empty and not_empty is no ok
        if isEmpty(value) and not self.__empty_ok:
            self.raise_invalid_value(value)

        # Evaluate minimum
        if self.__min and not isArray(value):
            if value < self.__min:
                self.raise_invalid_value(value)



        # Check if array and recursively validate
        if self.__array_ok and isinstance(value, (np.array, np.generic, \
            pd.Series, list, tuple)):
            value = [self.validate_coerce(e, notify_errors=False) for e in value]

        # Attempt to coerce to desired typ
        coerced_values = []
        invalid_values = []
        typ == float if self.__typ == float else typ == int
        if isArray(value):   
            for v in value:
                try:
                    coerced_v = typ(v)         
                except ValueError:
                    invalid_values.append(v)
                else:
                    coerced_values.append(coerced_v)
                
            if invalid_values:
                self.raise_invalid_elements(invalid_values)
            else:
                value = coerced_values
        else:
            try:
                value = typ(value)
            except ValueError:
                self.raise_invalid_value(value)

        # If no errors, good to go.
        return value        

# --------------------------------------------------------------------------- #
#                            ISBOOL VALIDATOR                                 #
# --------------------------------------------------------------------------- #    
class IsBool(Validator):

    def __init__(self):
        super(IsBool, self).__init__()     

    def validate(self, attribute):
        self.attribute = attribute
        if isinstance(self.attribute, bool):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a boolean."  \
                            % str(self.attribute)



# --------------------------------------------------------------------------- #
#                           ISITERABLE VALIDATOR                              #
# --------------------------------------------------------------------------- #    
class IsIterable(Validator):

    def __init__(self):
        super(IsIterable, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if isinstance(self.attribute, Iterable):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not an iteratable."  \
                            % str(self.attribute)   



  



# %%
