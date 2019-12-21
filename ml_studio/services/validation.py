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
import textwrap
import time

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#                              UTILITY FUNCTIONS                              #
# --------------------------------------------------------------------------- #
def isArray(x):
    if isinstance(x, (np.ndarray,pd.Series, list, tuple)):
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

def format_text(x):
    x = " ".join(x.split())
    formatted = textwrap.fill(textwrap.dedent(x))
    return formatted

# --------------------------------------------------------------------------- #
#                             NOTIFY FUNCTIONS                                #
# --------------------------------------------------------------------------- #
def raise_type_error(classname, category, attribute, value, var_type, 
                      valid_desc):
    """Renders an invalid type error message to the console."""
    # Format invalid values
    invalid_values = []
    for v in value:
        msg = str(v) + " is type: " + str(type(v)._name_) 
        invalid_values.append(msg)

    if isArray(value):
        msg = """\
        Invalid element(s) encountered by the """ + category + " property " + \
        attribute + ". Expected " + var_type + "s, but encountered the \
        following invalid elements including " + str(invalid_values[0:5]) + " "\
        + valid_desc
    
    else:
        msg = """\
            Invalid value encountered by the """ + category + " property " + \
            attribute + ". Expected " + var_type + "s, but encountered the \
            following invalid elements including " + str(invalid_values[0:5]) + " "\
            + valid_desc

    formatted = format_text(msg)

    raise ValueError(formatted)

# --------------------------------------------------------------------------- #
# Raise empty variable error
def raise_empty_error(classname, category, attribute, valid_desc):
    """Renders an invalid empty variable message to the console."""

    msg = """\
        Invalid empty value encountered by the """ +  classname + \
        " class for the " + category + " property, " + attribute + \
        ". " + valid_desc

    formatted = format_text(msg)

    raise ValueError(formatted)

# --------------------------------------------------------------------------- #
# Raise numeric out of range
def raise_number_out_of_range_error(classname, category, attribute, value, 
                                    min_range, max_range, valid_desc):
    """Renders a message when value or values are outside designated range."""

    if isArray(value):
        msg = """\
            Encountered element(s) in the """ + category + " property " + \
            attribute + ". that are outside [{min_range},{max_range}], ".format(
                min_range=min_range,
                max_range=max_range
                ) + "including: {values}. ".format(
                    values=value[0:5]
                ) + valid_desc
    else:
        msg = category + " property " + attribute + " = {value} is out of the\
        acceptable range [{min_range}, {max_range}]. ".format(
            value=value,
            min_range=min_range,
            max_range=max_range) + valid_desc

    formatted = format_text(msg)

    raise ValueError(formatted)

# --------------------------------------------------------------------------- #
# Raise array_error
def raise_array_error(classname, category, attribute, valid_desc):
    """Renders an invalid message if this is an array and arrays are incompatible."""

    msg = """\
        Encountered an array in the """ + category + " property " + \
        attribute + ". Arrays are incompatible with the requirements\
        of this attribute." + valid_desc

    formatted = format_text(msg)

    raise ValueError(formatted)
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

    value : Any
        The value being evaluated.

    empty_ok : bool. Optional. Default = False
        Indicates whether the value may be empty or None.

    array_ok : bool. Optional.  Default = False
        Specifies whether the value may be an array-like object.

    """

    def __init__(self, instance, category, attribute, value, empty_ok=True, 
                 array_ok=False, *args, **kwargs):
        self._classname = instance.__class__.__name__
        self._category = category
        self._attribute = attribute
        self._empty_ok = empty_ok
        self._array_ok = array_ok
    
    @abstractmethod
    def description(self):
        """Returns a string that describes the valid values for the validator."""
        raise NotImplementedError()
    
    @abstractmethod
    def validate_coerce(self, value, notify_errors=True):
        """ Validate and if incompatible, coerce if possible.

        Validate whether an input value is compatible with this property, and 
        coerce the value to be compatible of possible. This base class will 
        perform basic validation common to all subclasses, and more specific 
        validations will be left to the subclasses.

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
        # Check empty error must be raised. is empty
        if not self._empty_ok and isEmpty(value):
            raise_empty_error(classname=self._classname,
                              category=self._category,
                              attribute=self._attribute,
                              valid_desc=self.description())

        # Check if invalid array
        if not self._array_ok and isArray(value):
            raise_array_error(classname=self._classname,
                              category=self._category,
                              attribute=self._attribute,
                              valid_desc=self.description())            

        # If array, recurse until we have a non array-like value
        if isArray(value):
            value = [self.validate_coerce(e, notify_errors=False) for e in value]

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
        States whether spaces are permitted in the string.        

    """
    def __init__(self, instance, category, attribute, empty_ok=True, 
                 array_ok=False, spaces_ok=False, numbers_ok=False):
        super(StringValidator,self).__init__(
            instance = instance,
            category = category,
            attribute = attribute,
            empty_ok = empty_ok,
            array_ok = array_ok)        
        self._spaces_ok = spaces_ok
    
    def description(self):

        desc = """The {category} property, {attribute}, of the {classname} class\
            must be """.format(
                    category=self._category,
                    attribute=self._attribute,
                    classname=self._classname)

        if not self._empty_ok:
            desc = desc + """a non-empty string"""

        if self._array_ok:
            desc = desc + """ or tuple, list, Series, or a one-dimensional \
                numpy array"""

        if not self._spaces_ok:
            desc = desc + " with alphanumeric characters and without spaces"

        desc = desc + """."""

        return desc
    
   
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
        super(StringValidator,self).validate_coerce(value = value)        
        # Check if spaces error must be raised
        if not self._spaces_ok and " " in value:
            raise_spaces_in_string_error(
                classname=self._classname,
                category=self._category,
                attribute=self._attribute,
                value=value,
                valid_desc=self.description()
            )
        # Check if array and recursively validate
        if not self._array_ok and isArray(value):
            raise_array_error(
                classname=self._classname,
                category=self._category,
                attribute=self._attribute,
                valid_desc=self.description()
            )

        if self._array_ok and isArray(value):
            value = [self.validate_coerce(e, notify_errors=False) for e in value]
  
        else:
            for v in value:
                self.coerced_values(v)

        # Check if empty and not_empty is no ok
        if isEmpty(value) and not self._empty_ok:
            raise_empty_error(classname = self._classname, 
                              category=self._category, 
                              attribute=self._attribute,
                              valid_desc=self.description())

        # Check for spaces if not permitted
        if not self._spaces_ok:
            if isArray(value):
                invalid_values = []
                invalid.values.append([v for v in value if " " in v])
                if invalid_values:
                    raise_spaces_in_string_error(
                        classname=self._classname,
                        category=self._category,
                        attribute=self._attribute,
                        value=value,
                        valid_desc=self.description()

                    )


        
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

    empty_ok : bool. Optional. Default = True
        Indicates whether the value may be empty or None.

    array_ok : bool. Optional. Default = False
        Specifies whether the value may be an array-like object.

    """
    def __init__(self, instance, category, attribute, min=None,
                 max=None, empty_ok=True, array_ok=False):
        super(NumberValidator,self).__init__(
            instance = instance,
            category = category,
            attribute = attribute,
            empty_ok = empty_ok,
            array_ok = array_ok)

        if min is None:            
            self._min = float("-inf")
        else:
            self._min = min
        if max is None:
            self._max = float("inf")
        else:
            self._max = max

        if min is None and max is None:
            self._has_min_max = False
        else:
            self._has_min_max = True


    
    def description(self):

        desc = """The {category} property, {attribute}, of the {classname} class\
            must be """.format(
                    category=self._category,
                    attribute=self._attribute,
                    classname=self._classname)

        if not self._empty_ok:
            desc = desc + """a non-empty int or float """

        else:
            desc = desc + """none, empty or a int or float """

        if self._has_min_max:
            desc = desc + "in interval [{min_val},{max_val}]".format(
                min_val=self._min,
                max_val=self._max
            )

        if self._array_ok:
            desc = desc + """. Tuples, lists, Series, and one-dimensional \
                numpy arrays are acceptable"""

        desc = desc + """."""
    
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
        # If array not ok, raise validation error
        if not self._array_ok and isArray(value):
            raise_array_error(classname=self._classname,
                              category=self._category,
                              attribute=self._attribute,
                              valid_desc=self.description()
                              )

        # Otherwise, if array begin recursive validation
        if isArray(value) and self._array_ok:
            for v in value:
                self.validate_coerce(value=v, notify_errors=False)

        # Check if empty and not_empty is no ok
        if isEmpty(value) and not self._empty_ok:
            raise_empty_error(classname=self._classname,
                              category=self._category,
                              attribute=self._attribute,
                              valid_desc=self.description()
                              )

        # Evaluate minimum
        if isArray(value) and self._array_ok:
            invalid_values = [v for v in value if v < self._min]
            if invalid_values:
                raise_below_min_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=invalid_values, min=self._min,
                                     valid_desc=self.description())
            
        else:
            if value < self._min:
                raise_below_min_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=value, min=self._min,
                                     valid_desc=self.description())

        # Evaluate maximum
        if isArray(value) and self._array_ok:
            invalid_values = [v for v in value if v > self._max]
            if invalid_values:
                raise_above_max_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=invalid_values, max=self._max,
                                     valid_desc=self.description())
            
        else:
            if value > self._max:
                raise_above_max_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=value, max=self._max,
                                     valid_desc=self.description())

        # No errors, attempt to coerce
        invalid_values = []
        coerced_values = []        
        if isArray(value):   
            for v in value:
                try:
                    coerced_v = self._var_type(v)         
                except ValueError:
                    invalid_values.append(v)
                else:
                    coerced_values.append(coerced_v)
                
            if invalid_values:
                raise_type_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=value, 
                                     var_type=self._var_type,
                                     valid_desc=self.description())

        else:
            try:
                coerced_v = self._var_type(value)
            except:
                invalid_values.append(value)
            else:
                coerced_values.append(coerced_v)

        return coerced_values

# --------------------------------------------------------------------------- #
#                            DiscreteValidator                                #
# --------------------------------------------------------------------------- #    
class DiscreteValidator(BaseValidator):
    """Validates when discrete values are allowed.
    
    Parameters
    ----------
    instance : ML Studio object
        The instance of the class being evaluated.

    category : str
        Corresponds to a plotly object within the Layout API

    attribute : str
        The name of the attribute being evaluated. 

    allowed : int, float, str or array-like thereof.
        The value or values that are allowed for this variable.

    empty_ok : bool. Optional. Default = False
        Indicates whether the value may be empty or None.

    array_ok : bool. Optional. Default = False
        Specifies whether the value may be an array-like object.

    """
    def __init__(self, instance, category, attribute, min=None,
                 max=None, empty_ok=True, array_ok=False, var_type=float):
        super(NumberValidator,self).__init__(
            instance = instance,
            category = category,
            attribute = attribute,
            var_type = var_type,
            empty_ok = empty_ok,
            array_ok = array_ok)
        self._min = min
        self._max = max

    
    def description(self):

        desc = """The {category} property, {attribute}, of the {classname} class\
            must be """.format(
                    category=self._category,
                    attribute=self._attribute,
                    classname=self._classname)

        if not self._empty_ok:
            desc = desc + """a non-empty  """

        else:
            desc = desc + """none, empty or """

        if self._var_type == float:
            desc = desc + "a float, "
        else:
            desc = desc + "an integer, "

        if self._min:
            desc = desc + "greater than or equal to " + self._min

        if self._max:
            desc = desc + " and less than or equal to " + self._max 

        if self._array_ok:
            desc = desc + """. Tuples, lists, Series, and one-dimensional \
                numpy arrays are acceptable formats"""

        desc = desc + """."""
    
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
        # If array not ok, raise validation error
        if not self._array_ok and isArray(value):
            raise_empty_error(classname=self._classname,
                              category=self._category,
                              attribute=self._attribute,
                              value=value,
                              var_type=self._var_type,
                              valid_desc=self.description()
                              )

        # Otherwise, if array begin recursive validation
        if isArray(value) and self._array_ok:
            for v in value:
                self.validate_coerce(value=v, notify_errors=False)

        # Check if empty and not_empty is no ok
        if isEmpty(value) and not self._empty_ok:
            raise_empty_error(classname=self._classname,
                              category=self._category,
                              attribute=self._attribute,
                              value=value,
                              var_type=self._var_type,
                              valid_desc=self.description()
                              )

        # Evaluate minimum
        if isArray(value) and self._array_ok:
            invalid_values = [v for v in value if v < self._min]
            if invalid_values:
                raise_below_min_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=invalid_values, min=self._min,
                                     valid_desc=self.description())
            
        else:
            if value < self._min:
                raise_below_min_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=value, min=self._min,
                                     valid_desc=self.description())

        # Evaluate maximum
        if isArray(value) and self._array_ok:
            invalid_values = [v for v in value if v > self._max]
            if invalid_values:
                raise_above_max_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=invalid_values, max=self._max,
                                     valid_desc=self.description())
            
        else:
            if value > self._max:
                raise_above_max_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=value, max=self._max,
                                     valid_desc=self.description())

        # No errors, attempt to coerce
        invalid_values = []
        coerced_values = []        
        if isArray(value):   
            for v in value:
                try:
                    coerced_v = self._var_type(v)         
                except ValueError:
                    invalid_values.append(v)
                else:
                    coerced_values.append(coerced_v)
                
            if invalid_values:
                raise_type_error(classname=self._classname, category=self._category,
                                     attribute=self._attribute, value=value, 
                                     var_type=self._var_type,
                                     valid_desc=self.description())

        else:
            try:
                coerced_v = self._var_type(value)
            except:
                invalid_values.append(value)
            else:
                coerced_values.append(coerced_v)

        return coerced_values
