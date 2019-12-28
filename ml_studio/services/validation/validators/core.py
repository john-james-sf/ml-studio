#!/usr/bin/env python3
# =========================================================================== #
#                          VALIDATORS : CORE                                  #
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
# Create Date: Saturday December 21st 2019, 8:22:27 am                        #
# Last Modified: Friday December 27th 2019, 10:17:52 am                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the core Validator classes. 

Validation is governed by the following validator classes:

    Validators
    ----------
    * Validator : Abstract base class for the following Validator classes.
    * Validerator : Rule Iterator for Validator classes.
    * StringValidator : Ensures a property value is a valid string.
    * BooleanValidator : Ensures a property value is a valid Boolean.
    * AllowedValuesValidator : Ensures a property value is among the set of
        allowed values.
    * ForbiddenValuesValidator : Ensures a property value is not among 
        the set of forbidden values.        
    * NumberValidator : Ensures a property value is a valid number
    * IntegerValidator : Ensures a property value is a valid integer    
    * ArrayValidator : Ensures a property value is a valid array.
    
    
"""
#%%
from abc import ABC, abstractmethod, abstractproperty
import builtins
from collections.abc import Iterable, Iterator
from datetime import datetime
import getpass
import textwrap
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from ml_studio.utils.misc import format_text

# =========================================================================== #
#                              CORE VALIDATORS                                #
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
    def __init__(self, instance, target_name, parent_name,
                 object_type="Validator", **kwargs):
        self._instance = instance
        self._target_name = target_name
        self._parent_name = parent_name
        self._object_name = self.__class__.__name__
        self._object_type = object_type
        # Metadata
        self._id = uuid4()
        self._created = datetime.now()
        self._user = getpass.getuser()
        # Components
        self._components = []        
        self._results = []
        self._invalid_messages = []
        self._is_valid = True

    def __iter__(self):
        return Validerator(self._components)    

    # ----------------------------------------------------------------------- #
    #                            Properties                                   #
    # ----------------------------------------------------------------------- #
    @property
    def is_valid(self):
        return self._is_valid
    @property
    def target(self):
        return self._target_name

    @property
    def target_name(self):
        return self._target_name        
    
    @property
    def parent_name(self):
        return self._parent_name

    @property
    def invalid_messages(self):
        return self._invalid_messages

    # ----------------------------------------------------------------------- #
    #                            Composite Members                            #
    # ----------------------------------------------------------------------- #
    def get_component(self, name):
        """Retrieves a Rule or a RuleSet composite object from the validator.

        Parameters
        ----------
        name : str
            Component name. This is the class name for the target object. 

        Returns
        -------
        Rule object

        """        
        if self._components:
            for component in self._components:
                if name.lower() in component.__class__.__name__.lower():
                    return component

    def add_component(self, component):
        """Adds a Rule or a RuleSet composite object to the validator.

        Parameters
        ----------
        component : Validator, RuleSet, Rule, or Condition object
            Component to add to the object.

        """ 
        self._components.append(component)
        return self

    def del_component(self, component):
        """Deletes a component from the aggregation.

        Parameters
        ----------
        component : Validator, RuleSet, Rule, or Condition object
            Component to delete from the object.

        """        
        self._components.remove(component)   
        return self

    # ----------------------------------------------------------------------- #
    #                              Validate                                   #
    # ----------------------------------------------------------------------- #    
    def validate(self):
        if self._components:
            self._invalid_messages = []
            self._results = []
            for component in self._components:
                component.validate()
                self._results.append(component.is_valid)
                if not component.is_valid:
                    self._invalid_messages.append(component.invalid_messages)
            self._is_valid = all(self._results)

    # ----------------------------------------------------------------------- #
    #                              Print                                      #
    # ----------------------------------------------------------------------- #
    def print_validator(self):
        text = "\nThe {objectname} {objecttype} for the {parentname} property \
            {targetname}:".format(
                objectname = self._object_name,
                objecttype=self._object_type,
                parentname=self._parent_name,
                targetname=self._target_name
            )        
        print(text)        
        if self._components:
            for idx, component in enumerate(self._components):
                component.compile()
                component.print_component()
                self._components[idx] = component        


# --------------------------------------------------------------------------- #
#                             StringValidator                                 #
# --------------------------------------------------------------------------- #                
class StringValidator(BaseValidator):
    """Validates an optional string property."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(StringValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

# --------------------------------------------------------------------------- #
#                            BooleanValidator                                 #
# --------------------------------------------------------------------------- #                
class BooleanValidator(BaseValidator):
    """Ensures an attribute contains a Boolean value."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(BooleanValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)
            
# --------------------------------------------------------------------------- #
#                            AllowedValuesValidator                           #
# --------------------------------------------------------------------------- #                
class AllowedValuesValidator(BaseValidator):
    """Ensures an attribute contains one of a set of allowed values."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(AllowedValuesValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)
    
# --------------------------------------------------------------------------- #
#                          ForbiddenValuesValidator                           #
# --------------------------------------------------------------------------- #                
class ForbiddenValuesValidator(BaseValidator):
    """Ensures an attribute contains none of a set of forbidden values."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(ForbiddenValuesValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

# --------------------------------------------------------------------------- #
#                            NumberValidator                                  #
# --------------------------------------------------------------------------- #                
class NumberValidator(BaseValidator):
    """Ensures an attribute contains a number."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(NumberValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

# --------------------------------------------------------------------------- #
#                            IntegerValidator                                 #
# --------------------------------------------------------------------------- #                
class IntegerValidator(BaseValidator):
    """Ensures an attribute contains a number."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(IntegerValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)             

# --------------------------------------------------------------------------- #
#                             ArrayValidator                                  #
# --------------------------------------------------------------------------- #                
class ArrayValidator(BaseValidator):
    """Ensures an attribute contains a number."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(ArrayValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)
# --------------------------------------------------------------------------- #
#                            RegexValidator                                   #
# --------------------------------------------------------------------------- #                
class RegexValidator(BaseValidator):
    """Ensures an attribute value matches a regex pattern."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Attribute Validator", **kwargs):
        super(RegexValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)  

