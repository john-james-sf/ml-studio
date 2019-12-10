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

Validation is accomplished using classes that can be characterized as either
entity, configuration, object validation and attribute validation
classes. 

The entity classes include:

* Project : The top of the class hierarchy. Contains data, analysis, model,
and pipeline related objects.
* DataPackage : A container of datasets associated with a project. This
would typically include raw, staged, cleaned, and processed data.
* DateSet : An individual data source, uniquely identified by a name, 
an identifier, and a stage.
* Analysis : A set of processes and the outcome of data analysis efforts.
* Model : A machine learning or deep learning model with parameters, data,
training results and performance metrics.
* Pipeline : Set of processes executed upon data and model objects to achieve
a project objective.

The configuration classes include:  
 
 * Context : This class captures the relationship between entities and 
 and their contexts that define validation behavior.
 * Attribute : Defines the attributes of the entity classes, specifying
 it, name, type and validator.
 * Rule : Manages the relationship between entities, contexts, attributes
 and requirements. 

An object validation class is defined for each entity.  Attribute validation
classes execute validation logic for an individual entity.
    
"""
import os
import time
import math

from ml_studio.entities.package import DataPackage
from ml_studio.entities.dataset import DataSet
from ml_studio.entities.dataset import DataSet

# --------------------------------------------------------------------------- #
#                                VALIDATOR                                    #
# --------------------------------------------------------------------------- #

class Validator():
    """Abstract base class from which all Validator classes inherit.

    This defines the interface required by entity and attribute validation 
    classes.

    Attributes
    ----------
    name : str
        The name for the validator in lower snake case.

    attributes : str, array-like 
        The name or names of attributes to validated by this validator. 

    except : str, array-like
        The name or names of contexts for which validation should not be applied

    message : str
        Message to be displayed to user when validation requirements are 
        not met.

    contexts : str, array-like
        Contexts to which the validator can be applied

    ignore_empty : Bool
        Whether the validation rule should be ignored if attribute is empty.

    ignore_if_error : Bool
        Whether to ignore if previous validation error has been encountered on
        another attribute.

    """

    def __init__(self, name=None):
        """Instantiates the validation class."""
        self.name = name
        self.data_package = 

    def confirm_attributes(self):

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        new_name = snake(name)
        if name != new_name:
            warning.warn("The name parameter must be lower case, alphanumeric "
                         "only with the exception of underscores and hyphens. "
                         "The name parameter was changed from %s to %s" 
                         % (name, new_name))
            self._name = new_name

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        # Instantiate ClassManager
        # Iterate through 
        self._attributes = attributes

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = attributes


    