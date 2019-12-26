#!/usr/bin/env python3
# =========================================================================== #
#                            SERVICES: CLASSES                                #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \classes.py                                                           #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 14th 2019, 12:42:38 am                       #
# Last Modified: Saturday December 14th 2019, 12:42:55 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module holding ML Studio class definitions and convenience methods.

Each ML Studio class is contained, indexed by its class name. Methods are
included to reveal class attributes and instance variables.
    
"""
#%%
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#                                  CLASSES                                    #
# --------------------------------------------------------------------------- #
"""Instances of ML Studio classes and methods to expose class attributes.

This singleton class exposes the class and instance variables for the class.

"""

class Classes:

    classes = {}
    __instance = None
    
    def __new__(cls):
        if Classes.__instance is None:
            Classes.__instance = object.__new__(cls)        
        return Classes.__instance    

    def get_classnames(self):
        """Returns list of class names in the class."""
        classnames = []
        for k in self.classes.keys():
            classnames.append(k)
        return classnames

    def search_classnames(self, classname):
        """Searches by classname and returns True if found, False otherwise."""
        classnames = self.get_classnames()
        if classname in classnames:
            return True
        else:
            return False


    def add_class(self, instance):
        """Add class to Classes.

        Parameters
        ----------
        instance : Instance of an ML Studio class.
            An instance of the class to add to the inventory of classes

        """
        classname = instance.__class__.__name__
        self.classes[classname] = instance

    def remove_class(self, classname):
        """Removes a class from the inventory of classes.

        Parameters
        ----------
        classname : str. Required
            The class name to remove from the inventory of classes

        """
        try:
            del self.classes[classname]
        except KeyError:
            print("%s does not exist." % classname)

    def get_instance_attributes(self, classname, values=False):
        """Return a list of instance attributes.
        
        Parameters
        ----------
        classname : str. Required
            The class name for which the instance attributes are to be shown

        values : bool
            Indicates whether the attribute values should be returned

        """
        instance_attrs = list()

        try:
            instance = self.classes[classname]
        except KeyError:
            raise KeyError("%s class was not found." % classname)
        
        for attribute, value in instance.__dict__.items():   
            if values:     
                d = {}
                d[attribute] = value
                instance_attrs.append(d)
            else:
                instance_attrs.append(attribute)

        return instance_attrs

    def get_class_attributes(self, classname, values=False):
        """Print and return a list of class attributes.
        
        Parameters
        ----------
        classname : str. Required
            The class name for which the instance attributes are to be shown

        values : bool
            Indicates whether the attribute values should be returned            

        """
        class_attrs = list()

        try:
            instance = self.classes[classname]
        except KeyError:
            raise KeyError("%s class was not found." % classname)        
        
        for attribute in instance.__class__.__dict__.keys():
            if attribute[:1] != "_":
                value = getattr(instance, attribute)
                if not callable(value):        
                    if values:            
                        d = {}
                        d[attribute] = value
                        class_attrs.append(d)
                    else:
                        class_attrs.append(attribute)

        return class_attrs