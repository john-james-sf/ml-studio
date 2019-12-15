#!/usr/bin/env python3
# =========================================================================== #
#                              TEST CLASSES                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_classes.py                                                      #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 14th 2019, 1:08:05 am                        #
# Last Modified: Saturday December 14th 2019, 1:08:27 am                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Tests the classes class."""
#%%
import pytest
from pytest import mark

from ml_studio.entities.classes import Classes
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.visualate.regression.validity import Residuals, ScaleLocation
from ml_studio.visualate.regression.validity import StandardizedResiduals
from ml_studio.visualate.regression.validity import StudentizedResiduals
from ml_studio.services.validation import ValidationContexts
# --------------------------------------------------------------------------- #
#%%

class ClassesTests:

    @mark.classes
    @mark.classes_init
    def test_classes_init(self):
        a = Classes()
        b = Classes()
        assert a == b, "Singleton didn't work"

    @mark.classes
    @mark.classes_get_classnames
    def test_classes_get_classnames(self):
        model = LinearRegression()
        classes = [LinearRegression(),LassoRegression(), Residuals(model),StudentizedResiduals(model)]
        clsnames = ['LinearRegression','LassoRegression', 'Residuals','StudentizedResiduals']
        c = Classes()
        for cls in classes:            
            c.add_class(cls)
        classnames = c.get_classnames()
        assert len(classnames) == 4, "Get classnames didn;t return list of correct length."
        assert all(item in clsnames for item in classnames), "Get classnames didn't return correct result."
    

    @mark.classes
    @mark.classes_add_class
    def test_classes_add_class(self):
        model = LinearRegression()
        instance = Residuals(model=model)
        classname = instance.__class__.__name__
        c = Classes()
        c.add_class(instance)
        assert c.classes[classname] == instance, "Adding single class didn't work."
        
    @mark.classes
    @mark.classes_remove_class
    def test_classes_remove_class(self):
        model = LinearRegression()
        instance = ScaleLocation(model=model)
        classname = instance.__class__.__name__
        c = Classes()
        c.add_class(instance)
        c.remove_class(classname)
        with pytest.raises(KeyError): 
            c.classes[classname] 

    @mark.classes
    @mark.classes_instance_attributes
    def test_classes_instance_attributes(self):
        instance = LinearRegression()        
        classname = instance.__class__.__name__
        c = Classes()
        c.add_class(instance)
        attrs = c.get_instance_attributes(classname)
        assert isinstance(attrs, list), "Instance attributes not a list"
        assert len(attrs) > 0, "There are no instance attributes."

    @mark.classes
    @mark.classes_class_attributes
    def test_classes_class_attributes(self):        
        instance = ValidationContexts()
        classname = instance.__class__.__name__
        c = Classes()
        c.add_class(instance)
        attrs = c.get_class_attributes(classname)
        assert isinstance(attrs, list), "Instance attributes not a list"
        assert len(attrs) > 0, "There are no instance attributes."        