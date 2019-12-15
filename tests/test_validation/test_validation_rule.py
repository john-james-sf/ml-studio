#!/usr/bin/env python3
# =========================================================================== #
#                          TEST VALIDATION RULE                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_validation_rule.py                                              #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 14th 2019, 5:11:35 am                        #
# Last Modified: Saturday December 14th 2019, 5:13:21 am                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Test validation rule."""
#%%
import os
import shutil
import pytest
from pytest import mark
import numpy as np

from ml_studio.services.validation import ValidationRule
from ml_studio.entities.classes import Classes
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import LassoRegression
# --------------------------------------------------------------------------- #
#%%

class ValidationRuleTests:

    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_init
    def test_validation_rule_init(self, get_classes):
        c = get_classes
        print(c.get_classnames())
        classname = 'LinearRegression'
        classnames = c.get_classnames()
        instance = LinearRegression()
        attribute = 'learning_rate'
        r = ValidationRule(instance, attribute)
        assert r.instance == instance, "instance not initialized"
        assert r.attribute == attribute, "attribute not initialized"
        assert r.requirements == [], "requirements not initialized"
        assert r.dependencies == [], "dependencies not initialized"
        assert r.classname == 'LinearRegression', "classname not initialized"
        assert r.attribute_type == 'float', "attribute type not initialized"        
        assert r.classnames == classnames, "classes not initialized"
        assert r.classname == classname, "classname not initialized"                
        assert r.attribute == 'learning_rate', "learning rate attribute not posted"
        # Test invalid class at instantiation
        instance = int()
        attribute = 'freaky'
        with pytest.raises(KeyError):
            r = ValidationRule(instance, attribute)
        # Test invalid attribute at instantiation              
        instance = LinearRegression()
        attribute = 'costx'  
        with pytest.raises(KeyError):
            r = ValidationRule(instance, attribute)

    # Test allowed
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_allowed
    def test_validation_rule_add_allowed(self, get_classes):
        # Obtain classes
        c = get_classes
        # Instantiate 
        instance = LinearRegression()
        attribute = 'cost'
        r = ValidationRule(instance,attribute)
        # Test value that doesn't match type of attribute
        allowed = [1,2,3]                
        requirement = {'allowed' : allowed}
        with pytest.raises(TypeError):            
            r.add_requirement(requirement)
        # Test valid requirement
        allowed = ['log', 'quadratic', 'entropy']        
        requirement1 = {'allowed' : allowed}
        r.add_requirement(requirement1)
        requirements1 = r.requirements
        assert requirement1 == requirements1[0], "Added allowed requirement. Not valid "
        # Add 2nd requirement
        allowed = ['binary', 'multinomial', 'cross_entropy']                
        requirement2 = {'allowed' : allowed}
        r.add_requirement(requirement2)
        requirements2 = r.requirements
        assert requirements2[0] == requirement1, "requirement values not updated "
        assert requirements2[1] == requirement2, "requirement values not updated "
        assert len(requirements2) == 2, "Attempt to add 2nd allowed requirement failed. "        

    # Test forbidden
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_forbidden
    def test_validation_rule_add_forbidden(self, get_classes):
        # Obtain classes
        c = get_classes
        # Instantiate 
        instance = LinearRegression()
        attribute = 'metric'
        r = ValidationRule(instance,attribute)
        # Test value that doesn't match type of attribute
        forbidden = [1,2,3]                
        requirement = {'forbidden' : forbidden}
        with pytest.raises(TypeError):            
            r.add_requirement(requirement)
        # Test valid requirement
        forbidden = ['r2', 'mse', 'lmse']        
        requirement1 = {'forbidden' : forbidden}
        r.add_requirement(requirement1)
        requirements1 = r.requirements
        assert requirement1 == requirements1[0], "Added allowed requirement. Not valid "
        # Add 2nd requirement
        forbidden = ['accuracy', 'f1', 'nlmse']                
        requirement2 = {'forbidden' : forbidden}
        r.add_requirement(requirement2)
        requirements2 = r.requirements
        assert requirements2[0] == requirement1, "requirement values not updated "
        assert requirements2[1] == requirement2, "requirement values not updated "
        assert len(requirements2) == 2, "Attempt to add 2nd allowed requirement failed. "        

    # Test contains
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_contains
    def test_validation_rule_add_contains(self, get_classes):
        # Obtain classes
        c = get_classes
        # Instantiate 
        instance = LinearRegression()
        attribute = 'metric'
        r = ValidationRule(instance,attribute)
        # Test value that doesn't match type of attribute
        contains = [1,2,3]                
        requirement = {'contains' : contains}
        with pytest.raises(TypeError):            
            r.add_requirement(requirement)
        # Test valid requirement
        contains = ['r2', 'mse', 'lmse']        
        requirement1 = {'contains' : contains}
        r.add_requirement(requirement1)
        requirements1 = r.requirements
        assert requirement1 == requirements1[0], "Added allowed requirement. Not valid "
        # Add 2nd requirement
        contains = ['accuracy', 'f1', 'nlmse']                
        requirement2 = {'contains' : contains}
        r.add_requirement(requirement2)
        requirements2 = r.requirements
        assert requirements2[0] == requirement1, "requirement values not updated "
        assert requirements2[1] == requirement2, "requirement values not updated "
        assert len(requirements2) == 2, "Attempt to add 2nd allowed requirement failed. "        

    # Test excludes
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_excludes
    def test_validation_rule_add_excludes(self, get_classes):
        # Obtain classes
        c = get_classes
        # Instantiate 
        instance = LinearRegression()
        attribute = 'metric'
        r = ValidationRule(instance,attribute)
        # Test value that doesn't match type of attribute
        excludes = [1,2,3]                
        requirement = {'excludes' : excludes}
        with pytest.raises(ValueError):            
            r.add_requirement(requirement)
        # Test valid requirement
        excludes = ['learning_rate', 'verbose', 'seed']        
        requirement1 = {'excludes' : excludes}
        r.add_requirement(requirement1)
        requirements1 = r.requirements
        assert requirement1 == requirements1[0], "Added allowed requirement. Not valid "
        # Add 2nd requirement
        excludes = ['theta_init', 'early_stop', 'val_size']                
        requirement2 = {'excludes' : excludes}
        r.add_requirement(requirement2)
        requirements2 = r.requirements
        assert requirements2[0] == requirement1, "requirement values not updated "
        assert requirements2[1] == requirement2, "requirement values not updated "
        assert len(requirements2) == 2, "Attempt to add 2nd allowed requirement failed. "        

    # Test min max
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_min_max
    def test_validation_rule_add_min_max(self, get_classes):
        # Obtain classes
        c = get_classes
        # Instantiate 
        instance = LinearRegression()
        attribute = 'early_stop'
        r = ValidationRule(instance,attribute)
        # Test value that doesn't match type of attribute
        requirement1 = {'min' : 5}
        with pytest.raises(TypeError):            
            r.add_requirement(requirement1)
        # Test valid min
        attribute = 'seed'
        r = ValidationRule(instance,attribute)        
        r.add_requirement(requirement1)        
        assert requirement1 == r.requirements[0], "Added allowed requirement. Not valid "
        # Test valid max        
        requirement2 = {'max': 20}        
        r.add_requirement(requirement2)        
        requirements = r.requirements
        assert requirements[0] == requirement1, "requirement values not updated "
        assert requirements[1] == requirement2, "requirement values not updated "
        assert len(requirements) == 2, "Attempt to add 2nd allowed requirement failed. "          

    # Test minlength maxlength
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_minlength_maxlength
    def test_validation_rule_add_min_max_length(self, get_classes):
        # Create LinearRegression class with a list attribute
        lr = LinearRegression()
        lr.some_list = ['speakers', 'rekkids', 'turntables']        
        # Obtain classes and add updated Linear Regression class
        c = get_classes
        c.add_class(lr)
        # Instantiate 
        instance = lr        
        attribute = 'early_stop'
        r = ValidationRule(instance,attribute)
        # Test value that doesn't match type of attribute
        requirement1 = {'minlength' : 5}
        with pytest.raises(ValueError):            
            r.add_requirement(requirement1)
        # Test valid minlength
        attribute = 'some_list'
        r = ValidationRule(instance,attribute)        
        r.add_requirement(requirement1)        
        assert requirement1 == r.requirements[0], "Added allowed requirement. Not valid "
        # Test valid maxlength        
        requirement2 = {'maxlength': 20}        
        r.add_requirement(requirement2)        
        requirements = r.requirements
        assert requirements[0] == requirement1, "requirement values not updated "
        assert requirements[1] == requirement2, "requirement values not updated "
        assert len(requirements) == 2, "Attempt to add 2nd allowed requirement failed. "       

    # Test required
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_required
    def test_validation_rule_add_required(self, get_classes):
        # Obtain classes and add updated Linear Regression class
        c = get_classes        
        # Instantiate 
        instance = LassoRegression()        
        attribute = 'val_size'
        r = ValidationRule(instance,attribute)
        # Add required validation attribute
        requirement = {'required' : 'Hat'}        
        with pytest.raises(TypeError):
            r.add_requirement(requirement)  
        requirement = {'required' : True}        
        r.add_requirement(requirement)                
        assert requirement == r.requirements[0], "Added allowed requirement. Not valid "        
        assert len(requirement) == 1, "Attempt to add 2nd allowed requirement failed. "           

    # Test regex
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_regex
    def test_validation_rule_add_regex(self, get_classes):
        # Obtain classes and add updated Linear Regression class
        c = get_classes        
        # Instantiate 
        instance = LassoRegression()        
        attribute = 'metric'
        r = ValidationRule(instance,attribute)
        # Add required validation attribute
        requirement = {'regex' : 99}        
        with pytest.raises(ValueError):
            r.add_requirement(requirement)  
        requirement = {'regex' : 'ab*'}        
        r.add_requirement(requirement)                
        assert requirement == r.requirements[0], "Added allowed requirement. Not valid "        
        assert len(requirement) == 1, "Attempt to add 2nd allowed requirement failed. "       

    # Test type
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_add_requirement
    @mark.validation_rule_add_requirement_type
    def test_validation_rule_add_type(self, get_classes):
        # Obtain classes and add updated Linear Regression class
        c = get_classes        
        # Instantiate 
        instance = LassoRegression()        
        attribute = 'metric'
        r = ValidationRule(instance,attribute)
        # Add invalid type validation attribute
        requirement = {'type' : 'xxx'}        
        with pytest.raises(ValueError):
            r.add_requirement(requirement)  
        # Add valid type
        attribute = 'early_stop'
        r = ValidationRule(instance,attribute)
        requirement = {'type' : bool}        
        r.add_requirement(requirement)                
        assert requirement == r.requirements[0], "Added allowed requirement. Not valid "        
        assert len(requirement) == 1, "Attempt to add 2nd allowed requirement failed. "        

    # Remove requirement
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_remove
    @mark.validation_rule_remove_requirement
    def test_validation_rule_remove_requirement(self, get_classes):
        # Remove full requirement
        c = get_classes                
        instance = LassoRegression()                
        attribute = 'early_stop'
        r = ValidationRule(instance,attribute)
        requirement = {'type' : bool}        
        r.add_requirement(requirement)  
        assert r.requirements, "Requirement does not exist"    
        assert r.requirements[0] == requirement, "Requirement does not match input requirement"
        r.remove_requirement(requirement)
        assert r.requirements == [], "Remove requirement didn't work"
        # Remove requirement based upon key
        requirement = {'type' : bool}        
        r.add_requirement(requirement)  
        requirement = 'type'
        r.remove_requirement(requirement)
        assert r.requirements == [], "Remove requirement didn't work"
        requirement = {'type' : bool}        
        r.add_requirement(requirement)  
        r.remove_requirement()
        assert r.requirements == [], "Remove requirement didn't work"
         
    # Remove dependency
    @mark.validation
    @mark.validation_rule
    @mark.validation_rule_dependency
    def test_validation_rule_dependency(self, get_classes):
        # Remove full requirement
        c = get_classes                
        instance = LassoRegression()                
        attribute = 'early_stop'
        r = ValidationRule(instance,attribute)
        requirement = {'type' : bool}        
        dependency = {'cost' : requirement}
        r.add_dependency(dependency)  
        assert r.dependencies, "Dependency does not exist"    
        assert r.dependencies[0] == dependency, "Dependency does not match input dependency"
        r.remove_dependency(dependency)
        assert r.dependencies == [], "Remove dependency didn't work"
        # Remove dependency based upon key
        requirement = {'type' : bool}     
        dependency = {'cost' : requirement}           
        r.add_dependency(dependency)  
        dependency = 'cost'
        r.remove_dependency(dependency)
        assert r.dependencies == [], "Remove dependency didn't work"
