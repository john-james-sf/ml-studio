# =========================================================================== #
#                              TEST CONDITIONS                                #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_conditions.py                                                   #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 21st 2019, 1:15:37 am                        #
# Last Modified: Tuesday December 24th 2019, 2:34:40 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Tests conditions.

Validation conditions must be met before a validation rule can be applied.
There are two types of conditions, syntactic and semantic. Syntactic 
conditions take a single parameter and evaluates the type and state. 
Semantic conditions compares a value or the property of an instance,
against another value or property of an instance.

    Syntactic Conditions
    --------------------
    * isNone : Evaluates whether the argument is None.
    * isNotNone : Evaluates whether the argument is not None.
    * isEmpty : Evaluates whether the argument is empty string or whitespace.
    * isNotEmpty : Evaluates whether the argument is not empty string or whitespace.
    * isBool : Evaluates whether the argument is a Boolean.
    * isInt : Evaluates whether the argument is an integer.
    * isFloat : Evaluates whether the argument is an float.
    * isNumber : Evaluates whether the argument is a number.
    * isString : Evaluates whether the argument is a string. 

    Semantic Conditions
    -------------------
    * IsEqual : Evaluates whether two arguments are equal  
    * IsNotEqual : Evaluates whether two arguments are not equal  
    * IsIn : Evaluates whether a is in b.
    * IsLess : Evaluates whether a < b.
    * IsGreater : Evaluates whether a > b.
    * IsBetween : Evaluates whether argument a is between min and max. 
    * IsMatch : Evaluates whether a string matches a regex pattern.    


"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.services.validation.conditions import IsNone, IsNotNone
from ml_studio.services.validation.conditions import IsEmpty, IsNotEmpty
from ml_studio.services.validation.conditions import IsBool, IsInt
from ml_studio.services.validation.conditions import IsFloat, IsNumber, IsString
from ml_studio.services.validation.conditions import IsEqual, IsNotEqual
from ml_studio.services.validation.conditions import IsIn, IsNotIn
from ml_studio.services.validation.conditions import IsLess, IsGreater
from ml_studio.services.validation.conditions import IsBetween, IsMatch

class SyntacticConditionTests:

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isNone
    def test_syntactic_conditions_isNone(self, get_validation_rule_test_object):
        a = 5
        assert IsNone(a)() == False, "isNone incorrect evaluation."
        a = None
        assert IsNone(a)() == True, "isNone incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsNone(a)() == True, "isNone incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsNone(a)() == False, "isNone incorrect evaluation nested list."        
        a = [[False, None], [None, None]]
        assert IsNone(a)() == False, "isNone incorrect evaluation nested list."  
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='n')
        assert IsNone(a)() == True, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='s')
        assert IsNone(a)() == False, "isNone incorrect evaluation nested list."  

        
    @mark.syntactic_conditions
    @mark.syntactic_conditions_isNotNone
    def test_syntactic_conditions_isNotNone(self, get_validation_rule_test_object):
        a = 5
        assert IsNotNone(a)() == True, "IsNotNone incorrect evaluation."
        a = None
        assert IsNotNone(a)() == False, "IsNotNone incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsNotNone(a)() == False, "IsNotNone incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsNotNone(a)() == False, "IsNotNone incorrect evaluation nested list."           
        a = [[False, None], [None, None]]
        assert IsNotNone(a)() == False, "IsNotNone incorrect evaluation nested list."                   
        a = [[False, False], [False, False]]
        assert IsNotNone(a)() == True, "IsNotNone incorrect evaluation nested list."                   
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='n')
        assert IsNotNone(a)() == False, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='s')
        assert IsNotNone(a)() == True, "isNone incorrect evaluation nested list."  


    @mark.syntactic_conditions
    @mark.syntactic_conditions_isEmpty
    def test_syntactic_conditions_isEmpty(self, get_validation_rule_test_object):
        a = 5
        assert IsEmpty(a)() == False, "isEmpty incorrect evaluation."
        a = None
        assert IsEmpty(a)() == True, "isEmpty incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsEmpty(a)() == True, "isEmpty incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsEmpty(a)() == False, "isEmpty incorrect evaluation nested list."  
        a = [[False, None], [None, None]]
        assert IsEmpty(a)() == False, "isEmpty incorrect evaluation nested list."                
        a = ""
        assert IsEmpty(a)() == True, "isEmpty incorrect evaluation."
        a = ["", None, " "]
        assert IsEmpty(a)() == True, "isEmpty incorrect evaluation."
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='e')
        assert IsEmpty(a)() == True, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='s')
        assert IsEmpty(a)() == False, "isNone incorrect evaluation nested list."  


    @mark.syntactic_conditions
    @mark.syntactic_conditions_isNotEmpty
    def test_syntactic_conditions_isNotEmpty(self, get_validation_rule_test_object):
        a = 5
        assert IsNotEmpty(a)() == True, "isNotEmpty incorrect evaluation."
        a = None
        assert IsNotEmpty(a)() == False, "isNotEmpty incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsNotEmpty(a)() == False, "isNotEmpty incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsNotEmpty(a)() == True, "isNotEmpty incorrect evaluation nested list."        
        a = ""
        assert IsNotEmpty(a)() == False, "isNotEmpty incorrect evaluation."
        a = ["", None, " "]
        assert IsNotEmpty(a)() == False, "isNotEmpty incorrect evaluation."
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='e')
        assert IsNotEmpty(a)() == False, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='s')
        assert IsNotEmpty(a)() == True, "isNone incorrect evaluation nested list."  


    @mark.syntactic_conditions
    @mark.syntactic_conditions_isBool
    def test_syntactic_conditions_isBool(self, get_validation_rule_test_object):
        a = 5
        assert IsBool(a)() == False, "isBool incorrect evaluation."
        a = None
        assert IsBool(a)() == False, "isBool incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsBool(a)() == False, "isBool incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsBool(a)() == False, "isBool incorrect evaluation nested list."  
        a = [[False, None], [None, None]]
        assert IsBool(a)() == False, "isBool incorrect evaluation nested list."                
        a = [[False, True], [False, True]]
        assert IsBool(a)() == True, "isBool incorrect evaluation nested list."                        
        a = True
        assert IsBool(a)() == True, "isBool incorrect evaluation."
        a = ["", None, " "]
        assert IsBool(a)() == False, "isBool incorrect evaluation."    
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='b')
        assert IsBool(a)() == True, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='s')
        assert IsBool(a)() == False, "isNone incorrect evaluation nested list."  


    @mark.syntactic_conditions
    @mark.syntactic_conditions_isInt
    def test_syntactic_conditions_isInt(self, get_validation_rule_test_object):
        a = 5
        assert IsInt(a)() == True, "isInt incorrect evaluation."
        a = None
        assert IsInt(a)() == False, "isInt incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsInt(a)() == False, "isInt incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsInt(a)() == False, "isInt incorrect evaluation nested list."  
        a = [[False, 3], [4, 4]]
        assert IsInt(a)() == False, "isInt incorrect evaluation nested list."                
        a = [[2, 5], [3, 2]]
        assert IsInt(a)() == True, "isInt incorrect evaluation nested list."                        
        a = True
        assert IsInt(a)() == False, "isInt incorrect evaluation."
        a = [2.0, 3.5]
        assert IsInt(a)() == False, "isInt incorrect evaluation."   
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='i')
        assert IsInt(a)() == True, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='s')
        assert IsInt(a)() == False, "isNone incorrect evaluation nested list."  


    @mark.syntactic_conditions
    @mark.syntactic_conditions_isFloat
    def test_syntactic_conditions_isFloat(self, get_validation_rule_test_object):
        a = 5.0
        assert IsFloat(a)() == True, "isFloat incorrect evaluation."
        a = None
        assert IsFloat(a)() == False, "isFloat incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsFloat(a)() == False, "isFloat incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsFloat(a)() == False, "isFloat incorrect evaluation nested list."  
        a = [[False, 3], [4, 4]]
        assert IsFloat(a)() == False, "isFloat incorrect evaluation nested list."                
        a = [[2.2, 5.3], [3.2, 2.0]]
        assert IsFloat(a)() == True, "isFloat incorrect evaluation nested list."                        
        a = True
        assert IsFloat(a)() == False, "isFloat incorrect evaluation."
        a = [2, 3]
        assert IsFloat(a)() == False, "isFloat incorrect evaluation."          
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='f')
        assert IsFloat(a)() == True, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='i')
        assert IsFloat(a)() == False, "isNone incorrect evaluation nested list."  


    @mark.syntactic_conditions
    @mark.syntactic_conditions_isNumber
    def test_syntactic_conditions_isNumber(self, get_validation_rule_test_object):
        a = 5.0
        assert IsNumber(a)() == True, "isNumber incorrect evaluation."
        a = None
        assert IsNumber(a)() == False, "isNumber incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsNumber(a)() == False, "isNumber incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsNumber(a)() == False, "isNumber incorrect evaluation nested list."  
        a = [[False, 3], [4, 4]]
        assert IsNumber(a)() == False, "isNumber incorrect evaluation nested list."                
        a = [[2.2, 5.3], [3.2, 2.0]]
        assert IsNumber(a)() == True, "isNumber incorrect evaluation nested list."                        
        a = True
        assert IsNumber(a)() == False, "isNumber incorrect evaluation."
        a = [2, 3]
        assert IsNumber(a)() == True, "isNumber incorrect evaluation."       
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='f')
        assert IsNumber(a)() == True, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='i')
        assert IsNumber(a)() == True, "isNone incorrect evaluation nested list."          
        a = dict(instance=test_object, attribute_name='s')
        assert IsNumber(a)() == False, "isNone incorrect evaluation nested list."  

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isString
    def test_syntactic_conditions_isString(self, get_validation_rule_test_object):
        a = '5.0'
        assert IsString(a)() == True, "isString incorrect evaluation."
        a = None
        assert IsString(a)() == False, "isString incorrect evaluation."
        a = [[None, None], [None, None]]
        assert IsString(a)() == False, "isString incorrect evaluation nested list."
        a = [[None, None], [None, False]]
        assert IsString(a)() == False, "isString incorrect evaluation nested list."  
        a = [[False, 3], [4, 4]]
        assert IsString(a)() == False, "isString incorrect evaluation nested list."                
        a = [['2.2', '5.3'], ['3.2', '2.0']]
        assert IsString(a)() == True, "isString incorrect evaluation nested list."                        
        a = True
        assert IsString(a)() == False, "isString incorrect evaluation."
        a = ['2', '3']
        assert IsString(a)() == True, "isString incorrect evaluation."       
        test_object = get_validation_rule_test_object
        a = dict(instance=test_object, attribute_name='s')
        assert IsString(a)() == True, "isNone incorrect evaluation nested list."  
        a = dict(instance=test_object, attribute_name='i')
        assert IsString(a)() == False, "isNone incorrect evaluation nested list."          

class SemanticConditionTests:

    @mark.semantic_conditions
    @mark.semantic_conditions_IsEqual
    def test_semantic_conditions_IsEqual(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = 5
        b = 10
        assert IsEqual(a,b)() == False, "IsEqual incorrect evaluation."
        a = 5
        b = dict(instance=test_object, attribute_name='i')        
        assert IsEqual(a,b)() == True, "IsEqual incorrect evaluation."
        a = None
        b = dict(instance=test_object, attribute_name='n')        
        assert IsEqual(a,b)() == True, "IsEqual incorrect evaluation."
        a = 2.0
        b = dict(instance=test_object, attribute_name='f')        
        assert IsEqual(a,b)() == True, "IsEqual incorrect evaluation."        
        a = 'hats'
        b = dict(instance=test_object, attribute_name='s')        
        assert IsEqual(a,b)() == True, "IsEqual incorrect evaluation."    
        a = dict(instance=ref_object, attribute_name='s')        
        b = dict(instance=test_object, attribute_name='s')        
        assert IsEqual(a,b)() == True, "IsEqual incorrect evaluation."        
        a = dict(instance=ref_object, attribute_name='i')        
        b = dict(instance=test_object, attribute_name='i')        
        assert IsEqual(a,b)() == False, "IsEqual incorrect evaluation."    
        a = dict(instance=ref_object, attribute_name='a_b')        
        b = dict(instance=test_object, attribute_name='a_b')        
        assert IsEqual(a,b)() == True, "IsEqual incorrect evaluation."                              
        a = dict(instance=ref_object, attribute_name='a_i')        
        b = dict(instance=test_object, attribute_name='a_i')        
        assert IsEqual(a,b)() == False, "IsEqual incorrect evaluation."                                      
        a = dict(instance=ref_object, attribute_name='na_e')        
        b = dict(instance=test_object, attribute_name='na_e')        
        assert IsEqual(a,b)() == True, "IsEqual incorrect evaluation."                                              
        a = dict(instance=ref_object, attribute_name='na_ne')        
        b = dict(instance=test_object, attribute_name='na_ne')        
        assert IsEqual(a,b)() == False, "IsEqual incorrect evaluation."                                              
        
    @mark.semantic_conditions
    @mark.semantic_conditions_IsNotEqual
    def test_semantic_conditions_IsNotEqual(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = 5
        b = 10
        assert IsNotEqual(a,b)() == True, "IsNotEqual incorrect evaluation."
        a = 5
        b = dict(instance=test_object, attribute_name='i')        
        assert IsNotEqual(a,b)() == False, "IsNotEqual incorrect evaluation."
        a = None
        b = dict(instance=test_object, attribute_name='n')        
        assert IsNotEqual(a,b)() == False, "IsNotEqual incorrect evaluation."
        a = 2.0
        b = dict(instance=test_object, attribute_name='f')        
        assert IsNotEqual(a,b)() == False, "IsNotEqual incorrect evaluation."        
        a = 'hats'
        b = dict(instance=test_object, attribute_name='s')        
        assert IsNotEqual(a,b)() == False, "IsNotEqual incorrect evaluation."    
        a = dict(instance=ref_object, attribute_name='s')        
        b = dict(instance=test_object, attribute_name='s')        
        assert IsNotEqual(a,b)() == False, "IsNotEqual incorrect evaluation."        
        a = dict(instance=ref_object, attribute_name='i')        
        b = dict(instance=test_object, attribute_name='i')        
        assert IsNotEqual(a,b)() == True, "IsNotEqual incorrect evaluation."    
        a = dict(instance=ref_object, attribute_name='a_b')        
        b = dict(instance=test_object, attribute_name='a_b')        
        assert IsNotEqual(a,b)() == False, "IsNotEqual incorrect evaluation."                              
        a = dict(instance=ref_object, attribute_name='a_i')        
        b = dict(instance=test_object, attribute_name='a_i')        
        assert IsNotEqual(a,b)() == True, "IsNotEqual incorrect evaluation."                                      
        a = dict(instance=ref_object, attribute_name='na_e')        
        b = dict(instance=test_object, attribute_name='na_e')        
        assert IsNotEqual(a,b)() == False, "IsNotEqual incorrect evaluation."                                              
        a = dict(instance=ref_object, attribute_name='na_ne')        
        b = dict(instance=test_object, attribute_name='na_ne')        
        assert IsNotEqual(a,b)() == True, "IsNotEqual incorrect evaluation."           

    @mark.semantic_conditions
    @mark.semantic_conditions_IsIn
    def test_semantic_conditions_IsIn(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = 5
        b = 10
        assert IsIn(a,b)() == False, "IsIn incorrect evaluation."
        a = 5
        b = dict(instance=test_object, attribute_name='i')        
        assert IsIn(a,b)() == True, "IsIn incorrect evaluation."
        a = None
        b = dict(instance=test_object, attribute_name='n')        
        assert IsIn(a,b)() == True, "IsIn incorrect evaluation."
        a = 2.0
        b = dict(instance=test_object, attribute_name='f')        
        assert IsIn(a,b)() == True, "IsIn incorrect evaluation."        
        a = 'ha'
        b = dict(instance=test_object, attribute_name='s')        
        assert IsIn(a,b)() == True, "IsIn incorrect evaluation."    
        a = dict(instance=ref_object, attribute_name='s')        
        b = ["sets", "djs", "lights"]
        assert IsIn(a,b)() == False, "IsIn incorrect evaluation."        
        a = dict(instance=test_object, attribute_name='i')        
        b = dict(instance=ref_object, attribute_name='a_i')        
        assert IsIn(a,b)() == True, "IsIn incorrect evaluation."    
        a = dict(instance=test_object, attribute_name='a_i')        
        b = dict(instance=ref_object, attribute_name='a_i')        
        assert IsIn(a,b)() == True, "IsIn incorrect evaluation."                              
        a = dict(instance=test_object, attribute_name='na_e')        
        b = [None, "Hats", 2.0, 3, False, 33,55]
        assert IsIn(a,b)() == True, "IsIn incorrect evaluation."           

    @mark.semantic_conditions
    @mark.semantic_conditions_IsLess
    def test_semantic_conditions_IsLess(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = 5
        b = 10
        assert IsLess(a,b, inclusive=False)() == True, "IsLess incorrect evaluation."
        a = 5
        b = dict(instance=test_object, attribute_name='i')   
        assert IsLess(a,b, inclusive=False)() == False, "IsLess incorrect evaluation."
        with pytest.raises(ValueError):     
            a = dict(instance=test_object, attribute_name='a_i')   
            b = dict(instance=ref_object, attribute_name='a_i')   
            IsLess(a,b, inclusive=False)() 
        a = dict(instance=test_object, attribute_name='a_l')
        b = dict(instance=ref_object, attribute_name='a_l')
        assert IsLess(a,b, inclusive=False)() == True, "Invalid evaluation of IsLess"
        a = dict(instance=test_object, attribute_name='a_le')
        b = dict(instance=ref_object, attribute_name='a_le')
        assert IsLess(a,b, inclusive=False)() == False, "Invalid evaluation of IsLess"
        a = dict(instance=test_object, attribute_name='a_le')
        b = dict(instance=ref_object, attribute_name='a_ge')
        assert IsLess(a,b, inclusive=False)() == False, "Invalid evaluation of IsLess"        


    @mark.semantic_conditions
    @mark.semantic_conditions_IsLessEqual
    def test_semantic_conditions_IsLessEqual(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = 5
        b = 10
        assert IsLess(a,b)() == True, "IsLess incorrect evaluation."
        a = 5
        b = dict(instance=test_object, attribute_name='i')   
        assert IsLess(a,b)() == True, "IsLess incorrect evaluation."
        with pytest.raises(ValueError):     
            a = dict(instance=test_object, attribute_name='a_i')   
            b = dict(instance=ref_object, attribute_name='a_i')   
            IsLess(a,b)() 
        a = dict(instance=test_object, attribute_name='a_l')
        b = dict(instance=ref_object, attribute_name='a_l')
        assert IsLess(a,b)() == True, "Invalid evaluation of IsLess"
        a = dict(instance=test_object, attribute_name='a_le')
        b = dict(instance=ref_object, attribute_name='a_ge')
        assert IsLess(a,b)() == False, "Invalid evaluation of IsLess"        

    @mark.semantic_conditions
    @mark.semantic_conditions_IsGreater
    def test_semantic_conditions_IsGreater(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = 5
        b = 10
        assert IsGreater(a,b, inclusive=False)() == False, "IsGreater incorrect evaluation."
        a = 5
        b = dict(instance=test_object, attribute_name='i')   
        assert IsGreater(a,b, inclusive=False)() == False, "IsGreater incorrect evaluation."
        with pytest.raises(ValueError):     
            a = dict(instance=test_object, attribute_name='a_i')   
            b = dict(instance=ref_object, attribute_name='a_i')   
            IsGreater(a,b, inclusive=False)() 
        a = dict(instance=test_object, attribute_name='a_g')
        b = dict(instance=ref_object, attribute_name='a_g')
        assert IsGreater(a,b, inclusive=False)() == True, "Invalid evaluation of IsGreater"
        a = dict(instance=test_object, attribute_name='a_ge')
        b = dict(instance=ref_object, attribute_name='a_ge')
        assert IsGreater(a,b, inclusive=False)() == False, "Invalid evaluation of IsGreater"
        a = dict(instance=test_object, attribute_name='a_ge')
        b = dict(instance=ref_object, attribute_name='a_le')
        assert IsGreater(a,b, inclusive=False)() == False, "Invalid evaluation of IsGreater"        


    @mark.semantic_conditions
    @mark.semantic_conditions_IsGreaterEqual
    def test_semantic_conditions_IsGreaterEqual(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = 5
        b = 10
        assert IsGreater(a,b)() == False, "IsGreater incorrect evaluation."
        a = 5
        b = dict(instance=test_object, attribute_name='i')   
        assert IsGreater(a,b)() == True, "IsGreater incorrect evaluation."
        with pytest.raises(ValueError):     
            a = dict(instance=test_object, attribute_name='a_i')   
            b = dict(instance=ref_object, attribute_name='a_i')   
            IsGreater(a,b)() 
        a = dict(instance=test_object, attribute_name='a_g')
        b = dict(instance=ref_object, attribute_name='a_g')
        assert IsGreater(a,b)() == True, "Invalid evaluation of IsGreater"
        a = dict(instance=test_object, attribute_name='a_ge')
        b = dict(instance=ref_object, attribute_name='a_ge')
        assert IsGreater(a,b)() == True, "Invalid evaluation of IsGreater"
        a = dict(instance=test_object, attribute_name='a_ge')
        b = dict(instance=ref_object, attribute_name='a_le')
        assert IsGreater(a,b)() == False, "Invalid evaluation of IsGreater"        

    @mark.semantic_conditions
    @mark.semantic_conditions_IsBetween
    def test_semantic_conditions_IsBetween(self, get_validation_rule_test_object,
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        b1 = [0.1,100]
        b2 = [10,100]
        b3 = [0.1,1]
        # Test invalid b
        with pytest.raises(ValueError):     
            a = dict(instance=test_object, attribute_name='a_i')   
            b = dict(instance=ref_object, attribute_name='a_i')   
            IsBetween(a,b)()
        # Test invalid a
        with pytest.raises(TypeError):     
            a = "some string"
            b = b1
            IsBetween(a,b)()         
        # Test valid integer
        a = dict(instance=test_object, attribute_name='i')
        b = b1
        assert IsBetween(a,b)() == True, "Invalid evaluation of IsBetween"
        # Test valid float
        a = dict(instance=test_object, attribute_name='f')
        b = b1
        assert IsBetween(a,b)() == True, "Invalid evaluation of IsBetween"        
        # Test integer too low
        a = dict(instance=test_object, attribute_name='i')
        b = b2
        assert IsBetween(a,b)() == False, "Invalid evaluation of IsBetween"
        # Test float too low
        a = dict(instance=test_object, attribute_name='f')
        b = b2
        assert IsBetween(a,b)() == False, "Invalid evaluation of IsBetween"
        # Test integer too high
        a = dict(instance=test_object, attribute_name='i')
        b = b3
        assert IsBetween(a,b)() == False, "Invalid evaluation of IsBetween"
        # Test float too high
        a = dict(instance=test_object, attribute_name='f')
        b = b3
        assert IsBetween(a,b)() == False, "Invalid evaluation of IsBetween"
        # Test valid array
        a = dict(instance=test_object, attribute_name='a_i')
        b = b1
        assert IsBetween(a,b)() == True, "Invalid evaluation of IsBetween"
        # Test invalid array
        a = dict(instance=test_object, attribute_name='a_i')
        b = b2
        assert IsBetween(a,b)() == False, "Invalid evaluation of IsBetween"

    @mark.semantic_conditions
    @mark.semantic_conditions_IsMatch
    def test_semantic_conditions_IsMatch(self, get_validation_rule_test_object, 
                                         get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        ref_object = get_validation_rule_reference_object
        a = dict(instance=test_object, attribute_name='a_s')
        b = '[0-9]'    
        assert IsMatch(a,b)() == False, "Invalid IsMatch evaluation."
        a = dict(instance=test_object, attribute_name='a_s')
        b = "oranges"
        assert IsMatch(a,b)() == False, "Invalid IsMatch evaluation."
        a = dict(instance=test_object, attribute_name='s')
        b = dict(instance=ref_object, attribute_name='s')
        assert IsMatch(a,b)() == True, "Invalid IsMatch evaluation."

