# =========================================================================== #
#                               TEST RULES                                    #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_rules.py                                                        #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 24th 2019, 7:41:43 pm                         #
# Last Modified: Tuesday December 24th 2019, 7:42:06 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Tests Rules.

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
    * IntegerRule : IntegerRule, which evaluates whether the value of a specific 
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
import pytest
from pytest import mark
import numpy as np
import re

from ml_studio.services.validation.rules import NoneRule, NotNoneRule
from ml_studio.services.validation.rules import EmptyRule, NotEmptyRule
from ml_studio.services.validation.rules import BoolRule, IntegerRule
from ml_studio.services.validation.rules import FloatRule, NumberRule
from ml_studio.services.validation.rules import StringRule
from ml_studio.services.validation.rules import EqualRule, NotEqualRule
from ml_studio.services.validation.rules import AllowedRule, DisAllowedRule
from ml_studio.services.validation.rules import LessRule, GreaterRule
from ml_studio.services.validation.rules import BetweenRule, RegexRule
from ml_studio.services.validation.rules import RuleSet

from ml_studio.services.validation.conditions import IsNone, IsNotNone
from ml_studio.services.validation.conditions import IsEmpty, IsNotEmpty
from ml_studio.services.validation.conditions import IsEqual, IsNotEqual
from ml_studio.services.validation.conditions import IsBool, IsInt
from ml_studio.services.validation.conditions import IsFloat, IsNumber
from ml_studio.services.validation.conditions import IsString, IsGreater
from ml_studio.services.validation.conditions import IsLess, IsBetween

class SyntacticRuleTests:

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_nonerule
    def test_syntactic_rule_nonerule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object       
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = NoneRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)
        # Evaluate valid basic type 
        rule = NoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.n)
        assert rule.is_valid == True, "Invalid NoneRule evaluation"
        # Evaluate invalid basic type 
        rule = NoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)
        assert rule.is_valid == False, "Invalid NoneRule evaluation"    
        print(rule.invalid_messages)
        # Evaluate valid array type
        rule = NoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_n)
        assert rule.is_valid == True, "Invalid NoneRule evaluation"               
        # Evaluate invalid array type
        rule = NoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_xn)
        assert rule.is_valid == False, "Invalid NoneRule evaluation"               
        print(rule.invalid_messages)
        # Evaluate valid nested array
        rule = NoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_n)
        assert rule.is_valid == True, "Invalid NoneRule evaluation"                 
        # Evaluate invalid nested array
        rule = NoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_xn)
        assert rule.is_valid == False, "Invalid NoneRule evaluation"                 
        print(rule.invalid_messages)

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_NotNonerule
    def test_syntactic_rule_NotNonerule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = NotNoneRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)        
        # Evaluate valid basic type 
        rule = NotNoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.n)
        assert rule.is_valid == False, "Invalid NotNoneRule evaluation"
        print(rule.invalid_messages)
        # Evaluate invalid basic type 
        rule = NotNoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid NotNoneRule evaluation"    
        # Evaluate valid array type
        rule = NotNoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_n)
        assert rule.is_valid == False, "Invalid NotNoneRule evaluation"               
        print(rule.invalid_messages)
        # Evaluate invalid array type
        rule = NotNoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_xn)
        assert rule.is_valid == False, "Invalid NotNoneRule evaluation"                       
        # Evaluate valid nested array
        rule = NotNoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_n)
        assert rule.is_valid == False, "Invalid NotNoneRule evaluation"                 
        print(rule.invalid_messages)
        # Evaluate invalid nested array
        rule = NotNoneRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_xn)
        assert rule.is_valid == True, "Invalid NotNoneRule evaluation"   

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_emptyrule
    def test_syntactic_rule_EmptyRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = EmptyRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)        
        # Evaluate valid basic type 
        rule = EmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.e)
        assert rule.is_valid == True, "Invalid EmptyRule evaluation"        
        # Evaluate valid array-like
        rule = EmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_e)
        assert rule.is_valid == True, "Invalid EmptyRule evaluation"        
        # Evaluate valid array-like with Emptys
        rule = EmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_n)
        assert rule.is_valid == True, "Invalid EmptyRule evaluation"                
        # Evaluate valid nexted array-like with Emptys
        rule = EmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_n)
        assert rule.is_valid == True, "Invalid EmptyRule evaluation"                        
        # Evaluate invalid basic type
        rule = EmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)
        assert rule.is_valid == False, "Invalid EmptyRule evaluation"        
        print(rule.invalid_messages)
        # Evaluate invalid array-like
        rule = EmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_s)
        assert rule.is_valid == False, "Invalid EmptyRule evaluation"        
        print(rule.invalid_messages)
        # Evaluate invalid nested array-like
        rule = EmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_e)
        assert rule.is_valid == False, "Invalid EmptyRule evaluation"        
        print(rule.invalid_messages)        

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_NotEmptyrule
    def test_syntactic_rule_NotEmptyRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = NotEmptyRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)
        # Evaluate valid basic type 
        rule = NotEmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.e)
        assert rule.is_valid == False, "Invalid NotEmptyRule evaluation"        
        print(rule.invalid_messages) 
        # Evaluate valid array-like
        rule = NotEmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_e)
        assert rule.is_valid == False, "Invalid NotEmptyRule evaluation"        
        print(rule.invalid_messages) 
        # Evaluate valid array-like with NotEmptys
        rule = NotEmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_n)
        assert rule.is_valid == False, "Invalid NotEmptyRule evaluation"                
        print(rule.invalid_messages) 
        # Evaluate valid nexted array-like with NotEmptys
        rule = NotEmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_n)
        assert rule.is_valid == False, "Invalid NotEmptyRule evaluation"                        
        print(rule.invalid_messages) 
        # Evaluate invalid basic type
        rule = NotEmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid NotEmptyRule evaluation"                
        # Evaluate invalid array-like
        rule = NotEmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_s)
        assert rule.is_valid == True, "Invalid NotEmptyRule evaluation"                
        # Evaluate invalid nested array-like
        rule = NotEmptyRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_e)
        assert rule.is_valid == False, "Invalid NotEmptyRule evaluation"   

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_Boolrule
    def test_syntactic_rule_BoolRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object    
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = BoolRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)            
        # Evaluate valid basic type 
        rule = BoolRule(test_object,"i", array_ok=True)
        rule.validate(test_object.b)
        assert rule.is_valid == True, "Invalid BoolRule evaluation"                
        # Evaluate valid array-like
        rule = BoolRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_b)
        assert rule.is_valid == True, "Invalid BoolRule evaluation"        
        # Evaluate valid nested array-like
        rule = BoolRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_b)
        assert rule.is_valid == True, "Invalid BoolRule evaluation"                         
        # Evaluate invalid basic type
        rule = BoolRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)
        assert rule.is_valid == False, "Invalid BoolRule evaluation"                             
        print(rule.invalid_messages)
        # Evaluate invalid array
        rule = BoolRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_i)
        assert rule.is_valid == True, "Invalid BoolRule evaluation"                             
        print(rule.invalid_messages)        
        # Evaluate invalid nested array
        
    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_IntegerRule
    def test_syntactic_rule_IntegerRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object    
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = IntegerRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)   

        # Evaluate invalid nested array
        with pytest.raises(ValueError):
            rule = IntegerRule(test_object,"i", array_ok=True)
            rule.validate(test_object.na_e)
        # Evaluate valid basic type 
        rule = IntegerRule(test_object,"i", array_ok=True)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid IntegerRule evaluation"                
        # Evaluate valid array-like
        rule = IntegerRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_i)
        assert rule.is_valid == True, "Invalid IntegerRule evaluation"        
        # Evaluate valid nested array-like
        rule = IntegerRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_i)
        assert rule.is_valid == True, "Invalid IntegerRule evaluation"                         
        # Evaluate invalid basic type
        rule = IntegerRule(test_object,"i", array_ok=True)
        rule.validate(test_object.f)
        assert rule.is_valid == False, "Invalid IntegerRule evaluation"                                 
        # Evaluate string
        rule = IntegerRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)
        assert rule.is_valid == False, "Invalid IntegerRule evaluation"                         
        # Evaluate invalid array
        rule = IntegerRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_f)
        assert rule.is_valid == False, "Invalid IntegerRule evaluation"                             
        print(rule.invalid_messages)        

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_Floatrule
    def test_syntactic_rule_FloatRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object  
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = FloatRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n) 
        # Evaluate valid basic type 
        rule = FloatRule(test_object,"i", array_ok=True)
        rule.validate(test_object.f)
        assert rule.is_valid == True, "Invalid FloatRule evaluation"                
        # Evaluate valid array-like
        rule = FloatRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_f)
        assert rule.is_valid == True, "Invalid FloatRule evaluation"        
        # Evaluate valid nested array-like
        rule = FloatRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_f)
        assert rule.is_valid == True, "Invalid FloatRule evaluation"                         
        # Evaluate invalid array
        rule = FloatRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_i)
        assert rule.is_valid == True, "Invalid FloatRule evaluation"                             
        print(rule.invalid_messages)        
        # Evaluate invalid nested array
        rule = FloatRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_e)
        assert rule.is_valid == False, "Invalid FloatRule evaluation"                             
        print(rule.invalid_messages)       
        # Evaluate string to float
        rule = FloatRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)                   
        assert rule.is_valid == False, "Invalid FloatRule evaluation"                                     
        print(rule.invalid_messages)       

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_Numberrule
    def test_syntactic_rule_NumberRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object     
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = NumberRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)           
        # Evaluate valid basic type 
        rule = NumberRule(test_object,"i", array_ok=True)
        rule.validate(test_object.f)
        assert rule.is_valid == True, "Invalid NumberRule evaluation"                
        # Evaluate valid array-like
        rule = NumberRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_f)
        assert rule.is_valid == True, "Invalid NumberRule evaluation"        
        # Evaluate valid nested array-like
        rule = NumberRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_f)
        assert rule.is_valid == True, "Invalid NumberRule evaluation"                         
        # Evaluate invalid basic type
        rule = NumberRule(test_object,"i", array_ok=True)
        rule.validate(test_object.f)
        assert rule.is_valid == True, "Invalid NumberRule evaluation"                             
        print(rule.invalid_messages)
        # Evaluate invalid array
        rule = NumberRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_f)
        assert rule.is_valid == True, "Invalid NumberRule evaluation"                             
        print(rule.invalid_messages)        
        # Evaluate invalid nested array
        rule = NumberRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_e)
        assert rule.is_valid == False, "Invalid NumberRule evaluation"                             
        print(rule.invalid_messages)              

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_Stringrule
    def test_syntactic_rule_StringRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = StringRule(test_object,"i", array_ok=False)
            rule.validate(test_object.a_n)        
        # Evaluate valid basic type 
        rule = StringRule(test_object,"i", array_ok=True)
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid StringRule evaluation"                
        # Evaluate valid array-like
        rule = StringRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_s)
        assert rule.is_valid == True, "Invalid StringRule evaluation"        
        # Evaluate valid nested array-like
        rule = StringRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_s)
        assert rule.is_valid == True, "Invalid StringRule evaluation"                         
        # Evaluate invalid basic type
        rule = StringRule(test_object,"i", array_ok=True)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid StringRule evaluation"                             
        print(rule.invalid_messages)
        # Evaluate invalid array
        rule = StringRule(test_object,"i", array_ok=True)
        rule.validate(test_object.a_b)
        assert rule.is_valid == False, "Invalid StringRule evaluation"                             
        print(rule.invalid_messages)        
        # Evaluate invalid nested array
        rule = StringRule(test_object,"i", array_ok=True)
        rule.validate(test_object.na_e)
        assert rule.is_valid == True, "Invalid StringRule evaluation"                             
        print(rule.invalid_messages)          


class SyntacticRuleWithConditionsTests:

    @mark.rules
    @mark.syntactic_rules
    @mark.syntactic_rules_with_conditions    
    @mark.syntactic_rules_nonerule_with_conditions
    def test_syntactic_rule_nonerule_with_conditions(self, get_validation_rule_test_object,
                                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Invalid condition type
        with pytest.raises(TypeError):             
            rule = NoneRule(test_object,"i", array_ok=True).when([IsGreater(a=dict(instance=test_object,attribute_name='f'),
                                         b=dict(instance=ref_object, attribute_name='f')),
                                IsEqual(a=dict(instance=test_object,attribute_name='a_le'),
                                         b=dict(instance=ref_object, attribute_name='a_le'))])
            rule.validate(test_object.i)    
        # Invalid conditions type for when_any
        with pytest.raises(TypeError):             
            rule = NoneRule(test_object,"i", array_ok=True).when_any(IsGreater(a=dict(instance=test_object,attribute_name='i'),
                                         b=dict(instance=ref_object, attribute_name='i')))
            rule.validate(test_object.n)
        # Invalid conditions type for when_all
        with pytest.raises(TypeError):             
            rule = NoneRule(test_object,"i", array_ok=True).when_all(IsGreater(a=dict(instance=test_object,attribute_name='i'),
                                         b=dict(instance=ref_object, attribute_name='i')))
            rule.validate(test_object.n)

        # Evaluate with when condition is met
        rule = NoneRule(test_object,"i", array_ok=True).when(IsGreater(a=dict(instance=test_object,attribute_name='i'),
                                         b=dict(instance=ref_object, attribute_name='i')))
        rule.validate(test_object.n)
        assert rule.is_valid == True, "Invalid NoneRule evaluation"
        # Evaluate with when condition is not met
        rule = NoneRule(test_object,"i", array_ok=True).when(IsLess(a=dict(instance=test_object,attribute_name='i'),
                                      b=dict(instance=ref_object, attribute_name='i')))
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid NoneRule evaluation"        
        # Evaluate with when_any is met (but rule not)
        rule = NoneRule(test_object,"i", array_ok=True).when_any([IsGreater(a=dict(instance=test_object,attribute_name='f'),
                                         b=dict(instance=ref_object, attribute_name='f')),
                                IsEqual(a=dict(instance=test_object,attribute_name='a_le'),
                                         b=dict(instance=ref_object, attribute_name='a_le'))])
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid NoneRule evaluation"
        print(rule.invalid_messages)
        # Evaluate with when_any is not met (but rule not)
        rule = NoneRule(test_object,"i", array_ok=True).when_any([IsGreater(a=dict(instance=test_object,attribute_name='f'),
                                         b=dict(instance=ref_object, attribute_name='f')),
                                IsEqual(a=dict(instance=test_object,attribute_name='f'),
                                         b=dict(instance=ref_object, attribute_name='f'))])
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid NoneRule evaluation"
        # Evaluate with when_all is met (but rule not)
        rule = NoneRule(test_object,"i", array_ok=True).when_all([IsLess(a=dict(instance=test_object,attribute_name='f'),
                                         b=dict(instance=ref_object, attribute_name='f')),
                                IsEqual(a=dict(instance=test_object,attribute_name='a_le'),
                                         b=dict(instance=ref_object, attribute_name='a_le'))])
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid NoneRule evaluation"
        print(rule.invalid_messages)
        # Evaluate with when_all is not met (but rule not)
        rule = NoneRule(test_object,"i", array_ok=True).when_all([IsLess(a=dict(instance=test_object,attribute_name='f'),
                                         b=dict(instance=ref_object, attribute_name='f')),
                                IsEqual(a=dict(instance=test_object,attribute_name='f'),
                                         b=dict(instance=ref_object, attribute_name='f'))])
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid NoneRule evaluation"                        


class SemanticRuleTests:

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_equal
    def test_semantic_rule_equalrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = EqualRule(test_object, "a_n",5, array_ok=False)
            rule.validate(test_object.a_n)        
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = EqualRule(test_object, "i", 5, array_ok=True)
            rule.validate(test_object.x)            

        # Evaluate valid against literal value
        rule = EqualRule(test_object, "i", 5, array_ok=True)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid EqualRule evaluation"
        # Evaluate invalid against literal value
        rule = EqualRule(test_object, "i", 7)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid EqualRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value
        reference_value=dict(instance=ref_object, attribute_name="s")
        rule = EqualRule(test_object, "s", reference_value)
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid EqualRule evaluation"
        # Evaluate invalid against external value
        reference_value=dict(instance=ref_object, attribute_name="i")
        rule = EqualRule(ref_object, 'i', reference_value)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid EqualRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value array
        reference_value=dict(instance=ref_object, attribute_name="a_ge")
        rule = EqualRule(ref_object, 'a_ge', reference_value, array_ok=True)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == True, "Invalid EqualRule evaluation"
        # Evaluate invalid against external value array
        reference_value=dict(instance=ref_object, attribute_name="a_ge")
        rule = EqualRule(ref_object, 'a_s', reference_value, array_ok=True)
        rule.validate(test_object.a_s)
        assert rule.is_valid == False, "Invalid EqualRule evaluation"
        print(rule.invalid_messages)

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_notequal
    def test_semantic_rule_notequalrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = NotEqualRule(test_object, "i", 5, array_ok=False)
            rule.validate(test_object.a_n)                
        with pytest.raises(AttributeError):     
            rule = NotEqualRule(test_object, "x", 3)
            rule.validate(test_object.i)    
        with pytest.raises(AttributeError):     
            rule = NotEqualRule(ref_object, 'i', 4)
            rule.validate(test_object.x)            

        # Evaluate valid against literal value
        rule = NotEqualRule(test_object, "i", 2)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid NotEqualRule evaluation"
        # Evaluate invalid against literal value
        rule = NotEqualRule(test_object, "i", 5)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid NotEqualRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value
        rule = NotEqualRule(test_object, "s", "disc")
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid NotEqualRule evaluation"
        # Evaluate invalid against external value
        rule = NotEqualRule(ref_object, 's', "hats")
        rule.validate(test_object.s)
        assert rule.is_valid == False, "Invalid NotEqualRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value array
        rule = NotEqualRule(ref_object, 'a_s', ref_object.a_s, array_ok=True)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == True, "Invalid NotEqualRule evaluation"
        # Evaluate invalid against external value array
        rule = NotEqualRule(ref_object, 'a_ge', ref_object.a_ge, array_ok=True)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == False, "Invalid NotEqualRule evaluation"
        print(rule.invalid_messages)

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_allowed
    def test_semantic_rule_allowedrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = AllowedRule(test_object, "a_n", [1,2], array_ok=False)
            rule.validate(test_object.a_n)                
        with pytest.raises(AttributeError):     
            rule = AllowedRule(ref_object, 'x', 9)
            rule.validate(test_object.i)    
         

        # Evaluate valid against literal value
        rule = AllowedRule(test_object, "i", [1,2,3,4,5], array_ok=True)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid AllowedRule evaluation"
        # Evaluate invalid against literal value
        rule = AllowedRule(test_object, "i", [1,2,3], array_ok=True)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid AllowedRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value
        reference_value=dict(instance=ref_object, attribute_name="a_ge")
        rule = AllowedRule(test_object, 'a_ge', reference_value, array_ok=True)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == True, "Invalid AllowedRule evaluation"
        # Evaluate invalid against external value
        reference_value=dict(instance=ref_object, attribute_name="a_g")
        rule = AllowedRule(test_object, 'a_g', reference_value, array_ok=True)
        rule.validate(test_object.a_g)
        assert rule.is_valid == False, "Invalid AllowedRule evaluation"
        print(rule.invalid_messages)

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_disallowed
    def test_semantic_rule_disallowedrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = DisAllowedRule(test_object, "a_n", 5, array_ok=False)
            rule.validate(test_object.a_n)                
        with pytest.raises(AttributeError):     
            rule = DisAllowedRule(test_object, 'x',4)
            rule.validate(test_object.i)    
        with pytest.raises(AttributeError):     
            rule = DisAllowedRule(test_object, 'i',3)
            rule.validate(test_object.x)            

        # Evaluate invalid against literal value
        rule = DisAllowedRule(test_object, "i", [1,2,3,4,5], array_ok=True)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid DisAllowedRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against literal value
        rule = DisAllowedRule(test_object, "i", [1,2,3], array_ok=True)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid DisAllowedRule evaluation"        
        # Evaluate invalid against external value
        rule = DisAllowedRule(test_object,'a_ge', test_object.a_ge, array_ok=True)
        rule.validate(test_object.a_ge)
        print(rule.invalid_messages)
        assert rule.is_valid == False, "Invalid DisAllowedRule evaluation"
        # Evaluate valid against external value
        rule = DisAllowedRule(ref_object, 'a_l', ref_object.a_l, array_ok=True)
        rule.validate(test_object.a_l)
        assert rule.is_valid == True, "Invalid DisAllowedRule evaluation"        

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_less
    def test_semantic_rule_lessrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = LessRule(test_object, "a_n", 5, array_ok=False)
            rule.validate(test_object.a_n)                
        with pytest.raises(AttributeError):     
            rule = LessRule(test_object,'x', 4)
            rule.validate(test_object.i)    
        with pytest.raises(AttributeError):     
            rule = LessRule(test_object, 'i', 3)
            rule.validate(test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = LessRule(test_object, 'l',3)
            rule.validate(test_object.i)                        

        # Evaluate valid against literal value
        rule = LessRule(test_object, "i", 10)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against literal value
        rule = LessRule(test_object, "i", 2)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid LessRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value
        rule = LessRule(test_object, "f", ref_object.f)
        rule.validate(test_object.f)
        assert rule.is_valid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against external value
        rule = LessRule(test_object, 'i', ref_object.i)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid LessRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value array, inclusive = False
        rule = LessRule(test_object, 'a_l', ref_object.a_l, array_ok=True, inclusive=False)
        rule.validate(test_object.a_l)
        assert rule.is_valid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against external value array, inclusive=False
        rule = LessRule(test_object, 'a_ge', ref_object.a_ge, array_ok=True, inclusive=False)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == False, "Invalid LessRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value array, inclusive = True
        rule = LessRule(test_object, 'a_le', ref_object.a_le, array_ok=True, inclusive=True)
        rule.validate(test_object.a_le)
        assert rule.is_valid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against external value array, inclusive=True
        rule = LessRule(test_object, 'a_g', ref_object.a_g, array_ok=True, inclusive=True)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == False, "Invalid LessRule evaluation"
        print(rule.invalid_messages)        

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_greater
    def test_semantic_rule_greaterrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object

        # Validate rule
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = GreaterRule(test_object, "a_n", 5, array_ok=False)
            rule.validate(test_object.a_n)                
        with pytest.raises(AttributeError):     
            rule = GreaterRule(test_object,'x', 4)
            rule.validate(test_object.i)    
        with pytest.raises(AttributeError):     
            rule = GreaterRule(test_object, 'i', 3)
            rule.validate(test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = GreaterRule(test_object, 'l',3)
            rule.validate(test_object.i)                        

        # Evaluate valid against literal value
        rule = GreaterRule(test_object, "i", 2)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"
        # Evaluate invalid against literal value
        rule = GreaterRule(test_object, "i", 10)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value
        rule = GreaterRule(test_object, "f", ref_object.f)
        rule.validate(test_object.f)
        assert rule.is_valid == False, "Invalid GreaterRule evaluation"
        # Evaluate invalid against external value
        rule = GreaterRule(test_object, 'i', ref_object.i)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value array, inclusive = False
        rule = GreaterRule(test_object, 'a_l', ref_object.a_l, array_ok=True, inclusive=False)
        rule.validate(test_object.a_l)
        assert rule.is_valid == False, "Invalid GreaterRule evaluation"
        # Evaluate invalid against external value array, inclusive=False
        rule = GreaterRule(test_object, 'a_ge', ref_object.a_ge, array_ok=True, inclusive=False)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid against external value array, inclusive = True
        rule = GreaterRule(test_object, 'a_le', ref_object.a_le, array_ok=True, inclusive=True)
        rule.validate(test_object.a_le)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"
        # Evaluate invalid against external value array, inclusive=True
        rule = GreaterRule(test_object, 'a_g', ref_object.a_g, array_ok=True, inclusive=True)
        rule.validate(test_object.a_ge)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"
        print(rule.invalid_messages)               

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_between
    def test_semantic_rule_betweenrule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object                
        # Validate rule
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = BetweenRule(test_object, "a_n", 5, array_ok=False)
            rule.validate(test_object.a_n)                
        with pytest.raises(AttributeError):     
            rule = BetweenRule(test_object, 'x', 3)
            rule.validate(test_object.i)    
        with pytest.raises(AttributeError):     
            rule = BetweenRule(test_object, 'i', 5)
            rule.validate(test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = BetweenRule(test_object, 'l', 9)
            rule.validate(test_object.i)                        

        # Ranges
        r1 = [0,10]
        r2 = [0,1]
        r3 = [100,1000]

        # Integers
        # Evaluate valid basic type, inclusive=False
        rule = BetweenRule(test_object, "i", r1, inclusive=False)
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid basic type, inclusive=False
        rule = BetweenRule(test_object, "i", r2, inclusive=False)
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid array-like, inclusive=True
        rule = BetweenRule(test_object, "a_g", r1, inclusive=True, array_ok=True)
        rule.validate(test_object.a_g)
        assert rule.is_valid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid array-like, inclusive=False
        rule = BetweenRule(test_object, "a_g", r2, inclusive=True, array_ok=True)
        rule.validate(test_object.a_g)
        assert rule.is_valid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_messages)

        # Float
        # Evaluate valid basic type, inclusive=False
        rule = BetweenRule(test_object, "f", r1, inclusive=False)
        rule.validate(test_object.f)
        assert rule.is_valid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid basic type, inclusive=False
        rule = BetweenRule(test_object, "f", r2, inclusive=False)
        rule.validate(test_object.f)
        assert rule.is_valid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_messages)
        # Evaluate valid array-like, inclusive=True
        rule = BetweenRule(test_object, "na_f", r1, inclusive=True, array_ok=True)
        rule.validate(test_object.na_f)
        assert rule.is_valid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid array-like, inclusive=False
        rule = BetweenRule(test_object, "na_f", r3, inclusive=True, array_ok=True)
        rule.validate(test_object.na_f)
        assert rule.is_valid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_messages)

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_regex
    def test_semantic_rule_regexrule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Validate rule
        # Evaluate array_ok=False
        with pytest.raises(TypeError):
            rule = RegexRule(test_object, "a_n",5, array_ok=False)
            rule.validate(test_object.a_n)                
        with pytest.raises(AttributeError):     
            rule = RegexRule(test_object, 'x', 3)
            rule.validate(test_object.i)    
        with pytest.raises(AttributeError):     
            rule = RegexRule(test_object, 'i',3)
            rule.validate(test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = RegexRule(test_object, 'l',7)
            rule.validate(test_object.i)              
        with pytest.raises(TypeError):                 
            rule = RegexRule(test_object, 'i', 9)
            rule.validate(test_object.i)             
        regex = "[a-zA-Z]+"
        # Evaluate valid basic type
        rule = RegexRule(test_object, "s", regex)
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid BetweenRule evaluation"   
        # Evaluate valid array-like 
        rule = RegexRule(test_object, "a_s", regex, array_ok=True)
        rule.validate(test_object.a_s)
        assert rule.is_valid == True, "Invalid BetweenRule evaluation"                       
        regex = "[0-9]+"
        # Evaluate invalid basic type
        rule = RegexRule(test_object, "s", regex)
        rule.validate(test_object.s)
        assert rule.is_valid == False, "Invalid BetweenRule evaluation"   
        print(rule.invalid_messages)
        # Evaluate invalid array-like 
        rule = RegexRule(test_object, 'a_s', regex, array_ok=True)
        rule.validate(test_object.a_s)
        assert rule.is_valid == False, "Invalid BetweenRule evaluation"                               
        print(rule.invalid_messages)


class SemanticRuleWithConditionsTests:

    @mark.rules
    @mark.semantic_rules
    @mark.semantic_rules_with_conditions    
    @mark.semantic_rules_greaterrule_with_conditions
    def test_semantic_rule_greaterrule_with_conditions(self, get_validation_rule_test_object,
                                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Invalid condition type
        with pytest.raises(TypeError):             
            rule = GreaterRule(test_object, "i",5)\
                .when([IsGreater(a=dict(instance=test_object,attribute_name='f'),
                                 b=dict(instance=ref_object, attribute_name='f')),
                       IsEqual(a=dict(instance=test_object,attribute_name='a_le'),
                               b=dict(instance=ref_object, attribute_name='a_le'))])
            rule.validate(test_object.i)    
        # Invalid conditions type for when_any
        with pytest.raises(TypeError):             
            rule = GreaterRule(test_object, "n",1)\
                .when_any(IsGreater(a=dict(instance=test_object,attribute_name='i'),
                                    b=dict(instance=ref_object, attribute_name='i')))
            rule.validate(test_object.n)
        # Invalid conditions type for when_all
        with pytest.raises(TypeError):             
            rule = GreaterRule(test_object, "n",1)\
                .when_all(IsGreater(a=dict(instance=test_object,attribute_name='i'),
                                    b=dict(instance=ref_object, attribute_name='i')))
            rule.validate(test_object.n)

        # Evaluate with when condition is met
        rule = GreaterRule(test_object, "i",2)\
            .when(IsGreater(a=dict(instance=test_object,attribute_name='i'),
                            b=dict(instance=ref_object, attribute_name='i')))
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"
        # Evaluate with when condition is not met
        rule = GreaterRule(test_object, "s",1)\
            .when(IsLess(a=dict(instance=test_object,attribute_name='i'),
                         b=dict(instance=ref_object, attribute_name='i')))
        rule.validate(test_object.s)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"        
        # Evaluate with when_any is met (but rule not)
        rule = GreaterRule(test_object, "i",10)\
            .when_any([IsGreater(a=dict(instance=test_object,attribute_name='f'),
                                 b=dict(instance=ref_object, attribute_name='f')),
                       IsEqual(a=dict(instance=test_object,attribute_name='a_le'),
                               b=dict(instance=ref_object, attribute_name='a_le'))])
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_messages)
        # Evaluate with when_any is not met (but rule not)
        rule = GreaterRule(test_object, "i",1)\
            .when_any([IsGreater(a=dict(instance=test_object,attribute_name='f'),
                                 b=dict(instance=ref_object, attribute_name='f')),
                       IsEqual(a=dict(instance=test_object,attribute_name='f'),
                               b=dict(instance=ref_object, attribute_name='f'))])
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"
        # Evaluate with when_all is met (but rule not)
        rule = GreaterRule(test_object, "i",10)\
            .when_all([IsLess(a=dict(instance=test_object,attribute_name='f'),
                              b=dict(instance=ref_object, attribute_name='f')),
                       IsEqual(a=dict(instance=test_object,attribute_name='a_le'),
                               b=dict(instance=ref_object, attribute_name='a_le'))])
        rule.validate(test_object.i)
        assert rule.is_valid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_messages)
        # Evaluate with when_all is not met (but rule not)
        rule = GreaterRule(test_object, "i",1)\
            .when_all([IsLess(a=dict(instance=test_object,attribute_name='f'),
                              b=dict(instance=ref_object, attribute_name='f')),
                       IsEqual(a=dict(instance=test_object,attribute_name='f'),
                               b=dict(instance=ref_object, attribute_name='f'))])
        rule.validate(test_object.i)
        assert rule.is_valid == True, "Invalid GreaterRule evaluation"          

class RuleSetTests:

    @mark.rules
    @mark.ruleset
    def test_ruleset(self, get_validation_rule_test_object):
        # Store regex patterns that will comprise the ruleset
        re_hex="#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})"
        re_rgb_etc="(rgb|hsl|hsv)a?\\([\\d.]+%?(,[\\d.]+%?){2,3}\\)"
        re_ddk="var\\(\\-\\-.*\\)"
        # Obtain test and ref objects
        test_object = get_validation_rule_test_object     
        # Create rules
        hex_rule=RegexRule(test_object, "s",re_hex)
        rgb_rule=RegexRule(test_object, "s",re_rgb_etc)
        ddk_rule=RegexRule(test_object, "s",re_ddk)
        # Create RuleSet
        ruleset = RuleSet()
        ruleset.operation='or'
        ruleset.add_rule(hex_rule)
        ruleset.add_rule(rgb_rule)
        ruleset.add_rule(ddk_rule)
        # Evaluate valid ruleset
        ruleset.validate(test_object.color_hex)
        assert ruleset.is_valid == True, "Invalid RuleSet RegexRule evaluation"
        # Evaluate another valid ruleset
        ruleset.validate(test_object.color_rgb)
        assert ruleset.is_valid == True, "Invalid RuleSet RegexRule evaluation"
        # Evaluate invalid ruleset
        ruleset.validate(test_object.s)
        assert ruleset.is_valid == False, "Invalid RuleSet RegexRule evaluation"
        # Print ruleset
        ruleset.print_rule()

class PrintSyntacticRuleTests:

    @mark.syntactic_rules
    @mark.syntactic_rules_print   
    @mark.syntactic_rules_print_nonerule
    def test_syntactic_rule_print_nonerule(self, get_validation_rule_test_object,
                                           get_validation_rule_reference_object): 
        test_object = get_validation_rule_test_object       
        ref_object = get_validation_rule_reference_object
        rule = NoneRule(test_object,"i", array_ok=True).when(IsGreater(a=dict(instance=test_object, 
                                                attribute_name='i'),
                                         b=dict(instance=ref_object, 
                                                attribute_name='i')))\
                         .when_any([IsNotEqual(a=dict(instance=test_object, 
                                                attribute_name='i'),
                                               b=dict(instance=ref_object, 
                                                attribute_name='i')),
                                    IsLess(a=dict(instance=test_object, 
                                                attribute_name='i'),
                                           b=dict(instance=ref_object, 
                                                attribute_name='i'))])\
                         .when_all([IsNotEqual(a=dict(instance=test_object, 
                                                attribute_name='i'),
                                               b=dict(instance=ref_object, 
                                                attribute_name='i')),
                                    IsLess(a=dict(instance=test_object, 
                                                attribute_name='i'),
                                           b=dict(instance=ref_object, 
                                                attribute_name='i'))])
        
        rule.print_rule()

class PrintSemanticRuleTests:

    @mark.semantic_rules
    @mark.semantic_rules_print   
    @mark.semantic_rules_print_nonerule
    def test_semantic_rule_print_nonerule(self, get_validation_rule_test_object,
                                           get_validation_rule_reference_object): 
        test_object = get_validation_rule_test_object       
        ref_object = get_validation_rule_reference_object
        rule = GreaterRule(instance=ref_object,
                           attribute_name='f',
                           reference_value=5)\
                               .when(IsGreater(a=dict(instance=test_object, 
                                                attribute_name='i'),
                                         b=dict(instance=ref_object, 
                                                attribute_name='i')))\
                                .when_any([IsNotEqual(a=dict(instance=test_object, 
                                                        attribute_name='i'),
                                                    b=dict(instance=ref_object, 
                                                        attribute_name='i')),
                                            IsLess(a=dict(instance=test_object, 
                                                        attribute_name='i'),
                                                b=dict(instance=ref_object, 
                                                        attribute_name='i'))])\
                                .when_all([IsNotEqual(a=dict(instance=test_object, 
                                                        attribute_name='i'),
                                                    b=dict(instance=ref_object, 
                                                        attribute_name='i')),
                                            IsLess(a=dict(instance=test_object, 
                                                        attribute_name='i'),
                                                b=dict(instance=ref_object, 
                                                        attribute_name='i'))])
        
        rule.print_rule()


