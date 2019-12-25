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
import pytest
from pytest import mark
import numpy as np

from ml_studio.services.validation.rules import NoneRule, NotNoneRule
from ml_studio.services.validation.rules import EmptyRule, NotEmptyRule
from ml_studio.services.validation.rules import BoolRule, IntRule
from ml_studio.services.validation.rules import FloatRule, NumberRule
from ml_studio.services.validation.rules import StringRule
from ml_studio.services.validation.rules import EqualRule, NotEqualRule
from ml_studio.services.validation.rules import AllowedRule, DisAllowedRule
from ml_studio.services.validation.rules import LessRule, GreaterRule
from ml_studio.services.validation.rules import BetweenRule, RegexRule

class SyntacticRuleTests:

    @mark.syntactic_rules
    @mark.syntactic_rules_nonerule
    def test_syntactic_rule_nonerule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = NoneRule()
        rule.validate(test_object, 'n', test_object.n)
        assert rule.isValid == True, "Invalid NoneRule evaluation"
        # Evaluate invalid basic type 
        rule = NoneRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid NoneRule evaluation"    
        print(rule.invalid_message)
        # Evaluate valid array type
        rule = NoneRule()
        rule.validate(test_object, 'a_n', test_object.a_n)
        assert rule.isValid == True, "Invalid NoneRule evaluation"               
        # Evaluate invalid array type
        rule = NoneRule()
        rule.validate(test_object, 'a_xn', test_object.a_xn)
        assert rule.isValid == False, "Invalid NoneRule evaluation"               
        print(rule.invalid_message)
        # Evaluate valid nested array
        rule = NoneRule()
        rule.validate(test_object, 'na_n', test_object.na_n)
        assert rule.isValid == True, "Invalid NoneRule evaluation"                 
        # Evaluate invalid nested array
        rule = NoneRule()
        rule.validate(test_object, 'na_xn', test_object.na_xn)
        assert rule.isValid == False, "Invalid NoneRule evaluation"                 
        print(rule.invalid_message)

    @mark.syntactic_rules
    @mark.syntactic_rules_NotNonerule
    def test_syntactic_rule_NotNonerule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = NotNoneRule()
        rule.validate(test_object, 'n', test_object.n)
        assert rule.isValid == False, "Invalid NotNoneRule evaluation"
        print(rule.invalid_message)
        # Evaluate invalid basic type 
        rule = NotNoneRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == True, "Invalid NotNoneRule evaluation"    
        # Evaluate valid array type
        rule = NotNoneRule()
        rule.validate(test_object, 'a_n', test_object.a_n)
        assert rule.isValid == False, "Invalid NotNoneRule evaluation"               
        print(rule.invalid_message)
        # Evaluate invalid array type
        rule = NotNoneRule()
        rule.validate(test_object, 'a_xn', test_object.a_xn)
        assert rule.isValid == False, "Invalid NotNoneRule evaluation"               
        print(rule.invalid_message)
        # Evaluate valid nested array
        rule = NotNoneRule()
        rule.validate(test_object, 'na_n', test_object.na_n)
        assert rule.isValid == False, "Invalid NotNoneRule evaluation"                 
        print(rule.invalid_message)
        # Evaluate invalid nested array
        rule = NotNoneRule()
        rule.validate(test_object, 'na_xn', test_object.na_xn)
        assert rule.isValid == True, "Invalid NotNoneRule evaluation"   

    @mark.syntactic_rules
    @mark.syntactic_rules_emptyrule
    def test_syntactic_rule_EmptyRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = EmptyRule()
        rule.validate(test_object, 'e', test_object.e)
        assert rule.isValid == True, "Invalid EmptyRule evaluation"        
        # Evaluate valid array-like
        rule = EmptyRule()
        rule.validate(test_object, 'a_e', test_object.a_e)
        assert rule.isValid == True, "Invalid EmptyRule evaluation"        
        # Evaluate valid array-like with Emptys
        rule = EmptyRule()
        rule.validate(test_object, 'a_n', test_object.a_n)
        assert rule.isValid == True, "Invalid EmptyRule evaluation"                
        # Evaluate valid nexted array-like with Emptys
        rule = EmptyRule()
        rule.validate(test_object, 'na_n', test_object.na_n)
        assert rule.isValid == True, "Invalid EmptyRule evaluation"                        
        # Evaluate invalid basic type
        rule = EmptyRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid EmptyRule evaluation"        
        print(rule.invalid_message)
        # Evaluate invalid array-like
        rule = EmptyRule()
        rule.validate(test_object, 'a_s', test_object.a_s)
        assert rule.isValid == False, "Invalid EmptyRule evaluation"        
        print(rule.invalid_message)
        # Evaluate invalid nested array-like
        rule = EmptyRule()
        rule.validate(test_object, 'na_e', test_object.na_e)
        assert rule.isValid == False, "Invalid EmptyRule evaluation"        
        print(rule.invalid_message)        

    @mark.syntactic_rules
    @mark.syntactic_rules_NotEmptyrule
    def test_syntactic_rule_NotEmptyRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = NotEmptyRule()
        rule.validate(test_object, 'e', test_object.e)
        assert rule.isValid == False, "Invalid NotEmptyRule evaluation"        
        print(rule.invalid_message) 
        # Evaluate valid array-like
        rule = NotEmptyRule()
        rule.validate(test_object, 'a_e', test_object.a_e)
        assert rule.isValid == False, "Invalid NotEmptyRule evaluation"        
        print(rule.invalid_message) 
        # Evaluate valid array-like with NotEmptys
        rule = NotEmptyRule()
        rule.validate(test_object, 'a_n', test_object.a_n)
        assert rule.isValid == False, "Invalid NotEmptyRule evaluation"                
        print(rule.invalid_message) 
        # Evaluate valid nexted array-like with NotEmptys
        rule = NotEmptyRule()
        rule.validate(test_object, 'na_n', test_object.na_n)
        assert rule.isValid == False, "Invalid NotEmptyRule evaluation"                        
        print(rule.invalid_message) 
        # Evaluate invalid basic type
        rule = NotEmptyRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == True, "Invalid NotEmptyRule evaluation"                
        # Evaluate invalid array-like
        rule = NotEmptyRule()
        rule.validate(test_object, 'a_s', test_object.a_s)
        assert rule.isValid == True, "Invalid NotEmptyRule evaluation"                
        # Evaluate invalid nested array-like
        rule = NotEmptyRule()
        rule.validate(test_object, 'na_e', test_object.na_e)
        assert rule.isValid == True, "Invalid NotEmptyRule evaluation"   

    @mark.syntactic_rules
    @mark.syntactic_rules_Boolrule
    def test_syntactic_rule_BoolRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = BoolRule()
        rule.validate(test_object, 'b', test_object.b)
        assert rule.isValid == True, "Invalid BoolRule evaluation"                
        # Evaluate valid array-like
        rule = BoolRule()
        rule.validate(test_object, 'a_b', test_object.a_b)
        assert rule.isValid == True, "Invalid BoolRule evaluation"        
        # Evaluate valid nested array-like
        rule = BoolRule()
        rule.validate(test_object, 'na_b', test_object.na_b)
        assert rule.isValid == True, "Invalid BoolRule evaluation"                         
        # Evaluate invalid basic type
        rule = BoolRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid BoolRule evaluation"                             
        print(rule.invalid_message)
        # Evaluate invalid array
        rule = BoolRule()
        rule.validate(test_object, 'a_i', test_object.a_i)
        assert rule.isValid == False, "Invalid BoolRule evaluation"                             
        print(rule.invalid_message)        
        # Evaluate invalid nested array
        rule = BoolRule()
        rule.validate(test_object, 'na_e', test_object.na_e)
        assert rule.isValid == False, "Invalid BoolRule evaluation"                             
        print(rule.invalid_message)                

    @mark.syntactic_rules
    @mark.syntactic_rules_Intrule
    def test_syntactic_rule_IntRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = IntRule()
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid IntRule evaluation"                
        # Evaluate valid array-like
        rule = IntRule()
        rule.validate(test_object, 'a_i', test_object.a_i)
        assert rule.isValid == True, "Invalid IntRule evaluation"        
        # Evaluate valid nested array-like
        rule = IntRule()
        rule.validate(test_object, 'na_i', test_object.na_i)
        assert rule.isValid == True, "Invalid IntRule evaluation"                         
        # Evaluate invalid basic type
        rule = IntRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid IntRule evaluation"                             
        print(rule.invalid_message)
        # Evaluate invalid array
        rule = IntRule()
        rule.validate(test_object, 'a_f', test_object.a_f)
        assert rule.isValid == False, "Invalid IntRule evaluation"                             
        print(rule.invalid_message)        
        # Evaluate invalid nested array
        rule = IntRule()
        rule.validate(test_object, 'na_e', test_object.na_e)
        assert rule.isValid == False, "Invalid IntRule evaluation"                             
        print(rule.invalid_message)              

    @mark.syntactic_rules
    @mark.syntactic_rules_Floatrule
    def test_syntactic_rule_FloatRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = FloatRule()
        rule.validate(test_object, 'f', test_object.f)
        assert rule.isValid == True, "Invalid FloatRule evaluation"                
        # Evaluate valid array-like
        rule = FloatRule()
        rule.validate(test_object, 'a_f', test_object.a_f)
        assert rule.isValid == True, "Invalid FloatRule evaluation"        
        # Evaluate valid nested array-like
        rule = FloatRule()
        rule.validate(test_object, 'na_f', test_object.na_f)
        assert rule.isValid == True, "Invalid FloatRule evaluation"                         
        # Evaluate invalid basic type
        rule = FloatRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid FloatRule evaluation"                             
        print(rule.invalid_message)
        # Evaluate invalid array
        rule = FloatRule()
        rule.validate(test_object, 'a_i', test_object.a_i)
        assert rule.isValid == False, "Invalid FloatRule evaluation"                             
        print(rule.invalid_message)        
        # Evaluate invalid nested array
        rule = FloatRule()
        rule.validate(test_object, 'na_e', test_object.na_e)
        assert rule.isValid == False, "Invalid FloatRule evaluation"                             
        print(rule.invalid_message)               

    @mark.syntactic_rules
    @mark.syntactic_rules_Numberrule
    def test_syntactic_rule_NumberRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = NumberRule()
        rule.validate(test_object, 'f', test_object.f)
        assert rule.isValid == True, "Invalid NumberRule evaluation"                
        # Evaluate valid array-like
        rule = NumberRule()
        rule.validate(test_object, 'a_f', test_object.a_f)
        assert rule.isValid == True, "Invalid NumberRule evaluation"        
        # Evaluate valid nested array-like
        rule = NumberRule()
        rule.validate(test_object, 'na_f', test_object.na_f)
        assert rule.isValid == True, "Invalid NumberRule evaluation"                         
        # Evaluate invalid basic type
        rule = NumberRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid NumberRule evaluation"                             
        print(rule.invalid_message)
        # Evaluate invalid array
        rule = NumberRule()
        rule.validate(test_object, 'a_s', test_object.a_s)
        assert rule.isValid == False, "Invalid NumberRule evaluation"                             
        print(rule.invalid_message)        
        # Evaluate invalid nested array
        rule = NumberRule()
        rule.validate(test_object, 'na_e', test_object.na_e)
        assert rule.isValid == False, "Invalid NumberRule evaluation"                             
        print(rule.invalid_message)              

    @mark.syntactic_rules
    @mark.syntactic_rules_Stringrule
    def test_syntactic_rule_StringRule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Evaluate valid basic type 
        rule = StringRule()
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == True, "Invalid StringRule evaluation"                
        # Evaluate valid array-like
        rule = StringRule()
        rule.validate(test_object, 'a_s', test_object.a_s)
        assert rule.isValid == True, "Invalid StringRule evaluation"        
        # Evaluate valid nested array-like
        rule = StringRule()
        rule.validate(test_object, 'na_s', test_object.na_s)
        assert rule.isValid == True, "Invalid StringRule evaluation"                         
        # Evaluate invalid basic type
        rule = StringRule()
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid StringRule evaluation"                             
        print(rule.invalid_message)
        # Evaluate invalid array
        rule = StringRule()
        rule.validate(test_object, 'a_b', test_object.a_b)
        assert rule.isValid == False, "Invalid StringRule evaluation"                             
        print(rule.invalid_message)        
        # Evaluate invalid nested array
        rule = StringRule()
        rule.validate(test_object, 'na_e', test_object.na_e)
        assert rule.isValid == False, "Invalid StringRule evaluation"                             
        print(rule.invalid_message)          

class SemanticRuleTests:

    @mark.semantic_rules
    @mark.semantic_rules_equal
    def test_syntactic_rule_equalrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = EqualRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = EqualRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            

        # Evaluate valid against literal value
        rule = EqualRule(value=5)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid EqualRule evaluation"
        # Evaluate invalid against literal value
        rule = EqualRule(value=7)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid EqualRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value
        rule = EqualRule(instance=ref_object, attribute_name='s')
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == True, "Invalid EqualRule evaluation"
        # Evaluate invalid against external value
        rule = EqualRule(instance=ref_object, attribute_name='i')
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid EqualRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value array
        rule = EqualRule(instance=ref_object, attribute_name='a_ge')
        rule.validate(test_object, 'a_ge', test_object.a_ge)
        assert rule.isValid == True, "Invalid EqualRule evaluation"
        # Evaluate invalid against external value array
        rule = EqualRule(instance=ref_object, attribute_name='a_s')
        rule.validate(test_object, 'a_s', test_object.a_s)
        assert rule.isValid == False, "Invalid EqualRule evaluation"
        print(rule.invalid_message)

    @mark.semantic_rules
    @mark.semantic_rules_notequal
    def test_syntactic_rule_notequalrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = NotEqualRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = NotEqualRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            

        # Evaluate valid against literal value
        rule = NotEqualRule(value=2)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid NotEqualRule evaluation"
        # Evaluate invalid against literal value
        rule = NotEqualRule(value=5)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid NotEqualRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value
        rule = NotEqualRule(instance=ref_object, attribute_name='i')
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == True, "Invalid NotEqualRule evaluation"
        # Evaluate invalid against external value
        rule = NotEqualRule(instance=ref_object, attribute_name='s')
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid NotEqualRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value array
        rule = NotEqualRule(instance=ref_object, attribute_name='a_s')
        rule.validate(test_object, 'a_ge', test_object.a_ge)
        assert rule.isValid == True, "Invalid NotEqualRule evaluation"
        # Evaluate invalid against external value array
        rule = NotEqualRule(instance=ref_object, attribute_name='a_ge')
        rule.validate(test_object, 'a_ge', test_object.a_ge)
        assert rule.isValid == False, "Invalid NotEqualRule evaluation"
        print(rule.invalid_message)

    @mark.semantic_rules
    @mark.semantic_rules_allowed
    def test_syntactic_rule_allowedrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = AllowedRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = AllowedRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            

        # Evaluate valid against literal value
        rule = AllowedRule(value=[1,2,3,4,5])
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid AllowedRule evaluation"
        # Evaluate invalid against literal value
        rule = AllowedRule(value=[1,2,3])
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid AllowedRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value
        rule = AllowedRule(instance=ref_object, attribute_name='a_ge')
        rule.validate(test_object, 'a_ge', test_object.a_ge)
        assert rule.isValid == True, "Invalid AllowedRule evaluation"
        # Evaluate invalid against external value
        rule = AllowedRule(instance=ref_object, attribute_name='a_g')
        rule.validate(test_object, 'a_g', test_object.a_g)
        assert rule.isValid == False, "Invalid AllowedRule evaluation"
        print(rule.invalid_message)

    @mark.semantic_rules
    @mark.semantic_rules_disallowed
    def test_syntactic_rule_disallowedrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = DisAllowedRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = DisAllowedRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            

        # Evaluate invalid against literal value
        rule = DisAllowedRule(value=[1,2,3,4,5])
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid DisAllowedRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against literal value
        rule = DisAllowedRule(value=[1,2,3])
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid DisAllowedRule evaluation"        
        # Evaluate invalid against external value
        rule = DisAllowedRule(instance=ref_object, attribute_name='a_ge')
        rule.validate(test_object, 'a_ge', test_object.a_ge)
        print(rule.invalid_message)
        assert rule.isValid == False, "Invalid DisAllowedRule evaluation"
        # Evaluate valid against external value
        rule = DisAllowedRule(instance=ref_object, attribute_name='a_l')
        rule.validate(test_object, 'a_l', test_object.a_l)
        assert rule.isValid == True, "Invalid DisAllowedRule evaluation"        

    @mark.semantic_rules
    @mark.semantic_rules_less
    def test_syntactic_rule_lessrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = LessRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = LessRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = LessRule(instance=ref_object, attribute_name='l')
            rule.validate(test_object, 'i', test_object.i)                        

        # Evaluate valid against literal value
        rule = LessRule(value=10)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against literal value
        rule = LessRule(value=2)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid LessRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value
        rule = LessRule(instance=ref_object, attribute_name='f')
        rule.validate(test_object, 'f', test_object.f)
        assert rule.isValid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against external value
        rule = LessRule(instance=ref_object, attribute_name='i')
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid LessRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value array, inclusive = False
        rule = LessRule(instance=ref_object, attribute_name='a_l', inclusive=False)
        rule.validate(test_object, 'a_l', test_object.a_l)
        assert rule.isValid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against external value array, inclusive=False
        rule = LessRule(instance=ref_object, attribute_name='a_ge', inclusive=False)
        rule.validate(test_object, 'a_ge', test_object.a_ge)
        assert rule.isValid == False, "Invalid LessRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid against external value array, inclusive = True
        rule = LessRule(instance=ref_object, attribute_name='a_le', inclusive=True)
        rule.validate(test_object, 'a_le', test_object.a_le)
        assert rule.isValid == True, "Invalid LessRule evaluation"
        # Evaluate invalid against external value array, inclusive=True
        rule = LessRule(instance=ref_object, attribute_name='a_g', inclusive=True)
        rule.validate(test_object, 'a_ge', test_object.a_ge)
        assert rule.isValid == False, "Invalid LessRule evaluation"
        print(rule.invalid_message)        

    @mark.semantic_rules
    @mark.semantic_rules_greater
    def test_syntactic_rule_greaterrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = GreaterRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = GreaterRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = GreaterRule(instance=ref_object, attribute_name='l')
            rule.validate(test_object, 'i', test_object.i)                        

        # Evaluate valid against literal value
        rule = GreaterRule(value=10)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_message)
        # Evaluate invalid against literal value
        rule = GreaterRule(value=2)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid GreaterRule evaluation"
                # Evaluate valid against external value
        rule = GreaterRule(instance=ref_object, attribute_name='f')
        rule.validate(test_object, 'f', test_object.f)
        assert rule.isValid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_message)
        # Evaluate invalid against external value
        rule = GreaterRule(instance=ref_object, attribute_name='i')
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid GreaterRule evaluation"
        # Evaluate valid against external value array, inclusive = False
        rule = GreaterRule(instance=ref_object, attribute_name='a_l', inclusive=False)
        rule.validate(test_object, 'a_l', test_object.a_l)        
        assert rule.isValid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_message)
        # Evaluate invalid against external value array, inclusive=False
        rule = GreaterRule(instance=test_object, attribute_name='a_l', inclusive=False)
        rule.validate(ref_object, 'a_l', ref_object.a_l)
        assert rule.isValid == True, "Invalid GreaterRule evaluation"
        # Evaluate valid against external value array, inclusive = True
        rule = GreaterRule(instance=ref_object, attribute_name='a_le', inclusive=True)
        rule.validate(test_object, 'a_g', test_object.a_g)
        assert rule.isValid == False, "Invalid GreaterRule evaluation"
        print(rule.invalid_message)
        # Evaluate invalid against external value array, inclusive=True
        rule = GreaterRule(instance=ref_object, attribute_name='a_g', inclusive=True)
        rule.validate(test_object, 'a_g', test_object.a_g)
        assert rule.isValid == True, "Invalid GreaterRule evaluation"                

    @mark.semantic_rules
    @mark.semantic_rules_between
    def test_syntactic_rule_betweenrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = BetweenRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = BetweenRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = BetweenRule(instance=ref_object, attribute_name='l')
            rule.validate(test_object, 'i', test_object.i)                        

        # Ranges
        r1 = [0,10]
        r2 = [0,1]
        r3 = [100,1000]

        # Integers
        # Evaluate valid basic type, inclusive=False
        rule = BetweenRule(value=r1, inclusive=False)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid basic type, inclusive=False
        rule = BetweenRule(value=r2, inclusive=False)
        rule.validate(test_object, 'i', test_object.i)
        assert rule.isValid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid array-like, inclusive=True
        rule = BetweenRule(value=r1, inclusive=True)
        rule.validate(test_object, 'a_g', test_object.a_g)
        assert rule.isValid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid array-like, inclusive=False
        rule = BetweenRule(value=r3, inclusive=True)
        rule.validate(test_object, 'a_g', test_object.a_g)
        assert rule.isValid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_message)

        # Float
        # Evaluate valid basic type, inclusive=False
        rule = BetweenRule(value=r1, inclusive=False)
        rule.validate(test_object, 'f', test_object.f)
        assert rule.isValid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid basic type, inclusive=False
        rule = BetweenRule(value=r2, inclusive=False)
        rule.validate(test_object, 'f', test_object.f)
        assert rule.isValid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_message)
        # Evaluate valid array-like, inclusive=True
        rule = BetweenRule(value=r1, inclusive=True)
        rule.validate(test_object, 'na_f', test_object.na_f)
        assert rule.isValid == True, "Invalid BetweenRule evaluation"        
        # Evaluate invalid array-like, inclusive=False
        rule = BetweenRule(value=r3, inclusive=True)
        rule.validate(test_object, 'na_f', test_object.na_f)
        assert rule.isValid == False, "Invalid BetweenRule evaluation"
        print(rule.invalid_message)

    @mark.semantic_rules
    @mark.semantic_rules_regex
    def test_syntactic_rule_regexrule(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        # Validate rule
        with pytest.raises(AttributeError):     
            rule = RegexRule(instance=ref_object, attribute_name='x')
            rule.validate(test_object, 'i', test_object.i)    
        with pytest.raises(AttributeError):     
            rule = RegexRule(instance=ref_object, attribute_name='i')
            rule.validate(test_object, 'x', test_object.x)            
        with pytest.raises(AttributeError):                 
            rule = RegexRule(instance=ref_object, attribute_name='l')
            rule.validate(test_object, 'i', test_object.i)              
        with pytest.raises(TypeError):                 
            rule = RegexRule(value=9)
            rule.validate(test_object, 'i', test_object.i)             
        regex = "[a-zA-Z]+"
        # Evaluate valid basic type
        rule = RegexRule(value=regex)
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == True, "Invalid BetweenRule evaluation"   
        # Evaluate valid array-like 
        rule = RegexRule(value=regex)
        rule.validate(test_object, 'a_s', test_object.a_s)
        assert rule.isValid == True, "Invalid BetweenRule evaluation"                       
        regex = "[0-9]+"
        # Evaluate invalid basic type
        rule = RegexRule(value=regex)
        rule.validate(test_object, 's', test_object.s)
        assert rule.isValid == False, "Invalid BetweenRule evaluation"   
        print(rule.invalid_message)
        # Evaluate invalid array-like 
        rule = RegexRule(value=regex)
        rule.validate(test_object, 'a_s', test_object.a_s)
        assert rule.isValid == False, "Invalid BetweenRule evaluation"                               
        print(rule.invalid_message)

        