# =========================================================================== #
#                      SEMANTIC VALIDATION RULES                             #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_semantic_rules.py                                              #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 20th 2019, 7:12:34 pm                          #
# Last Modified: Friday December 20th 2019, 7:13:16 pm                        #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Tests semantic validation rules.

Test validation rules that evaluate type and whether an attribute is empty,
or None. These validation rules include:

    Semantic Rules
    --------------
    * SemanticRule : Base class for semantic rules.
    * EqualRule : Ensures that the value of a specific property is
        equal to a particular value or the value of another property.        
    * NotEqualRule : Ensures that the value of a specific property is
        not equal to a particular value or the value of another property.
    * AllowedRule :Ensures the value of a specific property is one of a 
        discrete set of allowed values. 
    * DisAllowedRule :Ensures the value of a specific property is none of a 
        discrete set of disallowed values.     
    * LessRule : Ensures the value of a specific property is less than
        a partiulcar value or less than the value of another property.
    * GreaterRule : Ensures the value of a specific property is greater
        than a particulcar value or greater than the value of another property.        
    * RegexRule : Ensures the value of a specific property matches
        the given regular expression(s).   
      

"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.services.validation.rules import EqualRule, NotEqualRule
from ml_studio.services.validation.rules import AllowedRule, DisAllowedRule
from ml_studio.services.validation.rules import LessRule, GreaterRule
from ml_studio.services.validation.rules import RegexRule

from ml_studio.services.validation.conditions import isEqual, isIn, isLess
from ml_studio.services.validation.conditions import isGreater, isMatch

class SemanticRuleTests:

    @mark.semantic_rules
    @mark.semantic_rules_equal_rule
    def test_semantic_rules_equal_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = EqualRule(instance=ref_object, attribute_name='i')
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "EqualRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "EqualRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == False, "EqualRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "EqualRule=fail no conditions, validation message failed"
        
    @mark.semantic_rules
    @mark.semantic_rules_not_equal_rule
    def test_semantic_rules_not_equal_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = NotEqualRule(instance=ref_object, attribute_name='i')
        validation_rule.validate(test_object, 'i', 3)
        assert validation_rule.isValid == True, "NotEqualRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "NotEqualRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == False, "NotEqualRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "NotEqualRule=fail no conditions, validation message failed"

    @mark.semantic_rules
    @mark.semantic_rules_allowed_rule
    def test_semantic_rules_allowed_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = AllowedRule(instance=ref_object, attribute_name='a_i')
        validation_rule.validate(test_object, 'i', 9)
        assert validation_rule.isValid == True, "AllowedRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "AllowedRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'i', 10)
        assert validation_rule.isValid == False, "AllowedRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "AllowedRule=fail no conditions, validation message failed"

    @mark.semantic_rules
    @mark.semantic_rules_disallowed_rule
    def test_semantic_rules_disallowed_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = DisAllowedRule(instance=ref_object, attribute_name='a_i')
        validation_rule.validate(test_object, 'i', 10)
        assert validation_rule.isValid == True, "DisallowedRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "DisallowedRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'i', 9)
        assert validation_rule.isValid == False, "DisallowedRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "DisallowedRule=fail no conditions, validation message failed"

    @mark.semantic_rules
    @mark.semantic_rules_less_rule
    def test_semantic_rules_less_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = LessRule(instance=ref_object, attribute_name='f')
        validation_rule.validate(test_object, 'i', 7)
        assert validation_rule.isValid == True, "LessRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "LessRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'i', 10)
        assert validation_rule.isValid == False, "LessRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "LessRule=fail no conditions, validation message failed"        

    @mark.semantic_rules
    @mark.semantic_rules_less_equal_rule
    def test_semantic_rules_less_equal_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = LessRule(instance=ref_object, attribute_name='f', equal_ok=True)
        validation_rule.validate(test_object, 'i', 9.3)
        assert validation_rule.isValid == True, "Less_equalRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "Less_equalRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'i', 10)
        assert validation_rule.isValid == False, "Less_equalRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "Less_equalRule=fail no conditions, validation message failed"                

    @mark.semantic_rules
    @mark.semantic_rules_greater_rule
    def test_semantic_rules_greater_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = GreaterRule(instance=ref_object, attribute_name='f')
        validation_rule.validate(test_object, 'i', 10)
        assert validation_rule.isValid == True, "GreaterRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "GreaterRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'i', 5)
        assert validation_rule.isValid == False, "GreaterRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "GreaterRule=fail no conditions, validation message failed"        

    @mark.semantic_rules
    @mark.semantic_rules_greater_equal_rule
    def test_semantic_rules_greater_equal_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = GreaterRule(instance=ref_object, attribute_name='f', equal_ok=True)
        validation_rule.validate(test_object, 'i', 9.3)
        assert validation_rule.isValid == True, "Greater_equalRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "Greater_equalRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 'i', 5)
        assert validation_rule.isValid == False, "Greater_equalRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "Greater_equalRule=fail no conditions, validation message failed"              

    @mark.semantic_rules
    @mark.semantic_rules_regex_rule
    def test_semantic_rules_regex_rule(self, get_validation_rule_test_object,
                               get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        ref_object = get_validation_rule_reference_object
        validation_rule = RegexRule(instance=ref_object, attribute_name='s')
        validation_rule.validate(test_object, 's', "hats")
        assert validation_rule.isValid == True, "RegexRule=pass, no conditions validation failed"
        assert validation_rule.invalid_message is None, "RegexRule=pass, no conditions validation message failed"
        validation_rule.validate(test_object, 's', "shoes")
        assert validation_rule.isValid == False, "RegexRule=fail no conditions, validation failed."        
        assert validation_rule.invalid_message is not None, "RegexRule=fail no conditions, validation message failed"                      

    # Semantic Conditions
    # -------------------
    # * isEqual : Evaluates whether two arguments are equal  
    # * isIn : Evaluates whether argument a is in argument b.
    # * isLess : Evaluates whether argument a is less than argument b.
    # * isGreater : Evaluates whether argument a is greater than argument b.
    # * isMatch : Evaluates whether a string matches a regea pattern.     
    
    class SemanticRuleConditionTests:

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_when_condition_isequal
        def test_semantic_rule_condition_isequal(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup passing when condition
            a = dict(instance=ref_object, attribute_name='s')
            b = dict(instance=test_object, attribute_name='s')
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isEqual, a,b)
            validation_rule.validate(test_object, 'i', 2)
            assert validation_rule.isValid == True, "When Condition=isEqual, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isEqual, pass, validation message failed"
            # Setup failing when condition
            a = dict(instance=ref_object, attribute_name='b')
            b = dict(instance=test_object, attribute_name='s')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isEqual, a,b)
            validation_rule.validate(test_object, 'n', 5)
            assert validation_rule.isValid == True, "When Condition=isEqual, fail, validation failed."        
            assert validation_rule.invalid_message is None, "Condition=isEqual, fail no conditions, validation message failed"

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_except_when_condition_isequal
        def test_semantic_rule_except_when_condition_isequal(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object            
            # Setup passing except when condition
            a = dict(instance=ref_object, attribute_name='s')
            b = dict(instance=test_object, attribute_name='s')
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isEqual, a,b)
            validation_rule.validate(test_object, 'i', 2)
            assert validation_rule.isValid == True, "Except When Condition=isEqual, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isEqual, pass, validation message failed"
            # Setup failing except when condition
            a = dict(instance=ref_object, attribute_name='b')
            b = dict(instance=test_object, attribute_name='s')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isEqual, a,b)
            validation_rule.validate(test_object, 'n', 5)
            assert validation_rule.isValid == False, "Except When Condition=isEqual, fail, validation failed."        
            assert validation_rule.invalid_message is not None, "Condition=isEqual, fail no conditions, validation message failed"

        
        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_when_condition_isin
        def test_semantic_rule_condition_isin(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup when passing condition
            a = dict(instance=ref_object, attribute_name='s')
            b = ['hats', 'coat', 'shoes']
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isIn, a,b)
            validation_rule.validate(test_object, 'i', 2)
            assert validation_rule.isValid == True, "When Condition=isIn, pass validation failed"
            assert validation_rule.invalid_message is None, "When Condition=isIn, pass, validation message failed"
            # Setup when failing condition
            a = dict(instance=ref_object, attribute_name='b')
            b = dict(instance=test_object, attribute_name='s')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isIn, a,b)
            validation_rule.validate(test_object, 'n', 5)
            assert validation_rule.isValid == True, "When Condition=isIn, fail, validation failed."        
            assert validation_rule.invalid_message is None, "When Condition=isIn, fail no conditions, validation message failed"

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_except_when_condition_isin
        def test_semantic_rule_except_whencondition_isin(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup except when passing condition
            a = dict(instance=ref_object, attribute_name='s')
            b = ['hats', 'coat', 'shoes']
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isIn, a,b)
            validation_rule.validate(test_object, 'i', 3)
            assert validation_rule.isValid == True, "Except When Condition=isIn, pass validation failed"
            assert validation_rule.invalid_message is None, "Except When Condition=isIn, pass, validation message failed"
            # Setup except when failing condition
            a = dict(instance=ref_object, attribute_name='b')
            b = dict(instance=test_object, attribute_name='s')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isIn, a,b)
            validation_rule.validate(test_object, 'n', 5)
            assert validation_rule.isValid == False, "Except When Condition=isIn, fail, validation failed."        
            assert validation_rule.invalid_message is not None, "Except When Condition=isIn, fail no conditions, validation message failed"            

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_when_condition_isless
        def test_semantic_rule_condition_isless(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup when passing condition
            a = dict(instance=ref_object, attribute_name='i')
            b = dict(instance=test_object, attribute_name='i')
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isLess, a,b)
            validation_rule.validate(test_object, 'i', 2)
            assert validation_rule.isValid == True, "When Condition=isLess, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isLess, pass, validation message failed"
            # Setup when failing condition
            a = dict(instance=ref_object, attribute_name='f')
            b = dict(instance=test_object, attribute_name='f')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isLess, a,b)
            validation_rule.validate(test_object, 'n', 5)
            assert validation_rule.isValid == True, "When Condition=isLess, fail, validation failed."        
            assert validation_rule.invalid_message is None, "Condition=isLess, fail no conditions, validation message failed"

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_except_when_condition_isless
        def test_semantic_rule_except_when_condition_isless(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup except when passing condition
            a = dict(instance=ref_object, attribute_name='i')
            b = dict(instance=test_object, attribute_name='i')
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isLess, a,b)
            validation_rule.validate(test_object, 'i', 3)
            assert validation_rule.isValid == True, "When Condition=isLess, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isLess, pass, validation message failed"
            # Setup except when failing condition
            a = dict(instance=ref_object, attribute_name='f')
            b = dict(instance=test_object, attribute_name='f')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isLess, a,b)
            validation_rule.validate(test_object, 'n',2)
            assert validation_rule.isValid == True, "When Condition=isLess, fail, validation failed."        
            assert validation_rule.invalid_message is None, "Condition=isLess, fail no conditions, validation message failed"

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_when_condition_isgreater
        def test_semantic_rule_condition_isgreater(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup when passing condition
            b = dict(instance=ref_object, attribute_name='i')
            a = dict(instance=test_object, attribute_name='i')
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isGreater, a,b)
            validation_rule.validate(test_object, 'i', 2)
            assert validation_rule.isValid == True, "When Condition=isGreater, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isGreater, pass, validation message failed"
            # Setup when failing condition
            b = dict(instance=ref_object, attribute_name='f')
            a = dict(instance=test_object, attribute_name='f')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isGreater, a,b)
            validation_rule.validate(test_object, 'n', 5)
            assert validation_rule.isValid == True, "When Condition=isGreater, fail, validation failed."        
            assert validation_rule.invalid_message is None, "Condition=isGreater, fail no conditions, validation message failed"

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_except_when_condition_isgreater
        def test_semantic_rule_except_when_condition_isgreater(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup except when passing condition
            b = dict(instance=ref_object, attribute_name='i')
            a = dict(instance=test_object, attribute_name='i')
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isGreater, a,b)
            validation_rule.validate(test_object, 'i', 3)
            assert validation_rule.isValid == True, "When Condition=isGreater, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isGreater, pass, validation message failed"
            # Setup except when failing condition
            b = dict(instance=ref_object, attribute_name='f')
            a = dict(instance=test_object, attribute_name='f')            
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isGreater, a,b)
            validation_rule.validate(test_object, 'n',2)
            assert validation_rule.isValid == True, "When Condition=isGreater, fail, validation failed."        
            assert validation_rule.invalid_message is None, "Condition=isGreater, fail no conditions, validation message failed"

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_when_condition_ismatch
        def test_semantic_rule_condition_ismatch(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup when passing condition
            a = dict(instance=ref_object, attribute_name='s')
            b = dict(instance=test_object, attribute_name='s')
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isMatch, a,b)
            validation_rule.validate(test_object, 'i', 2)
            assert validation_rule.isValid == True, "When Condition=isMatch, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isMatch, pass, validation message failed"
            # Setup when failing condition
            a = "cigaratte"
            b = "after"
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').when(isMatch, a,b)
            validation_rule.validate(test_object, 'n', 5)
            assert validation_rule.isValid == True, "When Condition=isMatch, fail, validation failed."        
            assert validation_rule.invalid_message is None, "Condition=isMatch, fail no conditions, validation message failed"

        @mark.semantic_rules
        @mark.semantic_rules_condition
        @mark.semantic_rules_except_when_condition_ismatch
        def test_semantic_rule_except_when_condition_ismatch(self, get_validation_rule_test_object,
                                get_validation_rule_reference_object):
            test_object = get_validation_rule_test_object        
            ref_object = get_validation_rule_reference_object
            # Setup except when passing condition
            a = "cunning linguistics"
            b = "cunning"
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isMatch, a,b)
            validation_rule.validate(test_object, 'i', 2)
            assert validation_rule.isValid == True, "When Condition=isMatch, pass validation failed"
            assert validation_rule.invalid_message is None, "Condition=isMatch, pass, validation message failed"
            # Setup except when failing condition
            a = "cunning linguistics"
            b = "smoke"
            validation_rule = EqualRule(instance=ref_object, attribute_name='i').except_when(isMatch, a,b)
            validation_rule.validate(test_object, 'n',2)
            assert validation_rule.isValid == True, "When Condition=isMatch, fail, validation failed."        
            assert validation_rule.invalid_message is None, "Condition=isMatch, fail no conditions, validation message failed"            