# =========================================================================== #
#                      SYNTACTIC VALIDATION RULES                             #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_syntactic_rules.py                                              #
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
"""Tests syntactic validation rules.

Test validation rules that evaluate type and whether an attribute is empty,
or None. These validation rules include:

    Syntactic Rules
    --------------- 
    * BaseRule : The abstract base class for rule classes.
    * NoneRule : Ensures that a specific property is None.
    * NotNoneRule : Ensures that a specific property is not None.
    * EmptyRule : Ensures that a specific property is None, empty or whitespace.
    * NotEmptyRule : Ensures that a specific property is not None, the
        empty string or whitespace.
    * BoolRule : Ensures that the value of a specific property is a Boolean.
    * IntRule : Ensures that the value of a specific property is an integer.
    * FloatRule : Ensures that the value of a specific property is a float.
    * NumberRule : Ensures that the value of a specific property is a number.
    * StringRule : Ensures that the value of a specific property is a string.   

"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.services.validation.rules import NoneRule, NotNoneRule, EmptyRule
from ml_studio.services.validation.rules import NotEmptyRule, BoolRule, FloatRule
from ml_studio.services.validation.rules import IntRule, NumberRule, StringRule

from ml_studio.services.validation.conditions import isNone, isEmpty, isBool, isInt
from ml_studio.services.validation.conditions import isFloat, isNumber, isString
from ml_studio.services.validation.conditions import isDate

from ml_studio.services.validation.conditions import isAllNone, isAllEmpty, isAllBool, isAllInt
from ml_studio.services.validation.conditions import isAllFloat, isAllNumber, isAllString
from ml_studio.services.validation.conditions import isAllDate

class SyntacticRuleTests:

    @mark.syntactic_rules
    @mark.syntactic_rules_nonerule
    def test_syntactic_nonerule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = NoneRule()
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == True, "Nonetype validation incorrectly failed."
        assert validation_rule.invalid_message is None, "Nonetype incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', 2)
        assert validation_rule.isValid == False, "Nonetype validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "Nonetype incorrectly failed to produced invalid message"
        
    @mark.syntactic_rules
    @mark.syntactic_rules_notnonerule
    def test_syntactic_notnonerule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = NotNoneRule()
        validation_rule.validate(test_object, 'n', 'x')
        assert validation_rule.isValid == True, "NotNoneRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "NotNoneRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == False, "NotNoneRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NotNoneRule incorrectly failed to produced invalid message"
        
    @mark.syntactic_rules
    @mark.syntactic_rules_empty_rule
    def test_syntactic_empty_rule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = EmptyRule()
        validation_rule.validate(test_object, 'n', ' ')
        assert validation_rule.isValid == True, "EmptyRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "EmptyRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', 'x')
        assert validation_rule.isValid == False, "EmptyRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "EmptyRule incorrectly failed to produced invalid message"        

    @mark.syntactic_rules
    @mark.syntactic_rules_NotEmpty_rule
    def test_syntactic_NotEmpty_rule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = NotEmptyRule()
        validation_rule.validate(test_object, 'n', 'x')
        assert validation_rule.isValid == True, "NotEmptyRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "NotEmptyRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == False, "NotEmptyRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NotEmptyRule incorrectly failed to produced invalid message"        

    @mark.syntactic_rules
    @mark.syntactic_rules_Bool_rule
    def test_syntactic_Bool_rule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = BoolRule()
        validation_rule.validate(test_object, 'n', True)
        assert validation_rule.isValid == True, "BoolRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "BoolRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', 'x')
        assert validation_rule.isValid == False, "BoolRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "BoolRule incorrectly failed to produced invalid message"        

    @mark.syntactic_rules
    @mark.syntactic_rules_Int_rule
    def test_syntactic_Int_rule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = IntRule()
        validation_rule.validate(test_object, 'n', 2)
        assert validation_rule.isValid == True, "IntRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "IntRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', '2.2')
        assert validation_rule.isValid == False, "IntRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "IntRule incorrectly failed to produced invalid message"        

    @mark.syntactic_rules
    @mark.syntactic_rules_Float_rule
    def test_syntactic_Float_rule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = FloatRule()
        validation_rule.validate(test_object, 'n', 2.0)
        assert validation_rule.isValid == True, "FloatRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "FloatRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', '2.2')
        assert validation_rule.isValid == False, "FloatRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "FloatRule incorrectly failed to produced invalid message"  

    @mark.syntactic_rules
    @mark.syntactic_rules_Number_rule
    def test_syntactic_Number_rule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = NumberRule()
        validation_rule.validate(test_object, 'n', 2)
        assert validation_rule.isValid == True, "NumberRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "NumberRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', '2')
        assert validation_rule.isValid == False, "NumberRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NumberRule incorrectly failed to produced invalid message"          

    @mark.syntactic_rules
    @mark.syntactic_rules_String_rule
    def test_syntactic_String_rule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        validation_rule = StringRule()
        validation_rule.validate(test_object, 'n', '2')
        assert validation_rule.isValid == True, "StringRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "StringRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', True)
        assert validation_rule.isValid == False, "StringRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "StringRule incorrectly failed to produced invalid message"                  

class SyntacticConditionTests:

    # Syntactic Conditions
    # --------------------
    # * isNone : Evaluates whether the argument is None.
    # * isEmpty : Evaluates whether the argument is empty string or whitespace.
    # * isBool : Evaluates whether the argument is a Boolean.
    # * isInt : Evaluates whether the argument is an integer.
    # * isFloat : Evaluates whether the argument is an float.
    # * isNumber : Evaluates whether the argument is a number.
    # * isString : Evaluates whether the argument is a string. 
    # * isDate : Evaluates whether a string is a valid datetime format.
    

    # Semantic Conditions
    # -------------------
    # * isEqual : Evaluates whether two arguments are equal  
    # * isIn : Evaluates whether argument a is in argument b.
    # * isLess : Evaluates whether argument a is less than argument b.
    # * isGreater : Evaluates whether argument a is greater than argument b.
    # * isBetween : Evaluates whether argument a is between min and maa. 
    # * isMatch : Evaluates whether a string matches a regea pattern.    
    # ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isNone    
    @mark.syntactic_rules_condition_isNone_when    
    def test_syntactic_when_condition_isNone(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isNone
        when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isNone=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isNone=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isNone
        when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isNone=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isNone=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isNone
        when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isNone=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isNone=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isNone
        when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isNone=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isNone=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isNone    
    @mark.syntactic_rules_condition_isNone_except_when_except    
    def test_syntactic_except_when_condition_isNone(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isNone
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isNone=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isNone=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isNone
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isNone=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isNone=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isNone
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isNone=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isNone=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isNone
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isNone=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isNone=fail, Message failed"    


    # ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isEmpty    
    @mark.syntactic_rules_condition_isEmpty_when    
    def test_syntactic_when_condition_isEmpty(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isEmpty
        when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isEmpty=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isEmpty=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isEmpty
        when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isEmpty=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isEmpty=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isEmpty
        when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isEmpty=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isEmpty=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isEmpty
        when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isEmpty=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isEmpty=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isEmpty    
    @mark.syntactic_rules_condition_isEmpty_except_when_except    
    def test_syntactic_except_when_condition_isEmpty(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isEmpty
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isEmpty=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isEmpty=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isEmpty
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='n',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isEmpty=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isEmpty=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isEmpty
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isEmpty=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isEmpty=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isEmpty
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isEmpty=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isEmpty=fail, Message failed"            

    # ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isBool    
    @mark.syntactic_rules_condition_isBool_when    
    def test_syntactic_when_condition_isBool(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isBool
        when_a_dict = dict(instance=test_object,
                        attribute_name='b',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isBool=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isBool=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isBool
        when_a_dict = dict(instance=test_object,
                        attribute_name='b',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isBool=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isBool=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isBool
        when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isBool=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isBool=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isBool
        when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isBool=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isBool=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isBool    
    @mark.syntactic_rules_condition_isBool_except_when_except    
    def test_syntactic_except_when_condition_isBool(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isBool
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='b',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isBool=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isBool=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isBool
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='b',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isBool=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isBool=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isBool
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isBool=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isBool=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isBool
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i',
                        value=None)
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isBool=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isBool=fail, Message failed"               

# ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isInt    
    @mark.syntactic_rules_condition_isInt_when    
    def test_syntactic_when_condition_isInt(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isInt
        when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isInt=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isInt=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isInt
        when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isInt=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isInt=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isInt
        when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isInt=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isInt=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isInt
        when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isInt=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isInt=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isInt    
    @mark.syntactic_rules_condition_isInt_except_when_except    
    def test_syntactic_except_when_condition_isInt(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isInt
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isInt=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isInt=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isInt
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isInt=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isInt=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isInt
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isInt=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isInt=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isInt
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isInt=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isInt=fail, Message failed"                       

# ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isFloat    
    @mark.syntactic_rules_condition_isFloat_when    
    def test_syntactic_when_condition_isFloat(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isFloat
        when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isFloat=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isFloat=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isFloat
        when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isFloat=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isFloat=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isFloat
        when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isFloat=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isFloat=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isFloat
        when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isFloat=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isFloat=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isFloat    
    @mark.syntactic_rules_condition_isFloat_except_when_except    
    def test_syntactic_except_when_condition_isFloat(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isFloat
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isFloat=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isFloat=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isFloat
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isFloat=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isFloat=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isFloat
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isFloat=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isFloat=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isFloat
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isFloat=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isFloat=fail, Message failed"                               

# ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isNumber    
    @mark.syntactic_rules_condition_isNumber_when    
    def test_syntactic_when_condition_isNumber(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isNumber
        when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isNumber=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isNumber=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isNumber
        when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isNumber=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isNumber=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isNumber
        when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isNumber=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isNumber=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isNumber
        when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isNumber=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isNumber=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isNumber    
    @mark.syntactic_rules_condition_isNumber_except_when_except    
    def test_syntactic_except_when_condition_isNumber(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isNumber
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isNumber=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isNumber=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isNumber
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='f')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isNumber=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isNumber=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isNumber
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isNumber=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isNumber=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isNumber
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isNumber=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isNumber=fail, Message failed"                               
                

# ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isString    
    @mark.syntactic_rules_condition_isString_when    
    def test_syntactic_when_condition_isString(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isString
        when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isString=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isString=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isString
        when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isString=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isString=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isString
        when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isString=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isString=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isString
        when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isString=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isString=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isString    
    @mark.syntactic_rules_condition_isString_except_when_except    
    def test_syntactic_except_when_condition_isString(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isString
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isString=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isString=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isString
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isString=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isString=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isString
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isString=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isString=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isString
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='i')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isString=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isString=fail, Message failed"                               

# ======================================================================= #    
    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isDate    
    @mark.syntactic_rules_condition_isDate_when    
    def test_syntactic_when_condition_isDate(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        when_func = isDate
        when_a_dict = dict(instance=test_object,
                        attribute_name='d')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isDate=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isDate=pass, Message failed"

        # Setup passing condition, failing rule
        when_func = isDate
        when_a_dict = dict(instance=test_object,
                        attribute_name='d')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=when, NumberRule=fail, isDate=pass, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=when, NumberRule=fail, isDate=pass, Message failed"

        # Setup failing condition, passing rule
        when_func = isDate
        when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=when, NumberRule=pass, isDate=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=pass, isDate=fail, Message failed"

        # Setup failing condition, failing rule
        when_func = isDate
        when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        when_dict=dict(a_dict=when_a_dict)    
        validation_rule = NumberRule().when(when_func, **when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=when, NumberRule=fail, isDate=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=when, NumberRule=fail, isDate=fail, Message failed"

    # ----------------------------------------------------------------------- #    

    @mark.syntactic_rules_condition
    @mark.syntactic_rules_condition_isDate    
    @mark.syntactic_rules_condition_isDate_except_when_except    
    def test_syntactic_except_when_condition_isDate(self, get_validation_rule_test_object,
                                     get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object        
        # Setup passing condition, passing rule
        except_when_func = isDate
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='d')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isDate=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isDate=pass, Message failed"

        # Setup passing condition, failing rule
        except_when_func = isDate
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='d')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=fail, isDate=pass, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=fail, isDate=pass, Message failed"

        # Setup failing condition, passing rule
        except_when_func = isDate
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', 2)
        assert validation_rule.isValid == True, "Condition=except_when, NumberRule=pass, isDate=fail, Validity failed"
        assert validation_rule.invalid_message is None, "Condition=except_when, NumberRule=pass, isDate=fail, Message failed"

        # Setup failing condition, failing rule
        except_when_func = isDate
        except_when_a_dict = dict(instance=test_object,
                        attribute_name='s')
        except_when_dict=dict(a_dict=except_when_a_dict)    
        validation_rule = NumberRule().except_when(except_when_func, **except_when_dict)
        validation_rule.validate(test_object, 'i', '2')
        assert validation_rule.isValid == False, "Condition=except_when, NumberRule=fail, isDate=fail, Validity failed"
        assert validation_rule.invalid_message is not None, "Condition=except_when, NumberRule=fail, isDate=fail, Message failed"                               
                                                                