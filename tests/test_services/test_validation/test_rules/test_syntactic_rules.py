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
from ml_studio.services.validation.rules import NotEmptyRule, BoolRule, IntRule
from ml_studio.services.validation.rules import FloatRule, NumberRule, StringRule

class SyntacticRuleTests:

    @mark.syntactic_rules
    @mark.syntactic_rules_nonerule
    def test_syntactic_nonerule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = NoneRule()
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == True, "Nonetype validation incorrectly failed."
        assert validation_rule.invalid_message is None, "Nonetype incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', 2)
        assert validation_rule.isValid == False, "Nonetype validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "Nonetype incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = NoneRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == False, "Nonetype validation incorrectly suceeded."        
        assert validation_rule.invalid_message is not None, "Nonetype incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = NoneRule().when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == True, "Nonetype validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "Nonetype incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = NoneRule().except_when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == False, "Nonetype validation incorrectly suceeded."        
        assert validation_rule.invalid_message is not None, "Nonetype incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = NoneRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == True, "Nonetype validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "Nonetype incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = NoneRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == False, "Nonetype validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "Nonetype incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = NoneRule().when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == True, "Nonetype validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "Nonetype incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = NoneRule().except_when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == False, "Nonetype validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "Nonetype incorrectly failed to produced invalid message"

        # Validate with except when not met from another object
        validation_rule = NoneRule().except_when(attribute_name='b', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 5)
        assert validation_rule.isValid == True, "Nonetype validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "Nonetype incorrectly produced invalid message"

    @mark.syntactic_rules
    @mark.syntactic_rules_notnonerule
    def test_syntactic_notnonerule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = NotNoneRule()
        validation_rule.validate(test_object, 'n', 'x')
        assert validation_rule.isValid == True, "NotNoneRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "NotNoneRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == False, "NotNoneRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NotNoneRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = NotNoneRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == False, "NotNoneRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NotNoneRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = NotNoneRule().when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == True, "NotNoneRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotNoneRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = NotNoneRule().except_when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == False, "NotNoneRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NotNoneRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = NotNoneRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == True, "NotNoneRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotNoneRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = NotNoneRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == False, "NotNoneRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NotNoneRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = NotNoneRule().when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == True, "NotNoneRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotNoneRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = NotNoneRule().except_when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == False, "NotNoneRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NotNoneRule incorrectly failed to produced invalid message"

        # Validate with except when not met from another object
        validation_rule = NotNoneRule().except_when(attribute_name='b', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == True, "NotNoneRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotNoneRule incorrectly produced invalid message"        

    @mark.syntactic_rules
    @mark.syntactic_rules_emptyrule
    def test_syntactic_emptyrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = EmptyRule()
        validation_rule.validate(test_object, 'n', '')
        assert validation_rule.isValid == True, "EmptyRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "EmptyRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == False, "EmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "EmptyRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = EmptyRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == False, "EmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "EmptyRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = EmptyRule().when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == True, "EmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "EmptyRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = EmptyRule().except_when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == False, "EmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "EmptyRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = EmptyRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == True, "EmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "EmptyRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = EmptyRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == False, "EmptyRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "EmptyRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = EmptyRule().when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == True, "EmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "EmptyRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = EmptyRule().except_when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == False, "EmptyRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "EmptyRule incorrectly failed to produced invalid message"

        # Validate with except when not met from another object
        validation_rule = EmptyRule().except_when(attribute_name='b', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "x")
        assert validation_rule.isValid == True, "EmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "EmptyRule incorrectly produced invalid message"                

    @mark.syntactic_rules
    @mark.syntactic_rules_notemptyrule
    def test_syntactic_notemptyrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = NotEmptyRule()
        validation_rule.validate(test_object, 'n', 'x')
        assert validation_rule.isValid == True, "NotEmptyRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "NotEmptyRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NotEmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NotEmptyRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = NotEmptyRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NotEmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NotEmptyRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = NotEmptyRule().when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "NotEmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotEmptyRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = NotEmptyRule().except_when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NotEmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NotEmptyRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = NotEmptyRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "NotEmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotEmptyRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = NotEmptyRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NotEmptyRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NotEmptyRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = NotEmptyRule().when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "NotEmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotEmptyRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = NotEmptyRule().except_when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NotEmptyRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NotEmptyRule incorrectly failed to produced invalid message"

        # Validate with except when not met from another object
        validation_rule = NotEmptyRule().except_when(attribute_name='b', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "NotEmptyRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NotEmptyRule incorrectly produced invalid message"        

    @mark.syntactic_rules
    @mark.syntactic_rules_boolrule
    def test_syntactic_boolrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = BoolRule()
        validation_rule.validate(test_object, 'n', False)
        assert validation_rule.isValid == True, "BoolRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "BoolRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "BoolRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "BoolRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = BoolRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "BoolRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "BoolRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = BoolRule().when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "BoolRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "BoolRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = BoolRule().except_when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "BoolRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "BoolRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = BoolRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "BoolRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "BoolRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = BoolRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "BoolRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "BoolRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = BoolRule().when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "BoolRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "BoolRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = BoolRule().except_when(attribute_name='i', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "BoolRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "BoolRule incorrectly failed to produced invalid message"

        # Validate with except when not met from another object
        validation_rule = BoolRule().except_when(attribute_name='b', rule=BoolRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "BoolRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "BoolRule incorrectly produced invalid message"                

    @mark.syntactic_rules
    @mark.syntactic_rules_intrule
    def test_syntactic_intrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = IntRule()
        validation_rule.validate(test_object, 'n', 2)
        assert validation_rule.isValid == True, "IntRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "IntRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "IntRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = IntRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "IntRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = IntRule().when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "IntRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = IntRule().except_when(attribute_name='i', rule=FloatRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "IntRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = IntRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "IntRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = IntRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "IntRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "IntRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = IntRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 2)
        assert validation_rule.isValid == True, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "IntRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = IntRule().except_when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "IntRule incorrectly produced invalid message"

        # Validate with except when not met from another object
        validation_rule = IntRule().except_when(attribute_name='f', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "IntRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "IntRule incorrectly produced invalid message"                   

    @mark.syntactic_rules
    @mark.syntactic_rules_floatrule
    def test_syntactic_floatrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = FloatRule()
        validation_rule.validate(test_object, 'n', 2.2)
        assert validation_rule.isValid == True, "FloatRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "FloatRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "FloatRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = FloatRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "FloatRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = FloatRule().when(attribute_name='i', rule=StringRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "FloatRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = FloatRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "FloatRule incorrectly failed to produced invalid message"
        
        # Validate with except when not met
        validation_rule = FloatRule().except_when(attribute_name='i', rule=BoolRule())
        validation_rule.validate(test_object, 'n', 2.0)
        assert validation_rule.isValid == True, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "FloatRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = FloatRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "FloatRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "FloatRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = FloatRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 2.0)
        assert validation_rule.isValid == True, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "FloatRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = FloatRule().except_when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "FloatRule incorrectly produced invalid message"

        # Validate with except when not met from another object
        validation_rule = FloatRule().except_when(attribute_name='f', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "FloatRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "FloatRule incorrectly produced invalid message"               

    @mark.syntactic_rules
    @mark.syntactic_rules_numberrule
    def test_syntactic_numberrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = NumberRule()
        validation_rule.validate(test_object, 'n', 2.2)
        assert validation_rule.isValid == True, "NumberRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "NumberRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NumberRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = NumberRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NumberRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = NumberRule().when(attribute_name='i', rule=StringRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NumberRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = NumberRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NumberRule incorrectly failed to produced invalid message"
        
        # Validate with except when not met
        validation_rule = NumberRule().except_when(attribute_name='i', rule=BoolRule())
        validation_rule.validate(test_object, 'n', 2.0)
        assert validation_rule.isValid == True, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NumberRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = NumberRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NumberRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "NumberRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = NumberRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 2.0)
        assert validation_rule.isValid == True, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NumberRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = NumberRule().except_when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "NumberRule incorrectly produced invalid message"

        # Validate with except when not met from another object
        validation_rule = NumberRule().except_when(attribute_name='f', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "NumberRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "NumberRule incorrectly produced invalid message"               

    @mark.syntactic_rules
    @mark.syntactic_rules_stringrule
    def test_syntactic_stringrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        validation_rule = StringRule()
        validation_rule.validate(test_object, 'n', "2.2")
        assert validation_rule.isValid == True, "StringRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "StringRule incorrectly produced invalid message"
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "StringRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "StringRule incorrectly failed to produced invalid message"
        
        # Validate with when met
        validation_rule = StringRule().when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "StringRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "StringRule incorrectly failed to produced invalid message"
        
        # Validate with when not met
        validation_rule = StringRule().when(attribute_name='i', rule=StringRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "StringRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "StringRule incorrectly failed to produced invalid message"

        # Validate with except when met
        validation_rule = StringRule().except_when(attribute_name='i', rule=IntRule())
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "StringRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "StringRule incorrectly failed to produced invalid message"
        
        # Validate with except when not met
        validation_rule = StringRule().except_when(attribute_name='i', rule=BoolRule())
        validation_rule.validate(test_object, 'n', "2.0")
        assert validation_rule.isValid == True, "StringRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "StringRule incorrectly failed to produced invalid message"

        # Validate with when met from another object
        validation_rule = StringRule().when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "StringRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "StringRule incorrectly failed to produced invalid message"

        # Validate with when not met from another object
        validation_rule = StringRule().when(attribute_name='i', rule=FloatRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', 2.0)
        assert validation_rule.isValid == True, "StringRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "StringRule incorrectly produced invalid message"

        # Validate with except when met from another object
        validation_rule = StringRule().except_when(attribute_name='i', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == True, "StringRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "StringRule incorrectly produced invalid message"

        # Validate with except when not met from another object
        validation_rule = StringRule().except_when(attribute_name='f', rule=IntRule(), instance=reference_object)
        validation_rule.validate(test_object, 'n', "")
        assert validation_rule.isValid == False, "StringRule validation incorrectly failed."        
        assert validation_rule.invalid_message is not None, "StringRule incorrectly produced invalid message"              