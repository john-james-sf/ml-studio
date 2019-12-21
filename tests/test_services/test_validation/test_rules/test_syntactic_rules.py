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
    def test_syntactic_nonerule(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        validation_rule = NoneRule()
        validation_rule.validate(test_object, 'n', None)
        assert validation_rule.isValid == True, "Nonetype validation incorrectly failed."
        assert validation_rule.
        validation_rule.validate(test_object, 'n', 2)
        assert validation_rule.isValid == False, "Nonetype validation incorrectly failed."        

