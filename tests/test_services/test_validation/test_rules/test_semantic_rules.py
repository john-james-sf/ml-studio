# =========================================================================== #
#                      SEMANTIC VALIDATION RULES                              #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_semantic_rules.py                                               #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 20th 2019, 10:06:34 pm                         #
# Last Modified: Friday December 20th 2019, 10:06:42 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Tests semantic validation rules.

Test validation rules that evaluate attributes within the context of other
values and/or attributes.

These validation rules include:

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

from ml_studio.services.validation.rules import EqualRule, NotEqualRule, AllowedRule
from ml_studio.services.validation.rules import DisAllowedRule, LessRule, GreaterRule
from ml_studio.services.validation.rules import RegexRule

class SemanticRuleTests:

    @mark.semantic_rules
    @mark.semantic_rules_equalrule
    def test_semantic_equalrule(self, get_validation_rule_test_object,
                                      get_validation_rule_reference_object):
        test_object = get_validation_rule_test_object
        reference_object = get_validation_rule_reference_object
        # Validate against val parameter, success
        validation_rule = EqualRule(val=5)
        validation_rule.validate(test_object, 'i', 5)
        assert validation_rule.isValid == True, "EqualRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "EqualRule incorrectly produced invalid message"
        # Validate against val parameter, fail
        validation_rule = EqualRule(val=5)
        validation_rule.validate(test_object, 'i', 'x')
        assert validation_rule.isValid == False, "EqualRule validation incorrectly succeeded."
        assert validation_rule.invalid_message is not None, "EqualRule incorrectly failed to produce invalid message"
        # Validate against reference object, success
        validation_rule = EqualRule(instance=reference_object, attribute_name='b')
        validation_rule.validate(test_object, 'b', False)
        assert validation_rule.isValid == True, "EqualRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "EqualRule incorrectly produced invalid message"
        # Validate against reference object, fail
        validation_rule = EqualRule(instance=reference_object, attribute_name='b')
        validation_rule.validate(test_object, 'b', True)
        assert validation_rule.isValid == False, "EqualRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "EqualRule incorrectly failed to produced invalid message"
        # ------------------------------------------------------------------- #
        # Validate against val parameter, when condition met success
        validation_rule = EqualRule(val=5).when('i', EqualRule, reference_object)
        validation_rule.validate(test_object, 'i', 5)
        assert validation_rule.isValid == True, "EqualRule validation incorrectly failed."
        assert validation_rule.invalid_message is None, "EqualRule incorrectly produced invalid message"
        # Validate against val parameter, when condition met fail
        validation_rule = EqualRule(val=5)
        validation_rule.validate(test_object, 'i', 'x')
        assert validation_rule.isValid == False, "EqualRule validation incorrectly succeeded."
        assert validation_rule.invalid_message is not None, "EqualRule incorrectly failed to produce invalid message"
        # Validate against reference object, when condition met success
        validation_rule = EqualRule(instance=reference_object, attribute_name='b')
        validation_rule.validate(test_object, 'b', False)
        assert validation_rule.isValid == True, "EqualRule validation incorrectly failed."        
        assert validation_rule.invalid_message is None, "EqualRule incorrectly produced invalid message"
        # Validate against reference object, when condition met fail
        validation_rule = EqualRule(instance=reference_object, attribute_name='b')
        validation_rule.validate(test_object, 'b', True)
        assert validation_rule.isValid == False, "EqualRule validation incorrectly succeeded."        
        assert validation_rule.invalid_message is not None, "EqualRule incorrectly failed to produced invalid message"
      