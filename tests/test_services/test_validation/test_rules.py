#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_rules.py                                                   #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 28th 2019, 8:04:47 pm                        #
# Last Modified: Sunday December 29th 2019, 8:01:34 pm                        #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
from collections import OrderedDict 
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.services.validation.rules import Rule, RuleSet

class SyntacticRuleTests:

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_none    
    def test_validation_rules_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('a_n')\
                          .is_none\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of none rule"
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('a_xn')\
                          .is_none\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of none rule"

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_not_none
    def test_validation_rules_not_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('a_n')\
                          .is_not_none\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not none rule"
        # Evaluates to false
        answer = rule.on(test_object)\
                          .when('a_xn')\
                          .is_not_none\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not none rule"

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_empty    
    def test_validation_rules_empty(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('e')\
                          .is_empty\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of empty rule"
        # Evaluates to false
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_empty\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of empty rule"

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_not_empty
    def test_validation_rules_not_empty(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('e')\
                          .is_not_empty\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not empty rule"
        # Evaluates to false
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_not_empty\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not empty rule"

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_bool    
    def test_validation_rules_bool(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('b')\
                          .is_bool\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of bool rule"
        # Evaluates to false
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_bool\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of bool rule"        

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_integer    
    def test_validation_rules_integer(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_integer\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of integer rule"
        # Evaluates to false
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_integer\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of integer rule"                

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_number    
    def test_validation_rules_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('f')\
                          .is_number\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of number rule"
        # Evaluates to false
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_number\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of number rule"                        

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_syntactic
    @mark.validation_rules_string    
    def test_validation_rules_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_string\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of string rule"
        # Evaluates to false
        answer = rule.on(test_object)\
                          .when('a_l')\
                          .is_string\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of string rule"                                

class SemanticRuleTests:

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_equal_number
    def test_validation_rules_equal_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_equal(5)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('a_g')\
                          .is_equal(50)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('f')\
                          .is_equal('i')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal rule"   


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_equal_string
    def test_validation_rules_equal_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_equal("hats")\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_equal("fott")\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_equal('a_s')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_equal('a_sg')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal rule"   


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_not_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_not_equal_number
    def test_validation_rules_not_equal_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_not_equal(6)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_not_equal(5)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('f')\
                          .is_not_equal('i')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_ge')\
                          .is_not_equal('a_g')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal rule"                        


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_not_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_not_equal_string
    def test_validation_rules_not_equal_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_not_equal("disc")\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_not_equal('hats')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_not_equal('a_sg')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_not_equal('a_s')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal rule"                        


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_less
    @mark.validation_rules_semantic
    @mark.validation_rules_less_numbers
    def test_validation_rules_less_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_less(6)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('a_g')\
                          .is_less(4)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('f')\
                          .is_less('i')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_less('f')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less rule"        


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_less
    @mark.validation_rules_semantic
    @mark.validation_rules_less_strings
    def test_validation_rules_less_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_less('z')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_less('a')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_less('z')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_less('a_s')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less rule"        



    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_less_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_less_equal_numbers
    def test_validation_rules_less_equal_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_less_equal(5)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less_equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('a_g')\
                          .is_less_equal(7)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less_equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_g')\
                          .is_less_equal('a_ge')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less_equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_ge')\
                          .is_less_equal('a_le')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less_equal rule"           


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_less_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_less_equal_strings
    def test_validation_rules_less_equal_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_less_equal('hats')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less_equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_less_equal('a')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of less_equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_less_equal('a_s')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of less_equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_less_equal('s')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of less_equal rule"           


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_greater
    @mark.validation_rules_semantic
    @mark.validation_rules_greater_numbers
    def test_validation_rules_greater_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_greater(3)\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('a_g')\
                          .is_greater(3)\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_greater('f')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('f')\
                          .is_greater('i')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater rule"        


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_greater
    @mark.validation_rules_semantic
    @mark.validation_rules_greater_strings
    def test_validation_rules_greater_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_greater('a')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_greater('z')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_sg')\
                          .is_greater('a_s')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_greater('a_sg')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater rule"        


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_greater_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_greater_equal_numbers
    def test_validation_rules_greater_equal_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('i')\
                          .is_greater_equal(5)\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('a_g')\
                          .is_greater_equal(50)\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_g')\
                          .is_greater_equal('a_ge')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_le')\
                          .is_greater_equal('a_ge')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal rule"                   


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_greater_equal
    @mark.validation_rules_semantic
    @mark.validation_rules_greater_equal_strings
    def test_validation_rules_greater_equal_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_greater_equal("hats")\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('s')\
                          .is_greater_equal('z')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal rule"        
        # Evaluates to true with attribute
        answer = rule.on(test_object)\
                          .when('a_sg')\
                          .is_greater_equal('a_s')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal rule"
        # Evaluates to false with attribute
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_greater_equal('a_sg')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal rule"                   



    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_semantic
    @mark.validation_rules_match
    def test_validation_rules_match(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        rule = Rule()
        # Evaluates to true with constant
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_match("[a-zA-Z]+")\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of match rule"
        # Evaluates to false with constant
        answer = rule.on(test_object)\
                          .when('a_s')\
                          .is_match("[0-9]+")\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of match rule"        



class RuleSetTests:

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_semantic
    @mark.validation_rules_set
    @mark.validation_rules_set_and
    def test_validation_rules_set_and(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Create True 
        rule1 = Rule().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        assert rule1.evaluate.is_valid is True, "Failed Rule 1 before assigning to set."
        # Create another True rule
        rule2 = Rule().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')
        assert rule2.evaluate.is_valid is True, "Failed Rule 2 before assigning to set."

        # Create a false rule
        rule3 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        assert rule3.evaluate.is_valid is False, "Failed Rule 3 before assigning to set."                          
        # Create another false rule
        rule4 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)
        assert rule4.evaluate.is_valid is False, "Failed Rule 4 before assigning to set."
        # Create True/True rule set where all must be true                           
        rs = RuleSet()
        rs.when_all_rules_are_true
        rs.add_rule(rule1).add_rule(rule2)
        # Check evaluation
        answer = rs.evaluate.is_valid
        assert answer is True, "Invalid evaluation of rules 1 and 2"
        
        # Create True/False rule set where all must be true
        rs.remove_rule(rule2).add_rule(rule3)                
        # Check evaluation
        answer = rs.evaluate.is_valid
        assert answer is False, "Invalid evaluation of rules 1 and 3"

        # Create False/False rule where all must be true
        rs.remove_rule(rule3).add_rule(rule4)                
        # Check evaluation
        answer = rs.evaluate.is_valid
        assert answer is False, "Invalid evaluation of rules 1 and 4"        

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_semantic
    @mark.validation_rules_set
    @mark.validation_rules_set_or
    def test_validation_rules_set_or(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        
        # Create True 
        rule1 = Rule().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True rule
        rule2 = Rule().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false rule
        rule3 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false rule
        rule4 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)
        # Create True/True rule set where all must be true                           
        rs = RuleSet()
        rs.when_any_rule_is_true
        rs.add_rule(rule1)
        rs.add_rule(rule2)
        answer = rs.evaluate.is_valid
        assert answer is True, "Invalid evaluation of rules 1 and 2"
        # Create True/False rule set where all must be true
        rs.remove_rule(rule2)        
        rs.add_rule(rule3)                
        answer = rs.evaluate.is_valid
        rs.print_rule_set
        assert answer is True, "Invalid evaluation of rules 1 and 3"
        # Create False/False rule where all must be true
        rs.remove_rule(rule1)
        rs.add_rule(rule4) 
        answer = rs.evaluate.is_valid           
        rs.print_rule_set
        assert answer is False, "Invalid evaluation of rules 3 and 4"                

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_semantic
    @mark.validation_rules_set
    @mark.validation_rules_set_none
    def test_validation_rules_set_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        
        # Create True 
        rule1 = Rule().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True rule
        rule2 = Rule().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false rule
        rule3 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false rule
        rule4 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)
        # Create True/True rule set where all must be true                           
        rs = RuleSet()
        rs.when_no_rules_are_true
        rs.add_rule(rule1)
        rs.add_rule(rule2)
        answer = rs.evaluate.is_valid
        assert answer is False, "Invalid evaluation of rules 1 and 2"
        # Create True/False rule set where all must be true
        rs.remove_rule(rule2)        
        rs.add_rule(rule3)                
        answer = rs.evaluate.is_valid
        rs.print_rule_set
        assert answer is False, "Invalid evaluation of rules 1 and 3"
        # Create False/False rule where all must be true
        rs.remove_rule(rule1)
        rs.add_rule(rule4) 
        answer = rs.evaluate.is_valid           
        rs.print_rule_set
        assert answer is True, "Invalid evaluation of rules 3 and 4"            

class ChildNodeTests:

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_printing
    def test_validation_rules_printing(self, get_validation_rule_test_object):
        """Testing propagation of data down through child nodes."""
        test_object = get_validation_rule_test_object        
        # Create some rules
        # True Test Rule
        rule1 = Rule().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True rule
        rule2 = Rule().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false rule
        rule3 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false rule
        rule4 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)        
        # Add 1 and 2 to a new Rule Set
        rs1 = RuleSet()
        rs1.add_rule(rule1).add_rule(rule2)
        # Add RS1 and rules 3 and 4 to new RuleSet
        rs2 = RuleSet()
        rs2.add_rule(rs1).add_rule(rule3).add_rule(rule4)
        rs2.when_no_rules_are_true
        # Print to see how it looks in the hierarchy
        rs2.print_rule_set


    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_propagation
    def test_validation_rules_propagation(self, get_validation_rule_test_object):
        """Testing propagation of data down through child nodes."""
        test_object = get_validation_rule_test_object        
        # Create some rules
        # True Test Rule
        rule1 = Rule().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True rule
        rule2 = Rule().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false rule
        rule3 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false rule
        rule4 = Rule().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)        
        # Add 1 and 2 to a new Rule Set
        rs1 = RuleSet()
        rs1.add_rule(rule1).add_rule(rule2)
        # Add RS1 and rules 3 and 4 to new RuleSet
        rs2 = RuleSet()
        rs2.add_rule(rs1).add_rule(rule3).add_rule(rule4)
        rs2.when_no_rules_are_true
        # Set the target object and attribute
        rs2.on(test_object)
        rs2.attribute('a_g')
        # Traverse through confirming updates
        def traverse(rule):         
            assert rule._evaluated_instance == test_object, "Test object not set" 
            assert rule._evaluated_attribute == "a_g", "Attribute not set" 
            if isinstance(rule, RuleSet):
                for _,rule in rule._rules.items():      
                    return traverse(rule)
            else:
                assert rule._evaluated_instance == test_object, "Test object not set" 
                assert rule._evaluated_attribute == "a_g", "Attribute not set" 
        traverse(rs2)


class ErrorHandlingTests:

    @mark.validation
    @mark.validation_rules
    @mark.validation_rules_error_handling
    def test_validation_rules_error_handling(self, get_validation_rule_test_object):
        """Testing propagation of data down through child nodes."""
        test_object = get_validation_rule_test_object        
        # Indicate action on error
        rule = Rule().on(test_object)\
                          .when('i')\
                          .is_equal(5)\
                          .on_fail_report_error

        assert rule._action_on_fail == "report", "Action assignment failed"
        # Report Error Message ForRule
        rule.error_message = "Dangerous for i to be 5."
        assert "Dangerous" in rule._error_message, "Error message assignment didn't work"
