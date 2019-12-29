#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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
# Create Date: Saturday December 28th 2019, 8:04:47 pm                        #
# Last Modified: Saturday December 28th 2019, 9:47:34 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.services.validation.conditions import Condition, ConditionSet

class SyntacticConditionTests:

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_none    
    def test_validation_conditions_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'a_n')\
                          .when_value\
                          .is_none\
                          .evaluate
        assert answer is True, "Invalid evaluation of none condition"
        # Evaluates to true
        answer = condition.on(test_object, 'a_xn')\
                          .when_value\
                          .is_none\
                          .evaluate
        assert answer is False, "Invalid evaluation of none condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_not_none
    def test_validation_conditions_not_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'a_n')\
                          .when_value\
                          .is_not_none\
                          .evaluate
        assert answer is False, "Invalid evaluation of not none condition"
        # Evaluates to false
        answer = condition.on(test_object, 'a_xn')\
                          .when_value\
                          .is_not_none\
                          .evaluate
        assert answer is True, "Invalid evaluation of not none condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_empty    
    def test_validation_conditions_empty(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'e')\
                          .when_value\
                          .is_empty\
                          .evaluate
        assert answer is True, "Invalid evaluation of empty condition"
        # Evaluates to false
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_empty\
                          .evaluate
        assert answer is False, "Invalid evaluation of empty condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_not_empty
    def test_validation_conditions_not_empty(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'e')\
                          .when_value\
                          .is_not_empty\
                          .evaluate
        assert answer is False, "Invalid evaluation of not empty condition"
        # Evaluates to false
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_not_empty\
                          .evaluate
        assert answer is True, "Invalid evaluation of not empty condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_bool    
    def test_validation_conditions_bool(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'b')\
                          .when_value\
                          .is_bool\
                          .evaluate
        assert answer is True, "Invalid evaluation of bool condition"
        # Evaluates to false
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_bool\
                          .evaluate
        assert answer is False, "Invalid evaluation of bool condition"        

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_integer    
    def test_validation_conditions_integer(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_integer\
                          .evaluate
        assert answer is True, "Invalid evaluation of integer condition"
        # Evaluates to false
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_integer\
                          .evaluate
        assert answer is False, "Invalid evaluation of integer condition"                

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_number    
    def test_validation_conditions_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'f')\
                          .when_value\
                          .is_number\
                          .evaluate
        assert answer is True, "Invalid evaluation of number condition"
        # Evaluates to false
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_number\
                          .evaluate
        assert answer is False, "Invalid evaluation of number condition"                        

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_string    
    def test_validation_conditions_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_string\
                          .evaluate
        assert answer is True, "Invalid evaluation of string condition"
        # Evaluates to false
        answer = condition.on(test_object, 'a_l')\
                          .when_value\
                          .is_string\
                          .evaluate
        assert answer is False, "Invalid evaluation of string condition"                                

class SemanticConditionTests:

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_equal
    @mark.validation_conditions_equal_number
    def test_validation_conditions_equal_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_equal(5)\
                          .evaluate
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 'a_g')\
                          .when_value\
                          .is_equal(50)\
                          .evaluate
        assert answer is False, "Invalid evaluation of equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_ge')\
                          .when_value\
                          .is_equal('a_g')\
                          .evaluate
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'f')\
                          .when_value\
                          .is_equal('i')\
                          .evaluate
        assert answer is False, "Invalid evaluation of equal condition"   


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_equal
    @mark.validation_conditions_equal_string
    def test_validation_conditions_equal_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_equal("hats")\
                          .evaluate
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_equal("fott")\
                          .evaluate
        assert answer is False, "Invalid evaluation of equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_equal('a_s')\
                          .evaluate
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_equal('a_sg')\
                          .evaluate
        assert answer is False, "Invalid evaluation of equal condition"   


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_not_equal
    @mark.validation_conditions_not_equal_number
    def test_validation_conditions_not_equal_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_not_equal(6)\
                          .evaluate
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_not_equal(5)\
                          .evaluate
        assert answer is False, "Invalid evaluation of not_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'f')\
                          .when_value\
                          .is_not_equal('i')\
                          .evaluate
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_ge')\
                          .when_value\
                          .is_not_equal('a_g')\
                          .evaluate
        assert answer is False, "Invalid evaluation of not_equal condition"                        




    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_not_equal
    @mark.validation_conditions_not_equal_string
    def test_validation_conditions_not_equal_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_not_equal("disc")\
                          .evaluate
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_not_equal('hats')\
                          .evaluate
        assert answer is False, "Invalid evaluation of not_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_not_equal('a_sg')\
                          .evaluate
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_not_equal('a_s')\
                          .evaluate
        assert answer is False, "Invalid evaluation of not_equal condition"                        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less
    @mark.validation_conditions_less_numbers
    def test_validation_conditions_less_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_less(6)\
                          .evaluate
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 'a_g')\
                          .when_value\
                          .is_less(4)\
                          .evaluate
        assert answer is False, "Invalid evaluation of less condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'f')\
                          .when_value\
                          .is_less('i')\
                          .evaluate
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_less('f')\
                          .evaluate
        assert answer is False, "Invalid evaluation of less condition"        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less
    @mark.validation_conditions_less_strings
    def test_validation_conditions_less_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_less('z')\
                          .evaluate
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_less('a')\
                          .evaluate
        assert answer is False, "Invalid evaluation of less condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_less('z')\
                          .evaluate
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_less('a_s')\
                          .evaluate
        assert answer is False, "Invalid evaluation of less condition"        



    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less_equal
    @mark.validation_conditions_less_equal_numbers
    def test_validation_conditions_less_equal_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_less_equal(5)\
                          .evaluate
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 'a_g')\
                          .when_value\
                          .is_less_equal(7)\
                          .evaluate
        assert answer is False, "Invalid evaluation of less_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_g')\
                          .when_value\
                          .is_less_equal('a_ge')\
                          .evaluate
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_ge')\
                          .when_value\
                          .is_less_equal('a_le')\
                          .evaluate
        assert answer is False, "Invalid evaluation of less_equal condition"           


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less_equal
    @mark.validation_conditions_less_equal_strings
    def test_validation_conditions_less_equal_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_less_equal('hats')\
                          .evaluate
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_less_equal('a')\
                          .evaluate
        assert answer is False, "Invalid evaluation of less_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_less_equal('a_s')\
                          .evaluate
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_less_equal('s')\
                          .evaluate
        assert answer is False, "Invalid evaluation of less_equal condition"           


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater
    @mark.validation_conditions_greater_numbers
    def test_validation_conditions_greater_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_greater(3)\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 'a_g')\
                          .when_value\
                          .is_greater(3)\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_greater('f')\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'f')\
                          .when_value\
                          .is_greater('i')\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater condition"        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater
    @mark.validation_conditions_greater_strings
    def test_validation_conditions_greater_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_greater('a')\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_greater('z')\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_sg')\
                          .when_value\
                          .is_greater('a_s')\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_greater('a_sg')\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater condition"        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater_equal
    @mark.validation_conditions_greater_equal_numbers
    def test_validation_conditions_greater_equal_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 'i')\
                          .when_value\
                          .is_greater_equal(5)\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 'a_g')\
                          .when_value\
                          .is_greater_equal(50)\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_g')\
                          .when_value\
                          .is_greater_equal('a_ge')\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_le')\
                          .when_value\
                          .is_greater_equal('a_ge')\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater_equal condition"                   


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater_equal
    @mark.validation_conditions_greater_equal_strings
    def test_validation_conditions_greater_equal_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_greater_equal("hats")\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 's')\
                          .when_value\
                          .is_greater_equal('z')\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object, 'a_sg')\
                          .when_value\
                          .is_greater_equal('a_s')\
                          .evaluate
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_greater_equal('a_sg')\
                          .evaluate
        assert answer is False, "Invalid evaluation of greater_equal condition"                   



    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_match
    def test_validation_conditions_match(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_match("[a-zA-Z]+")\
                          .evaluate
        assert answer is True, "Invalid evaluation of match condition"
        # Evaluates to false with constant
        answer = condition.on(test_object, 'a_s')\
                          .when_value\
                          .is_match("[0-9]+")\
                          .evaluate
        assert answer is False, "Invalid evaluation of match condition"        
