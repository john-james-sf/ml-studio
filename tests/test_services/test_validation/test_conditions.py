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
# Last Modified: Sunday December 29th 2019, 1:34:06 pm                        #
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

from ml_studio.services.validation.conditions import Condition, ConditionSet

class SyntacticConditionTests:

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_none    
    def test_validation_conditions_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('a_n')\
                          .is_none\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of none condition"
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('a_xn')\
                          .is_none\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of none condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_not_none
    def test_validation_conditions_not_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('a_n')\
                          .is_not_none\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not none condition"
        # Evaluates to false
        answer = condition.on(test_object)\
                          .when('a_xn')\
                          .is_not_none\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not none condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_empty    
    def test_validation_conditions_empty(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('e')\
                          .is_empty\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of empty condition"
        # Evaluates to false
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_empty\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of empty condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_not_empty
    def test_validation_conditions_not_empty(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('e')\
                          .is_not_empty\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not empty condition"
        # Evaluates to false
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_not_empty\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not empty condition"

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_bool    
    def test_validation_conditions_bool(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('b')\
                          .is_bool\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of bool condition"
        # Evaluates to false
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_bool\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of bool condition"        

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_integer    
    def test_validation_conditions_integer(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_integer\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of integer condition"
        # Evaluates to false
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_integer\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of integer condition"                

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_number    
    def test_validation_conditions_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('f')\
                          .is_number\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of number condition"
        # Evaluates to false
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_number\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of number condition"                        

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_syntactic
    @mark.validation_conditions_string    
    def test_validation_conditions_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_string\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of string condition"
        # Evaluates to false
        answer = condition.on(test_object)\
                          .when('a_l')\
                          .is_string\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of string condition"                                

class SemanticConditionTests:

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_equal_number
    def test_validation_conditions_equal_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_equal(5)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('a_g')\
                          .is_equal(50)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('f')\
                          .is_equal('i')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal condition"   


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_equal_string
    def test_validation_conditions_equal_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_equal("hats")\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_equal("fott")\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_equal('a_s')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_equal('a_sg')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of equal condition"   


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_not_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_not_equal_number
    def test_validation_conditions_not_equal_number(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_not_equal(6)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_not_equal(5)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('f')\
                          .is_not_equal('i')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_ge')\
                          .is_not_equal('a_g')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal condition"                        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_not_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_not_equal_string
    def test_validation_conditions_not_equal_string(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_not_equal("disc")\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_not_equal('hats')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_not_equal('a_sg')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of not_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_not_equal('a_s')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of not_equal condition"                        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less
    @mark.validation_conditions_semantic
    @mark.validation_conditions_less_numbers
    def test_validation_conditions_less_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_less(6)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('a_g')\
                          .is_less(4)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('f')\
                          .is_less('i')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_less('f')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less condition"        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less
    @mark.validation_conditions_semantic
    @mark.validation_conditions_less_strings
    def test_validation_conditions_less_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_less('z')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_less('a')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_less('z')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_less('a_s')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less condition"        



    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_less_equal_numbers
    def test_validation_conditions_less_equal_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_less_equal(5)\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('a_g')\
                          .is_less_equal(7)\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_g')\
                          .is_less_equal('a_ge')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_ge')\
                          .is_less_equal('a_le')\
                          .evaluate\
                          .is_valid
        assert answer is False, "Invalid evaluation of less_equal condition"           


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_less_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_less_equal_strings
    def test_validation_conditions_less_equal_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_less_equal('hats')\
                          .evaluate\
                          .is_valid
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_less_equal('a')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of less_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_less_equal('a_s')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of less_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_less_equal('s')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of less_equal condition"           


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater
    @mark.validation_conditions_semantic
    @mark.validation_conditions_greater_numbers
    def test_validation_conditions_greater_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_greater(3)\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('a_g')\
                          .is_greater(3)\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_greater('f')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('f')\
                          .is_greater('i')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater condition"        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater
    @mark.validation_conditions_semantic
    @mark.validation_conditions_greater_strings
    def test_validation_conditions_greater_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_greater('a')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_greater('z')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_sg')\
                          .is_greater('a_s')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_greater('a_sg')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater condition"        


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_greater_equal_numbers
    def test_validation_conditions_greater_equal_numbers(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('i')\
                          .is_greater_equal(5)\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('a_g')\
                          .is_greater_equal(50)\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_g')\
                          .is_greater_equal('a_ge')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_le')\
                          .is_greater_equal('a_ge')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal condition"                   


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_greater_equal
    @mark.validation_conditions_semantic
    @mark.validation_conditions_greater_equal_strings
    def test_validation_conditions_greater_equal_strings(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_greater_equal("hats")\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('s')\
                          .is_greater_equal('z')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal condition"        
        # Evaluates to true with attribute
        answer = condition.on(test_object)\
                          .when('a_sg')\
                          .is_greater_equal('a_s')\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of greater_equal condition"
        # Evaluates to false with attribute
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_greater_equal('a_sg')\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of greater_equal condition"                   



    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_semantic
    @mark.validation_conditions_match
    def test_validation_conditions_match(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        condition = Condition()
        # Evaluates to true with constant
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_match("[a-zA-Z]+")\
                          .evaluate.is_valid
        assert answer is True, "Invalid evaluation of match condition"
        # Evaluates to false with constant
        answer = condition.on(test_object)\
                          .when('a_s')\
                          .is_match("[0-9]+")\
                          .evaluate.is_valid
        assert answer is False, "Invalid evaluation of match condition"        



class ConditionSetTests:

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_semantic
    @mark.validation_conditions_set
    @mark.validation_conditions_set_and
    def test_validation_conditions_set_and(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object        
        # Create True 
        condition1 = Condition().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        assert condition1.evaluate.is_valid is True, "Failed Condition 1 before assigning to set."
        # Create another True condition
        condition2 = Condition().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')
        assert condition2.evaluate.is_valid is True, "Failed Condition 2 before assigning to set."

        # Create a false condition
        condition3 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        assert condition3.evaluate.is_valid is False, "Failed Condition 3 before assigning to set."                          
        # Create another false condition
        condition4 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)
        assert condition4.evaluate.is_valid is False, "Failed Condition 4 before assigning to set."
        # Create True/True condition set where all must be true                           
        cs = ConditionSet()
        cs.when_all_conditions_are_true
        cs.add_condition(condition1).add_condition(condition2)
        # Check evaluation
        answer = cs.evaluate.is_valid
        assert answer is True, "Invalid evaluation of conditions 1 and 2"
        
        # Create True/False condition set where all must be true
        cs.remove_condition(condition2).add_condition(condition3)                
        # Check evaluation
        answer = cs.evaluate.is_valid
        assert answer is False, "Invalid evaluation of conditions 1 and 3"

        # Create False/False condition where all must be true
        cs.remove_condition(condition3).add_condition(condition4)                
        # Check evaluation
        answer = cs.evaluate.is_valid
        assert answer is False, "Invalid evaluation of conditions 1 and 4"        

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_semantic
    @mark.validation_conditions_set
    @mark.validation_conditions_set_or
    def test_validation_conditions_set_or(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        
        # Create True 
        condition1 = Condition().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True condition
        condition2 = Condition().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false condition
        condition3 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false condition
        condition4 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)
        # Create True/True condition set where all must be true                           
        cs = ConditionSet()
        cs.when_any_condition_is_true
        cs.add_condition(condition1)
        cs.add_condition(condition2)
        answer = cs.evaluate.is_valid
        assert answer is True, "Invalid evaluation of conditions 1 and 2"
        # Create True/False condition set where all must be true
        cs.remove_condition(condition2)        
        cs.add_condition(condition3)                
        answer = cs.evaluate.is_valid
        cs.print_condition_set
        assert answer is True, "Invalid evaluation of conditions 1 and 3"
        # Create False/False condition where all must be true
        cs.remove_condition(condition1)
        cs.add_condition(condition4) 
        answer = cs.evaluate.is_valid           
        cs.print_condition_set
        assert answer is False, "Invalid evaluation of conditions 3 and 4"                

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_semantic
    @mark.validation_conditions_set
    @mark.validation_conditions_set_none
    def test_validation_conditions_set_none(self, get_validation_rule_test_object):
        test_object = get_validation_rule_test_object
        
        # Create True 
        condition1 = Condition().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True condition
        condition2 = Condition().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false condition
        condition3 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false condition
        condition4 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)
        # Create True/True condition set where all must be true                           
        cs = ConditionSet()
        cs.when_no_conditions_are_true
        cs.add_condition(condition1)
        cs.add_condition(condition2)
        answer = cs.evaluate.is_valid
        assert answer is False, "Invalid evaluation of conditions 1 and 2"
        # Create True/False condition set where all must be true
        cs.remove_condition(condition2)        
        cs.add_condition(condition3)                
        answer = cs.evaluate.is_valid
        cs.print_condition_set
        assert answer is False, "Invalid evaluation of conditions 1 and 3"
        # Create False/False condition where all must be true
        cs.remove_condition(condition1)
        cs.add_condition(condition4) 
        answer = cs.evaluate.is_valid           
        cs.print_condition_set
        assert answer is True, "Invalid evaluation of conditions 3 and 4"            

class ChildNodeTests:

    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_printing
    def test_validation_conditions_printing(self, get_validation_rule_test_object):
        """Testing propagation of data down through child nodes."""
        test_object = get_validation_rule_test_object        
        # Create some conditions
        # True Test Condition
        condition1 = Condition().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True condition
        condition2 = Condition().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false condition
        condition3 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false condition
        condition4 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)        
        # Add 1 and 2 to a new Condition Set
        cs1 = ConditionSet()
        cs1.add_condition(condition1).add_condition(condition2)
        # Add CS1 and conditions 3 and 4 to new ConditionSet
        cs2 = ConditionSet()
        cs2.add_condition(cs1).add_condition(condition3).add_condition(condition4)
        cs2.when_no_conditions_are_true
        # Print to see how it looks in the hierarchy
        cs2.print_condition_set


    @mark.validation
    @mark.validation_conditions
    @mark.validation_conditions_propagation
    def test_validation_conditions_propagation(self, get_validation_rule_test_object):
        """Testing propagation of data down through child nodes."""
        test_object = get_validation_rule_test_object        
        # Create some conditions
        # True Test Condition
        condition1 = Condition().on(test_object)\
                          .when('i')\
                          .is_equal(5)
        # Create another True condition
        condition2 = Condition().on(test_object)\
                          .when('a_ge')\
                          .is_equal('a_g')

        # Create a false condition
        condition3 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_equal(50)
        # Create another false condition
        condition4 = Condition().on(test_object)\
                          .when('a_g')\
                          .is_greater(50)        
        # Add 1 and 2 to a new Condition Set
        cs1 = ConditionSet()
        cs1.add_condition(condition1).add_condition(condition2)
        # Add CS1 and conditions 3 and 4 to new ConditionSet
        cs2 = ConditionSet()
        cs2.add_condition(cs1).add_condition(condition3).add_condition(condition4)
        cs2.when_no_conditions_are_true
        # Set the target object and attribute
        cs2.on(test_object)
        cs2.attribute('a_g')
        # Traverse through confirming updates
        def traverse(condition):         
            assert condition._evaluated_instance == test_object, "Test object not set" 
            assert condition._evaluated_attribute == "a_g", "Attribute not set" 
            if isinstance(condition, ConditionSet):
                for _,condition in condition._conditions.items():      
                    return traverse(condition)
            else:
                assert condition._evaluated_instance == test_object, "Test object not set" 
                assert condition._evaluated_attribute == "a_g", "Attribute not set" 
        traverse(cs2)