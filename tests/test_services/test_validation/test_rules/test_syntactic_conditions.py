# =========================================================================== #
#                       TEST SYNTACTIC CONDITIONS                             #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_syntactic_conditions.py                                         #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 21st 2019, 1:15:37 am                        #
# Last Modified: Saturday December 21st 2019, 1:16:13 am                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Tests syntactic conditions.

Test validation conditions are evaluated before a rule is validated. If 
a when condition is met, validation continues. If an except when condition
is met, validation stops and vice-versa. 

There are two types of conditions, syntactic and semantic. This module tests
the syntactic conditions. These are conditions that take a single parameter, 
the variable being evaluated, and evaluates the type and state.

    Syntactic Conditions
    --------------------
    * isNone : Evaluates whether the argument is None.
    * isEmpty : Evaluates whether the argument is empty string or whitespace.
    * isBool : Evaluates whether the argument is a Boolean.
    * isInt : Evaluates whether the argument is an integer.
    * isFloat : Evaluates whether the argument is an float.
    * isNumber : Evaluates whether the argument is a number.
    * isString : Evaluates whether the argument is a string. 
    * isDate : Evaluates whether a string is a valid datetime format.   

    * isAllNone : Evaluates whether the argument isAll None.
    * isAllEmpty : Evaluates whether the argument isAll empty string or whitespace.
    * isAllBool : Evaluates whether the argument isAll a Boolean.
    * isAllInt : Evaluates whether the argument isAll an integer.
    * isAllFloat : Evaluates whether the argument isAll an float.
    * isAllNumber : Evaluates whether the argument isAll a number.
    * isAllString : Evaluates whether the argument isAll a string. 
    * isAllDate : Evaluates whether a string isAll a valid datetime format.   

"""
import pytest
from pytest import mark
import numpy as np

from ml_studio.services.validation.conditions import isNone, isEmpty, isBool, isInt
from ml_studio.services.validation.conditions import isFloat, isNumber, isString
from ml_studio.services.validation.conditions import isDate

from ml_studio.services.validation.conditions import isAllNone, isAllEmpty, isAllBool, isAllInt
from ml_studio.services.validation.conditions import isAllFloat, isAllNumber, isAllString
from ml_studio.services.validation.conditions import isAllDate

class SyntacticConditionTests:

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isNone
    def test_syntactic_conditions_isNone(self):
        a = 5
        assert isNone(a) == False, "isNone incorrect evaluation."
        a = None
        assert isNone(a) == True, "isNone incorrect evaluation."
        
    @mark.syntactic_conditions
    @mark.syntactic_conditions_isEmpty
    def test_syntactic_conditions_isEmpty(self):
        a = "x"
        b = " "
        c = ""
        d = None
        assert isEmpty(a) == False, "isEmpty incorrect evaluation."
        assert isEmpty(b) == True, "isEmpty incorrect evaluation."
        assert isEmpty(c) == True, "isEmpty incorrect evaluation."
        assert isEmpty(d) == True, "isEmpty incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isBool
    def test_syntactic_conditions_isBool(self):
        a = False
        b = 'False'
        c = 1
        d = 0
        assert isBool(a) == True, "isBool incorrect evaluation."
        assert isBool(b) == False, "isBool incorrect evaluation."
        assert isBool(c) == False, "isBool incorrect evaluation."
        assert isBool(d) == False, "isBool incorrect evaluation."                

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isInt
    def test_syntactic_conditions_isInt(self):
        a = 1
        b = '1'
        c = 2.0
        assert isInt(a) == True, "isInt incorrect evaluation."
        assert isInt(b) == False, "isInt incorrect evaluation."
        assert isInt(c) == False, "isInt incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isFloat
    def test_syntactic_conditions_isFloat(self):
        a = 1.0
        b = '1.0'
        c = 2
        assert isFloat(a) == True, "isFloat incorrect evaluation."
        assert isFloat(b) == False, "isFloat incorrect evaluation."
        assert isFloat(c) == False, "isFloat incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isNumber
    def test_syntactic_conditions_isNumber(self):
        a = 1.0
        b = '1.0'
        c = 'x'
        d = 2
        assert isNumber(a) == True, "isNumber incorrect evaluation."
        assert isNumber(b) == False, "isNumber incorrect evaluation."
        assert isNumber(c) == False, "isNumber incorrect evaluation."        
        assert isNumber(d) == True, "isNumber incorrect evaluation."        

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isString
    def test_syntactic_conditions_isString(self):
        a = 1.0
        b = '1.0'
        c = 'x'
        d = 2
        assert isString(a) == False, "isString incorrect evaluation."
        assert isString(b) == True, "isString incorrect evaluation."
        assert isString(c) == True, "isString incorrect evaluation."        
        assert isString(d) == False, "isString incorrect evaluation."          

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isDate
    def test_syntactic_conditions_isDate(self):
        a = '12/21/2019'
        b = '1x.0'
        c = 'x'
        assert isDate(a) == True, "isDate incorrect evaluation."
        assert isDate(b) == False, "isDate incorrect evaluation."
        assert isDate(c) == False, "isDate incorrect evaluation."        

class SyntacticConditionAllTests:

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllNone
    def test_syntactic_conditions_isAllNone(self):
        a = [5, None, 3, True]
        assert isAllNone(a) == False, "isAllNone incorrect evaluation."
        a = [None, None, None]
        assert isAllNone(a) == True, "isAllNone incorrect evaluation."
        
    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllEmpty
    def test_syntactic_conditions_isAllEmpty(self):
        a = ["", " ", None]
        b = ["x", " ", None]
        assert isAllEmpty(a) == True, "isAllEmpty incorrect evaluation."
        assert isAllEmpty(b) == False, "isAllEmpty incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllBool
    def test_syntactic_conditions_isAllBool(self):
        a = [False, True, True]
        b = ['False', 1, 0]
        assert isAllBool(a) == True, "isAllBool incorrect evaluation."
        assert isAllBool(b) == False, "isAllBool incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllInt
    def test_syntactic_conditions_isAllInt(self):
        a = [1, 2, 3]
        b = ['1', 2, 3]
        assert isAllInt(a) == True, "isAllInt incorrect evaluation."
        assert isAllInt(b) == False, "isAllInt incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllFloat
    def test_syntactic_conditions_isAllFloat(self):
        a = [1.0, 2.3]
        b = [1,2]
        assert isAllFloat(a) == True, "isAllFloat incorrect evaluation."
        assert isAllFloat(b) == False, "isAllFloat incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllNumber
    def test_syntactic_conditions_isAllNumber(self):
        a = [1.0, 2, 3]
        b = ['1.0',2, 3]
        assert isAllNumber(a) == True, "isAllNumber incorrect evaluation."
        assert isAllNumber(b) == False, "isAllNumber incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllString
    def test_syntactic_conditions_isAllString(self):
        a = ['hat', 'shoes', 'belt']
        b = [True, 9, 2.0]
        assert isAllString(a) == True, "isAllString incorrect evaluation."
        assert isAllString(b) == False, "isAllString incorrect evaluation."

    @mark.syntactic_conditions
    @mark.syntactic_conditions_isAllDate
    def test_syntactic_conditions_isAllDate(self):
        a = ['12/21/2019', '1/1/2001', '12/31/2009']
        b = ['x', 'y', 'z']
        assert isAllDate(a) == True, "isAllDate incorrect evaluation."
        assert isAllDate(b) == False, "isAllDate incorrect evaluation."
