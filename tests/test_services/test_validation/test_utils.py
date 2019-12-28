#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_utils.py                                                        #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 28th 2019, 9:18:43 am                        #
# Last Modified: Saturday December 28th 2019, 6:29:14 pm                      #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from ml_studio.services.validation.utils import is_array, is_homogeneous_array
from ml_studio.services.validation.utils import is_simple_array, is_none
from ml_studio.services.validation.utils import is_not_none, is_empty
from ml_studio.services.validation.utils import is_not_empty, is_bool
from ml_studio.services.validation.utils import is_integer, is_number
from ml_studio.services.validation.utils import is_string, is_less
from ml_studio.services.validation.utils import is_less_equal, is_greater
from ml_studio.services.validation.utils import is_greater_equal, is_match
from ml_studio.services.validation.utils import is_equal, is_not_equal

class SyntacticFunctionsTests:

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_array
    def test_validation_utils_is_array(self):
        # Test scaler
        a = 1
        assert is_array(a) is False, "Invalid is_array evaluation on scaler"
        # Test string
        a = "dds"
        assert is_array(a) is False, "Invalid is_array evaluation on string"
        # Test list
        a = ["dds"]
        assert is_array(a) is True, "Invalid is_array evaluation on list"
        # Test set
        a = {"dds",2}
        assert is_array(a) is True, "Invalid is_array evaluation on tuple"
        # Test tuple
        a = ("dds",2)
        assert is_array(a) is True, "Invalid is_array evaluation on tuple"
        # Test numpy array
        a = np.array(["dds"])
        assert is_array(a) is True, "Invalid is_array evaluation on np.array"
        # Test pandas Series
        a = "dds"
        a = pd.Series(a)
        assert is_array(a) is True, "Invalid is_array evaluation on np.array"

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_homogeneous_array
    def test_validation_utils_is_homogeneous_array(self):
        # Test scaler
        a = 1
        assert is_homogeneous_array(a) is False, "Invalid is_homogeneous_array evaluation on scaler"
        # Test string
        a = "dds"
        assert is_homogeneous_array(a) is False, "Invalid is_homogeneous_array evaluation on string"
        # Test list
        a = ["dds"]
        assert is_homogeneous_array(a) is False, "Invalid is_homogeneous_array evaluation on list"
        # Test set
        a = {"dds",2}
        assert is_homogeneous_array(a) is False, "Invalid is_homogeneous_array evaluation on tuple"
        # Test tuple
        a = ("dds",2)
        assert is_homogeneous_array(a) is False, "Invalid is_homogeneous_array evaluation on tuple"
        # Test numpy array
        a = np.array(["dds"])
        assert is_homogeneous_array(a) is True, "Invalid is_homogeneous_array evaluation on np.array"
        # Test pandas Series
        a = "dds"
        a = pd.Series(a)
        assert is_homogeneous_array(a) is True, "Invalid is_homogeneous_array evaluation on np.array"

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_simple_array
    def test_validation_utils_is_simple_array(self):
        # Test scaler
        a = 1
        assert is_simple_array(a) is False, "Invalid is_simple_array evaluation on scaler"
        # Test string
        a = "dds"
        assert is_simple_array(a) is False, "Invalid is_simple_array evaluation on string"
        # Test list
        a = ["dds"]
        assert is_simple_array(a) is True, "Invalid is_simple_array evaluation on list"
        # Test set
        a = {"dds",2}
        assert is_simple_array(a) is True, "Invalid is_simple_array evaluation on tuple"
        # Test tuple
        a = ("dds",2)
        assert is_simple_array(a) is True, "Invalid is_simple_array evaluation on tuple"
        # Test numpy array
        a = np.array(["dds"])
        assert is_simple_array(a) is False, "Invalid is_simple_array evaluation on np.array"
        # Test pandas Series
        a = "dds"
        a = pd.Series(a)
        assert is_simple_array(a) is False, "Invalid is_simple_array evaluation on np.array"        

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_none
    def test_validation_utils_is_none(self):
        # Test Basictype
        a = None
        assert is_none(a) is True, "Invalid evaluation is_none" 
        a = False
        assert is_none(a) is False, "Invalid evaluation is_none" 
        # Test List that evaluates to True
        a = [None, None]
        assert is_none(a) is True, "Invalid evaluation is_none" 
        # Test set that evaluates to True
        a = {None, None}
        assert is_none(a) is True, "Invalid evaluation is_none" 
        # Test tuple that evaluates to True
        a = (None, None)
        assert is_none(a) is True, "Invalid evaluation is_none" 
        # Test List that evaluates to False
        a = [None, 2]
        assert is_none(a) is False, "Invalid evaluation is_none" 
        # Test set that evaluates to False
        a = {None, 2}
        assert is_none(a) is False, "Invalid evaluation is_none" 
        # Test tuple that evaluates to False
        a = (None, 2)
        assert is_none(a) is False, "Invalid evaluation is_none"         

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_not_none
    def test_validation_utils_is_not_none(self):
        # Test Basictype
        a = None
        assert is_not_none(a) is False, "Invalid evaluation is_not_none" 
        a = False
        assert is_not_none(a) is True, "Invalid evaluation is_not_none" 
        # Test List that evaluates to True
        a = [None, None]
        assert is_not_none(a) is False, "Invalid evaluation is_not_none" 
        # Test set that evaluates to True
        a = {None, None}
        assert is_not_none(a) is False, "Invalid evaluation is_not_none" 
        # Test tuple that evaluates to True
        a = (None, None)
        assert is_not_none(a) is False, "Invalid evaluation is_not_none" 
        # Test List that evaluates to False
        a = [None, 2]
        assert is_not_none(a) is True, "Invalid evaluation is_not_none" 
        # Test set that evaluates to False
        a = {None, 2}
        assert is_not_none(a) is True, "Invalid evaluation is_not_none" 
        # Test tuple that evaluates to False
        a = (None, 2)
        assert is_not_none(a) is True, "Invalid evaluation is_not_none"       

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_empty
    def test_validation_utils_is_empty(self):
        # Test Basictype
        a = None
        assert is_empty(a) is False, "Invalid evaluation is_empty" 
        a = False
        assert is_empty(a) is False, "Invalid evaluation is_empty" 
        # Test List that evaluates to True
        a = ["", ""]
        assert is_empty(a) is True, "Invalid evaluation is_empty" 
        # Test set that evaluates to True
        a = {"", ""}
        assert is_empty(a) is True, "Invalid evaluation is_empty" 
        # Test tuple that evaluates to True
        a = ("", "")
        assert is_empty(a) is True, "Invalid evaluation is_empty" 
        # Test List that evaluates to False
        a = [None, 2]
        assert is_empty(a) is False, "Invalid evaluation is_empty" 
        # Test set that evaluates to False
        a = {None, 2}
        assert is_empty(a) is False, "Invalid evaluation is_empty" 
        # Test tuple that evaluates to False
        a = (None, 2)
        assert is_empty(a) is False, "Invalid evaluation is_empty"    

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_not_empty
    def test_validation_utils_is_not_empty(self):
        # Test Basictype
        a = None
        assert is_not_empty(a) is True, "Invalid evaluation is_not_empty" 
        a = False
        assert is_not_empty(a) is True, "Invalid evaluation is_not_empty" 
        a = " "
        assert is_not_empty(a) is False, "Invalid evaluation is_not_empty" 
        # Test List that evaluates to False
        a = ["", " "]
        assert is_not_empty(a) is False, "Invalid evaluation is_not_empty" 
        # Test set that evaluates to False
        a = {"", " "}
        assert is_not_empty(a) is False, "Invalid evaluation is_not_empty" 
        # Test tuple that evaluates to False
        a = ("", " ")
        assert is_not_empty(a) is False, "Invalid evaluation is_not_empty" 
        # Test List that evaluates to True
        a = [None, 2]
        assert is_not_empty(a) is True, "Invalid evaluation is_not_empty" 
        # Test set that evaluates to False
        a = {None, 2}
        assert is_not_empty(a) is True, "Invalid evaluation is_not_empty" 
        # Test tuple that evaluates to False
        a = (None, 2)
        assert is_not_empty(a) is True, "Invalid evaluation is_not_empty"    

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_bool
    def test_validation_utils_is_bool(self):
        # Test basic type that evaluates to True
        a = False
        assert is_bool(a) is True, "Invalid evaluation is_bool" 
        # Test basic type that evaluates to False
        a = None
        assert is_bool(a) is False, "Invalid evaluation is_bool" 
        # Test array that evaluates to True
        a = [True, False]
        assert is_bool(a) is True, "Invalid evaluation is_bool" 
        # Test array that evaluates to False
        a = [True, "hat"]
        assert is_bool(a) is False, "Invalid evaluation is_bool"         

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_integer
    def test_validation_utils_is_integer(self):
        # Test basic type that evaluates to True
        a = 8
        assert is_integer(a) is True, "Invalid evaluation is_integer" 
        # Test basic type that evaluates to False
        a = None
        assert is_integer(a) is False, "Invalid evaluation is_integer" 
        # Test array that evaluates to True
        a = [7, 8]
        assert is_integer(a) is True, "Invalid evaluation is_integer" 
        # Test array that evaluates to False
        a = [9, "hat"]
        assert is_integer(a) is False, "Invalid evaluation is_integer"                 

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_number
    def test_validation_utils_is_number(self):
        # Test basic type that evaluates to True
        a = 8.0
        assert is_number(a) is True, "Invalid evaluation is_number" 
        # Test basic type that evaluates to False
        a = None
        assert is_number(a) is False, "Invalid evaluation is_number" 
        # Test array that evaluates to True
        a = [7.2, 8.1]
        assert is_number(a) is True, "Invalid evaluation is_number" 
        # Test array that evaluates to False
        a = [9.1, "hat"]
        assert is_number(a) is False, "Invalid evaluation is_number"           

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_string
    def test_validation_utils_is_string(self):
        # Test basic type that evaluates to True
        a = "7.0"
        assert is_string(a) is True, "Invalid evaluation is_string" 
        # Test basic type that evaluates to False
        a = None
        assert is_string(a) is False, "Invalid evaluation is_string" 
        # Test array that evaluates to True
        a = ["7.2", "8.1"]
        assert is_string(a) is True, "Invalid evaluation is_string" 
        # Test array that evaluates to False
        a = [9.1, "hat"]
        assert is_string(a) is False, "Invalid evaluation is_string"                

class SemanticFunctionsTests:

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_less
    @mark.validation_utils_is_less_number
    def test_validation_utils_is_less_numbers(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_less(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_less(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_less(a,b)
        # Test basic types for a and b that evaluates to True
        a = 3
        b = 4.5
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test basic types for a and b that evaluates to False
        a = 3
        b = 3
        assert is_less(a,b) is False, "Invalid evaluation of is_less"
        # Test a as basic type and b as an array that evaluates to True
        a = 3
        b = [4,5,6]
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test a as basic type and b as an array that evaluates to False
        a = 3
        b = [1,5,6]
        assert is_less(a,b) is False, "Invalid evaluation of is_less"
        # Test a and b are arrays and evaluates to True
        a = [3,9,2]
        b = [5,15,5]
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test a and b are arrays and evaluates to False
        a = [3,9,7]
        b = [5,15,5]
        assert is_less(a,b) is False, "Invalid evaluation of is_less"
        # Test a is array and b is a basic type and evaluates to True
        a = [3,9,7]
        b = 15
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        b = 5
        assert is_less(a,b) is False, "Invalid evaluation of is_less"


    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_less
    @mark.validation_utils_is_less_strings
    def test_validation_utils_is_less_strings(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_less(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_less(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_less(a,b)
        # Test basic types for a and b that evaluates to True
        a = "3"
        b = "4.5"
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test basic types for a and b that evaluates to False
        a = "3"
        b = "3"
        assert is_less(a,b) is False, "Invalid evaluation of is_less"
        # Test a as basic type and b as an array that evaluates to True
        a = "3"
        b = ["4","5","6"]
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test a as basic type and b as an array that evaluates to False
        a = "3"
        b = ["1","5","6"]
        assert is_less(a,b) is False, "Invalid evaluation of is_less"
        # Test a and b are arrays and evaluates to True
        a = ["3","6","2"]
        b = ["5","9","5"]
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test a and b are arrays and evaluates to False
        a = ["3","6","7"]
        b = ["5","9","5"]
        assert is_less(a,b) is False, "Invalid evaluation of is_less"
        # Test a is array and b is a basic type and evaluates to True
        a = ["3","8","7"]
        b = "9"
        assert is_less(a,b) is True, "Invalid evaluation of is_less"
        # Test a is array and b is a basic type and evaluates to False
        a = ["3","9","7"]
        b = "5"
        assert is_less(a,b) is False, "Invalid evaluation of is_less"



    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_less_equal
    @mark.validation_utils_is_less_equal_numbers
    def test_validation_utils_is_less_equal_numbers(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_less_equal(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_less_equal(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_less_equal(a,b)
        # Test basic types for a and b that evaluates to True
        a = 3
        b = 3
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test basic types for a and b that evaluates to False
        a = 3
        b = 2
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"
        # Test a as basic type and b as an array that evaluates to True
        a = 3
        b = [3,5,6]
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test a as basic type and b as an array that evaluates to False
        a = 3
        b = [1,5,6]
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"
        # Test a and b are arrays and evaluates to True
        a = [3,9,2]
        b = [5,15,5]
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test a and b are arrays and evaluates to False
        a = [3,9,7]
        b = [5,15,5]
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"
        # Test a is array and b is a basic type and evaluates to True
        a = [3,9,7]
        b = 15
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        b = 5
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"



    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_less_equal
    @mark.validation_utils_is_less_equal_strings
    def test_validation_utils_is_less_equal_strings(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_less_equal(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_less_equal(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_less_equal(a,b)
        # Test basic types for a and b that evaluates to True
        a = "3"
        b = "3"
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test basic types for a and b that evaluates to False
        a = "3"
        b = "2"
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"
        # Test a as basic type and b as an array that evaluates to True
        a = "3"
        b = ["3","5","6"]
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test a as basic type and b as an array that evaluates to False
        a = "3"
        b = ["1","5","6"]
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"
        # Test a and b are arrays and evaluates to True
        a = ["3","8","2"]
        b = ["5","9","5"]
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test a and b are arrays and evaluates to False
        a = ["3","8","7"]
        b = ["5","9","5"]
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"
        # Test a is array and b is a basic type and evaluates to True
        a = ["3","8","7"]
        b = "9"
        assert is_less_equal(a,b) is True, "Invalid evaluation of is_less_equal"
        # Test a is array and b is a basic type and evaluates to False
        a = ["3","9","7"]
        b = "5"
        assert is_less_equal(a,b) is False, "Invalid evaluation of is_less_equal"



    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_greater
    @mark.validation_utils_is_greater_numbers
    def test_validation_utils_is_greater_numbers(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_greater(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_greater(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_greater(a,b)
        # Test basic types for a and b that evaluates to True
        a = 6
        b = 4.5
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test basic types for a and b that evaluates to False
        a = 3
        b = 3
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"
        # Test a as basic type and b as an array that evaluates to True
        a = 10
        b = [4,5,6]
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test a as basic type and b as an array that evaluates to False
        a = 5
        b = [1,5,6]
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"
        # Test a and b are arrays and evaluates to True
        b = [3,9,2]
        a = [5,15,5]
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test a and b are arrays and evaluates to False
        b = [3,9,7]
        a = [5,15,5]
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"
        # Test a is array and b is a basic type and evaluates to True
        b = [3,9,7]
        a = 15
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        b = 5
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"


    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_greater
    @mark.validation_utils_is_greater_strings
    def test_validation_utils_is_greater_strings(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_greater(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_greater(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_greater(a,b)
        # Test basic types for a and b that evaluates to True
        a = "6"
        b = "4.5"
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test basic types for a and b that evaluates to False
        a = "3"
        b = "3"
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"
        # Test a as basic type and b as an array that evaluates to True
        a = "9"
        b = ["4","5","6"]
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test a as basic type and b as an array that evaluates to False
        a = "5"
        b = ["1","5","6"]
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"
        # Test a and b are arrays and evaluates to True
        b = ["3","8","2"]
        a = ["5","9","5"]
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test a and b are arrays and evaluates to False
        b = ["3","9","7"]
        a = [5,9,5]
        a = list(map(str,a))
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"
        # Test a is array and b is a basic type and evaluates to True
        b = [3,5,7]
        b = list(map(str,b))
        a = str(9)
        assert is_greater(a,b) is True, "Invalid evaluation of is_greater"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        a = list(map(str,a))
        b = str(5)
        assert is_greater(a,b) is False, "Invalid evaluation of is_greater"


    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_greater_equal
    @mark.validation_utils_is_greater_equal_numbers
    def test_validation_utils_is_greater_equal_numbers(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_greater_equal(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_greater_equal(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_greater_equal(a,b)
        # Test basic types for a and b that evaluates to True
        b = 3
        a = 3
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test basic types for a and b that evaluates to False
        b = 3
        a = 2
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"
        # Test a as basic type and b as an array that evaluates to True
        b = 3
        a = [3,5,6]
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test a as basic type and b as an array that evaluates to False
        b = 3
        a = [1,5,6]
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"
        # Test a and b are arrays and evaluates to True
        b = [3,9,2]
        a = [5,15,5]
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test a and b are arrays and evaluates to False
        b = [3,9,7]
        a = [5,15,5]
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"
        # Test a is array and b is a basic type and evaluates to True
        b = [3,9,7]
        a = 15
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        b = 5
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"        


    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_greater_equal
    @mark.validation_utils_is_greater_equal_strings
    def test_validation_utils_is_greater_equal_strings(self):
        # Test invalid types
        with pytest.raises(TypeError):
            a = "7.0"
            b = 6
            is_greater_equal(a,b)
        with pytest.raises(TypeError):
            a = 4
            b = "6"
            is_greater_equal(a,b)
        with pytest.raises(TypeError):
            a = 5
            b = [4,5,None]
            is_greater_equal(a,b)
        # Test basic types for a and b that evaluates to True
        b = str(3)
        a = str(3)
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test basic types for a and b that evaluates to False
        b = str(3)
        a = str(2)
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"
        # Test a as basic type and b as an array that evaluates to True
        b = str(3)
        a = [3,5,6]
        a = list(map(str,a))
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test a as basic type and b as an array that evaluates to False
        b = str(3)
        a = [1,5,6]
        a = list(map(str,a))
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"
        # Test a and b are arrays and evaluates to True
        b = [3,7,2]
        b = list(map(str,b))
        a = [5,9,5]
        a = list(map(str,a))
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test a and b are arrays and evaluates to False
        b = [3,4,7]
        b = list(map(str,b))
        a = [5,9,5]
        a = list(map(str,a))
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"
        # Test a is array and b is a basic type and evaluates to True
        b = [3,6,7]
        b = list(map(str,b))        
        a = str(9)
        assert is_greater_equal(a,b) is True, "Invalid evaluation of is_greater_equal"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        a = list(map(str,a))
        b = str(5)
        assert is_greater_equal(a,b) is False, "Invalid evaluation of is_greater_equal"        


    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_match
    def test_validation_utils_is_match(self):
        # Test basic types for a and b that evaluates to True
        a = "3"
        b = "3"
        assert is_match(a,b) is True, "Invalid evaluation of is_match"                    
        # Test basic types for a and b that evaluates to True
        a = "May 2013"
        b = '[a-zA-Z]+\\s+\\d{4}'
        assert is_match(a,b) is True, "Invalid evaluation of is_match"        

        # Test basic types for a and b that evaluates to False
        a = "3"
        b = "[a-zA-Z]+"
        assert is_match(a,b) is False, "Invalid evaluation of is_match"                    
        # Test basic types for a and b that evaluates to False
        a = "hats"
        b = '[a-zA-Z]+\\s+\\d{4}'
        assert is_match(a,b) is False, "Invalid evaluation of is_match"                

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_equal_numbers
    def test_validation_utils_is_equal_numbers(self):
        # Test basic types for a and b that evaluates to True
        b = 3
        a = 3
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test basic types for a and b that evaluates to False
        b = 3
        a = 2
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"
        # Test a as basic type and b as an array that evaluates to True
        b = 4
        a = [4,4,4]
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test a as basic type and b as an array that evaluates to False
        a = 2
        b = [4,4,4]
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"
        # Test a and b are arrays and evaluates to True
        a = [1,"hat",None, 6]
        b = [1,"hat",None, 6]
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test a and b are arrays and evaluates to False
        a = [2,9,7]
        b = [3,9,7]
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"
        # Test a is array and b is a basic type and evaluates to True
        b = [15,15,15]
        a = 15
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        b = 5
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"                            

    @mark.validation
    @mark.validation_utils
    @mark.validation_utils_is_equal_strings
    def test_validation_utils_is_equal_strings(self):
        # Test basic types for a and b that evaluates to True
        b = str(3)
        a = str(3)
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test basic types for a and b that evaluates to False
        b = str(3)
        a = str(2)
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"
        # Test a as basic type and b as an array that evaluates to True
        b = str(4)
        a = [4,4,4]
        a = list(map(str,a))
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test a as basic type and b as an array that evaluates to False
        a = "2"
        b = [4,4,4]
        b = list(map(str,b))
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"
        # Test a and b are arrays and evaluates to True
        a = [1,"hat",None, 6]
        b = [1,"hat",None, 6]
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test a and b are arrays and evaluates to False
        a = [2,9,7]
        a = list(map(str,a))
        b = [3,9,7]
        b = list(map(str,b))
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"
        # Test a is array and b is a basic type and evaluates to True
        b = [15,15,15]
        b = list(map(str,b))
        a = str(15)
        assert is_equal(a,b) is True, "Invalid evaluation of is_equal"
        # Test a is array and b is a basic type and evaluates to False
        a = [3,9,7]
        a = list(map(str,a))
        b = str(5)
        assert is_equal(a,b) is False, "Invalid evaluation of is_equal"            