#!/usr/bin/env python3
# =========================================================================== #
#                          TEST VALIDATION CONTEXTS                           #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \test_validation_contexts.py                                          #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 13th 2019, 9:53:10 pm                          #
# Last Modified: Friday December 13th 2019, 10:39:32 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Test validation rules."""
#%%
import os
import shutil
import pytest
from pytest import mark
import numpy as np

from ml_studio.services.validation import ValidationContexts
from ml_studio.services.validation import Classes
from ml_studio.supervised_learning.regression import LinearRegression
# --------------------------------------------------------------------------- #
#%%

class ValidationContextsTests:

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_init
    def test_validation_contexts_instantiation(self):
        a = ValidationContexts()
        b = ValidationContexts()
        assert a == b, "Singleton didn't work"

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_add
    @mark.validation_contexts_add_str_invalid_class
    def test_validation_contexts_add_context_invalid_class(self):
        classname = 'LinearRegression'
        contexts = 'raw'
        r = ValidationContexts()
        with pytest.raises(ValueError):
            r.add_context(classname, contexts)

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_add
    @mark.validation_contexts_add_str
    def test_validation_contexts_add_context_str(self):
        classname = 'LinearRegression'
        contexts = 'raw'
        c = Classes()
        c.add_class(LinearRegression())
        r = ValidationContexts()
        r.add_context(classname, contexts)
        assert r.contexts['LinearRegression'] == list('raw'), "Single context not added correctly."
        
    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_add
    @mark.validation_contexts_add_list
    def test_validation_contexts_add_context_list(self):
        classname = 'LinearRegression'
        contexts = ['raw', 'clean', 'processed']
        c = Classes()
        c.add_class(LinearRegression())        
        r = ValidationContexts()
        r.add_context(classname, contexts)
        assert all(item in contexts for item in r.contexts['LinearRegression']), \
            "Adding list of contexts is broke."     

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_add
    @mark.validation_contexts_add_tuple
    def test_validation_contexts_add_context_tuple(self):
        classname = 'LinearRegression'
        contexts = ('raw', 'clean', 'processed', 'trained')
        c = Classes()
        c.add_class(LinearRegression())        
        r = ValidationContexts()
        r.add_context(classname, contexts)
        assert all(item in contexts for item in r.contexts['LinearRegression']), \
            "Adding tuple of contexts is broke."     


    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_add
    @mark.validation_contexts_add_ndarray
    def test_validation_contexts_add_context_ndarray(self):
        classname = 'LinearRegression'
        contexts = np.array(['raw', 'clean', 'processed', 'trained', 'fitted'])
        c = Classes()
        c.add_class(LinearRegression())        
        r = ValidationContexts()
        r.add_context(classname, contexts)
        assert all(item in contexts for item in r.contexts['LinearRegression']), \
            "Adding nd array of contexts is broke."     

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_verify_success
    def test_validation_contexts_verify_success(self):
        classname = 'LinearRegression'
        contexts = np.array(['raw', 'clean', 'processed', 'trained', 'fitted'])
        c = Classes()
        c.add_class(LinearRegression())        
        r = ValidationContexts()
        r.add_context(classname, contexts)                
        r.verify()
        assert all(item in contexts for item in r.contexts['LinearRegression']), \
            "Adding nd array of contexts is broke."             

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_remove
    @mark.validation_contexts_remove_str
    def test_validation_contexts_remove_context_str(self):
        classname = 'LinearRegression'
        contexts_add = np.array(['raw', 'clean', 'processed', 'trained', 'fitted'])
        c = Classes()
        c.add_class(LinearRegression())        
        contexts_remove = 'raw'        
        r = ValidationContexts()
        r.add_context(classname, contexts_add)
        r.remove_context(classname, contexts_remove)
        assert ~any(item in contexts_remove for item in r.contexts['LinearRegression']), \
            "Removing string context is broke."                 

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_remove
    @mark.validation_contexts_remove_list
    def test_validation_contexts_remove_context_list(self):
        classname = 'LinearRegression'
        c = Classes()
        c.add_class(LinearRegression())        
        contexts_add = np.array(['raw', 'clean', 'processed', 'trained', 'fitted'])
        contexts_remove = ['raw', 'clean']
        remaining = ['processed', 'trained', 'fitted']
        r = ValidationContexts()
        r.add_context(classname, contexts_add)
        r.remove_context(classname, contexts_remove)
        assert ~any(item in contexts_remove for item in r.contexts['LinearRegression']), \
            "Removing list  context is broke."                             
        assert all(item in remaining for item in r.contexts['LinearRegression']), \
            "Removing ndarray context is broke."                  

    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_remove
    @mark.validation_contexts_remove_ndarray
    def test_validation_contexts_remove_context_ndarray(self):
        classname = 'LinearRegression'
        c = Classes()
        c.add_class(LinearRegression())        
        contexts_add = np.array(['raw', 'clean', 'processed', 'trained', 'fitted'])
        contexts_remove = np.array(['raw', 'clean'])
        remaining = ['processed', 'trained', 'fitted']
        r = ValidationContexts()
        r.add_context(classname, contexts_add)
        r.remove_context(classname, contexts_remove)
        assert ~any(item in contexts_remove for item in r.contexts['LinearRegression']), \
            "Removing list  context is broke."                             
        assert all(item in remaining for item in r.contexts['LinearRegression']), \
            "Removing ndarray context is broke."                  


    @mark.validation
    @mark.validation_contexts
    @mark.validation_contexts_remove
    @mark.validation_contexts_remove_all
    def test_validation_contexts_remove_all_context(self):
        classname = 'LinearRegression'                
        c = Classes()
        c.add_class(LinearRegression())
        contexts_add = np.array(['raw', 'clean', 'processed', 'trained', 'fitted'])
        r = ValidationContexts()
        r.add_context(classname, contexts_add)
        r.remove_context(classname)
        assert len(r.contexts[classname]) == 0, "Not all contexts were removed"                                         