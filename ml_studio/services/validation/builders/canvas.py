#!/usr/bin/env python3
# =========================================================================== #
#                          VALIDATION: BUILDERS                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \builders.py                                                          #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 27th 2019, 12:53:50 am                         #
# Last Modified: Friday December 27th 2019, 1:34:31 am                        #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the Validator Composite Builder classes. 

The Builder classes include:

    Builders
    ----------
    * ValidatorBuilder : Abstract base class for the validation Builders.
    * ClassValidatorBuilder : A collection of Builders for class validators.
    * AttributeValidatorBuilder : A collection of attribute validation Builders.
    * ValidationRuleSetBuilder : A set of validation RuleSet Builders
    * ValidationRuleBuilder : A set of validation Rule Builders
    * ValidationConditionBuilder : A set of validation condition Builders.    

    The above classes represent a tree-like Composite Pattern starting with
    the ClassValidatorBuilder at the top of the hierarchy to the 
    ValidationConditionBuilder at the bottom.    
    
"""
#%%
from abc import ABC, abstractmethod, abstractproperty
import builtins
from collections.abc import Iterable, Iterator
from datetime import datetime
import getpass
import textwrap
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from ml_studio.utils.misc import format_text
from ml_studio.visualate.canvas import  Canvas
from ml_studio.visualate.canvas import  CanvasTitle
from ml_studio.visualate.canvas import  CanvasLegend
from ml_studio.visualate.canvas import  CanvasMargins
from ml_studio.visualate.canvas import  CanvasSize
from ml_studio.visualate.canvas import  CanvasFont
from ml_studio.visualate.canvas import  CanvasColorBackground
from ml_studio.visualate.canvas import  CanvasColorScale
from ml_studio.visualate.canvas import  CanvasColorAxisDomain
from ml_studio.visualate.canvas import  CanvasColorAxisScales
from ml_studio.visualate.canvas import  CanvasColorAxisBarStyle
from ml_studio.visualate.canvas import  CanvasColorAxisBarPosition
from ml_studio.visualate.canvas import  CanvasColorAxisBarBoundary
from ml_studio.visualate.canvas import  CanvasColorAxisBarTicks
from ml_studio.visualate.canvas import  CanvasColorAxisBarTickStyle
from ml_studio.visualate.canvas import  CanvasColorAxisBarTickFont
from ml_studio.visualate.canvas import  CanvasColorAxisBarNumbers
from ml_studio.visualate.canvas import  CanvasColorAxisBarTitle

from ml_studio.services.validation.builders.core import ValidatorBuilder

from ml_studio.services.validation.validators.canvas import CanvasValidator
from ml_studio.services.validation.validators.canvas import CanvasTitleValidator
from ml_studio.services.validation.validators.canvas import CanvasLegendValidator
from ml_studio.services.validation.validators.canvas import CanvasMarginsValidator
from ml_studio.services.validation.validators.canvas import CanvasSizeValidator
from ml_studio.services.validation.validators.canvas import CanvasFontValidator
from ml_studio.services.validation.validators.canvas import CanvasColorBackgroundValidator
from ml_studio.services.validation.validators.canvas import CanvasColorScaleValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisDomainValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisScalesValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarStyleValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarPositionValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarBoundaryValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarTicksValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarTickStyleValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarTickFontValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarNumbersValidator
from ml_studio.services.validation.validators.canvas import CanvasColorAxisBarTitleValidator

# =========================================================================== #
#                        CANVAS VALIDATOR BUILDERS                            #
# =========================================================================== #             
class CanvasValidatorBuilder(ValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        instance = Canvas()
        target_name = "Canvas"
        parent_name = "Visualate"
        self._validator = CanvasValidator(instance, target_name, parent_name)
    
    @property
    def validator(self):
        validator = self._validator
        self.reset()
        return validator
    
    def with_component(self, component):
        self._validator.add_component(component)   


class CanvasTitleValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasTitle()
        target_name = "CanvasTitle"
        parent_name = "Canvas"
        self._validator = CanvasTitleValidator(instance, target_name, parent_name)

class CanvasLegendValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasLegend()
        target_name = "CanvasLegend"
        parent_name = "Canvas"
        self._validator = CanvasLegendValidator(instance, target_name, parent_name)

class CanvasMarginsValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasMargins()
        target_name = "CanvasMargins"
        parent_name = "Canvas"
        self._validator = CanvasMarginsValidator(instance, target_name, parent_name)

class CanvasSizeValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasSize()
        target_name = "CanvasSize"
        parent_name = "Canvas"
        self._validator = CanvasSizeValidator(instance, target_name, parent_name)

class CanvasFontValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasFont()
        target_name = "CanvasFont"
        parent_name = "Canvas"
        self._validator = CanvasFontValidator(instance, target_name, parent_name)

class CanvasColorBackgroundValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorBackground()
        target_name = "CanvasColorBackground"
        parent_name = "Canvas"
        self._validator = CanvasColorBackgroundValidator(instance, target_name, parent_name)

class CanvasColorScaleValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorScale()
        target_name = "CanvasColorScale"
        parent_name = "Canvas"
        self._validator = CanvasColorScaleValidator(instance, target_name, parent_name)

class CanvasColorAxisDomainValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisDomain()
        target_name = "CanvasColorBackground"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisDomainValidator(instance, target_name, parent_name)

class CanvasColorAxisScalesValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisScales()
        target_name = "CanvasColorAxisScales"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisScalesValidator(instance, target_name, parent_name)

class CanvasColorAxisBarStyleValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarStyle()
        target_name = "CanvasColorAxisBarStyle"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarStyleValidator(instance, target_name, parent_name)

class CanvasColorAxisBarPositionValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarPosition()
        target_name = "CanvasColorAxisBarPosition"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarPositionValidator(instance, target_name, parent_name)

class CanvasColorAxisBarBoundaryValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarBoundary()
        target_name = "CanvasColorAxisBarBoundary"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarBoundaryValidator(instance, target_name, parent_name)

class CanvasColorAxisBarTicksValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarTicks()
        target_name = "CanvasColorAxisBarTicks"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarTicksValidator(instance, target_name, parent_name)

class CanvasColorAxisBarTickStyleValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarTickStyle()
        target_name = "CanvasColorAxisBarTickStyle"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarTickStyleValidator(instance, target_name, parent_name)

class CanvasColorAxisBarTickFontValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarTickFont()
        target_name = "CanvasColorAxisBarTickFont"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarTickFontValidator(instance, target_name, parent_name)

class CanvasColorAxisBarNumbersValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarNumbers()
        target_name = "CanvasColorAxisBarNumbers"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarNumbersValidator(instance, target_name, parent_name)

class CanvasColorAxisBarTitleValidatorBuilder(CanvasValidatorBuilder):
    """Abstract class for syntactic validator builders."""

    def __init__(self):
        self.reset()

    def reset(self):
        instance = CanvasColorAxisBarTitle()
        target_name = "CanvasColorAxisBarTitle"
        parent_name = "Canvas"
        self._validator = CanvasColorAxisBarTitleValidator(instance, target_name, parent_name)

