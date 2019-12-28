#!/usr/bin/env python3
# =========================================================================== #
#                          VALIDATORS : CORE                                  #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \core.py                                                              #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Saturday December 21st 2019, 8:22:27 am                        #
# Last Modified: Friday December 27th 2019, 10:17:52 am                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the core Validator classes. 

Validation is governed by the following validator classes:

    Validators
    ----------
    * Validator : Abstract base class for the following Validator classes.
    * Validerator : Rule Iterator for Validator classes.
    * StringValidator : Ensures a property value is a valid string.
    * BooleanValidator : Ensures a property value is a valid Boolean.
    * AllowedValuesValidator : Ensures a property value is among the set of
        allowed values.
    * ForbiddenValuesValidator : Ensures a property value is not among 
        the set of forbidden values.        
    * NumberValidator : Ensures a property value is a valid number
    * IntegerValidator : Ensures a property value is a valid integer    
    * ArrayValidator : Ensures a property value is a valid array.
    
    
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
from ml_studio.services.validation.validators.core import BaseValidator 

# =========================================================================== #
#                           CANVAS VALIDATORS                                 #
# =========================================================================== #        
class CanvasValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)


class CanvasTitleValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasTitleValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasLegendValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasLegendValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasMarginsValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasMarginsValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)


class CanvasSizeValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasSizeValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)


class CanvasFontValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasFontValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)


class CanvasColorBackgroundValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorBackgroundValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)


class CanvasColorScaleValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorScaleValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)


class CanvasColorAxisDomainValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisDomainValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)


class CanvasColorAxisScalesValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisScalesValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)    

class CanvasColorAxisBarStyleValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarStyleValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasColorAxisBarPositionValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarPositionValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasColorAxisBarBoundaryValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarBoundaryValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasColorAxisBarTicksValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarTicksValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasColorAxisBarTickStyleValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarTickStyleValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasColorAxisBarTickFontValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarTickFontValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasColorAxisBarNumbersValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarNumbersValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)

class CanvasColorAxisBarTitleValidator(BaseValidator):
    """Validates the Canvas object."""

    def __init__(self, instance, target_name, parent_name,
                 object_type="Class Validator", **kwargs):
        super(CanvasColorAxisBarTitleValidator, self).__init__(instance=instance,
                                              target_name=target_name,
                                              parent_name=parent_name,
                                              object_type=object_type,
                                              **kwargs)
