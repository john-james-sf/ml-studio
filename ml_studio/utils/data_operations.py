# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \data_operations.py                                                   #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Wednesday December 25th 2019, 5:49:43 pm                       #
# Last Modified: Wednesday December 25th 2019, 6:04:26 pm                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Utility functions used to analyze and manipulate data."""
import numpy as np
import pandas as pd

def is_array(a):
    if isinstance(a, (pd.Series, np.ndarray, list, tuple)):
        return True
    else:
        return False

def is_simple_array(a):
    if isinstance(a, (list, tuple)):
        return True
    else:
        return False

def is_homogeneous_array(a):
    if isinstance(a, (pd.Series, pd.Index, np.ndarray)):
        return True
    else:
        return False

def is_numpy_convertable(a):
    """Evaluate whether an array can be converted to a numpy array."""
    return hasattr(a, "__array__") or hasattr(a, "__array_interface__")

def to_native_type(a):
    """Converts a numpy scalar or array to python native scalar or list."""
    # Handle scalar
    if np.isscalar(a) and hasattr(a, "item"):
        return a.item()
    # Handle simplearray
    if is_simple_array(a):
        return [to_native_type(e) for e in a]
    # Handle numpy array
    if isinstance(a, np.ndarray) and a.ndim == 0:
        return a.item()
    # Handle pandas Series
    if isinstance(a, (pd.Series, pd.Index)):
        return [to_native_type(e) for e in a]
    # Handle convertable array
    if is_numpy_convertable(a):
        return to_native_type(np.array(a))
    return a

def coerce_homogeneous_array(value, kind=None, force_numeric=False):
    """Coerces homogeneous array to numeric numpy array if possible.

    This code was inspired by:
    ******************************************************************
    * Title : copy_to_readonly_numpy_array
    * Author(s) : Nicolas Kruchten, Jon Mease  
    * Date : December 26, 2019
    * Version 4.4.1
    * Availability : https://github.com/plotly/plotly.py/blob/90f237060092d86d5b8bd9ec8cf158e0e5a7f728/packages/python/plotly/_plotly_utils/basevalidators.py#L56
    ******************************************************************
    """
    # Initialize kind
    if not kind:
        kind = ()
    elif isinstance(kind, str):
        kind = (kind,)

    # Extract first kind
    first_kind = kind[0] if kind else None

    # Designate numeric kinds and default types
    numeric_kinds = {"u", "i", "f"}
    kind_default_dtypes = {"u": "uint32", "i": "int32", "f": \
        "float64", "O": "object", "S": "string"}        
    
    # Coerce pandas Series and Index objects
    if isinstance(value, (pd.Series, pd.Index)):
        if value.dtype.kind is numeric_kinds:
            # Extract the numeric numpy array
            value = value.values
    # If value is not a numpy array, then attempt to convert
    if not isinstance(value, np.ndarray):            
        if is_numpy_convertable(value):
            return(coerce_homogeneous_array(value, kind, 
            force_numeric=force_numeric))
        else:
            # value is a simple array
            value = [to_native_type(e) for e in value]

            # Lookup dtype for the requested kind, if any
            dtype=kind_default_dtypes.get(first_kind, None)

            # Construct new array from list
            value = np.array(value, order="C", dtype=dtype)
    elif value.dtype.kind in numeric_kinds:
        # value is a homogeneous array
        if kind and value.dtype.kind not in kind:
            # Kind(s) specified but not matched
            # Convert to the default type for the first kind
            dtype = kind_default_dtypes.get(first_kind, None)
            value = np.ascontiguousarray(value.astype(dtype))
        else:
            # Either no kind was requested or requested kind is satisfied
            value = np.ascontiguousarray(value.copy())
    else:
        # value is a non-numeric homogeneous array
        value = value.copy()

    # Handle force numeric parameter
    if force_numeric and value.dtype.kind not in  numeric_kinds:
        raise ValueError("Unable to force non-numeric to numeric")

    # Force non-numeric arrays to object type
    if "U" not in kind and value.dtype.kind \
        not in kind_default_dtypes.keys():
            value = np.array(value, dtype="object") 

    return value

