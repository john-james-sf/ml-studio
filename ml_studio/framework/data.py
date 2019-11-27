# =========================================================================== #
#                             DATA CLASS                                      #
# =========================================================================== #
"""Data class. Contains data, transformations, and analysis objects."""
from abc import ABC, abstractmethod, ABCMeta
import datetime
import numpy as np
import pandas as pd
import warnings

from ml_studio.utils.data_manager import data_split
from ml_studio.framework.logger import get_logger
# --------------------------------------------------------------------------- #

class Data():
    """Data class."""

    def __init__(self):
        self.logger = get_logger(__name__)
    
    def load_raw(self,X,y=None):
        self.raw_data = {'X': x, 'y': y}
        




