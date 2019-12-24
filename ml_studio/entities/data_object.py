#!/usr/bin/env python3
# =========================================================================== #
#                          DATA PACKAGE: PACKAGE                              #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \package.py                                                           #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Monday December 9th 2019, 6:00:49 am                           #
# Last Modified: Monday December 9th 2019, 6:01:12 am                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the Package class.

Machine learning and deep learning practitioners must acquire, prepare,
analyze and manage data sources across several stages of the machine learning
pipeline. This module defines the Package class, which allows one to 
aggregate all datasets for a machine learning project.

"""
import os
import time
import math

from datapackage import Package, Resource, Profile, Exceptions, validate, infer

# --------------------------------------------------------------------------- #
#                                PACKAGE                                      #
# --------------------------------------------------------------------------- #

class DataPackage():
    """An aggregation of data resources and behaviors to manage them.

    Package objects contains the data, its metadata, and a log of preparation,
    cleaning, transformations, standardizations, and all activities 
    performed on member data objects.  

    Parameters
    ----------
    template : Template object
        Template containing Package class configurations. 

    name : str
        A human readable name in snake case

    title : str
        A title for the package in capital case. Used for visualizations

    description : str
        A brief description of the package, its purpose and contents

    url : str
        The URL for the package. This may be the link to a github LFS 
        repository or an otherwise external data source

    basedir : str
        The root directory for the package and its contents. 

    author_name : str
        String containing authors name

    author_email : str
        Author's email address

    author_website : str
        Author's website

    storage : str
        Storage backends. Currently supported backends include 'sql', 
        'bigquery', and 'csv.

    storage_options : dict
        storage options to use for storage creation

    """
    def __init__(self, template=None, name=None, title=None, description=None, url=None,
                 basedir=None, author_name=None, author_email=None, 
                 author_website=None, storage='csv', storage_options=None):                 
        """Instantiates an object of the Package class."""
        
        self.template = template
        self.name = name
        self.title = title
        self.description = description
        self.url = url
        self.basedir = basedir
        self.author_name = author_name
        self.author_email = author_email
        self.author_website = author_website
        self.storage = storage
        self.storage_options = storage_options

        # Components
        self._id = None
        self._metadata = None
        self._log = None
        self._datasets = []

        # 
#%%  
import os
repr(getpass.getuser())


# %%
