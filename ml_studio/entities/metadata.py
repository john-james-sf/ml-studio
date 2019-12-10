#!/usr/bin/env python3
# =========================================================================== #
#                          ENTITIES: MANAGE                                   #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \metadata.py                                                          #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Monday December 9th 2019, 7:29:41 am                           #
# Last Modified: Monday December 9th 2019, 5:23:31 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines classes that manage metadata.

Metadata, which is the essential data about data, is almost important as the
data itself. This module provides the data and behaviors that represent
metadata for the entity classes in ML Studio.

There are three main categories of metadata: Technical metadata, business 
metadata and process metadata. 

- Technical metadata defines the objects and processes in ML Studio, as seen 
from a technical point of view. The technical metadata includes the system 
metadata, which defines the data packages and structures such as tables, fields, 
data types.
- Business metadata tells you what data you have, where they come from, 
what they mean and what their relationship is to other objects. 
- Process metadata is used to describe the results of various operations 
such as training and optimization processes. This includes information such 
start time, end time, CPU seconds used, and training set sizes.

Metadata captured for technical, business and process objects and
events can be characterized as:

- Descriptive, or used for discovery and identification and including such 
information as title, author, description and keywords. 
- Structural, which shows how information is put together – datasets within 
a data package.
- Administrative, metadata captures information such as when, how and by whom
a resource was created. 

"""
#%%
import json
import math
import os
import time
import uuid
import warnings

from abc import ABC, abstractmethod, ABCMeta
from datapackage import Package, Resource, Profile, exceptions, validate, infer
from smalluuid import SmallUUID

from ..utils.misc import snake, proper

# --------------------------------------------------------------------------- #
#                                PACKAGE                                      #
# --------------------------------------------------------------------------- #

class MetaData(ABC):
    """The abstract base class for all metadata classes. 

    This class defines the common API and properties for all metadata classes.

    Parameters
    ----------
    name : str
        A human readable name in lower snake case with alphanumeric characters
        only. The name parameter should have no spaces; therefore, underscores 
        and hyphens may be used as separators. Spaces will be replaced
        with underscores. With the exception of underscores and hyphens,
        special characters will be removed from the name. 

    title : str
        A title for the package in capital case. Used for visualizations

    description : str
        A brief description of the package, its purpose and contents

    url : str
        The url for the remote location of the resource. This may be a Git LFS 
        repository or a cloud-based data warehouse.

    basedir : str
        The root directory in which the resource and its metadata are stored.

    storage : str
        The storage backend used. This can be 'sql', 'bigquery', or 
        its default 'hdf5'

    storage_options : dict
        Options for the storage backend.

    licensing : str
        The license by which access and usage rights are defined.

    sources : list of dicts
        A list of dictionaries containing the source title, a url, and 
        an optional email address.

    author_name : str
        String containing authors name

    author_email : str
        Author's email address

    author_website : str
        Author's website    

    version : str
        Version numbers follow the major.minor.patch semantic versioning
        standard.

    
    """
    
    def __init__(self, name, basedir, title=None, description=None, 
                 project=None, phase=None, url=None, storage='hdf5', 
                 storage_options=None, licensing='BSD', sources=None, 
                 author_name=None, author_email=None, author_website=None, 
                 version='0.1.0'):                 
        """Instantiates an object of the Package class."""
                
        self.name = name
        self.title = title
        self.description = description
        self.project = project
        self.phase = phase
        self.url = url
        self.basedir = basedir
        self.storage = storage
        self.storage_options = storage_options        
        self.licensing = licensing
        self.sources = sources
        self.author_name = author_name
        self.author_email = author_email
        self.author_website = author_website
        self.version = version

        # Generate id
        self.id = SmallUUID()
        # Create title if not provided


        self.filename = None
        self.metadata_filename = None

        # Create identifier
        self.id = SmallUUID()

        # Confirm name is snake case and reformat if appropriate
        new_name = snake(self.name) 
        if new_name != self.name:
            warnings.warn("The 'name' parameter was modified from %s to %s" % (self.name, new_name))
            self.name = new_name

        # 

        

        




# %%
