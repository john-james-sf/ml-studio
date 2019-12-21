# =========================================================================== #
#                           SINGLETON METACLASS                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \singleton.py                                                         #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 20th 2019, 8:29:22 am                          #
# Last Modified: Friday December 20th 2019, 8:30:29 am                        #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Provides an alternative to the Singleton pattern, using the Borg superclass.

More information about the Borg 'nonpattern' may be obtained from 
https://learning.oreilly.com/library/view/python-cookbook/0596001673/ch05s23.html

To implement, simply subclass this class and call the __init__ method just
as you always do for every base class's constructor.
 
"""
class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
