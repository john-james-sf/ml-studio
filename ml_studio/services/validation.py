#!/usr/bin/env python3
# =========================================================================== #
#                          SERVICES: VALIDATION                               #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \validation.py                                                        #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Monday December 9th 2019, 2:07:26 pm                           #
# Last Modified: Monday December 9th 2019, 2:08:13 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Module defines the classes responsible for validating entity classes.

Validation is accomplished by two types of classes, a configuration class and
a validator class. The former are:

    * ValidationAttr : contains the attribute / validator relationship     
    * ValidationRule : contains the attribute / validator / class validation rule
    * Validatrix : invokes the appropriate validator based upon the rule class.

Each attribute has a validator class with a validate method that is 
invoked by the controller. The interface for the validator classes is defined
by the Validator base class.
    
"""
#%%
from abc import ABC, abstractmethod
import builtins
from collections.abc import Iterable
import os
import math
import numbers
import re
import sys
import time

import numpy as np
import pandas as pd

from ml_studio.entities.classes import Classes

# --------------------------------------------------------------------------- #
#                            VALIDATIONRULES                                  #
# --------------------------------------------------------------------------- #
# class ValidationRules(object):
#     """Singleton class to create, maintain and execute validation rules.""" 
#     __instance = None

#     rules = pd.DataFrame()
#     def __new__(cls):
#         if ValidationRules.__instance is None:
#             ValidationRules.__instance = object.__new__(cls)        
#         return ValidationRules.__instance

#     def add_rule(rule):
#         """Adds a rule to the list of rules."""
#         r = dict()
#         r.rule_id = rule.rule_id
#         r.classname = rule.classname
#         r.attribute = rule.attribute
#         r.contexts = rule.contexts
#         r.exceptions = rule.exceptions
#         r.requirements = rule.requirements
#         r.dependencies = rule.dependencies
#         df = pd.DataFrame(data=r)
#         self.rules.append(df)

#     def remove_rules(rule_id):
#         """Removes the rules associated with the rule_ids parameter.

#         Parameter
#         ---------
#         rule_id : str, list, nd.array   
#             A single rule_id, a partial rule_id, or an iterable containing
#             multiples thereof

#         """

#         if len(rule_id) > 1:
#             for rule in rule_id:
#                 self.rules = self.rules[~self.rules.rule_id.str.contains(rule, case=False)]
#         else:
#             self.rules = self.rules[~self.rules.rule_id.str.contains(rule_id, case=False)]

# --------------------------------------------------------------------------- #
#                              VALIDATIONCONTEXTS                             #
# --------------------------------------------------------------------------- #
"""Maintains the relationships between classes and contexts for validation.

Validation context is a user defined abstraction assigned to a class. For 
instance, a DataSet class may have contexts pertaining to the stage of the 
machine learning process e.g. 'raw', 'clean', 'processed'. A DataPackage may
be 'assembled', 'documented', 'published'. A model may be 'fitted', 'tuned'
on a test set, 'final' which means the model is prepared for deployment.

This is a singleton class to ensure that there is a single set of contexts
for the application.

"""
class ValidationContexts:
    __instance = None
    contexts = {}
    classes = Classes()

    def __new__(cls):
        if ValidationContexts.__instance is None:
            ValidationContexts.__instance = object.__new__(cls)        
        return ValidationContexts.__instance    

    def verify(self):
        """Verifies that the context class is valid."""
        classnames = self.classes.get_classnames()
        for k, _ in self.contexts.items():
            if k not in classnames:
                raise ValueError("%s is not a valid class name" % k)                   

    def add_context(self, classname, contexts):
        """Adds a context to the inventory of contexts.

        Parameters
        ----------
        classname : str
            The name of the class to which the contexts apply

        contexts : str or array-like of str
            The contexts that are appropriate for a particular class.

        """
        classnames = self.classes.get_classnames()
        # Confirm classname is valid
        if classname not in classnames:
            raise ValueError("%s is not a valid classname" % classname)

        # Prepare contexts for addition, namely convert to list of strings
        if isinstance(contexts, (list, tuple, np.ndarray, np.generic)):            
            contexts = [str(c) for c in contexts]
        else:
            contexts = list(str(contexts))

        # Add context to existing iterable
        if self.contexts.get(classname):
            self.contexts[classname].extend(contexts)

        # Otherwise create dictionary entry for classname and assign contexts
        if len(contexts) > 1:
            self.contexts[classname] = contexts            
        else:
            self.contexts[classname] = [contexts]


    def remove_context(self, classname, context=None):
        """Removes an individual context or all contexts for a class.

        If context is None, all contexts for a class are removed after 
        obtaining user verification.

        Parameters
        ----------
        classname : str
            The name of the class to which the contexts apply

        contexts : str or list of str, Optional        
            The contexts to remove for the class. If None, all contexts for 
            the class will be removed.

        """         
        if context is None:
            remove_all = input("Are you sure you want to remove all contexts for this class? (y/n)")

            if remove_all in ['Y', 'y', 'Yes', 'yes', 'YES']:
                print('Removing contexts.')
                # removing contexts
                self.contexts[classname] = list()
            else:
                print('Leaving contexts in place. You may prefer to indicate \
                        a specific context to remove.')

        elif len(context)>1:
            for c in context:
                try:
                    self.contexts[classname].remove(c)
                except ValueError:
                    pass

        else:
            self.contexts[classname].remove(context)
            

# --------------------------------------------------------------------------- #
#                              VALIDATIONRULE                                 #
# --------------------------------------------------------------------------- #
class ValidationRule():
    """Class containing all validation rules and the ability to create them.

    A rule consists of:
        * rule_id : concatenation of classname, attribute, and a sequence no.        
        * instance : an instance of the class to which the validation occurs.
        * attribute : the attribute of the class to which the rule applies
        * dependencies : list of rule dictionaries that the current rule 
            depends upon.
        * requirements : a list of requirements dictionaries that must be 
            met. A requirement is a dictionary with one of the following keys:

              * allowed : a list containing the allowed values for a target
              * forbidden : opposite of allowed
              * contains : a list containing values that must be in target
              * excludes : attributes that must be excluded from the target object.
              * keys_rule : rules that apply to the keys of a mapping
              * values_rule : rules that apply to the values of a mapping
              * min, max : values must be greater or equal (less or equal)
              * minlength, maxlength : defines valid lengths for iterables
              * required : boolean. True if attribute is required.
              * regex : regex pattern that must be matched
              * type : the type that must be matched 

    """

    DIRECTIVES = ['allowed', 'forbidden', 'contains', 'excludes', 'keys_rule', 
                    'values_rule', 'min', 'max', 'minlength', 'maxlength',
                    'required', 'regex', 'type']

    def __init__(self, instance, attribute, requirements=None, dependencies=None):
        """Instantiates rule object."""        
                
        self.__instance = instance
        self.__attribute = attribute        
        self.__requirements = requirements or []
        self.__dependencies = dependencies or []

        # Derive classname
        self.classname = instance.__class__.__name__

        # Derive rule_id
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.rule_id = (self.classname + '_' + attribute + '_'  + str(timestr)).lower()   

        # Save all class attributes as instance variable 
        c = Classes()        
        self.attributes = c.get_instance_attributes(self.classname)           

        # Get attribute type for instance
        self.attribute_type = self._get_attribute_type()        
       

    @property
    def instance(self):
        return self.__instance
    
    @instance.setter
    def instance(self, i):
        if i is not None:
            # Save all valid class names
            c = Classes()        
            self.classnames = c.get_classnames()        

            # Obtain classname for current instance
            self.classname = i.__class__.__name__
            
            # Validate classname against all class names
            if self.classname not in self.classnames:
                raise ValueError("instance is not of a valid class.")

            # Save instance
            self.__instance = i

    @property
    def attribute(self):        
        return self.__attribute

    @attribute.setter
    def attribute(self, a):
        if a is not None:

            # Obtain classname for current instance
            self.classname = self.__instance.__class__.__name__           

            # Save all class attributes
            c = Classes()        
            self.attributes = c.get_instance_attributes(self.classname)        

            # Validate attribute against class attributes
            if a not in self.attributes:
                raise ValueError("%s is not a valid attribute for %s class" \
                                % (str(a), str(self.classname)))

            # Update attribute
            self.__attribute = a

            # Update attribute type
            self.attribute_type = self._get_attribute_type()        

    @property
    def requirements(self):
        return self.__requirements

    @property
    def dependencies(self):
        return self.__dependencies

    def _get_attribute(self):
        return getattr(self.instance, self.attribute)    
        

    def _get_attribute_type(self):        
        """Gets the type of the attribute from the instance."""
        try:
            attr_type = type(getattr(self.__instance, self.__attribute)).__name__
        except AttributeError:
            raise KeyError("attribute %s is not a valid attribute for class %s"\
                % (str(self.attribute), str(self.instance)))
        return attr_type


    def _validate_attribute_type(self, value):
        """Validates value accompanying the allowed requirement."""
        # Validate attribute type compatibility if the attribute type is 
        # not None.         
        if self.attribute_type != type(None).__name__:
            if isinstance(value, (list, pd.Series, np.ndarray, tuple, np.generic)):
                for v in value:
                    if type(v).__name__ != self.attribute_type:
                        raise TypeError("variable type %s not compatible \
                            with attribute type %s" % (type(v), self.attribute_type))
            else:
                if type(value).__name__ != self.attribute_type:
                        raise TypeError("variable type %s not compatible \
                            with attribute type %s" % (type(value), self.attribute_type))

    def _validate_attribute(self, value):

        # Get attributes for class from classes
        c = Classes()
        self.attributes = c.get_instance_attributes(self.classname)

        if isinstance(value, (list, tuple, pd.Series, np.array, np.generic)):
            for v in value:
                if v not in self.attributes:
                    raise ValueError("attribute %s is not a valid attribute for \
                        class %s" % (v, self.classname))
        else:
            if value not in self.attributes:
                raise ValueError("attribute %s is not a valid attribute for \
                        class %s" % (v, self.classname))

    def _validate_allowed(self, key=None, value=None):
        self._validate_attribute_type(value)

    def _validate_forbidden(self, key=None, value=None):
        self._validate_attribute_type(value)

    def _validate_contains(self, key=None, value=None):
        self._validate_attribute_type(value)

    def _validate_excludes(self, key=None, value=None):
        self._validate_attribute(value)

    def _validate_min(self, key=None, value=None):
        self._validate_attribute_type(value)
        if not value.__gt__:
            raise ValueError("%s is not valid for the 'min' requirement" % str(value))

    def _validate_max(self, key=None, value=None):
        self._validate_attribute_type(value)
        if not value.__lt__:
            raise ValueError("%s is not valid for the 'max' requirement" % str(value))        

    def _validate_length(self, key=None, value=None):
        attribute = self._get_attribute()        
        try:
            attribute.__len__
        except(AttributeError):
            raise ValueError("%s does not have a 'len' attribute" % str(self.attribute))                

    def _validate_minlength(self, key=None, value=None):
        self._validate_length(key, value)

    def _validate_maxlength(self, key=None, value=None):
        self._validate_length(key, value)

    def _validate_required(self, key=None, value=None):
        if not isinstance(value, bool):        
            raise TypeError("%s must be bool for the 'required' requirement." % str(value))

    def _validate_regex(self, key=None, value=None):
        try:
            re.compile(value)
        except TypeError:
            raise ValueError("%s is not valid regex" % str(value))

    def _validate_type(self, key=None, value=None):
        # Obtain builtin types
        try : 
            builtin_types = [t.__name__ for t in __builtin__.__dict__.itervalues() if\
                 isinstance(t, type)]
        except:
            builtin_types = [getattr(builtins, d).__name__ for d in dir(builtins) if \
                isinstance(getattr(builtins, d), type)]

        # Validate against built in and custom types                
        if isinstance(value, type):
            type_string = value.__name__
            if type_string not in builtin_types:
                c = Classes()
                classnames = c.get_classnames()
                if type_string not in classnames:
                    raise ValueError("%s is not a valid built_in or custom type" % str(type_string))

        else:
            raise ValueError("%s must be a valid custom or builtin type" % str(value))

    def _dispatcher(self, requirement):       

        # Extract requirement keys and values
        k = list(requirement.keys())[0]
        v = list(requirement.values())[0]

        # Confirm that there is but a single requirement
        if isinstance(k,(list, tuple, np.ndarray, np.generic)):
            raise KeyError("A requirement must have a single key")        

        # Create dispatch dictionary
        dispatcher = {'allowed': self._validate_allowed,
                      'forbidden': self._validate_forbidden,
                      'contains': self._validate_contains,
                      'excludes': self._validate_excludes,      
                      'min': self._validate_min,
                      'max':  self._validate_max,
                      'minlength': self._validate_minlength,
                      'maxlength':  self._validate_maxlength,                      
                      'required': self._validate_required,
                      'regex': self._validate_regex,
                      'type': self._validate_type}

        validation = dispatcher.get(k)
        validation(key=k, value=v)      
        

    def verify_requirement(self, requirement):
        """Validates a requirement dictionary."""

        # Confirm requirement contains a single directive
        directive = requirement.keys()
        if len(directive) > 1:
            raise KeyError("A requirement must have a single directive")
        
        # Validate each directive is valid
        for directive in requirement.keys():
            if ~(directive in self.DIRECTIVES):
                raise ValueError("%s is not a valid requirement" % str(directive))

        # Validate the requirement
        self._dispatcher(requirement)        

    def verify_dependency(self, dependency):
        """Validates the dependency parameter."""

        # Confirm a dependency has a single attribute key
        attribute = dependency.keys()
        if len(attribute) > 1:
            raise KeyError("A dependency must have a single attribute")

        # Confirm that attribute is valid
        if attribute not in self.attributes:
            raise KeyError("%s is not a valid attribute for %s class." \
                % (attribute, self.classname))

        # Validate the requirements associated with the dependency    
        for requirement in dependency.values():
            self._dispatcher(requirement)


    def verify(self):
        """Validates the validation rule instance for completeness."""

        # Validate instance is a valid ML Studio class
        if self.classname not in self.classnames:
            raise ValueError("%s is not an instance of a valid class" % str(self.instance))
        
        # Validate attribute
        if self.attribute not in self.attributes:
            raise ValueError("%s is not an instance of a attribute for the %s \
                class" % str(self.attribute, self.classname))

        # Validate requirements
        for requirement in self.requirements:
            self.verify_requirement(requirement)

        # Validate dependencies
        for dependency in self.dependencies:
            self.verify_dependency(dependency)


    def add_requirement(self, requirement):
        """Adds a requirement to the rule attribute

        Parameters
        ----------
        requirement : dict
            Key value pair consisting of a directive and a value 

        """        
        # Extract directive 
        directive = list(requirement.keys())[0]

        # Confirm the requirement is a dictionary
        if not isinstance(requirement, dict):
            raise TypeError("requirement %s must be a dictionary object" % str(requirement))

        # Validate the directive is one of the valid directives
        if directive not in self.DIRECTIVES:
            raise KeyError("%s is not a valid requirement. see \
                Class().DIRECTIVES" % directive)
        
        # Validate the directive value
        self._dispatcher(requirement)

        # Add requirement if no errors
        self.__requirements.append(dict(requirement.items()))
      


    def remove_requirement(self, requirement=None):
        """Removes requirement from a rule. 

        The requirement parameter may be one of several formats. If requirement
        is None, all requirements will be removed subject to verification.
        If requirement is a string, all requirements that match the string 
        will be removed, subject to verification. If the requirement 
        parameter is a dict, only the requirement in which both the key
        and value match, will be removed.
        
        Parameters
        ----------
        requirement : str, dict. Optional 
            String containing the key for a requirement or a dict containing 
            a key value pair containing a directive and a value.
        
        """
        if requirement:
            if isinstance(requirement,str):
                for i, r in enumerate(self.requirements):
                    for k, v in r.items():
                        if k == requirement:
                            del self.__requirements[i]

            elif isinstance(requirement, dict):
                for k, v in requirement.items():
                    for i, r in enumerate(self.requirements):
                        for k2, v2 in r.items():
                            if k==k2 and v==v2:
                                del self.__requirements[i]

        else:
            remove_all = input("Are you sure you want to remove all requirements for this class? (y/n)")

            if remove_all in ['Y', 'y', 'Yes', 'yes', 'YES']:
                print('Removing requirements.')
                # removing contexts
                self.__requirements = list()
            else:
                print('Leaving requirements in place. You may prefer to indicate \
                        a specific requirement to remove.')            

    def add_dependency(self, dependency):
        """Adds a dependency to the rule attribute

        Parameters
        ----------
        dependency : dict
            Key value pair consisting of an attribute and a list of 
            requirements 

        """        
        # Confirm the dependency is a dictionary 
        if not isinstance(dependency, dict):
            raise TypeError("dependency %s must be a dictionary object" % str(dependency))        
        
        # Extract dependency attribute
        attribute = list(dependency.keys())[0]

        # Validate the dependency attribute is valid 
        if attribute not in self.attributes:
            raise KeyError("%s is not a valid requirement. see \
                Class().REQUIREMENTS" % attribute)
        
        # Validate the dependency requirements        
        for requirement in dependency.values():            
            # requirement is a dict
            for k in requirement.keys():
                if k not in self.DIRECTIVES:
                    raise ValueError("%s is not a valid requirement" % str(k))
                # Validate the requirement
                self._dispatcher(requirement)

        # Add dependency if no errors
        self.__dependencies.append(dict(dependency.items()))


    def remove_dependency(self, dependency=None):
        """Removes a dependency from a rule.

        The dependency parameter may be one of several formats. If dependency
        is None, all dependencies will be removed subject to verification.
        If dependency is a string, all dependencies that match the string 
        will be removed, subject to verification. If the dependency 
        parameter is a dict, only the dependency in which both the key
        and value match, will be removed.
        
        Parameters
        ----------
        dependency : str, dict. Optional 
            String containing the key for a dependency or a dict containing 
            a key value pair containing the dependency and its requirements.
        
        """
        if dependency:
            if isinstance(dependency,str):
                for i, d in enumerate(self.__dependencies):
                    for k, v in d.items():
                        if k == dependency:
                            del self.__dependencies[i]

            elif isinstance(dependency, dict):
                for i, (k, v) in enumerate(dependency.items()):
                    for d in self.__dependencies:
                        for k2, v2 in d.items():
                            if k==k2 and v==v2:
                                del self.__dependencies[i]

        else:
            remove_all = input("Are you sure you want to remove all dependencies for this class? (y/n)")

            if remove_all in ['Y', 'y', 'Yes', 'yes', 'YES']:
                print('Removing dependencies.')
                # removing contexts
                self.__dependencies = list()
            else:
                print('Leaving requirements in place. You may prefer to indicate \
                        a specific requirement to remove.')     

# --------------------------------------------------------------------------- #
#                                VALIDATOR                                    #
# --------------------------------------------------------------------------- #

class Validator(ABC):
    """Abstract base class from which all Validator classes inherit."""

    def __init__(self):
        self.attribute = None
        # Return results
        self.result = None
        self.error_message = None


    @abstractmethod
    def validate(self, attribute):
        """Performs the validation & updates the result & message attributes.

        Parameters
        ----------
        attribute : str, array-like. Required.
            The name of the attribute to validated by this validator. 

        Returns
        -------
        self : Bool
            True if the validation passed. False otherwise.

        error_message : str, Defaults to None
            An error message or None if the validation passed.        

        """
        pass

    @abstractmethod
    def message(self, message_id=None):
        """Formats the error message."""
        pass

# --------------------------------------------------------------------------- #
#                            ISSTR VALIDATOR                                  #
# --------------------------------------------------------------------------- #    
class IsStr(Validator):

    def __init__(self):
        super(IsStr, self).__init__()     

    def validate(self,attribute):        
        self.attribute = attribute
        if isinstance(attribute, str):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a string."  \
                                % str(self.attribute)

# --------------------------------------------------------------------------- #
#                            ISINT VALIDATOR                                  #
# --------------------------------------------------------------------------- #    
class IsInt(Validator):

    def __init__(self):
        super(IsInt, self).__init__()     

    def validate(self,attribute):     
        self.attribute = attribute   
        if isinstance(attribute, int):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not an integer."  \
                            % str(self.attribute)    

# --------------------------------------------------------------------------- #
#                            ISFLOAT VALIDATOR                                #
# --------------------------------------------------------------------------- #    
class IsFloat(Validator):

    def __init__(self):
        super(IsFloat, self).__init__()     

    def validate(self,attribute):     
        self.attribute = attribute   
        if isinstance(attribute, float):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a float."  \
                            % str(self.attribute)

# --------------------------------------------------------------------------- #
#                            ISNUMBER VALIDATOR                               #
# --------------------------------------------------------------------------- #    
class IsNumber(Validator):

    def __init__(self):
        super(IsNumber, self).__init__() 
    

    def validate(self, attribute):
        self.attribute = attribute
        if isinstance(self.attribute, numbers.Number):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a number."  \
                            % str(self.attribute)

# --------------------------------------------------------------------------- #
#                            ISBOOL VALIDATOR                                 #
# --------------------------------------------------------------------------- #    
class IsBool(Validator):

    def __init__(self):
        super(IsBool, self).__init__()     

    def validate(self, attribute):
        self.attribute = attribute
        if isinstance(self.attribute, bool):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a boolean."  \
                            % str(self.attribute)

# --------------------------------------------------------------------------- #
#                            IS1DARRAY VALIDATOR                              #
# --------------------------------------------------------------------------- #    
class Is1dArray(Validator):

    def __init__(self):
        super(Is1dArray, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if isinstance(self.attribute, (np.ndarray, np.generic)):
            if len(self.attribute.shape) == 1:
                self.result = True
            else:
                self.result = False
                self.message()
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a 1D array."  \
                            % str(self.attribute)
        
# --------------------------------------------------------------------------- #
#                            ISNDARRAY VALIDATOR                              #
# --------------------------------------------------------------------------- #    
class IsNdArray(Validator):

    def __init__(self):
        super(IsNdArray, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if isinstance(self.attribute, (np.ndarray, np.generic)):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not an ND array."  \
                            % str(self.attribute)

# --------------------------------------------------------------------------- #
#                          ISARRAYLIKE VALIDATOR                              #
# --------------------------------------------------------------------------- #    
class IsArrayLike(Validator):

    def __init__(self):
        super(IsArrayLike, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if isinstance(self.attribute, (list, tuple, np.ndarray, np.generic)):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not an array-like."  \
                            % str(self.attribute)                      

# --------------------------------------------------------------------------- #
#                           ISITERABLE VALIDATOR                              #
# --------------------------------------------------------------------------- #    
class IsIterable(Validator):

    def __init__(self):
        super(IsIterable, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if isinstance(self.attribute, Iterable):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not an iteratable."  \
                            % str(self.attribute)   

# --------------------------------------------------------------------------- #
#                            ISSERIES VALIDATOR                               #
# --------------------------------------------------------------------------- #    
class IsSeries(Validator):

    def __init__(self):
        super(IsSeries, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if isinstance(self.attribute, pd.Series):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a Series."  \
                            % str(self.attribute)   
  

# --------------------------------------------------------------------------- #
#                           ISDATAFRAME VALIDATOR                             #
# --------------------------------------------------------------------------- #    
class IsDataFrame(Validator):

    def __init__(self):
        super(IsDataFrame, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if isinstance(self.attribute, pd.DataFrame):
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not a DataFrame."  \
                            % str(self.attribute) 

# --------------------------------------------------------------------------- #
#                              ISEMPTY VALIDATOR                              #
# --------------------------------------------------------------------------- #    
class IsEmpty(Validator):

    def __init__(self):
        super(IsEmpty, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if not self.attribute:
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is not empty."  \
                            % str(self.attribute)                          

# --------------------------------------------------------------------------- #
#                              ISNOTEMPTY VALIDATOR                           #
# --------------------------------------------------------------------------- #    
class IsNotEmpty(Validator):

    def __init__(self):
        super(IsNotEmpty, self).__init__() 

    def validate(self, attribute):
        self.attribute = attribute        
        if self.attribute:
            self.result = True
        else:
            self.result = False
            self.message()

    def message(self, message_id=None):
        self.error_message = "Attribute %s is empty."  \
                            % str(self.attribute)                                     

