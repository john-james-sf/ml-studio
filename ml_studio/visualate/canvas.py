#!/usr/bin/env python3
# =========================================================================== #
#                                  CANVAS                                     #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \canvas.py                                                            #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Sunday December 15th 2019, 7:17:51 am                          #
# Last Modified: Tuesday December 17th 2019, 1:01:56 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Classes that manage plot 'canvas' configurations. 

The configurable options for a plot are copious and it could be a burdonsome to
include these configuration options in each plot class. This would lead
to bloated interfaces that are difficult to read, let alone maintain.

The Canvas module allows users to specify configurable options separate from
the instantiation and rendering of plots. The Canvas object, containing all
plot configuration options, is passed into the constructors of the individual 
plot classes. This allows users the flexibility of defining the 'canvas' 
while reducing the burdon on the plot classes.  

The module is comprised of the following classes.

    Container 
    ---------
    * Canvas : Container class for Canvas component classes    

    Components
    ----------
    * CanvasComponent : Abstract base class for the following classes
    * CanvasTitle	:	Sets font and position of the plot title
    * CanvasLegend	:	Sets style, font, position and behavior of the legend
    * CanvasMargins	:	Sets plot margins
    * CanvasSize	:	Sets plot width and height
    * CanvasFont	:	Sets family, size and color of fonts
    * CanvasColorBackground	:	Sets plot and page background colors
    * CanvasColorScale	:	Sets sequential, divergent and colorway scales
    * CanvasColorAxisDomain	:	Sets min, max, and mid values of the color scales
    * CanvasColorAxisScales	:	Sets color scale
    * CanvasColorAxisBarStyle	:	Sets color axis bar thickness, length and color
    * CanvasColorAxisBarPosition	:	Sets the position of the color axis color bar
    * CanvasColorAxisBarBoundary	:	Sets color axis border and outline color and width
    * CanvasColorAxisBarTicks	:	Sets parameters for ticks
    * CanvasColorAxisBarTickStyle	:	Sets the style of the ticks 
    * CanvasColorAxisBarTickFont	:	Sets the font of the ticks.
    * CanvasColorAxisBarNumbers	:	Set number format
    * CanvasColorAxisBarTitle	:	Sets the axis bar title family, size and color.

"""
import os
import time

import autopep8
from abc import ABC, abstractmethod, ABCMeta
# --------------------------------------------------------------------------- #
#                                Canvas                                       #
# --------------------------------------------------------------------------- #
class Canvas():    
    """A container class holding the various Canvas components. 
    
    A Canvas is a collection of CanvasComponents, each of which contains
    a set of related visualization configuration parameters. Each 
    CanvasComponent class exposes its parameters are accessor properties. At
    instantiation, the parameters are set to their default values.

    """    
    def __init__(self):
        self.__components = {}

    def print_components(self):
        """Prints component parameters and values to sysout."""
        if bool(self.__components):
            for component in self.__components.values():
                component.print_parameters()
        else:
            print("The Canvas object has no parameters.")

    def add_component(self, component):
        """Adds a CanvasComponent object to the collection.
        
        Parameters
        ----------
        component : CanvasComponent object
            Class containing a group of parameters and accessors.

        Returns
        -------
        self
        
        """

        component_name = component.__class__.__name
        if self.__components[component_name]:
            raise Exception("CanvasComponent %s already exists in the Canvas."\
                % component_name)
        else:
            self.__components[component_name] = component
        
        return self

    def update_component(self, component):
        """Updates an existing CanvasComponent with values from component.

        Components are uniquely indexed by their class names. The existing
        CanvasComponent is obtained via the class name of component. Then,
        the existing CanvasComponent object parameter dictionary is updated 
        with the values from the component parameter dictionary.

        Parameters
        ----------
        component : CanvasComponent object
            Object containing the parameter values to update.

        Returns
        -------
        self

        """ 

        component_name = component.__class__.__name
        if not self.__components[component_name]:
            raise Exception("CanvasComponent object %s does not exist." \
                % component_name)
        else:
            parameters = component.get_parameters()
            self.__components[component_name].update(parameters)

        return self

    def delete_component(self, component_name=None):
        """Removes a CanvasComponent object from the container.

        Parameters
        ----------
        component_name : str. Optional
            The class name for the CanvasComponent class to delete. If None,
            all CanvasComponents will be deleted subject to verification.

        Returns
        -------
        self

        """

        if component_name is None:

            delete_all = input("Are you sure you want to delete all \
            CanvasComponent objects from this class? (y/n)")

            if delete_all in ['Y', 'y', 'Yes', 'yes', 'YES']:
                print('Deleting CanvasComponent objects.')
                self.__components[component_name] = dict()
            else:
                print('Leaving CanvasComponents in place.')
        else:
            try:
                del self.__components[component_name]
            except KeyError:
                print("CanvasComponent object %s does not exist." \
                        % component_name)
        return self

    def __iter__(self):
        return iter(self.__components)

    # ----------------------------------------------------------------------- #
    #                          RESET METHOD                                   #
    # ----------------------------------------------------------------------- #       
    def reset(self, component_name=None):
        """Resets CanvasComponent(s) to its/their default values."""

        if component_name is None:

            reset_all = input("Are you sure you want to reset all \
            CanvasComponent objects to their default values? (y/n)")

            if reset_all in ['Y', 'y', 'Yes', 'yes', 'YES']:
                print('Resetting all CanvasComponent objects.')
                for component in self.__components.values():
                    component.reset()                
            else:
                print('Leaving CanvasComponents unchanged.')        

        else:
            try:
                self.__components[component_name].reset()
            except KeyError:
                print("CanvasComponent object %s does not exist." \
                        % component_name)
        return self

   



# --------------------------------------------------------------------------- #
#                            CanvasComponent                                  #
# --------------------------------------------------------------------------- #    
class CanvasComponent(ABC):
    """Abstract base class for Canvas component classes."""

    def __init__(self):
        self.__parameters = {}

    @abstractmethod
    def reset(self):
        """Resets configuration to default values."""
        pass

    def print_parameters(self):
        """Prints current parameters and values."""
        classname = self.__class__.__name__
        print('\nParameters for %s' % classname)
        for k, v in self.__parameters.items():
            message = '   ' + k + ' = ' + str(v)
            print(message)


# --------------------------------------------------------------------------- #
#                              CanvasTitle                                    #
# --------------------------------------------------------------------------- #    
class CanvasTitle(CanvasComponent):
    """Configuration options for plot titles."""

    DEFAULTS = {
        'text' : '',
        'font_family' : None,
        'font_size' : None,
        'font_color' : None,
        'xref' : 'container',
        'yref' : 'container',
        'x' : 0.5,
        'y' : 'auto',
        'xanchor' : 'auto',
        'yanchor' : 'auto',
        'pad' : {'t':0, 'b': 0, 'l':0}
    }

    def __init__(self):
        self.__parameters = {
            'title_text' : '',
            'title_font_family' : None,   
            'title_font_size' : None,
            'title_font_color' : None,
            'title_xref' : 'container',
            'title_yref' : 'container',
            'title_x' : 0.5,
            'title_y' : 'auto',
            'title_xanchor' : 'auto',
            'title_yanchor' : 'auto',
            'title_pad' : {'t':0, 'b': 0, 'l':0}
        }

    def reset(self):
        self.__parameters = {
            'title_text' : self.DEFAULTS['text'],
            'title_font_family' : self.DEFAULTS['font_family'],
            'title_font_size' : self.DEFAULTS['font_size'],
            'title_font_color' : self.DEFAULTS['font_color'],
            'title_xref' : self.DEFAULTS['xref'],
            'title_yref' : self.DEFAULTS['yref'],
            'title_x' : self.DEFAULTS['x'],
            'title_y' : self.DEFAULTS['y'],
            'title_xanchor' : self.DEFAULTS['xanchor'],
            'title_yanchor' : self.DEFAULTS['yanchor'],
            'title_pad' : self.DEFAULTS['pad']
        }

    # ----------------------------------------------------------------------- #
    #                        TITLE TEXT PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def title_text(self):
        """Returns the title_text attribute."""
        return self.__parameters['title_text']

    @title_text.setter
    def title_text(self, value):
        """Sets the title_text attribute."""
        self.__parameters['title_text'] = value

    # ----------------------------------------------------------------------- #
    #                     TITLE FONT FAMILY PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def title_font_family(self):
        """Returns the title_font_family attribute."""
        return self.__parameters['title_font_family']

    @title_font_family.setter
    def title_font_family(self, value):
        """Sets the title_font_family attribute."""
        self.__parameters['title_font_family'] = value

    # ----------------------------------------------------------------------- #
    #                        TITLE FONT SIZE PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def title_font_size(self):
        """Returns the title_font_size attribute."""
        return self.__parameters['title_font_size']

    @title_font_size.setter
    def title_font_size(self, value):
        """Sets the title_font_size attribute."""
        if value >= 1 or value is None:            
            self.__parameters['title_font_size'] = value
        else:
            raise ValueError("Font size must be greater or equal to 1.")

    # ----------------------------------------------------------------------- #
    #                        TITLE FONT COLOR PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def title_font_color(self):
        """Returns the title_font_color attribute."""
        return self.__parameters['title_font_color']

    @title_font_color.setter
    def title_font_color(self, value):
        """Sets the title_font_color attribute."""
        self.__parameters['title_font_color'] = value        

    # ----------------------------------------------------------------------- #
    #                        TITLE XREF PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def title_xref(self):
        """Returns the title_xref attribute.

        xref may have one of two values, 'container', which spans the entire
        width of the plot and 'paper', which refers to the plotting area 
        only.
        """
        return self.__parameters['title_xref']

    @title_xref.setter
    def title_xref(self, value):
        """Sets the title_xref attribute.

        Parameters
        ----------
        xref : str. Default 'container'
            xref may have one of two values, 'container', which spans the entire
            width of the plot and 'paper', which refers to the plotting area 
            only.
        """
        valid_values = ['container', 'paper']
        if value in valid_values:
            self.__parameters['title_xref'] = value                
        else:
            raise ValueError("xref must be equal to 'container', or 'paper'. ")

    # ----------------------------------------------------------------------- #
    #                        TITLE YREF PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def title_yref(self):
        """Returns the title_yref attribute.

        yref may have one of two values, 'container', which spans the entire
        width of the plot and 'paper', which refers to the plotting area 
        only.
        """
        return self.__parameters['title_yref']

    @title_yref.setter
    def title_yref(self, value):
        """Sets the title_yref attribute.

        Parameters
        ----------
        value : str. Default = 'container'
            yref may have one of two values, 'container', which spans the entire
            height of the plot and 'paper', which refers to the plotting area 
            only.
        """
        valid_values = ['container', 'paper']
        if value in valid_values:
            self.__parameters['title_yref'] = value                
        else:
            raise ValueError("yref must be equal to 'container', or 'paper'. ")        


    # ----------------------------------------------------------------------- #
    #                        TITLE X PROPERTIES                               #
    # ----------------------------------------------------------------------- #
    @property
    def title_x(self):
        """Returns the title_x attribute.

        Specifies the x position with respect to 'xref' in normalized 
        coordinates from '0' (left) to '1' (right)
        """
        return self.__parameters['title_x']

    @title_x.setter
    def title_x(self, value):
        """Sets the title_x attribute.

        Parameters
        ----------
        value : float, Default 0.5
            Specifies the x position with respect to 'xref' in normalized 
            coordinates from '0' (left) to '1' (right)
        """        
        if value >= 0 and value <= 1:
            self.__parameters['title_x'] = value                
        else:
            raise ValueError("x must be between 0 and 1 inclusive.")               

    # ----------------------------------------------------------------------- #
    #                        TITLE Y PROPERTIES                               #
    # ----------------------------------------------------------------------- #
    @property
    def title_y(self):
        """Returns the title_y attribute.

        Specifies the y position with respect to 'xref' in normalized 
        coordinates from '0' (left) to '1' (right). "auto" places 
        the baseline of the title onto the vertical center of the 
        top margin.

        """

        return self.__parameters['title_y']

    @title_y.setter
    def title_y(self, value):
        """Sets the title_y attribute.

        Parameters
        ----------
        value : float, Default = 'auto'
            Specifies the y position with respect to 'xref' in normalized 
            coordinates from '0' (left) to '1' (right). "auto" places 
            the baseline of the title onto the vertical center of the 
            top margin.

        """ 
        if isinstance(value, str):
            if value == 'auto':
                self.__parameters['title_y'] = value
            else:
                raise ValueError("title_y must be 'auto' or int between 0 and 1 inclusive.")
        elif isinstance(value, type(float, int)):
            if value >= 0 and value <= 1:
                self.__parameters['title_y'] = value                
            else:
                raise ValueError("title_y must be 'auto' or int between 0 and 1 inclusive.")
        else:
            raise ValueError("title_y must be 'auto' or int between 0 and 1 inclusive.")
   

    # ----------------------------------------------------------------------- #
    #                        TITLE XANCHOR PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def title_xanchor(self):
        """Returns the title_xanchor attribute.

        Sets the horizontal alignment of the title with respect to the x
        position. "left" means that the title starts at x, "right" means 
        that the title ends at x and "center" means that the title's center 
        is at x. "auto" divides `xref` by three and calculates the `xanchor` 
        value automatically based on the value of `x`.

        """

        return self.__parameters['title_xanchor']

    @title_xanchor.setter
    def title_xanchor(self, value):
        """Sets the title_xanchor attribute.

        Parameters
        ----------
        value : str, Default 'auto'. One of 'auto', 'left', 'center', 'right'
             "left" means that the title starts at x, "right" means that the 
             title ends at x and "center" means that the title's center is at 
             x. "auto" divides `xref` by three and calculates the `xanchor` 
             value automatically based on the value of `x`.

        """  

        valid_values = ['auto', 'left', 'center', 'right']        
        if value in valid_values:
            self.__parameters['title_xanchor'] = value                
        else:
            raise ValueError("xanchor must be 'auto', 'left', 'center',\
                             or 'right'.")             

    # ----------------------------------------------------------------------- #
    #                        TITLE YANCHOR PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def title_yanchor(self):
        """Returns the title_yanchor attribute.

        Sets the horizontal alignment of the title with respect to the x
        position. "left" means that the title starts at x, "right" means 
        that the title ends at x and "center" means that the title's center 
        is at x. "auto" divides `xref` by three and calculates the `yanchor` 
        value automatically based on the value of `x`.

        """

        return self.__parameters['title_yanchor']

    @title_yanchor.setter
    def title_yanchor(self, value):
        """Sets the title_yanchor attribute.

        Parameters
        ----------
        value : str, Default 'auto'. One of 'auto', 'top', 'middle', 'bottom'
              "top" means that the title's cap line is at y, "bottom" 
              means that the title's baseline is at y and "middle" means 
              that the title's midline is at y. "auto" divides `yref` by 
              three and calculates the `yanchor` value automatically based 
              on the value of `y`.

        """   

        valid_values = ['auto', 'top', 'middle', 'bottom']        
        if value in valid_values:
            self.__parameters['title_yanchor'] = value                
        else:
            raise ValueError("yanchor must be 'auto', 'top', 'middle',\
                             or 'bottom'.")                                          

    # ----------------------------------------------------------------------- #
    #                        TITLE PADDING PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def title_pad(self):
        """Returns the title_pad attribute.

        Sets the padding of the title via three key/value pairs. The keys are
        't' for top, 'b' for bottom, and 'l' for left. Each padding value 
        applies only when the corresponding `xanchor`/`yanchor` value is 
        set. For instance, for left padding to take effect, `xanchor` must 
        be set to "left". The same rule applies if `xanchor`/`yanchor` is 
        determined automatically. Padding is ignored if the respective 
        anchor value is "middle"/"center".

        """

        return self.__parameters['title_pad']

    @title_pad.setter
    def title_pad(self, value):
        """Sets the title_pad attribute.

        Parameters
        ----------
        value : dict, Default {'t': 0, 'b' : 0, 'l' : 0}
            Sets the padding of the title via three key/value pairs. The keys are
            't' for top, 'b' for bottom, and 'l' for left. The values are 
            the amount of padding in pixels. Each padding value 
            applies only when the corresponding `xanchor`/`yanchor` value is 
            set. For instance, for left padding to take effect, `xanchor` must 
            be set to "left". The same rule applies if `xanchor`/`yanchor` is 
            determined automatically. Padding is ignored if the respective 
            anchor value is "middle"/"center".

        """   

        valid_keys = ['t', 'b', 'l']                
        if isinstance(value, dict):
            if all(item in valid_keys for item in value.keys()):
                if all(isinstance(v,int) for v in value.values()):
                    self.__parameters['title_pad'] = value
                else:
                    raise TypeError("Pad values must be integers")
            else:
                raise KeyError("Pad keys must be 't', 'b', or 'l'.")
        else:
            raise TypeError("pad must be a dictionary.")                                

# --------------------------------------------------------------------------- #
#                              CanvasLegend                                   #
# --------------------------------------------------------------------------- #    
class CanvasLegend(CanvasComponent):
    """Configuration options for plot legends."""

    DEFAULTS = {
        'show' : True,
        'bgcolor' : None,   
        'bordercolor' : '#444',
        'borderwidth' : 0,
        'font_family' : None,
        'font_size' : None,
        'font_color' : None,
        'orientation' : 'v',
        'itemsizing' : 'trace',
        'itemclick' : 'toggle',
        'x' : 1.02,
        'y' : 1,
        'xanchor' : 'left',
        'yanchor' : 'auto',
        'valign' : 'middle'
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['legend_show'] = True
        self.__parameters['legend_bgcolor'] = None   
        self.__parameters['legend_bordercolor'] = '#444'
        self.__parameters['legend_borderwidth'] = 0
        self.__parameters['legend_font_family'] = None
        self.__parameters['legend_font_size'] = None
        self.__parameters['legend_font_color'] = None
        self.__parameters['legend_orientation'] = 'v'
        self.__parameters['legend_itemsizing'] = 'trace'
        self.__parameters['legend_itemclick'] = 'toggle'
        self.__parameters['legend_x'] = 1.02
        self.__parameters['legend_y'] = 1
        self.__parameters['legend_xanchor'] = 'left'
        self.__parameters['legend_yanchor'] = 'auto'
        self.__parameters['legend_valign'] = 'middle'

    def reset(self):
        self.__parameters = {}
        self.__parameters['legend_show'] = self.DEFAULTS['show']
        self.__parameters['legend_bgcolor'] = self.DEFAULTS['bgcolor']
        self.__parameters['legend_bordercolor'] = self.DEFAULTS['bordercolor']
        self.__parameters['legend_borderwidth'] = self.DEFAULTS['borderwidth']
        self.__parameters['legend_font_family'] = self.DEFAULTS['font_family']
        self.__parameters['legend_font_size'] = self.DEFAULTS['font_size']
        self.__parameters['legend_font_color'] = self.DEFAULTS['font_color']
        self.__parameters['legend_orientation'] = self.DEFAULTS['orientation']
        self.__parameters['legend_itemsizing'] = self.DEFAULTS['itemsizing']
        self.__parameters['legend_itemclick'] = self.DEFAULTS['itemclick']
        self.__parameters['legend_x'] = self.DEFAULTS['x']
        self.__parameters['legend_y'] = self.DEFAULTS['y']
        self.__parameters['legend_xanchor'] = self.DEFAULTS['xanchor']
        self.__parameters['legend_yanchor'] = self.DEFAULTS['yanchor']
        self.__parameters['legend_valign'] = self.DEFAULTS['valign']

    # ----------------------------------------------------------------------- #
    #                       LEGEND SHOW PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def legend_show(self):
        """Returns the legend_show attribute."""
        return self.__parameters['legend_show']

    @legend_show.setter
    def legend_show(self, value):
        """Sets the legend_show attribute."""
        if isinstance(value, bool):
            self.__parameters['legend_show'] = value        
        else:
            raise TypeError("legend_show must be True or False")

    # ----------------------------------------------------------------------- #
    #                     LEGEND BGCOLOR PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def legend_bgcolor(self):
        """Returns the legend_bgcolor attribute."""
        return self.__parameters['legend_bgcolor']

    @legend_bgcolor.setter
    def legend_bgcolor(self, value):
        """Sets the legend_bgcolor attribute."""
        if isinstance(value, str) or value is None:
            self.__parameters['legend_bgcolor'] = value           
        else:
            raise TypeError("legend_bgcolor must be string")

    # ----------------------------------------------------------------------- #
    #                     LEGEND BORDER COLOR PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_bordercolor(self):
        """Returns the legend_bordercolor attribute."""
        return self.__parameters['legend_bordercolor']

    @legend_bordercolor.setter
    def legend_bordercolor(self, value):
        """Sets the legend_bordercolor attribute."""
        self.__parameters['legend_bordercolor'] = value              

    # ----------------------------------------------------------------------- #
    #                     LEGEND BORDER WIDTH PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_borderwidth(self):
        """Returns the legend_borderwidth attribute."""
        return self.__parameters['legend_borderwidth']

    @legend_borderwidth.setter
    def legend_borderwidth(self, value):
        """Sets the legend_borderwidth attribute."""
        if isinstance(value, int) and value >= 0:
            self.__parameters['legend_borderwidth'] = value            
        elif not isinstance(value, int):
            raise TypeError("value must be an integer >= 0")
        else:
            raise ValueError("value must be an integer >= 0")

    # ----------------------------------------------------------------------- #
    #                     LEGEND FONT FAMILY PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def legend_font_family(self):
        """Returns the legend_font_family attribute."""
        return self.__parameters['legend_font_family']

    @legend_font_family.setter
    def legend_font_family(self, value):
        """Sets the legend_font_family attribute."""
        self.__parameters['legend_font_family'] = value

    # ----------------------------------------------------------------------- #
    #                        LEGEND FONT SIZE PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_font_size(self):
        """Returns the legend_font_size attribute."""
        return self.__parameters['legend_font_size']

    @legend_font_size.setter
    def legend_font_size(self, value):
        """Sets the legend_font_size attribute."""
        if isinstance(value, int) and value >= 1:
            self.__parameters['legend_font_size'] = value
        elif not isinstance(value, int):
            raise TypeError("value must be an integer >= 1")
        else:
            raise ValueError("value must be an integer >= 1")

    # ----------------------------------------------------------------------- #
    #                        LEGEND FONT COLOR PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def legend_font_color(self):
        """Returns the legend_font_color attribute."""
        return self.__parameters['legend_font_color']

    @legend_font_color.setter
    def legend_font_color(self, value):
        """Sets the legend_font_color attribute."""
        self.__parameters['legend_font_color'] = value           

    # ----------------------------------------------------------------------- #
    #                        LEGEND ORIENTATION PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def legend_orientation(self):
        """Returns the legend_orientation attribute."""
        return self.__parameters['legend_orientation']

    @legend_orientation.setter
    def legend_orientation(self, value):
        """Sets the legend_orientation attribute."""
        valid_values = ['v', 'h']
        if value in valid_values:
            self.__parameters['legend_orientation'] = value             
        else:
            raise ValueError("legend_orientation must be 'v', or 'h'.")

    # ----------------------------------------------------------------------- #
    #                        LEGEND ITEMSIZING PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def legend_itemsizing(self):
        """Returns the legend_itemsizing attribute."""
        return self.__parameters['legend_itemsizing']

    @legend_itemsizing.setter
    def legend_itemsizing(self, value):
        """Sets the legend_itemsizing attribute."""
        valid_values = ['trace', 'constant']
        if value in valid_values:
            self.__parameters['legend_itemsizing'] = value               
        else:
            raise ValueError("legend_itemsizing must be 'trace' or 'constant'")

    # ----------------------------------------------------------------------- #
    #                        LEGEND ITEMCLICK PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_itemclick(self):
        """Returns the legend_itemclick attribute."""
        return self.__parameters['legend_itemclick']

    @legend_itemclick.setter
    def legend_itemclick(self, value):
        """Sets the legend_itemclick attribute."""
        valid_values = ['toggle', 'toggleothers', False]
        if value in valid_values:
            self.__parameters['legend_itemclick'] = value                       
        else:
            raise ValueError("legend_itemclick must be 'toggle', \
                'toggleothers' or False")

    # ----------------------------------------------------------------------- #
    #                        LEGEND X PROPERTIES                              #
    # ----------------------------------------------------------------------- #
    @property
    def legend_x(self):
        """Returns the legend_x attribute.

        Specifies the x position with respect to 'xref' in normalized 
        coordinates from '0' (left) to '1' (right)
        """
        return self.__parameters['legend_x']

    @legend_x.setter
    def legend_x(self, value):
        """Sets the legend_x attribute.

        Parameters
        ----------
        value : float, Default 0.5
            Sets the x position (in normalized coordinates) of the legend. 
            Defaults to "1.02" for vertical legends and defaults to 
            "0" for horizontal legends.
        """                
        if isinstance(value,(int, float)) and value >= -2 and value <= 3:
            self.__parameters['legend_x'] = value                
        elif not isinstance(value, (int, float)):
            raise TypeError("legend_x must be a number between -2 and 3 inclusive.")               
        else:
            raise ValueError("legend_x must be a number between -2 and 3 inclusive.")               

    # ----------------------------------------------------------------------- #
    #                        LEGEND Y PROPERTIES                              #
    # ----------------------------------------------------------------------- #
    @property
    def legend_y(self):
        """Returns the legend_y attribute.

        Specifies the y position with respect to 'yref' in normalized 
        coordinates from '0' (bottom) to '1' (top). 

        """

        return self.__parameters['legend_y']

    @legend_y.setter
    def legend_y(self, value):
        """Sets the legend_y attribute.

        Parameters
        ----------
        value : float, Default = 1
            Sets the y position (in normalized coordinates) of the legend. 
            Defaults to "1" for vertical legends, defaults to "-0.1" 
            for horizontal legends on graphs w/o range sliders and 
            defaults to "1.1" for horizontal legends on graph with 
            one or multiple range sliders.

        """ 

        if isinstance(value,(int, float)) and value >= -2 and value <= 3:
            self.__parameters['legend_y'] = value                
        elif not isinstance(value, (int, float)):
            raise TypeError("legend_y must be a number between -2 and 3 inclusive.")               
        else:
            raise ValueError("legend_y must be a number between -2 and 3 inclusive.")                      

    # ----------------------------------------------------------------------- #
    #                        LEGEND XANCHOR PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def legend_xanchor(self):
        """Returns the legend_xanchor attribute.

        Sets the legend's horizontal position anchor. This anchor binds the 
        `x` position to the "left", "center" or "right" of the legend. 
        Value "auto" anchors legends to the right for `x` values greater 
        than or equal to 2/3, anchors legends to the left for `x` values 
        less than or equal to 1/3 and anchors legends with respect to their 
        center otherwise.

        """

        return self.__parameters['legend_xanchor']

    @legend_xanchor.setter
    def legend_xanchor(self, value):
        """Sets the legend_xanchor attribute.

        Parameters
        ----------
        value : str, Default 'left'. One of 'auto', 'left', 'center', 'right'
            Sets the legend's horizontal position anchor. This anchor binds the 
            `x` position to the "left", "center" or "right" of the legend. 
            Value "auto" anchors legends to the right for `x` values greater 
            than or equal to 2/3, anchors legends to the left for `x` values 
            less than or equal to 1/3 and anchors legends with respect to 
            their center otherwise.
        """  

        valid_values = ['auto', 'left', 'center', 'right']        
        if value in valid_values:
            self.__parameters['legend_xanchor'] = value                
        else:
            raise ValueError("xanchor must be 'auto', 'left', 'center',\
                             or 'right'.")             

    # ----------------------------------------------------------------------- #
    #                        LEGEND YANCHOR PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def legend_yanchor(self):
        """Returns the legend_yanchor attribute.

        Sets the horizontal alignment of the legend with respect to the x
        position. "left" means that the legend starts at x, "right" means 
        that the legend ends at x and "center" means that the legend's center 
        is at x. "auto" divides `xref` by three and calculates the `yanchor` 
        value automatically based on the value of `x`.

        """

        return self.__parameters['legend_yanchor']

    @legend_yanchor.setter
    def legend_yanchor(self, value):
        """Sets the legend_yanchor attribute.

        Parameters
        ----------
        value : str, Default 'auto'. One of 'auto', 'top', 'middle', 'bottom'
            Sets the legend's vertical position anchor This anchor binds the 
            `y` position to the "top", "middle" or "bottom" of the legend. 
            Value "auto" anchors legends at their bottom for `y` values 
            less than or equal to 1/3, anchors legends to at their top 
            for `y` values greater than or equal to 2/3 and anchors legends 
            with respect to their middle otherwise.

        """   

        valid_values = ['auto', 'top', 'middle', 'bottom']        
        if value in valid_values:
            self.__parameters['legend_yanchor'] = value                
        else:
            raise ValueError("yanchor must be 'auto', 'top', 'middle',\
                             or 'bottom'.")               

    # ----------------------------------------------------------------------- #
    #                        LEGEND VALIGN PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def legend_valign(self):
        """Returns the legend_valign attribute.

        Sets the vertical alignment of the symbols with respect to their
        associated text.

        """

        return self.__parameters['legend_valign']

    @legend_valign.setter
    def legend_valign(self, value):
        """Sets the legend_valign attribute.

        Parameters
        ----------
        value : str, Default 'auto'. One of 'auto', 'top', 'middle', 'bottom'
            Sets the vertical alignment of the symbols with respect to their 
            associated text.

        """   

        valid_values = ['top', 'middle', 'bottom']        
        if value in valid_values:
            self.__parameters['legend_valign'] = value                
        else:
            raise ValueError("valign must be 'top', 'middle',\
                             or 'bottom'.")                                     

# --------------------------------------------------------------------------- #
#                              CanvasMargins                                  #
# --------------------------------------------------------------------------- #    
class CanvasMargins(CanvasComponent):
    """Configuration options for plot margins."""

    DEFAULTS = {
        'left' : 80,
        'top' : 100,   
        'bottom' : 80,
        'pad' : 0
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['margins_left'] = 80
        self.__parameters['margins_top'] = 100   
        self.__parameters['margins_bottom'] = 80
        self.__parameters['margins_pad'] = 0

    def reset(self):
        self.__parameters = {}
        self.__parameters['margins_left'] = self.DEFAULTS['left']
        self.__parameters['margins_top'] = self.DEFAULTS['top']   
        self.__parameters['margins_bottom'] = self.DEFAULTS['bottom']
        self.__parameters['margins_pad'] = self.DEFAULTS['pad']
        

    # ----------------------------------------------------------------------- #
    #                       MARGINS_LEFT PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def margins_left(self):
        """Returns the margins_left attribute."""
        return self.__parameters['margins_left']

    @margins_left.setter
    def margins_left(self, value):
        """Sets the margins_left attribute."""
        if isinstance(value,int) and value >= 0:
            self.__parameters['margins_left'] = value        
        else:
            raise ValueError("value must be an integer >= 0")

    # ----------------------------------------------------------------------- #
    #                       MARGINS_TOP PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def margins_top(self):
        """Returns the margins_top attribute."""
        return self.__parameters['margins_top']

    @margins_top.setter
    def margins_top(self, value):
        """Sets the margins_top attribute."""
        if isinstance(value,int) and value >= 0:
            self.__parameters['margins_top'] = value        
        else:
            raise ValueError("value must be an integer >= 0")      

    # ----------------------------------------------------------------------- #
    #                       MARGINS_BOTTOM PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def margins_bottom(self):
        """Returns the margins_bottom attribute."""
        return self.__parameters['margins_bottom']

    @margins_bottom.setter
    def margins_bottom(self, value):
        """Sets the margins_bottom attribute."""
        if isinstance(value,int) and value >= 0:
            self.__parameters['margins_bottom'] = value        
        else:
            raise ValueError("value must be an integer >= 0")      

    # ----------------------------------------------------------------------- #
    #                       MARGINS_PAD PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def margins_pad(self):
        """Returns the margins_pad attribute."""
        return self.__parameters['margins_pad']

    @margins_pad.setter
    def margins_pad(self, value):
        """Sets the margins_pad attribute."""
        if isinstance(value,int) and value >= 0:
            self.__parameters['margins_pad'] = value        
        else:
            raise ValueError("value must be an integer >= 0")             

# --------------------------------------------------------------------------- #
#                              CanvasSize                                     #
# --------------------------------------------------------------------------- #    
class CanvasSize(CanvasComponent):
    """Configuration options for plot size."""

    DEFAULTS = {
        'autosize' : True,
        'width' : 700,   
        'height' : 450
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['size_autosize'] = True
        self.__parameters['size_width'] = 700   
        self.__parameters['size_height'] = 450

    def reset(self):
        self.__parameters = {}
        self.__parameters['size_autosize'] = self.DEFAULTS['autosize']
        self.__parameters['size_width'] = self.DEFAULTS['width']   
        self.__parameters['size_height'] = self.DEFAULTS['height']
        

    # ----------------------------------------------------------------------- #
    #                         AUTOSIZE PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def size_autosize(self):
        """Returns the size_autosize attribute.
        
        Determines whether or not a layout width or height that has been left 
        undefined by the user is initialized on each relayout.

        """

        return self.__parameters['size_autosize']

    @size_autosize.setter
    def size_autosize(self, value):
        """Sets the size_autosize attribute.
        
        Parameters
        ----------
        value : bool
            Determines whether or not a layout width or height that has 
            been left undefined by the user is initialized on each relayout.

        """

        if isinstance(value, bool):
            self.__parameters['size_autosize'] = value        
        else:
            raise ValueError("size_autosize must be True or False.")

    # ----------------------------------------------------------------------- #
    #                         WIDTH PROPERTIES                                #
    # ----------------------------------------------------------------------- #
    @property
    def size_width(self):
        """Returns the size_width attribute."""

        return self.__parameters['size_width']

    @size_width.setter
    def size_width(self, value):
        """Sets the size_width attribute."""
        if isinstance(value, (int,float)) and value >= 10:
            self.__parameters['size_width'] = value        
        else:
            raise ValueError("Width must be a number greater or equal to 10.")

    # ----------------------------------------------------------------------- #
    #                         HEIGHT PROPERTIES                               #
    # ----------------------------------------------------------------------- #
    @property
    def size_height(self):
        """Returns the size_height attribute."""

        return self.__parameters['size_height']

    @size_height.setter
    def size_height(self, value):
        """Sets the size_height attribute."""
        if isinstance(value, (int, float)) and value >= 10:
            self.__parameters['size_height'] = value        
        else:
            raise ValueError("height must be a number greater or equal to 10.")        

# --------------------------------------------------------------------------- #
#                              CanvasFont                                     #
# --------------------------------------------------------------------------- #    
class CanvasFont(CanvasComponent):
    """Configuration options for plot font."""

    DEFAULTS = {
        'family' : None,
        'size' : 12,   
        'color' : '#444',
        'separators' : '.,'
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['font_family'] = None
        self.__parameters['font_size'] = 12   
        self.__parameters['font_color'] = '#444'
        self.__parameters['font_separators'] = '.,'

    def reset(self):
        self.__parameters = {}
        self.__parameters['font_family'] = self.DEFAULTS['family']
        self.__parameters['font_size'] = self.DEFAULTS['size']   
        self.__parameters['font_color'] = self.DEFAULTS['color']
        self.__parameters['font_separators'] = self.DEFAULTS['separators']
        

    # ----------------------------------------------------------------------- #
    #                         FONT FAMILY PROPERTIES                          #
    # ----------------------------------------------------------------------- #
    @property
    def font_family(self):
        """Returns the font_family attribute."""
        return self.__parameters['font_family']

    @font_family.setter
    def font_family(self, value):
        """Sets the font_family attribute."""
        self.__parameters['font_family'] = value              

    # ----------------------------------------------------------------------- #
    #                         FONT SIZE PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def font_size(self):
        """Returns the font_size attribute."""
        return self.__parameters['font_size']

    @font_size.setter
    def font_size(self, value):
        """Sets the font_size attribute."""
        if isinstance(value, int) and value >= 1:
            self.__parameters['font_size'] = value              
        else:
            raise ValueError("value must be a number >= 1")

    # ----------------------------------------------------------------------- #
    #                         FONT COLOR PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def font_color(self):
        """Returns the font_color attribute."""
        return self.__parameters['font_color']

    @font_color.setter
    def font_color(self, value):
        """Sets the font_color attribute."""
        self.__parameters['font_color'] = value                  

    # ----------------------------------------------------------------------- #
    #                          SEPARATORS PROPERTIES                          #
    # ----------------------------------------------------------------------- #
    @property
    def font_separators(self):
        """Returns the font_separators attribute.
        
        The variable name is misleading since the separator parameter is
        for numbers.  BTAIM, this parameter sets the decimal and thousand 
        separators. For example, ".," puts a '.' before decimals and a ',' 
        between thousands.
        """
        return self.__parameters['font_separators']

    @font_separators.setter
    def font_separators(self, value):
        """Sets the font_separators attribute.
        
        Parameters
        ----------
        value : str, Default '.,'
            The variable name is misleading since the separator parameter is
            for numbers.  BTAIM, this parameter sets the decimal and thousand 
            separators. For example, ".," puts a '.' before decimals and a ',' 
            between thousands.

        """

        self.__parameters['font_separators'] = value            

# --------------------------------------------------------------------------- #
#                          CanvasColorsBackground                             #
# --------------------------------------------------------------------------- #       
class CanvasColorsBackground(CanvasComponent):
    """Configures background colors for paper and plot."""
    DEFAULTS = {
        'paper_bgcolor' : '#fff',
        'plot_bgcolor' : '#fff'
    }      

    def __init__(self):
        self.__parameters = {}
        self.__parameters['paper_bgcolor'] = '#fff'
        self.__parameters['plot_bgcolor'] = '#fff'   

    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['paper_bgcolor'] = self.DEFAULTS["paper_bgcolor"]
        self.__parameters['plot_bgcolor'] = self.DEFAULTS["plot_bgcolor"]      

    # ----------------------------------------------------------------------- #
    #                         PAPER BGCOLOR PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def paper_bgcolor(self):
        """Returns the paper_bgcolor attribute.
        
        Sets the color of paper where the graph is drawn.
        
        """
        
        return self.__parameters['paper_bgcolor']

    @paper_bgcolor.setter
    def paper_bgcolor(self, value):
        """Sets the paper_bgcolor attribute.
        
        Parameters
        ----------
        value : str, Default = '#fff'
            Sets the color of paper where the graph is drawn.
        """
        self.__parameters['paper_bgcolor'] = value                      

    # ----------------------------------------------------------------------- #
    #                         PLOT BGCOLOR PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def plot_bgcolor(self):
        """Returns the plot_bgcolor attribute.
        
        Sets the color of plot where the graph is drawn.
        
        """
        
        return self.__parameters['plot_bgcolor']

    @plot_bgcolor.setter
    def plot_bgcolor(self, value):
        """Sets the plot_bgcolor attribute.
        
        Parameters
        ----------
        value : str, Default = '#fff'
            Sets the color of plot where the graph is drawn.
        """
        self.__parameters['plot_bgcolor'] = value              

# --------------------------------------------------------------------------- #
#                              CanvasColorScale                               #
# --------------------------------------------------------------------------- #    
class CanvasColorScale(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'colorscale_sequential' : [[0, 'rgb(220,220,220)'], 
                                   [0.2, 'rgb(245,195,157)'], 
                                   [0.4, 'rgb(245,160,105)'], 
                                   [1, 'rgb(178,10,28)'], ],
        'colorscale_sequentialminus' : [[0, 'rgb(5,10,172)'], 
                                        [0.35, 'rgb(40,60,190)'], 
                                        [0.5, 'rgb(70,100,245)'], 
                                        [0.6, 'rgb(90,120,245)'], 
                                        [0.7, 'rgb(106,137,247)'], 
                                        [1, 'rgb(220,220,220)'], ],
        'colorscale_diverging' : [[0, 'rgb(5,10,172)'], 
                                  [0.35, 'rgb(106,137,247)'], 
                                  [0.5, 'rgb(190,190,190)'], 
                                  [0.6, 'rgb(220,170,132)'], 
                                  [0.7, 'rgb(230,145,90)'], 
                                  [1, 'rgb(178,10,28)'], ],
        'colorway' : ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['colorscale_sequential'] = [[0, 'rgb(220,220,220)'], [0.2, 'rgb(245,195,157)'], 
                                  [0.4, 'rgb(245,160,105)'], [1, 'rgb(178,10,28)'], ]
        self.__parameters['colorscale_sequentialminus'] = [[0, 'rgb(5,10,172)'], 
                                        [0.35, 'rgb(40,60,190)'], 
                                        [0.5, 'rgb(70,100,245)'], 
                                        [0.6, 'rgb(90,120,245)'], 
                                        [0.7, 'rgb(106,137,247)'], 
                                        [1, 'rgb(220,220,220)'], ]
        self.__parameters['colorscale_diverging'] = [[0, 'rgb(5,10,172)'], [0.35, 'rgb(106,137,247)'], 
                                [0.5, 'rgb(190,190,190)'], [0.6, 'rgb(220,170,132)'], 
                                [0.7, 'rgb(230,145,90)'], [1, 'rgb(178,10,28)'], ]
        self.__parameters['colorway'] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['colorscale_sequential'] = self.DEFAULTS["colorscale_sequential"]
        self.__parameters['colorscale_sequentialminus'] = self.DEFAULTS["colorscale_sequentialminus"]
        self.__parameters['colorscale_diverging'] = self.DEFAULTS["colorscale_diverging"]
        self.__parameters['colorway'] = self.DEFAULTS["colorway"]

# ----------------------------------------------------------------------- #
    #                   COLORSCALE SEQUENTIAL PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def colorscale_sequential(self):
        """Returns the colorscale_sequential attribute.
        
        Sets the default sequential colorscale for positive values.
        
        """
        
        return self.__parameters['colorscale_sequential']

    @colorscale_sequential.setter
    def colorscale_sequential(self, value):
        """Sets the colorscale_sequential attribute.
        
        Parameters
        ----------
        value : list Default = [[0, rgb(220,220,220)], [0.2, rgb(245,195,157)], 
                                  [0.4, rgb(245,160,105)], [1, rgb(178,10,28)], ]
            Sets the default sequential colorscale for positive values. 
        """
        self.__parameters['colorscale_sequential'] = value                   

    # ----------------------------------------------------------------------- #
    #                   COLORSCALE SEQUENTIALMINUS PROPERTIES                 #
    # ----------------------------------------------------------------------- #
    @property
    def colorscale_sequentialminus(self):
        """Returns the colorscale_sequentialminus attribute.
        
        Sets the default sequential colorscale for negative values.
        
        """
        
        return self.__parameters['colorscale_sequentialminus']

    @colorscale_sequentialminus.setter
    def colorscale_sequentialminus(self, value):
        """Sets the colorscale_sequentialminus attribute.
        
        Parameters
        ----------
        value : list Default = [[0, rgb(5,10,172)], 
                                        [0.35, rgb(40,60,190)], 
                                        [0.5, rgb(70,100,245)], 
                                        [0.6, rgb(90,120,245)], 
                                        [0.7, rgb(106,137,247)], 
                                        [1, rgb(220,220,220)], ]
            Sets the default sequential colorscale for negative values. 
        """
        self.__parameters['colorscale_sequentialminus'] = value                           

    # ----------------------------------------------------------------------- #
    #                   COLORSCALE DIVERGING PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def colorscale_diverging(self):
        """Returns the colorscale_diverging attribute.
        
        Sets the default diverging colorscale.
        
        """
        
        return self.__parameters['colorscale_diverging']

    @colorscale_diverging.setter
    def colorscale_diverging(self, value):
        """Sets the colorscale_diverging attribute.
        
        Parameters
        ----------
        value : list, Default = [[0, rgb(5,10,172)], [0.35, rgb(106,137,247)], 
                                [0.5, rgb(190,190,190)], [0.6, rgb(220,170,132)], 
                                [0.7, rgb(230,145,90)], [1, rgb(178,10,28)], ]
            Sets the default diverging colorscale.
        """
        self.__parameters['colorscale_diverging'] = value                           
        
    # ----------------------------------------------------------------------- #
    #                         COLORWAY PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def colorway(self):
        """Returns the colorway attribute.
        
        Sets the default trace colors.
        
        """
        
        return self.__parameters['colorway']

    @colorway.setter
    def colorway(self, value):
        """Sets the colorway attribute.
        
        Parameters
        ----------
        value : list. Default = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                                 '#bcbd22', '#17becf']
            Sets the default trace colors.
        """
        self.__parameters['colorway'] = value     

# --------------------------------------------------------------------------- #
#                          CanvasColorAxisDomain                              #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisDomain(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_cauto' : True,
        'coloraxis_cmin' : None,
        'coloraxis_cmax' : None,
        'coloraxis_cmid' : None
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_cauto'] = True
        self.__parameters['coloraxis_cmin'] = None
        self.__parameters['coloraxis_cmax'] = None
        self.__parameters['coloraxis_cmid'] = None


    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_cauto'] = self.DEFAULTS["coloraxis_cauto"]
        self.__parameters['coloraxis_cmin'] = self.DEFAULTS["coloraxis_cmin"]
        self.__parameters['coloraxis_cmax'] = self.DEFAULTS["coloraxis_cmax"]
        self.__parameters['coloraxis_cmid'] = self.DEFAULTS["coloraxis_cmid"]

    # ----------------------------------------------------------------------- #
    #                     COLORAXIS CAUTO PROPERTIES                          #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_cauto(self):
        """Returns the coloraxis_cauto attribute.
        
        Determines whether or not the color domain is computed with respect 
        to the input data (here corresponding trace color array(s)) or the 
        bounds set in `cmin` and `cmax` Defaults to `False` when `cmin` 
        and `cmax` are set by the user.
        
        """
        
        return self.__parameters['coloraxis_cauto']

    @coloraxis_cauto.setter
    def coloraxis_cauto(self, value):
        """Sets the coloraxis_cauto attribute.
        
        Parameters
        ----------
        value : bool Default = True
            Determines whether or not the color domain is computed with respect 
            to the input data (here corresponding trace color array(s)) or the 
            bounds set in `cmin` and `cmax` Defaults to `False` when `cmin` 
            and `cmax` are set by the user.

        """
        if isinstance(value, bool):
            self.__parameters['coloraxis_cauto'] = value         
        else:
            raise TypeError("value must be boolean True or False.")

    # ----------------------------------------------------------------------- #
    #                     COLORAXIS CMIN PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_cmin(self):
        """Returns the coloraxis_cmin attribute.
        
        Sets the lower bound of the color domain. Value should have the same 
        units as corresponding trace color array(s) and if set, `cmax` must 
        be set as well.
        
        """
        
        return self.__parameters['coloraxis_cmin']

    @coloraxis_cmin.setter
    def coloraxis_cmin(self, value):
        """Sets the coloraxis_cmin attribute.
        
        Parameters
        ----------
        value : int
            Sets the lower bound of the color domain. Value should have the same 
            units as corresponding trace color array(s) and if set, `cmax` must 
            be set as well.

        """
        if isinstance(value, (float, int)):
            self.__parameters['coloraxis_cmin'] = value                 
        else:
            raise TypeError("value must be a number")

    # ----------------------------------------------------------------------- #
    #                     COLORAXIS CMAX PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_cmax(self):
        """Returns the coloraxis_cmax attribute.
        
        Sets the upper bound of the color domain. Value should have the same 
        units as corresponding trace color array(s) and if set, `cmin` 
        must be set as well.
        
        """
        
        return self.__parameters['coloraxis_cmax']

    @coloraxis_cmax.setter
    def coloraxis_cmax(self, value):
        """Sets the coloraxis_cmax attribute.
        
        Parameters
        ----------
        value : int
            Sets the upper bound of the color domain. Value should have the same 
            units as corresponding trace color array(s) and if set, `cmin` 
            must be set as well.

        """
        
        if isinstance(value, (float, int)):
            self.__parameters['coloraxis_cmax'] = value                 
        else:
            raise TypeError("value must be a number")         
        
    # ----------------------------------------------------------------------- #
    #                     COLORAXIS CMID PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_cmid(self):
        """Returns the coloraxis_cmid attribute.
        
        Sets the mid-point of the color domain by scaling `cmin` and/or 
        `cmax` to be equidistant to this point. Value should have the 
        same units as corresponding trace color array(s). Has no effect 
        when `cauto` is `False`.
        
        """
        
        return self.__parameters['coloraxis_cmid']

    @coloraxis_cmid.setter
    def coloraxis_cmid(self, value):
        """Sets the coloraxis_cmid attribute.
        
        Parameters
        ----------
        value : int
            Sets the mid-point of the color domain by scaling `cmin` and/or 
            `cmax` to be equidistant to this point. Value should have the 
            same units as corresponding trace color array(s). Has no effect 
            when `cauto` is `False`.

        """
        
        if isinstance(value, (float, int)):
            self.__parameters['coloraxis_cmid'] = value                 
        else:
            raise TypeError("value must be a number")       
        
# --------------------------------------------------------------------------- #
#                         CanvasColorAxisScales                               #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisScales(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorscale' : [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
        'coloraxis_autoscale' : True,
        'coloraxis_reversescale' : True,
        'coloraxis_showscale' : True
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorscale'] = [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]
        self.__parameters['coloraxis_autoscale'] = True
        self.__parameters['coloraxis_reversescale'] = True
        self.__parameters['coloraxis_showscale'] = True

    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorscale'] = self.DEFAULTS["coloraxis_colorscale"]
        self.__parameters['coloraxis_autoscale'] = self.DEFAULTS["coloraxis_autoscale"]
        self.__parameters['coloraxis_reversescale'] = self.DEFAULTS["coloraxis_reversescale"]
        self.__parameters['coloraxis_showscale'] = self.DEFAULTS["coloraxis_showscale"]

    # ----------------------------------------------------------------------- #
    #                   COLORAXIS COLORSCALE PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorscale(self):
        """Returns the coloraxis_colorscale attribute.
        
        Sets the colorscale. See `Plotly Colorscale 
        <https://plot.ly/python/reference/#layout-coloraxis-colorscale>`_
        
        """
        
        return self.__parameters['coloraxis_colorscale']

    @coloraxis_colorscale.setter
    def coloraxis_colorscale(self, value):
        """Sets the coloraxis_colorscale attribute.
        
        Parameters
        ----------
        value : list. Default = [[0, rgb(0,0,255)], [1, rgb(255,0,0)]]
            Sets the colorscale. See `Plotly Colorscale 
            <https://plot.ly/python/reference/#layout-coloraxis-colorscale>`_

        """
        if isinstance(value, list):
            self.__parameters['coloraxis_colorscale'] = value                
        else:
            raise TypeError("value must be a list.")
              

    # ----------------------------------------------------------------------- #
    #                   COLORAXIS AUTOSCALE PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_autoscale(self):
        """Returns the coloraxis_autoscale attribute.
        
        Determines whether the colorscale is a default palette 
        (`autocolorscale: True`) or the palette determined by `colorscale`. 
        In case `colorscale` is unspecified or `autocolorscale` is True, 
        the default palette will be chosen according to whether numbers 
        in the `color` array are all positive, all negative or mixed.
        
        """
        
        return self.__parameters['coloraxis_autoscale']

    @coloraxis_autoscale.setter
    def coloraxis_autoscale(self, value):
        """Sets the coloraxis_autoscale attribute.
        
        Parameters
        ----------
        value : bool. Default = True
            Determines whether the colorscale is a default palette 
            (`autocolorscale: True`) or the palette determined by `colorscale`. 
            In case `colorscale` is unspecified or `autocolorscale` is True, 
            the default palette will be chosen according to whether numbers 
            in the `color` array are all positive, all negative or mixed.

        """
        if isinstance(value, bool):
            self.__parameters['coloraxis_autoscale'] = value                  
        else:
            raise TypeError("value must be boolean True or False.")

    # ----------------------------------------------------------------------- #
    #                   COLORAXIS REVERSESCALE PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_reversescale(self):
        """Returns the coloraxis_reversescale attribute.
        
        Reverses the color mapping if True. If True, `cmin` will correspond 
        to the last color in the array and `cmax` will correspond to the first 
        color.
        
        """
        
        return self.__parameters['coloraxis_reversescale']

    @coloraxis_reversescale.setter
    def coloraxis_reversescale(self, value):
        """Sets the coloraxis_reversescale attribute.
        
        Parameters
        ----------
        value : bool. Default = True
            Reverses the color mapping if True. If True, `cmin` will correspond 
            to the last color in the array and `cmax` will correspond to the first 
            color.

        """

        if isinstance(value, bool):
            self.__parameters['coloraxis_reversescale'] = value                  
        else:
            raise TypeError("value must be boolean True or False.")             

    # ----------------------------------------------------------------------- #
    #                   COLORAXIS SHOWSCALE PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_showscale(self):
        """Returns the coloraxis_showscale attribute.
        
        Determines whether or not a colorbar is displayed for this trace.
        
        """
        
        return self.__parameters['coloraxis_showscale']

    @coloraxis_showscale.setter
    def coloraxis_showscale(self, value):
        """Sets the coloraxis_showscale attribute.
        
        Parameters
        ----------
        value : bool. Default = True
            Determines whether or not a colorbar is displayed for this trace.

        """
        
        if isinstance(value, bool):
            self.__parameters['coloraxis_showscale'] = value                  
        else:
            raise TypeError("value must be boolean True or False.")    


# --------------------------------------------------------------------------- #
#                         CanvasColorAxisBarStyle                             #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisBarStyle(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_thicknessmode' : 'pixels',
        'coloraxis_colorbar_thickness' : 30,
        'coloraxis_colorbar_lenmode' : 'fraction',
        'coloraxis_colorbar_len' : 1,
        'coloraxis_colorbar_bgcolor' : "rgba(0000)"
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_thicknessmode'] = 'pixels'
        self.__parameters['coloraxis_colorbar_thickness'] = 30
        self.__parameters['coloraxis_colorbar_lenmode'] = 'fraction'
        self.__parameters['coloraxis_colorbar_len'] = 1
        self.__parameters['coloraxis_colorbar_bgcolor'] = "rgba(0000)"


    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_thicknessmode'] = self.DEFAULTS["coloraxis_colorbar_thicknessmode"]
        self.__parameters['coloraxis_colorbar_thickness'] = self.DEFAULTS["coloraxis_colorbar_thickness"]
        self.__parameters['coloraxis_colorbar_lenmode'] = self.DEFAULTS["coloraxis_colorbar_lenmode"]
        self.__parameters['coloraxis_colorbar_len'] = self.DEFAULTS["coloraxis_colorbar_len"]
        self.__parameters['coloraxis_colorbar_bgcolor'] = self.DEFAULTS["coloraxis_colorbar_bgcolor"]

    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR THICKNESSMODE PROPERTIES                #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_thicknessmode(self):
        """Returns the coloraxis_colorbar_thicknessmode attribute.
        
        Determines whether this color bar's thickness (i.e. the measure in 
        the constant color direction) is set in units of plot "fraction" 
        or in "pixels". Use `thickness` to set the value.
        
        """
        
        return self.__parameters['coloraxis_colorbar_thicknessmode']

    @coloraxis_colorbar_thicknessmode.setter
    def coloraxis_colorbar_thicknessmode(self, value):
        """Sets the coloraxis_colorbar_thicknessmode attribute.
        
        Parameters
        ----------
        value : str. One of 'fraction' or 'pixels'. Default = 'pixels'.
            Determines whether this color bar's thickness (i.e. the measure in 
            the constant color direction) is set in units of plot "fraction" 
            or in "pixels". Use `thickness` to set the value.

        """
        valid_values = ['fraction', 'pixels']
        if value in valid_values:
            self.__parameters['coloraxis_colorbar_thicknessmode'] = value              
        else:
            raise ValueError("colorbar_thicknessmode must be either 'fraction'\
                 or 'pixels'.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR THICKNESS PROPERTIES                  #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_thickness(self):
        """Returns the coloraxis_colorbar_thickness attribute.
        
        Sets the thickness of the color bar This measure excludes the 
        size of the padding, ticks and labels.
        
        """
        
        return self.__parameters['coloraxis_colorbar_thickness']

    @coloraxis_colorbar_thickness.setter
    def coloraxis_colorbar_thickness(self, value):
        """Sets the coloraxis_colorbar_thickness attribute.
        
        Parameters
        ----------
        value : int Default = 30
            Sets the thickness of the color bar This measure excludes the 
            size of the padding, ticks and labels.

        """
        
        if isinstance(value, (int, float)) and value >= 0:
            self.__parameters['coloraxis_colorbar_thickness'] = value              
        else:
            raise ValueError("colorbar_thickness must be an integer >= 0.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR LENMODE PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_lenmode(self):
        """Returns the coloraxis_colorbar_lenmode attribute.
        
        Determines whether this color bar's length (i.e. the measure in 
        the color variation direction) is set in units of plot "fraction" 
        or in "pixels. Use `len` to set the value.
        
        """
        
        return self.__parameters['coloraxis_colorbar_lenmode']

    @coloraxis_colorbar_lenmode.setter
    def coloraxis_colorbar_lenmode(self, value):
        """Sets the coloraxis_colorbar_lenmode attribute.
        
        Parameters
        ----------
        value : str. One of 'fraction' or 'pixels'. Default = 'pixels'.
            Determines whether this color bar's length (i.e. the measure in 
            the color variation direction) is set in units of plot "fraction" 
            or in "pixels. Use `len` to set the value.

        """
        valid_values = ['fraction', 'pixels']
        if value in valid_values:
            self.__parameters['coloraxis_colorbar_lenmode'] = value         
        else:
            raise ValueError("colorbar_lenmode must be either 'fraction', \
                or 'pixels'.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR LEN PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_len(self):
        """Returns the coloraxis_colorbar_len attribute.
        
        Sets the length of the color bar This measure excludes the padding 
        of both ends. That is, the color bar length is this length minus 
        the padding on both ends.
        
        """
        
        return self.__parameters['coloraxis_colorbar_len']

    @coloraxis_colorbar_len.setter
    def coloraxis_colorbar_len(self, value):
        """Sets the coloraxis_colorbar_len attribute.
        
        Parameters
        ----------
        value : int. Default = 1
            Sets the length of the color bar This measure excludes the padding 
            of both ends. That is, the color bar length is this length minus 
            the padding on both ends.

        """
        
        if isinstance(value, (int, float)) and value >= 0:
            self.__parameters['coloraxis_colorbar_len'] = value                        
        else:
            raise ValueError("colorbar_len must be an integer >= 0.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR BGCOLOR PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_bgcolor(self):
        """Returns the coloraxis_colorbar_bgcolor attribute.
        
        Sets the color of padded area.
        
        """
        
        return self.__parameters['coloraxis_colorbar_bgcolor']

    @coloraxis_colorbar_bgcolor.setter
    def coloraxis_colorbar_bgcolor(self, value):
        """Sets the coloraxis_colorbar_bgcolor attribute.
        
        Parameters
        ----------
        value : color. Default = "rgba(0,0,0,0)"
            Sets the color of padded area.

        """
        
        self.__parameters['coloraxis_colorbar_bgcolor'] = value               

# --------------------------------------------------------------------------- #
#                        CanvasColorAxisBarPosition                           #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisBarPosition(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_x' : 1.02,
        'coloraxis_colorbar_y' : 0.5,
        'coloraxis_colorbar_xanchor' : 'left',
        'coloraxis_colorbar_yanchor' : 'middle',
        'coloraxis_colorbar_xpad' : 10,
        'coloraxis_colorbar_ypad' : 10
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_x'] = 1.02
        self.__parameters['coloraxis_colorbar_y'] = 0.5
        self.__parameters['coloraxis_colorbar_xanchor'] = 'left'
        self.__parameters['coloraxis_colorbar_yanchor'] = 'middle'
        self.__parameters['coloraxis_colorbar_xpad'] = 10
        self.__parameters['coloraxis_colorbar_ypad'] = 10        

    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_x'] = self.DEFAULTS["coloraxis_colorbar_x"]
        self.__parameters['coloraxis_colorbar_y'] = self.DEFAULTS["coloraxis_colorbar_y"]
        self.__parameters['coloraxis_colorbar_xanchor'] = self.DEFAULTS["coloraxis_colorbar_xanchor"]
        self.__parameters['coloraxis_colorbar_yanchor'] = self.DEFAULTS["coloraxis_colorbar_yanchor"]
        self.__parameters['coloraxis_colorbar_xpad'] = self.DEFAULTS["coloraxis_colorbar_xpad"]
        self.__parameters['coloraxis_colorbar_ypad'] = self.DEFAULTS["coloraxis_colorbar_ypad"]

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR X PROPERTIES                          #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_x(self):
        """Returns the coloraxis_colorbar_x attribute.
        
        Sets the x position of the color bar (in plot fraction).
        
        """
        
        return self.__parameters['coloraxis_colorbar_x']

    @coloraxis_colorbar_x.setter
    def coloraxis_colorbar_x(self, value):
        """Sets the coloraxis_colorbar_x attribute.
        
        Parameters
        ----------
        value : int between -2 and 3. Default = 1.02
            Sets the x position of the color bar (in plot fraction).

        """
        
        if isinstance(value, (int, float)) and value >= -2 and value <= 3:
            self.__parameters['coloraxis_colorbar_x'] = value                 
        else:
            raise ValueError("colorbar_x must be an integer between \
                -2 and 3 inclusive.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR Y PROPERTIES                          #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_y(self):
        """Returns the coloraxis_colorbar_y attribute.
        
        Sets the x position of the color bar (in plot fraction).
        
        """
        
        return self.__parameters['coloraxis_colorbar_y']

    @coloraxis_colorbar_y.setter
    def coloraxis_colorbar_y(self, value):
        """Sets the coloraxis_colorbar_y attribute.
        
        Parameters
        ----------
        value : int between -2 and 3. Default = 0.5
            Sets the x position of the color bar (in plot fraction).

        """
        if isinstance(value, (int, float)) and value >= -2 and value <= 3:
            self.__parameters['coloraxis_colorbar_y'] = value                 
        else:
            raise ValueError("colorbar_y must be an integer between \
                -2 and 3 inclusive.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR XANCHOR PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_xanchor(self):
        """Returns the coloraxis_colorbar_xanchor attribute.
        
        Sets this color bar's horizontal position anchor. This anchor 
        binds the `x` position to the "left", "center" or "right" of the 
        color bar.
        
        """
        
        return self.__parameters['coloraxis_colorbar_xanchor']

    @coloraxis_colorbar_xanchor.setter
    def coloraxis_colorbar_xanchor(self, value):
        """Sets the coloraxis_colorbar_xanchor attribute.
        
        Parameters
        ----------
        value : str. One of 'left', 'center', 'right'. Default = 'left'
            Sets this color bar's horizontal position anchor. This anchor 
            binds the `x` position to the "left", "center" or "right" of the 
            color bar.

        """
        valid_values = ['left', 'center', 'right']
        if value in valid_values:            
            self.__parameters['coloraxis_colorbar_xanchor'] = value          
        else:
            raise ValueError("colorbar_xanchor must be either 'left', \
                'center', or 'right'.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR YANCHOR PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_yanchor(self):
        """Returns the coloraxis_colorbar_yanchor attribute.
        
        Sets this color bar's vertical position anchor This anchor binds 
        the `y` position to the "top", "middle" or "bottom" of the color bar.
        
        """
        
        return self.__parameters['coloraxis_colorbar_yanchor']

    @coloraxis_colorbar_yanchor.setter
    def coloraxis_colorbar_yanchor(self, value):
        """Sets the coloraxis_colorbar_yanchor attribute.
        
        Parameters
        ----------
        value : str. One of 'top', 'middle', 'bottom'. Default = 'middle'
            Sets this color bar's vertical position anchor This anchor 
            binds the `y` position to the "top", "middle" or "bottom" 
            of the color bar.

        """
        valid_values = ['middle', 'bottom', 'top']
        if value in valid_values:            
            self.__parameters['coloraxis_colorbar_yanchor'] = value               
        else:
            raise ValueError("colorbar_yanchor must be either 'middle', \
                            'bottom', or 'top'.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR XPAD PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_xpad(self):
        """Returns the coloraxis_colorbar_xpad attribute.
        
        Sets the amount of padding (in px) along the x direction.
        
        """
        
        return self.__parameters['coloraxis_colorbar_xpad']

    @coloraxis_colorbar_xpad.setter
    def coloraxis_colorbar_xpad(self, value):
        """Sets the coloraxis_colorbar_xpad attribute.
        
        Parameters
        ----------
        value : int. Default = 10
            Sets the amount of padding (in px) along the x direction.

        """
        
        if isinstance(value, (int, float)) and value >= 0:
            self.__parameters['coloraxis_colorbar_xpad'] = value                    
        else:
            raise ValueError("colorbar_xpad must be an integer >= 0.")    

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR YPAD PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_ypad(self):
        """Returns the coloraxis_colorbar_ypad attribute.
        
        Sets the amount of padding (in px) along the y direction.
        
        """
        
        return self.__parameters['coloraxis_colorbar_ypad']

    @coloraxis_colorbar_ypad.setter
    def coloraxis_colorbar_ypad(self, value):
        """Sets the coloraxis_colorbar_ypad attribute.
        
        Parameters
        ----------
        value : int. Default = 10
            Sets the amount of padding (in px) along the y direction.

        """
        if isinstance(value, (int, float)) and value >= 0:
            self.__parameters['coloraxis_colorbar_ypad'] = value                    
        else:
            raise ValueError("colorbar_ypad must be an integer >= 0.")        
        self.__parameters['coloraxis_colorbar_ypad'] = value                    


# --------------------------------------------------------------------------- #
#                       CanvasColorAxisBarBoundary                            #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisBarBoundary(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_outlinecolor' : '#444',        
        'coloraxis_colorbar_outlinewidth' : 1,        
        'coloraxis_colorbar_bordercolor' : '#444',        
        'coloraxis_colorbar_borderwidth' : 0
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_outlinecolor'] = '#444'        
        self.__parameters['coloraxis_colorbar_outlinewidth'] = 1        
        self.__parameters['coloraxis_colorbar_bordercolor'] = '#444'        
        self.__parameters['coloraxis_colorbar_borderwidth'] = 0        

    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_outlinecolor'] = self.DEFAULTS["coloraxis_colorbar_outlinecolor"]
        self.__parameters['coloraxis_colorbar_outlinewidth'] = self.DEFAULTS["coloraxis_colorbar_outlinewidth"]
        self.__parameters['coloraxis_colorbar_bordercolor'] = self.DEFAULTS["coloraxis_colorbar_bordercolor"]
        self.__parameters['coloraxis_colorbar_borderwidth'] = self.DEFAULTS["coloraxis_colorbar_borderwidth"]

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR OUTLINECOLOR PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_outlinecolor(self):
        """Returns the coloraxis_colorbar_outlinecolor attribute.
        
        Sets the axis line color.
        
        """
        
        return self.__parameters['coloraxis_colorbar_outlinecolor']

    @coloraxis_colorbar_outlinecolor.setter
    def coloraxis_colorbar_outlinecolor(self, value):
        """Sets the coloraxis_colorbar_outlinecolor attribute.
        
        Parameters
        ----------
        value : str. Default = '#444'
            Sets the axis line color.

        """
        if isinstance(value, str):
            self.__parameters['coloraxis_colorbar_outlinecolor'] = value       
        else:
            raise TypeError("value must be a string.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR OUTLINEWIDTH PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_outlinewidth(self):
        """Returns the coloraxis_colorbar_outlinewidth attribute.
        
        Sets the width (in px) of the axis line.
        
        """
        
        return self.__parameters['coloraxis_colorbar_outlinewidth']

    @coloraxis_colorbar_outlinewidth.setter
    def coloraxis_colorbar_outlinewidth(self, value):
        """Sets the coloraxis_colorbar_outlinewidth attribute.
        
        Parameters
        ----------
        value : int greater than or equal to 0. Default = 1
            Sets the width (in px) of the axis line.

        """
        
        if isinstance(value, (int, float)) and value >= 0:
            self.__parameters['coloraxis_colorbar_outlinewidth'] = value                    
        else:
            raise ValueError("colorbar_outlinewidth must be an integer >= 0.")            

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR BORDERCOLOR PROPERTIES                #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_bordercolor(self):
        """Returns the coloraxis_colorbar_bordercolor attribute.
        
        Sets the axis line color.
        
        """
        
        return self.__parameters['coloraxis_colorbar_bordercolor']

    @coloraxis_colorbar_bordercolor.setter
    def coloraxis_colorbar_bordercolor(self, value):
        """Sets the coloraxis_colorbar_bordercolor attribute.
        
        Parameters
        ----------
        value : str. Default = '#444'
            Sets the axis line color.

        """
        
        if isinstance(value, str):
            self.__parameters['coloraxis_colorbar_bordercolor'] = value       
        else:
            raise TypeError("value must be a string.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR BORDERWIDTH PROPERTIES                #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_borderwidth(self):
        """Returns the coloraxis_colorbar_borderwidth attribute.
        
        Sets the axis line color.
        
        """
        
        return self.__parameters['coloraxis_colorbar_borderwidth']

    @coloraxis_colorbar_borderwidth.setter
    def coloraxis_colorbar_borderwidth(self, value):
        """Sets the coloraxis_colorbar_borderwidth attribute.
        
        Parameters
        ----------
        value : int greater than or equal to 0. Default = 0
            Sets the axis line color.

        """

        if isinstance(value, (int, float)) and value >= 0:
            self.__parameters['coloraxis_colorbar_borderwidth'] = value                    
        else:
            raise ValueError("colorbar_borderwidth must be an integer >= 0.")

# --------------------------------------------------------------------------- #
#                        CanvasColorAxisBarTicks                              #
# --------------------------------------------------------------------------- #    

class CanvasColorAxisBarTicks(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_tickmode' : "array",
        'coloraxis_colorbar_nticks' : 0,
        'coloraxis_colorbar_tick0' : None,
        'coloraxis_colorbar_dtick' : None,
        'coloraxis_colorbar_tickvals' : None,
        'coloraxis_colorbar_ticktext' : "",
        'coloraxis_colorbar_ticks' : None
        }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_tickmode'] = "array"
        self.__parameters['coloraxis_colorbar_nticks'] = 0
        self.__parameters['coloraxis_colorbar_tick0'] = None
        self.__parameters['coloraxis_colorbar_dtick'] = None
        self.__parameters['coloraxis_colorbar_tickvals'] = None
        self.__parameters['coloraxis_colorbar_ticktext'] = ""
        self.__parameters['coloraxis_colorbar_ticks'] = ""


    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_tickmode'] = self.DEFAULTS["coloraxis_colorbar_tickmode"]
        self.__parameters['coloraxis_colorbar_nticks'] = self.DEFAULTS["coloraxis_colorbar_nticks"]
        self.__parameters['coloraxis_colorbar_tick0'] = self.DEFAULTS["coloraxis_colorbar_tick0"]
        self.__parameters['coloraxis_colorbar_dtick'] = self.DEFAULTS["coloraxis_colorbar_dtick"]
        self.__parameters['coloraxis_colorbar_tickvals'] = self.DEFAULTS["coloraxis_colorbar_tickvals"]
        self.__parameters['coloraxis_colorbar_ticktext'] = self.DEFAULTS["coloraxis_colorbar_ticktext"]
        self.__parameters['coloraxis_colorbar_ticks'] = self.DEFAULTS["coloraxis_colorbar_ticks"]

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKMODE PROPERTIES                   #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickmode(self):
        """Returns the coloraxis_colorbar_tickmode attribute.
        
        Sets the tick mode for this axis. If "auto", the number of ticks is 
        set via `nticks`. If "linear", the placement of the ticks is 
        determined by a starting position `tick0` and a tick step `dtick` 
        ("linear" is the default value if `tick0` and `dtick` are provided). 
        If "array", the placement of the ticks is set via `tickvals` and the 
        tick text is `ticktext`. ("array" is the default value if 
        `tickvals` is provided).
        
        """
        
        return self.__parameters['coloraxis_colorbar_tickmode']

    @coloraxis_colorbar_tickmode.setter
    def coloraxis_colorbar_tickmode(self, value):
        """Sets the coloraxis_colorbar_tickmode attribute.
        
        Parameters
        ----------
        value : str. One of 'auto', 'linear', or 'array'. Default = "array"
            Sets the tick mode for this axis. If "auto", the number of ticks is 
            set via `nticks`. If "linear", the placement of the ticks is 
            determined by a starting position `tick0` and a tick step `dtick` 
            ("linear" is the default value if `tick0` and `dtick` are provided). 
            If "array", the placement of the ticks is set via `tickvals` and the 
            tick text is `ticktext`. ("array" is the default value if 
            `tickvals` is provided).

        """
        valid_values = ['auto', 'linear', 'array']
        if value in valid_values:
            self.__parameters['coloraxis_colorbar_tickmode'] = value        
        else:
            raise ValueError("'colorbar_tickmode' must be either \
                'auto', 'linear', oir 'array'.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR NTICKS PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_nticks(self):
        """Returns the coloraxis_colorbar_nticks attribute.
        
        Specifies the maximum number of ticks for the particular axis. 
        The actual number of ticks will be chosen automatically to be less 
        than or equal to `nticks`. Has an effect only if `tickmode` is set 
        to "auto".
        
        """
        
        return self.__parameters['coloraxis_colorbar_nticks']

    @coloraxis_colorbar_nticks.setter
    def coloraxis_colorbar_nticks(self, value):
        """Sets the coloraxis_colorbar_nticks attribute.
        
        Parameters
        ----------
        value : int greater than or equal to 0. Default = 0
            Specifies the maximum number of ticks for the particular axis. 
            The actual number of ticks will be chosen automatically to be less 
            than or equal to `nticks`. Has an effect only if `tickmode` is set 
            to "auto".

        """
        
        if isinstance(value, (int, float)) and value >= 0:
            self.__parameters['coloraxis_colorbar_nticks'] = value                      
        else:
            raise ValueError("colorbar_nticks must be a number >= 0.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICK0 PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tick0(self):
        """Returns the coloraxis_colorbar_tick0 attribute.
        
        Sets the placement of the first tick on this axis. Use with `dtick`. 
        If the axis `type` is "log", then you must take the log of your 
        starting tick (e.g. to set the starting tick to 100, set the `tick0` 
        to 2) except when `dtick`="L<f>" (see `dtick` for more info). If 
        the axis `type` is "date", it should be a date string, like date 
        data. If the axis `type` is "category", it should be a number, 
        using the scale where each category is assigned a serial number 
        from zero in the order it appears.
        
        """
        
        return self.__parameters['coloraxis_colorbar_tick0']

    @coloraxis_colorbar_tick0.setter
    def coloraxis_colorbar_tick0(self, value):
        """Sets the coloraxis_colorbar_tick0 attribute.
        
        Parameters
        ----------
        value : int or str
            Sets the placement of the first tick on this axis. Use with `dtick`. 
            If the axis `type` is "log", then you must take the log of your 
            starting tick (e.g. to set the starting tick to 100, set the `tick0` 
            to 2) except when `dtick`="L<f>" (see `dtick` for more info). If 
            the axis `type` is "date", it should be a date string, like date 
            data. If the axis `type` is "category", it should be a number, 
            using the scale where each category is assigned a serial number 
            from zero in the order it appears.

        """
        
        self.__parameters['coloraxis_colorbar_tick0'] = value           

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR DTICK PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_dtick(self):
        """Returns the coloraxis_colorbar_dtick attribute.
        
        Sets the step in-between ticks on this axis. Use with `tick0`. 
        Must be a positive number, or special strings available to "log" 
        and "date" axes. If the axis `type` is "log", then ticks are set 
        every 10^(n"dtick) where n is the tick number. For example, to set 
        a tick mark at 1, 10, 100, 1000, ... set dtick to 1. To set tick 
        marks at 1, 100, 10000, ... set dtick to 2. To set tick marks at 
        1, 5, 25, 125, 625, 3125, ... set dtick to log_10(5), or 0.69897000433.
        "log" has several special values; "L<f>", where `f` is a positive 
        number, gives ticks linearly spaced in value (but not position). 
        For example `tick0`'] = 0.1, `dtick`'] = "L0.5" will put ticks at 0.1, 
        0.6, 1.1, 1.6 etc. To show powers of 10 plus small digits between, 
        use "D1" (all digits) or "D2" (only 2 and 5). `tick0` is ignored for 
        "D1" and "D2". If the axis `type` is "date", then you must convert 
        the time to milliseconds. For example, to set the interval between 
        ticks to one day, set `dtick` to 86400000.0. "date" also has special 
        values "M<n>" gives ticks spaced by a number of months. `n` must be 
        a positive integer. To set ticks on the 15th of every third month, 
        set `tick0` to "2000-01-15" and `dtick` to "M3". To set ticks every 
        4 years, set `dtick` to "M48"
        
        """
        
        return self.__parameters['coloraxis_colorbar_dtick']

    @coloraxis_colorbar_dtick.setter
    def coloraxis_colorbar_dtick(self, value):
        """Sets the coloraxis_colorbar_dtick attribute.
        
        Parameters
        ----------
        value : int or str
            Sets the step in-between ticks on this axis. Use with `tick0`. 
            Must be a positive number, or special strings available to "log" 
            and "date" axes. If the axis `type` is "log", then ticks are set 
            every 10^(n"dtick) where n is the tick number. For example, to set 
            a tick mark at 1, 10, 100, 1000, ... set dtick to 1. To set tick 
            marks at 1, 100, 10000, ... set dtick to 2. To set tick marks at 
            1, 5, 25, 125, 625, 3125, ... set dtick to log_10(5), or 0.69897000433.
            "log" has several special values; "L<f>", where `f` is a positive 
            number, gives ticks linearly spaced in value (but not position). 
            For example `tick0`'] = 0.1, `dtick`'] = "L0.5" will put ticks at 0.1, 
            0.6, 1.1, 1.6 etc. To show powers of 10 plus small digits between, 
            use "D1" (all digits) or "D2" (only 2 and 5). `tick0` is ignored for 
            "D1" and "D2". If the axis `type` is "date", then you must convert 
            the time to milliseconds. For example, to set the interval between 
            ticks to one day, set `dtick` to 86400000.0. "date" also has special 
            values "M<n>" gives ticks spaced by a number of months. `n` must be 
            a positive integer. To set ticks on the 15th of every third month, 
            set `tick0` to "2000-01-15" and `dtick` to "M3". To set ticks every 
            4 years, set `dtick` to "M48"

        """
        
        self.__parameters['coloraxis_colorbar_dtick'] = value                             

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKVALS PROPERTIES                   #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickvals(self):
        """Returns the coloraxis_colorbar_tickvals attribute.
        
        Sets the values at which ticks on this axis appear. Only has an 
        effect if `tickmode` is set to "array". Used with `ticktext`
        
        """
        
        return self.__parameters['coloraxis_colorbar_tickvals']

    @coloraxis_colorbar_tickvals.setter
    def coloraxis_colorbar_tickvals(self, value):
        """Sets the coloraxis_colorbar_tickvals attribute.
        
        Parameters
        ----------
        value : array-like numbers, strings, or datetimes.
            Sets the values at which ticks on this axis appear. Only has an 
            effect if `tickmode` is set to "array". Used with `ticktext`

        """

        self.__parameters['coloraxis_colorbar_tickvals'] = value         

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKTEXT PROPERTIES                   #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_ticktext(self):
        """Returns the coloraxis_colorbar_ticktext attribute.
        
        Sets the text displayed at the ticks position via `tickvals`. 
        Only has an effect if `tickmode` is set to "array". 
        Used with `tickvals`.
        
        """

        return self.__parameters['coloraxis_colorbar_ticktext']

    @coloraxis_colorbar_ticktext.setter
    def coloraxis_colorbar_ticktext(self, value):
        """Sets the coloraxis_colorbar_ticktext attribute.
        
        Parameters
        ----------
        value : array-like numbers, strings, or datetimes.
            Sets the text displayed at the ticks position via `tickvals`. 
            Only has an effect if `tickmode` is set to "array". 
            Used with `tickvals`.

        """                

        self.__parameters['coloraxis_colorbar_ticktext'] = value         

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKS PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_ticks(self):
        """Returns the coloraxis_colorbar_ticks attribute.
        
        Determines whether ticks are drawn or not. If "", this axis' 
        ticks are not drawn. If "outside" ("inside"), this axis' are 
        drawn outside (inside) the axis lines.
        
        """

        return self.__parameters['coloraxis_colorbar_ticks']

    @coloraxis_colorbar_ticks.setter
    def coloraxis_colorbar_ticks(self, value):
        """Sets the coloraxis_colorbar_ticks attribute.
        
        Parameters
        ----------
        value : str. One of 'outside', 'inside', and "". Default = ""
            Determines whether ticks are drawn or not. If "", this axis' 
            ticks are not drawn. If "outside" ("inside"), this axis' are 
            drawn outside (inside) the axis lines.

        """                

        valid_values = ['outside', 'inside', ""]
        if value in valid_values:
            self.__parameters['coloraxis_colorbar_ticks'] = value                 
        else:
            raise ValueError("colorbar_ticks must be either 'outside', \
                'inside', or ''.")


# --------------------------------------------------------------------------- #
#                     CanvasColorAxisBarTickStyle                             #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisBarTickStyle(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_ticklen' : 5,
        'coloraxis_colorbar_tickwidth' : 1,
        'coloraxis_colorbar_tickcolor' : '#444',
        'coloraxis_colorbar_showticklabels' : True,
        'coloraxis_colorbar_tickangle' : 'auto',
        'coloraxis_colorbar_tickprefix' : '',
        'coloraxis_colorbar_showtickprefix' : 'all',
        'coloraxis_colorbar_ticksuffix' : '',
        'coloraxis_colorbar_showticksuffix' : 'all'

    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_ticklen'] = 5
        self.__parameters['coloraxis_colorbar_tickwidth'] = 1
        self.__parameters['coloraxis_colorbar_tickcolor'] = '#444'
        self.__parameters['coloraxis_colorbar_showticklabels'] = True
        self.__parameters['coloraxis_colorbar_tickangle'] = 'auto'
        self.__parameters['coloraxis_colorbar_tickprefix'] = ''
        self.__parameters['coloraxis_colorbar_showtickprefix'] = 'all'
        self.__parameters['coloraxis_colorbar_ticksuffix'] = ''
        self.__parameters['coloraxis_colorbar_showticksuffix'] = 'all'



    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_ticklen'] = self.DEFAULTS["coloraxis_colorbar_ticklen"]
        self.__parameters['coloraxis_colorbar_tickwidth'] = self.DEFAULTS["coloraxis_colorbar_tickwidth"]
        self.__parameters['coloraxis_colorbar_tickcolor'] = self.DEFAULTS["coloraxis_colorbar_tickcolor"]
        self.__parameters['coloraxis_colorbar_showticklabels'] = self.DEFAULTS["coloraxis_colorbar_showticklabels"]
        self.__parameters['coloraxis_colorbar_tickangle'] = self.DEFAULTS["coloraxis_colorbar_tickangle"]
        self.__parameters['coloraxis_colorbar_tickprefix'] = self.DEFAULTS["coloraxis_colorbar_tickprefix"]
        self.__parameters['coloraxis_colorbar_showtickprefix'] = self.DEFAULTS["coloraxis_colorbar_showtickprefix"]
        self.__parameters['coloraxis_colorbar_ticksuffix'] = self.DEFAULTS["coloraxis_colorbar_ticksuffix"]
        self.__parameters['coloraxis_colorbar_showticksuffix'] = self.DEFAULTS["coloraxis_colorbar_showticksuffix"]


    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKLEN PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_ticklen(self):
        """Returns the coloraxis_colorbar_ticklen attribute.
        
        Sets the tick length (in px).
        
        """

        return self.__parameters['coloraxis_colorbar_ticklen']

    @coloraxis_colorbar_ticklen.setter
    def coloraxis_colorbar_ticklen(self, value):
        """Sets the coloraxis_colorbar_ticklen attribute.
        
        Parameters
        ----------
        value : int >= 0. Default = 5
            Sets the tick length (in px).

        """                
        
        if isinstance(value, int) and value >= 0:
            self.__parameters['coloraxis_colorbar_ticklen'] = value                 
        else:
            raise ValueError("colorbar_ticklen must be an integer >= 0.")                

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKWIDTH PROPERTIES                  #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickwidth(self):
        """Returns the coloraxis_colorbar_tickwidth attribute.
        
        Sets the tick length (in px).
        
        """

        return self.__parameters['coloraxis_colorbar_tickwidth']

    @coloraxis_colorbar_tickwidth.setter
    def coloraxis_colorbar_tickwidth(self, value):
        """Sets the coloraxis_colorbar_tickwidth attribute.
        
        Parameters
        ----------
        value : int >= 0. Default = 1
            Sets the tick length (in px).

        """                
        
        if isinstance(value, int) and value >= 0:
            self.__parameters['coloraxis_colorbar_tickwidth'] = value                 
        else:
            raise ValueError("colorbar_tickwidth must be an integer >= 0.")                      

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKCOLOR PROPERTIES                  #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickcolor(self):
        """Returns the coloraxis_colorbar_tickcolor attribute.
        
        Sets the tick color.
        
        """

        return self.__parameters['coloraxis_colorbar_tickcolor']

    @coloraxis_colorbar_tickcolor.setter
    def coloraxis_colorbar_tickcolor(self, value):
        """Sets the coloraxis_colorbar_tickcolor attribute.
        
        Parameters
        ----------
        value : str
            Sets the tick color.

        """                
        
        self.__parameters['coloraxis_colorbar_tickcolor'] = value                 
        

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR SHOWTICKLABELS PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_showticklabels(self):
        """Returns the coloraxis_colorbar_showticklabels attribute.
        
        Determines whether or not the tick labels are drawn.
        
        """

        return self.__parameters['coloraxis_colorbar_showticklabels']

    @coloraxis_colorbar_showticklabels.setter
    def coloraxis_colorbar_showticklabels(self, value):
        """Sets the coloraxis_colorbar_showticklabels attribute.
        
        Parameters
        ----------
        value : bool. Default = True
            Determines whether or not the tick labels are drawn.

        """                
        if isinstance(value, bool):
            self.__parameters['coloraxis_colorbar_showticklabels'] = value                 
        else:
            raise TypeError("value must be a boolean, True or False.")

    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKANGLE PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickangle(self):
        """Returns the coloraxis_colorbar_tickangle attribute.
        
        Sets tick angle.
        
        """

        return self.__parameters['coloraxis_colorbar_tickangle']

    @coloraxis_colorbar_tickangle.setter
    def coloraxis_colorbar_tickangle(self, value):
        """Sets the coloraxis_colorbar_tickangle attribute.
        
        Parameters
        ----------
        value : str or int
            Sets tick angle.

        """                
        if isinstance(value, str) and value == 'auto':
            self.__parameters['coloraxis_colorbar_tickangle'] = value             
        elif isinstance(value, int):
            self.__parameters['coloraxis_colorbar_tickangle'] = value             
        else:
            raise TypeError("value must be 'auto' or a number.")

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR TICKPREFIX PROPERTIES                  #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickprefix(self):
        """Returns the coloraxis_colorbar_tickprefix attribute.
        
        Sets a tick label prefix 
        
        """

        return self.__parameters['coloraxis_colorbar_tickprefix']

    @coloraxis_colorbar_tickprefix.setter
    def coloraxis_colorbar_tickprefix(self, value):
        """Sets the coloraxis_colorbar_tickprefix attribute.
        
        Parameters
        ----------
        value : str
             Sets a tick label prefix

        """                
        
        self.__parameters['coloraxis_colorbar_tickprefix'] = value            

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR SHOWTICKPREFIX PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_showtickprefix(self):
        """Returns the coloraxis_colorbar_showtickprefix attribute.
        
        If "all", all tick labels are displayed with a prefix. If 
        "first", only the first tick is displayed with a prefix. 
        If "last", only the last tick is displayed with a suffix. 
        If "none", tick prefixes are hidden.
        
        """

        return self.__parameters['coloraxis_colorbar_showtickprefix']

    @coloraxis_colorbar_showtickprefix.setter
    def coloraxis_colorbar_showtickprefix(self, value):
        """Sets the coloraxis_colorbar_showtickprefix attribute.
        
        Parameters
        ----------
        value : str
             If "all", all tick labels are displayed with a prefix. If 
             "first", only the first tick is displayed with a prefix. 
             If "last", only the last tick is displayed with a suffix. 
             If "none", tick prefixes are hidden.

        """                
        valid_values = ['all', 'first', 'last', 'none']
        if value in valid_values:
            self.__parameters['coloraxis_colorbar_showtickprefix'] = value
        else:
            raise ValueError("showtickprefix must be 'all', 'first', 'last'\
                , or 'none'.")

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR TICKSUFFIX PROPERTIES                  #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_ticksuffix(self):
        """Returns the coloraxis_colorbar_ticksuffix attribute.
        
        Sets a tick label suffix 
        
        """

        return self.__parameters['coloraxis_colorbar_ticksuffix']

    @coloraxis_colorbar_ticksuffix.setter
    def coloraxis_colorbar_ticksuffix(self, value):
        """Sets the coloraxis_colorbar_ticksuffix attribute.
        
        Parameters
        ----------
        value : str
             Sets a tick label suffix

        """                
        
        self.__parameters['coloraxis_colorbar_ticksuffix'] = value                   

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR SHOWTICKSUFFIX PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_showticksuffix(self):
        """Returns the coloraxis_colorbar_showticksuffix attribute.
        
        If "all", all tick labels are displayed with a suffix. If 
        "first", only the first tick is displayed with a suffix. 
        If "last", only the last tick is displayed with a suffix. 
        If "none", tick suffixes are hidden.
        
        """

        return self.__parameters['coloraxis_colorbar_showticksuffix']

    @coloraxis_colorbar_showticksuffix.setter
    def coloraxis_colorbar_showticksuffix(self, value):
        """Sets the coloraxis_colorbar_showticksuffix attribute.
        
        Parameters
        ----------
        value : str
             If "all", all tick labels are displayed with a suffix. If 
             "first", only the first tick is displayed with a suffix. 
             If "last", only the last tick is displayed with a suffix. 
             If "none", tick suffixes are hidden.

        """                
        valid_values = ['all', 'first', 'last', 'none']
        if value in valid_values:
            self.__parameters['coloraxis_colorbar_showticksuffix'] = value
        else:
            raise ValueError("showticksuffix must be 'all', 'first', 'last'\
                , or 'none'.")


# --------------------------------------------------------------------------- #
#                      CanvasColorAxisBarTickFont                             #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisBarTickFont(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_tickfont_family' : None,
        'coloraxis_colorbar_tickfont_size' : 1,
        'coloraxis_colorbar_tickfont_color' : None
        }

    def __init__(self):
        self.__parameters['coloraxis_colorbar_tickfont_family'] = None
        self.__parameters['coloraxis_colorbar_tickfont_size'] = 1
        self.__parameters['coloraxis_colorbar_tickfont_color'] = None
        
    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_tickfont_family'] = self.DEFAULTS["coloraxis_colorbar_tickfont_family"]
        self.__parameters['coloraxis_colorbar_tickfont_size'] = self.DEFAULTS["coloraxis_colorbar_tickfont_size"]
        self.__parameters['coloraxis_colorbar_tickfont_color'] = self.DEFAULTS["coloraxis_colorbar_tickfont_color"]

    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKFONT_FAMILY PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickfont_family(self):
        """Returns the coloraxis_colorbar_tickfont_family attribute.
        
        Sets tick font family.
        
        """
        return self.__parameters['coloraxis_colorbar_tickfont_family']


    @coloraxis_colorbar_tickfont_family.setter
    def coloraxis_colorbar_tickfont_family(self, value):
        """Sets the coloraxis_colorbar_tickfont_family attribute.
        
        Parameters
        ----------
        value : str
            Sets tick font family.

        """                

        if isinstance(value, str):        
            self.__parameters['coloraxis_colorbar_tickfont_family'] = value                    
        else:
            raise TypeError("value must be a string.")

    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKFONT_SIZE PROPERTIES                #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickfont_size(self):
        """Returns the coloraxis_colorbar_tickfont_size attribute.
        
        Sets tick font size.
        
        """

        return self.__parameters['coloraxis_colorbar_tickfont_size']

    @coloraxis_colorbar_tickfont_size.setter
    def coloraxis_colorbar_tickfont_size(self, value):
        """Sets the coloraxis_colorbar_tickfont_size attribute.
        
        Parameters
        ----------
        value : int
            Sets tick font size.

        """                
        if isinstance(value, int) and value >= 1:
            self.__parameters['coloraxis_colorbar_tickfont_size'] = value     
        else:
            raise ValueError("value must be an integer >= 1.")
        
    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKFONT_COLOR PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickfont_color(self):
        """Returns the coloraxis_colorbar_tickfont_color attribute.
        
        Sets tick font color.
        
        """

        return self.__parameters['coloraxis_colorbar_tickfont_color']

    @coloraxis_colorbar_tickfont_color.setter
    def coloraxis_colorbar_tickfont_color(self, value):
        """Sets the coloraxis_colorbar_tickfont_color attribute.
        
        Parameters
        ----------
        value : str
            Sets tick font color.

        """                
        
        self.__parameters['coloraxis_colorbar_tickfont_color'] = value     



# --------------------------------------------------------------------------- #
#                          CanvasColorAxisBarNumbers                          #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisBarNumbers(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_separatethousands' : True,
        'coloraxis_colorbar_exponentformat' : 'B',
        'coloraxis_colorbar_showexponent' : 'all'
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_separatethousands'] = True
        self.__parameters['coloraxis_colorbar_exponentformat'] = 'B'
        self.__parameters['coloraxis_colorbar_showexponent'] = 'all'

    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_separatethousands'] = self.DEFAULTS["coloraxis_colorbar_separatethousands"]
        self.__parameters['coloraxis_colorbar_exponentformat'] = self.DEFAULTS["coloraxis_colorbar_exponentformat"]
        self.__parameters['coloraxis_colorbar_showexponent'] = self.DEFAULTS["coloraxis_colorbar_showexponent"]

    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR SEPARATETHOUSANDS PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_separatethousands(self):
        """Returns the coloraxis_colorbar_separatethousands attribute.
        
        If "True", even 4-digit integers are separated.

        """

        return self.__parameters['coloraxis_colorbar_separatethousands']

    @coloraxis_colorbar_separatethousands.setter
    def coloraxis_colorbar_separatethousands(self, value):
        """Sets the coloraxis_colorbar_separatethousands attribute.
        
        Parameters
        ----------
        value : bool
            If "True", even 4-digit integers are separated.

        """                

        if isinstance(value, bool):
            self.__parameters['coloraxis_colorbar_separatethousands'] = value
        else:
            raise TypeError("value must be a boolean, True or False.")

    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR EXPONENTIALFORMAT PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_exponentialformat(self):
        """Returns the coloraxis_colorbar_exponentialformat attribute.
        
        Determines a formatting rule for the tick exponents.

        """

        return self.__parameters['coloraxis_colorbar_exponentialformat']

    @coloraxis_colorbar_exponentialformat.setter
    def coloraxis_colorbar_exponentialformat(self, value):
        """Sets the coloraxis_colorbar_exponentialformat attribute.
        
        Parameters
        ----------
        value : bool. One of "none" | "e" | "E" | "power" | "SI" | "B"
            Determines a formatting rule for the tick exponents.

        """                

        valid_values = ["none", "e", "E", "power", "SI", "B"]
        if value in valid_values:            
            self.__parameters['coloraxis_colorbar_exponentialformat'] = value        
        else:
            raise ValueError("exponentialformat must be 'none', 'e'\
                , 'E', 'power', 'SI', or 'B'.")

    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR SHOWEXPONENT PROPERTIES                   #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_showexponent(self):
        """Returns the coloraxis_colorbar_showexponent attribute.
        
        If "all", all exponents are shown besides their significands. 
        If "first", only the exponent of the first tick is shown. If "last", 
        only the exponent of the last tick is shown. If "none", 
        no exponents appear.

        """

        return self.__parameters['coloraxis_colorbar_showexponent']

    @coloraxis_colorbar_showexponent.setter
    def coloraxis_colorbar_showexponent(self, value):
        """Sets the coloraxis_colorbar_showexponent attribute.
        
        Parameters
        ----------
        value : bool. One of "all" | "first" | "last" | "none" 
            If "all", all exponents are shown besides their significands. 
            If "first", only the exponent of the first tick is shown. If "last", 
            only the exponent of the last tick is shown. If "none", 
            no exponents appear.

        """                

        valid_values = ['all', 'first', 'last', 'none']
        if value in valid_values:            
            self.__parameters['coloraxis_colorbar_showexponent'] = value        
        else:
            raise ValueError("showexponent must be 'all', 'first', 'last'\
                , 'none'.")                

# --------------------------------------------------------------------------- #
#                         CanvasColorAxisBarTitle                             #
# --------------------------------------------------------------------------- #    
class CanvasColorAxisBarTitle(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'coloraxis_colorbar_title_text' : "",
        'coloraxis_colorbar_title_font_family' : None,
        'coloraxis_colorbar_title_font_size' : 1,
        'coloraxis_colorbar_title_font_color' : None,
        'coloraxis_colorbar_title_side' : 'top'
    }

    def __init__(self):
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_title_text'] = ""
        self.__parameters['coloraxis_colorbar_title_font_family'] = None
        self.__parameters['coloraxis_colorbar_title_font_size'] = 1
        self.__parameters['coloraxis_colorbar_title_font_color'] = None
        self.__parameters['coloraxis_colorbar_title_side'] = 'top'


    def reset(self):
        """Sets parameters back to their defaults."""
        self.__parameters = {}
        self.__parameters['coloraxis_colorbar_title_text'] = self.DEFAULTS["coloraxis_colorbar_title_text"]
        self.__parameters['coloraxis_colorbar_title_font_family'] = self.DEFAULTS["coloraxis_colorbar_title_font_family"]
        self.__parameters['coloraxis_colorbar_title_font_size'] = self.DEFAULTS["coloraxis_colorbar_title_font_size"]
        self.__parameters['coloraxis_colorbar_title_font_color'] = self.DEFAULTS["coloraxis_colorbar_title_font_color"]
        self.__parameters['coloraxis_colorbar_title_side'] = self.DEFAULTS["coloraxis_colorbar_title_side"]        

    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR TITLE_TEXT PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_title_text(self):
        """Returns the coloraxis_colorbar_title_text attribute.
        
        Sets the title of the color bar. 

        """

        return self.__parameters['coloraxis_colorbar_title_text']

    @coloraxis_colorbar_title_text.setter
    def coloraxis_colorbar_title_text(self, value):
        """Sets the coloraxis_colorbar_title_text attribute.
        
        Parameters
        ----------
        value : str
            Sets the title of the color bar. 

        """                

        self.__parameters['coloraxis_colorbar_title_text'] = value        
          
    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR TITLE_FONT_FAMILY PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_title_font_family(self):
        """Returns the coloraxis_colorbar_title_font_family attribute.
        
        Sets the font family of the title of the color bar. 

        """

        return self.__parameters['coloraxis_colorbar_title_font_family']

    @coloraxis_colorbar_title_font_family.setter
    def coloraxis_colorbar_title_font_family(self, value):
        """Sets the coloraxis_colorbar_title_font_family attribute.
        
        Parameters
        ----------
        value : str
            Sets the font family of the title of the color bar. 

        """                

        self.__parameters['coloraxis_colorbar_title_font_family'] = value                  

    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR TITLE_FONT_SIZE PROPERTIES                #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_title_font_size(self):
        """Returns the coloraxis_colorbar_title_font_size attribute.
        
        Sets the font size of the title of the color bar. 

        """

        return self.__parameters['coloraxis_colorbar_title_font_size']

    @coloraxis_colorbar_title_font_size.setter
    def coloraxis_colorbar_title_font_size(self, value):
        """Sets the coloraxis_colorbar_title_font_size attribute.
        
        Parameters
        ----------
        value : int >= 1
            Sets the font size of the title of the color bar. 

        """                
        if isinstance(value, (int, float)) and value >= 1:
            self.__parameters['coloraxis_colorbar_title_font_size'] = value          
        else:
            raise ValueError("value must be an number >= 1.")

    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR TITLE_FONT_COLOR PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_title_font_color(self):
        """Returns the coloraxis_colorbar_title_font_color attribute.
        
        Sets the font color of the title of the color bar. 

        """

        return self.__parameters['coloraxis_colorbar_title_font_color']

    @coloraxis_colorbar_title_font_color.setter
    def coloraxis_colorbar_title_font_color(self, value):
        """Sets the coloraxis_colorbar_title_font_color attribute.
        
        Parameters
        ----------
        value : str
            Sets the font color of the title of the color bar. 

        """                

        self.__parameters['coloraxis_colorbar_title_font_color'] = value              

    # ----------------------------------------------------------------------- #
    #            COLORAXIS COLORBAR TITLE_SIDE PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_title_side(self):
        """Returns the coloraxis_colorbar_title_side attribute.
        
        Determines the location of color bar's title with respect to the 
        color bar. 

        """

        return self.__parameters['coloraxis_colorbar_title_side']

    @coloraxis_colorbar_title_side.setter
    def coloraxis_colorbar_title_side(self, value):
        """Sets the coloraxis_colorbar_title_side attribute.
        
        Parameters
        ----------
        value : str. One of 'right', 'top', 'bottom'. Default = 'top'
            Determines the location of color bar's title with respect to the 
            color bar. 0

        """                

        valid_values = ['right', 'top', 'bottom']
        if value in valid_values:
            self.__parameters['coloraxis_colorbar_title_side'] = value              
        else:
            raise ValueError("colorbar_title_side must be 'right', 'top', \
                or 'bottom'.")
