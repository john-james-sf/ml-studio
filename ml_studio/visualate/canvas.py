#!/usr/bin/env python3
# =========================================================================== #
#                                  CANVAS                                     #
# =========================================================================== #
# =========================================================================== #
# Project: Visualate                                                          #
# Version: 0.1.0                                                              #
# File: \base.py                                                              #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Wednesday November 27th 2019, 10:28:47 am                      #
# Last Modified: Wednesday November 27th 2019, 12:51:57 pm                    #
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
    * CanvasTitle : Configuration class for plot title options
    * CanvasLegend : Configuration class for plot legend options
    * CanvasMargins : Configuration class for plot margin options
    * CanvasSize : Configuration class for plot size options
    * CanvasFont : Configuration class for plot font options
    * CanvasColors : Configuration class for plot color options
    * CanvasXAxis : Configuration class for plot x-axis options
    * CanvasYAxis : Configuration class for plot y-axis options

    Factory
    ---------
    * CanvasFactor : Creates the appropriate component class

"""
import os
import time

import autopep8
from abc import ABC, abstractmethod, ABCMeta
# --------------------------------------------------------------------------- #
#                                Canvas                                       #
# --------------------------------------------------------------------------- #
class Canvas():    
    """
    A container class holding the various Canvas components. A Canvas is a 
    collection of configuration options for a visualization. The set of 
    configurable options are divided into groups and classes for managing
    the configurations in the group.

    The Canvas components align, more or less, with the structure of the 
    plotly layout structure and are:

        * Title             * Font
        * Legend            * Colors
        * Margins           * X Axis
        * Size              * Y Axis

    Canvas components are exposed as properties of the Canvas class. 
    Instantiation will generate a set of default components, containing 
    its default settings.

    Attributes
    ----------
    title : CanvasTitle, Optional 
        Configuration options for the plot title.

    legend : CanvasLegend, Optional 
        Configuration options for the plot legend.

    margins : CanvasMargins, Optional 
        Configuration options for the plot margins.

    size : CanvasSize, Optional 
        Configuration options for the plot size.

    font : CanvasFont, Optional 
        Configuration options for the plot fonts.

    colors : CanvasColors, Optional 
        Configuration options for the plot colors.

    x_axis : CanvasXAxis, Optional 
        Configuration options for the plot x_axis.

    y_axis : CanvasYAxis, Optional 
        Configuration options for the plot y_axis.                                                

    """
    canvas_components = ['title', 'legend', 'margins', 'size', 'font', 'colors',
                         'x_axis', 'y_axis']
    def __init__(self):
        self.components = {}

    # ----------------------------------------------------------------------- #
    #                          RESET METHODS                                  #
    # ----------------------------------------------------------------------- #       
    def reset(self):
        """Resets the Canvas with default components."""
        # Instantiate a factory for obtaining the Canvas* component.
        factory = CanvasFactory()
        for component in canvas_components:
            self.components[component] = factory.get_component(component)


    # ----------------------------------------------------------------------- #
    #                            TITLE PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def title(self):
        """Returns the CanvasTitle object."""
        return self.components.get('title')

    @title.setter
    def title(self, value):
        """Sets the CanvasTitle object."""
        self.components['title'] = value

    @title.deleter
    def title(self, value):
        """Deletes the CanvasTitle object."""
        try:
            del self.components['title']
        except KeyError:
            print("There is no 'title' attribute for the Canvas class.")

    # ----------------------------------------------------------------------- #
    #                            LEGEND PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def legend(self):
        """Returns the CanvasLegend object."""
        return self.components.get('legend')

    @legend.setter
    def legend(self, value):
        """Sets the CanvasLegend object."""
        self.components['legend'] = value

    @legend.deleter
    def legend(self, value):
        """Deletes the CanvasLegend object."""
        try:
            del self.components['legend']
        except KeyError:
            print("There is no 'legend' attribute for the Canvas class.")

    # ----------------------------------------------------------------------- #
    #                            MARGINS PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def margins(self):
        """Returns the CanvasMargins object."""
        return self.components.get('margins')

    @margins.setter
    def margins(self, value):
        """Sets the CanvasMargins object."""
        self.components['margins'] = value

    @margins.deleter
    def margins(self, value):
        """Deletes the CanvasMargins object."""
        try:
            del self.components['margins']
        except KeyError:
            print("There is no 'margins' attribute for the Canvas class.")

    # ----------------------------------------------------------------------- #
    #                            SIZE PROPERTIES                              #
    # ----------------------------------------------------------------------- #
    @property
    def size(self):
        """Returns the CanvasSize object."""
        return self.components.get('size')

    @size.setter
    def size(self, value):
        """Sets the CanvasSize object."""
        self.components['size'] = value

    @size.deleter
    def size(self, value):
        """Deletes the CanvasSize object."""
        try:
            del self.components['size']
        except KeyError:
            print("There is no 'size' attribute for the Canvas class.")

    # ----------------------------------------------------------------------- #
    #                            FONT PROPERTIES                              #
    # ----------------------------------------------------------------------- #
    @property
    def font(self):
        """Returns the CanvasFont object."""
        return self.components.get('font')

    @font.setter
    def font(self, value):
        """Sets the CanvasFont object."""
        self.components['font'] = value

    @font.deleter
    def font(self, value):
        """Deletes the CanvasFont object."""
        try:
            del self.components['font']
        except KeyError:
            print("There is no 'font' attribute for the Canvas class.")
    # ----------------------------------------------------------------------- #
    #                           COLORS PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def colors(self):
        """Returns the CanvasColors object."""
        return self.components.get('colors')

    @colors.setter
    def colors(self, value):
        """Sets the CanvasColors object."""
        self.components['colors'] = value

    @colors.deleter
    def colors(self, value):
        """Deletes the CanvasColors object."""
        try:
            del self.components['colors']
        except KeyError:
            print("There is no 'colors' attribute for the Canvas class.")
    # ----------------------------------------------------------------------- #
    #                           X_AXIS PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def x_axis(self):
        """Returns the CanvasXAxis object."""
        return self.components.get('x_axis')

    @x_axis.setter
    def x_axis(self, value):
        """Sets the CanvasXAxis object."""
        self.components['x_axis'] = value

    @x_axis.deleter
    def x_axis(self, value):
        """Deletes the CanvasXAxis object."""
        try:
            del self.components['x_axis']
        except KeyError:
            print("There is no 'x_axis' attribute for the Canvas class.")       

    # ----------------------------------------------------------------------- #
    #                           Y_AXIS PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def y_axis(self):
        """Returns the CanvasYAxis object."""
        return self.components.get('y_axis')

    @y_axis.setter
    def y_axis(self, value):
        """Sets the CanvasYAxis object."""
        self.components['y_axis'] = value

    @y_axis.deleter
    def y_axis(self, value):
        """Deletes the CanvasYAxis object."""
        try:
            del self.components['y_axis']
        except KeyError:
            print("There is no 'y_axis' attribute for the Canvas class.")     


# --------------------------------------------------------------------------- #
#                            CanvasFactory                                    #
# --------------------------------------------------------------------------- #
class CanvasFactory():    
    """
    The CanvasFactory class determines on the basis of the component parameter,
    which component to create, instantiates the component and returns the 
    Canvas* component to the user.

    The factory currently supports the following Canvas components:

        * Title             * Font
        * Legend            * Colors
        * Margins           * X Axis
        * Size              * Y Axis

    Canvas components are instantiated with default settings for the 
    configurable options they contain. 

    """
    canvas_components = ['title', 'legend', 'margins', 'size', 'font', 'colors',
                         'x_axis', 'y_axis']
    def __init__(self):
        pass

    def make_component(self, component):
        """Returns the component specified by the component str parameter."""

        if component not in canvas_components:
            raise ValueError("Component %s is not a valid component. Valid \
                              Canvas components include %s." \
                                  % (component, canvas_components))

        dispatcher = {
            'title' : CanvasTitle(),
            'legend' : CanvasLegend(),
            'margins' : CanvasMargins(),
            'size' : CanvasSize(),
            'font' : CanvasFont(),
            'colors' : CanvasColors(),
            'x_axis' : CanvasXAxis(),
            'y_axis' : CanvasYAxis()
        }
        return dispatcher.get(component)

# --------------------------------------------------------------------------- #
#                            CanvasComponent                                  #
# --------------------------------------------------------------------------- #    
class CanvasComponent(ABC):
    """Abstract base class for Canvas component classes."""

    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        """Resets configuration to default values."""
        pass

# --------------------------------------------------------------------------- #
#                              CanvasTitle                                    #
# --------------------------------------------------------------------------- #    
class CanvasTitle(CanvasComponent):
    """Configuration options for plot titles."""

    DEFAULTS = {
        'text' : '',
        'font_family' : '',
        'font_size' : None,
        'font_color' : '',
        'xref' : 'container',
        'yref' : 'container',
        'x' : 0.5,
        'y' : 'auto',
        'xanchor' : 'auto',
        'yanchor' : 'auto',
        'pad' : {'t':0, 'b': 0, 'l':0}
    }

    def __init__(self):
        self.__title_text = ''
        self.__title_font_family = ''   
        self.__title_font_size = None
        self.__title_font_color = ''
        self.__title_xref = 'container'
        self.__title_yref = 'container'
        self.__title_x = 0.5
        self.__title_y = 'auto'
        self.__title_xanchor = 'auto'
        self.__title_yanchor = 'auto'
        self.__title_pad = {'t':0, 'b': 0, 'l':0}

    def reset(self):
        self.__title_text = self.DEFAULTS['text']
        self.__title_font_family = self.DEFAULTS['font_family']
        self.__title_font_size = self.DEFAULTS['font_size']
        self.__title_font_color = self.DEFAULTS['font_color']
        self.__title_xref = self.DEFAULTS['xref']
        self.__title_yref = self.DEFAULTS['yref']
        self.__title_x = self.DEFAULTS['x']
        self.__title_y = self.DEFAULTS['y']
        self.__title_xanchor = self.DEFAULTS['xanchor']
        self.__title_yanchor = self.DEFAULTS['yanchor']
        self.__title_pad = self.DEFAULTS['pad']

    # ----------------------------------------------------------------------- #
    #                        TITLE TEXT PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def title_text(self):
        """Returns the title_text attribute."""
        return self.__title_text

    @title_text.setter
    def title_text(self, value):
        """Sets the title_text attribute."""
        self.__title_text = value

    # ----------------------------------------------------------------------- #
    #                     TITLE FONT FAMILY PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def title_font_family(self):
        """Returns the title_font_family attribute."""
        return self.__title_font_family

    @title_font_family.setter
    def title_font_family(self, value):
        """Sets the title_font_family attribute."""
        self.__title_font_family = value

    # ----------------------------------------------------------------------- #
    #                        TITLE FONT SIZE PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def title_font_size(self):
        """Returns the title_font_size attribute."""
        return self.__title_font_size

    @title_font_size.setter
    def title_font_size(self, value):
        """Sets the title_font_size attribute."""
        if value >= 1:            
            self.__title_font_size = value
        else:
            raise ValueError("Font size must be greater or equal to 1.")

    # ----------------------------------------------------------------------- #
    #                        TITLE FONT COLOR PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def title_font_color(self):
        """Returns the title_font_color attribute."""
        return self.__title_font_color

    @title_font_color.setter
    def title_font_color(self, value):
        """Sets the title_font_color attribute."""
        self.__title_font_color = value        

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
        return self.__title_xref

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
            self.__title_xref = value                
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
        return self.__title_yref

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
            self.__title_yref = value                
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
        return self.__title_x

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
            self.__title_x = value                
        else:
            raise ValueError("x must be between 0 and 1 inclusive.")               

    # ----------------------------------------------------------------------- #
    #                        TITLE Y PROPERTIES                               #
    # ----------------------------------------------------------------------- #
    @property
    def title_y(self):
        """Returns the title_y attribute.

        Specifies the y position with respect to 'yref' in normalized 
        coordinates from '0' (bottom) to '1' (top). 

        """

        return self.__title_y

    @title_y.setter
    def title_y(self, value):
        """Sets the title_y attribute.

        Parameters
        ----------
        value : float, Default = 'auto'
            Specifies the x position with respect to 'xref' in normalized 
            coordinates from '0' (left) to '1' (right)

        """ 

        if value >= 0 and value <= 1:
            self.__title_y = value                
        else:
            raise ValueError("x must be between 0 and 1 inclusive.")                       

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

        return self.__title_xanchor

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
            self.__title_xanchor = value                
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

        return self.__title_yanchor

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
            self.__title_yanchor = value                
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

        return self.__title_pad

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
                    self.__title_pad = value
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
        'bgcolor' : '',   
        'bordercolor' : '#444',
        'borderwidth' : 0,
        'font_family' : '',
        'font_size' : None,
        'font_color' : '',
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
        self.__legend_show = True
        self.__legend_bgcolor = ''   
        self.__legend_bordercolor = '#444'
        self.__legend_borderwidth = 0
        self.__legend_font_family = ''
        self.__legend_font_size = None
        self.__legend_font_color = ''
        self.__legend_orientation = 'v'
        self.__legend_itemsizing = 'trace'
        self.__legend_itemclick = 'toggle'
        self.__legend_x = 1.02
        self.__legend_y = 1
        self.__legend_xanchor = 'left'
        self.__legend_yanchor = 'auto'
        self.__legend_valign = 'middle'

    def reset(self):
        self.__legend_show = self.DEFAULTS['show']
        self.__legend_bgcolor = self.DEFAULTS['bgcolor']
        self.__legend_bordercolor = self.DEFAULTS['bordercolor']
        self.__legend_borderwidth = self.DEFAULTS['borderwidth']
        self.__legend_font_family = self.DEFAULTS['font_family']
        self.__legend_font_size = self.DEFAULTS['font_size']
        self.__legend_font_color = self.DEFAULTS['font_color']
        self.__legend_orientation = self.DEFAULTS['orientation']
        self.__legend_itemsizing = self.DEFAULTS['itemsizing']
        self.__legend_itemclick = self.DEFAULTS['itemclick']
        self.__legend_x = self.DEFAULTS['x']
        self.__legend_y = self.DEFAULTS['y']
        self.__legend_xanchor = self.DEFAULTS['xanchor']
        self.__legend_yanchor = self.DEFAULTS['yanchor']
        self.__legend_valign = self.DEFAULTS['valign']

    # ----------------------------------------------------------------------- #
    #                       LEGEND SHOW PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def legend_show(self):
        """Returns the legend_show attribute."""
        return self.__legend_show

    @legend_show.setter
    def legend_show(self, value):
        """Sets the legend_show attribute."""
        self.__legend_show = value        

    # ----------------------------------------------------------------------- #
    #                     LEGEND BGCOLOR PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def legend_bgcolor(self):
        """Returns the legend_bgcolor attribute."""
        return self.__legend_bgcolor

    @legend_bgcolor.setter
    def legend_bgcolor(self, value):
        """Sets the legend_bgcolor attribute."""
        self.__legend_bgcolor = value           

    # ----------------------------------------------------------------------- #
    #                     LEGEND BORDER COLOR PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_bordercolor(self):
        """Returns the legend_bordercolor attribute."""
        return self.__legend_bordercolor

    @legend_bordercolor.setter
    def legend_bordercolor(self, value):
        """Sets the legend_bordercolor attribute."""
        self.__legend_bordercolor = value              

    # ----------------------------------------------------------------------- #
    #                     LEGEND BORDER WIDTH PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_borderwidth(self):
        """Returns the legend_borderwidth attribute."""
        return self.__legend_borderwidth

    @legend_borderwidth.setter
    def legend_borderwidth(self, value):
        """Sets the legend_borderwidth attribute."""
        self.__legend_borderwidth = value            

    # ----------------------------------------------------------------------- #
    #                     LEGEND FONT FAMILY PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def legend_font_family(self):
        """Returns the legend_font_family attribute."""
        return self.__legend_font_family

    @legend_font_family.setter
    def legend_font_family(self, value):
        """Sets the legend_font_family attribute."""
        self.__legend_font_family = value

    # ----------------------------------------------------------------------- #
    #                        LEGEND FONT SIZE PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_font_size(self):
        """Returns the legend_font_size attribute."""
        return self.__legend_font_size

    @legend_font_size.setter
    def legend_font_size(self, value):
        """Sets the legend_font_size attribute."""
        if value >= 1:            
            self.__legend_font_size = value
        else:
            raise ValueError("Font size must be greater or equal to 1.")

    # ----------------------------------------------------------------------- #
    #                        LEGEND FONT COLOR PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def legend_font_color(self):
        """Returns the legend_font_color attribute."""
        return self.__legend_font_color

    @legend_font_color.setter
    def legend_font_color(self, value):
        """Sets the legend_font_color attribute."""
        self.__legend_font_color = value           

    # ----------------------------------------------------------------------- #
    #                        LEGEND ORIENTATION PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def legend_orientation(self):
        """Returns the legend_orientation attribute."""
        return self.__legend_orientation

    @legend_orientation.setter
    def legend_orientation(self, value):
        """Sets the legend_orientation attribute."""
        self.__legend_orientation = value             

    # ----------------------------------------------------------------------- #
    #                        LEGEND ITEMSIZING PROPERTIES                     #
    # ----------------------------------------------------------------------- #
    @property
    def legend_itemsizing(self):
        """Returns the legend_itemsizing attribute."""
        return self.__legend_itemsizing

    @legend_itemsizing.setter
    def legend_itemsizing(self, value):
        """Sets the legend_itemsizing attribute."""
        valid_values = ['trace', 'constant']
        if value in valid_values:
            self.__legend_itemsizing = value               
        else:
            raise ValueError("legend_itemsizing must be 'trace' or 'constant'")

    # ----------------------------------------------------------------------- #
    #                        LEGEND ITEMCLICK PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def legend_itemclick(self):
        """Returns the legend_itemclick attribute."""
        return self.__legend_itemclick

    @legend_itemclick.setter
    def legend_itemclick(self, value):
        """Sets the legend_itemclick attribute."""
        valid_values = ['toggle', 'toggleothers', False]
        if value in valid_values:
            self.__legend_itemclick = value                       
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
        return self.__legend_x

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
        if value >= -2 and value <= 3:
            self.__legend_x = value                
        else:
            raise ValueError("x must be between -2 and 3 inclusive.")               

    # ----------------------------------------------------------------------- #
    #                        LEGEND Y PROPERTIES                              #
    # ----------------------------------------------------------------------- #
    @property
    def legend_y(self):
        """Returns the legend_y attribute.

        Specifies the y position with respect to 'yref' in normalized 
        coordinates from '0' (bottom) to '1' (top). 

        """

        return self.__legend_y

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

        if value >= -2 and value <= 3:
            self.__legend_y = value                
        else:
            raise ValueError("x must be between -2 and 3 inclusive.")                       

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

        return self.__legend_xanchor

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
            self.__legend_xanchor = value                
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

        return self.__legend_yanchor

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
            self.__legend_yanchor = value                
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

        return self.__legend_valign

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
            self.__legend_valign = value                
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
        self.__margins_left = 80
        self.__margins_top = 100   
        self.__margins_bottom = 80
        self.__margins_pad = 0

    def reset(self):
        self.__margins_left = self.DEFAULTS['left']
        self.__margins_top = self.DEFAULTS['top']   
        self.__margins_bottom = self.DEFAULTS['bottom']
        self.__margins_pad = self.DEFAULTS['pad']
        

    # ----------------------------------------------------------------------- #
    #                       MARGINS_LEFT PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def margins_left(self):
        """Returns the margins_left attribute."""
        return self.__margins_left

    @margins_left.setter
    def margins_left(self, value):
        """Sets the margins_left attribute."""
        self.__margins_left = value        

    # ----------------------------------------------------------------------- #
    #                       MARGINS_TOP PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def margins_top(self):
        """Returns the margins_top attribute."""
        return self.__margins_top

    @margins_top.setter
    def margins_top(self, value):
        """Sets the margins_top attribute."""
        self.__margins_top = value        

    # ----------------------------------------------------------------------- #
    #                       MARGINS_BOTTOM PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def margins_bottom(self):
        """Returns the margins_bottom attribute."""
        return self.__margins_bottom

    @margins_bottom.setter
    def margins_bottom(self, value):
        """Sets the margins_bottom attribute."""
        self.__margins_bottom = value             

    # ----------------------------------------------------------------------- #
    #                       MARGINS_PAD PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def margins_pad(self):
        """Returns the margins_pad attribute."""
        return self.__margins_pad

    @margins_pad.setter
    def margins_pad(self, value):
        """Sets the margins_pad attribute."""
        self.__margins_pad = value             

# --------------------------------------------------------------------------- #
#                              CanvasSize                                  #
# --------------------------------------------------------------------------- #    
class CanvasSize(CanvasComponent):
    """Configuration options for plot size."""

    DEFAULTS = {
        'autosize' : True,
        'width' : 700,   
        'height' : 450
    }

    def __init__(self):
        self.__size_autosize = True
        self.__size_width = 700   
        self.__size_height = 450

    def reset(self):
        self.__size_autosize= self.DEFAULTS['autosize']
        self.__size_width = self.DEFAULTS['width']   
        self.__size_height = self.DEFAULTS['height']
        

    # ----------------------------------------------------------------------- #
    #                         AUTOSIZE PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def size_autosize(self):
        """Returns the size_autosize attribute.
        
        Determines whether or not a layout width or height that has been left 
        undefined by the user is initialized on each relayout.

        """

        return self.__size_autosize

    @size_autosize.setter
    def size_autosize(self, value):
        """Sets the size_autosize attribute.
        
        Parameters
        ----------
        value : bool
            Determines whether or not a layout width or height that has 
            been left undefined by the user is initialized on each relayout.

        """

        self.__size_autosize = value        

    # ----------------------------------------------------------------------- #
    #                         WIDTH PROPERTIES                                #
    # ----------------------------------------------------------------------- #
    @property
    def size_width(self):
        """Returns the size_width attribute."""

        return self.__size_width

    @size_width.setter
    def size_width(self, value):
        """Sets the size_width attribute."""
        if value >= 10:
            self.__size_width = value        
        else:
            raise ValueError("Width must be a number greater or equal to 10.")

    # ----------------------------------------------------------------------- #
    #                         HEIGHT PROPERTIES                               #
    # ----------------------------------------------------------------------- #
    @property
    def size_height(self):
        """Returns the size_height attribute."""

        return self.__size_height

    @size_height.setter
    def size_height(self, value):
        """Sets the size_height attribute."""
        if value >= 10:
            self.__size_height = value        
        else:
            raise ValueError("height must be a number greater or equal to 10.")        

# --------------------------------------------------------------------------- #
#                              CanvasFont                                     #
# --------------------------------------------------------------------------- #    
class CanvasFont(CanvasComponent):
    """Configuration options for plot font."""

    DEFAULTS = {
        'family' : '',
        'size' : 12,   
        'color' : '#444',
        'separators' : '.,'
    }

    def __init__(self):
        self.__font_family = ''
        self.__font_size = 12   
        self.__font_color = '#444'
        self.__font_separators = '.,'

    def reset(self):
        self.__font_family= self.DEFAULTS['family']
        self.__font_size = self.DEFAULTS['size']   
        self.__font_color = self.DEFAULTS['color']
        self.__font_separators = self.DEFAULTS['separators']
        

    # ----------------------------------------------------------------------- #
    #                         FONT FAMILY PROPERTIES                          #
    # ----------------------------------------------------------------------- #
    @property
    def font_family(self):
        """Returns the font_family attribute."""
        return self.__font_family

    @font_family.setter
    def font_family(self, value):
        """Sets the font_family attribute."""
        self.__font_family = value              

    # ----------------------------------------------------------------------- #
    #                         FONT SIZE PROPERTIES                            #
    # ----------------------------------------------------------------------- #
    @property
    def font_size(self):
        """Returns the font_size attribute."""
        return self.__font_size

    @font_size.setter
    def font_size(self, value):
        """Sets the font_size attribute."""
        self.__font_size = value              

    # ----------------------------------------------------------------------- #
    #                         FONT COLOR PROPERTIES                           #
    # ----------------------------------------------------------------------- #
    @property
    def font_color(self):
        """Returns the font_color attribute."""
        return self.__font_color

    @font_color.setter
    def font_color(self, value):
        """Sets the font_color attribute."""
        self.__font_color = value                  

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
        return self.__font_separators

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

        self.__font_separators = value            

# --------------------------------------------------------------------------- #
#                              CanvasColors                                   #
# --------------------------------------------------------------------------- #    
class CanvasColors(CanvasComponent):
    """Configuration options for plot colors."""

    DEFAULTS = {
        'paper_bgcolor' : '#fff',
        'plot_bgcolor' : '#fff',   
        'colorscale_sequential' : [[0, rgb(220,220,220)], [0.2, rgb(245,195,157)], 
                                  [0.4, rgb(245,160,105)], [1, rgb(178,10,28)], ],
        'colorscale_sequentialminus' : [[0, rgb(5,10,172)], 
                                        [0.35, rgb(40,60,190)], 
                                        [0.5, rgb(70,100,245)], 
                                        [0.6, rgb(90,120,245)], 
                                        [0.7, rgb(106,137,247)], 
                                        [1, rgb(220,220,220)], ],
        'colorscale_diverging' : [[0, rgb(5,10,172)], [0.35, rgb(106,137,247)], 
                                [0.5, rgb(190,190,190)], [0.6, rgb(220,170,132)], 
                                [0.7, rgb(230,145,90)], [1, rgb(178,10,28)], ],
        'colorway' : ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],

        'coloraxis_cauto' : True,
        'coloraxis_cmin' : None,
        'coloraxis_cmax' : None,
        'coloraxis_cmid' : None,
        'coloraxis_colorscale' : [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
        'coloraxis_autocolorscale' : True,
        'coloraxis_reversescale' : True,
        'coloraxis_showscale' : True,
        'coloraxis_colorbar_thicknessmode' : 'pixels',
        'coloraxis_colorbar_thickness' : 30,
        'coloraxis_colorbar_lenmode' : 'fraction',
        'coloraxis_colorbar_len' : 1,
        'coloraxis_colorbar_x' : 1.02,
        'coloraxis_colorbar_y' : 0.5,
        'coloraxis_colorbar_xanchor' : 'left',
        'coloraxis_colorbar_yanchor' : 'middle',
        'coloraxis_colorbar_xpad' : 10,
        'coloraxis_colorbar_ypad' : 10,        
        'coloraxis_colorbar_outlinecolor' : '#444',        
        'coloraxis_colorbar_outlinewidth' : 1,        
        'coloraxis_colorbar_bordercolor' : '#444',        
        'coloraxis_colorbar_borderwidth' : 0,        
        'coloraxis_colorbar_bgcolor' : "rgba(0,0,0,0)",
        'coloraxis_colorbar_tickmode' : "array",
        'coloraxis_colorbar_nticks' : 0,
        'coloraxis_colorbar_tick0' : None,
        'coloraxis_colorbar_dtick' : None,
        'coloraxis_colorbar_tickvals' : None,
        'coloraxis_colorbar_ticktext' : None,
        'coloraxis_colorbar_ticks' : "",
        'coloraxis_colorbar_ticklen' : 5,
        'coloraxis_colorbar_tickwidth' : 1,
        'coloraxis_colorbar_tickcolor' : '#444',
        'coloraxis_colorbar_showticklabels' : True,
        'coloraxis_colorbar_tickfont_family' : None,
        'coloraxis_colorbar_tickfont_size' : 1,
        'coloraxis_colorbar_tickfont_color' : None,
        'coloraxis_colorbar_tickangle' : 'auto',
        'coloraxis_colorbar_tickformatstops_enabled' : True,
        'coloraxis_colorbar_tickformatstops_dtickrange' : None,
        'coloraxis_colorbar_tickformatstops_value' : '',
        'coloraxis_colorbar_tickformatstops_name' : None,
        'coloraxis_colorbar_tickformatstops_templateitemname' : None,
        'coloraxis_colorbar_tickprefix' : '',
        'coloraxis_colorbar_showtickprefix' : 'all',
        'coloraxis_colorbar_ticksuffix' : '',
        'coloraxis_colorbar_showticksuffix' : 'all',
        'coloraxis_colorbar_separatethousands' : True,
        'coloraxis_colorbar_exponentialformat' : 'B',
        'coloraxis_colorbar_showexponent' : 'all',
        'coloraxis_colorbar_title_text' : "",
        'coloraxis_colorbar_title_font_family' : None,
        'coloraxis_colorbar_title_font_size' : 1,
        'coloraxis_colorbar_title_font_color' : None,
        'coloraxis_colorbar_title_side' : 'top'
    }

    def __init__(self):
        self.__paper_bgcolor = '#fff'
        self.__plot_bgcolor = '#fff'   
        self.__colorscale_sequential = [[0, 'rgb(220,220,220)'], [0.2, 'rgb(245,195,157)'], 
                                  [0.4, 'rgb(245,160,105)'], [1, 'rgb(178,10,28)'], ]
        self.__colorscale_sequentialminus = [[0, 'rgb(5,10,172)'], 
                                        [0.35, 'rgb(40,60,190)'], 
                                        [0.5, 'rgb(70,100,245)'], 
                                        [0.6, 'rgb(90,120,245)'], 
                                        [0.7, 'rgb(106,137,247)'], 
                                        [1, 'rgb(220,220,220)'], ]
        self.__colorscale_diverging = [[0, 'rgb(5,10,172)'], [0.35, 'rgb(106,137,247)'], 
                                [0.5, 'rgb(190,190,190)'], [0.6, 'rgb(220,170,132)'], 
                                [0.7, 'rgb(230,145,90)'], [1, 'rgb(178,10,28)'], ]
        self.__colorway = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        self.__coloraxis_cauto = True
        self.__coloraxis_cmin = None
        self.__coloraxis_cmax = None
        self.__coloraxis_cmid = None
        self.__coloraxis_colorscale = [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]
        self.__coloraxis_autoscale = True
        self.__coloraxis_reversescale = True
        self.__coloraxis_showscale = True
        self.__coloraxis_colorbar_thicknessmode = 'pixels'
        self.__coloraxis_colorbar_thickness = 30
        self.__coloraxis_colorbar_lenmode = 'fraction'
        self.__coloraxis_colorbar_len = 1
        self.__coloraxis_colorbar_x = 1.02
        self.__coloraxis_colorbar_y = 0.5
        self.__coloraxis_colorbar_xanchor = 'left'
        self.__coloraxis_colorbar_yanchor = 'middle'
        self.__coloraxis_colorbar_xpad = 10
        self.__coloraxis_colorbar_ypad = 10        
        self.__coloraxis_colorbar_outlinecolor = '#444'        
        self.__coloraxis_colorbar_outlinewidth = 1        
        self.__coloraxis_colorbar_bordercolor = '#444'        
        self.__coloraxis_colorbar_borderwidth = 0        
        self.__coloraxis_colorbar_bgcolor = "rgba(0000)"
        self.__coloraxis_colorbar_tickmode = "array"
        self.__coloraxis_colorbar_nticks = 0
        self.__coloraxis_colorbar_tick0 = None
        self.__coloraxis_colorbar_dtick = None
        self.__coloraxis_colorbar_tickvals = None
        self.__coloraxis_colorbar_ticktext = None
        self.__coloraxis_colorbar_ticks = ""
        self.__coloraxis_colorbar_ticklen = 5
        self.__coloraxis_colorbar_tickwidth = 1
        self.__coloraxis_colorbar_tickcolor = '#444'
        self.__coloraxis_colorbar_showticklabels = True
        #TODO: pickup from here
        self.__coloraxis_colorbar_tickfont_family = None
        self.__coloraxis_colorbar_tickfont_size = 1
        self.__coloraxis_colorbar_tickfont_color = None
        self.__coloraxis_colorbar_tickangle = 'auto'
        self.__coloraxis_colorbar_tickformatstops_enabled = True
        self.__coloraxis_colorbar_tickformatstops_dtickrange = None
        self.__coloraxis_colorbar_tickformatstops_value = None
        self.__coloraxis_colorbar_tickformatstops_name = None
        self.__coloraxis_colorbar_tickformatstops_templateitemname = None
        self.__coloraxis_colorbar_tickprefix = ''
        self.__coloraxis_colorbar_showtickprefix = 'all'
        self.__coloraxis_colorbar_ticksuffix = ''
        self.__coloraxis_colorbar_showticksuffix = 'all'
        self.__coloraxis_colorbar_separatethousands = True
        self.__coloraxis_colorbar_exponentialformat = 'B'
        self.__coloraxis_colorbar_showexponent = 'all'
        self.__coloraxis_colorbar_title_text = ""
        self.__coloraxis_colorbar_title_font_family = None
        self.__coloraxis_colorbar_title_font_size = 1
        self.__coloraxis_colorbar_title_font_color = None
        self.__coloraxis_colorbar_title_side = 'top'


    def reset(self):
        """Sets parameters back to their defaults."""
        self.__paper_bgcolor = self.DEFAULTS["paper_bgcolor"]
        self.__plot_bgcolor = self.DEFAULTS["plot_bgcolor"]
        self.__colorscale_sequential = self.DEFAULTS["colorscale_sequential"]
        self.__colorscale_sequentialminus = self.DEFAULTS["colorscale_sequentialminus"]
        self.__colorscale_diverging = self.DEFAULTS["colorscale_diverging"]
        self.__colorway = self.DEFAULTS["colorway"]
        self.__coloraxis_cauto = self.DEFAULTS["coloraxis_cauto"]
        self.__coloraxis_cmin = self.DEFAULTS["coloraxis_cmin"]
        self.__coloraxis_cmax = self.DEFAULTS["coloraxis_cmax"]
        self.__coloraxis_cmid = self.DEFAULTS["coloraxis_cmid"]
        self.__coloraxis_colorscale = self.DEFAULTS["coloraxis_colorscale"]
        self.__coloraxis_autocolorscale = self.DEFAULTS["coloraxis_autocolorscale"]
        self.__coloraxis_reversescale = self.DEFAULTS["coloraxis_reversescale"]
        self.__coloraxis_showscale = self.DEFAULTS["coloraxis_showscale"]
        self.__coloraxis_colorbar_thicknessmode = self.DEFAULTS["coloraxis_colorbar_thicknessmode"]
        self.__coloraxis_colorbar_thickness = self.DEFAULTS["coloraxis_colorbar_thickness"]
        self.__coloraxis_colorbar_lenmode = self.DEFAULTS["coloraxis_colorbar_lenmode"]
        self.__coloraxis_colorbar_len = self.DEFAULTS["coloraxis_colorbar_len"]
        self.__coloraxis_colorbar_x = self.DEFAULTS["coloraxis_colorbar_x"]
        self.__coloraxis_colorbar_y = self.DEFAULTS["coloraxis_colorbar_y"]
        self.__coloraxis_colorbar_xanchor = self.DEFAULTS["coloraxis_colorbar_xanchor"]
        self.__coloraxis_colorbar_yanchor = self.DEFAULTS["coloraxis_colorbar_yanchor"]
        self.__coloraxis_colorbar_xpad = self.DEFAULTS["coloraxis_colorbar_xpad"]
        self.__coloraxis_colorbar_ypad = self.DEFAULTS["coloraxis_colorbar_ypad"]
        self.__coloraxis_colorbar_outlinecolor = self.DEFAULTS["coloraxis_colorbar_outlinecolor"]
        self.__coloraxis_colorbar_outlinewidth = self.DEFAULTS["coloraxis_colorbar_outlinewidth"]
        self.__coloraxis_colorbar_bordercolor = self.DEFAULTS["coloraxis_colorbar_bordercolor"]
        self.__coloraxis_colorbar_borderwidth = self.DEFAULTS["coloraxis_colorbar_borderwidth"]
        self.__coloraxis_colorbar_bgcolor = self.DEFAULTS["coloraxis_colorbar_bgcolor"]
        self.__coloraxis_colorbar_tickmode = self.DEFAULTS["coloraxis_colorbar_tickmode"]
        self.__coloraxis_colorbar_nticks = self.DEFAULTS["coloraxis_colorbar_nticks"]
        self.__coloraxis_colorbar_tick0 = self.DEFAULTS["coloraxis_colorbar_tick0"]
        self.__coloraxis_colorbar_dtick = self.DEFAULTS["coloraxis_colorbar_dtick"]
        self.__coloraxis_colorbar_tickvals = self.DEFAULTS["coloraxis_colorbar_tickvals"]
        self.__coloraxis_colorbar_ticktext = self.DEFAULTS["coloraxis_colorbar_ticktext"]
        self.__coloraxis_colorbar_ticks = self.DEFAULTS["coloraxis_colorbar_ticks"]
        self.__coloraxis_colorbar_ticklen = self.DEFAULTS["coloraxis_colorbar_ticklen"]
        self.__coloraxis_colorbar_tickwidth = self.DEFAULTS["coloraxis_colorbar_tickwidth"]
        self.__coloraxis_colorbar_tickcolor = self.DEFAULTS["coloraxis_colorbar_tickcolor"]
        self.__coloraxis_colorbar_showticklabels = self.DEFAULTS["coloraxis_colorbar_showticklabels"]
        self.__coloraxis_colorbar_tickfont_family = self.DEFAULTS["coloraxis_colorbar_tickfont_family"]
        self.__coloraxis_colorbar_tickfont_size = self.DEFAULTS["coloraxis_colorbar_tickfont_size"]
        self.__coloraxis_colorbar_tickfont_color = self.DEFAULTS["coloraxis_colorbar_tickfont_color"]
        self.__coloraxis_colorbar_tickangle = self.DEFAULTS["coloraxis_colorbar_tickangle"]
        self.__coloraxis_colorbar_tickformatstops_enabled = self.DEFAULTS["coloraxis_colorbar_tickformatstops_enabled"]
        self.__coloraxis_colorbar_tickformatstops_dtickrange = self.DEFAULTS["coloraxis_colorbar_tickformatstops_dtickrange"]
        self.__coloraxis_colorbar_tickformatstops_value = self.DEFAULTS["coloraxis_colorbar_tickformatstops_value"]
        self.__coloraxis_colorbar_tickformatstops_name = self.DEFAULTS["coloraxis_colorbar_tickformatstops_name"]
        self.__coloraxis_colorbar_tickformatstops_templateitemname = self.DEFAULTS["coloraxis_colorbar_tickformatstops_templateitemname"]
        self.__coloraxis_colorbar_tickprefix = self.DEFAULTS["coloraxis_colorbar_tickprefix"]
        self.__coloraxis_colorbar_showtickprefix = self.DEFAULTS["coloraxis_colorbar_showtickprefix"]
        self.__coloraxis_colorbar_ticksuffix = self.DEFAULTS["coloraxis_colorbar_ticksuffix"]
        self.__coloraxis_colorbar_showticksuffix = self.DEFAULTS["coloraxis_colorbar_showticksuffix"]
        self.__coloraxis_colorbar_separatethousands = self.DEFAULTS["coloraxis_colorbar_separatethousands"]
        self.__coloraxis_colorbar_exponentialformat = self.DEFAULTS["coloraxis_colorbar_exponentialformat"]
        self.__coloraxis_colorbar_showexponent = self.DEFAULTS["coloraxis_colorbar_showexponent"]
        self.__coloraxis_colorbar_title_text = self.DEFAULTS["coloraxis_colorbar_title_text"]
        self.__coloraxis_colorbar_title_font_family = self.DEFAULTS["coloraxis_colorbar_title_font_family"]
        self.__coloraxis_colorbar_title_font_size = self.DEFAULTS["coloraxis_colorbar_title_font_size"]
        self.__coloraxis_colorbar_title_font_color = self.DEFAULTS["coloraxis_colorbar_title_font_color"]
        self.__coloraxis_colorbar_title_side = self.DEFAULTS["coloraxis_colorbar_title_side"]        

    # ----------------------------------------------------------------------- #
    #                         PAPER BGCOLOR PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def paper_bgcolor(self):
        """Returns the paper_bgcolor attribute.
        
        Sets the color of paper where the graph is drawn.
        
        """
        
        return self.__paper_bgcolor

    @paper_bgcolor.setter
    def paper_bgcolor(self, value):
        """Sets the paper_bgcolor attribute.
        
        Parameters
        ----------
        value : str, Default = '#fff'
            Sets the color of paper where the graph is drawn.
        """
        self.__paper_bgcolor = value                      

    # ----------------------------------------------------------------------- #
    #                         PLOT BGCOLOR PROPERTIES                         #
    # ----------------------------------------------------------------------- #
    @property
    def plot_bgcolor(self):
        """Returns the plot_bgcolor attribute.
        
        Sets the color of plot where the graph is drawn.
        
        """
        
        return self.__plot_bgcolor

    @plot_bgcolor.setter
    def plot_bgcolor(self, value):
        """Sets the plot_bgcolor attribute.
        
        Parameters
        ----------
        value : str, Default = '#fff'
            Sets the color of plot where the graph is drawn.
        """
        self.__plot_bgcolor = value                 

    # ----------------------------------------------------------------------- #
    #                   COLORSCALE SEQUENTIAL PROPERTIES                      #
    # ----------------------------------------------------------------------- #
    @property
    def colorscale_sequential(self):
        """Returns the colorscale_sequential attribute.
        
        Sets the default sequential colorscale for positive values.
        
        """
        
        return self.__colorscale_sequential

    @colorscale_sequential.setter
    def colorscale_sequential(self, value):
        """Sets the colorscale_sequential attribute.
        
        Parameters
        ----------
        value : list Default = [[0, rgb(220,220,220)], [0.2, rgb(245,195,157)], 
                                  [0.4, rgb(245,160,105)], [1, rgb(178,10,28)], ]
            Sets the default sequential colorscale for positive values. 
        """
        self.__colorscale_sequential = value                   

    # ----------------------------------------------------------------------- #
    #                   COLORSCALE SEQUENTIALMINUS PROPERTIES                 #
    # ----------------------------------------------------------------------- #
    @property
    def colorscale_sequentialminus(self):
        """Returns the colorscale_sequentialminus attribute.
        
        Sets the default sequential colorscale for negative values.
        
        """
        
        return self.__colorscale_sequentialminus

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
        self.__colorscale_sequentialminus = value                           

    # ----------------------------------------------------------------------- #
    #                   COLORSCALE DIVERGING PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def colorscale_diverging(self):
        """Returns the colorscale_diverging attribute.
        
        Sets the default diverging colorscale.
        
        """
        
        return self.__colorscale_diverging

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
        self.__colorscale_diverging = value                           
        
    # ----------------------------------------------------------------------- #
    #                         COLORWAY PROPERTIES                             #
    # ----------------------------------------------------------------------- #
    @property
    def colorway(self):
        """Returns the colorway attribute.
        
        Sets the default trace colors.
        
        """
        
        return self.__colorway

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
        self.__colorway = value                             

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
        
        return self.__coloraxis_cauto

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

        self.__coloraxis_cauto = value         

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
        
        return self.__coloraxis_cmin

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
        
        self.__coloraxis_cmin = value                 

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
        
        return self.__coloraxis_cmax

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
        
        self.__coloraxis_cmax = value            
        
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
        
        return self.__coloraxis_cmid

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
        
        self.__coloraxis_cmid = value            
        
    # ----------------------------------------------------------------------- #
    #                   COLORAXIS COLORSCALE PROPERTIES                       #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorscale(self):
        """Returns the coloraxis_colorscale attribute.
        
        Sets the colorscale. See `Plotly Colorscale 
        <https://plot.ly/python/reference/#layout-coloraxis-colorscale>`_
        
        """
        
        return self.__coloraxis_colorscale

    @coloraxis_colorscale.setter
    def coloraxis_colorscale(self, value):
        """Sets the coloraxis_colorscale attribute.
        
        Parameters
        ----------
        value : list. Default = [[0, rgb(0,0,255)], [1, rgb(255,0,0)]]
            Sets the colorscale. See `Plotly Colorscale 
            <https://plot.ly/python/reference/#layout-coloraxis-colorscale>`_

        """
        
        self.__coloraxis_colorscale = value                
              

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
        
        return self.__coloraxis_autoscale

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
        
        self.__coloraxis_autoscale = value                  

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
        
        return self.__coloraxis_reversescale

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
        
        self.__coloraxis_reversescale = value              

    # ----------------------------------------------------------------------- #
    #                   COLORAXIS SHOWSCALE PROPERTIES                        #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_showscale(self):
        """Returns the coloraxis_showscale attribute.
        
        Determines whether or not a colorbar is displayed for this trace.
        
        """
        
        return self.__coloraxis_showscale

    @coloraxis_showscale.setter
    def coloraxis_showscale(self, value):
        """Sets the coloraxis_showscale attribute.
        
        Parameters
        ----------
        value : bool. Default = True
            Determines whether or not a colorbar is displayed for this trace.

        """
        
        self.__coloraxis_showscale = value          

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
        
        return self.__coloraxis_colorbar_thicknessmode

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
            self.__coloraxis_colorbar_thicknessmode = value              
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
        
        return self.__coloraxis_colorbar_thickness

    @coloraxis_colorbar_thickness.setter
    def coloraxis_colorbar_thickness(self, value):
        """Sets the coloraxis_colorbar_thickness attribute.
        
        Parameters
        ----------
        value : int Default = 30
            Sets the thickness of the color bar This measure excludes the 
            size of the padding, ticks and labels.

        """
        
        if value > 0:
            self.__coloraxis_colorbar_thickness = value              
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
        
        return self.__coloraxis_colorbar_lenmode

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
            self.__coloraxis_colorbar_lenmode = value         
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
        
        return self.__coloraxis_colorbar_len

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
        
        if value > 0:
            self.__coloraxis_colorbar_len = value                        
        else:
            raise ValueError("colorbar_len must be an integer >= 0.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR X PROPERTIES                          #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_x(self):
        """Returns the coloraxis_colorbar_x attribute.
        
        Sets the x position of the color bar (in plot fraction).
        
        """
        
        return self.__coloraxis_colorbar_x

    @coloraxis_colorbar_x.setter
    def coloraxis_colorbar_x(self, value):
        """Sets the coloraxis_colorbar_x attribute.
        
        Parameters
        ----------
        value : int between -2 and 3. Default = 1.02
            Sets the x position of the color bar (in plot fraction).

        """
        
        if value >= -2 and value <= 3:
            self.__coloraxis_colorbar_x = value                 
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
        
        return self.__coloraxis_colorbar_y

    @coloraxis_colorbar_y.setter
    def coloraxis_colorbar_y(self, value):
        """Sets the coloraxis_colorbar_y attribute.
        
        Parameters
        ----------
        value : int between -2 and 3. Default = 0.5
            Sets the x position of the color bar (in plot fraction).

        """
        if value >= -2 and value <= 3:
            self.__coloraxis_colorbar_y = value                 
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
        
        return self.__coloraxis_colorbar_xanchor

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
            self.__coloraxis_colorbar_xanchor = value          
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
        
        return self.__coloraxis_colorbar_yanchor

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
            self.__coloraxis_colorbar_yanchor = value               
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
        
        return self.__coloraxis_colorbar_xpad

    @coloraxis_colorbar_xpad.setter
    def coloraxis_colorbar_xpad(self, value):
        """Sets the coloraxis_colorbar_xpad attribute.
        
        Parameters
        ----------
        value : int. Default = 10
            Sets the amount of padding (in px) along the x direction.

        """
        
        if value >= 0:
            self.__coloraxis_colorbar_xpad = value                    
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
        
        return self.__coloraxis_colorbar_ypad

    @coloraxis_colorbar_ypad.setter
    def coloraxis_colorbar_ypad(self, value):
        """Sets the coloraxis_colorbar_ypad attribute.
        
        Parameters
        ----------
        value : int. Default = 10
            Sets the amount of padding (in px) along the y direction.

        """
        if value >= 0:
            self.__coloraxis_colorbar_ypad = value                    
        else:
            raise ValueError("colorbar_ypad must be an integer >= 0.")        
        self.__coloraxis_colorbar_ypad = value                    

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR OUTLINECOLOR PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_outlinecolor(self):
        """Returns the coloraxis_colorbar_outlinecolor attribute.
        
        Sets the axis line color.
        
        """
        
        return self.__coloraxis_colorbar_outlinecolor

    @coloraxis_colorbar_outlinecolor.setter
    def coloraxis_colorbar_outlinecolor(self, value):
        """Sets the coloraxis_colorbar_outlinecolor attribute.
        
        Parameters
        ----------
        value : str. Default = '#444'
            Sets the axis line color.

        """
        
        self.__coloraxis_colorbar_outlinecolor = value       

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR OUTLINEWIDTH PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_outlinewidth(self):
        """Returns the coloraxis_colorbar_outlinewidth attribute.
        
        Sets the width (in px) of the axis line.
        
        """
        
        return self.__coloraxis_colorbar_outlinewidth

    @coloraxis_colorbar_outlinewidth.setter
    def coloraxis_colorbar_outlinewidth(self, value):
        """Sets the coloraxis_colorbar_outlinewidth attribute.
        
        Parameters
        ----------
        value : int greater than or equal to 0. Default = 1
            Sets the width (in px) of the axis line.

        """
        
        if value >= 0:
            self.__coloraxis_colorbar_outlinewidth = value                    
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
        
        return self.__coloraxis_colorbar_bordercolor

    @coloraxis_colorbar_bordercolor.setter
    def coloraxis_colorbar_bordercolor(self, value):
        """Sets the coloraxis_colorbar_bordercolor attribute.
        
        Parameters
        ----------
        value : str. Default = '#444'
            Sets the axis line color.

        """
        
        self.__coloraxis_colorbar_bordercolor = value            

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR BORDERWIDTH PROPERTIES                #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_borderwidth(self):
        """Returns the coloraxis_colorbar_borderwidth attribute.
        
        Sets the axis line color.
        
        """
        
        return self.__coloraxis_colorbar_borderwidth

    @coloraxis_colorbar_borderwidth.setter
    def coloraxis_colorbar_borderwidth(self, value):
        """Sets the coloraxis_colorbar_borderwidth attribute.
        
        Parameters
        ----------
        value : int greater than or equal to 0. Default = 0
            Sets the axis line color.

        """

        if value >= 0:
            self.__coloraxis_colorbar_borderwidth = value                    
        else:
            raise ValueError("colorbar_borderwidth must be an integer >= 0.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR BGCOLOR PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_bgcolor(self):
        """Returns the coloraxis_colorbar_bgcolor attribute.
        
        Sets the color of padded area.
        
        """
        
        return self.__coloraxis_colorbar_bgcolor

    @coloraxis_colorbar_bgcolor.setter
    def coloraxis_colorbar_bgcolor(self, value):
        """Sets the coloraxis_colorbar_bgcolor attribute.
        
        Parameters
        ----------
        value : color. Default = "rgba(0,0,0,0)"
            Sets the color of padded area.

        """
        
        self.__coloraxis_colorbar_bgcolor = value                            

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
        
        return self.__coloraxis_colorbar_tickmode

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
            self.__coloraxis_colorbar_tickmode = value        
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
        
        return self.__coloraxis_colorbar_nticks

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
        
        if value >= 0:
            self.__coloraxis_colorbar_nticks = value                      
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
        
        return self.__coloraxis_colorbar_tick0

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
        
        self.__coloraxis_colorbar_tick0 = value           

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
        For example `tick0` = 0.1, `dtick` = "L0.5" will put ticks at 0.1, 
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
        
        return self.__coloraxis_colorbar_dtick

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
            For example `tick0` = 0.1, `dtick` = "L0.5" will put ticks at 0.1, 
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
        
        self.__coloraxis_colorbar_dtick = value                             

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKVALS PROPERTIES                   #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickvals(self):
        """Returns the coloraxis_colorbar_tickvals attribute.
        
        Sets the values at which ticks on this axis appear. Only has an 
        effect if `tickmode` is set to "array". Used with `ticktext`
        
        """
        
        return self.__coloraxis_colorbar_tickvals

    @coloraxis_colorbar_tickvals.setter
    def coloraxis_colorbar_tickvals(self, value):
        """Sets the coloraxis_colorbar_tickvals attribute.
        
        Parameters
        ----------
        value : array-like numbers, strings, or datetimes.
            Sets the values at which ticks on this axis appear. Only has an 
            effect if `tickmode` is set to "array". Used with `ticktext`

        """

        self.__coloraxis_colorbar_tickvals = value         

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

        return self.__coloraxis_colorbar_ticktext

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

        self.__coloraxis_colorbar_ticktext = value         

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

        return self.__coloraxis_colorbar_ticks

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
            self.__coloraxis_colorbar_ticks = value                 
        else:
            raise ValueError("colorbar_ticks must be either 'outside', \
                'inside', or ''.")

    # ----------------------------------------------------------------------- #
    #                COLORAXIS COLORBAR TICKLEN PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_ticklen(self):
        """Returns the coloraxis_colorbar_ticklen attribute.
        
        Sets the tick length (in px).
        
        """

        return self.__coloraxis_colorbar_ticklen

    @coloraxis_colorbar_ticklen.setter
    def coloraxis_colorbar_ticklen(self, value):
        """Sets the coloraxis_colorbar_ticklen attribute.
        
        Parameters
        ----------
        value : int >= 0. Default = 5
            Sets the tick length (in px).

        """                
        
        if value >= 0:
            self.__coloraxis_colorbar_ticklen = value                 
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

        return self.__coloraxis_colorbar_tickwidth

    @coloraxis_colorbar_tickwidth.setter
    def coloraxis_colorbar_tickwidth(self, value):
        """Sets the coloraxis_colorbar_tickwidth attribute.
        
        Parameters
        ----------
        value : int >= 0. Default = 1
            Sets the tick length (in px).

        """                
        
        if value >= 0:
            self.__coloraxis_colorbar_tickwidth = value                 
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

        return self.__coloraxis_colorbar_tickcolor

    @coloraxis_colorbar_tickcolor.setter
    def coloraxis_colorbar_tickcolor(self, value):
        """Sets the coloraxis_colorbar_tickcolor attribute.
        
        Parameters
        ----------
        value : int >= 0. Default = 1
            Sets the tick color.

        """                
        
        self.__coloraxis_colorbar_tickcolor = value                 
        

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR SHOWTICKLABELS PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_showticklabels(self):
        """Returns the coloraxis_colorbar_showticklabels attribute.
        
        Determines whether or not the tick labels are drawn.
        
        """

        return self.__coloraxis_colorbar_showticklabels

    @coloraxis_colorbar_showticklabels.setter
    def coloraxis_colorbar_showticklabels(self, value):
        """Sets the coloraxis_colorbar_showticklabels attribute.
        
        Parameters
        ----------
        value : bool. Default = True
            Determines whether or not the tick labels are drawn.

        """                
        
        self.__coloraxis_colorbar_showticklabels = value                    

    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKFONT_FAMILY PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickfont_family(self):
        """Returns the coloraxis_colorbar_tickfont_family attribute.
        
        Sets tick font family.
        
        """

        return self.__coloraxis_colorbar_tickfont_family

    @coloraxis_colorbar_tickfont_family.setter
    def coloraxis_colorbar_tickfont_family(self, value):
        """Sets the coloraxis_colorbar_tickfont_family attribute.
        
        Parameters
        ----------
        value : str
            Sets tick font family.

        """                
        
        self.__coloraxis_colorbar_tickfont_family = value                    

    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKFONT_SIZE PROPERTIES                #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickfont_size(self):
        """Returns the coloraxis_colorbar_tickfont_size attribute.
        
        Sets tick font size.
        
        """

        return self.__coloraxis_colorbar_tickfont_size

    @coloraxis_colorbar_tickfont_size.setter
    def coloraxis_colorbar_tickfont_size(self, value):
        """Sets the coloraxis_colorbar_tickfont_size attribute.
        
        Parameters
        ----------
        value : str
            Sets tick font size.

        """                
        
        self.__coloraxis_colorbar_tickfont_size = value     
        
    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKFONT_COLOR PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickfont_color(self):
        """Returns the coloraxis_colorbar_tickfont_color attribute.
        
        Sets tick font color.
        
        """

        return self.__coloraxis_colorbar_tickfont_color

    @coloraxis_colorbar_tickfont_color.setter
    def coloraxis_colorbar_tickfont_color(self, value):
        """Sets the coloraxis_colorbar_tickfont_color attribute.
        
        Parameters
        ----------
        value : str
            Sets tick font color.

        """                
        
        self.__coloraxis_colorbar_tickfont_color = value     

    # ----------------------------------------------------------------------- #
    #              COLORAXIS COLORBAR TICKANGLE PROPERTIES                    #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickangle(self):
        """Returns the coloraxis_colorbar_tickangle attribute.
        
        Sets tick angle.
        
        """

        return self.__coloraxis_colorbar_tickangle

    @coloraxis_colorbar_tickangle.setter
    def coloraxis_colorbar_tickangle(self, value):
        """Sets the coloraxis_colorbar_tickangle attribute.
        
        Parameters
        ----------
        value : str or int
            Sets tick angle.

        """                
        
        self.__coloraxis_colorbar_tickangle = value            
                

    # ----------------------------------------------------------------------- #
    #        COLORAXIS COLORBAR TICKFORMATSTOPS_ENABLED PROPERTIES            #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickformatstops_enabled(self):
        """Returns the coloraxis_colorbar_tickformatstops_enabled attribute.
        
        Determines whether or not this stop is used. If `False`, this stop 
        is ignored even within its `dtickrange`.
        
        """

        return self.__coloraxis_colorbar_tickformatstops_enabled

    @coloraxis_colorbar_tickformatstops_enabled.setter
    def coloraxis_colorbar_tickformatstops_enabled(self, value):
        """Sets the coloraxis_colorbar_tickformatstops_enabled attribute.
        
        Parameters
        ----------
        value : bool
            Determines whether or not this stop is used. If `False`, this stop 
            is ignored even within its `dtickrange`.

        """                
        
        self.__coloraxis_colorbar_tickformatstops_enabled = value                            

    # ----------------------------------------------------------------------- #
    #        COLORAXIS COLORBAR TICKFORMATSTOPS_DTICKRANGE PROPERTIES         #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickformatstops_dtickrange(self):
        """Returns the coloraxis_colorbar_tickformatstops_dtickrange attribute.
        
        range ["min", "max"], where "min", "max" - dtick values 
        which describe some zoom level, it is possible to omit 
        "min" or "max" value by passing "null"

        
        """

        return self.__coloraxis_colorbar_tickformatstops_dtickrange

    @coloraxis_colorbar_tickformatstops_dtickrange.setter
    def coloraxis_colorbar_tickformatstops_dtickrange(self, value):
        """Sets the coloraxis_colorbar_tickformatstops_dtickrange attribute.
        
        Parameters
        ----------
        value : list
            range ["min", "max"], where "min", "max" - dtick values 
            which describe some zoom level, it is possible to omit 
            "min" or "max" value by passing "null"

        """                
        
        self.__coloraxis_colorbar_tickformatstops_dtickrange = value           

    # ----------------------------------------------------------------------- #
    #        COLORAXIS COLORBAR TICKFORMATSTOPS_VALUE PROPERTIES              #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickformatstops_value(self):
        """Returns the coloraxis_colorbar_tickformatstops_value attribute.
        
         dtickformat for described zoom level, the same as "tickformat"
        
        """

        return self.__coloraxis_colorbar_tickformatstops_value

    @coloraxis_colorbar_tickformatstops_value.setter
    def coloraxis_colorbar_tickformatstops_value(self, value):
        """Sets the coloraxis_colorbar_tickformatstops_value attribute.
        
        Parameters
        ----------
        value : str
             dtickformat for described zoom level, the same as "tickformat"

        """                
        
        self.__coloraxis_colorbar_tickformatstops_value = value                   

    # ----------------------------------------------------------------------- #
    #        COLORAXIS COLORBAR TICKFORMATSTOPS_NAME PROPERTIES               #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickformatstops_name(self):
        """Returns the coloraxis_colorbar_tickformatstops_name attribute.
        
        When used in a template, named items are created in the 
        output figure in addition to any items the figure already 
        has in this array. 
        
        """

        return self.__coloraxis_colorbar_tickformatstops_name

    @coloraxis_colorbar_tickformatstops_name.setter
    def coloraxis_colorbar_tickformatstops_name(self, value):
        """Sets the coloraxis_colorbar_tickformatstops_name attribute.
        
        Parameters
        ----------
        value : str
             When used in a template, named items are created in the 
             output figure in addition to any items the figure already 
             has in this array. 

        """                
        
        self.__coloraxis_colorbar_tickformatstops_name = value                  

    # ----------------------------------------------------------------------- #
    #    COLORAXIS COLORBAR TICKFORMATSTOPS_TEMPLATEITEMNAME PROPERTIES       #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickformatstops_templateitemname(self):
        """Returns the coloraxis_colorbar_tickformatstops_templateitemname attribute.
        
        Used to refer to a named item in this array in the template. 
        
        """

        return self.__coloraxis_colorbar_tickformatstops_templateitemname

    @coloraxis_colorbar_tickformatstops_templateitemname.setter
    def coloraxis_colorbar_tickformatstops_templateitemname(self, value):
        """Sets the coloraxis_colorbar_tickformatstops_templateitemname attribute.
        
        Parameters
        ----------
        value : str
             Used to refer to a named item in this array in the template. 

        """                
        
        self.__coloraxis_colorbar_tickformatstops_templateitemname = value          

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR TICKPREFIX PROPERTIES                  #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_tickprefix(self):
        """Returns the coloraxis_colorbar_tickprefix attribute.
        
        Sets a tick label prefix 
        
        """

        return self.__coloraxis_colorbar_tickprefix

    @coloraxis_colorbar_tickprefix.setter
    def coloraxis_colorbar_tickprefix(self, value):
        """Sets the coloraxis_colorbar_tickprefix attribute.
        
        Parameters
        ----------
        value : str
             Sets a tick label prefix

        """                
        
        self.__coloraxis_colorbar_tickprefix = value            

    # ----------------------------------------------------------------------- #
    #               COLORAXIS COLORBAR SHOWTICKPREFIX PROPERTIES                  #
    # ----------------------------------------------------------------------- #
    @property
    def coloraxis_colorbar_showtickprefix(self):
        """Returns the coloraxis_colorbar_showtickprefix attribute.
        
        If "all", all tick labels are displayed with a prefix. If 
        "first", only the first tick is displayed with a prefix. 
        If "last", only the last tick is displayed with a suffix. 
        If "none", tick prefixes are hidden.
        
        """

        return self.__coloraxis_colorbar_showtickprefix

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
            self.__coloraxis_colorbar_showtickprefix = value
        else:
            raise ValueError("showtickprefix must be 'all', 'first', 'last'\
                , or 'none'.")