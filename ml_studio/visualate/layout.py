#!/usr/bin/env python3
# =========================================================================== #
#                                  LAYOUT                                     #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \layout.py                                                            #
# Python Version: 3.7.3                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday December 17th 2019, 1:48:51 pm                         #
# Last Modified: Tuesday December 17th 2019, 1:49:22 pm                       #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Classes that create layout for a figure based upon CanvasComponents. 

The Layout module contains classes that configure the plotly layout object
based upon the parameters in the CanvasComponents classes.

The module is comprised of the following classes.

    * LayoutConFig : Configures plotly layout based upon ConfigComponent parameters
    * LayoutBase : Abstract base class for the following classes
    * LayoutTitle	:	Sets font and position of the plot title
    * LayoutLegend	:	Sets style, font, position and behavior of the legend
    * LayoutMargins	:	Sets plot margins
    * LayoutSize	:	Sets plot width and height
    * LayoutFont	:	Sets family, size and color of fonts
    * LayoutColorBackground	:	Sets plot and page background colors
    * LayoutColorScale	:	Sets sequential, divergent and colorway scales
    * LayoutColorAxisDomain	:	Sets min, max, and mid values of the color scales
    * LayoutColorAxisScales	:	Sets color scale
    * LayoutColorAxisBarStyle	:	Sets color axis bar thickness, length and color
    * LayoutColorAxisBarPosition	:	Sets the position of the color axis color bar
    * LayoutColorAxisBarBoundary	:	Sets color axis border and outline color and width
    * LayoutColorAxisBarTicks	:	Sets parameters for ticks
    * LayoutColorAxisBarTickStyle	:	Sets the style of the ticks 
    * LayoutColorAxisBarTickFont	:	Sets the font of the ticks.
    * LayoutColorAxisBarTickFormatStops	:	Sets tick format stop parameters
    * LayoutColorAxisBarNumbers	:	Set number format
    * LayoutColorAxisBarTitle	:	Sets the axis bar title family, size and color.

"""
from abc import ABC, abstractmethod, ABCMeta
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
#                            LayoutBase                                       #
# --------------------------------------------------------------------------- #
class LayoutBase(ABC):
    """Abstract base class for layout classes."""

    def __init__(self):
        pass

    @abstractmethod
    def update_layout(self, component, fig):
        pass

# --------------------------------------------------------------------------- #
#                            LayoutTitle                                      #
# --------------------------------------------------------------------------- #
class LayoutTitle(LayoutBase):
    """Configures title for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasTitle component parameters.
        
        Parameters
        ----------
        component : CanvasTitle 
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            title=dict(text=component.title_text,
                       font=dict(family=component.title_font_family,
                                 size=component.title_font_size,
                                 color=component.title_font_color),
                       xref=component.title_xref,
                       yref=component.title_yref,
                       x=component.title_x,
                       y=component.title_y,
                       xanchor=component.title_xanchor,
                       yanchor=component.title_yanchor,
                       pd=component.title_pad))

        return fig

# --------------------------------------------------------------------------- #
#                            LayoutLegend                                     #
# --------------------------------------------------------------------------- #
class LayoutLegend(LayoutBase):
    """Configures legend for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasLegend component parameters.
        
        Parameters
        ----------
        component : CanvasLegend 
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            showlegend=component.legend_show,
            legend=dict(bgcolor=component.legend_bgcolor,
                        bordercolor=component.legend_bordercolor,
                        borderwidth=component.legend_borderwidth,
                        font=dict(family=component.legend_font_family,
                                  size=component.legend_font_size,
                                  color=component.legend_font_color),
                        orientation=component.legend_orientation,
                        itemsizing=component.legend_itemsizing,
                        itemclick=component.legend_itemclick,
                        x=component.legend_x,
                        y=component.legend_y,
                        xanchor=component.legend_xanchor,
                        yanchor=component.legend_yanchor,
                        valign=component.legend_valign))

        return fig

# --------------------------------------------------------------------------- #
#                            LayoutLegend                                     #
# --------------------------------------------------------------------------- #
class LayoutMargins(LayoutBase):
    """Configures margins for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasMargins component parameters.
        
        Parameters
        ----------
        component : CanvasMargins 
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            margin=dict(l=component.margins_left,
                        t=component.margins_top,
                        b=component.margins_bottom,
                        pad=component.margins_pad)
        )

        return fig        

# --------------------------------------------------------------------------- #
#                             LayoutSize                                      #
# --------------------------------------------------------------------------- #
class LayoutSize(LayoutBase):
    """Configures size for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasSize component parameters.
        
        Parameters
        ----------
        component : CanvasSize
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            size=component.size_autosize,
            width=component.size_width,
            height=component.size_height
        )

        return fig                

# --------------------------------------------------------------------------- #
#                             LayoutFont                                      #
# --------------------------------------------------------------------------- #
class LayoutFont(LayoutBase):
    """Configures fonts for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasFont component parameters.
        
        Parameters
        ----------
        component : CanvasFont
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            font=dict(
                family=component.font_family,
                size=component.font_size,
                color=component.font_color),
            separators=component.font_separators
        )

        return fig              

# --------------------------------------------------------------------------- #
#                         LayoutColorBackground                               #
# --------------------------------------------------------------------------- #
class LayoutColorBackground(LayoutBase):
    """Configures background color for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorBackground component.
        
        Parameters
        ----------
        component : CanvasColorBackground
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            paper_bgcolor=component.paper_bgcolor,
            plot_bgcolor=component.plot_bgcolor
        )
        
        return fig             

# --------------------------------------------------------------------------- #
#                           LayoutColorScale                                  #
# --------------------------------------------------------------------------- #
class LayoutColorScale(LayoutBase):
    """Configures color scales for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorScale component.
        
        Parameters
        ----------
        component : CanvasColorScale
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            colorscale=dict(
                sequential=component.colorscale_sequential,
                sequentialminus=component.colorscale_sequentialminus,
                diverging=component.colorscale_diverging),
            colorway=component.colorway
        )

        return fig

# --------------------------------------------------------------------------- #
#                           LayoutColorAxisDomain                             #
# --------------------------------------------------------------------------- #
class LayoutColorAxisDomain(LayoutBase):
    """Configures color axis domains for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisDomain component.
        
        Parameters
        ----------
        component : CanvasColorAxisDomain
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                cauto=component.coloraxis_cauto,
                cmin=component.coloraxis_cmin,
                cmax=component.coloraxis_cmax,
                cmid=component.coloraxis_cmid)
        )

        return fig       

# --------------------------------------------------------------------------- #
#                           LayoutColorAxisScales                             #
# --------------------------------------------------------------------------- #
class LayoutColorAxisScales(LayoutBase):
    """Configures color axis scales for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisScales component.
        
        Parameters
        ----------
        component : CanvasColorAxisScales
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorscale=component.coloraxis_colorscale,
                autocolorscale=component.coloraxis_autoscale,
                reversescale=component.coloraxis_reversescale,
                showscale=component.coloraxis_showscale)
        )

        return fig                                

# --------------------------------------------------------------------------- #
#                         LayoutColorAxisBarStyle                             #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarStyle(LayoutBase):
    """Configures color axis bar style for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisBarStyle component.
        
        Parameters
        ----------
        component : CanvasColorAxisBarStyle
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    thicknessmode=component.coloraxis_colorbar_thicknessmode,
                    thickness=component.coloraxis_colorbar_thickness,
                    lenmode=component.coloraxis_colorbar_lenmode,
                    len=component.coloraxis_colorbar_len,
                    bgcolor=component.coloraxis_colorbar_bgcolor
                )
            )
        )

        return fig             

# --------------------------------------------------------------------------- #
#                         LayoutColorAxisBarStyle                             #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarPosition(LayoutBase):
    """Configures color axis bar position for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisBarPosition component.
        
        Parameters
        ----------
        component : CanvasColorAxisBarPosition
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    x=component.coloraxis_colorbar_x,
                    y=component.coloraxis_colorbar_y,
                    xanchor=component.coloraxis_colorbar_xanchor,
                    yanchor=component.coloraxis_colorbar_yanchor,
                    xpad=component.coloraxis_colorbar_xpad,
                    ypad=component.coloraxis_colorbar_ypad,
                )
            )
        )

        return fig             

# --------------------------------------------------------------------------- #
#                         LayoutColorAxisBarStyle                             #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarBoundary(LayoutBase):
    """Configures color axis bar boundaries for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisBarBoundary component.
        
        Parameters
        ----------
        component : CanvasColorAxisBarBoundary
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    outlinecolor=component.coloraxis_colorbar_outlinecolor,
                    outlinewidth=component.coloraxis_colorbar_outlinewidth,
                    bordercolor=component.coloraxis_colorbar_bordercolor,
                    borderwidth=component.coloraxis_colorbar_borderwidth
                )
            )
        )

        return fig                     

# --------------------------------------------------------------------------- #
#                         LayoutColorAxisBarTicks                             #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarTicks(LayoutBase):
    """Configures color axis bar ticks for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisBarTicks component.
        
        Parameters
        ----------
        component : CanvasColorAxisBarTicks
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    tickmode=component.coloraxis_colorbar_tickmode,
                    nticks=component.coloraxis_colorbar_nticks,
                    tick0=component.coloraxis_colorbar_tick0,
                    dtick=component.coloraxis_colorbar_dtick,
                    tickvals=component.coloraxis_colorbar_tickvals,
                    ticktext=component.coloraxis_colorbar_ticktext,
                    ticks=component.coloraxis_colorbar_ticks
                )
            )
        )

        return fig         

# --------------------------------------------------------------------------- #
#                      LayoutColorAxisBarTickStyle                            #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarTickStyle(LayoutBase):
    """Configures color axis bar tick styles for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisBarTickStyle component.
        
        Parameters
        ----------
        component : CanvasColorAxisBarTickStyle
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    ticklen=component.coloraxis_colorbar_ticklen,
                    tickwidth=component.coloraxis_colorbar_tickwidth,
                    tickcolor=component.coloraxis_colorbar_tickcolor,
                    showticklabels=component.coloraxis_colorbar_showticklabels,
                    tickangle=component.coloraxis_colorbar_tickangle,
                    tickprefix=component.coloraxis_colorbar_tickprefix,
                    showtickprefix=component.coloraxis_colorbar_showtickprefix,
                    ticksuffix=component.coloraxis_colorbar_ticksuffix,
                    showticksuffix=component.coloraxis_colorbar_showticksuffix                    
                )
            )
        )

        return fig              

# --------------------------------------------------------------------------- #
#                      LayoutColorAxisBarTickFont                             #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarTickFont(LayoutBase):
    """Configures color axis bar tick fonts for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisBarTickFont component.
        
        Parameters
        ----------
        component : CanvasColorAxisBarTickFont
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    tickfont=dict(
                        family= component.coloraxis_colorbar_tickfont_family,
                        size=component.coloraxis_colorbar_tickfont_size,
                        color=component.coloraxis_colorbar_tickfont_color
                    )
                )
            )
        )

        return fig         

# --------------------------------------------------------------------------- #
#                    LayoutColorAxisBarTickFormatStops                        #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarNumbers(LayoutBase):
    """Configures color axis bar number formats for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis ColorAxisBarNumbers component.
        
        Parameters
        ----------
        component : ColorAxisBarNumbers
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    separatethousands=component.coloraxis_colorbar_separatethousands,
                    exponentformat=component.coloraxis_colorbar_exponentformat,
                    showexponent=component.coloraxis_colorbar_showexponent
                    )
                )
            )        

        return fig           

# --------------------------------------------------------------------------- #
#                        LayoutColorAxisBarTitle                              #
# --------------------------------------------------------------------------- #
class LayoutColorAxisBarTitle(LayoutBase):
    """Configures color axis bar number formats for layout."""

    def __init__(self):
        pass

    def update_layout(self, component, fig):
        """Updates the figure layout basis CanvasColorAxisBarTitle component.
        
        Parameters
        ----------
        component : CanvasColorAxisBarTitle
            Component containing parameters

        fig : Plotly Figure
            The figure for which the layout is being updated.
        
        """ 

        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title=dict(
                        text=component.coloraxis_colorbar_title_text,
                        font=dict(
                            family=component.coloraxis_colorbar_title_font_family,
                            size=component.coloraxis_colorbar_title_font_size,
                            color=component.coloraxis_colorbar_title_font_color
                        )
                    )
                )
            )
        )        

        return fig             

# --------------------------------------------------------------------------- #
#                            LayoutConFig                                     #
# --------------------------------------------------------------------------- #
class LayoutConFig:
    """Configures a plotly layout based upon a Canvas object.

    Parameters
    ----------
    canvas : Canvas
        A Canvas object containing CanvasComponent objects.

    fig : Ploty Figure object
        Figure object to which the layout is configured.

    Returns
    -------
    fig : Plotly Figure Object
        Figure object with updated Layout

    """
    LAYOUTS = {
        "CanvasTitle" :  LayoutTitle(),
        "CanvasLegend" :  LayoutLegend(),
        "CanvasMargins" :  LayoutMargins(),
        "CanvasSize" :  LayoutSize(),
        "CanvasFont" :  LayoutFont(),
        "CanvasColorBackground" :  LayoutColorBackground(),
        "CanvasColorScale" :  LayoutColorScale(),
        "CanvasColorAxisDomain" :  LayoutColorAxisDomain(),
        "CanvasColorAxisScales" :  LayoutColorAxisScales(),
        "CanvasColorAxisBarStyle" :  LayoutColorAxisBarStyle(),
        "CanvasColorAxisBarPosition" :  LayoutColorAxisBarPosition(),
        "CanvasColorAxisBarBoundary" :  LayoutColorAxisBarBoundary(),
        "CanvasColorAxisBarTicks" :  LayoutColorAxisBarTicks(),
        "CanvasColorAxisBarTickStyle" :  LayoutColorAxisBarTickStyle(),
        "CanvasColorAxisBarTickFont" :  LayoutColorAxisBarTickFont(),
        "CanvasColorAxisBarNumbers" :  LayoutColorAxisBarNumbers(),
        "CanvasColorAxisBarTitle" :  LayoutColorAxisBarTitle()
    }

    def __init__(self, canvas, fig):
        self.__canvas = canvas
        self.__fig = fig

    def update_layout(self):
        for name, component in self.__canvas.items():
            self.__fig = self.LAYOUTS[name].update_layout(component, self.__fig)
        return self.__fig
