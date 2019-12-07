# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \visual.py                                                            #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Friday December 6th 2019, 4:18:12 pm                           #
# Last Modified: Friday December 6th 2019, 4:18:28 pm                         #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #
"""Classes and functions to support visualization."""
import os
from textwrap import dedent

import dash_html_components as html
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output

def show_callbacks(app):

    def format_regs(registrations, padding=10):
        # TODO: -- switch to single line printing if > 79 chars                                                                                                                                
        vals = sorted("{}.{}".format(i['id'], i['property'])
                      for i in registrations)
        return ", ".join(vals)

    output_list = []

    for callback_id, callback in app.callback_map.items():
        wrapped_func = callback['callback'].__wrapped__
        inputs = callback['inputs']
        states = callback['state']
        #events = callback['events']

        str_values = {
            'callback': wrapped_func.__name__,
            'output': callback_id,
            'filename': os.path.split(wrapped_func.__code__.co_filename)[-1],
            'lineno': wrapped_func.__code__.co_firstlineno,
            'num_inputs': len(inputs),
            'num_states': len(states),
            #'num_events': len(events),
            'inputs': format_regs(inputs),
            'states': format_regs(states)
            #'events': format_regs(events)
        }

        output = """                                                                                                                                                                           
        callback      {callback} @ {filename}:{lineno}                                                                                                                                         
        Output        {output}                                                                                                                                                                 
        Inputs  {num_inputs:>4}  {inputs}                                                                                                                                                      
        States  {num_states:>4}  {states}                                                                                                                                                      
        
        """.format(**str_values)

        output_list.append(output)
    return "\n".join(output_list)


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Custom Display Components
def Card(children, **kwargs):
    return html.Section(
        children,
        style=_merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
        }, kwargs.get('style', {})),
        **_omit(['style'], kwargs)
    )


def FormattedSlider(**kwargs):
    return html.Div(
        style=kwargs.get('style', {}),
        children=dcc.Slider(**_omit(['style'], kwargs))
    )


def NamedSlider(name, **kwargs):
    return html.Div(
        style={'padding': '20px 10px 25px 4px'},
        children=[
            html.P(f'{name}:'),
            html.Div(
                style={'margin-left': '6px'},
                children=dcc.Slider(**kwargs)
            )
        ]
    )


def NamedDropdown(name, **kwargs):
    return html.Div(
        style={'margin': '10px 0px'},
        children=[
            html.P(
                children=f'{name}:',
                style={'margin-left': '3px'}
            ),

            dcc.Dropdown(**kwargs)
        ]
    )


def NamedRadioItems(name, **kwargs):
    return html.Div(
        style={'padding': '20px 10px 25px 4px'},
        children=[
            html.P(children=f'{name}:'),
            dcc.RadioItems(**kwargs)
        ]
    )


# Non-generic
def DemoDescription(filename, strip=False):
    with open(filename, 'r') as file:
        text = file.read()

    if strip:
        text = text.split('<Start Description>')[-1]
        text = text.split('<End Description>')[0]

    return html.Div(
            className='row',
            style={
                'padding': '15px 30px 27px',
                'margin': '45px auto 45px',
                'width': '80%',
                'max-width': '1024px',
                'borderRadius': 5,
                'border': 'thin lightgrey solid',
                'font-family': 'Roboto, sans-serif'
            },
            children=dcc.Markdown(dedent(text))
    )    