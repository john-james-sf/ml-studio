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