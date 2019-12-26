# =========================================================================== #
#                                 MISC                                        #
# =========================================================================== #
"""Miscellaneous utilities and functions."""
#%%
import re
import random
import string
import textwrap

def proper(s):
    """Strips then capitalizes each word in a string.""" 
    s = s.replace("-", " ").title()
    s = s.replace("_", " ").title()
    return s    

def snake(s):
    """Converts string to snake case suitable for filenames."""
    s = re.sub(r"[^a-zA-Z0-9._// ]+", '', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    s = s.replace(" ", "_")
    pattern = '_' + '{2,}'
    s = re.sub(pattern, '_', s)
    return s

def format_text(x):
    x = " ".join(x.split())
    formatted = textwrap.fill(textwrap.dedent(x))
    return formatted        


# %%
