# =========================================================================== #
#                                 MISC                                        #
# =========================================================================== #
"""Miscellaneous utilities and functions."""
#%%
import re
import random
import string

def randomString(stringLength=5):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

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
    return s
