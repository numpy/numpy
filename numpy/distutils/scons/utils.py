import os
import re

from copy import deepcopy

_START_WITH_MINUS = re.compile('^\s*-')

def popen_wrapper(cmd, merge = False):
    """This works like popen, but it returns both the status and the output.

    If merge is True, then output contains both stdout and stderr. 
    
    Returns: (st, out), where st is an integer (the return status of the
    subprocess, and out the string of the output).
    
    NOTE:
        - it tries to be robust to find non existing command. For example, is
          cmd starts with a minus, a nonzero status is returned, and no junk is
          displayed on the interpreter stdout."""
    if _START_WITH_MINUS.match(cmd):
        return 1, ''
    if merge:
        # XXX: I have a bad feeling about this. Does it really work reliably on
        # all supported platforms ?
        cmd += ' 2>& 1 '
    output = os.popen(cmd)
    out = output.read()
    st = output.close()
    if st:
        status = st
    else:
        status = 0

    return status, out

def get_empty(dict, key):
    """Assuming dict is a dictionary with lists as values, returns an empty
    list if key is not found, or a (deep) copy of the existing value if it
    does."""
    try:
        return deepcopy(dict[key])
    except KeyError, e:
        return []

