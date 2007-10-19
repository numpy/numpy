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

def rsplit(s, sep, max = -1):
    """Equivalent of rsplit, but works on 2.3."""
    try:
        return s.rsplit(sep, max)
    except AttributeError:
        return _rsplit(s, sep, max)

def _rsplit(s, sep, max):
    """Equivalent of rsplit, but works on 2.3."""
    l = s.split(sep)
    if len(l) < 2 or max == 0:
        return [s]
    elif max < 0:
        return l[-len(l):]
    else:
        st = sep.join(l[0:-max])
        return [st] + l[-max:]

class curry:
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.pending = args[:]
        self.kwargs = kwargs.copy()

    def __call__(self, *args, **kwargs):
        if kwargs and self.kwargs:
            kw = self.kwargs.copy()
            kw.update(kwargs)
        else:
            kw = kwargs or self.kwargs

        return self.fun(*(self.pending + args), **kw)

if __name__ == '__main__':
    a1 = 'a.b.c'
    assert a1.split('.', -1) == a1.rsplit('.', -1) == _rsplit(a1, '.', -1)

    assert a1.rsplit('.', 1) == _rsplit(a1, '.', 1)

    assert a1.rsplit('.', 0) == _rsplit(a1, '.', 0)

    assert a1.rsplit('.', 2) == _rsplit(a1, '.', 2)

    a2 = 'floupi'
    assert a2.rsplit('.') ==  _rsplit(a2, '.', -1)
    assert a2.rsplit('.', 1) == _rsplit(a2, '.', 1)
