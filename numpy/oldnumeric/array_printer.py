from __future__ import division, absolute_import, print_function

__all__ = ['array2string']

from numpy import array2string as _array2string

def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ',
                 array_output=0):
    if array_output:
        prefix="array("
        style=repr
    else:
        prefix = ""
        style=str
    return _array2string(a, max_line_width, precision,
                         suppress_small, separator, prefix, style)
