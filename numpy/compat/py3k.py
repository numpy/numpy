"""
Python 3 compatibility tools.

"""

__all__ = ['bytes', 'asbytes', 'isfileobj']

import sys
if sys.version_info[0] >= 3:
    import io
    bytes = bytes
    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('iso-8859-1')
    def isfileobj(f):
        return isinstance(f, io.IOBase)
else:
    bytes = str
    asbytes = str
    def isfileobj(f):
        return isinstance(f, file)
