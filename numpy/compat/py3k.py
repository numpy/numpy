"""
Python 3 compatibility tools.

"""

__all__ = ['bytes', 'asbytes', 'isfile']

import sys
if sys.version_info[0] >= 3:
    import io
    bytes = bytes
    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('iso-8859-1')
    def isfile(f):
        return isinstance(f, io.IOBase)
else:
    bytes = str
    asbytes = str
    def isfile(f):
        return isinstance(f, file)
