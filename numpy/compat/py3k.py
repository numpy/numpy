"""
Python 3 compatibility tools.

"""

__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar',
           'asunicode']

import sys

if sys.version_info[0] >= 3:
    import io
    bytes = bytes
    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('iso-8859-1')
    asunicode = str
    def isfileobj(f):
        return isinstance(f, io.IOBase)
    strchar = 'U'
else:
    bytes = str
    asbytes = str
    strchar = 'S'
    def isfileobj(f):
        return isinstance(f, file)
    def asunicode(s):
        if isinstance(s, unicode):
            return s
        return s.decode('iso-8859-1')

def getexception():
    return sys.exc_info()[1]

