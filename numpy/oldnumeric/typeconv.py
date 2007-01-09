__all__ = ['oldtype2dtype', 'convtypecode', 'convtypecode2', 'oldtypecodes']

import numpy as N

oldtype2dtype = {'1': N.dtype(N.byte),
                 's': N.dtype(N.short),
#                 'i': N.dtype(N.intc),
#                 'l': N.dtype(int),
#                 'b': N.dtype(N.ubyte),
                 'w': N.dtype(N.ushort),
                 'u': N.dtype(N.uintc),
#                 'f': N.dtype(N.single),
#                 'd': N.dtype(float),
#                 'F': N.dtype(N.csingle),
#                 'D': N.dtype(complex),
#                 'O': N.dtype(object),
#                 'c': N.dtype('c'),
                 None:N.dtype(int)
    }

# converts typecode=None to int
def convtypecode(typecode, dtype=None):
    if dtype is None:
        try:
            return oldtype2dtype[typecode]
        except:
            return N.dtype(typecode)
    else:
        return dtype

#if both typecode and dtype are None
#  return None
def convtypecode2(typecode, dtype=None):
    if dtype is None:
        if typecode is None:
            return None
        else:
            try:
                return oldtype2dtype[typecode]
            except:
                return N.dtype(typecode)
    else:
        return dtype

_changedtypes = {'B': 'b',
                 'b': '1',
                 'h': 's',
                 'H': 'w',
                 'I': 'u'}

class _oldtypecodes(dict):
    def __getitem__(self, obj):
        char = N.dtype(obj).char
        try:
            return _changedtypes[char]
        except KeyError:
            return char


oldtypecodes = _oldtypecodes()
