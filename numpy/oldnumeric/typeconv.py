__all__ = ['oldtype2dtype', 'convtypecode', 'convtypecode2', 'oldtypecodes']

import numpy as np

oldtype2dtype = {'1': np.dtype(np.byte),
                 's': np.dtype(np.short),
#                 'i': np.dtype(np.intc),
#                 'l': np.dtype(int),
#                 'b': np.dtype(np.ubyte),
                 'w': np.dtype(np.ushort),
                 'u': np.dtype(np.uintc),
#                 'f': np.dtype(np.single),
#                 'd': np.dtype(float),
#                 'F': np.dtype(np.csingle),
#                 'D': np.dtype(complex),
#                 'O': np.dtype(object),
#                 'c': np.dtype('c'),
                 None: np.dtype(int)
    }

# converts typecode=None to int
def convtypecode(typecode, dtype=None):
    if dtype is None:
        try:
            return oldtype2dtype[typecode]
        except:
            return np.dtype(typecode)
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
                return np.dtype(typecode)
    else:
        return dtype

_changedtypes = {'B': 'b',
                 'b': '1',
                 'h': 's',
                 'H': 'w',
                 'I': 'u'}

class _oldtypecodes(dict):
    def __getitem__(self, obj):
        char = np.dtype(obj).char
        try:
            return _changedtypes[char]
        except KeyError:
            return char


oldtypecodes = _oldtypecodes()
