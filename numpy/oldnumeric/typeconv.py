
__all__ = ['oldtype2dtype', 'convtypecode', 'convtypecode2']

import numpy as N

oldtype2dtype = {'1': N.dtype(N.byte),
                 's': N.dtype(N.short),
                 'i': N.dtype(N.intc),
                 'l': N.dtype(int),
                 'b': N.dtype(N.ubyte),
                 'w': N.dtype(N.ushort),
                 'u': N.dtype(N.uintc),
                 'f': N.dtype(N.single),
                 'd': N.dtype(float),
                 'F': N.dtype(N.csingle),
                 'D': N.dtype(complex),
                 'O': N.dtype(object),
                 'c': N.dtype('c'),
                 None:N.dtype(int)
    }

def convtypecode(typecode, dtype=None):
    if dtype is None:
        try:
            return oldtype2dtype[typecode]
        except:
            return N.dtype(typecode)

def convtypecode2(typecode, dtype=None):
    if typecode is None and dtype is None:
        return None
    elif dtype is None:
        try:
            return oldtype2dtype[typecode]
        except:
            return N.dtype(typecode)
