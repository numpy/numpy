#!/usr/bin/env python
"""Prints type-coercion tables for the built-in NumPy types"""

import numpy as np

# Generic object that can be added, but doesn't do anything else
class GenericObject(object):
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    dtype = np.dtype('O')

def print_cancast_table(ntypes):
    print 'X',
    for char in ntypes: print char,
    print
    for row in ntypes:
        print row,
        for col in ntypes:
            print int(np.can_cast(row, col)),
        print

def print_coercion_table(ntypes, inputfirstvalue, inputsecondvalue, firstarray, use_promote_types=False):
    print '+',
    for char in ntypes: print char,
    print
    for row in ntypes:
        if row == 'O':
            rowtype = GenericObject
        else:
            rowtype = np.obj2sctype(row)

        print row,
        for col in ntypes:
            if col == 'O':
                coltype = GenericObject
            else:
                coltype = np.obj2sctype(col)
            try:
                if firstarray:
                    rowvalue = np.array([rowtype(inputfirstvalue)], dtype=rowtype)
                else:
                    rowvalue = rowtype(inputfirstvalue)
                colvalue = coltype(inputsecondvalue)
                if use_promote_types:
                    char = np.promote_types(rowvalue.dtype, colvalue.dtype).char
                else:
                    value = np.add(rowvalue,colvalue)
                    if isinstance(value, np.ndarray):
                        char = value.dtype.char
                    else:
                        char = np.dtype(type(value)).char
            except ValueError:
                char = '!'
            except OverflowError:
                char = '@'
            except TypeError:
                char = '#'
            print char,
        print

print "can cast"
print_cancast_table(np.typecodes['All'])
print
print "In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'"
print
print "scalar + scalar"
print_coercion_table(np.typecodes['All'], 0, 0, False)
print
print "scalar + neg scalar"
print_coercion_table(np.typecodes['All'], 0, -1, False)
print
print "array + scalar"
print_coercion_table(np.typecodes['All'], 0, 0, True)
print
print "array + neg scalar"
print_coercion_table(np.typecodes['All'], 0, -1, True)
print
print "promote_types"
print_coercion_table(np.typecodes['All'], 0, 0, False, True)
