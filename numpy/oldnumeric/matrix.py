"""This module is for compatibility only.

"""
from __future__ import division, absolute_import, print_function

__all__ = ['UserArray', 'squeeze', 'Matrix', 'asarray', 'dot', 'k', 'Numeric', 'LinearAlgebra', 'identity', 'multiply', 'types', 'string']

import types
from .user_array import UserArray, asarray
import numpy.oldnumeric as Numeric
from numpy.oldnumeric import dot, identity, multiply
import numpy.oldnumeric.linear_algebra as LinearAlgebra
from numpy import matrix as Matrix, squeeze

# Hidden names that will be the same.

_table = [None]*256
for k in range(256):
    _table[k] = chr(k)
_table = ''.join(_table)

_numchars = '0123456789.-+jeEL'
_todelete = []
for k in _table:
    if k not in _numchars:
        _todelete.append(k)
_todelete = ''.join(_todelete)


def _eval(astr):
    return eval(astr.translate(_table, _todelete))

def _convert_from_string(data):
    data.find
    rows = data.split(';')
    newdata = []
    count = 0
    for row in rows:
        trow = row.split(',')
        newrow = []
        for col in trow:
            temp = col.split()
            newrow.extend(map(_eval, temp))
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError("Rows not the same size.")
        count += 1
        newdata.append(newrow)
    return newdata


_lkup = {'0':'000',
         '1':'001',
         '2':'010',
         '3':'011',
         '4':'100',
         '5':'101',
         '6':'110',
         '7':'111'}

def _binary(num):
    ostr = oct(num)
    bin = ''
    for ch in ostr[1:]:
        bin += _lkup[ch]
    ind = 0
    while bin[ind] == '0':
        ind += 1
    return bin[ind:]
