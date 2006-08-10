# Functions that already have the correct syntax or miscellaneous functions


__all__ = ['load', 'sort', 'copy_reg', 'clip', 'putmask', 'Unpickler', 'rank',
           'sign', 'shape', 'types', 'allclose', 'size',
           'choose', 'swapaxes', 'array_str',
           'pi', 'math', 'concatenate',
           'around', 'vdot', 'transpose', 'array2string', 'diagonal',
           'searchsorted', 'put', 'fromfunction', 'copy', 'resize',
           'array_repr', 'e', 'StringIO', 'pickle', 'average',
           'argsort', 'convolve', 'loads', 'cross_correlate',
           'Pickler', 'dot', 'outerproduct', 'innerproduct']

import types
import StringIO
import pickle
import math
import copy
import copy_reg
from pickle import load, loads

from numpy import sort, clip, putmask, rank, sign, shape, allclose, size,\
     choose, swapaxes, array_str, array_repr, e, pi, \
     fromfunction, resize, around, concatenate, vdot, transpose, \
     diagonal, searchsorted, put, average, argsort, convolve, dot, \
     outer as outerproduct, inner as innerproduct, correlate as cross_correlate

from array_printer import array2string


class Unpickler(pickle.Unpickler):
    def __init__(self, *args, **kwds):
        raise NotImplemented    
    def load_array(self):
        raise NotImplemented

class Pickler(pickle.Pickler):
    def __init__(self, *args, **kwds):
        raise NotImplemented
    def save_array(self, object):
        raise NotImplemented
