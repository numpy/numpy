# Functions that already have the correct syntax or miscellaneous functions


__all__ = ['load', 'sort', 'copy_reg', 'clip', 'putmask', 'Unpickler', 'rank',
           'sign', 'shape', 'types', 'allclose', 'size',
           'argmax', 'choose', 'swapaxes', 'array_str',
           'pi', 'math', 'compress', 'concatenate',
           'around', 'vdot', 'transpose', 'array2string', 'diagonal',
           'searchsorted', 'put', 'fromfunction', 'copy', 'resize',
           'array_repr', 'e', 'argmin', 'StringIO', 'pickle', 'average',
           'argsort', 'convolve', 'loads',
           'Pickler', 'dot']

import types
import StringIO
import pickle
import math
import copy
import copy_reg
from pickle import load, loads

from numpy import sort, clip, putmask, rank, sign, shape, allclose, size,\
     argmax, choose, swapaxes, array_str, array_repr, argmin, e, pi, \
     fromfunction, resize, around, compress, concatenate, vdot, transpose, \
     diagonal, searchsorted, put, average, argsort, convolve, dot

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
