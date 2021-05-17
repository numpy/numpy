from __future__ import annotations

from ._dtypes import (_boolean_dtypes, _floating_dtypes,
                      _integer_dtypes, _integer_or_boolean_dtypes,
                      _numeric_dtypes)
from ._array_object import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import array

import numpy as np

def abs(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.abs <numpy.abs>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in abs')
    return ndarray._new(np.abs(x._array))

# Note: the function name is different here
@np.errstate(all='ignore')
def acos(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arccos <numpy.arccos>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in acos')
    return ndarray._new(np.arccos(x._array))

# Note: the function name is different here
@np.errstate(all='ignore')
def acosh(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arccosh <numpy.arccosh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in acosh')
    return ndarray._new(np.arccosh(x._array))

@np.errstate(all='ignore')
def add(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.add <numpy.add>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in add')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.add(x1._array, x2._array))

# Note: the function name is different here
@np.errstate(all='ignore')
def asin(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arcsin <numpy.arcsin>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in asin')
    return ndarray._new(np.arcsin(x._array))

# Note: the function name is different here
@np.errstate(all='ignore')
def asinh(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arcsinh <numpy.arcsinh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in asinh')
    return ndarray._new(np.arcsinh(x._array))

# Note: the function name is different here
def atan(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arctan <numpy.arctan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in atan')
    return ndarray._new(np.arctan(x._array))

# Note: the function name is different here
def atan2(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arctan2 <numpy.arctan2>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in atan2')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.arctan2(x1._array, x2._array))

# Note: the function name is different here
@np.errstate(all='ignore')
def atanh(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arctanh <numpy.arctanh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in atanh')
    return ndarray._new(np.arctanh(x._array))

def bitwise_and(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_and <numpy.bitwise_and>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer_or_boolean dtypes are allowed in bitwise_and')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.bitwise_and(x1._array, x2._array))

# Note: the function name is different here
def bitwise_left_shift(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.left_shift <numpy.left_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError('Only integer dtypes are allowed in bitwise_left_shift')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    # Note: bitwise_left_shift is only defined for x2 nonnegative.
    if np.any(x2._array < 0):
        raise ValueError('bitwise_left_shift(x1, x2) is only defined for x2 >= 0')
    # Note: The spec requires the return dtype of bitwise_left_shift to be the
    # same as the first argument. np.left_shift() returns a type that is the
    # type promotion of the two input types.
    return ndarray._new(np.left_shift(x1._array, x2._array).astype(x1.dtype))

# Note: the function name is different here
def bitwise_invert(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.invert <numpy.invert>`.

    See its docstring for more information.
    """
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_invert')
    return ndarray._new(np.invert(x._array))

def bitwise_or(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_or <numpy.bitwise_or>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_or')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.bitwise_or(x1._array, x2._array))

# Note: the function name is different here
def bitwise_right_shift(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.right_shift <numpy.right_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError('Only integer dtypes are allowed in bitwise_right_shift')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    # Note: bitwise_right_shift is only defined for x2 nonnegative.
    if np.any(x2._array < 0):
        raise ValueError('bitwise_right_shift(x1, x2) is only defined for x2 >= 0')
    # Note: The spec requires the return dtype of bitwise_left_shift to be the
    # same as the first argument. np.left_shift() returns a type that is the
    # type promotion of the two input types.
    return ndarray._new(np.right_shift(x1._array, x2._array).astype(x1.dtype))

def bitwise_xor(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_xor <numpy.bitwise_xor>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError('Only integer or boolean dtypes are allowed in bitwise_xor')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.bitwise_xor(x1._array, x2._array))

def ceil(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.ceil <numpy.ceil>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in ceil')
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of ceil is the same as the input
        return x
    return ndarray._new(np.ceil(x._array))

@np.errstate(all='ignore')
def cos(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.cos <numpy.cos>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in cos')
    return ndarray._new(np.cos(x._array))

@np.errstate(all='ignore')
def cosh(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.cosh <numpy.cosh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in cosh')
    return ndarray._new(np.cosh(x._array))

@np.errstate(all='ignore')
def divide(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.divide <numpy.divide>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in divide')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.divide(x1._array, x2._array))

def equal(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.equal <numpy.equal>`.

    See its docstring for more information.
    """
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.equal(x1._array, x2._array))

@np.errstate(all='ignore')
def exp(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.exp <numpy.exp>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in exp')
    return ndarray._new(np.exp(x._array))

@np.errstate(all='ignore')
def expm1(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.expm1 <numpy.expm1>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in expm1')
    return ndarray._new(np.expm1(x._array))

def floor(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.floor <numpy.floor>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in floor')
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of floor is the same as the input
        return x
    return ndarray._new(np.floor(x._array))

@np.errstate(all='ignore')
def floor_divide(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.floor_divide <numpy.floor_divide>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in floor_divide')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.floor_divide(x1._array, x2._array))

def greater(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.greater <numpy.greater>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in greater')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.greater(x1._array, x2._array))

def greater_equal(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.greater_equal <numpy.greater_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in greater_equal')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.greater_equal(x1._array, x2._array))

def isfinite(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.isfinite <numpy.isfinite>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in isfinite')
    return ndarray._new(np.isfinite(x._array))

def isinf(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.isinf <numpy.isinf>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in isinf')
    return ndarray._new(np.isinf(x._array))

def isnan(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.isnan <numpy.isnan>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in isnan')
    return ndarray._new(np.isnan(x._array))

def less(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.less <numpy.less>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in less')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.less(x1._array, x2._array))

def less_equal(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.less_equal <numpy.less_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in less_equal')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.less_equal(x1._array, x2._array))

@np.errstate(all='ignore')
def log(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.log <numpy.log>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log')
    return ndarray._new(np.log(x._array))

@np.errstate(all='ignore')
def log1p(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.log1p <numpy.log1p>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log1p')
    return ndarray._new(np.log1p(x._array))

@np.errstate(all='ignore')
def log2(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.log2 <numpy.log2>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log2')
    return ndarray._new(np.log2(x._array))

@np.errstate(all='ignore')
def log10(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.log10 <numpy.log10>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in log10')
    return ndarray._new(np.log10(x._array))

def logaddexp(x1: array, x2: array) -> array:
    """
    Array API compatible wrapper for :py:func:`np.logaddexp <numpy.logaddexp>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in logaddexp')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.logaddexp(x1._array, x2._array))

def logical_and(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.logical_and <numpy.logical_and>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_and')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.logical_and(x1._array, x2._array))

def logical_not(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.logical_not <numpy.logical_not>`.

    See its docstring for more information.
    """
    if x.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_not')
    return ndarray._new(np.logical_not(x._array))

def logical_or(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.logical_or <numpy.logical_or>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_or')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.logical_or(x1._array, x2._array))

def logical_xor(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.logical_xor <numpy.logical_xor>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError('Only boolean dtypes are allowed in logical_xor')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.logical_xor(x1._array, x2._array))

@np.errstate(all='ignore')
def multiply(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.multiply <numpy.multiply>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in multiply')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.multiply(x1._array, x2._array))

def negative(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.negative <numpy.negative>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in negative')
    return ndarray._new(np.negative(x._array))

def not_equal(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.not_equal <numpy.not_equal>`.

    See its docstring for more information.
    """
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.not_equal(x1._array, x2._array))

def positive(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.positive <numpy.positive>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in positive')
    return ndarray._new(np.positive(x._array))

# Note: the function name is different here
@np.errstate(all='ignore')
def pow(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.power <numpy.power>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in pow')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.power(x1._array, x2._array))

@np.errstate(all='ignore')
def remainder(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.remainder <numpy.remainder>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in remainder')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.remainder(x1._array, x2._array))

def round(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.round <numpy.round>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in round')
    return ndarray._new(np.round(x._array))

def sign(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.sign <numpy.sign>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in sign')
    return ndarray._new(np.sign(x._array))

@np.errstate(all='ignore')
def sin(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.sin <numpy.sin>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in sin')
    return ndarray._new(np.sin(x._array))

@np.errstate(all='ignore')
def sinh(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.sinh <numpy.sinh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in sinh')
    return ndarray._new(np.sinh(x._array))

@np.errstate(all='ignore')
def square(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.square <numpy.square>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in square')
    return ndarray._new(np.square(x._array))

@np.errstate(all='ignore')
def sqrt(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.sqrt <numpy.sqrt>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in sqrt')
    return ndarray._new(np.sqrt(x._array))

@np.errstate(all='ignore')
def subtract(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.subtract <numpy.subtract>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in subtract')
    x1, x2 = ndarray._normalize_two_args(x1, x2)
    return ndarray._new(np.subtract(x1._array, x2._array))

@np.errstate(all='ignore')
def tan(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.tan <numpy.tan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in tan')
    return ndarray._new(np.tan(x._array))

def tanh(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.tanh <numpy.tanh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in tanh')
    return ndarray._new(np.tanh(x._array))

def trunc(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.trunc <numpy.trunc>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in trunc')
    return ndarray._new(np.trunc(x._array))
