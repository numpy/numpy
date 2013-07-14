"""
Array methods which are called by the both the C-code for the method
and the Python code for the NumPy-namespace function

"""
from __future__ import division, absolute_import, print_function

import warnings

from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core.numeric import asanyarray
from numpy.core import numerictypes as nt

def _amax(a, axis=None, out=None, keepdims=False):
    return um.maximum.reduce(a, axis=axis,
                            out=out, keepdims=keepdims)

def _amin(a, axis=None, out=None, keepdims=False):
    return um.minimum.reduce(a, axis=axis,
                            out=out, keepdims=keepdims)

def _sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return um.add.reduce(a, axis=axis, dtype=dtype,
                            out=out, keepdims=keepdims)

def _prod(a, axis=None, dtype=None, out=None, keepdims=False):
    return um.multiply.reduce(a, axis=axis, dtype=dtype,
                            out=out, keepdims=keepdims)

def _any(a, axis=None, dtype=None, out=None, keepdims=False):
    return um.logical_or.reduce(a, axis=axis, dtype=dtype, out=out,
                                keepdims=keepdims)

def _all(a, axis=None, dtype=None, out=None, keepdims=False):
    return um.logical_and.reduce(a, axis=axis, dtype=dtype, out=out,
                                 keepdims=keepdims)

def _count_reduce_items(arr, axis):
    if axis is None:
        axis = tuple(range(arr.ndim))
    if not isinstance(axis, tuple):
        axis = (axis,)
    items = 1
    for ax in axis:
        items *= arr.shape[ax]
    return items

def _mean(a, axis=None, dtype=None, out=None, keepdims=False):
    arr = asanyarray(a)

    rcount = _count_reduce_items(arr, axis)
    # Make this warning show up first
    if rcount == 0:
        warnings.warn("Mean of empty slice.", RuntimeWarning)


    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
        dtype = mu.dtype('f8')

    ret = um.add.reduce(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
    else:
        ret = ret.dtype.type(ret / rcount)

    return ret

def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    arr = asanyarray(a)

    rcount = _count_reduce_items(arr, axis)
    # Make this warning show up on top.
    if ddof >= rcount:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
        dtype = mu.dtype('f8')

    # Compute the mean.
    # Note that if dtype is not of inexact type then arraymean will
    # not be either.
    arrmean = um.add.reduce(arr, axis=axis, dtype=dtype, keepdims=True)
    if isinstance(arrmean, mu.ndarray):
        arrmean = um.true_divide(
                arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
    else:
        arrmean = arrmean.dtype.type(arrmean / rcount)

    # Compute sum of squared deviations from mean
    # Note that x may not be inexact
    x = arr - arrmean
    if issubclass(arr.dtype.type, nt.complexfloating):
        x = um.multiply(x, um.conjugate(x), out=x).real
    else:
        x = um.multiply(x, x, out=x)
    ret = um.add.reduce(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    # Compute degrees of freedom and make sure it is not negative.
    rcount = max([rcount - ddof, 0])

    # divide by degrees of freedom
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
    else:
        ret = ret.dtype.type(ret / rcount)

    return ret

def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    else:
        ret = ret.dtype.type(um.sqrt(ret))

    return ret

