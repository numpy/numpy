"""
Array methods which are called by both the C-code for the method
and the Python code for the NumPy-namespace function

"""
from __future__ import division, absolute_import, print_function

import warnings

from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core.numeric import asanyarray
from numpy.core import numerictypes as nt

# save those O(100) nanoseconds!
umr_maximum = um.maximum.reduce
umr_minimum = um.minimum.reduce
umr_sum = um.add.reduce
umr_prod = um.multiply.reduce
umr_any = um.logical_or.reduce
umr_all = um.logical_and.reduce

# avoid keyword arguments to speed up parsing, saves about 15%-20% for very
# small reductions
def _amax(a, axis=None, out=None, keepdims=False):
    return umr_maximum(a, axis, None, out, keepdims)

def _amin(a, axis=None, out=None, keepdims=False):
    return umr_minimum(a, axis, None, out, keepdims)

def _sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return umr_sum(a, axis, dtype, out, keepdims)

def _prod(a, axis=None, dtype=None, out=None, keepdims=False):
    return umr_prod(a, axis, dtype, out, keepdims)

def _any(a, axis=None, dtype=None, out=None, keepdims=False):
    return umr_any(a, axis, dtype, out, keepdims)

def _all(a, axis=None, dtype=None, out=None, keepdims=False):
    return umr_all(a, axis, dtype, out, keepdims)

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

    ret = umr_sum(arr, axis, dtype, out, keepdims)
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret

def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
         fweights=None, aweights=None):
    from numpy.lib.function_base import _product_fweights_aweights
    arr = asanyarray(a)

    rcount = _count_reduce_items(arr, axis)
    w, aweights = _product_fweights_aweights(fweights, aweights, rcount)

    if w is None:
        n_items = rcount
    else:
        n_items = umr_sum(w)

    # Compute normalization factor and make if it is negative, floor to 0
    if aweights is not None:
        waweights = umr_sum(um.multiply(w, aweights))
        nf = (n_items ** 2- ddof * waweights) / float(n_items)
    else:
        nf = n_items - ddof
    if nf <= 0:
        nf = 0
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
        dtype = mu.dtype('f8')

    if w is not None and axis == 0:
        w = w[:, None]

    # Compute the mean.
    # Note that if dtype is not of inexact type then arraymean will
    # not be either.
    if w is None:
        arrmean = umr_sum(arr, axis, dtype, keepdims=True)
    else:
        arrmean = umr_sum(w * arr, axis, dtype, keepdims=True)

    if isinstance(arrmean, mu.ndarray):
        arrmean = um.true_divide(
                arrmean, n_items, out=arrmean, casting='unsafe', subok=False)
    else:
        arrmean = arrmean.dtype.type(arrmean / n_items)

    # Compute sum of squared deviations from mean
    # Note that x may not be inexact and that we need it to be an array,
    # not a scalar.
    x = asanyarray(arr - arrmean)

    if issubclass(arr.dtype.type, nt.complexfloating):
        x = um.multiply(x, um.conjugate(x), out=x).real
    else:
        x = um.multiply(x, x, out=x)

    if w is not None:
        x = um.multiply(w, x, out=x)
    ret = umr_sum(x, axis, dtype, out, keepdims)

    # divide by degrees of freedom
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, nf, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / nf)
    else:
        ret = ret / nf

    return ret

def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)

    return ret
