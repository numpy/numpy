# Array methods which are called by the both the C-code for the method
# and the Python code for the NumPy-namespace function

from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core.numeric import asanyarray

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
        axis = tuple(xrange(arr.ndim))
    if not isinstance(axis, tuple):
        axis = (axis,)
    items = 1
    for ax in axis:
        items *= arr.shape[ax]
    return items

def _mean(a, axis=None, dtype=None, out=None, keepdims=False):
    arr = asanyarray(a)

    # Upgrade bool, unsigned int, and int to float64
    if dtype is None and arr.dtype.kind in ['b','u','i']:
        ret = um.add.reduce(arr, axis=axis, dtype='f8',
                            out=out, keepdims=keepdims)
    else:
        ret = um.add.reduce(arr, axis=axis, dtype=dtype,
                            out=out, keepdims=keepdims)
    rcount = _count_reduce_items(arr, axis)
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(ret, rcount,
                        out=ret, casting='unsafe', subok=False)
    else:
        ret = ret / float(rcount)
    return ret

def _var(a, axis=None, dtype=None, out=None, ddof=0,
                            keepdims=False):
    arr = asanyarray(a)

    # First compute the mean, saving 'rcount' for reuse later
    if dtype is None and arr.dtype.kind in ['b','u','i']:
        arrmean = um.add.reduce(arr, axis=axis, dtype='f8', keepdims=True)
    else:
        arrmean = um.add.reduce(arr, axis=axis, dtype=dtype, keepdims=True)
    rcount = _count_reduce_items(arr, axis)
    if isinstance(arrmean, mu.ndarray):
        arrmean = um.true_divide(arrmean, rcount,
                            out=arrmean, casting='unsafe', subok=False)
    else:
        arrmean = arrmean / float(rcount)

    # arr - arrmean
    x = arr - arrmean

    # (arr - arrmean) ** 2
    if arr.dtype.kind == 'c':
        x = um.multiply(x, um.conjugate(x), out=x).real
    else:
        x = um.multiply(x, x, out=x)

    # add.reduce((arr - arrmean) ** 2, axis)
    ret = um.add.reduce(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    # add.reduce((arr - arrmean) ** 2, axis) / (n - ddof)
    if not keepdims and isinstance(rcount, mu.ndarray):
        rcount = rcount.squeeze(axis=axis)
    rcount -= ddof
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(ret, rcount,
                        out=ret, casting='unsafe', subok=False)
    else:
        ret = ret / float(rcount)

    return ret

def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    else:
        ret = um.sqrt(ret)

    return ret
