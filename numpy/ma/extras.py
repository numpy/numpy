"""Masked arrays add-ons.

A collection of utilities for maskedarray

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: extras.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

__all__ = ['apply_along_axis', 'atleast_1d', 'atleast_2d', 'atleast_3d',
           'average',
           'column_stack','compress_cols','compress_rowcols', 'compress_rows',
           'count_masked', 'corrcoef', 'cov',
           'diagflat', 'dot','dstack',
           'ediff1d','expand_dims',
           'flatnotmasked_contiguous','flatnotmasked_edges',
           'hsplit','hstack',
           'mask_cols','mask_rowcols','mask_rows','masked_all','masked_all_like',
           'median','mr_',
           'notmasked_contiguous','notmasked_edges',
           'polyfit',
           'row_stack',
           'vander','vstack',
           ]

from itertools import groupby
import warnings

import core as ma
from core import MaskedArray, MAError, add, array, asarray, concatenate, count,\
    filled, getmask, getmaskarray, masked, masked_array, mask_or, nomask, ones,\
    sort, zeros
#from core import *

import numpy as np
from numpy import ndarray, array as nxarray
import numpy.core.umath as umath
from numpy.lib.index_tricks import AxisConcatenator
from numpy.lib.polynomial import _lstsq

#...............................................................................
def issequence(seq):
    """Is seq a sequence (ndarray, list or tuple)?"""
    if isinstance(seq, ndarray):
        return True
    elif isinstance(seq, tuple):
        return True
    elif isinstance(seq, list):
        return True
    return False

def count_masked(arr, axis=None):
    """Count the number of masked elements along the given axis.

    Parameters
    ----------
        axis : int, optional
            Axis along which to count.
            If None (default), a flattened version of the array is used.

    """
    m = getmaskarray(arr)
    return m.sum(axis)

def masked_all(shape, dtype=float):
    """Return an empty masked array of the given shape and dtype,
    where all the data are masked.

    Parameters
    ----------
        dtype : dtype, optional
            Data type of the output.

    """
    a = masked_array(np.empty(shape, dtype),
                     mask=np.ones(shape, bool))
    return a

def masked_all_like(arr):
    """Return an empty masked array of the same shape and dtype as
    the array `a`, where all the data are masked.

    """
    a = masked_array(np.empty_like(arr),
                     mask=np.ones(arr.shape, bool))
    return a


#####--------------------------------------------------------------------------
#---- --- Standard functions ---
#####--------------------------------------------------------------------------
class _fromnxfunction:
    """Defines a wrapper to adapt numpy functions to masked arrays."""

    def __init__(self, funcname):
        self.__name__ = funcname
        self.__doc__ = self.getdoc()

    def getdoc(self):
        "Retrieves the __doc__ string from the function."
        inidoc = getattr(np, self.__name__).__doc__
        if inidoc:
            locdoc = "Notes\n-----\nThe function is applied to both the _data"\
                     " and the _mask, if any."
            return '\n'.join((inidoc, locdoc))
        return


    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        if len(args)==1:
            x = args[0]
            if isinstance(x, ndarray):
                _d = func(np.asarray(x), **params)
                _m = func(getmaskarray(x), **params)
                return masked_array(_d, mask=_m)
            elif isinstance(x, tuple) or isinstance(x, list):
                _d = func(tuple([np.asarray(a) for a in x]), **params)
                _m = func(tuple([getmaskarray(a) for a in x]), **params)
                return masked_array(_d, mask=_m)
        else:
            arrays = []
            args = list(args)
            while len(args)>0 and issequence(args[0]):
                arrays.append(args.pop(0))
            res = []
            for x in arrays:
                _d = func(np.asarray(x), *args, **params)
                _m = func(getmaskarray(x), *args, **params)
                res.append(masked_array(_d, mask=_m))
            return res

atleast_1d = _fromnxfunction('atleast_1d')
atleast_2d = _fromnxfunction('atleast_2d')
atleast_3d = _fromnxfunction('atleast_3d')

vstack = row_stack = _fromnxfunction('vstack')
hstack = _fromnxfunction('hstack')
column_stack = _fromnxfunction('column_stack')
dstack = _fromnxfunction('dstack')

hsplit = _fromnxfunction('hsplit')

diagflat = _fromnxfunction('diagflat')

def expand_dims(a, axis):
    """Expands the shape of a by including newaxis before axis.
    """
    if not isinstance(a, MaskedArray):
        return np.expand_dims(a, axis)
    elif getmask(a) is nomask:
        return np.expand_dims(a, axis).view(MaskedArray)
    m = getmaskarray(a)
    return masked_array(np.expand_dims(a, axis),
                        mask=np.expand_dims(m, axis))

#####--------------------------------------------------------------------------
#----
#####--------------------------------------------------------------------------
def flatten_inplace(seq):
    """Flatten a sequence in place."""
    k = 0
    while (k != len(seq)):
        while hasattr(seq[k],'__iter__'):
            seq[k:(k+1)] = seq[k]
        k += 1
    return seq


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Execute func1d(arr[i],*args) where func1d takes 1-D arrays and
    arr is an N-d array.  i varies so as to apply the function along
    the given axis for each 1-d subarray in arr.
    """
    arr = array(arr, copy=False, subok=True)
    nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis,nd))
    ind = [0]*(nd-1)
    i = np.zeros(nd,'O')
    indlist = range(nd)
    indlist.remove(axis)
    i[axis] = slice(None,None)
    outshape = np.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    j = i.copy()
    res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
    #  if res is a number, then we have a smaller output array
    asscalar = np.isscalar(res)
    if not asscalar:
        try:
            len(res)
        except TypeError:
            asscalar = True
    # Note: we shouldn't set the dtype of the output from the first result...
    #...so we force the type to object, and build a list of dtypes
    #...we'll just take the largest, to avoid some downcasting
    dtypes = []
    if asscalar:
        dtypes.append(np.asarray(res).dtype)
        outarr = zeros(outshape, object)
        outarr[tuple(ind)] = res
        Ntot = np.product(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(ind)] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    else:
        res = array(res, copy=False, subok=True)
        j = i.copy()
        j[axis] = ([slice(None, None)] * res.ndim)
        j.put(indlist, ind)
        Ntot = np.product(outshape)
        holdshape = outshape
        outshape = list(arr.shape)
        outshape[axis] = res.shape
        dtypes.append(asarray(res).dtype)
        outshape = flatten_inplace(outshape)
        outarr = zeros(outshape, object)
        outarr[tuple(flatten_inplace(j.tolist()))] = res
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            j.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(flatten_inplace(j.tolist()))] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    max_dtypes = np.dtype(np.asarray(dtypes).max())
    if not hasattr(arr, '_mask'):
        result = np.asarray(outarr, dtype=max_dtypes)
    else:
        result = asarray(outarr, dtype=max_dtypes)
        result.fill_value = ma.default_fill_value(result)
    return result

def average(a, axis=None, weights=None, returned=False):
    """Average the array over the given axis.

    Parameters
    ----------
    axis : {None,int}, optional
        Axis along which to perform the operation.
        If None, applies to a flattened version of the array.
    weights : {None, sequence}, optional
        Sequence of weights.
        The weights must have the shape of a, or be 1D with length
        the size of a along the given axis.
        If no weights are given, weights are assumed to be 1.
    returned : {False, True}, optional
        Flag indicating whether a tuple (result, sum of weights/counts)
        should be returned as output (True), or just the result (False).

    """
    a = asarray(a)
    mask = a.mask
    ash = a.shape
    if ash == ():
        ash = (1,)
    if axis is None:
        if mask is nomask:
            if weights is None:
                n = a.sum(axis=None)
                d = float(a.size)
            else:
                w = filled(weights, 0.0).ravel()
                n = umath.add.reduce(a._data.ravel() * w)
                d = umath.add.reduce(w)
                del w
        else:
            if weights is None:
                n = a.filled(0).sum(axis=None)
                d = umath.add.reduce((-mask).ravel().astype(int))
            else:
                w = array(filled(weights, 0.0), float, mask=mask).ravel()
                n = add.reduce(a.ravel() * w)
                d = add.reduce(w)
                del w
    else:
        if mask is nomask:
            if weights is None:
                d = ash[axis] * 1.0
                n = add.reduce(a._data, axis, dtype=float)
            else:
                w = filled(weights, 0.0)
                wsh = w.shape
                if wsh == ():
                    wsh = (1,)
                if wsh == ash:
                    w = np.array(w, float, copy=0)
                    n = add.reduce(a*w, axis)
                    d = add.reduce(w, axis)
                    del w
                elif wsh == (ash[axis],):
                    ni = ash[axis]
                    r = [None]*len(ash)
                    r[axis] = slice(None, None, 1)
                    w = eval ("w["+ repr(tuple(r)) + "] * ones(ash, float)")
                    n = add.reduce(a*w, axis, dtype=float)
                    d = add.reduce(w, axis, dtype=float)
                    del w, r
                else:
                    raise ValueError, 'average: weights wrong shape.'
        else:
            if weights is None:
                n = add.reduce(a, axis, dtype=float)
                d = umath.add.reduce((-mask), axis=axis, dtype=float)
            else:
                w = filled(weights, 0.0)
                wsh = w.shape
                if wsh == ():
                    wsh = (1,)
                if wsh == ash:
                    w = array(w, dtype=float, mask=mask, copy=0)
                    n = add.reduce(a*w, axis, dtype=float)
                    d = add.reduce(w, axis, dtype=float)
                elif wsh == (ash[axis],):
                    ni = ash[axis]
                    r = [None]*len(ash)
                    r[axis] = slice(None, None, 1)
                    w = eval ("w["+ repr(tuple(r)) + \
                              "] * masked_array(ones(ash, float), mask)")
                    n = add.reduce(a*w, axis, dtype=float)
                    d = add.reduce(w, axis, dtype=float)
                else:
                    raise ValueError, 'average: weights wrong shape.'
                del w
    if n is masked or d is masked:
        return masked
    result = n/d
    del n

    if isinstance(result, MaskedArray):
        if ((axis is None) or (axis==0 and a.ndim == 1)) and \
           (result.mask is nomask):
            result = result._data
        if returned:
            if not isinstance(d, MaskedArray):
                d = masked_array(d)
            if isinstance(d, ndarray) and (not d.shape == result.shape):
                d = ones(result.shape, dtype=float) * d
    if returned:
        return result, d
    else:
        return result



def median(a, axis=None, out=None, overwrite_input=False):
    """Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array
    axis : {None, int}, optional
        Axis along which the medians are computed. The default (axis=None) is to
        compute the median along a flattened version of the array.
    out : {None, ndarray}, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.
    overwrite_input : {False, True}, optional
       If True, then allow use of memory of input array (a) for
       calculations. The input array will be modified by the call to
       median. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. Note that, if overwrite_input is true, and the input
       is not already an ndarray, an error will be raised.

    Returns
    -------
    median : ndarray.
        A new array holding the result is returned unless out is
        specified, in which case a reference to out is returned.
        Return datatype is float64 for ints and floats smaller than
        float64, or the input datatype otherwise.

    See Also
    -------
    mean

    Notes
    -----
    Given a vector V with N non masked values, the median of V is the middle
    value of a sorted copy of V (Vs) - i.e. Vs[(N-1)/2], when N is odd, or
    {Vs[N/2 - 1] + Vs[N/2]}/2. when N is even.

    """
    def _median1D(data):
        counts = filled(count(data),0)
        (idx, rmd) = divmod(counts, 2)
        if rmd:
            choice = slice(idx, idx+1)
        else:
            choice = slice(idx-1, idx+1)
        return data[choice].mean(0)
    #
    if overwrite_input:
        if axis is None:
            asorted = a.ravel()
            asorted.sort()
        else:
            a.sort(axis=axis)
            asorted = a
    else:
        asorted = sort(a, axis=axis)
    if axis is None:
        result = _median1D(asorted)
    else:
        result = apply_along_axis(_median1D, axis, asorted)
    if out is not None:
        out = result
    return result




#..............................................................................
def compress_rowcols(x, axis=None):
    """Suppress the rows and/or columns of a 2D array that contains
    masked values.

    The suppression behavior is selected with the `axis`parameter.
        - If axis is None, rows and columns are suppressed.
        - If axis is 0, only rows are suppressed.
        - If axis is 1 or -1, only columns are suppressed.

    Parameters
    ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.

    Returns
    -------
        compressed_array : an ndarray.

    """
    x = asarray(x)
    if x.ndim != 2:
        raise NotImplementedError, "compress2d works for 2D arrays only."
    m = getmask(x)
    # Nothing is masked: return x
    if m is nomask or not m.any():
        return x._data
    # All is masked: return empty
    if m.all():
        return nxarray([])
    # Builds a list of rows/columns indices
    (idxr, idxc) = (range(len(x)), range(x.shape[1]))
    masked = m.nonzero()
    if not axis:
        for i in np.unique(masked[0]):
            idxr.remove(i)
    if axis in [None, 1, -1]:
        for j in np.unique(masked[1]):
            idxc.remove(j)
    return x._data[idxr][:,idxc]

def compress_rows(a):
    """Suppress whole rows of a 2D array that contain masked values.

    """
    return compress_rowcols(a, 0)

def compress_cols(a):
    """Suppress whole columnss of a 2D array that contain masked values.

    """
    return compress_rowcols(a, 1)

def mask_rowcols(a, axis=None):
    """Mask whole rows and/or columns of a 2D array that contain
    masked values.  The masking behavior is selected with the
    `axis`parameter.

        - If axis is None, rows and columns are masked.
        - If axis is 0, only rows are masked.
        - If axis is 1 or -1, only columns are masked.

    Parameters
    ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.

    Returns
    -------
         a *pure* ndarray.

    """
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError, "compress2d works for 2D arrays only."
    m = getmask(a)
    # Nothing is masked: return a
    if m is nomask or not m.any():
        return a
    maskedval = m.nonzero()
    a._mask = a._mask.copy()
    if not axis:
        a[np.unique(maskedval[0])] = masked
    if axis in [None, 1, -1]:
        a[:,np.unique(maskedval[1])] = masked
    return a

def mask_rows(a, axis=None):
    """Mask whole rows of a 2D array that contain masked values.

    Parameters
    ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
    """
    return mask_rowcols(a, 0)

def mask_cols(a, axis=None):
    """Mask whole columns of a 2D array that contain masked values.

    Parameters
    ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
    """
    return mask_rowcols(a, 1)


def dot(a,b, strict=False):
    """Return the dot product of two 2D masked arrays a and b.

    Like the generic numpy equivalent, the product sum is over the last
    dimension of a and the second-to-last dimension of b.  If strict is True,
    masked values are propagated: if a masked value appears in a row or column,
    the whole row or column is considered masked.

    Parameters
    ----------
    strict : {boolean}
        Whether masked data are propagated (True) or set to 0 for the computation.

    Notes
    -----
    The first argument is not conjugated.

    """
    #!!!: Works only with 2D arrays. There should be a way to get it to run with higher dimension
    if strict and (a.ndim == 2) and (b.ndim == 2):
        a = mask_rows(a)
        b = mask_cols(b)
    #
    d = np.dot(filled(a, 0), filled(b, 0))
    #
    am = (~getmaskarray(a))
    bm = (~getmaskarray(b))
    m = ~np.dot(am, bm)
    return masked_array(d, mask=m)

#...............................................................................
def ediff1d(array, to_end=None, to_begin=None):
    """Return the differences between consecutive elements of an
    array, possibly with prefixed and/or appended values.

    Parameters
    ----------
        array : {array}
            Input array,  will be flattened before the difference is taken.
        to_end : {number}, optional
            If provided, this number will be tacked onto the end of the returned
            differences.
        to_begin : {number}, optional
            If provided, this number will be taked onto the beginning of the
            returned differences.

    Returns
    -------
          ed : {array}
            The differences. Loosely, this will be (ary[1:] - ary[:-1]).

    """
    a = masked_array(array, copy=True)
    if a.ndim > 1:
        a.reshape((a.size,))
    (d, m, n) = (a._data, a._mask, a.size-1)
    dd = d[1:]-d[:-1]
    if m is nomask:
        dm = nomask
    else:
        dm = m[1:]-m[:-1]
    #
    if to_end is not None:
        to_end = asarray(to_end)
        nend = to_end.size
        if to_begin is not None:
            to_begin = asarray(to_begin)
            nbegin = to_begin.size
            r_data = np.empty((n+nend+nbegin,), dtype=a.dtype)
            r_mask = np.zeros((n+nend+nbegin,), dtype=bool)
            r_data[:nbegin] = to_begin._data
            r_mask[:nbegin] = to_begin._mask
            r_data[nbegin:-nend] = dd
            r_mask[nbegin:-nend] = dm
        else:
            r_data = np.empty((n+nend,), dtype=a.dtype)
            r_mask = np.zeros((n+nend,), dtype=bool)
            r_data[:-nend] = dd
            r_mask[:-nend] = dm
        r_data[-nend:] = to_end._data
        r_mask[-nend:] = to_end._mask
    #
    elif to_begin is not None:
        to_begin = asarray(to_begin)
        nbegin = to_begin.size
        r_data = np.empty((n+nbegin,), dtype=a.dtype)
        r_mask = np.zeros((n+nbegin,), dtype=bool)
        r_data[:nbegin] = to_begin._data
        r_mask[:nbegin] = to_begin._mask
        r_data[nbegin:] = dd
        r_mask[nbegin:] = dm
    #
    else:
        r_data = dd
        r_mask = dm
    return masked_array(r_data, mask=r_mask)


def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    """
    Private function for the computation of covariance and correlation
    coefficients.

    """
    x = ma.array(x, ndmin=2, copy=True, dtype=float)
    xmask = ma.getmaskarray(x)
    # Quick exit if we can't process masked data
    if not allow_masked and xmask.any():
        raise ValueError("Cannot process masked data...")
    #
    if x.shape[0] == 1:
        rowvar = True
    # Make sure that rowvar is either 0 or 1
    rowvar = int(bool(rowvar))
    axis = 1-rowvar
    if rowvar:
        tup = (slice(None), None)
    else:
        tup = (None, slice(None))
    #
    if y is None:
        xnotmask = np.logical_not(xmask).astype(int)
    else:
        y = array(y, copy=False, ndmin=2, dtype=float)
        ymask = ma.getmaskarray(y)
        if not allow_masked and ymask.any():
            raise ValueError("Cannot process masked data...")
        if xmask.any() or ymask.any():
            if y.shape == x.shape:
                # Define some common mask
                common_mask = np.logical_or(xmask, ymask)
                if common_mask is not nomask:
                    x.unshare_mask()
                    y.unshare_mask()
                    xmask = x._mask = y._mask = ymask = common_mask
        x = ma.concatenate((x,y),axis)
        xnotmask = np.logical_not(np.concatenate((xmask, ymask), axis)).astype(int)
    x -= x.mean(axis=rowvar)[tup]
    return (x, xnotmask, rowvar)


def cov(x, y=None, rowvar=True, bias=False, allow_masked=True):
    """Estimates the covariance matrix.

    Normalization is by (N-1) where N is the number of observations (unbiased
    estimate).  If bias is True then normalization is by N.

    By default, masked values are recognized as such. If x and y have the same
    shape, a common mask is allocated: if x[i,j] is masked, then y[i,j] will also
    be masked.
    Setting `allow_masked` to False will raise an exception if values are missing
    in either of the input arrays.

    Parameters
    ----------
    x : array-like
        Input data.
        If x is a 1D array, returns the variance.
        If x is a 2D array, returns the covariance matrix.
    y : {None, array-like}, optional
        Optional set of variables.
    rowvar : {False, True} optional
        If rowvar is true, then each row is a variable with observations in columns.
        If rowvar is False, each column is a variable and the observations are in
        the rows.
    bias : {False, True} optional
        Whether to use a biased (True) or unbiased (False) estimate of the covariance.
        If bias is True, then the normalization is by N, the number of observations.
        Otherwise, the normalization is by (N-1).
    allow_masked : {True, False} optional
        If True, masked values are propagated pair-wise: if a value is masked in x,
        the corresponding value is masked in y.
        If False, raises a ValueError exception when some values are missing.

    Raises
    ------
    ValueError:
        Raised if some values are missing and allow_masked is False.

    """
    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    if not rowvar:
        fact = np.dot(xnotmask.T, xnotmask)*1. - (1 - bool(bias))
        result = (dot(x.T, x.conj(), strict=False) / fact).squeeze()
    else:
        fact = np.dot(xnotmask, xnotmask.T)*1. - (1 - bool(bias))
        result = (dot(x, x.T.conj(), strict=False) / fact).squeeze()
    return result


def corrcoef(x, y=None, rowvar=True, bias=False, allow_masked=True):
    """The correlation coefficients formed from the array x, where the
    rows are the observations, and the columns are variables.

    corrcoef(x,y) where x and y are 1d arrays is the same as
    corrcoef(transpose([x,y]))

    Parameters
    ----------
    x : ndarray
        Input data. If x is a 1D array, returns the variance.
        If x is a 2D array, returns the covariance matrix.
    y : {None, ndarray} optional
        Optional set of variables.
    rowvar : {False, True} optional
        If True, then each row is a variable with observations in columns.
        If False, each column is a variable and the observations are in the rows.
    bias : {False, True} optional
        Whether to use a biased (True) or unbiased (False) estimate of the
        covariance.
        If True, then the normalization is by N, the number of non-missing
        observations.
        Otherwise, the normalization is by (N-1).
    allow_masked : {True, False} optional
        If True, masked values are propagated pair-wise: if a value is masked
        in x, the corresponding value is masked in y.
        If False, raises an exception.

    See Also
    --------
    cov

    """
    # Get the data
    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    # Compute the covariance matrix
    if not rowvar:
        fact = np.dot(xnotmask.T, xnotmask)*1. - (1 - bool(bias))
        c = (dot(x.T, x.conj(), strict=False) / fact).squeeze()
    else:
        fact = np.dot(xnotmask, xnotmask.T)*1. - (1 - bool(bias))
        c = (dot(x, x.T.conj(), strict=False) / fact).squeeze()
    # Check whether we have a scalar
    try:
        diag = ma.diagonal(c)
    except ValueError:
        return 1
    #
    if xnotmask.all():
        _denom = ma.sqrt(ma.multiply.outer(diag, diag))
    else:
        _denom = diagflat(diag)
        n = x.shape[1-rowvar]
        if rowvar:
            for i in range(n-1):
                for j in range(i+1,n):
                    _x = mask_cols(vstack((x[i], x[j]))).var(axis=1,
                                                             ddof=1-bias)
                    _denom[i,j] = _denom[j,i] = ma.sqrt(ma.multiply.reduce(_x))
        else:
            for i in range(n-1):
                for j in range(i+1,n):
                    _x = mask_cols(vstack((x[:,i], x[:,j]))).var(axis=1,
                                                                 ddof=1-bias)
                    _denom[i,j] = _denom[j,i] = ma.sqrt(ma.multiply.reduce(_x))
    return c/_denom

#####--------------------------------------------------------------------------
#---- --- Concatenation helpers ---
#####--------------------------------------------------------------------------

class MAxisConcatenator(AxisConcatenator):
    """Translate slice objects to concatenation along an axis.

    """

    def __init__(self, axis=0):
        AxisConcatenator.__init__(self, axis, matrix=False)

    def __getitem__(self,key):
        if isinstance(key, str):
            raise MAError, "Unavailable for masked array."
        if type(key) is not tuple:
            key = (key,)
        objs = []
        scalars = []
        final_dtypedescr = None
        for k in range(len(key)):
            scalar = False
            if type(key[k]) is slice:
                step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if type(step) is type(1j):
                    size = int(abs(step))
                    newobj = np.linspace(start, stop, num=size)
                else:
                    newobj = np.arange(start, stop, step)
            elif type(key[k]) is str:
                if (key[k] in 'rc'):
                    self.matrix = True
                    self.col = (key[k] == 'c')
                    continue
                try:
                    self.axis = int(key[k])
                    continue
                except (ValueError, TypeError):
                    raise ValueError, "Unknown special directive"
            elif type(key[k]) in np.ScalarType:
                newobj = asarray([key[k]])
                scalars.append(k)
                scalar = True
            else:
                newobj = key[k]
            objs.append(newobj)
            if isinstance(newobj, ndarray) and not scalar:
                if final_dtypedescr is None:
                    final_dtypedescr = newobj.dtype
                elif newobj.dtype > final_dtypedescr:
                    final_dtypedescr = newobj.dtype
        if final_dtypedescr is not None:
            for k in scalars:
                objs[k] = objs[k].astype(final_dtypedescr)
        res = concatenate(tuple(objs),axis=self.axis)
        return self._retval(res)

class mr_class(MAxisConcatenator):
    """Translate slice objects to concatenation along the first axis.

    For example:
        >>> np.ma.mr_[np.ma.array([1,2,3]), 0, 0, np.ma.array([4,5,6])]
        array([1, 2, 3, 0, 0, 4, 5, 6])

    """
    def __init__(self):
        MAxisConcatenator.__init__(self, 0)

mr_ = mr_class()

#####--------------------------------------------------------------------------
#---- Find unmasked data ---
#####--------------------------------------------------------------------------

def flatnotmasked_edges(a):
    """Find the indices of the first and last not masked values in a
    1D masked array.  If all values are masked, returns None.

    """
    m = getmask(a)
    if m is nomask or not np.any(m):
        return [0,-1]
    unmasked = np.flatnonzero(~m)
    if len(unmasked) > 0:
        return unmasked[[0,-1]]
    else:
        return None

def notmasked_edges(a, axis=None):
    """Find the indices of the first and last not masked values along
    the given axis in a masked array.

    If all values are masked, return None.  Otherwise, return a list
    of 2 tuples, corresponding to the indices of the first and last
    unmasked values respectively.

    Parameters
    ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
    """
    a = asarray(a)
    if axis is None or a.ndim == 1:
        return flatnotmasked_edges(a)
    m = getmask(a)
    idx = array(np.indices(a.shape), mask=np.asarray([m]*a.ndim))
    return [tuple([idx[i].min(axis).compressed() for i in range(a.ndim)]),
            tuple([idx[i].max(axis).compressed() for i in range(a.ndim)]),]

def flatnotmasked_contiguous(a):
    """Find contiguous unmasked data in a flattened masked array.

    Return a sorted sequence of slices (start index, end index).

    """
    m = getmask(a)
    if m is nomask:
        return (a.size, [0,-1])
    unmasked = np.flatnonzero(~m)
    if len(unmasked) == 0:
        return None
    result = []
    for k, group in groupby(enumerate(unmasked), lambda (i,x):i-x):
        tmp = np.array([g[1] for g in group], int)
#        result.append((tmp.size, tuple(tmp[[0,-1]])))
        result.append( slice(tmp[0], tmp[-1]) )
    result.sort()
    return result

def notmasked_contiguous(a, axis=None):
    """Find contiguous unmasked data in a masked array along the given
    axis.

    Parameters
    ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.

    Returns
    -------
        A sorted sequence of slices (start index, end index).

    Notes
    -----
        Only accepts 2D arrays at most.

    """
    a = asarray(a)
    nd = a.ndim
    if nd > 2:
        raise NotImplementedError,"Currently limited to atmost 2D array."
    if axis is None or nd == 1:
        return flatnotmasked_contiguous(a)
    #
    result = []
    #
    other = (axis+1)%2
    idx = [0,0]
    idx[axis] = slice(None, None)
    #
    for i in range(a.shape[other]):
        idx[other] = i
        result.append( flatnotmasked_contiguous(a[idx]) )
    return result


#####--------------------------------------------------------------------------
#---- Polynomial fit ---
#####--------------------------------------------------------------------------

def vander(x, n=None):
    """%s
    Notes
    -----
        Masked values in x will result in rows of zeros.
    """
    _vander = np.vander(x, n)
    m = getmask(x)
    if m is not nomask:
        _vander[m] = 0
    return _vander


def polyfit(x, y, deg, rcond=None, full=False):
    """
    Least squares polynomial fit.

    Do a best fit polynomial of degree 'deg' of 'x' to 'y'.  Return value is a
    vector of polynomial coefficients [pk ... p1 p0].  Eg, for ``deg = 2``::

        p2*x0^2 +  p1*x0 + p0 = y1
        p2*x1^2 +  p1*x1 + p0 = y1
        p2*x2^2 +  p1*x2 + p0 = y2
        .....
        p2*xk^2 +  p1*xk + p0 = yk

    Parameters
    ----------
    x : array_like
        1D vector of sample points.
    y : array_like
        1D vector or 2D array of values to fit. The values should run down the
        columns in the 2D case.
    deg : integer
        Degree of the fitting polynomial
    rcond: {None, float}, optional
        Relative condition number of the fit. Singular values smaller than this
        relative to the largest singular value will be ignored. The defaul value
        is len(x)*eps, where eps is the relative precision of the float type,
        about 2e-16 in most cases.
    full : {False, boolean}, optional
        Switch determining nature of return value. When it is False just the
        coefficients are returned, when True diagnostic information from the
        singular value decomposition is also returned.

    Returns
    -------
    coefficients, [residuals, rank, singular_values, rcond] : variable
        When full=False, only the coefficients are returned, running down the
        appropriate colume when y is a 2D array. When full=True, the rank of the
        scaled Vandermonde matrix, its effective rank in light of the rcond
        value, its singular values, and the specified value of rcond are also
        returned.

    Warns
    -----
    RankWarning : if rank is reduced and not full output
        The warnings can be turned off by:
        >>> import warnings
        >>> warnings.simplefilter('ignore',np.RankWarning)


    See Also
    --------
    polyval : computes polynomial values.

    Notes
    -----
    If X is a the Vandermonde Matrix computed from x (see
    http://mathworld.wolfram.com/VandermondeMatrix.html), then the
    polynomial least squares solution is given by the 'p' in

        X*p = y

    where X.shape is a matrix of dimensions (len(x), deg + 1), p is a vector of
    dimensions (deg + 1, 1), and y is a vector of dimensions (len(x), 1).

    This equation can be solved as

        p = (XT*X)^-1 * XT * y

    where XT is the transpose of X and -1 denotes the inverse. However, this
    method is susceptible to rounding errors and generally the singular value
    decomposition of the matrix X is preferred and that is what is done here.
    The singular value method takes a paramenter, 'rcond', which sets a limit on
    the relative size of the smallest singular value to be used in solving the
    equation. This may result in lowering the rank of the Vandermonde matrix, in
    which case a RankWarning is issued. If polyfit issues a RankWarning, try a
    fit of lower degree or replace x by x - x.mean(), both of which will
    generally improve the condition number. The routine already normalizes the
    vector x by its maximum absolute value to help in this regard. The rcond
    parameter can be set to a value smaller than its default, but the resulting
    fit may be spurious. The current default value of rcond is len(x)*eps, where
    eps is the relative precision of the floating type being used, generally
    around 1e-7 and 2e-16 for IEEE single and double precision respectively.
    This value of rcond is fairly conservative but works pretty well when x -
    x.mean() is used in place of x.


    DISCLAIMER: Power series fits are full of pitfalls for the unwary once the
    degree of the fit becomes large or the interval of sample points is badly
    centered. The problem is that the powers x**n are generally a poor basis for
    the polynomial functions on the sample interval, resulting in a Vandermonde
    matrix is ill conditioned and coefficients sensitive to rounding erros. The
    computation of the polynomial values will also sensitive to rounding errors.
    Consequently, the quality of the polynomial fit should be checked against
    the data whenever the condition number is large.  The quality of polynomial
    fits *can not* be taken for granted. If all you want to do is draw a smooth
    curve through the y values and polyfit is not doing the job, try centering
    the sample range or look into scipy.interpolate, which includes some nice
    spline fitting functions that may be of use.

    For more info, see
    http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html,
    but note that the k's and n's in the superscripts and subscripts
    on that page.  The linear algebra is correct, however.



    Notes
    -----
        Any masked values in x is propagated in y, and vice-versa.

    """
    order = int(deg) + 1
    x = asarray(x)
    mx = getmask(x)
    y = asarray(y)
    if y.ndim == 1:
        m = mask_or(mx, getmask(y))
    elif y.ndim == 2:
        y = mask_rows(y)
        my = getmask(y)
        if my is not nomask:
            m = mask_or(mx, my[:,0])
        else:
            m = mx
    else:
        raise TypeError,"Expected a 1D or 2D array for y!"
    if m is not nomask:
        x[m] = y[m] = masked
    # Set rcond
    if rcond is None :
        rcond = len(x)*np.finfo(x.dtype).eps
    # Scale x to improve condition number
    scale = abs(x).max()
    if scale != 0 :
        x = x / scale
    # solve least squares equation for powers of x
    v = vander(x, order)
    c, resids, rank, s = _lstsq(v, y.filled(0), rcond)
    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        warnings.warn("Polyfit may be poorly conditioned", np.RankWarning)
    # scale returned coefficients
    if scale != 0 :
        if c.ndim == 1 :
            c /= np.vander([scale], order)[0]
        else :
            c /= np.vander([scale], order).T
    if full :
        return c, resids, rank, s, rcond
    else :
        return c

################################################################################
