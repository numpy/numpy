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
           'count_masked',
           'dot','dstack',
           'expand_dims',
           'flatnotmasked_contiguous','flatnotmasked_edges', 
           'hsplit','hstack', 
           'mask_cols','mask_rowcols','mask_rows','masked_all','masked_all_like',
           'median','mediff1d','mr_', 
           'notmasked_contiguous','notmasked_edges', 
           'row_stack', 
           'vstack',
           ]

from itertools import groupby

import core
from core import *

import numpy
from numpy import float_
import numpy.core.umath as umath
import numpy.core.numeric as numeric
from numpy.core.numeric import ndarray
from numpy.core.numeric import array as nxarray
from numpy.core.fromnumeric import asarray as nxasarray

from numpy.lib.index_tricks import AxisConcatenator
import numpy.lib.function_base as function_base

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

def masked_all(shape, dtype=float_):
    """Return an empty masked array of the given shape and dtype,
    where all the data are masked.

    Parameters
    ----------
        dtype : dtype, optional
            Data type of the output.

    """
    a = masked_array(numeric.empty(shape, dtype),
                     mask=numeric.ones(shape, bool_))
    return a

def masked_all_like(arr):
    """Return an empty masked array of the same shape and dtype as
    the array `a`, where all the data are masked.

    """
    a = masked_array(numeric.empty_like(arr),
                     mask=numeric.ones(arr.shape, bool_))
    return a


#####--------------------------------------------------------------------------
#---- --- Standard functions ---
#####--------------------------------------------------------------------------
class _fromnxfunction:
    """Defines a wrapper to adapt numpy functions to masked arrays."""
    def __init__(self, funcname):
        self._function = funcname
        self.__doc__ = self.getdoc()
    def getdoc(self):
        "Retrieves the __doc__ string from the function."
        return getattr(numpy, self._function).__doc__ +\
            "*Notes*:\n    (The function is applied to both the _data and the _mask, if any.)"
    def __call__(self, *args, **params):
        func = getattr(numpy, self._function)
        if len(args)==1:
            x = args[0]
            if isinstance(x,ndarray):
                _d = func(nxasarray(x), **params)
                _m = func(getmaskarray(x), **params)
                return masked_array(_d, mask=_m)
            elif isinstance(x, tuple) or isinstance(x, list):
                _d = func(tuple([nxasarray(a) for a in x]), **params)
                _m = func(tuple([getmaskarray(a) for a in x]), **params)
                return masked_array(_d, mask=_m)
        else:
            arrays = []
            args = list(args)
            while len(args)>0 and issequence(args[0]):
                arrays.append(args.pop(0))
            res = []
            for x in arrays:
                _d = func(nxasarray(x), *args, **params)
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

def expand_dims(a, axis):
    """Expands the shape of a by including newaxis before axis.
    """
    if not isinstance(a, MaskedArray):
        return numpy.expand_dims(a,axis)
    elif getmask(a) is nomask:
        return numpy.expand_dims(a,axis).view(MaskedArray)
    m = getmaskarray(a)
    return masked_array(numpy.expand_dims(a,axis),
                        mask=numpy.expand_dims(m,axis))

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


def apply_along_axis(func1d,axis,arr,*args,**kwargs):
    """Execute func1d(arr[i],*args) where func1d takes 1-D arrays and
    arr is an N-d array.  i varies so as to apply the function along
    the given axis for each 1-d subarray in arr.
    """
    arr = core.array(arr, copy=False, subok=True)
    nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis,nd))
    ind = [0]*(nd-1)
    i = numeric.zeros(nd,'O')
    indlist = range(nd)
    indlist.remove(axis)
    i[axis] = slice(None,None)
    outshape = numeric.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    j = i.copy()
    res = func1d(arr[tuple(i.tolist())],*args,**kwargs)
    #  if res is a number, then we have a smaller output array
    asscalar = numeric.isscalar(res)
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
        dtypes.append(numeric.asarray(res).dtype)
        outarr = zeros(outshape, object_)
        outarr[tuple(ind)] = res
        Ntot = numeric.product(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist,ind)
            res = func1d(arr[tuple(i.tolist())],*args,**kwargs)
            outarr[tuple(ind)] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    else:
        res = core.array(res, copy=False, subok=True)
        j = i.copy()
        j[axis] = ([slice(None,None)] * res.ndim)
        j.put(indlist, ind)
        Ntot = numeric.product(outshape)
        holdshape = outshape
        outshape = list(arr.shape)
        outshape[axis] = res.shape
        dtypes.append(asarray(res).dtype)
        outshape = flatten_inplace(outshape)
        outarr = zeros(outshape, object_)
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
            res = func1d(arr[tuple(i.tolist())],*args,**kwargs)
            outarr[tuple(flatten_inplace(j.tolist()))] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    max_dtypes = numeric.dtype(numeric.asarray(dtypes).max())
    if not hasattr(arr, '_mask'):
        result = numeric.asarray(outarr, dtype=max_dtypes)
    else:
        result = core.asarray(outarr, dtype=max_dtypes)
        result.fill_value = core.default_fill_value(result)
    return result

def average(a, axis=None, weights=None, returned=False):
    """Average the array over the given axis.

    Parameters
    ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        weights : sequence, optional
            Sequence of weights.
            The weights must have the shape of a, or be 1D with length
            the size of a along the given axis.
            If no weights are given, weights are assumed to be 1.
        returned : bool
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
                d = umath.add.reduce((-mask).ravel().astype(int_))
            else:
                w = array(filled(weights, 0.0), float, mask=mask).ravel()
                n = add.reduce(a.ravel() * w)
                d = add.reduce(w)
                del w
    else:
        if mask is nomask:
            if weights is None:
                d = ash[axis] * 1.0
                n = add.reduce(a._data, axis, dtype=float_)
            else:
                w = filled(weights, 0.0)
                wsh = w.shape
                if wsh == ():
                    wsh = (1,)
                if wsh == ash:
                    w = numeric.array(w, float_, copy=0)
                    n = add.reduce(a*w, axis)
                    d = add.reduce(w, axis)
                    del w
                elif wsh == (ash[axis],):
                    ni = ash[axis]
                    r = [None]*len(ash)
                    r[axis] = slice(None, None, 1)
                    w = eval ("w["+ repr(tuple(r)) + "] * ones(ash, float)")
                    n = add.reduce(a*w, axis, dtype=float_)
                    d = add.reduce(w, axis, dtype=float_)
                    del w, r
                else:
                    raise ValueError, 'average: weights wrong shape.'
        else:
            if weights is None:
                n = add.reduce(a, axis, dtype=float_)
                d = umath.add.reduce((-mask), axis=axis, dtype=float_)
            else:
                w = filled(weights, 0.0)
                wsh = w.shape
                if wsh == ():
                    wsh = (1,)
                if wsh == ash:
                    w = array(w, dtype=float_, mask=mask, copy=0)
                    n = add.reduce(a*w, axis, dtype=float_)
                    d = add.reduce(w, axis, dtype=float_)
                elif wsh == (ash[axis],):
                    ni = ash[axis]
                    r = [None]*len(ash)
                    r[axis] = slice(None, None, 1)
                    w = eval ("w["+ repr(tuple(r)) + \
                              "] * masked_array(ones(ash, float), mask)")
                    n = add.reduce(a*w, axis, dtype=float_)
                    d = add.reduce(w, axis, dtype=float_)
                else:
                    raise ValueError, 'average: weights wrong shape.'
                del w
    if n is masked or d is masked:
        return masked
    result = n/d
    del n

    if isMaskedArray(result):
        if ((axis is None) or (axis==0 and a.ndim == 1)) and \
           (result.mask is nomask):
            result = result._data
        if returned:
            if not isMaskedArray(d):
                d = masked_array(d)
            if isinstance(d, ndarray) and (not d.shape == result.shape):
                d = ones(result.shape, dtype=float_) * d
    if returned:
        return result, d
    else:
        return result



def median(a, axis=0, out=None, overwrite_input=False):
    """Compute the median along the specified axis.

    Returns the median of the array elements.  The median is taken
    over the first axis of the array by default, otherwise over
    the specified axis.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array
    axis : {int, None}, optional
        Axis along which the medians are computed. The default is to
        compute the median along the first dimension.  axis=None
        returns the median of the flattened array

    out : ndarray, optional
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
    Given a vector V length N, the median of V is the middle value of
    a sorted copy of V (Vs) - i.e. Vs[(N-1)/2], when N is odd. It is
    the mean of the two middle values of Vs, when N is even.

    """
    def _median1D(data):
        counts = filled(count(data,axis),0)
        (idx,rmd) = divmod(counts, 2)
        if rmd:
            choice = slice(idx,idx+1)
        else:
            choice = slice(idx-1,idx+1)
        return data[choice].mean(0)
    # 
    if overwrite_input:
        if axis is None:
            sorted = a.ravel()
            sorted.sort()
        else:
            a.sort(axis=axis)
            sorted = a
    else:
        sorted = sort(a, axis=axis)
    if axis is None:
        result = _median1D(sorted)
    else:
        result = apply_along_axis(_median1D, axis, sorted)
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
        for i in function_base.unique(masked[0]):
            idxr.remove(i)
    if axis in [None, 1, -1]:
        for j in function_base.unique(masked[1]):
            idxc.remove(j)
    return x._data[idxr][:,idxc]

def compress_rows(a):
    """Suppress whole rows of a 2D array that contain masked values.

    """
    return compress_rowcols(a,0)

def compress_cols(a):
    """Suppress whole columnss of a 2D array that contain masked values.

    """
    return compress_rowcols(a,1)

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
        a[function_base.unique(maskedval[0])] = masked
    if axis in [None, 1, -1]:
        a[:,function_base.unique(maskedval[1])] = masked
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

    Like the generic numpy equivalent, the product sum is over the
    last dimension of a and the second-to-last dimension of b.  If
    strict is True, masked values are propagated: if a masked value
    appears in a row or column, the whole row or column is considered
    masked.

    Parameters
    ----------
        strict : {boolean}
            Whether masked data are propagated (True) or set to 0 for
            the computation.

    Notes
    -----
        The first argument is not conjugated.

    """
    #TODO: Works only with 2D arrays. There should be a way to get it to run with higher dimension
    if strict and (a.ndim == 2) and (b.ndim == 2):
        a = mask_rows(a)
        b = mask_cols(b)
    #
    d = numpy.dot(filled(a, 0), filled(b, 0))
    #
    am = (~getmaskarray(a))
    bm = (~getmaskarray(b))
    m = ~numpy.dot(am,bm)
    return masked_array(d, mask=m)

#...............................................................................
def mediff1d(array, to_end=None, to_begin=None):
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
            r_data = numeric.empty((n+nend+nbegin,), dtype=a.dtype)
            r_mask = numeric.zeros((n+nend+nbegin,), dtype=bool_)
            r_data[:nbegin] = to_begin._data
            r_mask[:nbegin] = to_begin._mask
            r_data[nbegin:-nend] = dd
            r_mask[nbegin:-nend] = dm
        else:
            r_data = numeric.empty((n+nend,), dtype=a.dtype)
            r_mask = numeric.zeros((n+nend,), dtype=bool_)
            r_data[:-nend] = dd
            r_mask[:-nend] = dm
        r_data[-nend:] = to_end._data
        r_mask[-nend:] = to_end._mask
    #
    elif to_begin is not None:
        to_begin = asarray(to_begin)
        nbegin = to_begin.size
        r_data = numeric.empty((n+nbegin,), dtype=a.dtype)
        r_mask = numeric.zeros((n+nbegin,), dtype=bool_)
        r_data[:nbegin] = to_begin._data
        r_mask[:nbegin] = to_begin._mask
        r_data[nbegin:] = dd
        r_mask[nbegin:] = dm
    #
    else:
        r_data = dd
        r_mask = dm
    return masked_array(r_data, mask=r_mask)




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
                    newobj = function_base.linspace(start, stop, num=size)
                else:
                    newobj = numeric.arange(start, stop, step)
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
            elif type(key[k]) in numeric.ScalarType:
                newobj = asarray([key[k]])
                scalars.append(k)
                scalar = True
            else:
                newobj = key[k]
            objs.append(newobj)
            if isinstance(newobj, numeric.ndarray) and not scalar:
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
        >>> mr_[array([1,2,3]), 0, 0, array([4,5,6])]
        array([1, 2, 3, 0, 0, 4, 5, 6])

    """
    def __init__(self):
        MAxisConcatenator.__init__(self, 0)

mr_ = mr_class()

#####--------------------------------------------------------------------------
#---- ---
#####--------------------------------------------------------------------------

def flatnotmasked_edges(a):
    """Find the indices of the first and last not masked values in a
    1D masked array.  If all values are masked, returns None.

    """
    m = getmask(a)
    if m is nomask or not numpy.any(m):
        return [0,-1]
    unmasked = numeric.flatnonzero(~m)
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
    idx = array(numpy.indices(a.shape), mask=nxasarray([m]*a.ndim))
    return [tuple([idx[i].min(axis).compressed() for i in range(a.ndim)]),
            tuple([idx[i].max(axis).compressed() for i in range(a.ndim)]),]

def flatnotmasked_contiguous(a):
    """Find contiguous unmasked data in a flattened masked array.

    Return a sorted sequence of slices (start index, end index).

    """
    m = getmask(a)
    if m is nomask:
        return (a.size, [0,-1])
    unmasked = numeric.flatnonzero(~m)
    if len(unmasked) == 0:
        return None
    result = []
    for k, group in groupby(enumerate(unmasked), lambda (i,x):i-x):
        tmp = numpy.array([g[1] for g in group], int_)
#        result.append((tmp.size, tuple(tmp[[0,-1]])))
        result.append( slice(tmp[0],tmp[-1]) )
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
        a sorted sequence of slices (start index, end index).

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
    idx[axis] = slice(None,None)
    #
    for i in range(a.shape[other]):
        idx[other] = i
        result.append( flatnotmasked_contiguous(a[idx]) )
    return result

################################################################################
