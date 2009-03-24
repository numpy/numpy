# pylint: disable-msg=E1002
"""
numpy.ma : a package to handle missing or invalid values.

This package was initially written for numarray by Paul F. Dubois
at Lawrence Livermore National Laboratory. 
In 2006, the package was completely rewritten by Pierre Gerard-Marchant
(University of Georgia) to make the MaskedArray class a subclass of ndarray,
and to improve support of structured arrays.


Copyright 1999, 2000, 2001 Regents of the University of California.
Released for unlimited redistribution.
* Adapted for numpy_core 2005 by Travis Oliphant and (mainly) Paul Dubois.
* Subclassing of the base ndarray 2006 by Pierre Gerard-Marchant 
  (pgmdevlist_AT_gmail_DOT_com)
* Improvements suggested by Reggie Dugard (reggie_AT_merfinllc_DOT_com)

.. moduleauthor:: Pierre Gerard-Marchant


"""
__author__ = "Pierre GF Gerard-Marchant"
__docformat__ = "restructuredtext en"

__all__ = ['MAError', 'MaskError', 'MaskType', 'MaskedArray',
           'bool_',
           'abs', 'absolute', 'add', 'all', 'allclose', 'allequal', 'alltrue',
           'amax', 'amin', 'anom', 'anomalies', 'any', 'arange',
           'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2',
           'arctanh', 'argmax', 'argmin', 'argsort', 'around',
           'array', 'asarray','asanyarray',
           'bitwise_and', 'bitwise_or', 'bitwise_xor',
           'ceil', 'choose', 'clip', 'common_fill_value', 'compress',
           'compressed', 'concatenate', 'conjugate', 'copy', 'cos', 'cosh',
           'count', 'cumprod', 'cumsum',
           'default_fill_value', 'diag', 'diagonal', 'divide', 'dump', 'dumps',
           'empty', 'empty_like', 'equal', 'exp', 'expand_dims',
           'fabs', 'flatten_mask', 'fmod', 'filled', 'floor', 'floor_divide',
           'fix_invalid', 'flatten_structured_array', 'frombuffer', 'fromflex',
           'fromfunction',
           'getdata','getmask', 'getmaskarray', 'greater', 'greater_equal',
           'harden_mask', 'hypot',
           'identity', 'ids', 'indices', 'inner', 'innerproduct',
           'isMA', 'isMaskedArray', 'is_mask', 'is_masked', 'isarray',
           'left_shift', 'less', 'less_equal', 'load', 'loads', 'log', 'log10',
           'logical_and', 'logical_not', 'logical_or', 'logical_xor',
           'make_mask', 'make_mask_descr', 'make_mask_none', 'mask_or',
           'masked', 'masked_array', 'masked_equal', 'masked_greater',
           'masked_greater_equal', 'masked_inside', 'masked_invalid',
           'masked_less','masked_less_equal', 'masked_not_equal',
           'masked_object','masked_outside', 'masked_print_option',
           'masked_singleton','masked_values', 'masked_where', 'max', 'maximum',
           'maximum_fill_value', 'mean', 'min', 'minimum', 'minimum_fill_value',
           'mod', 'multiply',
           'negative', 'nomask', 'nonzero', 'not_equal',
           'ones', 'outer', 'outerproduct',
           'power', 'prod', 'product', 'ptp', 'put', 'putmask',
           'rank', 'ravel', 'remainder', 'repeat', 'reshape', 'resize',
           'right_shift', 'round_', 'round',
           'set_fill_value', 'shape', 'sin', 'sinh', 'size', 'sometrue',
           'sort', 'soften_mask', 'sqrt', 'squeeze', 'std', 'subtract', 'sum',
           'swapaxes',
           'take', 'tan', 'tanh', 'trace', 'transpose', 'true_divide',
           'var', 'where',
           'zeros']

import cPickle
import operator

import numpy as np
from numpy import ndarray, amax, amin, iscomplexobj, bool_
from numpy import array as narray

import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy import expand_dims as n_expand_dims
import warnings


MaskType = np.bool_
nomask = MaskType(0)

np.seterr(all='ignore')



def doc_note(initialdoc, note):
    """
    Adds a Notes section to an existing docstring.
    """
    if initialdoc is None:
        return
    if note is None:
        return initialdoc
    newdoc = """
    %s

    Notes
    -----
    %s
    """
    return newdoc % (initialdoc, note)

def get_object_signature(obj):
    """
    Get the signature from obj
    """
    import inspect
    try:
        sig = inspect.formatargspec(*inspect.getargspec(obj))
    except TypeError, errmsg:
        msg = "Unable to retrieve the signature of %s '%s'\n"\
              "(Initial error message: %s)"
#        warnings.warn(msg % (type(obj),
#                             getattr(obj, '__name__', '???'),
#                             errmsg))
        sig = ''
    return sig

#####--------------------------------------------------------------------------
#---- --- Exceptions ---
#####--------------------------------------------------------------------------
class MAError(Exception):
    "Class for MA related errors."
    pass
class MaskError(MAError):
    "Class for mask related errors."
    pass


#####--------------------------------------------------------------------------
#---- --- Filling options ---
#####--------------------------------------------------------------------------
# b: boolean - c: complex - f: floats - i: integer - O: object - S: string
default_filler = {'b': True,
                  'c' : 1.e20 + 0.0j,
                  'f' : 1.e20,
                  'i' : 999999,
                  'O' : '?',
                  'S' : 'N/A',
                  'u' : 999999,
                  'V' : '???',
                  }
max_filler = ntypes._minvals
max_filler.update([(k, -np.inf) for k in [np.float32, np.float64]])
min_filler = ntypes._maxvals
min_filler.update([(k, +np.inf) for k in [np.float32, np.float64]])
if 'float128' in ntypes.typeDict:
    max_filler.update([(np.float128, -np.inf)])
    min_filler.update([(np.float128, +np.inf)])


def default_fill_value(obj):
    """
    Calculate the default fill value for the argument object.

    """
    if hasattr(obj,'dtype'):
        defval = _check_fill_value(None, obj.dtype)
    elif isinstance(obj, np.dtype):
        if obj.subdtype:
            defval = default_filler[obj.subdtype[0].kind]
        else:
            defval = default_filler[obj.kind]
    elif isinstance(obj, float):
        defval = default_filler['f']
    elif isinstance(obj, int) or isinstance(obj, long):
        defval = default_filler['i']
    elif isinstance(obj, str):
        defval = default_filler['S']
    elif isinstance(obj, complex):
        defval = default_filler['c']
    else:
        defval = default_filler['O']
    return defval


def _recursive_extremum_fill_value(ndtype, extremum):
    names = ndtype.names
    if names:
        deflist = []
        for name in names:
            fval = _recursive_extremum_fill_value(ndtype[name], extremum)
            deflist.append(fval)
        return tuple(deflist)
    return extremum[ndtype]


def minimum_fill_value(obj):
    """
    Calculate the default fill value suitable for taking the minimum of ``obj``.

    """
    errmsg = "Unsuitable type for calculating minimum."
    if hasattr(obj, 'dtype'):
        return _recursive_extremum_fill_value(obj.dtype, min_filler)
    elif isinstance(obj, float):
        return min_filler[ntypes.typeDict['float_']]
    elif isinstance(obj, int):
        return min_filler[ntypes.typeDict['int_']]
    elif isinstance(obj, long):
        return min_filler[ntypes.typeDict['uint']]
    elif isinstance(obj, np.dtype):
        return min_filler[obj]
    else:
        raise TypeError(errmsg)


def maximum_fill_value(obj):
    """
    Calculate the default fill value suitable for taking the maximum of ``obj``.

    """
    errmsg = "Unsuitable type for calculating maximum."
    if hasattr(obj, 'dtype'):
        return _recursive_extremum_fill_value(obj.dtype, max_filler)
    elif isinstance(obj, float):
        return max_filler[ntypes.typeDict['float_']]
    elif isinstance(obj, int):
        return max_filler[ntypes.typeDict['int_']]
    elif isinstance(obj, long):
        return max_filler[ntypes.typeDict['uint']]
    elif isinstance(obj, np.dtype):
        return max_filler[obj]
    else:
        raise TypeError(errmsg)


def _recursive_set_default_fill_value(dtypedescr):
    deflist = []
    for currentdescr in dtypedescr:
        currenttype = currentdescr[1]
        if isinstance(currenttype, list):
            deflist.append(tuple(_recursive_set_default_fill_value(currenttype)))
        else:
            deflist.append(default_fill_value(np.dtype(currenttype)))
    return tuple(deflist)

def _recursive_set_fill_value(fillvalue, dtypedescr):
    fillvalue = np.resize(fillvalue, len(dtypedescr))
    output_value = []
    for (fval, descr) in zip(fillvalue, dtypedescr):
        cdtype = descr[1]
        if isinstance(cdtype, list):
            output_value.append(tuple(_recursive_set_fill_value(fval, cdtype)))
        else:
            output_value.append(np.array(fval, dtype=cdtype).item())
    return tuple(output_value)


def _check_fill_value(fill_value, ndtype):
    """
    Private function validating the given `fill_value` for the given dtype.

    If fill_value is None, it is set to the default corresponding to the dtype
    if this latter is standard (no fields). If the datatype is flexible (named
    fields), fill_value is set to a tuple whose elements are the default fill
    values corresponding to each field.

    If fill_value is not None, its value is forced to the given dtype.

    """
    ndtype = np.dtype(ndtype)
    fields = ndtype.fields
    if fill_value is None:
        if fields:
            descr = ndtype.descr
            fill_value = np.array(_recursive_set_default_fill_value(descr),
                                  dtype=ndtype,)
        else:
            fill_value = default_fill_value(ndtype)
    elif fields:
        fdtype = [(_[0], _[1]) for _ in ndtype.descr]
        if isinstance(fill_value, ndarray):
            try:
                fill_value = np.array(fill_value, copy=False, dtype=fdtype)
            except ValueError:
                err_msg = "Unable to transform %s to dtype %s"
                raise ValueError(err_msg % (fill_value, fdtype))
        else:
            descr = ndtype.descr
            fill_value = np.array(_recursive_set_fill_value(fill_value, descr),
                                  dtype=ndtype)
    else:
        if isinstance(fill_value, basestring) and (ndtype.char not in 'SV'):
            fill_value = default_fill_value(ndtype)
        else:
            # In case we want to convert 1e+20 to int...
            try:
                fill_value = np.array(fill_value, copy=False, dtype=ndtype).item()
            except OverflowError:
                fill_value = default_fill_value(ndtype)
    return fill_value


def set_fill_value(a, fill_value):
    """
    Set the filling value of a, if a is a masked array.

    This function changes the fill value of the masked array `a` in place.
    If `a` is not a masked array, the function returns silently, without
    doing anything.

    Parameters
    ----------
    a : array_like
        Input array.
    fill_value : dtype
        Filling value. A consistency test is performed to make sure
        the value is compatible with the dtype of `a`.

    Returns
    -------
    None
        Nothing returned by this function.

    See Also
    --------
    maximum_fill_value : Return the default fill value for a dtype.
    MaskedArray.fill_value : Return current fill value.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> a = ma.masked_where(a < 3, a)
    >>> a
    masked_array(data = [-- -- -- 3 4],
          mask = [ True  True  True False False],
          fill_value=999999)
    >>> ma.set_fill_value(a, -999)
    >>> a
    masked_array(data = [-- -- -- 3 4],
          mask = [ True  True  True False False],
          fill_value=-999)

    Nothing happens if `a` is not a masked array.

    >>> a = range(5)
    >>> a
    [0, 1, 2, 3, 4]
    >>> ma.set_fill_value(a, 100)
    >>> a
    [0, 1, 2, 3, 4]
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> ma.set_fill_value(a, 100)
    >>> a
    array([0, 1, 2, 3, 4])

    """
    if isinstance(a, MaskedArray):
        a._fill_value = _check_fill_value(fill_value, a.dtype)
    return

def get_fill_value(a):
    """
    Return the filling value of a, if any.  Otherwise, returns the
    default filling value for that type.

    """
    if isinstance(a, MaskedArray):
        result = a.fill_value
    else:
        result = default_fill_value(a)
    return result

def common_fill_value(a, b):
    """
    Return the common filling value of a and b, if any.
    If a and b have different filling values, returns None.

    """
    t1 = get_fill_value(a)
    t2 = get_fill_value(b)
    if t1 == t2:
        return t1
    return None


#####--------------------------------------------------------------------------
def filled(a, fill_value = None):
    """
    Return `a` as an array where masked data have been replaced by `value`.

    If `a` is not a MaskedArray, `a` itself is returned.
    If `a` is a MaskedArray and `fill_value` is None, `fill_value` is set to
    `a.fill_value`.

    Parameters
    ----------
    a : maskedarray or array_like
        An input object.
    fill_value : {var}, optional
        Filling value. If None, the output of :func:`get_fill_value(a)` is used
        instead.

    Returns
    -------
    a : array_like

    """
    if hasattr(a, 'filled'):
        return a.filled(fill_value)
    elif isinstance(a, ndarray):
        # Should we check for contiguity ? and a.flags['CONTIGUOUS']:
        return a
    elif isinstance(a, dict):
        return np.array(a, 'O')
    else:
        return np.array(a)

#####--------------------------------------------------------------------------
def get_masked_subclass(*arrays):
    """
    Return the youngest subclass of MaskedArray from a list of (masked) arrays.
    In case of siblings, the first listed takes over.

    """
    if len(arrays) == 1:
        arr = arrays[0]
        if isinstance(arr, MaskedArray):
            rcls = type(arr)
        else:
            rcls = MaskedArray
    else:
        arrcls = [type(a) for a in arrays]
        rcls = arrcls[0]
        if not issubclass(rcls, MaskedArray):
            rcls = MaskedArray
        for cls in arrcls[1:]:
            if issubclass(cls, rcls):
                rcls = cls
    return rcls

#####--------------------------------------------------------------------------
def getdata(a, subok=True):
    """
    Return the data of a masked array as an ndarray.

    Return the data of `a` (if any) as an ndarray if `a` is a ``MaskedArray``,
    else return `a` as a ndarray or subclass (depending on `subok`) if not.

    Parameters
    ----------
    a : array_like
        Input ``MaskedArray``, alternatively a ndarray or a subclass thereof.
    subok : bool
        Whether to force the output to be a `pure` ndarray (False) or to
        return a subclass of ndarray if approriate (True - default).

    See Also
    --------
    getmask : Return the mask of a masked array, or nomask.
    getmaskarray : Return the mask of a masked array, or full array of False.

    Examples
    --------

    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(data =
     [[1 --]
     [3 4]],
          mask =
     [[False  True]
     [False False]],
          fill_value=999999)
    >>> ma.getdata(a)
    array([[1, 2],
           [3, 4]])

    Equivalently use the ``MaskedArray`` `data` attribute.

    >>> a.data
    array([[1, 2],
           [3, 4]])

    """
    data = getattr(a, '_data', np.array(a, subok=subok))
    if not subok:
        return data.view(ndarray)
    return data
get_data = getdata


def fix_invalid(a, mask=nomask, copy=True, fill_value=None):
    """
    Return (a copy of) `a` where invalid data (nan/inf) are masked
    and replaced by `fill_value`.

    Note that a copy is performed by default (just in case...).

    Parameters
    ----------
    a : array_like
        A (subclass of) ndarray.
    copy : bool
        Whether to use a copy of `a` (True) or to fix `a` in place (False).
    fill_value : {var}, optional
        Value used for fixing invalid data.  If not given, the output
        of get_fill_value(a) is used instead.

    Returns
    -------
    b : MaskedArray

    """
    a = masked_array(a, copy=copy, mask=mask, subok=True)
    #invalid = (numpy.isnan(a._data) | numpy.isinf(a._data))
    invalid = np.logical_not(np.isfinite(a._data))
    if not invalid.any():
        return a
    a._mask |= invalid
    if fill_value is None:
        fill_value = a.fill_value
    a._data[invalid] = fill_value
    return a



#####--------------------------------------------------------------------------
#---- --- Ufuncs ---
#####--------------------------------------------------------------------------
ufunc_domain = {}
ufunc_fills = {}

class _DomainCheckInterval:
    """Define a valid interval, so that :

    ``domain_check_interval(a,b)(x) = true`` where
    ``x < a`` or ``x > b``.

    """
    def __init__(self, a, b):
        "domain_check_interval(a,b)(x) = true where x < a or y > b"
        if (a > b):
            (a, b) = (b, a)
        self.a = a
        self.b = b

    def __call__ (self, x):
        "Execute the call behavior."
        return umath.logical_or(umath.greater (x, self.b),
                                umath.less(x, self.a))
#............................
class _DomainTan:
    """Define a valid interval for the `tan` function, so that:

    ``domain_tan(eps) = True`` where ``abs(cos(x)) < eps``

    """
    def __init__(self, eps):
        "domain_tan(eps) = true where abs(cos(x)) < eps)"
        self.eps = eps
    def __call__ (self, x):
        "Executes the call behavior."
        return umath.less(umath.absolute(umath.cos(x)), self.eps)
#............................
class _DomainSafeDivide:
    """Define a domain for safe division."""
    def __init__ (self, tolerance=None):
        self.tolerance = tolerance
    def __call__ (self, a, b):
        # Delay the selection of the tolerance to here in order to reduce numpy
        # import times. The calculation of these parameters is a substantial
        # component of numpy's import time.
        if self.tolerance is None:
            self.tolerance = np.finfo(float).tiny
        return umath.absolute(a) * self.tolerance >= umath.absolute(b)
#............................
class _DomainGreater:
    "DomainGreater(v)(x) = true where x <= v"
    def __init__(self, critical_value):
        "DomainGreater(v)(x) = true where x <= v"
        self.critical_value = critical_value

    def __call__ (self, x):
        "Executes the call behavior."
        return umath.less_equal(x, self.critical_value)
#............................
class _DomainGreaterEqual:
    "DomainGreaterEqual(v)(x) = true where x < v"
    def __init__(self, critical_value):
        "DomainGreaterEqual(v)(x) = true where x < v"
        self.critical_value = critical_value

    def __call__ (self, x):
        "Executes the call behavior."
        return umath.less(x, self.critical_value)

#..............................................................................
class _MaskedUnaryOperation:
    """Defines masked version of unary operations, where invalid
    values are pre-masked.

    Parameters
    ----------
    f : callable
    fill :
        Default filling value (0).
    domain :
        Default domain (None).

    """
    def __init__ (self, mufunc, fill=0, domain=None):
        """ _MaskedUnaryOperation(aufunc, fill=0, domain=None)
            aufunc(fill) must be defined
            self(x) returns aufunc(x)
            with masked values where domain(x) is true or getmask(x) is true.
        """
        self.f = mufunc
        self.fill = fill
        self.domain = domain
        self.__doc__ = getattr(mufunc, "__doc__", str(mufunc))
        self.__name__ = getattr(mufunc, "__name__", str(mufunc))
        ufunc_domain[mufunc] = domain
        ufunc_fills[mufunc] = fill
    #
    def __call__ (self, a, *args, **kwargs):
        "Execute the call behavior."
        #
        m = getmask(a)
        d1 = getdata(a)
        #
        if self.domain is not None:
            dm = np.array(self.domain(d1), copy=False)
            m = np.logical_or(m, dm)
            # The following two lines control the domain filling methods.
            d1 = d1.copy()
            # We could use smart indexing : d1[dm] = self.fill ...
            # ... but np.putmask looks more efficient, despite the copy.
            np.putmask(d1, dm, self.fill)
        # Take care of the masked singletong first ...
        if (not m.ndim) and m:
            return masked
        elif m is nomask:
            result = self.f(d1, *args, **kwargs)
        else:
            result = np.where(m, d1, self.f(d1, *args, **kwargs))
        # If result is not a scalar
        if result.ndim:
            # Get the result subclass:
            if isinstance(a, MaskedArray):
                subtype = type(a)
            else:
                subtype = MaskedArray
            result = result.view(subtype)
            result._mask = m
            result._update_from(a)
        return result
    #
    def __str__ (self):
        return "Masked version of %s. [Invalid values are masked]" % str(self.f)

#..............................................................................
class _MaskedBinaryOperation:
    """Define masked version of binary operations, where invalid
    values are pre-masked.

    Parameters
    ----------
    f : callable
    fillx :
        Default filling value for the first argument (0).
    filly :
        Default filling value for the second argument (0).
    domain :
        Default domain (None).

    """
    def __init__ (self, mbfunc, fillx=0, filly=0):
        """abfunc(fillx, filly) must be defined.
           abfunc(x, filly) = x for all x to enable reduce.
        """
        self.f = mbfunc
        self.fillx = fillx
        self.filly = filly
        self.__doc__ = getattr(mbfunc, "__doc__", str(mbfunc))
        self.__name__ = getattr(mbfunc, "__name__", str(mbfunc))
        ufunc_domain[mbfunc] = None
        ufunc_fills[mbfunc] = (fillx, filly)

    def __call__ (self, a, b, *args, **kwargs):
        "Execute the call behavior."
        m = mask_or(getmask(a), getmask(b), shrink=False)
        (da, db) = (getdata(a), getdata(b))
        # Easy case: there's no mask...
        if m is nomask:
            result = self.f(da, db, *args, **kwargs)
        # There are some masked elements: run only on the unmasked
        else:
            result = np.where(m, da, self.f(da, db, *args, **kwargs))
        # Transforms to a (subclass of) MaskedArray if we don't have a scalar
        if result.shape:
            result = result.view(get_masked_subclass(a, b))
            # If we have a mask, make sure it's broadcasted properly
            if m.any():
                result._mask = mask_or(getmaskarray(a), getmaskarray(b))
            # If some initial masks where not shrunk, don't shrink the result
            elif m.shape:
                result._mask = make_mask_none(result.shape, result.dtype)
            if isinstance(a, MaskedArray):
                result._update_from(a)
            if isinstance(b, MaskedArray):
                result._update_from(b)
        # ... or return masked if we have a scalar and the common mask is True
        elif m:
            return masked
        return result
#
#        result = self.f(d1, d2, *args, **kwargs).view(get_masked_subclass(a, b))
#        if len(result.shape):
#            if m is not nomask:
#                result._mask = make_mask_none(result.shape)
#                result._mask.flat = m
#                #!!!!!
#                # Force m to be at least 1D
#                m.shape = m.shape or (1,)
#                print "Resetting data"
#                result.data[m].flat = d1.flat
#                #!!!!!
#            if isinstance(a, MaskedArray):
#                result._update_from(a)
#            if isinstance(b, MaskedArray):
#                result._update_from(b)
#        elif m:
#            return masked
#        return result

    def reduce(self, target, axis=0, dtype=None):
        """Reduce `target` along the given `axis`."""
        if isinstance(target, MaskedArray):
            tclass = type(target)
        else:
            tclass = MaskedArray
        m = getmask(target)
        t = filled(target, self.filly)
        if t.shape == ():
            t = t.reshape(1)
            if m is not nomask:
                m = make_mask(m, copy=1)
                m.shape = (1,)
        if m is nomask:
            return self.f.reduce(t, axis).view(tclass)
        t = t.view(tclass)
        t._mask = m
        tr = self.f.reduce(getdata(t), axis, dtype=dtype or t.dtype)
        mr = umath.logical_and.reduce(m, axis)
        tr = tr.view(tclass)
        if mr.ndim > 0:
            tr._mask = mr
            return tr
        elif mr:
            return masked
        return tr

    def outer (self, a, b):
        """Return the function applied to the outer product of a and b.

        """
        ma = getmask(a)
        mb = getmask(b)
        if ma is nomask and mb is nomask:
            m = nomask
        else:
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = umath.logical_or.outer(ma, mb)
        if (not m.ndim) and m:
            return masked
        (da, db) = (getdata(a), getdata(b))
        if m is nomask:
            d = self.f.outer(da, db)
        else:
            d = np.where(m, da, self.f.outer(da, db))
        if d.shape:
            d = d.view(get_masked_subclass(a, b))
            d._mask = m
        return d

    def accumulate (self, target, axis=0):
        """Accumulate `target` along `axis` after filling with y fill
        value.

        """
        if isinstance(target, MaskedArray):
            tclass = type(target)
        else:
            tclass = MaskedArray
        t = filled(target, self.filly)
        return self.f.accumulate(t, axis).view(tclass)

    def __str__ (self):
        return "Masked version of " + str(self.f)

#..............................................................................
class _DomainedBinaryOperation:
    """
    Define binary operations that have a domain, like divide.

    They have no reduce, outer or accumulate.

    Parameters
    ----------
    f : function.
    domain : Default domain.
    fillx : Default filling value for the first argument (0).
    filly : Default filling value for the second argument (0).

    """
    def __init__ (self, dbfunc, domain, fillx=0, filly=0):
        """abfunc(fillx, filly) must be defined.
           abfunc(x, filly) = x for all x to enable reduce.
        """
        self.f = dbfunc
        self.domain = domain
        self.fillx = fillx
        self.filly = filly
        self.__doc__ = getattr(dbfunc, "__doc__", str(dbfunc))
        self.__name__ = getattr(dbfunc, "__name__", str(dbfunc))
        ufunc_domain[dbfunc] = domain
        ufunc_fills[dbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        "Execute the call behavior."
        ma = getmask(a)
        mb = getmaskarray(b)
        da = getdata(a)
        db = getdata(b)
        t = narray(self.domain(da, db), copy=False)
        if t.any(None):
            mb = mask_or(mb, t, shrink=False)
            # The following line controls the domain filling
            if t.size == db.size:
                db = np.where(t, self.filly, db)
            else:
                db = np.where(np.resize(t, db.shape), self.filly, db)
        # Shrink m if a.mask was nomask, otherwise don't.
        m = mask_or(ma, mb, shrink=(getattr(a, '_mask', nomask) is nomask))
        if (not m.ndim) and m:
            return masked
        elif (m is nomask):
            result = self.f(da, db, *args, **kwargs)
        else:
            result = np.where(m, da, self.f(da, db, *args, **kwargs))
        if result.shape:
            result = result.view(get_masked_subclass(a, b))
            # If we have a mask, make sure it's broadcasted properly
            if m.any():
                result._mask = mask_or(getmaskarray(a), mb)
            # If some initial masks where not shrunk, don't shrink the result
            elif m.shape:
                result._mask = make_mask_none(result.shape, result.dtype)
            if isinstance(a, MaskedArray):
                result._update_from(a)
            if isinstance(b, MaskedArray):
                result._update_from(b)
        return result

    def __str__ (self):
        return "Masked version of " + str(self.f)

#..............................................................................
# Unary ufuncs
exp = _MaskedUnaryOperation(umath.exp)
conjugate = _MaskedUnaryOperation(umath.conjugate)
sin = _MaskedUnaryOperation(umath.sin)
cos = _MaskedUnaryOperation(umath.cos)
tan = _MaskedUnaryOperation(umath.tan)
arctan = _MaskedUnaryOperation(umath.arctan)
arcsinh = _MaskedUnaryOperation(umath.arcsinh)
sinh = _MaskedUnaryOperation(umath.sinh)
cosh = _MaskedUnaryOperation(umath.cosh)
tanh = _MaskedUnaryOperation(umath.tanh)
abs = absolute = _MaskedUnaryOperation(umath.absolute)
fabs = _MaskedUnaryOperation(umath.fabs)
negative = _MaskedUnaryOperation(umath.negative)
floor = _MaskedUnaryOperation(umath.floor)
ceil = _MaskedUnaryOperation(umath.ceil)
around = _MaskedUnaryOperation(np.round_)
logical_not = _MaskedUnaryOperation(umath.logical_not)
# Domained unary ufuncs .......................................................
sqrt = _MaskedUnaryOperation(umath.sqrt, 0.0,
                             _DomainGreaterEqual(0.0))
log = _MaskedUnaryOperation(umath.log, 1.0,
                            _DomainGreater(0.0))
log10 = _MaskedUnaryOperation(umath.log10, 1.0,
                              _DomainGreater(0.0))
tan = _MaskedUnaryOperation(umath.tan, 0.0,
                            _DomainTan(1.e-35))
arcsin = _MaskedUnaryOperation(umath.arcsin, 0.0,
                               _DomainCheckInterval(-1.0, 1.0))
arccos = _MaskedUnaryOperation(umath.arccos, 0.0,
                               _DomainCheckInterval(-1.0, 1.0))
arccosh = _MaskedUnaryOperation(umath.arccosh, 1.0,
                                _DomainGreaterEqual(1.0))
arctanh = _MaskedUnaryOperation(umath.arctanh, 0.0,
                                _DomainCheckInterval(-1.0+1e-15, 1.0-1e-15))
# Binary ufuncs ...............................................................
add = _MaskedBinaryOperation(umath.add)
subtract = _MaskedBinaryOperation(umath.subtract)
multiply = _MaskedBinaryOperation(umath.multiply, 1, 1)
arctan2 = _MaskedBinaryOperation(umath.arctan2, 0.0, 1.0)
equal = _MaskedBinaryOperation(umath.equal)
equal.reduce = None
not_equal = _MaskedBinaryOperation(umath.not_equal)
not_equal.reduce = None
less_equal = _MaskedBinaryOperation(umath.less_equal)
less_equal.reduce = None
greater_equal = _MaskedBinaryOperation(umath.greater_equal)
greater_equal.reduce = None
less = _MaskedBinaryOperation(umath.less)
less.reduce = None
greater = _MaskedBinaryOperation(umath.greater)
greater.reduce = None
logical_and = _MaskedBinaryOperation(umath.logical_and)
alltrue = _MaskedBinaryOperation(umath.logical_and, 1, 1).reduce
logical_or = _MaskedBinaryOperation(umath.logical_or)
sometrue = logical_or.reduce
logical_xor = _MaskedBinaryOperation(umath.logical_xor)
bitwise_and = _MaskedBinaryOperation(umath.bitwise_and)
bitwise_or = _MaskedBinaryOperation(umath.bitwise_or)
bitwise_xor = _MaskedBinaryOperation(umath.bitwise_xor)
hypot = _MaskedBinaryOperation(umath.hypot)
# Domained binary ufuncs ......................................................
divide = _DomainedBinaryOperation(umath.divide, _DomainSafeDivide(), 0, 1)
true_divide = _DomainedBinaryOperation(umath.true_divide,
                                        _DomainSafeDivide(), 0, 1)
floor_divide = _DomainedBinaryOperation(umath.floor_divide,
                                         _DomainSafeDivide(), 0, 1)
remainder = _DomainedBinaryOperation(umath.remainder,
                                      _DomainSafeDivide(), 0, 1)
fmod = _DomainedBinaryOperation(umath.fmod, _DomainSafeDivide(), 0, 1)
mod = _DomainedBinaryOperation(umath.mod, _DomainSafeDivide(), 0, 1)


#####--------------------------------------------------------------------------
#---- --- Mask creation functions ---
#####--------------------------------------------------------------------------

def _recursive_make_descr(datatype, newtype=bool_):
    "Private function allowing recursion in make_descr."
    # Do we have some name fields ?
    if datatype.names:
        descr = []
        for name in datatype.names:
            field = datatype.fields[name]
            if len(field) == 3:
                # Prepend the title to the name
                name = (field[-1], name)
            descr.append((name, _recursive_make_descr(field[0], newtype)))
        return descr
    # Is this some kind of composite a la (np.float,2)
    elif datatype.subdtype:
        mdescr = list(datatype.subdtype)
        mdescr[0] = newtype
        return tuple(mdescr)
    else:
        return newtype

def make_mask_descr(ndtype):
    """
    Construct a dtype description list from a given dtype.

    Returns a new dtype object, with the type of all fields in `ndtype` to a
    boolean type. Field names are not altered.

    Parameters
    ----------
    ndtype : dtype
        The dtype to convert.

    Returns
    -------
    result : dtype
        A dtype that looks like `ndtype`, the type of all fields is boolean.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> dtype = np.dtype({'names':['foo', 'bar'],
                          'formats':[np.float32, np.int]})
    >>> dtype
    dtype([('foo', '<f4'), ('bar', '<i4')])
    >>> ma.make_mask_descr(dtype)
    dtype([('foo', '|b1'), ('bar', '|b1')])
    >>> ma.make_mask_descr(np.float32)
    <type 'numpy.bool_'>

    """
    # Make sure we do have a dtype
    if not isinstance(ndtype, np.dtype):
        ndtype = np.dtype(ndtype)
    return np.dtype(_recursive_make_descr(ndtype, np.bool))

def getmask(a):
    """
    Return the mask of a masked array, or nomask.

    Return the mask of `a` as an ndarray if `a` is a `MaskedArray` and the
    mask is not `nomask`, else return `nomask`. To guarantee a full array
    of booleans of the same shape as a, use `getmaskarray`.

    Parameters
    ----------
    a : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getdata : Return the data of a masked array as an ndarray.
    getmaskarray : Return the mask of a masked array, or full array of False.

    Examples
    --------

    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(data =
     [[1 --]
     [3 4]],
          mask =
     [[False  True]
     [False False]],
          fill_value=999999)
    >>> ma.getmask(a)
    array([[False,  True],
           [False, False]], dtype=bool)

    Equivalently use the `MaskedArray` `mask` attribute.

    >>> a.mask
    array([[False,  True],
           [False, False]], dtype=bool)

    Result when mask == `nomask`

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(data =
     [[1 2]
     [3 4]],
          mask =
     False,
          fill_value=999999)
    >>> ma.nomask
    False
    >>> ma.getmask(b) == ma.nomask
    True
    >>> b.mask == ma.nomask
    True

    """
    return getattr(a, '_mask', nomask)
get_mask = getmask

def getmaskarray(arr):
    """
    Return the mask of a masked array, or full boolean array of False.

    Return the mask of `arr` as an ndarray if `arr` is a `MaskedArray` and
    the mask is not `nomask`, else return a full boolean array of False of
    the same shape as `arr`.

    Parameters
    ----------
    arr : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getmask : Return the mask of a masked array, or nomask.
    getdata : Return the data of a masked array as an ndarray.

    Examples
    --------

    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(data =
     [[1 --]
     [3 4]],
          mask =
     [[False  True]
     [False False]],
          fill_value=999999)
    >>> ma.getmaskarray(a)
    array([[False,  True],
           [False, False]], dtype=bool)

    Result when mask == ``nomask``

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(data =
     [[1 2]
     [3 4]],
          mask =
     False,
          fill_value=999999)
    >>> >ma.getmaskarray(b)
    array([[False, False],
           [False, False]], dtype=bool)

    """
    mask = getmask(arr)
    if mask is nomask:
        mask = make_mask_none(np.shape(arr), getdata(arr).dtype)
    return mask

def is_mask(m):
    """
    Return True if m is a valid, standard mask.

    This function does not check the contents of the input, only that the
    type is MaskType. In particular, this function returns False if the
    mask has a flexible dtype.

    Parameters
    ----------
    m : array_like
        Array to test.

    Returns
    -------
    result : bool
        True if `m.dtype.type` is MaskType, False otherwise.

    See Also
    --------
    isMaskedArray : Test whether input is an instance of MaskedArray.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> m = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> m
    masked_array(data = [-- 1 -- 2 3],
          mask = [ True False  True False False],
          fill_value=999999)
    >>> ma.is_mask(m)
    False
    >>> ma.is_mask(m.mask)
    True

    Input must be an ndarray (or have similar attributes)
    for it to be considered a valid mask.

    >>> m = [False, True, False]
    >>> ma.is_mask(m)
    False
    >>> m = np.array([False, True, False])
    >>> m
    array([False,  True, False], dtype=bool)
    >>> ma.is_mask(m)
    True

    Arrays with complex dtypes don't return True.

    >>> dtype = np.dtype({'names':['monty', 'pithon'],
                          'formats':[np.bool, np.bool]})
    >>> dtype
    dtype([('monty', '|b1'), ('pithon', '|b1')])
    >>> m = np.array([(True, False), (False, True), (True, False)],
                     dtype=dtype)
    >>> m
    array([(True, False), (False, True), (True, False)],
          dtype=[('monty', '|b1'), ('pithon', '|b1')])
    >>> ma.is_mask(m)
    False

    """
    try:
        return m.dtype.type is MaskType
    except AttributeError:
        return False

def make_mask(m, copy=False, shrink=True, flag=None, dtype=MaskType):
    """
    Create a boolean mask from an array.

    Return `m` as a boolean mask, creating a copy if necessary or requested.
    The function can accept any sequence that is convertible to integers,
    or ``nomask``.  Does not require that contents must be 0s and 1s, values
    of 0 are interepreted as False, everything else as True.

    Parameters
    ----------
    m : array_like
        Potential mask.
    copy : bool
        Whether to return a copy of `m` (True) or `m` itself (False).
    shrink : bool
        Whether to shrink `m` to ``nomask`` if all its values are False.
    flag : bool
        Deprecated equivalent of `shrink`.
    dtype : dtype
        Data-type of the output mask. By default, the output mask has
        a dtype of MaskType (bool). If the dtype is flexible, each field
        has a boolean dtype.

    Returns
    -------
    result : ndarray
        A boolean mask derived from `m`.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> m = [True, False, True, True]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True], dtype=bool)
    >>> m = [1, 0, 1, 1]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True], dtype=bool)
    >>> m = [1, 0, 2, -3]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True], dtype=bool)

    Effect of the `shrink` parameter.

    >>> m = np.zeros(4)
    >>> m
    array([ 0.,  0.,  0.,  0.])
    >>> ma.make_mask(m)
    False
    >>> ma.make_mask(m, shrink=False)
    array([False, False, False, False], dtype=bool)

    Using a flexible `dtype`.

    >>> m = [1, 0, 1, 1]
    >>> n = [0, 1, 0, 0]
    >>> arr = []
    >>> for man, mouse in zip(m, n):
    ...     arr.append((man, mouse))
    >>> arr
    [(1, 0), (0, 1), (1, 0), (1, 0)]
    >>> dtype = np.dtype({'names':['man', 'mouse'],
                          'formats':[np.int, np.int]})
    >>> arr = np.array(arr, dtype=dtype)
    >>> arr
    array([(1, 0), (0, 1), (1, 0), (1, 0)],
          dtype=[('man', '<i4'), ('mouse', '<i4')])
    >>> ma.make_mask(arr, dtype=dtype)
    array([(True, False), (False, True), (True, False), (True, False)],
          dtype=[('man', '|b1'), ('mouse', '|b1')])

    """
    if flag is not None:
        warnings.warn("The flag 'flag' is now called 'shrink'!",
                      DeprecationWarning)
        shrink = flag
    if m is nomask:
        return nomask
    elif isinstance(m, ndarray):
        # We won't return after this point to make sure we can shrink the mask
        # Fill the mask in case there are missing data
        m = filled(m, True)
        # Make sure the input dtype is valid
        dtype = make_mask_descr(dtype)
        if m.dtype == dtype:
            if copy:
                result = m.copy()
            else:
                result = m
        else:
            result = np.array(m, dtype=dtype, copy=copy)
    else:
        result = np.array(filled(m, True), dtype=MaskType)
    # Bas les masques !
    if shrink and (not result.dtype.names) and (not result.any()):
        return nomask
    else:
        return result


def make_mask_none(newshape, dtype=None):
    """
    Return a boolean mask of the given shape, filled with False.

    This function returns a boolean ndarray with all entries False, that can
    be used in common mask manipulations. If a complex dtype is specified, the
    type of each field is converted to a boolean type.

    Parameters
    ----------
    newshape : tuple
        A tuple indicating the shape of the mask.
    dtype: {None, dtype}, optional
        If None, use a MaskType instance. Otherwise, use a new datatype with
        the same fields as `dtype`, converted to boolean types.

    Returns
    -------
    result : ndarray
        An ndarray of appropriate shape and dtype, filled with False.

    See Also
    --------
    make_mask : Create a boolean mask from an array.
    make_mask_descr : Construct a dtype description list from a given dtype.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> ma.make_mask_none((3,))
    array([False, False, False], dtype=bool)

    Defining a more complex dtype.

    >>> dtype = np.dtype({'names':['foo', 'bar'],
                          'formats':[np.float32, np.int]})
    >>> dtype
    dtype([('foo', '<f4'), ('bar', '<i4')])
    >>> ma.make_mask_none((3,), dtype=dtype)
    array([(False, False), (False, False), (False, False)],
          dtype=[('foo', '|b1'), ('bar', '|b1')])

    """
    if dtype is None:
        result = np.zeros(newshape, dtype=MaskType)
    else:
        result = np.zeros(newshape, dtype=make_mask_descr(dtype))
    return result

def mask_or (m1, m2, copy=False, shrink=True):
    """
    Return the combination of two masks m1 and m2.

    The masks are combined with the *logical_or* operator, treating
    nomask as False.  The result may equal m1 or m2 if the other is
    nomask.

    Parameters
    ----------
    m1 : array_like
        First mask.
    m2 : array_like
        Second mask
    copy : {False, True}, optional
        Whether to return a copy.
    shrink : {True, False}, optional
        Whether to shrink m to nomask if all its values are False.

    Raises
    ------
    ValueError
        If m1 and m2 have different flexible dtypes.

    """
    def _recursive_mask_or(m1, m2, newmask):
        names = m1.dtype.names
        for name in names:
            current1 = m1[name]
            if current1.dtype.names:
                _recursive_mask_or(current1, m2[name], newmask[name])
            else:
                umath.logical_or(current1, m2[name], newmask[name])
        return
    #
    if (m1 is nomask) or (m1 is False):
        dtype = getattr(m2, 'dtype', MaskType)
        return make_mask(m2, copy=copy, shrink=shrink, dtype=dtype)
    if (m2 is nomask) or (m2 is False):
        dtype = getattr(m1, 'dtype', MaskType)
        return make_mask(m1, copy=copy, shrink=shrink, dtype=dtype)
    if m1 is m2 and is_mask(m1):
        return m1
    (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
    if (dtype1 != dtype2):
        raise ValueError("Incompatible dtypes '%s'<>'%s'" % (dtype1, dtype2))
    if dtype1.names:
        newmask = np.empty_like(m1)
        _recursive_mask_or(m1, m2, newmask)
        return newmask
    return make_mask(umath.logical_or(m1, m2), copy=copy, shrink=shrink)


def flatten_mask(mask):
    """
    Returns a completely flattened version of the mask, where nested fields
    are collapsed.

    Parameters
    ----------
    mask : array_like
        Array of booleans

    Returns
    -------
    flattened_mask : ndarray
        Boolean array.

    Examples
    --------
    >>> mask = np.array([0, 0, 1], dtype=np.bool)
    >>> flatten_mask(mask)
    array([False, False,  True], dtype=bool)
    >>> mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
    >>> flatten_mask(mask)
    array([False, False, False,  True], dtype=bool)
    >>> mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
    >>> mask = np.array([(0, (0, 0)), (0, (0, 1))], dtype=mdtype)
    >>> flatten_mask(mask)
    array([False, False, False, False, False,  True], dtype=bool)

    """
    #
    def _flatmask(mask):
        "Flatten the mask and returns a (maybe nested) sequence of booleans."
        mnames = mask.dtype.names
        if mnames:
            return [flatten_mask(mask[name]) for name in mnames]
        else:
            return mask
    #
    def _flatsequence(sequence):
        "Generates a flattened version of the sequence."
        try:
            for element in sequence:
                if hasattr(element, '__iter__'):
                    for f in _flatsequence(element):
                        yield f
                else:
                    yield element
        except TypeError:
            yield sequence
    #
    mask = np.asarray(mask)
    flattened = _flatsequence(_flatmask(mask))
    return np.array([_ for _ in flattened], dtype=bool)


#####--------------------------------------------------------------------------
#--- --- Masking functions ---
#####--------------------------------------------------------------------------

def masked_where(condition, a, copy=True):
    """
    Mask an array where a condition is met.

    Return `a` as an array masked where `condition` is True.
    Any masked values of `a` or `condition` are also masked in the output.

    Parameters
    ----------
    condition : array_like
        Masking condition.  When `condition` tests floating point values for
        equality, consider using ``masked_values`` instead.
    a : array_like
        Array to mask.
    copy : bool
        If True (default) make a copy of `a` in the result.  If False modify
        `a` in place and return a view.

    Returns
    -------
    result : MaskedArray
        The result of masking `a` where `condition` is True.

    See Also
    --------
    masked_values : Mask using floating point equality.
    masked_equal : Mask where equal to a given value.
    masked_not_equal : Mask where `not` equal to a given value.
    masked_less_equal : Mask where less than or equal to a given value.
    masked_greater_equal : Mask where greater than or equal to a given value.
    masked_less : Mask where less than a given value.
    masked_greater : Mask where greater than a given value.
    masked_inside : Mask inside a given interval.
    masked_outside : Mask outside a given interval.
    masked_invalid : Mask invalid values (NaNs or infs).

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_where(a <= 2, a)
    masked_array(data = [-- -- -- 3],
          mask = [ True  True  True False],
          fill_value=999999)

    Mask array `b` conditional on `a`.

    >>> b = ['a', 'b', 'c', 'd']
    >>> ma.masked_where(a == 2, b)
    masked_array(data = [a b -- d],
          mask = [False False  True False],
          fill_value=N/A)

    Effect of the `copy` argument.

    >>> c = ma.masked_where(a <= 2, a)
    >>> c
    masked_array(data = [-- -- -- 3],
          mask = [ True  True  True False],
          fill_value=999999)
    >>> c[0] = 99
    >>> c
    masked_array(data = [99 -- -- 3],
          mask = [False  True  True False],
          fill_value=999999)
    >>> a
    array([0, 1, 2, 3])
    >>> c = ma.masked_where(a <= 2, a, copy=False)
    >>> c[0] = 99
    >>> c
    masked_array(data = [99 -- -- 3],
          mask = [False  True  True False],
          fill_value=999999)
    >>> a
    array([99,  1,  2,  3])

    When `condition` or `a` contain masked values.

    >>> a = np.arange(4)
    >>> a = ma.masked_where(a == 2, a)
    >>> a
    masked_array(data = [0 1 -- 3],
          mask = [False False  True False],
          fill_value=999999)
    >>> b = np.arange(4)
    >>> b = ma.masked_where(b == 0, b)
    >>> b
    masked_array(data = [-- 1 2 3],
          mask = [ True False False False],
          fill_value=999999)
    >>> ma.masked_where(a == 3, b)
    masked_array(data = [-- 1 -- --],
          mask = [ True False  True  True],
          fill_value=999999)

    """
    # Make sure that condition is a valid standard-type mask.
    cond = make_mask(condition)
    a = np.array(a, copy=copy, subok=True)

    (cshape, ashape) = (cond.shape, a.shape)
    if cshape and cshape != ashape:
        raise IndexError("Inconsistant shape between the condition and the input"\
                         " (got %s and %s)" % (cshape, ashape))
    if hasattr(a, '_mask'):
        cond = mask_or(cond, a._mask)
        cls = type(a)
    else:
        cls = MaskedArray
    result = a.view(cls)
    result._mask = cond
    return result


def masked_greater(x, value, copy=True):
    """
    Mask an array where greater than a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x > value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_greater(a, 2)
    masked_array(data = [0 1 2 --],
          mask = [False False False  True],
          fill_value=999999)

    """
    return masked_where(greater(x, value), x, copy=copy)


def masked_greater_equal(x, value, copy=True):
    """
    Mask an array where greater than or equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x >= value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_greater_equal(a, 2)
    masked_array(data = [0 1 -- --],
          mask = [False False  True  True],
          fill_value=999999)

    """
    return masked_where(greater_equal(x, value), x, copy=copy)


def masked_less(x, value, copy=True):
    """
    Mask an array where less than a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x < value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_less(a, 2)
    masked_array(data = [-- -- 2 3],
          mask = [ True  True False False],
          fill_value=999999)

    """
    return masked_where(less(x, value), x, copy=copy)


def masked_less_equal(x, value, copy=True):
    """
    Mask an array where less than or equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x <= value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_less_equal(a, 2)
    masked_array(data = [-- -- -- 3],
          mask = [ True  True  True False],
          fill_value=999999)

    """
    return masked_where(less_equal(x, value), x, copy=copy)


def masked_not_equal(x, value, copy=True):
    """
    Mask an array where `not` equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x != value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_not_equal(a, 2)
    masked_array(data = [-- -- 2 --],
          mask = [ True  True False  True],
          fill_value=999999)

    """
    return masked_where(not_equal(x, value), x, copy=copy)


def masked_equal(x, value, copy=True):
    """
    Mask an array where equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x == value).  For floating point arrays,
    consider using ``masked_values(x, value)``.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_values : Mask using floating point equality.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_equal(a, 2)
    masked_array(data = [0 1 -- 3],
          mask = [False False  True False],
          fill_value=999999)

    """
    # An alternative implementation relies on filling first: probably not needed.
    # d = filled(x, 0)
    # c = umath.equal(d, value)
    # m = mask_or(c, getmask(x))
    # return array(d, mask=m, copy=copy)
    return masked_where(equal(x, value), x, copy=copy)


def masked_inside(x, v1, v2, copy=True):
    """
    Mask an array inside a given interval.

    Shortcut to ``masked_where``, where `condition` is True for `x` inside
    the interval [v1,v2] (v1 <= x <= v2).  The boundaries `v1` and `v2`
    can be given in either order.

    See Also
    --------
    masked_where : Mask where a condition is met.

    Notes
    -----
    The array `x` is prefilled with its filling value.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
    >>> ma.masked_inside(x, -0.3, 0.3)
    masked_array(data = [0.31 1.2 -- -- -0.4 -1.1],
          mask = [False False  True  True False False],
          fill_value=1e+20)

    The order of `v1` and `v2` doesn't matter.

    >>> ma.masked_inside(x, 0.3, -0.3)
    masked_array(data = [0.31 1.2 -- -- -0.4 -1.1],
          mask = [False False  True  True False False],
          fill_value=1e+20)

    """
    if v2 < v1:
        (v1, v2) = (v2, v1)
    xf = filled(x)
    condition = (xf >= v1) & (xf <= v2)
    return masked_where(condition, x, copy=copy)


def masked_outside(x, v1, v2, copy=True):
    """
    Mask an array outside a given interval.

    Shortcut to ``masked_where``, where `condition` is True for `x` outside
    the interval [v1,v2] (x < v1)|(x > v2).
    The boundaries `v1` and `v2` can be given in either order.

    See Also
    --------
    masked_where : Mask where a condition is met.

    Notes
    -----
    The array `x` is prefilled with its filling value.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
    >>> ma.masked_outside(x, -0.3, 0.3)
    masked_array(data = [-- -- 0.01 0.2 -- --],
          mask = [ True  True False False  True  True],
          fill_value=1e+20)

    The order of `v1` and `v2` doesn't matter.

    >>> ma.masked_outside(x, 0.3, -0.3)
    masked_array(data = [-- -- 0.01 0.2 -- --],
          mask = [ True  True False False  True  True],
          fill_value=1e+20)

    """
    if v2 < v1:
        (v1, v2) = (v2, v1)
    xf = filled(x)
    condition = (xf < v1) | (xf > v2)
    return masked_where(condition, x, copy=copy)


def masked_object(x, value, copy=True, shrink=True):
    """
    Mask the array `x` where the data are exactly equal to value.

    This function is similar to `masked_values`, but only suitable
    for object arrays: for floating point, use `masked_values` instead.

    Parameters
    ----------
    x : array_like
        Array to mask
    value : object
        Comparison value
    copy : {True, False}, optional
        Whether to return a copy of `x`.
    shrink : {True, False}, optional
        Whether to collapse a mask full of False to nomask

    Returns
    -------
    result : MaskedArray
        The result of masking `x` where equal to `value`.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_equal : Mask where equal to a given value (integers).
    masked_values : Mask using floating point equality.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> food = np.array(['green_eggs', 'ham'], dtype=object)
    >>> # don't eat spoiled food
    >>> eat = ma.masked_object(food, 'green_eggs')
    >>> print eat
    [-- ham]
    >>> # plain ol` ham is boring
    >>> fresh_food = np.array(['cheese', 'ham', 'pineapple'], dtype=object)
    >>> eat = ma.masked_object(fresh_food, 'green_eggs')
    >>> print eat
    [cheese ham pineapple]

    Note that `mask` is set to ``nomask`` if possible.

    >>> eat
    masked_array(data = [cheese ham pineapple],
          mask = False,
          fill_value=?)

    """
    if isMaskedArray(x):
        condition = umath.equal(x._data, value)
        mask = x._mask
    else:
        condition = umath.equal(np.asarray(x), value)
        mask = nomask
    mask = mask_or(mask, make_mask(condition, shrink=shrink))
    return masked_array(x, mask=mask, copy=copy, fill_value=value)


def masked_values(x, value, rtol=1.e-5, atol=1.e-8, copy=True, shrink=True):
    """
    Mask using floating point equality.

    Return a MaskedArray, masked where the data in array `x` are approximately
    equal to `value`, i.e. where the following condition is True

    (abs(x - value) <= atol+rtol*abs(value))

    The fill_value is set to `value` and the mask is set to ``nomask`` if
    possible.  For integers, consider using ``masked_equal``.

    Parameters
    ----------
    x : array_like
        Array to mask.
    value : float
        Masking value.
    rtol : float, optional
        Tolerance parameter.
    atol : float, optional
        Tolerance parameter (1e-8).
    copy : {True, False}, optional
        Whether to return a copy of `x`.
    shrink : {True, False}, optional
        Whether to collapse a mask full of False to ``nomask``

    Returns
    -------
    result : MaskedArray
        The result of masking `x` where approximately equal to `value`.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_equal : Mask where equal to a given value (integers).

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = np.array([1, 1.1, 2, 1.1, 3])
    >>> ma.masked_values(x, 1.1)
    masked_array(data = [1.0 -- 2.0 -- 3.0],
          mask = [False  True False  True False],
          fill_value=1.1)

    Note that `mask` is set to ``nomask`` if possible.

    >>> ma.masked_values(x, 1.5)
    masked_array(data = [ 1.   1.1  2.   1.1  3. ],
          mask = False,
          fill_value=1.5)

    For integers, the fill value will be different in general to the
    result of ``masked_equal``.

    >>> x = np.arange(5)
    >>> x
    array([0, 1, 2, 3, 4])
    >>> ma.masked_values(x, 2)
    masked_array(data = [0 1 -- 3 4],
          mask = [False False  True False False],
          fill_value=2)
    >>> ma.masked_equal(x, 2)
    masked_array(data = [0 1 -- 3 4],
          mask = [False False  True False False],
          fill_value=999999)

    """
    mabs = umath.absolute
    xnew = filled(x, value)
    if issubclass(xnew.dtype.type, np.floating):
        condition = umath.less_equal(mabs(xnew-value), atol + rtol*mabs(value))
        mask = getattr(x, '_mask', nomask)
    else:
        condition = umath.equal(xnew, value)
        mask = nomask
    mask = mask_or(mask, make_mask(condition, shrink=shrink))
    return masked_array(xnew, mask=mask, copy=copy, fill_value=value)


def masked_invalid(a, copy=True):
    """
    Mask an array where invalid values occur (NaNs or infs).

    This function is a shortcut to ``masked_where``, with
    `condition` = ~(np.isfinite(a)). Any pre-existing mask is conserved.
    Only applies to arrays with a dtype where NaNs or infs make sense
    (i.e. floating point types), but accepts any array_like object.

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(5, dtype=np.float)
    >>> a[2] = np.NaN
    >>> a[3] = np.PINF
    >>> a
    array([  0.,   1.,  NaN,  Inf,   4.])
    >>> ma.masked_invalid(a)
    masked_array(data = [0.0 1.0 -- -- 4.0],
          mask = [False False  True  True False],
          fill_value=1e+20)

    """
    a = np.array(a, copy=copy, subok=True)
    condition = ~(np.isfinite(a))
    if hasattr(a, '_mask'):
        condition = mask_or(condition, a._mask)
        cls = type(a)
    else:
        cls = MaskedArray
    result = a.view(cls)
    result._mask = condition
    return result


#####--------------------------------------------------------------------------
#---- --- Printing options ---
#####--------------------------------------------------------------------------

class _MaskedPrintOption:
    """
    Handle the string used to represent missing data in a masked array.

    """
    def __init__ (self, display):
        "Create the masked_print_option object."
        self._display = display
        self._enabled = True

    def display(self):
        "Display the string to print for masked values."
        return self._display

    def set_display (self, s):
        "Set the string to print for masked values."
        self._display = s

    def enabled(self):
        "Is the use of the display value enabled?"
        return self._enabled

    def enable(self, shrink=1):
        "Set the enabling shrink to `shrink`."
        self._enabled = shrink

    def __str__ (self):
        return str(self._display)

    __repr__ = __str__

#if you single index into a masked location you get this object.
masked_print_option = _MaskedPrintOption('--')


def _recursive_printoption(result, mask, printopt):
    """
    Puts printoptions in result where mask is True.
    Private function allowing for recursion
    """
    names = result.dtype.names
    for name in names:
        (curdata, curmask) = (result[name], mask[name])
        if curdata.dtype.names:
            _recursive_printoption(curdata, curmask, printopt)
        else:
            np.putmask(curdata, curmask, printopt)
    return

_print_templates = dict(long = """\
masked_%(name)s(data =
 %(data)s,
       %(nlen)s mask =
 %(mask)s,
 %(nlen)s fill_value = %(fill)s)
""",
                        short = """\
masked_%(name)s(data = %(data)s,
       %(nlen)s mask = %(mask)s,
%(nlen)s  fill_value = %(fill)s)
""",
                        long_flx = """\
masked_%(name)s(data =
 %(data)s,
       %(nlen)s mask =
 %(mask)s,
%(nlen)s  fill_value = %(fill)s,
      %(nlen)s dtype = %(dtype)s)
""",
                        short_flx = """\
masked_%(name)s(data = %(data)s,
%(nlen)s        mask = %(mask)s,
%(nlen)s  fill_value = %(fill)s,
%(nlen)s       dtype = %(dtype)s)
""")

#####--------------------------------------------------------------------------
#---- --- MaskedArray class ---
#####--------------------------------------------------------------------------

def _recursive_filled(a, mask, fill_value):
    """
    Recursively fill `a` with `fill_value`.
    Private function
    """
    names = a.dtype.names
    for name in names:
        current = a[name]
        if current.dtype.names:
            _recursive_filled(current, mask[name], fill_value[name])
        else:
            np.putmask(current, mask[name], fill_value[name])

#...............................................................................
class _arraymethod(object):
    """
    Define a wrapper for basic array methods.

    Upon call, returns a masked array, where the new _data array is
    the output of the corresponding method called on the original
    _data.

    If onmask is True, the new mask is the output of the method called
    on the initial mask. Otherwise, the new mask is just a reference
    to the initial mask.

    Parameters
    ----------
    _name : String
        Name of the function to apply on data.
    _onmask : bool
        Whether the mask must be processed also (True) or left
        alone (False). Default: True.
    obj : Object
        The object calling the arraymethod.

    """
    def __init__(self, funcname, onmask=True):
        self.__name__ = funcname
        self._onmask = onmask
        self.obj = None
        self.__doc__ = self.getdoc()
    #
    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        methdoc = getattr(ndarray, self.__name__, None) or \
                  getattr(np, self.__name__, None)
        if methdoc is not None:
            return methdoc.__doc__
    #
    def __get__(self, obj, objtype=None):
        self.obj = obj
        return self
    #
    def __call__(self, *args, **params):
        methodname = self.__name__
        data = self.obj._data
        mask = self.obj._mask
        cls = type(self.obj)
        result = getattr(data, methodname)(*args, **params).view(cls)
        result._update_from(self.obj)
        if result.ndim:
            if not self._onmask:
                result.__setmask__(mask)
            elif mask is not nomask:
                result.__setmask__(getattr(mask, methodname)(*args, **params))
        else:
            if mask.ndim and (not mask.dtype.names and mask.all()):
                return masked
        return result
#..........................................................

class MaskedIterator(object):
    "Define an interator."
    def __init__(self, ma):
        self.ma = ma
        self.dataiter = ma._data.flat
        #
        if ma._mask is nomask:
            self.maskiter = None
        else:
            self.maskiter = ma._mask.flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        result = self.dataiter.__getitem__(indx).view(type(self.ma))
        if self.maskiter is not None:
            _mask = self.maskiter.__getitem__(indx)
            _mask.shape = result.shape
            result._mask = _mask
        return result

    ### This won't work is ravel makes a copy
    def __setitem__(self, index, value):
        self.dataiter[index] = getdata(value)
        if self.maskiter is not None:
            self.maskiter[index] = getmaskarray(value)
#        self.ma1d[index] = value

    def next(self):
        "Returns the next element of the iterator."
        d = self.dataiter.next()
        if self.maskiter is not None and self.maskiter.next():
            d = masked
        return d


def flatten_structured_array(a):
    """
    Flatten a strutured array.

    The datatype of the output is the largest datatype of the (nested) fields.

    Returns
    -------
    output : var
        Flatten MaskedArray if the input is a MaskedArray,
        standard ndarray otherwise.

    Examples
    --------
    >>> ndtype = [('a', int), ('b', float)]
    >>> a = np.array([(1, 1), (2, 2)], dtype=ndtype)
    >>> flatten_structured_array(a)
    array([[1., 1.],
           [2., 2.]])

    """
    #
    def flatten_sequence(iterable):
        """Flattens a compound of nested iterables."""
        for elm in iter(iterable):
            if hasattr(elm,'__iter__'):
                for f in flatten_sequence(elm):
                    yield f
            else:
                yield elm
    #
    a = np.asanyarray(a)
    inishape = a.shape
    a = a.ravel()
    if isinstance(a, MaskedArray):
        out = np.array([tuple(flatten_sequence(d.item())) for d in a._data])
        out = out.view(MaskedArray)
        out._mask = np.array([tuple(flatten_sequence(d.item()))
                              for d in getmaskarray(a)])
    else:
        out = np.array([tuple(flatten_sequence(d.item())) for d in a])
    if len(inishape) > 1:
        newshape = list(out.shape)
        newshape[0] = inishape
        out.shape = tuple(flatten_sequence(newshape))
    return out




class MaskedArray(ndarray):
    """
    Arrays with possibly masked values.  Masked values of True
    exclude the corresponding element from any computation.

    Construction:
        x = MaskedArray(data, mask=nomask, dtype=None, copy=True,
        fill_value=None, keep_mask=True, hard_mask=False, shrink=True)

    Parameters
    ----------
    data : {var}
        Input data.
    mask : {nomask, sequence}, optional
        Mask.  Must be convertible to an array of booleans with
        the same shape as data: True indicates a masked (eg.,
        invalid) data.
    dtype : {dtype}, optional
        Data type of the output.
        If dtype is None, the type of the data argument (`data.dtype`) is used.
        If dtype is not None and different from `data.dtype`, a copy is performed.
    copy : {False, True}, optional
        Whether to copy the input data (True), or to use a reference instead.
        Note: data are NOT copied by default.
    subok : {True, False}, optional
        Whether to return a subclass of MaskedArray (if possible)
        or a plain MaskedArray.
    ndmin : {0, int}, optional
        Minimum number of dimensions
    fill_value : {var}, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the datatype is used.
    keep_mask : {True, boolean}, optional
        Whether to combine mask with the mask of the input data,
        if any (True), or to use only mask for the output (False).
    hard_mask : {False, boolean}, optional
        Whether to use a hard mask or not.
        With a hard mask, masked values cannot be unmasked.
    shrink : {True, boolean}, optional
        Whether to force compression of an empty mask.

    """

    __array_priority__ = 15
    _defaultmask = nomask
    _defaulthardmask = False
    _baseclass = ndarray

    def __new__(cls, data=None, mask=nomask, dtype=None, copy=False,
                subok=True, ndmin=0, fill_value=None,
                keep_mask=True, hard_mask=None, flag=None, shrink=True,
                **options):
        """
    Create a new masked array from scratch.

    Notes
    -----
    A masked array can also be created by taking a .view(MaskedArray).

        """
        if flag is not None:
            warnings.warn("The flag 'flag' is now called 'shrink'!",
                          DeprecationWarning)
            shrink = flag
        # Process data............
        _data = np.array(data, dtype=dtype, copy=copy, subok=True, ndmin=ndmin)
        _baseclass = getattr(data, '_baseclass', type(_data))
        # Check that we're not erasing the mask..........
        if isinstance(data, MaskedArray) and (data.shape != _data.shape):
            copy = True
        # Careful, cls might not always be MaskedArray...
        if not isinstance(data, cls) or not subok:
            _data = ndarray.view(_data, cls)
        else:
            _data = ndarray.view(_data, type(data))
        # Backwards compatibility w/ numpy.core.ma .......
        if hasattr(data,'_mask') and not isinstance(data, ndarray):
            _data._mask = data._mask
            _sharedmask = True
        # Process mask ...............................
        # Number of named fields (or zero if none)
        names_ = _data.dtype.names or ()
        # Type of the mask
        if names_:
            mdtype = make_mask_descr(_data.dtype)
        else:
            mdtype = MaskType
        # Case 1. : no mask in input ............
        if mask is nomask:
             # Erase the current mask ?
            if not keep_mask:
                # With a reduced version
                if shrink:
                    _data._mask = nomask
                # With full version
                else:
                    _data._mask = np.zeros(_data.shape, dtype=mdtype)
            # Check whether we missed something
            elif isinstance(data, (tuple,list)):
                try:
                    # If data is a sequence of masked array
                    mask = np.array([getmaskarray(m) for m in data],
                                    dtype=mdtype)
                except ValueError:
                    # If data is nested
                    mask = nomask
                # Force shrinking of the mask if needed (and possible)
                if (mdtype == MaskType) and mask.any():
                    _data._mask = mask
                    _data._sharedmask = False
            else:
                if copy:
                    _data._mask = _data._mask.copy()
                    _data._sharedmask = False
                    # Reset the shape of the original mask
                    if getmask(data) is not nomask:
                        data._mask.shape = data.shape
                else:
                    _data._sharedmask = True
        # Case 2. : With a mask in input ........
        else:
            # Read the mask with the current mdtype
            try:
                mask = np.array(mask, copy=copy, dtype=mdtype)
            # Or assume it's a sequence of bool/int
            except TypeError:
                mask = np.array([tuple([m]*len(mdtype)) for m in mask],
                                 dtype=mdtype)
            # Make sure the mask and the data have the same shape
            if mask.shape != _data.shape:
                (nd, nm) = (_data.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, _data.shape)
                elif nm == nd:
                    mask = np.reshape(mask, _data.shape)
                else:
                    msg = "Mask and data not compatible: data size is %i, "+\
                          "mask size is %i."
                    raise MaskError, msg % (nd, nm)
                copy = True
            # Set the mask to the new value
            if _data._mask is nomask:
                _data._mask = mask
                _data._sharedmask = not copy
            else:
                if not keep_mask:
                    _data._mask = mask
                    _data._sharedmask = not copy
                else:
                    if names_:
                        def _recursive_or(a, b):
                            "do a|=b on each field of a, recursively"
                            for name in a.dtype.names:
                                (af, bf) = (a[name], b[name])
                                if af.dtype.names:
                                    _recursive_or(af, bf)
                                else:
                                    af |= bf
                            return
                        _recursive_or(_data._mask, mask)
                    else:
                        _data._mask = np.logical_or(mask, _data._mask)
                    _data._sharedmask = False
        # Update fill_value.......
        if fill_value is None:
            fill_value = getattr(data, '_fill_value', None)
        # But don't run the check unless we have something to check....
        if fill_value is not None:
            _data._fill_value = _check_fill_value(fill_value, _data.dtype)
        # Process extra options ..
        if hard_mask is None:
            _data._hardmask = getattr(data, '_hardmask', False)
        else:
            _data._hardmask = hard_mask
        _data._baseclass = _baseclass
        return _data
    #
    def _update_from(self, obj):
        """Copies some attributes of obj to self.
        """
        if obj is not None and isinstance(obj, ndarray):
            _baseclass = type(obj)
        else:
            _baseclass = ndarray
        # We need to copy the _basedict to avoid backward propagation
        _optinfo = {}
        _optinfo.update(getattr(obj, '_optinfo', {}))
        _optinfo.update(getattr(obj, '_basedict', {}))
        if not isinstance(obj, MaskedArray):
            _optinfo.update(getattr(obj, '__dict__', {}))
        _dict = dict(_fill_value=getattr(obj, '_fill_value', None),
                     _hardmask=getattr(obj, '_hardmask', False),
                     _sharedmask=getattr(obj, '_sharedmask', False),
                     _isfield=getattr(obj, '_isfield', False),
                     _baseclass=getattr(obj,'_baseclass', _baseclass),
                     _optinfo=_optinfo,
                     _basedict=_optinfo)
        self.__dict__.update(_dict)
        self.__dict__.update(_optinfo)
        return
    #........................
    def __array_finalize__(self, obj):
        """Finalizes the masked array.
        """
        # Get main attributes .........
        self._update_from(obj)
        if isinstance(obj, ndarray):
            odtype = obj.dtype
            if odtype.names:
                _mask = getattr(obj, '_mask', make_mask_none(obj.shape, odtype))
            else:
                _mask = getattr(obj, '_mask', nomask)
        else:
            _mask = nomask
        self._mask = _mask
        # Finalize the mask ...........
        if self._mask is not nomask:
            self._mask.shape = self.shape
        return
    #..................................
    def __array_wrap__(self, obj, context=None):
        """
        Special hook for ufuncs.
        Wraps the numpy array and sets the mask according to context.
        """
        result = obj.view(type(self))
        result._update_from(self)
        #..........
        if context is not None:
            result._mask = result._mask.copy()
            (func, args, _) = context
            m = reduce(mask_or, [getmaskarray(arg) for arg in args])
            # Get the domain mask................
            domain = ufunc_domain.get(func, None)
            if domain is not None:
                # Take the domain, and make sure it's a ndarray
                if len(args) > 2:
                    d = filled(reduce(domain, args), True)
                else:
                    d = filled(domain(*args), True)
                # Fill the result where the domain is wrong
                try:
                    # Binary domain: take the last value
                    fill_value = ufunc_fills[func][-1]
                except TypeError:
                    # Unary domain: just use this one
                    fill_value = ufunc_fills[func]
                except KeyError:
                    # Domain not recognized, use fill_value instead
                    fill_value = self.fill_value
                result = result.copy()
                np.putmask(result, d, fill_value)
                # Update the mask
                if m is nomask:
                    if d is not nomask:
                        m = d
                else:
                    # Don't modify inplace, we risk back-propagation
                    m = (m | d)
            # Make sure the mask has the proper size
            if result.shape == () and m:
                return masked
            else:
                result._mask = m
                result._sharedmask = False
        #....
        return result
    #.............................................
    def view(self, dtype=None, type=None):
        if dtype is None:
            if type is None:
                output = ndarray.view(self)
            else:
                output = ndarray.view(self, type)
        elif type is None:
            try:
                if issubclass(dtype, ndarray):
                    output = ndarray.view(self, dtype)
                    dtype = None
                else:
                    output = ndarray.view(self, dtype)
            except TypeError:
                output = ndarray.view(self, dtype)
        else:
            output = ndarray.view(self, dtype, type)
        # Should we update the mask ?
        if (getattr(output, '_mask', nomask) is not nomask):
            if dtype is None:
                dtype = output.dtype
            mdtype = make_mask_descr(dtype)

            output._mask = self._mask.view(mdtype, ndarray)
            output._mask.shape = output.shape
        # Make sure to reset the _fill_value if needed
        if getattr(output, '_fill_value', None):
            output._fill_value = None
        return output
    view.__doc__ = ndarray.view.__doc__
    #.............................................
    def astype(self, newtype):
        """
        Returns a copy of the MaskedArray cast to given newtype.

        Returns
        -------
        output : MaskedArray
            A copy of self cast to input newtype.
            The returned record shape matches self.shape.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3.1],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> print x
        [[1.0 -- 3.1]
         [-- 5.0 --]
         [7.0 -- 9.0]]
        >>> print x.astype(int32)
        [[1 -- 3]
         [-- 5 --]
         [7 -- 9]]

        """
        newtype = np.dtype(newtype)
        output = self._data.astype(newtype).view(type(self))
        output._update_from(self)
        names = output.dtype.names
        if names is None:
            output._mask = self._mask.astype(bool)
        else:
            if self._mask is nomask:
                output._mask = nomask
            else:
                output._mask = self._mask.astype([(n, bool) for n in names])
        # Don't check _fill_value if it's None, that'll speed things up
        if self._fill_value is not None:
            output._fill_value = _check_fill_value(self._fill_value, newtype)
        return output
    #.............................................
    def __getitem__(self, indx):
        """x.__getitem__(y) <==> x[y]

        Return the item described by i, as a masked array.

        """
        # This test is useful, but we should keep things light...
#        if getmask(indx) is not nomask:
#            msg = "Masked arrays must be filled before they can be used as indices!"
#            raise IndexError, msg
        dout = ndarray.__getitem__(ndarray.view(self, ndarray), indx)
        # We could directly use ndarray.__getitem__ on self...
        # But then we would have to modify __array_finalize__ to prevent the
        # mask of being reshaped if it hasn't been set up properly yet...
        # So it's easier to stick to the current version
        _mask = self._mask
        if not getattr(dout, 'ndim', False):
            # A record ................
            if isinstance(dout, np.void):
                mask = _mask[indx]
                if flatten_mask(mask).any():
                    dout = masked_array(dout, mask=mask)
                else:
                    return dout
            # Just a scalar............
            elif _mask is not nomask and _mask[indx]:
                return masked
        else:
            # Force dout to MA ........
            dout = dout.view(type(self))
            # Inherit attributes from self
            dout._update_from(self)
            # Check the fill_value ....
            if isinstance(indx, basestring):
                if self._fill_value is not None:
                    dout._fill_value = self._fill_value[indx]
                dout._isfield = True
            # Update the mask if needed
            if _mask is not nomask:
                dout._mask = _mask[indx]
                dout._sharedmask = True
#               Note: Don't try to check for m.any(), that'll take too long...
        return dout
    #........................
    def __setitem__(self, indx, value):
        """x.__setitem__(i, y) <==> x[i]=y

        Set item described by index. If value is masked, masks those
        locations.

        """
        if self is masked:
            raise MaskError, 'Cannot alter the masked element.'
        # This test is useful, but we should keep things light...
#        if getmask(indx) is not nomask:
#            msg = "Masked arrays must be filled before they can be used as indices!"
#            raise IndexError, msg
        if isinstance(indx, basestring):
            ndarray.__setitem__(self._data, indx, value)
            if self._mask is nomask:
                self._mask = make_mask_none(self.shape, self.dtype)
            ndarray.__setitem__(self._mask, indx, getmask(value))
            return
        #........................................
#        ndgetattr = ndarray.__getattribute__
        _data = self._data
        _dtype = ndarray.__getattribute__(_data,'dtype')
        _mask = ndarray.__getattribute__(self,'_mask')
        nbfields = len(_dtype.names or ())
        #........................................
        if value is masked:
            # The mask wasn't set: create a full version...
            if _mask is nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
            # Now, set the mask to its value.
            if nbfields:
                _mask[indx] = tuple([True] * nbfields)
            else:
                _mask[indx] = True
            if not self._isfield:
                self._sharedmask = False
            return
        #........................................
        # Get the _data part of the new value
        dval = value
        # Get the _mask part of the new value
        mval = getattr(value, '_mask', nomask)
        if nbfields and mval is nomask:
            mval = tuple([False] * nbfields)
        if _mask is nomask:
            # Set the data, then the mask
            ndarray.__setitem__(_data, indx, dval)
            if mval is not nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
                ndarray.__setitem__(_mask, indx, mval)
        elif not self._hardmask:
            # Unshare the mask if necessary to avoid propagation
            if not self._isfield:
                self.unshare_mask()
                _mask = ndarray.__getattribute__(self,'_mask')
            # Set the data, then the mask
            ndarray.__setitem__(_data, indx, dval)
            ndarray.__setitem__(_mask, indx, mval)
        elif hasattr(indx, 'dtype') and (indx.dtype==MaskType):
            indx = indx * umath.logical_not(_mask)
            ndarray.__setitem__(_data, indx, dval)
        else:
            if nbfields:
                err_msg = "Flexible 'hard' masks are not yet supported..."
                raise NotImplementedError(err_msg)
            mindx = mask_or(_mask[indx], mval, copy=True)
            dindx = self._data[indx]
            if dindx.size > 1:
                dindx[~mindx] = dval
            elif mindx is nomask:
                dindx = dval
            ndarray.__setitem__(_data, indx, dindx)
            _mask[indx] = mindx
        return


    def __getslice__(self, i, j):
        """x.__getslice__(i, j) <==> x[i:j]

        Return the slice described by (i, j).  The use of negative
        indices is not supported.

        """
        return self.__getitem__(slice(i, j))


    def __setslice__(self, i, j, value):
        """x.__setslice__(i, j, value) <==> x[i:j]=value

    Set the slice (i,j) of a to value. If value is masked, mask
    those locations.

        """
        self.__setitem__(slice(i, j), value)


    def __setmask__(self, mask, copy=False):
        """Set the mask.

        """
        idtype = ndarray.__getattribute__(self,'dtype')
        current_mask = ndarray.__getattribute__(self,'_mask')
        if mask is masked:
            mask = True
        # Make sure the mask is set
        if (current_mask is nomask):
            # Just don't do anything is there's nothing to do...
            if mask is nomask:
                return
            current_mask = self._mask = make_mask_none(self.shape, idtype)
        # No named fields.........
        if idtype.names is None:
            # Hardmask: don't unmask the data
            if self._hardmask:
                current_mask |= mask
            # Softmask: set everything to False
            else:
                current_mask.flat = mask
        # Named fields w/ ............
        else:
            mdtype = current_mask.dtype
            mask = np.array(mask, copy=False)
            # Mask is a singleton
            if not mask.ndim:
                # It's a boolean : make a record
                if mask.dtype.kind == 'b':
                    mask = np.array(tuple([mask.item()]*len(mdtype)),
                                    dtype=mdtype)
                # It's a record: make sure the dtype is correct
                else:
                    mask = mask.astype(mdtype)
            # Mask is a sequence
            else:
                # Make sure the new mask is a ndarray with the proper dtype
                try:
                    mask = np.array(mask, copy=copy, dtype=mdtype)
                # Or assume it's a sequence of bool/int
                except TypeError:
                    mask = np.array([tuple([m]*len(mdtype)) for m in mask],
                                    dtype=mdtype)
            # Hardmask: don't unmask the data
            if self._hardmask:
                for n in idtype.names:
                    current_mask[n] |= mask[n]
            # Softmask: set everything to False
            else:
                current_mask.flat = mask
        # Reshape if needed
        if current_mask.shape:
            current_mask.shape = self.shape
        return
    _set_mask = __setmask__
    #....
    def _get_mask(self):
        """Return the current mask.

        """
        # We could try to force a reshape, but that wouldn't work in some cases.
#        return self._mask.reshape(self.shape)
        return self._mask
    mask = property(fget=_get_mask, fset=__setmask__, doc="Mask")


    def _get_recordmask(self):
        """
    Return the mask of the records.
    A record is masked when all the fields are masked.

        """
        _mask = ndarray.__getattribute__(self, '_mask').view(ndarray)
        if _mask.dtype.names is None:
            return _mask
        return np.all(flatten_structured_array(_mask), axis=-1)


    def _set_recordmask(self):
        """Return the mask of the records.
    A record is masked when all the fields are masked.

        """
        raise NotImplementedError("Coming soon: setting the mask per records!")
    recordmask = property(fget=_get_recordmask)

    #............................................
    def harden_mask(self):
        """Force the mask to hard.

        """
        self._hardmask = True

    def soften_mask(self):
        """Force the mask to soft.

        """
        self._hardmask = False

    hardmask = property(fget=lambda self: self._hardmask,
                        doc="Hardness of the mask")


    def unshare_mask(self):
        """Copy the mask and set the sharedmask flag to False.

        """
        if self._sharedmask:
            self._mask = self._mask.copy()
            self._sharedmask = False

    sharedmask = property(fget=lambda self: self._sharedmask,
                          doc="Share status of the mask (read-only).")

    def shrink_mask(self):
        """Reduce a mask to nomask when possible.

        """
        m = self._mask
        if m.ndim and not m.any():
            self._mask = nomask

    #............................................

    baseclass = property(fget= lambda self:self._baseclass,
                         doc="Class of the underlying data (read-only).")
    
    def _get_data(self):
        """Return the current data, as a view of the original
        underlying data.

        """
        return ndarray.view(self, self._baseclass)
    _data = property(fget=_get_data)
    data = property(fget=_get_data)

    def raw_data(self):
        """Return the _data part of the MaskedArray.

        DEPRECATED: You should really use ``.data`` instead...

        """
        warnings.warn('Use .data instead.', DeprecationWarning)
        return self._data
    #............................................
    def _get_flat(self):
        """Return a flat iterator.

        """
        return MaskedIterator(self)
    #
    def _set_flat (self, value):
        """Set a flattened version of self to value.

        """
        y = self.ravel()
        y[:] = value
    #
    flat = property(fget=_get_flat, fset=_set_flat,
                    doc="Flat version of the array.")
    #............................................
    def get_fill_value(self):
        """Return the filling value.

        """
        if self._fill_value is None:
            self._fill_value = _check_fill_value(None, self.dtype)
        return self._fill_value

    def set_fill_value(self, value=None):
        """Set the filling value to value.

        If value is None, use a default based on the data type.

        """
        self._fill_value = _check_fill_value(value, self.dtype)

    fill_value = property(fget=get_fill_value, fset=set_fill_value,
                          doc="Filling value.")


    def filled(self, fill_value=None):
        """
    Return a copy of self, where masked values are filled with `fill_value`.

    If `fill_value` is None, `self.fill_value` is used instead.

    Notes
    -----
    + Subclassing is preserved
    + The result is NOT a MaskedArray !

    Examples
    --------
    >>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
    >>> x.filled()
    array([1,2,-999,4,-999])
    >>> type(x.filled())
    <type 'numpy.ndarray'>

        """
        m = self._mask
        if m is nomask:
            return self._data
        #
        if fill_value is None:
            fill_value = self.fill_value
        else:
            fill_value = _check_fill_value(fill_value, self.dtype)
        #
        if self is masked_singleton:
            return np.asanyarray(fill_value)
        #
        if m.dtype.names:
            result = self._data.copy()
            _recursive_filled(result, self._mask, fill_value)
        elif not m.any():
            return self._data
        else:
            result = self._data.copy()
            try:
                np.putmask(result, m, fill_value)
            except (TypeError, AttributeError):
                fill_value = narray(fill_value, dtype=object)
                d = result.astype(object)
                result = np.choose(m, (d, fill_value))
            except IndexError:
                #ok, if scalar
                if self._data.shape:
                    raise
                elif m:
                    result = np.array(fill_value, dtype=self.dtype)
                else:
                    result = self._data
        return result

    def compressed(self):
        """
        Return a 1-D array of all the non-masked data.

        Returns
        -------
        data : ndarray.
            A new ndarray holding the non-masked data is returned.

        Notes
        -----
        + The result is NOT a MaskedArray !

        Examples
        --------
        >>> x = array(arange(5), mask=[0]+[1]*4)
        >>> print x.compressed()
        [0]
        >>> print type(x.compressed())
        <type 'numpy.ndarray'>

        """
        data = ndarray.ravel(self._data)
        if self._mask is not nomask:
            data = data.compress(np.logical_not(ndarray.ravel(self._mask)))
        return data


    def compress(self, condition, axis=None, out=None):
        """
        Return `a` where condition is ``True``.

        If condition is a `MaskedArray`, missing values are considered
        as ``False``.

        Parameters
        ----------
        condition : var
            Boolean 1-d array selecting which entries to return. If len(condition)
            is less than the size of a along the axis, then output is truncated
            to length of condition array.
        axis : {None, int}, optional
            Axis along which the operation must be performed.
        out : {None, ndarray}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

        Returns
        -------
        result : MaskedArray
            A :class:`MaskedArray` object.

        Notes
        -----
        Please note the difference with :meth:`compressed` !
        The output of :meth:`compress` has a mask, the output of
        :meth:`compressed` does not.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> print x
        [[1 -- 3]
         [-- 5 --]
         [7 -- 9]]
        >>> x.compress([1, 0, 1])
        masked_array(data = [1 3],
              mask = [False False],
              fill_value=999999)

        >>> x.compress([1, 0, 1], axis=1)
        masked_array(data =
         [[1 3]
         [-- --]
         [7 9]],
              mask =
         [[False False]
         [ True  True]
         [False False]],
              fill_value=999999)

        """
        # Get the basic components
        (_data, _mask) = (self._data, self._mask)
        # Force the condition to a regular ndarray (forget the missing values...)
        condition = np.array(condition, copy=False, subok=False)
        #
        _new = _data.compress(condition, axis=axis, out=out).view(type(self))
        _new._update_from(self)
        if _mask is not nomask:
            _new._mask = _mask.compress(condition, axis=axis)
        return _new

    #............................................
    def __str__(self):
        """String representation.

        """
        if masked_print_option.enabled():
            f = masked_print_option
            if self is masked:
                return str(f)
            m = self._mask
            if m is nomask:
                res = self._data
            else:
                if m.shape == ():
                    if m.dtype.names:
                        m = m.view((bool, len(m.dtype)))
                        if m.any():
                            r = np.array(self._data.tolist(), dtype=object)
                            np.putmask(r, m, f)
                            return str(tuple(r))
                        else:
                            return str(self._data)
                    elif m:
                        return str(f)
                    else:
                        return str(self._data)
                # convert to object array to make filled work
                names = self.dtype.names
                if names is None:
                    res = self._data.astype("|O8")
                    res[m] = f
                else:
                    rdtype = _recursive_make_descr(self.dtype, "|O8")
                    res = self._data.astype(rdtype)
                    _recursive_printoption(res, m, f)
        else:
            res = self.filled(self.fill_value)
        return str(res)

    def __repr__(self):
        """Literal string representation.

        """
        n = len(self.shape)
        name = repr(self._data).split('(')[0]
        parameters =  dict(name=name, nlen=" "*len(name),
                           data=str(self), mask=str(self._mask),
                           fill=str(self.fill_value), dtype=str(self.dtype))
        if self.dtype.names:
            if n <= 1:
                return _print_templates['short_flx'] % parameters
            return  _print_templates['long_flx'] % parameters
        elif n <= 1:
            return _print_templates['short'] % parameters
        return _print_templates['long'] % parameters
    #............................................
    def __eq__(self, other):
        "Check whether other equals self elementwise"
        omask = getattr(other, '_mask', nomask)
        if omask is nomask:
            check = ndarray.__eq__(self.filled(0), other).view(type(self))
            check._mask = self._mask
        else:
            odata = filled(other, 0)
            check = ndarray.__eq__(self.filled(0), odata).view(type(self))
            if self._mask is nomask:
                check._mask = omask
            else:
                mask = mask_or(self._mask, omask)
                if mask.dtype.names:
                    if mask.size > 1:
                        axis = 1
                    else:
                        axis = None
                    try:
                        mask = mask.view((bool_, len(self.dtype))).all(axis)
                    except ValueError:
                        mask =  np.all([[f[n].all() for n in mask.dtype.names]
                                        for f in mask], axis=axis)
                check._mask = mask
        return check
    #
    def __ne__(self, other):
        "Check whether other doesn't equal self elementwise"
        omask = getattr(other, '_mask', nomask)
        if omask is nomask:
            check = ndarray.__ne__(self.filled(0), other).view(type(self))
            check._mask = self._mask
        else:
            odata = filled(other, 0)
            check = ndarray.__ne__(self.filled(0), odata).view(type(self))
            if self._mask is nomask:
                check._mask = omask
            else:
                mask = mask_or(self._mask, omask)
                if mask.dtype.names:
                    if mask.size > 1:
                        axis = 1
                    else:
                        axis = None
                    try:
                        mask = mask.view((bool_, len(self.dtype))).all(axis)
                    except ValueError:
                        mask =  np.all([[f[n].all() for n in mask.dtype.names]
                                        for f in mask], axis=axis)
                check._mask = mask
        return check
    #
    def __add__(self, other):
        "Add other to self, and return a new masked array."
        return add(self, other)
    #
    def __radd__(self, other):
        "Add other to self, and return a new masked array."
        return add(self, other)
    #
    def __sub__(self, other):
        "Subtract other to self, and return a new masked array."
        return subtract(self, other)
    #
    def __rsub__(self, other):
        "Subtract other to self, and return a new masked array."
        return subtract(other, self)
    #
    def __mul__(self, other):
        "Multiply other by self, and return a new masked array."
        return multiply(self, other)
    #
    def __rmul__(self, other):
        "Multiply other by self, and return a new masked array."
        return multiply(self, other)
    #
    def __div__(self, other):
        "Divide other into self, and return a new masked array."
        return divide(self, other)
    #
    def __truediv__(self, other):
        "Divide other into self, and return a new masked array."
        return true_divide(self, other)
    #
    def __floordiv__(self, other):
        "Divide other into self, and return a new masked array."
        return floor_divide(self, other)
    #
    def __pow__(self, other):
        "Raise self to the power other, masking the potential NaNs/Infs"
        return power(self, other)
    #............................................
    def __iadd__(self, other):
        "Add other to self in-place."
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        else:
            if m is not nomask:
                self._mask += m
        ndarray.__iadd__(self._data, np.where(self._mask, 0, getdata(other)))
        return self
    #....
    def __isub__(self, other):
        "Subtract other from self in-place."
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        ndarray.__isub__(self._data, np.where(self._mask, 0, getdata(other)))
        return self
    #....
    def __imul__(self, other):
        "Multiply self by other in-place."
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        ndarray.__imul__(self._data, np.where(self._mask, 1, getdata(other)))
        return self
    #....
    def __idiv__(self, other):
        "Divide self by other in-place."
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        # The following 3 lines control the domain filling
        if dom_mask.any():
            (_, fval) = ufunc_fills[np.divide]
            other_data = np.where(dom_mask, fval, other_data)
#        self._mask = mask_or(self._mask, new_mask)
        self._mask |= new_mask
        ndarray.__idiv__(self._data, np.where(self._mask, 1, other_data))
        return self
    #...
    def __ipow__(self, other):
        "Raise self to the power other, in place."
        other_data = getdata(other)
        other_mask = getmask(other)
        ndarray.__ipow__(self._data, np.where(self._mask, 1, other_data))
        invalid = np.logical_not(np.isfinite(self._data))
        if invalid.any():
            if self._mask is not nomask:
                self._mask |= invalid
            else:
                self._mask = invalid
            np.putmask(self._data, invalid, self.fill_value)
        new_mask = mask_or(other_mask, invalid)
        self._mask = mask_or(self._mask, new_mask)
        return self
    #............................................
    def __float__(self):
        "Convert to float."
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted "\
                            "to Python scalars")
        elif self._mask:
            warnings.warn("Warning: converting a masked element to nan.")
            return np.nan
        return float(self.item())

    def __int__(self):
        "Convert to int."
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted "\
                            "to Python scalars")
        elif self._mask:
            raise MaskError, 'Cannot convert masked element to a Python int.'
        return int(self.item())
    #............................................
    def get_imag(self):
        "Returns the imaginary part."
        result = self._data.imag.view(type(self))
        result.__setmask__(self._mask)
        return result
    imag = property(fget=get_imag, doc="Imaginary part.")

    def get_real(self):
        "Returns the real part."
        result = self._data.real.view(type(self))
        result.__setmask__(self._mask)
        return result
    real = property(fget=get_real, doc="Real part")


    #............................................
    def count(self, axis=None):
        """
        Count the non-masked elements of the array along the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to count the non-masked elements. If axis is None,
            all the non masked elements are counted.

        Returns
        -------
        result : MaskedArray
            A masked array where the mask is True where all data are
            masked.  If axis is None, returns either a scalar ot the
            masked singleton if all values are masked.

        """
        m = self._mask
        s = self.shape
        ls = len(s)
        if m is nomask:
            if ls == 0:
                return 1
            if ls == 1:
                return s[0]
            if axis is None:
                return self.size
            else:
                n = s[axis]
                t = list(s)
                del t[axis]
                return np.ones(t) * n
        n1 = np.size(m, axis)
        n2 = m.astype(int).sum(axis)
        if axis is None:
            return (n1-n2)
        else:
            return narray(n1 - n2)
    #............................................
    flatten = _arraymethod('flatten')
    #
    def ravel(self):
        """
        Returns a 1D version of self, as a view.

        Returns
        -------
        MaskedArray
            Output view is of shape ``(self.size,)`` (or
            ``(np.ma.product(self.shape),)``).

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> print x
        [[1 -- 3]
         [-- 5 --]
         [7 -- 9]]
        >>> print x.ravel()
        [1 -- 3 -- 5 -- 7 -- 9]

        """
        r = ndarray.ravel(self._data).view(type(self))
        r._update_from(self)
        if self._mask is not nomask:
            r._mask = ndarray.ravel(self._mask).reshape(r.shape)
        else:
            r._mask = nomask
        return r
    #
    repeat = _arraymethod('repeat')
    #
    def reshape (self, *s, **kwargs):
        """
        Returns a masked array containing the data of a, but with a new shape.
        The result is a view to the original array; if this is not possible,
        a ValueError is raised.

        Parameters
        ----------
        shape : shape tuple or int
           The new shape should be compatible with the original shape. If an
           integer, then the result will be a 1D array of that length.
        order : {'C', 'F'}, optional
            Determines whether the array data should be viewed as in C
            (row-major) order or FORTRAN (column-major) order.

        Returns
        -------
        reshaped_array : array
            A new view to the array.

        Notes
        -----
        If you want to modify the shape in place, please use ``a.shape = s``

        Examples
        --------
        >>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
        >>> print x
        [[-- 2]
         [3 --]]
        >>> x = x.reshape((4,1))
        >>> print x
        [[--]
         [2]
         [3]
         [--]]

        """
        kwargs.update(order=kwargs.get('order','C'))
        result = self._data.reshape(*s, **kwargs).view(type(self))
        result._update_from(self)
        mask = self._mask
        if mask is not nomask:
            result._mask = mask.reshape(*s, **kwargs)
        return result
    #
    def resize(self, newshape, refcheck=True, order=False):
        """
    Change shape and size of array in-place.

        """
        # Note : the 'order' keyword looks broken, let's just drop it
#        try:
#            ndarray.resize(self, newshape, refcheck=refcheck)
#            if self.mask is not nomask:
#                self._mask.resize(newshape, refcheck=refcheck)
#        except ValueError:
#            raise ValueError("Cannot resize an array that has been referenced "
#                             "or is referencing another array in this way.\n"
#                             "Use the numpy.ma.resize function.")
#        return None
        errmsg = "A masked array does not own its data "\
                 "and therefore cannot be resized.\n" \
                 "Use the numpy.ma.resize function instead."
        raise ValueError(errmsg)
    #
    def put(self, indices, values, mode='raise'):
        """
        Set storage-indexed locations to corresponding values.

        Sets self._data.flat[n] = values[n] for each n in indices.
        If `values` is shorter than `indices` then it will repeat.
        If `values` has some masked values, the initial mask is updated
        in consequence, else the corresponding values are unmasked.

        Parameters
        ----------
        indices : 1-D array_like
            Target indices, interpreted as integers.
        values : array_like
            Values to place in self._data copy at target indices.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            'raise' : raise an error.
            'wrap' : wrap around.
            'clip' : clip to the range.

        Notes
        -----
        `values` can be a scalar or length 1 array.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> print x
        [[1 -- 3]
         [-- 5 --]
         [7 -- 9]]
        >>> x.put([0,4,8],[10,20,30])
        >>> print x
        [[10 -- 3]
         [-- 20 --]
         [7 -- 30]]

        >>> x.put(4,999)
        >>> print x
        [[10 -- 3]
         [-- 999 --]
         [7 -- 30]]

        """
        m = self._mask
        # Hard mask: Get rid of the values/indices that fall on masked data
        if self._hardmask and self._mask is not nomask:
            mask = self._mask[indices]
            indices = narray(indices, copy=False)
            values = narray(values, copy=False, subok=True)
            values.resize(indices.shape)
            indices = indices[~mask]
            values = values[~mask]
        #....
        self._data.put(indices, values, mode=mode)
        #....
        if m is nomask:
            m = getmask(values)
        else:
            m = m.copy()
            if getmask(values) is nomask:
                m.put(indices, False, mode=mode)
            else:
                m.put(indices, values._mask, mode=mode)
            m = make_mask(m, copy=False, shrink=True)
        self._mask = m
    #............................................
    def ids (self):
        """Return the addresses of the data and mask areas."""
        if self._mask is nomask:
            return (self.ctypes.data, id(nomask))
        return (self.ctypes.data, self._mask.ctypes.data)

    def iscontiguous(self):
        "Is the data contiguous?"
        return self.flags['CONTIGUOUS']

    #............................................
    def all(self, axis=None, out=None):
        """
    Check if all of the elements of `a` are true.

    Performs a :func:`logical_and` over the given axis and returns the result.
    Masked values are considered as True during computation.
    For convenience, the output array is masked where ALL the values along the
    current axis are masked: if the output would have been a scalar and that
    all the values are masked, then the output is `masked`.

    Parameters
    ----------
    axis : {None, integer}
        Axis to perform the operation over.
        If None, perform over flattened array.
    out : {None, array}, optional
        Array into which the result can be placed. Its type is preserved
        and it must be of the right shape to hold the output.

    See Also
    --------
    all : equivalent function

    Examples
    --------
    >>> np.ma.array([1,2,3]).all()
    True
    >>> a = np.ma.array([1,2,3], mask=True)
    >>> (a.all() is np.ma.masked)
    True

        """
        mask = self._mask.all(axis)
        if out is None:
            d = self.filled(True).all(axis=axis).view(type(self))
            if d.ndim:
                d.__setmask__(mask)
            elif mask:
                return masked
            return d
        self.filled(True).all(axis=axis, out=out)
        if isinstance(out, MaskedArray):
            if out.ndim or mask:
                out.__setmask__(mask)
        return out


    def any(self, axis=None, out=None):
        """
        Check if any of the elements of `a` are true.

        Performs a logical_or over the given axis and returns the result.
        Masked values are considered as False during computation.

        Parameters
        ----------
        axis : {None, integer}
            Axis to perform the operation over.
            If None, perform over flattened array and return a scalar.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        See Also
        --------
        any : equivalent function

        """
        mask = self._mask.all(axis)
        if out is None:
            d = self.filled(False).any(axis=axis).view(type(self))
            if d.ndim:
                d.__setmask__(mask)
            elif mask:
                d = masked
            return d
        self.filled(False).any(axis=axis, out=out)
        if isinstance(out, MaskedArray):
            if out.ndim or mask:
                out.__setmask__(mask)
        return out


    def nonzero(self):
        """
    Return the indices of the elements of a that are not zero
    nor masked, as a tuple of arrays.

    There are as many tuples as dimensions of a, each tuple
    contains the indices of the non-zero elements in that
    dimension.  The corresponding non-zero values can be obtained
    with ``a[a.nonzero()]``.

    To group the indices by element, rather than dimension, use
    instead: ``transpose(a.nonzero())``.

    The result of this is always a 2d array, with a row for each
    non-zero element.

        """
        return narray(self.filled(0), copy=False).nonzero()


    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """
        (this docstring should be overwritten)
        """
        #!!!: implement out + test!
        m = self._mask
        if m is nomask:
            result = super(MaskedArray, self).trace(offset=offset, axis1=axis1,
                                                    axis2=axis2, out=out)
            return result.astype(dtype)
        else:
            D = self.diagonal(offset=offset, axis1=axis1, axis2=axis2)
            return D.astype(dtype).filled(0).sum(axis=None, out=out)
    trace.__doc__ = ndarray.trace.__doc__

    def sum(self, axis=None, dtype=None, out=None):
        """
        Return the sum of the array elements over the given axis.
        Masked elements are set to 0 internally.

        Parameters
        ----------
        axis : {None, -1, int}, optional
            Axis along which the sum is computed. The default
            (`axis` = None) is to compute over the flattened array.
        dtype : {None, dtype}, optional
            Determines the type of the returned array and of the accumulator
            where the elements are summed. If dtype has the value None and
            the type of a is an integer type of precision less than the default
            platform integer, then the default platform integer precision is
            used.  Otherwise, the dtype is the same as that of a.
        out :  {None, ndarray}, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary.

        Returns
        -------
        sum_along_axis : MaskedArray or scalar
            An array with the same shape as self, with the specified
            axis removed.   If self is a 0-d array, or if `axis` is None, a scalar
            is returned.  If an output array is specified, a reference to
            `out` is returned.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> print x
        [[1 -- 3]
         [-- 5 --]
         [7 -- 9]]
        >>> print x.sum()
        25
        >>> print x.sum(axis=1)
        [4 5 16]
        >>> print x.sum(axis=0)
        [8 5 12]
        >>> print type(x.sum(axis=0, dtype=np.int64)[0])
        <type 'numpy.int64'>

        """
        _mask = ndarray.__getattribute__(self, '_mask')
        newmask = _mask.all(axis=axis)
        # No explicit output
        if out is None:
            result = self.filled(0).sum(axis, dtype=dtype).view(type(self))
            if result.ndim:
                result.__setmask__(newmask)
            elif newmask:
                result = masked
            return result
        # Explicit output
        result = self.filled(0).sum(axis, dtype=dtype, out=out)
        if isinstance(out, MaskedArray):
            outmask = getattr(out, '_mask', nomask)
            if (outmask is nomask):
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        return out


    def cumsum(self, axis=None, dtype=None, out=None):
        """
    Return the cumulative sum of the elements along the given axis.
    The cumulative sum is calculated over the flattened array by
    default, otherwise over the specified axis.

    Masked values are set to 0 internally during the computation.
    However, their position is saved, and the result will be masked at
    the same locations.

    Parameters
    ----------
    axis : {None, -1, int}, optional
        Axis along which the sum is computed. The default (`axis` = None) is to
        compute over the flattened array. `axis` may be negative, in which case
        it counts from the   last to the first axis.
    dtype : {None, dtype}, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Warnings
    --------
    The mask is lost if out is not a valid :class:`MaskedArray` !

    Returns
    -------
    cumsum : ndarray.
        A new array holding the result is returned unless ``out`` is
        specified, in which case a reference to ``out`` is returned.

    Examples
    --------
    >>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
    >>> print marr.cumsum()
    [0 1 3 -- -- -- 9 16 24 33]


    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

        """
        result = self.filled(0).cumsum(axis=axis, dtype=dtype, out=out)
        if out is not None:
            if isinstance(out, MaskedArray):
                out.__setmask__(self.mask)
            return out
        result = result.view(type(self))
        result.__setmask__(self._mask)
        return result


    def prod(self, axis=None, dtype=None, out=None):
        """
        Return the product of the array elements over the given axis.
        Masked elements are set to 1 internally for computation.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis over which the product is taken. If None is used, then the
            product is over all the array elements.
        dtype : {None, dtype}, optional
            Determines the type of the returned array and of the accumulator
            where the elements are multiplied. If ``dtype`` has the value ``None``
            and the type of a is an integer type of precision less than the default
            platform integer, then the default platform integer precision is
            used.  Otherwise, the dtype is the same as that of a.
        out : {None, array}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

        Returns
        -------
        product_along_axis : {array, scalar}, see dtype parameter above.
            Returns an array whose shape is the same as a with the specified
            axis removed. Returns a 0d array when a is 1d or axis=None.
            Returns a reference to the specified output array if specified.

        See Also
        --------
        prod : equivalent function

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is raised
        on overflow.

        Examples
        --------
        >>> np.prod([1.,2.])
        2.0
        >>> np.prod([1.,2.], dtype=np.int32)
        2
        >>> np.prod([[1.,2.],[3.,4.]])
        24.0
        >>> np.prod([[1.,2.],[3.,4.]], axis=1)
        array([  2.,  12.])

        """
        _mask = ndarray.__getattribute__(self, '_mask')
        newmask = _mask.all(axis=axis)
        # No explicit output
        if out is None:
            result = self.filled(1).prod(axis, dtype=dtype).view(type(self))
            if result.ndim:
                result.__setmask__(newmask)
            elif newmask:
                result = masked
            return result
        # Explicit output
        result = self.filled(1).prod(axis, dtype=dtype, out=out)
        if isinstance(out, MaskedArray):
            outmask = getattr(out, '_mask', nomask)
            if (outmask is nomask):
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        return out

    product = prod

    def cumprod(self, axis=None, dtype=None, out=None):
        """
    Return the cumulative product of the elements along the given axis.
    The cumulative product is taken over the flattened array by
    default, otherwise over the specified axis.

    Masked values are set to 1 internally during the computation.
    However, their position is saved, and the result will be masked at
    the same locations.

    Parameters
    ----------
    axis : {None, -1, int}, optional
        Axis along which the product is computed. The default
        (`axis` = None) is to compute over the flattened array.
    dtype : {None, dtype}, optional
        Determines the type of the returned array and of the accumulator
        where the elements are multiplied. If ``dtype`` has the value ``None`` and
        the type of ``a`` is an integer type of precision less than the default
        platform integer, then the default platform integer precision is
        used.  Otherwise, the dtype is the same as that of ``a``.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Warnings
    --------
    The mask is lost if out is not a valid MaskedArray !

    Returns
    -------
    cumprod : ndarray
        A new array holding the result is returned unless out is specified,
        in which case a reference to out is returned.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    """
        result = self.filled(1).cumprod(axis=axis, dtype=dtype, out=out)
        if out is not None:
            if isinstance(out, MaskedArray):
                out.__setmask__(self._mask)
            return out
        result = result.view(type(self))
        result.__setmask__(self._mask)
        return result


    def mean(self, axis=None, dtype=None, out=None):
        """
    Returns the average of the array elements along given axis.
    Refer to `numpy.mean` for full documentation.

    See Also
    --------
    numpy.mean : equivalent function'
        """
        if self._mask is nomask:
            result = super(MaskedArray, self).mean(axis=axis, dtype=dtype)
        else:
            dsum = self.sum(axis=axis, dtype=dtype)
            cnt = self.count(axis=axis)
            result = dsum*1./cnt
        if out is not None:
            out.flat = result
            if isinstance(out, MaskedArray):
                outmask = getattr(out, '_mask', nomask)
                if (outmask is nomask):
                    outmask = out._mask = make_mask_none(out.shape)
                outmask.flat = getattr(result, '_mask', nomask)
            return out
        return result

    def anom(self, axis=None, dtype=None):
        """
    Return the anomalies (deviations from the average) along the given axis.

    Parameters
    ----------
    axis : int, optional
        Axis along which to perform the operation.
        If None, applies to a flattened version of the array.
    dtype : {dtype}, optional
        Datatype for the intermediary computation.
        If not given, the current dtype is used instead.

    """
        m = self.mean(axis, dtype)
        if not axis:
            return (self - m)
        else:
            return (self - expand_dims(m, axis))

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        ""
        # Easy case: nomask, business as usual
        if self._mask is nomask:
            return self._data.var(axis=axis, dtype=dtype, out=out, ddof=ddof)
        # Some data are masked, yay!
        cnt = self.count(axis=axis)-ddof
        danom = self.anom(axis=axis, dtype=dtype)
        if iscomplexobj(self):
            danom = umath.absolute(danom)**2
        else:
            danom *= danom
        dvar = divide(danom.sum(axis), cnt).view(type(self))
        # Apply the mask if it's not a scalar
        if dvar.ndim:
            dvar._mask = mask_or(self._mask.all(axis), (cnt<=ddof))
            dvar._update_from(self)
        elif getattr(dvar,'_mask', False):
        # Make sure that masked is returned when the scalar is masked.
            dvar = masked
            if out is not None:
                if isinstance(out, MaskedArray):
                    out.__setmask__(True)
                elif out.dtype.kind in 'biu':
                    errmsg = "Masked data information would be lost in one or "\
                             "more location."
                    raise MaskError(errmsg)
                else:
                    out.flat = np.nan
                return out
        # In case with have an explicit output
        if out is not None:
            # Set the data
            out.flat = dvar
            # Set the mask if needed
            if isinstance(out, MaskedArray):
                out.__setmask__(dvar.mask)
            return out
        return dvar
    var.__doc__ = np.var.__doc__


    def std(self, axis=None, dtype=None, out=None, ddof=0):
        ""
        dvar = self.var(axis=axis, dtype=dtype, out=out, ddof=ddof)
        if dvar is not masked:
            dvar = sqrt(dvar)
            if out is not None:
                out **= 0.5
                return out
        return dvar
    std.__doc__ = np.std.__doc__

    #............................................
    def round(self, decimals=0, out=None):
        """
        Return an array rounded a to the given number of decimals.

        Refer to `numpy.around` for full documentation.

        See Also
        --------
        numpy.around : equivalent function

        """
        result = self._data.round(decimals=decimals, out=out).view(type(self))
        result._mask = self._mask
        result._update_from(self)
        # No explicit output: we're done
        if out is None:
            return result
        if isinstance(out, MaskedArray):
            out.__setmask__(self._mask)
        return out
    round.__doc__ = ndarray.round.__doc__

    #............................................
    def argsort(self, axis=None, fill_value=None, kind='quicksort',
                order=None):
        """
    Return an ndarray of indices that sort the array along the
    specified axis.  Masked values are filled beforehand to
    fill_value.

    Parameters
    ----------
    axis : int, optional
        Axis along which to sort.  If not given, the flattened array is used.
    kind : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm.
    order : list, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  Not all fields need be
        specified.
    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort `a` along the specified axis.
        In other words, ``a[index_array]`` yields a sorted `a`.

    See Also
    --------
    sort : Describes sorting algorithms used.
    lexsort : Indirect stable sort with multiple keys.
    ndarray.sort : Inplace sort.

    Notes
    -----
    See `sort` for notes on the different sorting algorithms.

        """
        if fill_value is None:
            fill_value = default_fill_value(self)
        d = self.filled(fill_value).view(ndarray)
        return d.argsort(axis=axis, kind=kind, order=order)


    def argmin(self, axis=None, fill_value=None, out=None):
        """
        Return array of indices to the minimum values along the given axis.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : {var}, optional
            Value used to fill in the masked values.  If None, the output of
            minimum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        {ndarray, scalar}
            If multi-dimension input, returns a new ndarray of indices to the
            minimum values along the given axis.  Otherwise, returns a scalar
            of index to the minimum values along the given axis.

        Examples
        --------
        >>> x = np.ma.array(arange(4), mask=[1,1,0,0])
        >>> x.shape = (2,2)
        >>> print x
        [[-- --]
         [2 3]]
        >>> print x.argmin(axis=0, fill_value=-1)
        [0 0]
        >>> print x.argmin(axis=0, fill_value=9)
        [1 1]

        """
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        d = self.filled(fill_value).view(ndarray)
        return d.argmin(axis, out=out)


    def argmax(self, axis=None, fill_value=None, out=None):
        """
        Returns array of indices of the maximum values along the given axis.
        Masked values are treated as if they had the value fill_value.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : {var}, optional
            Value used to fill in the masked values.  If None, the output of
            maximum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        index_array : {integer_array}

        Examples
        --------
        >>> a = np.arange(6).reshape(2,3)
        >>> a.argmax()
        5
        >>> a.argmax(0)
        array([1, 1, 1])
        >>> a.argmax(1)
        array([2, 2])

        """
        if fill_value is None:
            fill_value = maximum_fill_value(self._data)
        d = self.filled(fill_value).view(ndarray)
        return d.argmax(axis, out=out)


    def sort(self, axis=-1, kind='quicksort', order=None,
             endwith=True, fill_value=None):
        """
    Return a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm. Default is 'quicksort'.
    order : list, optional
        When `a` is a structured array, this argument specifies which fields
        to compare first, second, and so on.  This list does not need to
        include all of the fields.
    endwith : {True, False}, optional
        Whether missing values (if any) should be forced in the upper indices
        (at the end of the array) (True) or lower indices (at the beginning).
    fill_value : {var}
        Value used to fill in the masked values.  If None, use
        the the output of minimum_fill_value().

    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as `a`.

    See Also
    --------
    ndarray.sort : Method to sort an array in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted array.

    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The three available algorithms have the following
    properties:

    =========== ======= ============= ============ =======
       kind      speed   worst case    work space  stable
    =========== ======= ============= ============ =======
    'quicksort'    1     O(n^2)            0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'heapsort'     3     O(n*log(n))       0          no
    =========== ======= ============= ============ =======

    All the sort algorithms make temporary copies of the data when
    sorting along any but the last axis.  Consequently, sorting along
    the last axis is faster and uses less space than sorting along
    any other axis.

    Examples
    --------
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
    ...           ('Galahad', 1.7, 38)]
    >>> a = np.array(values, dtype=dtype)       # create a structured array
    >>> np.sort(a, order='height')                        # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
           ('Lancelot', 1.8999999999999999, 38)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    Sort by age, then height if ages are equal:

    >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
           ('Arthur', 1.8, 41)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

        """
        if self._mask is nomask:
            ndarray.sort(self, axis=axis, kind=kind, order=order)
        else:
            if fill_value is None:
                if endwith:
                    filler = minimum_fill_value(self)
                else:
                    filler = maximum_fill_value(self)
            else:
                filler = fill_value
            idx = np.indices(self.shape)
            idx[axis] = self.filled(filler).argsort(axis=axis, kind=kind,
                                                    order=order)
            idx_l = idx.tolist()
            tmp_mask = self._mask[idx_l].flat
            tmp_data = self._data[idx_l].flat
            self._data.flat = tmp_data
            self._mask.flat = tmp_mask
        return

    #............................................
    def min(self, axis=None, out=None, fill_value=None):
        """
    Return the minimum along a given axis.

    Parameters
    ----------
    axis : {None, int}, optional
        Axis along which to operate.  By default, ``axis`` is None and the
        flattened input is used.
    out : array_like, optional
        Alternative output array in which to place the result.  Must be of
        the same shape and buffer length as the expected output.
    fill_value : {var}, optional
        Value used to fill in the masked values.
        If None, use the output of `minimum_fill_value`.

    Returns
    -------
    amin : array_like
        New array holding the result.
        If ``out`` was specified, ``out`` is returned.

    See Also
    --------
    minimum_fill_value
        Returns the minimum filling value for a given datatype.

        """
        _mask = ndarray.__getattribute__(self, '_mask')
        newmask = _mask.all(axis=axis)
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        # No explicit output
        if out is None:
            result = self.filled(fill_value).min(axis=axis, out=out).view(type(self))
            if result.ndim:
                # Set the mask
                result.__setmask__(newmask)
                # Get rid of Infs
                if newmask.ndim:
                    np.putmask(result, newmask, result.fill_value)
            elif newmask:
                result = masked
            return result
        # Explicit output
        result = self.filled(fill_value).min(axis=axis, out=out)
        if isinstance(out, MaskedArray):
            outmask = getattr(out, '_mask', nomask)
            if (outmask is nomask):
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        else:
            if out.dtype.kind in 'biu':
                errmsg = "Masked data information would be lost in one or more"\
                         " location."
                raise MaskError(errmsg)
            np.putmask(out, newmask, np.nan)
        return out

    def mini(self, axis=None):
        if axis is None:
            return minimum(self)
        else:
            return minimum.reduce(self, axis)

    #........................
    def max(self, axis=None, out=None, fill_value=None):
        """
        Return the maximum along a given axis.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to operate.  By default, ``axis`` is None and the
            flattened input is used.
        out : array_like, optional
            Alternative output array in which to place the result.  Must
            be of the same shape and buffer length as the expected output.
        fill_value : {var}, optional
            Value used to fill in the masked values.
            If None, use the output of maximum_fill_value().

        Returns
        -------
        amax : array_like
            New array holding the result.
            If ``out`` was specified, ``out`` is returned.

        See Also
        --------
        maximum_fill_value
            Returns the maximum filling value for a given datatype.

        """
        _mask = ndarray.__getattribute__(self, '_mask')
        newmask = _mask.all(axis=axis)
        if fill_value is None:
            fill_value = maximum_fill_value(self)
        # No explicit output
        if out is None:
            result = self.filled(fill_value).max(axis=axis, out=out).view(type(self))
            if result.ndim:
                # Set the mask
                result.__setmask__(newmask)
                # Get rid of Infs
                if newmask.ndim:
                    np.putmask(result, newmask, result.fill_value)
            elif newmask:
                result = masked
            return result
        # Explicit output
        result = self.filled(fill_value).max(axis=axis, out=out)
        if isinstance(out, MaskedArray):
            outmask = getattr(out, '_mask', nomask)
            if (outmask is nomask):
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        else:

            if out.dtype.kind in 'biu':
                errmsg = "Masked data information would be lost in one or more"\
                         " location."
                raise MaskError(errmsg)
            np.putmask(out, newmask, np.nan)
        return out

    def ptp(self, axis=None, out=None, fill_value=None):
        """
        Return (maximum - minimum) along the the given dimension
        (i.e. peak-to-peak value).

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to find the peaks.  If None (default) the
            flattened array is used.
        out : {None, array_like}, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary.
        fill_value : {var}, optional
            Value used to fill in the masked values.

        Returns
        -------
        ptp : ndarray.
            A new array holding the result, unless ``out`` was
            specified, in which case a reference to ``out`` is returned.

        """
        if out is None:
            result = self.max(axis=axis, fill_value=fill_value)
            result -= self.min(axis=axis, fill_value=fill_value)
            return result
        out.flat = self.max(axis=axis, out=out, fill_value=fill_value)
        out -= self.min(axis=axis, fill_value=fill_value)
        return out


    # Array methods ---------------------------------------
    copy = _arraymethod('copy')
    diagonal = _arraymethod('diagonal')
    take = _arraymethod('take')
    transpose = _arraymethod('transpose')
    T = property(fget=lambda self:self.transpose())
    swapaxes = _arraymethod('swapaxes')
    clip = _arraymethod('clip', onmask=False)
    copy = _arraymethod('copy')
    squeeze = _arraymethod('squeeze')
    #--------------------------------------------
    def tolist(self, fill_value=None):
        """
    Copy the data portion of the array to a hierarchical python
    list and returns that list.

    Data items are converted to the nearest compatible Python
    type.  Masked values are converted to fill_value. If
    fill_value is None, the corresponding entries in the output
    list will be ``None``.

        """
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        result = self.filled().tolist()
        # Set temps to save time when dealing w/ mrecarrays...
        _mask = self._mask
        if _mask is nomask:
            return result
        nbdims = self.ndim
        dtypesize = len(self.dtype)
        if nbdims == 0:
            return tuple([None] * dtypesize)
        elif nbdims == 1:
            maskedidx = _mask.nonzero()[0].tolist()
            if dtypesize:
                nodata = tuple([None] * dtypesize)
            else:
                nodata = None
            [operator.setitem(result,i,nodata) for i in maskedidx]
        else:
            for idx in zip(*[i.tolist() for i in _mask.nonzero()]):
                tmp = result
                for i in idx[:-1]:
                    tmp = tmp[i]
                tmp[idx[-1]] = None
        return result
    #........................
    def tostring(self, fill_value=None, order='C'):
        """
        Return a copy of array data as a Python string containing the raw bytes
        in the array.  The array is filled beforehand.

        Parameters
        ----------
        fill_value : {var}, optional
            Value used to fill in the masked values.
            If None, uses self.fill_value instead.
        order : {string}
            Order of the data item in the copy {'C','F','A'}.
            'C'       -- C order (row major)
            'Fortran' -- Fortran order (column major)
            'Any'     -- Current order of array.
            None      -- Same as "Any"

        Notes
        -----
        As for method:`ndarray.tostring`, information about the shape, dtype...,
        but also fill_value will be lost.

        """
        return self.filled(fill_value).tostring(order=order)
    #........................
    def tofile(self, fid, sep="", format="%s"):
        raise NotImplementedError("Not implemented yet, sorry...")

    def toflex(self):
        """
        Transforms a MaskedArray into a flexible-type array with two fields:

        * the ``_data`` field stores the ``_data`` part of the array;
        * the ``_mask`` field stores the ``_mask`` part of the array;

        Returns
        -------
        record : ndarray
            A new flexible-type ndarray with two fields: the first element
            containing a value, the second element containing the corresponding
            mask boolean.  The returned record shape matches self.shape.

        Notes
        -----
        A side-effect of transforming a masked array into a flexible ndarray is
        that meta information (``fill_value``, ...) will be lost.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> print x
        [[1 -- 3]
         [-- 5 --]
         [7 -- 9]]
        >>> print x.toflex()
        [[(1, False) (2, True) (3, False)]
         [(4, True) (5, False) (6, True)]
         [(7, False) (8, True) (9, False)]]

        """
        # Get the basic dtype ....
        ddtype = self.dtype
        # Make sure we have a mask
        _mask = self._mask
        if _mask is None:
            _mask = make_mask_none(self.shape, ddtype)
        # And get its dtype
        mdtype = self._mask.dtype
        #
        record = np.ndarray(shape=self.shape,
                            dtype=[('_data',ddtype),('_mask',mdtype)])
        record['_data'] = self._data
        record['_mask'] = self._mask
        return record
    torecords = toflex
    #--------------------------------------------
    # Pickling
    def __getstate__(self):
        """Return the internal state of the masked array, for pickling
        purposes.

        """
        state = (1,
                 self.shape,
                 self.dtype,
                 self.flags.fnc,
                 self._data.tostring(),
                 getmaskarray(self).tostring(),
                 self._fill_value,
                 )
        return state
    #
    def __setstate__(self, state):
        """Restore the internal state of the masked array, for
        pickling purposes.  ``state`` is typically the output of the
        ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.

        """
        (ver, shp, typ, isf, raw, msk, flv) = state
        ndarray.__setstate__(self, (shp, typ, isf, raw))
        self._mask.__setstate__((shp, make_mask_descr(typ), isf, msk))
        self.fill_value = flv
    #
    def __reduce__(self):
        """Return a 3-tuple for pickling a MaskedArray.

        """
        return (_mareconstruct,
                (self.__class__, self._baseclass, (0,), 'b', ),
                self.__getstate__())
    #
    def __deepcopy__(self, memo=None):
        from copy import deepcopy
        copied = MaskedArray.__new__(type(self), self, copy=True)
        if memo is None:
            memo = {}
        memo[id(self)] = copied
        for (k,v) in self.__dict__.iteritems():
            copied.__dict__[k] = deepcopy(v, memo)
        return copied


def _mareconstruct(subtype, baseclass, baseshape, basetype,):
    """Internal function that builds a new MaskedArray from the
    information stored in a pickle.

    """
    _data = ndarray.__new__(baseclass, baseshape, basetype)
    _mask = ndarray.__new__(ndarray, baseshape, make_mask_descr(basetype))
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)


#####--------------------------------------------------------------------------
#---- --- Shortcuts ---
#####---------------------------------------------------------------------------
def isMaskedArray(x):
    """
    Test whether input is an instance of MaskedArray.

    This function returns True if `x` is an instance of MaskedArray
    and returns False otherwise.  Any object is accepted as input.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray.

    See Also
    --------
    isMA : Alias to isMaskedArray.
    isarray : Alias to isMaskedArray.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.eye(3, 3)
    >>> a
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> m = ma.masked_values(a, 0)
    >>> m
    masked_array(data =
     [[1.0 -- --]
     [-- 1.0 --]
     [-- -- 1.0]],
          mask =
     [[False  True  True]
     [ True False  True]
     [ True  True False]],
          fill_value=0.0)
    >>> ma.isMaskedArray(a)
    False
    >>> ma.isMaskedArray(m)
    True
    >>> ma.isMaskedArray([0, 1, 2])
    False

    """
    return isinstance(x, MaskedArray)
isarray = isMaskedArray
isMA = isMaskedArray  #backward compatibility
# We define the masked singleton as a float for higher precedence...
# Note that it can be tricky sometimes w/ type comparison
masked_singleton = MaskedArray(0, dtype=np.float_, mask=True)
masked = masked_singleton

masked_array = MaskedArray

def array(data, dtype=None, copy=False, order=False,
          mask=nomask, fill_value=None,
          keep_mask=True, hard_mask=False, shrink=True, subok=True, ndmin=0,
          ):
    """array(data, dtype=None, copy=False, order=False, mask=nomask,
             fill_value=None, keep_mask=True, hard_mask=False, shrink=True,
             subok=True, ndmin=0)

    Acts as shortcut to MaskedArray, with options in a different order
    for convenience.  And backwards compatibility...

    """
    #!!!: we should try to put 'order' somwehere
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy, subok=subok,
                       keep_mask=keep_mask, hard_mask=hard_mask,
                       fill_value=fill_value, ndmin=ndmin, shrink=shrink)
array.__doc__ = masked_array.__doc__

def is_masked(x):
    """
    Determine whether input has masked values.

    Accepts any object as input, but always returns False unless the
    input is a MaskedArray containing masked values.

    Parameters
    ----------
    x : array_like
        Array to check for masked values.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray with masked values, False otherwise.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> x
    masked_array(data = [-- 1 -- 2 3],
          mask = [ True False  True False False],
          fill_value=999999)
    >>> ma.is_masked(x)
    True
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 42)
    >>> x
    masked_array(data = [0 1 0 2 3],
          mask = False,
          fill_value=999999)
    >>> ma.is_masked(x)
    False

    Always returns False if `x` isn't a MaskedArray.

    >>> x = [False, True, False]
    >>> ma.is_masked(x)
    False
    >>> x = 'a string'
    >>> ma.is_masked(x)
    False

    """
    m = getmask(x)
    if m is nomask:
        return False
    elif m.any():
        return True
    return False


#####---------------------------------------------------------------------------
#---- --- Extrema functions ---
#####---------------------------------------------------------------------------
class _extrema_operation(object):
    "Generic class for maximum/minimum functions."
    def __call__(self, a, b=None):
        "Executes the call behavior."
        if b is None:
            return self.reduce(a)
        return where(self.compare(a, b), a, b)
    #.........
    def reduce(self, target, axis=None):
        "Reduce target along the given axis."
        target = narray(target, copy=False, subok=True)
        m = getmask(target)
        if axis is not None:
            kargs = { 'axis' : axis }
        else:
            kargs = {}
            target = target.ravel()
            if not (m is nomask):
                m = m.ravel()
        if m is nomask:
            t = self.ufunc.reduce(target, **kargs)
        else:
            target = target.filled(self.fill_value_func(target)).view(type(target))
            t = self.ufunc.reduce(target, **kargs)
            m = umath.logical_and.reduce(m, **kargs)
            if hasattr(t, '_mask'):
                t._mask = m
            elif m:
                t = masked
        return t
    #.........
    def outer (self, a, b):
        "Return the function applied to the outer product of a and b."
        ma = getmask(a)
        mb = getmask(b)
        if ma is nomask and mb is nomask:
            m = nomask
        else:
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = logical_or.outer(ma, mb)
        result = self.ufunc.outer(filled(a), filled(b))
        if not isinstance(result, MaskedArray):
            result = result.view(MaskedArray)
        result._mask = m
        return result

#............................
class _minimum_operation(_extrema_operation):
    "Object to calculate minima"
    def __init__ (self):
        """minimum(a, b) or minimum(a)
In one argument case, returns the scalar minimum.
        """
        self.ufunc = umath.minimum
        self.afunc = amin
        self.compare = less
        self.fill_value_func = minimum_fill_value

#............................
class _maximum_operation(_extrema_operation):
    "Object to calculate maxima"
    def __init__ (self):
        """maximum(a, b) or maximum(a)
           In one argument case returns the scalar maximum.
        """
        self.ufunc = umath.maximum
        self.afunc = amax
        self.compare = greater
        self.fill_value_func = maximum_fill_value

#..........................................................
def min(obj, axis=None, out=None, fill_value=None):
    try:
        return obj.min(axis=axis, fill_value=fill_value, out=out)
    except (AttributeError, TypeError):
        # If obj doesn't have a max method,
        # ...or if the method doesn't accept a fill_value argument
        return asanyarray(obj).min(axis=axis, fill_value=fill_value, out=out)
min.__doc__ = MaskedArray.min.__doc__

def max(obj, axis=None, out=None, fill_value=None):
    try:
        return obj.max(axis=axis, fill_value=fill_value, out=out)
    except (AttributeError, TypeError):
        # If obj doesn't have a max method,
        # ...or if the method doesn't accept a fill_value argument
        return asanyarray(obj).max(axis=axis, fill_value=fill_value, out=out)
max.__doc__ = MaskedArray.max.__doc__

def ptp(obj, axis=None, out=None, fill_value=None):
    """a.ptp(axis=None) =  a.max(axis)-a.min(axis)"""
    try:
        return obj.ptp(axis, out=out, fill_value=fill_value)
    except (AttributeError, TypeError):
        # If obj doesn't have a max method,
        # ...or if the method doesn't accept a fill_value argument
        return asanyarray(obj).ptp(axis=axis, fill_value=fill_value, out=out)
ptp.__doc__ = MaskedArray.ptp.__doc__


#####---------------------------------------------------------------------------
#---- --- Definition of functions from the corresponding methods ---
#####---------------------------------------------------------------------------
class _frommethod:
    """Define functions from existing MaskedArray methods.

    Parameters
    ----------
        _methodname : string
            Name of the method to transform.

    """
    def __init__(self, methodname):
        self.__name__ = methodname
        self.__doc__ = self.getdoc()
    #
    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        meth = getattr(MaskedArray, self.__name__, None) or\
               getattr(np, self.__name__, None)
        signature = self.__name__ + get_object_signature(meth)
        if meth is not None:
            doc = """    %s\n%s""" % (signature, getattr(meth, '__doc__', None))
            return doc
    #
    def __call__(self, a, *args, **params):
        if isinstance(a, MaskedArray):
            return getattr(a, self.__name__).__call__(*args, **params)
        #FIXME ----
        #As x is not a MaskedArray, we transform it to a ndarray with asarray
        #... and call the corresponding method.
        #Except that sometimes it doesn't work (try reshape([1,2,3,4],(2,2)))
        #we end up with a "SystemError: NULL result without error in PyObject_Call"
        #A dirty trick is then to call the initial numpy function...
        method = getattr(narray(a, copy=False), self.__name__)
        try:
            return method(*args, **params)
        except SystemError:
            return getattr(np,self.__name__).__call__(a, *args, **params)

all = _frommethod('all')
anomalies = anom = _frommethod('anom')
any = _frommethod('any')
compress = _frommethod('compress')
cumprod = _frommethod('cumprod')
cumsum = _frommethod('cumsum')
copy = _frommethod('copy')
diagonal = _frommethod('diagonal')
harden_mask = _frommethod('harden_mask')
ids = _frommethod('ids')
maximum = _maximum_operation()
mean = _frommethod('mean')
minimum = _minimum_operation ()
nonzero = _frommethod('nonzero')
prod = _frommethod('prod')
product = _frommethod('prod')
ravel = _frommethod('ravel')
repeat = _frommethod('repeat')
shrink_mask = _frommethod('shrink_mask')
soften_mask = _frommethod('soften_mask')
std = _frommethod('std')
sum = _frommethod('sum')
swapaxes = _frommethod('swapaxes')
take = _frommethod('take')
trace = _frommethod('trace')
var = _frommethod('var')

#..............................................................................
def power(a, b, third=None):
    """Computes a**b elementwise.

    """
    if third is not None:
        raise MaskError, "3-argument power not supported."
    # Get the masks
    ma = getmask(a)
    mb = getmask(b)
    m = mask_or(ma, mb)
    # Get the rawdata
    fa = getdata(a)
    fb = getdata(b)
    # Get the type of the result (so that we preserve subclasses)
    if isinstance(a, MaskedArray):
        basetype = type(a)
    else:
        basetype = MaskedArray
    # Get the result and view it as a (subclass of) MaskedArray
    result = np.where(m, fa, umath.power(fa, fb)).view(basetype)
    result._update_from(a)
    # Find where we're in trouble w/ NaNs and Infs
    invalid = np.logical_not(np.isfinite(result.view(ndarray)))
    # Add the initial mask
    if m is not nomask:
        if not (result.ndim):
            return masked
        m |= invalid
        result._mask = m
    # Fix the invalid parts
    if invalid.any():
        if not result.ndim:
            return masked
        elif result._mask is nomask:
            result._mask = invalid
        result._data[invalid] = result.fill_value
    return result

#    if fb.dtype.char in typecodes["Integer"]:
#        return masked_array(umath.power(fa, fb), m)
#    m = mask_or(m, (fa < 0) & (fb != fb.astype(int)))
#    if m is nomask:
#        return masked_array(umath.power(fa, fb))
#    else:
#        fa = fa.copy()
#        if m.all():
#            fa.flat = 1
#        else:
#            np.putmask(fa,m,1)
#        return masked_array(umath.power(fa, fb), m)

#..............................................................................
def argsort(a, axis=None, kind='quicksort', order=None, fill_value=None):
    "Function version of the eponymous method."
    if fill_value is None:
        fill_value = default_fill_value(a)
    d = filled(a, fill_value)
    if axis is None:
        return d.argsort(kind=kind, order=order)
    return d.argsort(axis, kind=kind, order=order)
argsort.__doc__ = MaskedArray.argsort.__doc__

def argmin(a, axis=None, fill_value=None):
    "Function version of the eponymous method."
    if fill_value is None:
        fill_value = default_fill_value(a)
    d = filled(a, fill_value)
    return d.argmin(axis=axis)
argmin.__doc__ = MaskedArray.argmin.__doc__

def argmax(a, axis=None, fill_value=None):
    "Function version of the eponymous method."
    if fill_value is None:
        fill_value = default_fill_value(a)
        try:
            fill_value = - fill_value
        except:
            pass
    d = filled(a, fill_value)
    return d.argmax(axis=axis)
argmin.__doc__ = MaskedArray.argmax.__doc__

def sort(a, axis=-1, kind='quicksort', order=None, endwith=True, fill_value=None):
    "Function version of the eponymous method."
    a = narray(a, copy=True, subok=True)
    if axis is None:
        a = a.flatten()
        axis = 0
    if fill_value is None:
        if endwith:
            filler = minimum_fill_value(a)
        else:
            filler = maximum_fill_value(a)
    else:
        filler = fill_value
#    return
    indx = np.indices(a.shape).tolist()
    indx[axis] = filled(a, filler).argsort(axis=axis, kind=kind, order=order)
    return a[indx]
sort.__doc__ = MaskedArray.sort.__doc__


def compressed(x):
    """
    Return a 1-D array of all the non-masked data.

    See Also
    --------
    MaskedArray.compressed
        equivalent method

    """
    if getmask(x) is nomask:
        return np.asanyarray(x)
    else:
        return x.compressed()

def concatenate(arrays, axis=0):
    "Concatenate the arrays along the given axis."
    d = np.concatenate([getdata(a) for a in arrays], axis)
    rcls = get_masked_subclass(*arrays)
    data = d.view(rcls)
    # Check whether one of the arrays has a non-empty mask...
    for x in arrays:
        if getmask(x) is not nomask:
            break
    else:
        return data
    # OK, so we have to concatenate the masks
    dm = np.concatenate([getmaskarray(a) for a in arrays], axis)
    # If we decide to keep a '_shrinkmask' option, we want to check that ...
    # ... all of them are True, and then check for dm.any()
#    shrink = numpy.logical_or.reduce([getattr(a,'_shrinkmask',True) for a in arrays])
#    if shrink and not dm.any():
    if not dm.dtype.fields and not dm.any():
        data._mask = nomask
    else:
        data._mask = dm.reshape(d.shape)
    return data

def count(a, axis = None):
    if isinstance(a, MaskedArray):
        return a.count(axis)
    return masked_array(a, copy=False).count(axis)
count.__doc__ = MaskedArray.count.__doc__


def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-dimensional array, return a copy of
        its `k`-th diagonal. If `v` is a 1-dimensional array,
        return a 2-dimensional array with `v` on the `k`-th diagonal.
    k : int, optional
        Diagonal in question.  The defaults is 0.

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    """
    output = np.diag(v, k).view(MaskedArray)
    if getmask(v) is not nomask:
        output._mask = np.diag(v._mask, k)
    return output


def expand_dims(x, axis):
    """
    Expand the shape of the array by including a new axis before
    the given one.

    """
    result = n_expand_dims(x, axis)
    if isinstance(x, MaskedArray):
        new_shape = result.shape
        result = x.view()
        result.shape = new_shape
        if result._mask is not nomask:
            result._mask.shape = new_shape
    return result

#......................................
def left_shift (a, n):
    "Left shift n bits."
    m = getmask(a)
    if m is nomask:
        d = umath.left_shift(filled(a), n)
        return masked_array(d)
    else:
        d = umath.left_shift(filled(a, 0), n)
        return masked_array(d, mask=m)

def right_shift (a, n):
    "Right shift n bits."
    m = getmask(a)
    if m is nomask:
        d = umath.right_shift(filled(a), n)
        return masked_array(d)
    else:
        d = umath.right_shift(filled(a, 0), n)
        return masked_array(d, mask=m)

#......................................
def put(a, indices, values, mode='raise'):
    """Set storage-indexed locations to corresponding values.

    Values and indices are filled if necessary.

    """
    # We can't use 'frommethod', the order of arguments is different
    try:
        return a.put(indices, values, mode=mode)
    except AttributeError:
        return narray(a, copy=False).put(indices, values, mode=mode)

def putmask(a, mask, values): #, mode='raise'):
    """Set a.flat[n] = values[n] for each n where mask.flat[n] is true.

    If values is not the same size of a and mask then it will repeat
    as necessary.  This gives different behavior than
    a[mask] = values.

    Note: Using a masked array as values will NOT transform a ndarray in
          a maskedarray.

    """
    # We can't use 'frommethod', the order of arguments is different
    if not isinstance(a, MaskedArray):
        a = a.view(MaskedArray)
    (valdata, valmask) = (getdata(values), getmask(values))
    if getmask(a) is nomask:
        if valmask is not nomask:
            a._sharedmask = True
            a._mask = make_mask_none(a.shape, a.dtype)
            np.putmask(a._mask, mask, valmask)
    elif a._hardmask:
        if valmask is not nomask:
            m = a._mask.copy()
            np.putmask(m, mask, valmask)
            a.mask |= m
    else:
        if valmask is nomask:
            valmask = getmaskarray(values)
        np.putmask(a._mask, mask, valmask)
    np.putmask(a._data, mask, valdata)
    return

def transpose(a, axes=None):
    """
    Return a view of the array with dimensions permuted according to axes,
    as a masked array.

    If ``axes`` is None (default), the output view has reversed
    dimensions compared to the original.

    """
    #We can't use 'frommethod', as 'transpose' doesn't take keywords
    try:
        return a.transpose(axes)
    except AttributeError:
        return narray(a, copy=False).transpose(axes).view(MaskedArray)

def reshape(a, new_shape, order='C'):
    """Change the shape of the array a to new_shape."""
    #We can't use 'frommethod', it whine about some parameters. Dmmit.
    try:
        return a.reshape(new_shape, order=order)
    except AttributeError:
        _tmp = narray(a, copy=False).reshape(new_shape, order=order)
        return _tmp.view(MaskedArray)

def resize(x, new_shape):
    """Return a new array with the specified shape.

    The total size of the original array can be any size.  The new
    array is filled with repeated copies of a. If a was masked, the
    new array will be masked, and the new mask will be a repetition of
    the old one.

    """
    # We can't use _frommethods here, as N.resize is notoriously whiny.
    m = getmask(x)
    if m is not nomask:
        m = np.resize(m, new_shape)
    result = np.resize(x, new_shape).view(get_masked_subclass(x))
    if result.ndim:
        result._mask = m
    return result


#................................................
def rank(obj):
    "maskedarray version of the numpy function."
    return np.rank(getdata(obj))
rank.__doc__ = np.rank.__doc__
#
def shape(obj):
    "maskedarray version of the numpy function."
    return np.shape(getdata(obj))
shape.__doc__ = np.shape.__doc__
#
def size(obj, axis=None):
    "maskedarray version of the numpy function."
    return np.size(getdata(obj), axis)
size.__doc__ = np.size.__doc__
#................................................

#####--------------------------------------------------------------------------
#---- --- Extra functions ---
#####--------------------------------------------------------------------------
def where (condition, x=None, y=None):
    """where(condition | x, y)

    Returns a (subclass of) masked array, shaped like condition, where
    the elements are x when condition is True, and y otherwise.  If
    neither x nor y are given, returns a tuple of indices where
    condition is True (a la condition.nonzero()).

    Parameters
    ----------
    condition : {var}
        The condition to meet. Must be convertible to an integer
        array.
    x : {var}, optional
        Values of the output when the condition is met
    y : {var}, optional
        Values of the output when the condition is not met.

    """
    if x is None and y is None:
        return filled(condition, 0).nonzero()
    elif x is None or y is None:
        raise ValueError, "Either both or neither x and y should be given."
    # Get the condition ...............
    fc = filled(condition, 0).astype(MaskType)
    notfc = np.logical_not(fc)
    # Get the data ......................................
    xv = getdata(x)
    yv = getdata(y)
    if x is masked:
        ndtype = yv.dtype
    elif y is masked:
        ndtype = xv.dtype
    else:
        ndtype = np.max([xv.dtype, yv.dtype])
    # Construct an empty array and fill it
    d = np.empty(fc.shape, dtype=ndtype).view(MaskedArray)
    _data = d._data
    np.putmask(_data, fc, xv.astype(ndtype))
    np.putmask(_data, notfc, yv.astype(ndtype))
    # Create an empty mask and fill it
    _mask = d._mask = np.zeros(fc.shape, dtype=MaskType)
    np.putmask(_mask, fc, getmask(x))
    np.putmask(_mask, notfc, getmask(y))
    _mask |= getmaskarray(condition)
    if not _mask.any():
        d._mask = nomask
    return d

def choose (indices, choices, out=None, mode='raise'):
    """
    choose(a, choices, out=None, mode='raise')

    Use an index array to construct a new array from a set of choices.

    Given an array of integers and a set of n choice arrays, this method
    will create a new array that merges each of the choice arrays.  Where a
    value in `a` is i, the new array will have the value that choices[i]
    contains in the same place.

    Parameters
    ----------
    a : int array
        This array must contain integers in [0, n-1], where n is the number
        of choices.
    choices : sequence of arrays
        Choice arrays. The index array and all of the choices should be
        broadcastable to the same shape.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.
        'raise' : raise an error
        'wrap' : wrap around
        'clip' : clip to the range

    Returns
    -------
    merged_array : array

    See Also
    --------
    choose : equivalent function

    """
    def fmask (x):
        "Returns the filled array, or True if masked."
        if x is masked:
            return True
        return filled(x)
    def nmask (x):
        "Returns the mask, True if ``masked``, False if ``nomask``."
        if x is masked:
            return True
        return getmask(x)
    # Get the indices......
    c = filled(indices, 0)
    # Get the masks........
    masks = [nmask(x) for x in choices]
    data = [fmask(x) for x in choices]
    # Construct the mask
    outputmask = np.choose(c, masks, mode=mode)
    outputmask = make_mask(mask_or(outputmask, getmask(indices)),
                           copy=0, shrink=True)
    # Get the choices......
    d = np.choose(c, data, mode=mode, out=out).view(MaskedArray)
    if out is not None:
        if isinstance(out, MaskedArray):
            out.__setmask__(outputmask)
        return out
    d.__setmask__(outputmask)
    return d


def round_(a, decimals=0, out=None):
    """
    Return a copy of a, rounded to 'decimals' places.

    When 'decimals' is negative, it specifies the number of positions
    to the left of the decimal point.  The real and imaginary parts of
    complex numbers are rounded separately. Nothing is done if the
    array is not of float type and 'decimals' is greater than or equal
    to 0.

    Parameters
    ----------
    decimals : int
        Number of decimals to round to. May be negative.
    out : array_like
        Existing array to use for output.
        If not given, returns a default copy of a.

    Notes
    -----
    If out is given and does not have a mask attribute, the mask of a
    is lost!

    """
    if out is None:
        return np.round_(a, decimals, out)
    else:
        np.round_(getdata(a), decimals, out)
        if hasattr(out, '_mask'):
            out._mask = getmask(a)
        return out
round = round_

def inner(a, b):
    """
    Returns the inner product of a and b for arrays of floating point types.

    Like the generic NumPy equivalent the product sum is over the last dimension
    of a and b.

    Notes
    -----
    The first argument is not conjugated.

    """
    fa = filled(a, 0)
    fb = filled(b, 0)
    if len(fa.shape) == 0:
        fa.shape = (1,)
    if len(fb.shape) == 0:
        fb.shape = (1,)
    return np.inner(fa, fb).view(MaskedArray)
inner.__doc__ = doc_note(np.inner.__doc__,
                         "Masked values are replaced by 0.")
innerproduct = inner

def outer(a, b):
    "maskedarray version of the numpy function."
    fa = filled(a, 0).ravel()
    fb = filled(b, 0).ravel()
    d = np.outer(fa, fb)
    ma = getmask(a)
    mb = getmask(b)
    if ma is nomask and mb is nomask:
        return masked_array(d)
    ma = getmaskarray(a)
    mb = getmaskarray(b)
    m = make_mask(1-np.outer(1-ma, 1-mb), copy=0)
    return masked_array(d, mask=m)
outer.__doc__ = doc_note(np.outer.__doc__,
                         "Masked values are replaced by 0.")
outerproduct = outer

def allequal (a, b, fill_value=True):
    """
    Return True if all entries of a and b are equal, using
    fill_value as a truth value where either or both are masked.

    """
    m = mask_or(getmask(a), getmask(b))
    if m is nomask:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        return d.all()
    elif fill_value:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        dm = array(d, mask=m, copy=False)
        return dm.filled(True).all(None)
    else:
        return False

def allclose (a, b, masked_equal=True, rtol=1.e-5, atol=1.e-8, fill_value=None):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    This function is equivalent to `allclose` except that masked values
    are treated as equal (default) or unequal, depending on the `masked_equal`
    argument.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    masked_equal : boolean, optional
        Whether masked values in `a` and `b` are considered equal (True) or not
        (False). They are considered equal by default.
    rtol : Relative tolerance
        The relative difference is equal to `rtol` * `b`.
    atol : Absolute tolerance
        The absolute difference is equal to `atol`.
    fill_value : boolean, optional
        *Deprecated* - Whether masked values in a or b are considered equal
        (True) or not (False).

    Returns
    -------
    y : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise. If either array contains NaN, then
        False is returned.

    See Also
    --------
    all, any
    numpy.allclose : the non-masked allclose

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    Return True if all elements of a and b are equal subject to
    given tolerances.

    Examples
    --------
    >>> a = ma.array([1e10, 1e-7, 42.0], mask=[0, 0, 1])
    >>> a
    masked_array(data = [10000000000.0 1e-07 --],
                 mask = [False False  True],
           fill_value = 1e+20)
    >>> b = ma.array([1e10, 1e-8, -42.0], mask=[0, 0, 1])
    >>> ma.allclose(a, b)
    False
    >>> a = ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
    >>> b = ma.array([1.00001e10, 1e-9, -42.0], mask=[0, 0, 1])
    >>> ma.allclose(a, b)
    True
    >>> ma.allclose(a, b, masked_equal=False)
    False

    Masked values are not compared directly.

    >>> a = ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
    >>> b = ma.array([1.00001e10, 1e-9, 42.0], mask=[0, 0, 1])
    >>> ma.allclose(a, b)
    True
    >>> ma.allclose(a, b, masked_equal=False)
    False

    """
    if fill_value is not None:
        warnings.warn("The use of fill_value is deprecated."\
                      " Please use masked_equal instead.")
        masked_equal = fill_value
    #
    x = masked_array(a, copy=False)
    y = masked_array(b, copy=False)
    m = mask_or(getmask(x), getmask(y))
    xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)
    # If we have some infs, they should fall at the same place.
    if not np.all(xinf == filled(np.isinf(y), False)):
        return False
    # No infs at all
    if not np.any(xinf):
        d = filled(umath.less_equal(umath.absolute(x-y),
                                    atol + rtol * umath.absolute(y)),
                   masked_equal)
        return np.all(d)
    if not np.all(filled(x[xinf] == y[xinf], masked_equal)):
        return False
    x = x[~xinf]
    y = y[~xinf]
    d = filled(umath.less_equal(umath.absolute(x-y),
                                atol + rtol * umath.absolute(y)),
               masked_equal)
    return np.all(d)

#..............................................................................
def asarray(a, dtype=None, order=None):
    """
    Convert the input `a` to a masked array of the given datatype.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Defaults to 'C'.

    Returns
    -------
    out : ndarray
        MaskedArray interpretation of `a`.  No copy is performed if the input
        is already an ndarray.  If `a` is a subclass of MaskedArray, a base
        class MaskedArray is returned.

    """
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=False)

def asanyarray(a, dtype=None):
    """
    Convert the input `a` to a masked array of the given datatype.
    If `a` is a subclass of MaskedArray, its class is conserved.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Defaults to 'C'.

    Returns
    -------
    out : ndarray
        MaskedArray interpretation of `a`.  No copy is performed if the input
        is already an ndarray.

    """
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=True)


#####--------------------------------------------------------------------------
#---- --- Pickling ---
#####--------------------------------------------------------------------------
def dump(a, F):
    """
    Pickle the MaskedArray `a` to the file `F`.  `F` can either be
    the handle of an exiting file, or a string representing a file
    name.

    """
    if not hasattr(F,'readline'):
        F = open(F, 'w')
    return cPickle.dump(a, F)

def dumps(a):
    """
    Return a string corresponding to the pickling of the MaskedArray.

    """
    return cPickle.dumps(a)

def load(F):
    """
    Wrapper around ``cPickle.load`` which accepts either a file-like object
    or a filename.

    """
    if not hasattr(F, 'readline'):
        F = open(F,'r')
    return cPickle.load(F)

def loads(strg):
    "Load a pickle from the current string."""
    return cPickle.loads(strg)

################################################################################
def fromfile(file, dtype=float, count=-1, sep=''):
    raise NotImplementedError("Not yet implemented. Sorry")


def fromflex(fxarray):
    """
    Rebuilds a masked_array from a flexible-type array output by the '.torecord'
    array
    """
    return masked_array(fxarray['_data'], mask=fxarray['_mask'])



class _convert2ma:
    """Convert functions from numpy to numpy.ma.

    Parameters
    ----------
        _methodname : string
            Name of the method to transform.

    """
    __doc__ = None
    def __init__(self, funcname):
        self._func = getattr(np, funcname)
        self.__doc__ = self.getdoc()
    #
    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        doc = getattr(self._func, '__doc__', None)
        sig = get_object_signature(self._func)
        if doc:
            # Add the signature of the function at the beginning of the doc
            if sig:
                sig = "%s%s\n" % (self._func.__name__, sig)
            doc = sig + doc
        return doc
    #
    def __call__(self, a, *args, **params):
        return self._func.__call__(a, *args, **params).view(MaskedArray)

arange = _convert2ma('arange')
clip = np.clip
empty = _convert2ma('empty')
empty_like = _convert2ma('empty_like')
frombuffer = _convert2ma('frombuffer')
fromfunction = _convert2ma('fromfunction')
identity = _convert2ma('identity')
indices = np.indices
ones = _convert2ma('ones')
zeros = _convert2ma('zeros')
squeeze = np.squeeze

###############################################################################
