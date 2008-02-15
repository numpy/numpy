# pylint: disable-msg=E1002
"""MA: a facility for dealing with missing observations
MA is generally used as a numpy.array look-alike.
by Paul F. Dubois.

Copyright 1999, 2000, 2001 Regents of the University of California.
Released for unlimited redistribution.
Adapted for numpy_core 2005 by Travis Oliphant and
(mainly) Paul Dubois.

Subclassing of the base ndarray 2006 by Pierre Gerard-Marchant.
pgmdevlist_AT_gmail_DOT_com
Improvements suggested by Reggie Dugard (reggie_AT_merfinllc_DOT_com)

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
"""
__author__ = "Pierre GF Gerard-Marchant"
__docformat__ = "restructuredtext en"

__all__ = ['MAError', 'MaskType', 'MaskedArray',
           'bool_', 'complex_', 'float_', 'int_', 'object_',
           'abs', 'absolute', 'add', 'all', 'allclose', 'allequal', 'alltrue',
           'amax', 'amin', 'anom', 'anomalies', 'any', 'arange',
           'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2',
           'arctanh', 'argmax', 'argmin', 'argsort', 'around',
           'array', 'asarray','asanyarray',
           'bitwise_and', 'bitwise_or', 'bitwise_xor',
           'ceil', 'choose', 'compress', 'compressed', 'concatenate',
           'conjugate', 'cos', 'cosh', 'count',
           'default_fill_value', 'diagonal', 'divide', 'dump', 'dumps',
           'empty', 'empty_like', 'equal', 'exp',
           'fabs', 'fmod', 'filled', 'floor', 'floor_divide','fix_invalid',
           'getdata','getmask', 'getmaskarray', 'greater', 'greater_equal', 
           'hypot',
           'ids', 'inner', 'innerproduct',
           'isMA', 'isMaskedArray', 'is_mask', 'is_masked', 'isarray',
           'left_shift', 'less', 'less_equal', 'load', 'loads', 'log', 'log10',
           'logical_and', 'logical_not', 'logical_or', 'logical_xor',
           'make_mask', 'make_mask_none', 'mask_or', 'masked',
           'masked_array', 'masked_equal', 'masked_greater',
           'masked_greater_equal', 'masked_inside', 'masked_less',
           'masked_less_equal', 'masked_not_equal', 'masked_object',
           'masked_outside', 'masked_print_option', 'masked_singleton',
           'masked_values', 'masked_where', 'max', 'maximum', 'mean', 'min',
           'minimum', 'multiply',
           'negative', 'nomask', 'nonzero', 'not_equal',
           'ones', 'outer', 'outerproduct',
           'power', 'product', 'ptp', 'put', 'putmask',
           'rank', 'ravel', 'remainder', 'repeat', 'reshape', 'resize',
           'right_shift', 'round_',
           'shape', 'sin', 'sinh', 'size', 'sometrue', 'sort', 'sqrt', 'std',
           'subtract', 'sum', 'swapaxes',
           'take', 'tan', 'tanh', 'transpose', 'true_divide',
           'var', 'where',
           'zeros']

import sys
import types
import cPickle
import operator

import numpy
from numpy.core import bool_, complex_, float_, int_, object_, str_

import numpy.core.umath as umath
import numpy.core.fromnumeric  as fromnumeric
import numpy.core.numeric as numeric
import numpy.core.numerictypes as ntypes
from numpy import bool_, dtype, typecodes, amax, amin, ndarray
from numpy import expand_dims as n_expand_dims
from numpy import array as narray
import warnings


MaskType = bool_
nomask = MaskType(0)

divide_tolerance = 1.e-35
numpy.seterr(all='ignore')

def doc_note(note):
    return "\nNotes\n-----\n%s" % note

#####--------------------------------------------------------------------------
#---- --- Exceptions ---
#####--------------------------------------------------------------------------
class MAError(Exception):
    "Class for MA related errors."
    def __init__ (self, args=None):
        "Creates an exception."
        Exception.__init__(self,args)
        self.args = args
    def __str__(self):
        "Calculates the string representation."
        return str(self.args)
    __repr__ = __str__

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
max_filler.update([(k,-numpy.inf) for k in [numpy.float32, numpy.float64]])
min_filler = ntypes._maxvals
min_filler.update([(k,numpy.inf) for k in [numpy.float32, numpy.float64]])
if 'float128' in ntypes.typeDict:
    max_filler.update([(numpy.float128,-numpy.inf)])
    min_filler.update([(numpy.float128, numpy.inf)])

def default_fill_value(obj):
    """Calculate the default fill value for the argument object.

    """
    if hasattr(obj,'dtype'):
        defval = default_filler[obj.dtype.kind]
    elif isinstance(obj, numeric.dtype):
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

def minimum_fill_value(obj):
    """Calculate the default fill value suitable for taking the
    minimum of ``obj``.

    """
    if hasattr(obj, 'dtype'):
        objtype = obj.dtype
        filler = min_filler[objtype]
        if filler is None:
            raise TypeError, 'Unsuitable type for calculating minimum.'
        return filler
    elif isinstance(obj, float):
        return min_filler[ntypes.typeDict['float_']]
    elif isinstance(obj, int):
        return min_filler[ntypes.typeDict['int_']]
    elif isinstance(obj, long):
        return min_filler[ntypes.typeDict['uint']]
    elif isinstance(obj, numeric.dtype):
        return min_filler[obj]
    else:
        raise TypeError, 'Unsuitable type for calculating minimum.'

def maximum_fill_value(obj):
    """Calculate the default fill value suitable for taking the maximum
    of ``obj``.

    """
    if hasattr(obj, 'dtype'):
        objtype = obj.dtype
        filler = max_filler[objtype]
        if filler is None:
            raise TypeError, 'Unsuitable type for calculating minimum.'
        return filler
    elif isinstance(obj, float):
        return max_filler[ntypes.typeDict['float_']]
    elif isinstance(obj, int):
        return max_filler[ntypes.typeDict['int_']]
    elif isinstance(obj, long):
        return max_filler[ntypes.typeDict['uint']]
    elif isinstance(obj, numeric.dtype):
        return max_filler[obj]
    else:
        raise TypeError, 'Unsuitable type for calculating minimum.'


def _check_fill_value(fill_value, dtype):
    descr = numpy.dtype(dtype).descr
    if fill_value is None:
        if len(descr) > 1:
            fill_value = [default_fill_value(numeric.dtype(d[1]))
                          for d in descr]
        else:
            fill_value = default_fill_value(dtype)
    else:
        fill_value = narray(fill_value).tolist() 
        fval = numpy.resize(fill_value, len(descr))
        if len(descr) > 1:
            fill_value = [numpy.asarray(f).astype(d[1]).item()
                          for (f,d) in zip(fval, descr)]
        else:
            fill_value = narray(fval, copy=False, dtype=dtype).item()
    return fill_value


def set_fill_value(a, fill_value):
    """Set the filling value of a, if a is a masked array.  Otherwise,
    do nothing.

    Returns
    -------
    None

    """
    if isinstance(a, MaskedArray):
        a._fill_value = _check_fill_value(fill_value, a.dtype)
    return

def get_fill_value(a):
    """Return the filling value of a, if any.  Otherwise, returns the
    default filling value for that type.

    """
    if isinstance(a, MaskedArray):
        result = a.fill_value
    else:
        result = default_fill_value(a)
    return result

def common_fill_value(a, b):
    """Return the common filling value of a and b, if any.
    If a and b have different filling values, returns None.

    """
    t1 = get_fill_value(a)
    t2 = get_fill_value(b)
    if t1 == t2:
        return t1
    return None


#####--------------------------------------------------------------------------
def filled(a, value = None):
    """Return a as an array with masked data replaced by value.  If
    value is None, get_fill_value(a) is used instead.  If a is already
    a ndarray, a itself is returned.

    Parameters
    ----------
    a : maskedarray or array_like
        An input object.
    value : {var}, optional
        Filling value. If not given, the output of get_fill_value(a)
        is used instead.

    Returns
    -------
    a : array_like

    """
    if hasattr(a, 'filled'):
        return a.filled(value)
    elif isinstance(a, ndarray): 
        # Should we check for contiguity ? and a.flags['CONTIGUOUS']:
        return a
    elif isinstance(a, dict):
        return narray(a, 'O')
    else:
        return narray(a)

#####--------------------------------------------------------------------------
def get_masked_subclass(*arrays):
    """Return the youngest subclass of MaskedArray from a list of
    (masked) arrays.  In case of siblings, the first takes over.

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
def get_data(a, subok=True):
    """Return the _data part of a (if any), or a as a ndarray.

    Parameters
    ----------
    a : array_like
        A ndarray or a subclass of.
    subok : bool
        Whether to force the output to a 'pure' ndarray (False) or to
        return a subclass of ndarray if approriate (True).

    """
    data = getattr(a, '_data', numpy.array(a, subok=subok))
    if not subok:
        return data.view(ndarray)
    return data
getdata = get_data

def fix_invalid(a, copy=True, fill_value=None):
    """Return (a copy of) a where invalid data (nan/inf) are masked
    and replaced by fill_value.

    Note that a copy is performed by default (just in case...).

    Parameters
    ----------
    a : array_like
        A (subclass of) ndarray.
    copy : bool
        Whether to use a copy of a (True) or to fix a in place (False).
    fill_value : {var}, optional
        Value used for fixing invalid data.  If not given, the output
        of get_fill_value(a) is used instead.

    Returns
    -------
    b : MaskedArray

    """
    a = masked_array(a, copy=copy, subok=True)
    invalid = (numpy.isnan(a._data) | numpy.isinf(a._data))
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
    def __init__ (self, tolerance=divide_tolerance):
        self.tolerance = tolerance
    def __call__ (self, a, b):
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
        d1 = get_data(a)
        #
        if self.domain is not None:
            dm = narray(self.domain(d1), copy=False)
            m = numpy.logical_or(m, dm)
            # The following two lines control the domain filling methods.
            d1 = d1.copy()
            # We could use smart indexing : d1[dm] = self.fill ...
            # ... but numpy.putmask looks more efficient, despite the copy.
            numpy.putmask(d1, dm, self.fill)
        # Take care of the masked singletong first ...
        if not m.ndim and m:
            return masked
        # Get the result class .......................
        if isinstance(a, MaskedArray):
            subtype = type(a)
        else:
            subtype = MaskedArray
        # Get the result  as a view of the subtype ...
        result = self.f(d1, *args, **kwargs).view(subtype)
        # Fix the mask if we don't have a scalar
        if result.ndim > 0:
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
    #
    def __call__ (self, a, b, *args, **kwargs):
        "Execute the call behavior."
        m = mask_or(getmask(a), getmask(b))
        (d1, d2) = (get_data(a), get_data(b))
        result = self.f(d1, d2, *args, **kwargs).view(get_masked_subclass(a,b))
        if result.size > 1:
            if m is not nomask:
                result._mask = make_mask_none(result.shape)
                result._mask.flat = m
            if isinstance(a,MaskedArray):
                result._update_from(a)
            if isinstance(b,MaskedArray):
                result._update_from(b)
        elif m:
            return masked
        return result
    #
    def reduce (self, target, axis=0, dtype=None):
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
        rcls = get_masked_subclass(a,b)
        # We could fill the arguments first, butis it useful ?
        # d = self.f.outer(filled(a, self.fillx), filled(b, self.filly)).view(rcls)
        d = self.f.outer(getdata(a), getdata(b)).view(rcls)
        if d.ndim > 0:
            d._mask = m
        return d

    def accumulate (self, target, axis=0):
        """Accumulate `target` along `axis` after filling with y fill
        value.

        """
        if isinstance(target, MaskedArray):
            tclass = type(target)
        else:
            tclass = masked_array
        t = filled(target, self.filly)
        return self.f.accumulate(t, axis).view(tclass)

    def __str__ (self):
        return "Masked version of " + str(self.f)

#..............................................................................
class _DomainedBinaryOperation:
    """Define binary operations that have a domain, like divide.

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

    def __call__(self, a, b):
        "Execute the call behavior."
        ma = getmask(a)
        mb = getmask(b)
        d1 = getdata(a)
        d2 = get_data(b)
        t = narray(self.domain(d1, d2), copy=False)
        if t.any(None):
            mb = mask_or(mb, t)
            # The following two lines control the domain filling
            d2 = d2.copy()
            numpy.putmask(d2, t, self.filly)
        m = mask_or(ma, mb)
        if (not m.ndim) and m:
            return masked
        result =  self.f(d1, d2).view(get_masked_subclass(a,b))
        if result.ndim > 0:
            result._mask = m
            if isinstance(a,MaskedArray):
                result._update_from(a)
            if isinstance(b,MaskedArray):
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
around = _MaskedUnaryOperation(fromnumeric.round_)
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


#####--------------------------------------------------------------------------
#---- --- Mask creation functions ---
#####--------------------------------------------------------------------------
def get_mask(a):
    """Return the mask of a, if any, or nomask.

    To get a full array of booleans of the same shape as a, use
    getmaskarray.

    """
    return getattr(a, '_mask', nomask)
getmask = get_mask

def getmaskarray(a):
    """Return the mask of a, if any, or a boolean array of the shape
    of a, full of False.

    """
    m = getmask(a)
    if m is nomask:
        m = make_mask_none(fromnumeric.shape(a))
    return m

def is_mask(m):
    """Return True if m is a legal mask.

    Does not check contents, only type.

    """
    try:
        return m.dtype.type is MaskType
    except AttributeError:
        return False
#
def make_mask(m, copy=False, shrink=True, flag=None):
    """Return m as a mask, creating a copy if necessary or requested.

    The function can accept any sequence of integers or nomask.  Does
    not check that contents must be 0s and 1s.

    Parameters
    ----------
    m : array_like
        Potential mask.
    copy : bool
        Whether to return a copy of m (True) or m itself (False).
    shrink : bool
        Whether to shrink m to nomask if all its values are False.

    """
    if flag is not None:
        warnings.warn("The flag 'flag' is now called 'shrink'!",
                      DeprecationWarning)
        shrink = flag
    if m is nomask:
        return nomask
    elif isinstance(m, ndarray):
        m = filled(m, True)
        if m.dtype.type is MaskType:
            if copy:
                result = narray(m, dtype=MaskType, copy=copy)
            else:
                result = m
        else:
            result = narray(m, dtype=MaskType)
    else:
        result = narray(filled(m, True), dtype=MaskType)
    # Bas les masques !
    if shrink and not result.any():
        return nomask
    else:
        return result

def make_mask_none(s):
    """Return a mask of shape s, filled with False.

    Parameters
    ----------
    s : tuple
        A tuple indicating the shape of the final mask.

    """
    result = numeric.zeros(s, dtype=MaskType)
    return result

def mask_or (m1, m2, copy=False, shrink=True):
    """Return the combination of two masks m1 and m2.

    The masks are combined with the *logical_or* operator, treating
    nomask as False.  The result may equal m1 or m2 if the other is
    nomask.

    Parameters
    ----------
    m1 : array_like
        First mask.
    m2 : array_like
        Second mask
    copy : bool
        Whether to return a copy.
    shrink : bool
        Whether to shrink m to nomask if all its values are False.

     """
    if m1 is nomask:
        return make_mask(m2, copy=copy, shrink=shrink)
    if m2 is nomask:
        return make_mask(m1, copy=copy, shrink=shrink)
    if m1 is m2 and is_mask(m1):
        return m1
    return make_mask(umath.logical_or(m1, m2), copy=copy, shrink=shrink)

#####--------------------------------------------------------------------------
#--- --- Masking functions ---
#####--------------------------------------------------------------------------
def masked_where(condition, a, copy=True):
    """Return a as an array masked where condition is true.

    Masked values of a or condition are kept.

    Parameters
    ----------
    condition : array_like
        Masking condition.
    a : array_like
        Array to mask.
    copy : bool
        Whether to return a copy of a (True) or modify a in place.

    """
    cond = filled(condition,1)
    a = narray(a, copy=copy, subok=True)
    if hasattr(a, '_mask'):
        cond = mask_or(cond, a._mask)
        cls = type(a)
    else:
        cls = MaskedArray
    result = a.view(cls)
    result._mask = cond
    return result

def masked_greater(x, value, copy=True):
    "Shortcut to masked_where, with condition = (x > value)."
    return masked_where(greater(x, value), x, copy=copy)

def masked_greater_equal(x, value, copy=True):
    "Shortcut to masked_where, with condition = (x >= value)."
    return masked_where(greater_equal(x, value), x, copy=copy)

def masked_less(x, value, copy=True):
    "Shortcut to masked_where, with condition = (x < value)."
    return masked_where(less(x, value), x, copy=copy)

def masked_less_equal(x, value, copy=True):
    "Shortcut to masked_where, with condition = (x <= value)."
    return masked_where(less_equal(x, value), x, copy=copy)

def masked_not_equal(x, value, copy=True):
    "Shortcut to masked_where, with condition = (x != value)."
    return masked_where((x != value), x, copy=copy)

#
def masked_equal(x, value, copy=True):
    """Shortcut to masked_where, with condition = (x == value).  For
    floating point, consider `masked_values(x, value)` instead.

    """
    # An alternative implementation relies on filling first: probably not needed.
    # d = filled(x, 0)
    # c = umath.equal(d, value)
    # m = mask_or(c, getmask(x))
    # return array(d, mask=m, copy=copy)
    return masked_where((x == value), x, copy=copy)

def masked_inside(x, v1, v2, copy=True):
    """Shortcut to masked_where, where condition is True for x inside
    the interval [v1,v2] (v1 <= x <= v2).  The boundaries v1 and v2
    can be given in either order.

    Notes
    -----
    The array x is prefilled with its filling value.

    """
    if v2 < v1:
        (v1, v2) = (v2, v1)
    xf = filled(x)
    condition = (xf >= v1) & (xf <= v2)
    return masked_where(condition, x, copy=copy)

def masked_outside(x, v1, v2, copy=True):
    """Shortcut to masked_where, where condition is True for x outside
    the interval [v1,v2] (x < v1)|(x > v2).  The boundaries v1 and v2
    can be given in either order.

    Notes
    -----
    The array x is prefilled with its filling value.

    """
    if v2 < v1:
        (v1, v2) = (v2, v1)
    xf = filled(x)
    condition = (xf < v1) | (xf > v2)
    return masked_where(condition, x, copy=copy)

#
def masked_object(x, value, copy=True):
    """Mask the array x where the data are exactly equal to value.

    This function is suitable only for object arrays: for floating
    point, please use ``masked_values`` instead.

    Notes
    -----
    The mask is set to `nomask` if posible.

    """
    if isMaskedArray(x):
        condition = umath.equal(x._data, value)
        mask = x._mask
    else:
        condition = umath.equal(fromnumeric.asarray(x), value)
        mask = nomask
    mask = mask_or(mask, make_mask(condition, shrink=True))
    return masked_array(x, mask=mask, copy=copy, fill_value=value)

def masked_values(x, value, rtol=1.e-5, atol=1.e-8, copy=True):
    """Mask the array x where the data are approximately equal in
    value, i.e.

    (abs(x - value) <= atol+rtol*abs(value))

    Suitable only for floating points. For integers, please use
    ``masked_equal``.  The mask is set to nomask if posible.

    Parameters
    ----------
    x : array_like
        Array to fill.
    value : float
        Masking value.
    rtol : float
        Tolerance parameter.
    atol : float
        Tolerance parameter (1e-8).
    copy : bool
        Whether to return a copy of x.

    """
    abs = umath.absolute
    xnew = filled(x, value)
    if issubclass(xnew.dtype.type, numeric.floating):
        condition = umath.less_equal(abs(xnew-value), atol+rtol*abs(value))
        mask = getattr(x, '_mask', nomask)
    else:
        condition = umath.equal(xnew, value)
        mask = nomask
    mask = mask_or(mask, make_mask(condition, shrink=True))
    return masked_array(xnew, mask=mask, copy=copy, fill_value=value)

def masked_invalid(a, copy=True):
    """Mask the array for invalid values (nans or infs).  Any
    preexisting mask is conserved.

    """
    a = narray(a, copy=copy, subok=True)
    condition = (numpy.isnan(a) | numpy.isinf(a))
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
    """Handle the string used to represent missing data in a masked
    array.

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

#####--------------------------------------------------------------------------
#---- --- MaskedArray class ---
#####--------------------------------------------------------------------------

#...............................................................................
class _arraymethod(object):
    """Define a wrapper for basic array methods.

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
        self._name = funcname
        self._onmask = onmask
        self.obj = None
        self.__doc__ = self.getdoc()
    #
    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        methdoc = getattr(ndarray, self._name, None)
        methdoc = getattr(numpy, self._name, methdoc)
        if methdoc is not None:
            return methdoc.__doc__
    #
    def __get__(self, obj, objtype=None):
        self.obj = obj
        return self
    #
    def __call__(self, *args, **params):
        methodname = self._name
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
            if mask.ndim and mask.all():
                return masked
        return result
#..........................................................

class FlatIter(object):
    "Define an interator."
    def __init__(self, ma):
        self.ma = ma
        self.ma_iter = numpy.asarray(ma).flat

        if ma._mask is nomask:
            self.maskiter = None
        else:
            self.maskiter = ma._mask.flat

    def __iter__(self):
        return self

    ### This won't work is ravel makes a copy
    def __setitem__(self, index, value):
        a = self.ma.ravel()
        a[index] = value

    def next(self):
        d = self.ma_iter.next()
        if self.maskiter is not None and self.maskiter.next():
            d = masked
        return d


class MaskedArray(numeric.ndarray):
    """Arrays with possibly masked values.  Masked values of True
    exclude the corresponding element from any computation.

    Construction:
        x = MaskedArray(data, mask=nomask, dtype=None, copy=True,
        fill_value=None, keep_mask=True, hard_mask=False, shrink=True)

    Parameters
    ----------
    data : {var}
        Input data.
    mask : {nomask, sequence}
        Mask.  Must be convertible to an array of booleans with
        the same shape as data: True indicates a masked (eg.,
        invalid) data.
    dtype : dtype
        Data type of the output. If None, the type of the data
        argument is used.  If dtype is not None and different from
        data.dtype, a copy is performed.
    copy : bool
        Whether to copy the input data (True), or to use a
        reference instead.  Note: data are NOT copied by default.
    subok : {True, boolean}
        Whether to return a subclass of MaskedArray (if possible)
        or a plain MaskedArray.
    ndmin : {0, int}
        Minimum number of dimensions
    fill_value : {var}
        Value used to fill in the masked values when necessary. If
        None, a default based on the datatype is used.
    keep_mask : {True, boolean}
        Whether to combine mask with the mask of the input data,
        if any (True), or to use only mask for the output (False).
    hard_mask : {False, boolean}
        Whether to use a hard mask or not. With a hard mask,
        masked values cannot be unmasked.
    shrink : {True, boolean}
        Whether to force compression of an empty mask.

    """

    __array_priority__ = 15
    _defaultmask = nomask
    _defaulthardmask = False
    _baseclass =  numeric.ndarray

    def __new__(cls, data=None, mask=nomask, dtype=None, copy=False,
                subok=True, ndmin=0, fill_value=None,
                keep_mask=True, hard_mask=False, flag=None,shrink=True,
                **options):
        """Create a new masked array from scratch.

        Note: you can also create an array with the .view(MaskedArray)
        method.

        """
        if flag is not None:
            warnings.warn("The flag 'flag' is now called 'shrink'!",
                          DeprecationWarning)
            shrink = flag
        # Process data............
        _data = narray(data, dtype=dtype, copy=copy, subok=True, ndmin=ndmin)
        _baseclass = getattr(data, '_baseclass', type(_data))
        _basedict = getattr(data, '_basedict', getattr(data, '__dict__', None))
        if not isinstance(data, MaskedArray) or not subok:
            _data = _data.view(cls)
        else:
            _data = _data.view(type(data))
        # Backwards compatibility w/ numpy.core.ma .......
        if hasattr(data,'_mask') and not isinstance(data, ndarray):
            _data._mask = data._mask
            _sharedmask = True
        # Process mask ...........
        if mask is nomask:
            if not keep_mask:
                if shrink:
                    _data._mask = nomask
                else:
                    _data._mask = make_mask_none(_data)
            if copy:
                _data._mask = _data._mask.copy()
                _data._sharedmask = False
            else:
                _data._sharedmask = True
        else:
            mask = narray(mask, dtype=MaskType, copy=copy)
            if mask.shape != _data.shape:
                (nd, nm) = (_data.size, mask.size)
                if nm == 1:
                    mask = numeric.resize(mask, _data.shape)
                elif nm == nd:
                    mask = fromnumeric.reshape(mask, _data.shape)
                else:
                    msg = "Mask and data not compatible: data size is %i, "+\
                          "mask size is %i."
                    raise MAError, msg % (nd, nm)
                copy = True
            if _data._mask is nomask:
                _data._mask = mask
                _data._sharedmask = not copy
            else:
                if not keep_mask:
                    _data._mask = mask
                    _data._sharedmask = not copy
                else:
                    _data._mask = umath.logical_or(mask, _data._mask)
                    _data._sharedmask = False

        # Update fill_value.......
        _data._fill_value = _check_fill_value(fill_value, _data.dtype)
        # Process extra options ..
        _data._hardmask = hard_mask
        _data._baseclass = _baseclass
        _data._basedict = _basedict
        return _data
    #
    def _update_from(self, obj):
        """Copies some attributes of obj to self.
        """
        self._hardmask = getattr(obj, '_hardmask', self._defaulthardmask)
        self._sharedmask = getattr(obj, '_sharedmask', False)
        if obj is not None:
            self._baseclass = getattr(obj, '_baseclass', type(obj))
        else:
            self._baseclass = ndarray
        self._fill_value = getattr(obj, '_fill_value', None)
        return
    #........................
    def __array_finalize__(self,obj):
        """Finalizes the masked array.
        """
        # Get main attributes .........
        self._mask = getattr(obj, '_mask', nomask)
        self._update_from(obj)
        # Update special attributes ...
        self._basedict = getattr(obj, '_basedict', getattr(obj, '__dict__', None))
        if self._basedict is not None:
            self.__dict__.update(self._basedict)
        # Finalize the mask ...........
        if self._mask is not nomask:
            self._mask.shape = self.shape
        return
    #..................................
    def __array_wrap__(self, obj, context=None):
        """Special hook for ufuncs.
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
                if len(args) > 2:
                    d = reduce(domain, args)
                else:
                    d = domain(*args)
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
                numpy.putmask(result, d, fill_value)
                # Update the mask
                if m is nomask:
                    if d is not nomask:
                        m = d
                else:
                    m |= d
            # Make sure the mask has the proper size
            if result.shape == () and m:
                return masked
            else:
                result._mask = m
                result._sharedmask = False
        #....
        return result
    #.............................................
    def __getitem__(self, indx):
        """x.__getitem__(y) <==> x[y]

        Return the item described by i, as a masked array.

        """
        # This test is useful, but we should keep things light...
#        if getmask(indx) is not nomask:
#            msg = "Masked arrays must be filled before they can be used as indices!"
#            raise IndexError, msg
        dout = ndarray.__getitem__(self.view(ndarray), indx)
        m = self._mask
        if not getattr(dout,'ndim', False):
            # Just a scalar............
            if m is not nomask and m[indx]:
                return masked
        else:
            # Force dout to MA ........
            dout = dout.view(type(self))
            # Inherit attributes from self
            dout._update_from(self)
            # Check the fill_value ....
            if isinstance(indx, basestring):
                fvindx = list(self.dtype.names).index(indx)
                dout._fill_value = self.fill_value[fvindx]
            # Update the mask if needed
            if m is not nomask:
                if isinstance(indx, basestring):
                    dout._mask = m.reshape(dout.shape)
                else:
                    dout._mask = ndarray.__getitem__(m, indx).reshape(dout.shape)
#               Note: Don't try to check for m.any(), that'll take too long...
#                mask = ndarray.__getitem__(m, indx).reshape(dout.shape)
#                if self._shrinkmask and not m.any():
#                    dout._mask = nomask
#                else:
#                    dout._mask = mask
        return dout
    #........................
    def __setitem__(self, indx, value):
        """x.__setitem__(i, y) <==> x[i]=y

        Set item described by index. If value is masked, masks those
        locations.

        """
        if self is masked:
            raise MAError, 'Cannot alter the masked element.'
        # This test is useful, but we should keep things light...
#        if getmask(indx) is not nomask:
#            msg = "Masked arrays must be filled before they can be used as indices!"
#            raise IndexError, msg
        if isinstance(indx, basestring):
            ndarray.__setitem__(self._data,indx, getdata(value))
            warnings.warn("The mask is NOT affected!")
            return
        #....
        if value is masked:
            m = self._mask
            if m is nomask:
                m = numpy.zeros(self.shape, dtype=MaskType)
            m[indx] = True
            self._mask = m
            self._sharedmask = False
            return
        #....
        dval = narray(value, copy=False, dtype=self.dtype)
        valmask = getmask(value)
        if self._mask is nomask:
            if valmask is not nomask:
                self._mask = numpy.zeros(self.shape, dtype=MaskType)
                self._mask[indx] = valmask
        elif not self._hardmask:
            # Unshare the mask if necessary to avoid propagation
            self.unshare_mask()
            self._mask[indx] = valmask
        elif hasattr(indx, 'dtype') and (indx.dtype==bool_):
            indx = indx * umath.logical_not(self._mask)
        else:
            mindx = mask_or(self._mask[indx], valmask, copy=True)
            dindx = self._data[indx]
            if dindx.size > 1:
                dindx[~mindx] = dval
            elif mindx is nomask:
                dindx = dval
            dval = dindx
            self._mask[indx] = mindx
        # Set data ..........
        ndarray.__setitem__(self._data,indx,dval)
    #............................................
    def __getslice__(self, i, j):
        """x.__getslice__(i, j) <==> x[i:j]

        Return the slice described by (i, j).  The use of negative
        indices is not supported.

        """
        return self.__getitem__(slice(i,j))
    #........................
    def __setslice__(self, i, j, value):
        """x.__setslice__(i, j, value) <==> x[i:j]=value

        Set the slice (i,j) of a to value. If value is masked, mask
        those locations.

        """
        self.__setitem__(slice(i,j), value)
    #............................................
    def __setmask__(self, mask, copy=False):
        """Set the mask.

        """
        if mask is not nomask:
            mask = narray(mask, copy=copy, dtype=MaskType)
            # We could try to check whether shrinking is needed..
            # ... but we would waste some precious time
#            if self._shrinkmask and not mask.any():
#                mask = nomask
        if self._mask is nomask:
            self._mask = mask
        elif self._hardmask:
            if mask is not nomask:
                self._mask.__ior__(mask)
        else:
            # This one is tricky: if we set the mask that way, we may break the
            # propagation. But if we don't, we end up with a mask full of False
            # and a test on nomask fails...
            if mask is nomask:
                self._mask = nomask
            else:
                self.unshare_mask()
                self._mask.flat = mask
        if self._mask.shape:
            self._mask = numeric.reshape(self._mask, self.shape)
    _set_mask = __setmask__
    #....
    def _get_mask(self):
        """Return the current mask.

        """
        # We could try to force a reshape, but that wouldn't work in some cases.
#        return self._mask.reshape(self.shape)
        return self._mask
    mask = property(fget=_get_mask, fset=__setmask__, doc="Mask")
    #............................................
    def harden_mask(self):
        """Force the mask to hard.

        """
        self._hardmask = True

    def soften_mask(self):
        """Force the mask to soft.

        """
        self._hardmask = False

    def unshare_mask(self):
        """Copy the mask and set the sharedmask flag to False.

        """
        if self._sharedmask:
            self._mask = self._mask.copy()
            self._sharedmask = False

    def shrink_mask(self):
        """Reduce a mask to nomask when possible.

        """
        m = self._mask
        if m.ndim and not m.any():
            self._mask = nomask

    #............................................
    def _get_data(self):
        """Return the current data, as a view of the original
        underlying data.

        """
        return self.view(self._baseclass)
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
        return FlatIter(self)
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
        self._fill_value = _check_fill_value(value,self.dtype)

    fill_value = property(fget=get_fill_value, fset=set_fill_value,
                          doc="Filling value.")

    def filled(self, fill_value=None):
        """Return a copy of self._data, where masked values are filled
        with fill_value.

        If fill_value is None, self.fill_value is used instead.

        Notes
        -----
        + Subclassing is preserved
        + The result is NOT a MaskedArray !

        Examples
        --------
        >>> x = array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
        >>> x.filled()
        array([1,2,-999,4,-999])
        >>> type(x.filled())
        <type 'numpy.ndarray'>

        """
        m = self._mask
        if m is nomask or not m.any():
            return self._data
        #
        if fill_value is None:
            fill_value = self.fill_value
        #
        if self is masked_singleton:
            result = numeric.asanyarray(fill_value)
        else:
            result = self._data.copy()
            try:
                numpy.putmask(result, m, fill_value)
            except (TypeError, AttributeError):
                fill_value = narray(fill_value, dtype=object)
                d = result.astype(object)
                result = fromnumeric.choose(m, (d, fill_value))
            except IndexError:
                #ok, if scalar
                if self._data.shape:
                    raise
                elif m:
                    result = narray(fill_value, dtype=self.dtype)
                else:
                    result = self._data
        return result

    def compressed(self):
        """Return a 1-D array of all the non-masked data.

        """
        data = ndarray.ravel(self._data).view(type(self))
        data._update_from(self)
        if self._mask is not nomask:
            data = data[numpy.logical_not(ndarray.ravel(self._mask))]
        return data
    
    
    def compress(self, condition, axis=None, out=None):
        """Return a where condition is True.
        If condition is a MaskedArray, missing values are considered as False.
        
        Returns
        -------
        A MaskedArray object.
        
        Notes
        -----
        Please note the difference with compressed() ! 
        The output of compress has a mask, the output of compressed does not.
    
        """
        # Get the basic components
        (_data, _mask) = (self._data, self._mask)
        # Force the condition to a regular ndarray (forget the missing values...)
        condition = narray(condition, copy=False, subok=False)
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
                    if m:
                        return str(f)
                    else:
                        return str(self._data)
                # convert to object array to make filled work
#CHECK: the two lines below seem more robust than the self._data.astype
#                res = numeric.empty(self._data.shape, object_)
#                numeric.putmask(res,~m,self._data)
                res = self._data.astype("|O8")
                res[m] = f
        else:
            res = self.filled(self.fill_value)
        return str(res)

    def __repr__(self):
        """Literal string representation.

        """
        with_mask = """\
masked_%(name)s(data =
 %(data)s,
      mask =
 %(mask)s,
      fill_value=%(fill)s)
"""
        with_mask1 = """\
masked_%(name)s(data = %(data)s,
      mask = %(mask)s,
      fill_value=%(fill)s)
"""
        n = len(self.shape)
        name = repr(self._data).split('(')[0]
        if n <= 1:
            return with_mask1 % {
                'name': name,
                'data': str(self),
                'mask': str(self._mask),
                'fill': str(self.fill_value),
                }
        return with_mask % {
            'name': name,
            'data': str(self),
            'mask': str(self._mask),
            'fill': str(self.fill_value),
            }
    #............................................
    def __add__(self, other):
        "Add other to self, and return a new masked array."
        return add(self, other)
    #
    def __sub__(self, other):
        "Subtract other to self, and return a new masked array."
        return subtract(self, other)
    #
    def __mul__(self, other):
        "Multiply other by self, and return a new masked array."
        return multiply(self, other)
    #
    def __div__(self, other):
        "Divides other into self, and return a new masked array."
        return divide(self, other)
    #
    def __truediv__(self, other):
        "Divide other into self, and return a new masked array."
        return true_divide(self, other)
    #
    def __floordiv__(self, other):
        "Divide other into self, and return a new masked array."
        return floor_divide(self, other)

    #............................................
    def __iadd__(self, other):
        "Add other to self in-place."
        ndarray.__iadd__(self._data, getdata(other))
        m = getmask(other)
        if self._mask is nomask:
            self._mask = m
        elif m is not nomask:
            self._mask += m
        return self
    #....
    def __isub__(self, other):
        "Subtract other from self in-place."
        ndarray.__isub__(self._data, getdata(other))
        m = getmask(other)
        if self._mask is nomask:
            self._mask = m
        elif m is not nomask:
            self._mask += m
        return self
    #....
    def __imul__(self, other):
        "Multiply self by other in-place."
        ndarray.__imul__(self._data, getdata(other))
        m = getmask(other)
        if self._mask is nomask:
            self._mask = m
        elif m is not nomask:
            self._mask += m
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
            other_data = other_data.copy()
            numpy.putmask(other_data, dom_mask, 1)
        ndarray.__idiv__(self._data, other_data)
        self._mask = mask_or(self._mask, new_mask)
        return self
    #............................................
    def __float__(self):
        "Convert to float."
        if self._mask is not nomask:
            warnings.warn("Warning: converting a masked element to nan.")
            return numpy.nan
        return float(self.item())

    def __int__(self):
        "Convert to int."
        if self._mask is not nomask:
            raise MAError, 'Cannot convert masked element to a Python int.'
        return int(self.item())
    #............................................
    def get_imag(self):
        result = self._data.imag.view(type(self))
        result.__setmask__(self._mask)
        return result
    imag = property(fget=get_imag,doc="Imaginary part")

    def get_real(self):
        result = self._data.real.view(type(self))
        result.__setmask__(self._mask)
        return result
    real = property(fget=get_real,doc="Real part")


    #............................................
    def count(self, axis=None):
        """Count the non-masked elements of the array along the given
        axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to count the non-masked elements. If
            not given, all the non masked elements are counted.

        Returns
        -------
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
                return numeric.ones(t) * n
        n1 = numpy.size(m, axis)
        n2 = m.astype(int_).sum(axis)
        if axis is None:
            return (n1-n2)
        else:
            return masked_array(n1 - n2)
    #............................................
    flatten = _arraymethod('flatten')
    #
    def ravel(self):
        """Returns a 1D version of self, as a view."""
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
    def reshape (self, *s):
        """Reshape the array to shape s.

        Returns
        -------
        A new masked array.

        Notes
        -----
        If you want to modify the shape in place, please use
        ``a.shape = s``

        """
        result = self._data.reshape(*s).view(type(self))
        result.__dict__.update(self.__dict__)
        if result._mask is not nomask:
            result._mask = self._mask.copy()
            result._mask.shape = result.shape
        return result
    #
    def resize(self, newshape, refcheck=True, order=False):
        """Attempt to modify the size and the shape of the array in place.

        The array must own its own memory and not be referenced by
        other arrays.

        Returns
        -------
        None.

        """
        try:
            self._data.resize(newshape, refcheck, order)
            if self.mask is not nomask:
                self._mask.resize(newshape, refcheck, order)
        except ValueError:
            raise ValueError("Cannot resize an array that has been referenced "
                             "or is referencing another array in this way.\n"
                             "Use the resize function.")
        return None
    #
    def put(self, indices, values, mode='raise'):
        """Set storage-indexed locations to corresponding values.

        a.put(values, indices, mode) sets a.flat[n] = values[n] for
        each n in indices.  If ``values`` is shorter than ``indices``
        then it will repeat.  If ``values`` has some masked values, the
        initial mask is updated in consequence, else the corresponding
        values are unmasked.

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
    #............................................
    def all(self, axis=None, out=None):
        """Return True if all entries along the given axis are True,
        False otherwise.  Masked values are considered as True during
        computation.

        Parameter
        ----------
            axis : int, optional
                Axis along which the operation is performed.  If None,
                the operation is performed on a flatten array
            out : {MaskedArray}, optional
                Alternate optional output.  If not None, out should be
                a valid MaskedArray of the same shape as the output of
                self._data.all(axis).

        Returns            A masked array, where the mask is True if all data along
        -------
        the axis are masked.

        Notes
        -----
        An exception is raised if ``out`` is not None and not of the
        same type as self.

        """
        if out is None:
            d = self.filled(True).all(axis=axis).view(type(self))
            if d.ndim > 0:
                d.__setmask__(self._mask.all(axis))
            return d
        elif type(out) is not type(self):
            raise TypeError("The external array should have " \
                            "a type %s (got %s instead)" %\
                            (type(self), type(out)))
        self.filled(True).all(axis=axis, out=out)
        if out.ndim:
            out.__setmask__(self._mask.all(axis))
        return out


    def any(self, axis=None, out=None):
        """Returns True if at least one entry along the given axis is
        True.

        Returns False if all entries are False.
        Masked values are considered as True during computation.

        Parameter
        ----------
            axis : int, optional
                Axis along which the operation is performed.
                If None, the operation is performed on a flatten array
            out : {MaskedArray}, optional
                Alternate optional output.  If not None, out should be
                a valid MaskedArray of the same shape as the output of
                self._data.all(axis).

        Returns            A masked array, where the mask is True if all data along
        -------
        the axis are masked.

        Notes
        -----
        An exception is raised if ``out`` is not None and not of the
        same type as self.

        """
        if out is None:
            d = self.filled(False).any(axis=axis).view(type(self))
            if d.ndim > 0:
                d.__setmask__(self._mask.all(axis))
            return d
        elif type(out) is not type(self):
            raise TypeError("The external array should have a type %s "\
                            "(got %s instead)" %\
                            (type(self), type(out)))
        self.filled(False).any(axis=axis, out=out)
        if out.ndim:
            out.__setmask__(self._mask.all(axis))
        return out


    def nonzero(self):
        """Return the indices of the elements of a that are not zero
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
    #............................................
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

        Return the sum along the offset diagonal of the array's
        indicated `axis1` and `axis2`.

        """
        # TODO: What are we doing with `out`?
        m = self._mask
        if m is nomask:
            result = super(MaskedArray, self).trace(offset=offset, axis1=axis1,
                                                    axis2=axis2, out=out)
            return result.astype(dtype)
        else:
            D = self.diagonal(offset=offset, axis1=axis1, axis2=axis2)
            return D.astype(dtype).filled(0).sum(axis=None)
    #............................................
    def sum(self, axis=None, dtype=None):
        """Sum the array over the given axis.

        Masked elements are set to 0 internally.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : dtype, optional
            Datatype for the intermediary computation. If not given,
            the current dtype is used instead.

        """
        if self._mask is nomask:
            mask = nomask
        else:
            mask = self._mask.all(axis)
            if (not mask.ndim) and mask:
                return masked
        result = self.filled(0).sum(axis, dtype=dtype).view(type(self))
        if result.ndim > 0:
            result.__setmask__(mask)
        return result

    def cumsum(self, axis=None, dtype=None):
        """Return the cumulative sum of the elements of the array
        along the given axis.

        Masked values are set to 0 internally.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : {dtype}, optional
            Datatype for the intermediary computation. If not
            given, the current dtype is used instead.

        """
        result = self.filled(0).cumsum(axis=axis, dtype=dtype).view(type(self))
        result.__setmask__(self.mask)
        return result

    def prod(self, axis=None, dtype=None):
        """Return the product of the elements of the array along the
        given axis.

        Masked elements are set to 1 internally.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : {dtype}, optional
            Datatype for the intermediary computation. If not
            given, the current dtype is used instead.

        """
        if self._mask is nomask:
            mask = nomask
        else:
            mask = self._mask.all(axis)
            if (not mask.ndim) and mask:
                return masked
        result = self.filled(1).prod(axis=axis, dtype=dtype).view(type(self))
        if result.ndim:
            result.__setmask__(mask)
        return result

    product = prod

    def cumprod(self, axis=None, dtype=None):
        """Return the cumulative product of the elements of the array
        along the given axis.

        Masked values are set to 1 internally.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : {dtype}, optional
            Datatype for the intermediary computation. If not
            given, the current dtype is used instead.

        """
        result = self.filled(1).cumprod(axis=axis, dtype=dtype).view(type(self))
        result.__setmask__(self.mask)
        return result

    def mean(self, axis=None, dtype=None):
        """Average the array over the given axis.  Equivalent to

        a.sum(axis, dtype) / a.size(axis).

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : {dtype}, optional
            Datatype for the intermediary computation. If not
            given, the current dtype is used instead.

        """
        if self._mask is nomask:
            return super(MaskedArray, self).mean(axis=axis, dtype=dtype)
        else:
            dsum = self.sum(axis=axis, dtype=dtype)
            cnt = self.count(axis=axis)
            return dsum*1./cnt

    def anom(self, axis=None, dtype=None):
        """Return the anomalies (deviations from the average) along
        the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : {dtype}, optional
            Datatype for the intermediary computation. If not
            given, the current dtype is used instead.

        """
        m = self.mean(axis, dtype)
        if not axis:
            return (self - m)
        else:
            return (self - expand_dims(m,axis))

    def var(self, axis=None, dtype=None):
        """Return the variance, a measure of the spread of a distribution.

        The variance is the average of the squared deviations from the
        mean, i.e. var = mean((x - x.mean())**2).

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : {dtype}, optional
            Datatype for the intermediary computation. If not
            given, the current dtype is used instead.

        Notes
        -----
        The value returned is a biased estimate of the true variance.
        For the (more standard) unbiased estimate, use varu.

        """
        if self._mask is nomask:
            # TODO: Do we keep super, or var _data and take a view ?
            return super(MaskedArray, self).var(axis=axis, dtype=dtype)
        else:
            cnt = self.count(axis=axis)
            danom = self.anom(axis=axis, dtype=dtype)
            danom *= danom
            dvar = narray(danom.sum(axis) / cnt).view(type(self))
            if axis is not None:
                dvar._mask = mask_or(self._mask.all(axis), (cnt==1))
            dvar._update_from(self)
            return dvar

    def std(self, axis=None, dtype=None):
        """Return the standard deviation, a measure of the spread of a
        distribution.

        The standard deviation is the square root of the average of
        the squared deviations from the mean, i.e.

        std = sqrt(mean((x - x.mean())**2)).

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        dtype : {dtype}, optional
            Datatype for the intermediary computation.
            If not given, the current dtype is used instead.

        Notes
        -----
        The value returned is a biased estimate of the true
        standard deviation.  For the more standard unbiased
        estimate, use stdu.

        """
        dvar = self.var(axis,dtype)
        if axis is not None or dvar is not masked:
            dvar = sqrt(dvar)
        return dvar

    #............................................
    def argsort(self, axis=None, fill_value=None, kind='quicksort',
                order=None):
        """Return an ndarray of indices that sort the array along the
        specified axis.  Masked values are filled beforehand to
        fill_value.

        Parameters
        ----------
        axis : int, optional
            Axis to be indirectly sorted.
            If not given, uses a flatten version of the array.
        fill_value : {var}
            Value used to fill in the masked values.
            If not given, self.fill_value is used instead.
        kind : {string}
            Sorting algorithm (default 'quicksort')
            Possible values: 'quicksort', 'mergesort', or 'heapsort'

        Notes
        -----
        This method executes an indirect sort along the given axis
        using the algorithm specified by the kind keyword. It returns
        an array of indices of the same shape as 'a' that index data
        along the given axis in sorted order.

        The various sorts are characterized by average speed, worst
        case performance need for work space, and whether they are
        stable.  A stable sort keeps items with the same key in the
        same relative order. The three available algorithms have the
        following properties:

        |------------------------------------------------------|
        |    kind   | speed |  worst case | work space | stable|
        |------------------------------------------------------|
        |'quicksort'|   1   | O(n^2)      |     0      |   no  |
        |'mergesort'|   2   | O(n*log(n)) |    ~n/2    |   yes |
        |'heapsort' |   3   | O(n*log(n)) |     0      |   no  |
        |------------------------------------------------------|

        All the sort algorithms make temporary copies of the data when
        the sort is not along the last axis. Consequently, sorts along
        the last axis are faster and use less space than sorts along
        other axis.

        """
        if fill_value is None:
            fill_value = default_fill_value(self)
        d = self.filled(fill_value).view(ndarray)
        return d.argsort(axis=axis, kind=kind, order=order)
    #........................
    def argmin(self, axis=None, fill_value=None):
        """Return an ndarray of indices for the minimum values of a
        along the specified axis.

        Masked values are treated as if they had the value fill_value.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        fill_value : {var}, optional
            Value used to fill in the masked values.  If None, the
            output of minimum_fill_value(self._data) is used.

        """
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        d = self.filled(fill_value).view(ndarray)
        return d.argmin(axis)
    #........................
    def argmax(self, axis=None, fill_value=None):
        """Returns the array of indices for the maximum values of `a`
        along the specified axis.

        Masked values are treated as if they had the value fill_value.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        fill_value : {var}, optional
            Value used to fill in the masked values.  If None, the
            output of maximum_fill_value(self._data) is used.

        """
        if fill_value is None:
            fill_value = maximum_fill_value(self._data)
        d = self.filled(fill_value).view(ndarray)
        return d.argmax(axis)

    def sort(self, axis=-1, kind='quicksort', order=None,
             endwith=True, fill_value=None):
        """Sort along the given axis.

        Parameters
        ----------
        axis : int
            Axis to be indirectly sorted.
        kind : {string}
            Sorting algorithm (default 'quicksort')
            Possible values: 'quicksort', 'mergesort', or 'heapsort'.
        order : {var}
            If a has fields defined, then the order keyword can be
            the field name to sort on or a list (or tuple) of
            field names to indicate the order that fields should
            be used to define the sort.
        fill_value : {var}
            Value used to fill in the masked values.  If None, use
            the the output of minimum_fill_value().
        endwith : bool
            Whether missing values (if any) should be forced in
            the upper indices (at the end of the array) (True) or
            lower indices (at the beginning).

        Returns
        -------
        When used as method, returns None.
            When used as a function, returns an array.

        Notes
        -----
        This method sorts 'a' in place along the given axis using
        the algorithm specified by the kind keyword.

        The various sorts may characterized by average speed,
        worst case performance need for work space, and whether
        they are stable.  A stable sort keeps items with the same
        key in the same relative order and is most useful when
        used w/ argsort where the key might differ from the items
        being sorted.  The three available algorithms have the
        following properties:

        |------------------------------------------------------|
        |    kind   | speed |  worst case | work space | stable|
        |------------------------------------------------------|
        |'quicksort'|   1   | O(n^2)      |     0      |   no  |
        |'mergesort'|   2   | O(n*log(n)) |    ~n/2    |   yes |
        |'heapsort' |   3   | O(n*log(n)) |     0      |   no  |
        |------------------------------------------------------|

        """
        if self._mask is nomask:
            ndarray.sort(self,axis=axis, kind=kind, order=order)
        else:
            if fill_value is None:
                if endwith:
                    filler = minimum_fill_value(self)
                else:
                    filler = maximum_fill_value(self)
            else:
                filler = fill_value
            idx = numpy.indices(self.shape)
            idx[axis] = self.filled(filler).argsort(axis=axis,kind=kind,order=order)
            idx_l = idx.tolist()
            tmp_mask = self._mask[idx_l].flat
            tmp_data = self._data[idx_l].flat
            self.flat = tmp_data
            self._mask.flat = tmp_mask
        return

    #............................................
    def min(self, axis=None, fill_value=None):
        """Return the minimum of a along the given axis.

        Masked values are filled with fill_value.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        fill_value : {var}, optional
            Value used to fill in the masked values.
            If None, use the the output of minimum_fill_value().

    """
        mask = self._mask
        # Check all/nothing case ......
        if mask is nomask:
            return super(MaskedArray, self).min(axis=axis)
        elif (not mask.ndim) and mask:
            return masked
        # Get the mask ................
        if axis is None:
            mask = umath.logical_and.reduce(mask.flat)
        else:
            mask = umath.logical_and.reduce(mask, axis=axis)
        # Skip if all masked ..........
        if not mask.ndim and mask:
            return masked
        # Get the fill value ...........
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        # Get the data ................
        result = self.filled(fill_value).min(axis=axis).view(type(self))
        if result.ndim > 0:
            result._mask = mask
        return result

    def mini(self, axis=None):
        if axis is None:
            return minimum(self)
        else:
            return minimum.reduce(self, axis)

    #........................
    def max(self, axis=None, fill_value=None):
        """Return the maximum/a along the given axis.

        Masked values are filled with fill_value.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        fill_value : {var}, optional
            Value used to fill in the masked values.
            If None, use the the output of maximum_fill_value().
        """
        mask = self._mask
        # Check all/nothing case ......
        if mask is nomask:
            return super(MaskedArray, self).max(axis=axis)
        elif (not mask.ndim) and mask:
            return masked
        # Check the mask ..............
        if axis is None:
            mask = umath.logical_and.reduce(mask.flat)
        else:
            mask = umath.logical_and.reduce(mask, axis=axis)
        # Skip if all masked ..........
        if not mask.ndim and mask:
            return masked
        # Get the fill value ..........
        if fill_value is None:
            fill_value = maximum_fill_value(self)
        # Get the data ................
        result = self.filled(fill_value).max(axis=axis).view(type(self))
        if result.ndim > 0:
            result._mask = mask
        return result
    #........................
    def ptp(self, axis=None, fill_value=None):
        """Return the visible data range (max-min) along the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to perform the operation.
            If None, applies to a flattened version of the array.
        fill_value : {var}, optional
            Value used to fill in the masked values.  If None, the
            maximum uses the maximum default, the minimum uses the
            minimum default.

        """
        return self.max(axis, fill_value) - self.min(axis, fill_value)

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
        """Copy the data portion of the array to a hierarchical python
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
            return tuple([None]*dtypesize)
        elif nbdims == 1:
            maskedidx = _mask.nonzero()[0].tolist()
            if dtypesize:
                nodata = tuple([None]*dtypesize)
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
        """Return a copy of array data as a Python string containing the raw
        bytes in the array.

        Parameters
        ----------
        fill_value : {var}, optional
            Value used to fill in the masked values.
            If None, uses self.fill_value instead.
        order : {string}
            Order of the data item in the copy {"C","F","A"}.
            "C"       -- C order (row major)
            "Fortran" -- Fortran order (column major)
            "Any"     -- Current order of array.
            None      -- Same as "Any"

        """
        return self.filled(fill_value).tostring(order=order)
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
        self._mask.__setstate__((shp, dtype(bool), isf, msk))
        self.fill_value = flv
    #
    def __reduce__(self):
        """Return a 3-tuple for pickling a MaskedArray.

        """
        return (_mareconstruct,
                (self.__class__, self._baseclass, (0,), 'b', ),
                self.__getstate__())


def _mareconstruct(subtype, baseclass, baseshape, basetype,):
    """Internal function that builds a new MaskedArray from the
    information stored in a pickle.

    """
    _data = ndarray.__new__(baseclass, baseshape, basetype)
    _mask = ndarray.__new__(ndarray, baseshape, 'b1')
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)


#####--------------------------------------------------------------------------
#---- --- Shortcuts ---
#####---------------------------------------------------------------------------
def isMaskedArray(x):
    "Is x a masked array, that is, an instance of MaskedArray?"
    return isinstance(x, MaskedArray)
isarray = isMaskedArray
isMA = isMaskedArray  #backward compatibility
# We define the masked singleton as a float for higher precedence...
# Note that it can be tricky sometimes w/ type comparison
masked_singleton = MaskedArray(0, dtype=float_, mask=True)
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
    #TODO: we should try to put 'order' somwehere
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy, subok=subok,
                       keep_mask=keep_mask, hard_mask=hard_mask,
                       fill_value=fill_value, ndmin=ndmin, shrink=shrink)
array.__doc__ = masked_array.__doc__

def is_masked(x):
    """Does x have masked values?"""
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
def min(array, axis=None, out=None):
    """Return the minima along the given axis.

    If `axis` is None, applies to the flattened array.

    """
    if out is not None:
        raise TypeError("Output arrays Unsupported for masked arrays")
    if axis is None:
        return minimum(array)
    else:
        return minimum.reduce(array, axis)
min.__doc__ = MaskedArray.min.__doc__
#............................
def max(obj, axis=None, out=None):
    if out is not None:
        raise TypeError("Output arrays Unsupported for masked arrays")
    if axis is None:
        return maximum(obj)
    else:
        return maximum.reduce(obj, axis)
max.__doc__ = MaskedArray.max.__doc__
#.............................
def ptp(obj, axis=None):
    """a.ptp(axis=None) =  a.max(axis)-a.min(axis)"""
    try:
        return obj.max(axis)-obj.min(axis)
    except AttributeError:
        return max(obj, axis=axis) - min(obj, axis=axis)
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
        self._methodname = methodname
        self.__doc__ = self.getdoc()
    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        try:
            return getattr(MaskedArray, self._methodname).__doc__
        except:
            return getattr(numpy, self._methodname).__doc__
    def __call__(self, a, *args, **params):
        if isinstance(a, MaskedArray):
            return getattr(a, self._methodname).__call__(*args, **params)
        #FIXME ----
        #As x is not a MaskedArray, we transform it to a ndarray with asarray
        #... and call the corresponding method.
        #Except that sometimes it doesn't work (try reshape([1,2,3,4],(2,2)))
        #we end up with a "SystemError: NULL result without error in PyObject_Call"
        #A dirty trick is then to call the initial numpy function...
        method = getattr(narray(a, copy=False), self._methodname)
        try:
            return method(*args, **params)
        except SystemError:
            return getattr(numpy,self._methodname).__call__(a, *args, **params)

all = _frommethod('all')
anomalies = anom = _frommethod('anom')
any = _frommethod('any')
conjugate = _frommethod('conjugate')
ids = _frommethod('ids')
nonzero = _frommethod('nonzero')
diagonal = _frommethod('diagonal')
maximum = _maximum_operation()
mean = _frommethod('mean')
minimum = _minimum_operation ()
product = _frommethod('prod')
ptp = _frommethod('ptp')
ravel = _frommethod('ravel')
repeat = _frommethod('repeat')
std = _frommethod('std')
sum = _frommethod('sum')
swapaxes = _frommethod('swapaxes')
take = _frommethod('take')
var = _frommethod('var')
compress = _frommethod('compress')

#..............................................................................
def power(a, b, third=None):
    """Computes a**b elementwise.

    Masked values are set to 1.

    """
    if third is not None:
        raise MAError, "3-argument power not supported."
    ma = getmask(a)
    mb = getmask(b)
    m = mask_or(ma, mb)
    fa = getdata(a)
    fb = getdata(b)
    if fb.dtype.char in typecodes["Integer"]:
        return masked_array(umath.power(fa, fb), m)
    md = make_mask((fa < 0), shrink=True)
    m = mask_or(m, md)
    if m is nomask:
        return masked_array(umath.power(fa, fb))
    else:
        fa = fa.copy()
        fa[(fa < 0)] = 1
        return masked_array(umath.power(fa, fb), m)

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
    a = narray(a, copy=False, subok=True)
    if fill_value is None:
        if endwith:
            filler = minimum_fill_value(a)
        else:
            filler = maximum_fill_value(a)
    else:
        filler = fill_value
#    return
    indx = numpy.indices(a.shape).tolist()
    indx[axis] = filled(a,filler).argsort(axis=axis,kind=kind,order=order)
    return a[indx]
sort.__doc__ = MaskedArray.sort.__doc__


def compressed(x):
    """Return a 1-D array of all the non-masked data."""
    if getmask(x) is nomask:
        return x
    else:
        return x.compressed()

def concatenate(arrays, axis=0):
    "Concatenate the arrays along the given axis."
    d = numpy.concatenate([getdata(a) for a in arrays], axis)
    rcls = get_masked_subclass(*arrays)
    data = d.view(rcls)
    # Check whether one of the arrays has a non-empty mask...
    for x in arrays:
        if getmask(x) is not nomask:
            break
    else:
        return data
    # OK, so we have to concatenate the masks
    dm = numpy.concatenate([getmaskarray(a) for a in arrays], axis)
    # If we decide to keep a '_shrinkmask' option, we want to check that ...
    # ... all of them are True, and then check for dm.any()
#    shrink = numpy.logical_or.reduce([getattr(a,'_shrinkmask',True) for a in arrays])
#    if shrink and not dm.any():
    if not dm.any():
        data._mask = nomask
    else:
        data._mask = dm.reshape(d.shape)
    return data

def count(a, axis = None):
    return masked_array(a, copy=False).count(axis)
count.__doc__ = MaskedArray.count.__doc__


def expand_dims(x,axis):
    """Expand the shape of the array by including a new axis before
    the given one.

    """
    result = n_expand_dims(x,axis)
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
            a.mask = numpy.zeros(a.shape, dtype=bool_)
            numpy.putmask(a._mask, mask, valmask)
    elif a._hardmask:
        if valmask is not nomask:
            m = a._mask.copy()
            numpy.putmask(m, mask, valmask)
            a.mask |= m
    else:
        if valmask is nomask:
            valmask = getmaskarray(values)
        numpy.putmask(a._mask, mask, valmask)
    numpy.putmask(a._data, mask, valdata)
    return

def transpose(a,axes=None):
    """Return a view of the array with dimensions permuted according to axes,
    as a masked array.

    If ``axes`` is None (default), the output view has reversed
    dimensions compared to the original.

    """
    #We can't use 'frommethod', as 'transpose' doesn't take keywords
    try:
        return a.transpose(axes)
    except AttributeError:
        return narray(a, copy=False).transpose(axes).view(MaskedArray)

def reshape(a, new_shape):
    """Change the shape of the array a to new_shape."""
    #We can't use 'frommethod', it whine about some parameters. Dmmit.
    try:
        return a.reshape(new_shape)
    except AttributeError:
        return narray(a, copy=False).reshape(new_shape).view(MaskedArray)

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
        m = numpy.resize(m, new_shape)
    result = numpy.resize(x, new_shape).view(get_masked_subclass(x))
    if result.ndim:
        result._mask = m
    return result


#................................................
def rank(obj):
    "maskedarray version of the numpy function."
    return fromnumeric.rank(getdata(obj))
rank.__doc__ = numpy.rank.__doc__
#
def shape(obj):
    "maskedarray version of the numpy function."
    return fromnumeric.shape(getdata(obj))
shape.__doc__ = numpy.shape.__doc__
#
def size(obj, axis=None):
    "maskedarray version of the numpy function."
    return fromnumeric.size(getdata(obj), axis)
size.__doc__ = numpy.size.__doc__
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
    fc = filled(condition, 0).astype(bool_)
    notfc = numpy.logical_not(fc)
    # Get the data ......................................
    xv = getdata(x)
    yv = getdata(y)
    if x is masked:
        ndtype = yv.dtype
        xm = numpy.ones(fc.shape, dtype=MaskType)
    elif y is masked:
        ndtype = xv.dtype
        ym = numpy.ones(fc.shape, dtype=MaskType)
    else:
        ndtype = numpy.max([xv.dtype, yv.dtype])
        xm = getmask(x)
    d = numpy.empty(fc.shape, dtype=ndtype).view(MaskedArray)
    numpy.putmask(d._data, fc, xv.astype(ndtype))
    numpy.putmask(d._data, notfc, yv.astype(ndtype))
    d._mask = numpy.zeros(fc.shape, dtype=MaskType)
    numpy.putmask(d._mask, fc, getmask(x))
    numpy.putmask(d._mask, notfc, getmask(y))
    d._mask |= getmaskarray(condition)
    if not d._mask.any():
        d._mask = nomask
    return d

def choose (indices, t, out=None, mode='raise'):
    "Return array shaped like indices with elements chosen from t"
    #TODO: implement options `out` and `mode`, if possible.
    def fmask (x):
        "Returns the filled array, or True if masked."
        if x is masked:
            return 1
        return filled(x)
    def nmask (x):
        "Returns the mask, True if ``masked``, False if ``nomask``."
        if x is masked:
            return 1
        m = getmask(x)
        if m is nomask:
            return 0
        return m
    c = filled(indices, 0)
    masks = [nmask(x) for x in t]
    a = [fmask(x) for x in t]
    d = numpy.choose(c, a)
    m = numpy.choose(c, masks)
    m = make_mask(mask_or(m, getmask(indices)), copy=0, shrink=True)
    return masked_array(d, mask=m)

def round_(a, decimals=0, out=None):
    """Return a copy of a, rounded to 'decimals' places.

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
        result = fromnumeric.round_(getdata(a), decimals, out)
        if isinstance(a,MaskedArray):
            result = result.view(type(a))
            result._mask = a._mask
        else:
            result = result.view(MaskedArray)
        return result
    else:
        fromnumeric.round_(getdata(a), decimals, out)
        if hasattr(out, '_mask'):
            out._mask = getmask(a)
        return out

def arange(stop, start=None, step=1, dtype=None):
    "maskedarray version of the numpy function."
    return numpy.arange(stop, start, step, dtype).view(MaskedArray)
arange.__doc__ = numpy.arange.__doc__

def inner(a, b):
    "maskedarray version of the numpy function."
    fa = filled(a, 0)
    fb = filled(b, 0)
    if len(fa.shape) == 0:
        fa.shape = (1,)
    if len(fb.shape) == 0:
        fb.shape = (1,)
    return numpy.inner(fa, fb).view(MaskedArray)
inner.__doc__ = numpy.inner.__doc__
inner.__doc__ += doc_note("Masked values are replaced by 0.")
innerproduct = inner

def outer(a, b):
    "maskedarray version of the numpy function."
    fa = filled(a, 0).ravel()
    fb = filled(b, 0).ravel()
    d = numeric.outer(fa, fb)
    ma = getmask(a)
    mb = getmask(b)
    if ma is nomask and mb is nomask:
        return masked_array(d)
    ma = getmaskarray(a)
    mb = getmaskarray(b)
    m = make_mask(1-numeric.outer(1-ma, 1-mb), copy=0)
    return masked_array(d, mask=m)
outer.__doc__ = numpy.outer.__doc__
outer.__doc__ += doc_note("Masked values are replaced by 0.")
outerproduct = outer

def allequal (a, b, fill_value=True):
    """Return True if all entries of a and b are equal, using
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

def allclose (a, b, fill_value=True, rtol=1.e-5, atol=1.e-8):
    """ Return True if all elements of a and b are equal subject to
    given tolerances.

    If fill_value is True, masked values are considered equal.
    If fill_value is False, masked values considered unequal.
    The relative error rtol should be positive and << 1.0
    The absolute error atol comes into play for those elements of b
    that are very small or zero; it says how small `a` must be also.

    """
    m = mask_or(getmask(a), getmask(b))
    d1 = getdata(a)
    d2 = getdata(b)
    x = filled(array(d1, copy=0, mask=m), fill_value).astype(float)
    y = filled(array(d2, copy=0, mask=m), 1).astype(float)
    d = umath.less_equal(umath.absolute(x-y), atol + rtol * umath.absolute(y))
    return fromnumeric.alltrue(fromnumeric.ravel(d))

#..............................................................................
def asarray(a, dtype=None):
    """asarray(data, dtype) = array(data, dtype, copy=0, subok=0)

    Return a as a MaskedArray object of the given dtype.
    If dtype is not given or None, is is set to the dtype of a.
    No copy is performed if a is already an array.
    Subclasses are converted to the base class MaskedArray.

    """
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=False)

def asanyarray(a, dtype=None):
    """asanyarray(data, dtype) = array(data, dtype, copy=0, subok=1)

    Return a as an masked array.
    If dtype is not given or None, is is set to the dtype of a.
    No copy is performed if a is already an array.
    Subclasses are conserved.

    """
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=True)


def empty(new_shape, dtype=float):
    "maskedarray version of the numpy function."
    return numpy.empty(new_shape, dtype).view(MaskedArray)
empty.__doc__ = numpy.empty.__doc__

def empty_like(a):
    "maskedarray version of the numpy function."
    return numpy.empty_like(a).view(MaskedArray)
empty_like.__doc__ = numpy.empty_like.__doc__

def ones(new_shape, dtype=float):
    "maskedarray version of the numpy function."
    return numpy.ones(new_shape, dtype).view(MaskedArray)
ones.__doc__ = numpy.ones.__doc__

def zeros(new_shape, dtype=float):
    "maskedarray version of the numpy function."
    return numpy.zeros(new_shape, dtype).view(MaskedArray)
zeros.__doc__ = numpy.zeros.__doc__

#####--------------------------------------------------------------------------
#---- --- Pickling ---
#####--------------------------------------------------------------------------
def dump(a,F):
    """Pickle the MaskedArray `a` to the file `F`.  `F` can either be
    the handle of an exiting file, or a string representing a file
    name.

    """
    if not hasattr(F,'readline'):
        F = open(F,'w')
    return cPickle.dump(a,F)

def dumps(a):
    """Return a string corresponding to the pickling of the
    MaskedArray.

    """
    return cPickle.dumps(a)

def load(F):
    """Wrapper around ``cPickle.load`` which accepts either a
    file-like object or a filename.

    """
    if not hasattr(F, 'readline'):
        F = open(F,'r')
    return cPickle.load(F)

def loads(strg):
    "Load a pickle from the current string."""
    return cPickle.loads(strg)

################################################################################

if 1:
    from testutils import assert_equal
    if 1:
        mtype = [('f',float_),('s','|S3')]
        x = array([(1,'a'),(2,'b'),(numpy.pi,'pi')], dtype=mtype)
        x[0] = (10,'A')
        (xf, xs) = (x['f'], x['s'])
        assert_equal(xf.data, [10,2,numpy.pi])
        assert_equal(xf.dtype, float_)
        assert_equal(xs.data, ['A', 'b', 'pi'])
        assert_equal(xs.dtype, '|S3')
