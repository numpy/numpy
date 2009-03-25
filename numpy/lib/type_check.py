## Automatically adapted for numpy Sep 19, 2005 by convertcode.py

__all__ = ['iscomplexobj','isrealobj','imag','iscomplex',
           'isreal','nan_to_num','real','real_if_close',
           'typename','asfarray','mintypecode','asscalar',
           'common_type']

import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, asanyarray, array, isnan, \
                obj2sctype, zeros
from ufunclike import isneginf, isposinf

_typecodes_by_elsize = 'GDFgdfQqLlIiHhBb?'

def mintypecode(typechars,typeset='GDFgdf',default='d'):
    """ Return a minimum data type character from typeset that
    handles all typechars given

    The returned type character must be the smallest size such that
    an array of the returned type can handle the data from an array of
    type t for each t in typechars (or if typechars is an array,
    then its dtype.char).

    If the typechars does not intersect with the typeset, then default
    is returned.

    If t in typechars is not a string then t=asarray(t).dtype.char is
    applied.
    """
    typecodes = [(type(t) is type('') and t) or asarray(t).dtype.char\
                 for t in typechars]
    intersection = [t for t in typecodes if t in typeset]
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    l = []
    for t in intersection:
        i = _typecodes_by_elsize.index(t)
        l.append((i,t))
    l.sort()
    return l[0][1]

def asfarray(a, dtype=_nx.float_):
    """
    Return an array converted to a float type.

    Parameters
    ----------
    a : array_like
        The input array.
    dtype : str or dtype object, optional
        Float type code to coerce input array `a`.  If `dtype` is one of the
        'int' dtypes, it is replaced with float64.

    Returns
    -------
    out : ndarray
        The input `a` as a float ndarray.

    Examples
    --------
    >>> np.asfarray([2, 3])
    array([ 2.,  3.])
    >>> np.asfarray([2, 3], dtype='float')
    array([ 2.,  3.])
    >>> np.asfarray([2, 3], dtype='int8')
    array([ 2.,  3.])

    """
    dtype = _nx.obj2sctype(dtype)
    if not issubclass(dtype, _nx.inexact):
        dtype = _nx.float_
    return asarray(a,dtype=dtype)

def real(val):
    """
    Return the real part of the elements of the array.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray
        If `val` is real, the type of `val` is used for the output.  If `val`
        has complex elements, the returned type is float.

    See Also
    --------
    real_if_close, imag, angle

    Examples
    --------
    >>> a = np.array([1+2j,3+4j,5+6j])
    >>> a.real
    array([ 1.,  3.,  5.])
    >>> a.real = 9
    >>> a
    array([ 9.+2.j,  9.+4.j,  9.+6.j])
    >>> a.real = np.array([9,8,7])
    >>> a
    array([ 9.+2.j,  8.+4.j,  7.+6.j])

    """
    return asanyarray(val).real

def imag(val):
    """
    Return the imaginary part of array.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray, real or int
        Real part of each element, same shape as `val`.

    """
    return asanyarray(val).imag

def iscomplex(x):
    """
    Returns a bool array, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray, bool
        Output array.

    See Also
    --------
    isreal: Returns a bool array, where True if input element is real.
    iscomplexobj: Return True if x is a complex type or an array of complex
                  numbers.

    Examples
    --------
    >>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
    array([ True, False, False, False, False,  True], dtype=bool)

    """
    ax = asanyarray(x)
    if issubclass(ax.dtype.type, _nx.complexfloating):
        return ax.imag != 0
    res = zeros(ax.shape, bool)
    return +res  # convet to array-scalar if needed

def isreal(x):
    """
    Returns a bool array, where True if input element is real.

    If the input value has a complex type but with complex part zero, the
    return value is True.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray, bool
        Boolean array of same shape as `x`.

    See Also
    --------
    iscomplex: Return a bool array, where True if input element is complex
               (non-zero imaginary part).
    isrealobj: Return True if x is not a complex type.

    Examples
    --------
    >>> np.isreal([1+1j, 1+0j, 4.5, 3, 2, 2j])
    >>> array([False,  True,  True,  True,  True, False], dtype=bool)

    """
    return imag(x) == 0

def iscomplexobj(x):
    """
    Return True if x is a complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True
    if the data type is complex.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    y : bool
        The return value, True if `x` is of a complex type.

    See Also
    --------
    isrealobj, iscomplex

    Examples
    --------
    >>> np.iscomplexobj(1)
    False
    >>> np.iscomplexobj(1+0j)
    True
    np.iscomplexobj([3, 1+0j, True])
    True

    """
    return issubclass( asarray(x).dtype.type, _nx.complexfloating)

def isrealobj(x):
    """
    Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    y : bool
        The return value, False if `x` is of a complex type.

    See Also
    --------
    iscomplexobj, isreal

    Examples
    --------
    >>> np.isrealobj(1)
    True
    >>> np.isrealobj(1+0j)
    False
    >>> np.isrealobj([3, 1+0j, True])
    False

    """
    return not issubclass( asarray(x).dtype.type, _nx.complexfloating)

#-----------------------------------------------------------------------------

def _getmaxmin(t):
    import getlimits
    f = getlimits.finfo(t)
    return f.max, f.min

def nan_to_num(x):
    """
    Replace nan with zero and inf with finite numbers.

    Returns an array or scalar replacing Not a Number (NaN) with zero,
    (positive) infinity with a very large number and negative infinity
    with a very small (or negative) number.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    out : ndarray, float
        Array with the same shape as `x` and dtype of the element in `x`  with
        the greatest precision. NaN is replaced by zero, and infinity
        (-infinity) is replaced by the largest (smallest or most negative)
        floating point value that fits in the output dtype. All finite numbers
        are upcast to the output dtype (default float64).

    See Also
    --------
    isinf : Shows which elements are negative or negative infinity.
    isneginf : Shows which elements are negative infinity.
    isposinf : Shows which elements are positive infinity.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite : Shows which elements are finite (not NaN, not infinity)

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.


    Examples
    --------
    >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
    >>> np.nan_to_num(x)
    array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000,
            -1.28000000e+002,   1.28000000e+002])

    """
    try:
        t = x.dtype.type
    except AttributeError:
        t = obj2sctype(type(x))
    if issubclass(t, _nx.complexfloating):
        return nan_to_num(x.real) + 1j * nan_to_num(x.imag)
    else:
        try:
            y = x.copy()
        except AttributeError:
            y = array(x)
    if not issubclass(t, _nx.integer):
        if not y.shape:
            y = array([x])
            scalar = True
        else:
            scalar = False
        are_inf = isposinf(y)
        are_neg_inf = isneginf(y)
        are_nan = isnan(y)
        maxf, minf = _getmaxmin(y.dtype.type)
        y[are_nan] = 0
        y[are_inf] = maxf
        y[are_neg_inf] = minf
        if scalar:
            y = y[0]
    return y

#-----------------------------------------------------------------------------

def real_if_close(a,tol=100):
    """
    If complex input returns a real array if complex parts are close to zero.

    "Close to zero" is defined as `tol` * (machine epsilon of the type for
    `a`).

    Parameters
    ----------
    a : array_like
        Input array.
    tol : float
        Tolerance in machine epsilons for the complex part of the elements
        in the array.

    Returns
    -------
    out : ndarray
        If `a` is real, the type of `a` is used for the output.  If `a`
        has complex elements, the returned type is float.

    See Also
    --------
    real, imag, angle

    Notes
    -----
    Machine epsilon varies from machine to machine and between data types
    but Python floats on most platforms have a machine epsilon equal to
    2.2204460492503131e-16.  You can use 'np.finfo(np.float).eps' to print
    out the machine epsilon for floats.

    Examples
    --------
    >>> np.finfo(np.float).eps
    2.2204460492503131e-16

    >>> np.real_if_close([2.1 + 4e-14j], tol=1000)
    array([ 2.1])
    >>> np.real_if_close([2.1 + 4e-13j], tol=1000)
    array([ 2.1 +4.00000000e-13j])

    """
    a = asanyarray(a)
    if not issubclass(a.dtype.type, _nx.complexfloating):
        return a
    if tol > 1:
        import getlimits
        f = getlimits.finfo(a.dtype.type)
        tol = f.eps * tol
    if _nx.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a


def asscalar(a):
    """
    Convert an array of size 1 to its scalar equivalent.

    Parameters
    ----------
    a : ndarray
        Input array of size 1.

    Returns
    -------
    out : scalar
        Scalar representation of `a`. The input data type is preserved.

    Examples
    --------
    >>> np.asscalar(np.array([24]))
    24

    """
    return a.item()

#-----------------------------------------------------------------------------

_namefromtype = {'S1' : 'character',
                 '?' : 'bool',
                 'b' : 'signed char',
                 'B' : 'unsigned char',
                 'h' : 'short',
                 'H' : 'unsigned short',
                 'i' : 'integer',
                 'I' : 'unsigned integer',
                 'l' : 'long integer',
                 'L' : 'unsigned long integer',
                 'q' : 'long long integer',
                 'Q' : 'unsigned long long integer',
                 'f' : 'single precision',
                 'd' : 'double precision',
                 'g' : 'long precision',
                 'F' : 'complex single precision',
                 'D' : 'complex double precision',
                 'G' : 'complex long double precision',
                 'S' : 'string',
                 'U' : 'unicode',
                 'V' : 'void',
                 'O' : 'object'
                 }

def typename(char):
    """
    Return a description for the given data type code.

    Parameters
    ----------
    char : str
        Data type code.

    Returns
    -------
    out : str
        Description of the input data type code.

    See Also
    --------
    typecodes
    dtype

    """
    return _namefromtype[char]

#-----------------------------------------------------------------------------

#determine the "minimum common type" for a group of arrays.
array_type = [[_nx.single, _nx.double, _nx.longdouble],
              [_nx.csingle, _nx.cdouble, _nx.clongdouble]]
array_precision = {_nx.single : 0,
                   _nx.double : 1,
                   _nx.longdouble : 2,
                   _nx.csingle : 0,
                   _nx.cdouble : 1,
                   _nx.clongdouble : 2}
def common_type(*arrays):
    """
    Return the inexact scalar type which is most common in a list of arrays.

    The return type will always be an inexact scalar type, even if all the
    arrays are integer arrays

    Parameters
    ----------
    array1, array2, ... : ndarray
        Input arrays.

    Returns
    -------
    out : data type code
        Data type code.

    See Also
    --------
    dtype

    Examples
    --------
    >>> np.common_type(np.arange(4), np.array([45,6]), np.array([45.0, 6.0]))
    <type 'numpy.float64'>

    """
    is_complex = False
    precision = 0
    for a in arrays:
        t = a.dtype.type
        if iscomplexobj(a):
            is_complex = True
        if issubclass(t, _nx.integer):
            p = 1
        else:
            p = array_precision.get(t, None)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]
