"""
Wrapper functions to more user-friendly calling of certain math functions
whose output data-type is different than the input data-type in certain
domains of the input.

For example, for functions like log() with branch cuts, the versions in this
module provide the mathematically valid answers in the complex plane:

>>> import math
>>> from numpy.lib import scimath
>>> scimath.log(-math.exp(1)) == (1+1j*math.pi)
True

Similarly, sqrt(), other base logarithms, power() and trig functions are
correctly handled.  See their respective docstrings for specific examples.
"""

__all__ = ['sqrt', 'log', 'log2', 'logn','log10', 'power', 'arccos',
           'arcsin', 'arctanh']

import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.lib.type_check import isreal

_ln2 = nx.log(2.0)

def _tocomplex(arr):
    """Convert its input `arr` to a complex array.

    The input is returned as a complex array of the smallest type that will fit
    the original data: types like single, byte, short, etc. become csingle,
    while others become cdouble.

    A copy of the input is always made.

    Parameters
    ----------
    arr : array

    Returns
    -------
    array
        An array with the same input data as the input but in complex form.

    Examples
    --------

    First, consider an input of type short:

    >>> a = np.array([1,2,3],np.short)

    >>> ac = np.lib.scimath._tocomplex(a); ac
    array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)

    >>> ac.dtype
    dtype('complex64')

    If the input is of type double, the output is correspondingly of the
    complex double type as well:

    >>> b = np.array([1,2,3],np.double)

    >>> bc = np.lib.scimath._tocomplex(b); bc
    array([ 1.+0.j,  2.+0.j,  3.+0.j])

    >>> bc.dtype
    dtype('complex128')

    Note that even if the input was complex to begin with, a copy is still
    made, since the astype() method always copies:

    >>> c = np.array([1,2,3],np.csingle)

    >>> cc = np.lib.scimath._tocomplex(c); cc
    array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)

    >>> c *= 2; c
    array([ 2.+0.j,  4.+0.j,  6.+0.j], dtype=complex64)

    >>> cc
    array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)
    """
    if issubclass(arr.dtype.type, (nt.single, nt.byte, nt.short, nt.ubyte,
                                   nt.ushort,nt.csingle)):
        return arr.astype(nt.csingle)
    else:
        return arr.astype(nt.cdouble)

def _fix_real_lt_zero(x):
    """Convert `x` to complex if it has real, negative components.

    Otherwise, output is just the array version of the input (via asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array

    Examples
    --------
    >>> np.lib.scimath._fix_real_lt_zero([1,2])
    array([1, 2])

    >>> np.lib.scimath._fix_real_lt_zero([-1,2])
    array([-1.+0.j,  2.+0.j])
    """
    x = asarray(x)
    if any(isreal(x) & (x<0)):
        x = _tocomplex(x)
    return x

def _fix_int_lt_zero(x):
    """Convert `x` to double if it has real, negative components.

    Otherwise, output is just the array version of the input (via asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array

    Examples
    --------
    >>> np.lib.scimath._fix_int_lt_zero([1,2])
    array([1, 2])

    >>> np.lib.scimath._fix_int_lt_zero([-1,2])
    array([-1.,  2.])
    """
    x = asarray(x)
    if any(isreal(x) & (x < 0)):
        x = x * 1.0
    return x

def _fix_real_abs_gt_1(x):
    """Convert `x` to complex if it has real components x_i with abs(x_i)>1.

    Otherwise, output is just the array version of the input (via asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array

    Examples
    --------
    >>> np.lib.scimath._fix_real_abs_gt_1([0,1])
    array([0, 1])

    >>> np.lib.scimath._fix_real_abs_gt_1([0,2])
    array([ 0.+0.j,  2.+0.j])
    """
    x = asarray(x)
    if any(isreal(x) & (abs(x)>1)):
        x = _tocomplex(x)
    return x

def sqrt(x):
    """Return the square root of x.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like output.

    Examples
    --------

    For real, non-negative inputs this works just like numpy.sqrt():
    >>> np.lib.scimath.sqrt(1)
    1.0

    >>> np.lib.scimath.sqrt([1,4])
    array([ 1.,  2.])

    But it automatically handles negative inputs:
    >>> np.lib.scimath.sqrt(-1)
    (0.0+1.0j)

    >>> np.lib.scimath.sqrt([-1,4])
    array([ 0.+1.j,  2.+0.j])

    Notes
    -----

    As the numpy.sqrt, this returns the principal square root of x, which is
    what most people mean when they use square root; the principal square root
    of x is not any number z such as z^2 = x.

    For positive numbers, the principal square root is defined as the positive
    number z such as z^2 = x.

    The principal square root of -1 is i, the principal square root of any
    negative number -x is defined a i * sqrt(x). For any non zero complex
    number, it is defined by using the following branch cut: x = r e^(i t) with
    r > 0 and -pi < t <= pi. The principal square root is then
    sqrt(r) e^(i t/2).
    """
    x = _fix_real_lt_zero(x)
    return nx.sqrt(x)

def log(x):
    """Return the natural logarithm of x.

    If x contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like

    Examples
    --------
    >>> import math
    >>> np.lib.scimath.log(math.exp(1))
    1.0

    Negative arguments are correctly handled (recall that for negative
    arguments, the identity exp(log(z))==z does not hold anymore):

    >>> np.lib.scimath.log(-math.exp(1)) == (1+1j*math.pi)
    True
    """
    x = _fix_real_lt_zero(x)
    return nx.log(x)

def log10(x):
    """Return the base 10 logarithm of x.

    If x contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like

    Examples
    --------

    (We set the printing precision so the example can be auto-tested)
    >>> np.set_printoptions(precision=4)

    >>> np.lib.scimath.log10([10**1,10**2])
    array([ 1.,  2.])


    >>> np.lib.scimath.log10([-10**1,-10**2,10**2])
    array([ 1.+1.3644j,  2.+1.3644j,  2.+0.j    ])
    """
    x = _fix_real_lt_zero(x)
    return nx.log10(x)

def logn(n, x):
    """Take log base n of x.

    If x contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like

    Examples
    --------

    (We set the printing precision so the example can be auto-tested)
    >>> np.set_printoptions(precision=4)

    >>> np.lib.scimath.logn(2,[4,8])
    array([ 2.,  3.])

    >>> np.lib.scimath.logn(2,[-4,-8,8])
    array([ 2.+4.5324j,  3.+4.5324j,  3.+0.j    ])
    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return nx.log(x)/nx.log(n)

def log2(x):
    """ Take log base 2 of x.

    If x contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like

    Examples
    --------

    (We set the printing precision so the example can be auto-tested)
    >>> np.set_printoptions(precision=4)

    >>> np.lib.scimath.log2([4,8])
    array([ 2.,  3.])

    >>> np.lib.scimath.log2([-4,-8,8])
    array([ 2.+4.5324j,  3.+4.5324j,  3.+0.j    ])
    """
    x = _fix_real_lt_zero(x)
    return nx.log(x)/_ln2

def power(x, p):
    """Return x**p.

    If x contains negative values, it is converted to the complex domain.

    If p contains negative values, it is converted to floating point.

    Parameters
    ----------
    x : array_like
    p : array_like of integers

    Returns
    -------
    array_like

    Examples
    --------
    (We set the printing precision so the example can be auto-tested)
    >>> np.set_printoptions(precision=4)

    >>> np.lib.scimath.power([2,4],2)
    array([ 4, 16])

    >>> np.lib.scimath.power([2,4],-2)
    array([ 0.25  ,  0.0625])

    >>> np.lib.scimath.power([-2,4],2)
    array([  4.+0.j,  16.+0.j])
    """
    x = _fix_real_lt_zero(x)
    p = _fix_int_lt_zero(p)
    return nx.power(x, p)

def arccos(x):
    """Compute the inverse cosine of x.

    For real x with abs(x)<=1, this returns the principal value.

    If abs(x)>1, the complex arccos() is computed.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like

    Examples
    --------
    >>> np.set_printoptions(precision=4)

    >>> np.lib.scimath.arccos(1)
    0.0

    >>> np.lib.scimath.arccos([1,2])
    array([ 0.-0.j   ,  0.+1.317j])
    """
    x = _fix_real_abs_gt_1(x)
    return nx.arccos(x)

def arcsin(x):
    """Compute the inverse sine of x.

    For real x with abs(x)<=1, this returns the principal value.

    If abs(x)>1, the complex arcsin() is computed.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like

    Examples
    --------
    (We set the printing precision so the example can be auto-tested)
    >>> np.set_printoptions(precision=4)

    >>> np.lib.scimath.arcsin(0)
    0.0

    >>> np.lib.scimath.arcsin([0,1])
    array([ 0.    ,  1.5708])
    """
    x = _fix_real_abs_gt_1(x)
    return nx.arcsin(x)

def arctanh(x):
    """Compute the inverse hyperbolic tangent of x.

    For real x with abs(x)<=1, this returns the principal value.

    If abs(x)>1, the complex arctanh() is computed.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like

    Examples
    --------
    (We set the printing precision so the example can be auto-tested)
    >>> np.set_printoptions(precision=4)

    >>> np.lib.scimath.arctanh(0)
    0.0

    >>> np.lib.scimath.arctanh([0,2])
    array([ 0.0000+0.j    ,  0.5493-1.5708j])
    """
    x = _fix_real_abs_gt_1(x)
    return nx.arctanh(x)
