"""
Objects for dealing with Chebyshev series.

This module provides a number of objects (mostly functions) useful for
dealing with Chebyshev series, including a `Chebyshev` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Constants
---------
- `chebdomain` -- Chebyshev series default domain, [-1,1].
- `chebzero` -- (Coefficients of the) Chebyshev series that evaluates
  identically to 0.
- `chebone` -- (Coefficients of the) Chebyshev series that evaluates
  identically to 1.
- `chebx` -- (Coefficients of the) Chebyshev series for the identity map,
  ``f(x) = x``.

Arithmetic
----------
- `chebadd` -- add two Chebyshev series.
- `chebsub` -- subtract one Chebyshev series from another.
- `chebmul` -- multiply two Chebyshev series.
- `chebdiv` -- divide one Chebyshev series by another.
- `chebpow` -- raise a Chebyshev series to an positive integer power
- `chebval` -- evaluate a Chebyshev series at given points.

Calculus
--------
- `chebder` -- differentiate a Chebyshev series.
- `chebint` -- integrate a Chebyshev series.

Misc Functions
--------------
- `chebfromroots` -- create a Chebyshev series with specified roots.
- `chebroots` -- find the roots of a Chebyshev series.
- `chebvander` -- Vandermonde-like matrix for Chebyshev polynomials.
- `chebfit` -- least-squares fit returning a Chebyshev series.
- `chebpts1` -- Chebyshev points of the first kind.
- `chebpts2` -- Chebyshev points of the second kind.
- `chebtrim` -- trim leading coefficients from a Chebyshev series.
- `chebline` -- Chebyshev series representing given straight line.
- `cheb2poly` -- convert a Chebyshev series to a polynomial.
- `poly2cheb` -- convert a polynomial to a Chebyshev series.

Classes
-------
- `Chebyshev` -- A Chebyshev series class.

See also
--------
`numpy.polynomial`

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities [1]_:

.. math ::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math :: x = \\frac{z + z^{-1}}{2}.

These identities allow a Chebyshev series to be expressed as a finite,
symmetric Laurent series.  In this module, this sort of Laurent series
is referred to as a "z-series."

References
----------
.. [1] A. T. Benjamin, et al., "Combinatorial Trigonometry with Chebyshev
  Polynomials," *Journal of Statistical Planning and Inference 14*, 2008
  (preprint: http://www.math.hmc.edu/~benjamin/papers/CombTrig.pdf, pg. 4)

"""
from __future__ import division

__all__ = ['chebzero', 'chebone', 'chebx', 'chebdomain', 'chebline',
        'chebadd', 'chebsub', 'chebmulx', 'chebmul', 'chebdiv', 'chebpow',
        'chebval', 'chebder', 'chebint', 'cheb2poly', 'poly2cheb',
        'chebfromroots', 'chebvander', 'chebfit', 'chebtrim', 'chebroots',
        'chebpts1', 'chebpts2', 'Chebyshev']

import numpy as np
import numpy.linalg as la
import polyutils as pu
import warnings
from polytemplate import polytemplate

chebtrim = pu.trimcoef

#
# A collection of functions for manipulating z-series. These are private
# functions and do minimal error checking.
#

def _cseries_to_zseries(cs) :
    """Covert Chebyshev series to z-series.

    Covert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    cs : 1-d ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-d ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = cs.size
    zs = np.zeros(2*n-1, dtype=cs.dtype)
    zs[n-1:] = cs/2
    return zs + zs[::-1]

def _zseries_to_cseries(zs) :
    """Covert z-series to a Chebyshev series.

    Covert a z series to the equivalent Chebyshev series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    zs : 1-d ndarray
        Odd length symmetric z-series, ordered from  low to high.

    Returns
    -------
    cs : 1-d ndarray
        Chebyshev coefficients, ordered from  low to high.

    """
    n = (zs.size + 1)//2
    cs = zs[n-1:].copy()
    cs[1:n] *= 2
    return cs

def _zseries_mul(z1, z2) :
    """Multiply two z-series.

    Multiply two z-series to produce a z-series.

    Parameters
    ----------
    z1, z2 : 1-d ndarray
        The arrays must be 1-d but this is not checked.

    Returns
    -------
    product : 1-d ndarray
        The product z-series.

    Notes
    -----
    This is simply convolution. If symmetic/anti-symmetric z-series are
    denoted by S/A then the following rules apply:

    S*S, A*A -> S
    S*A, A*S -> A

    """
    return np.convolve(z1, z2)

def _zseries_div(z1, z2) :
    """Divide the first z-series by the second.

    Divide `z1` by `z2` and return the quotient and remainder as z-series.
    Warning: this implementation only applies when both z1 and z2 have the
    same symmetry, which is sufficient for present purposes.

    Parameters
    ----------
    z1, z2 : 1-d ndarray
        The arrays must be 1-d and have the same symmetry, but this is not
        checked.

    Returns
    -------

    (quotient, remainder) : 1-d ndarrays
        Quotient and remainder as z-series.

    Notes
    -----
    This is not the same as polynomial division on account of the desired form
    of the remainder. If symmetic/anti-symmetric z-series are denoted by S/A
    then the following rules apply:

    S/S -> S,S
    A/A -> S,A

    The restriction to types of the same symmetry could be fixed but seems like
    uneeded generality. There is no natural form for the remainder in the case
    where there is no symmetry.

    """
    z1 = z1.copy()
    z2 = z2.copy()
    len1 = len(z1)
    len2 = len(z2)
    if len2 == 1 :
        z1 /= z2
        return z1, z1[:1]*0
    elif len1 < len2 :
        return z1[:1]*0, z1
    else :
        dlen = len1 - len2
        scl = z2[0]
        z2 /= scl
        quo = np.empty(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        while i < j :
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r*z2
            z1[i:i+len2] -= tmp
            z1[j:j+len2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r*z2
        z1[i:i+len2] -= tmp
        quo /= scl
        rem = z1[i+1:i-1+len2].copy()
        return quo, rem

def _zseries_der(zs) :
    """Differentiate a z-series.

    The derivative is with respect to x, not z. This is achieved using the
    chain rule and the value of dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to differentiate.

    Returns
    -------
    derivative : z-series
        The derivative

    Notes
    -----
    The zseries for x (ns) has been multiplied by two in order to avoid
    using floats that are incompatible with Decimal and likely other
    specialized scalar types. This scaling has been compensated by
    multiplying the value of zs by two also so that the two cancels in the
    division.

    """
    n = len(zs)//2
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    zs *= np.arange(-n, n+1)*2
    d, r = _zseries_div(zs, ns)
    return d

def _zseries_int(zs) :
    """Integrate a z-series.

    The integral is with respect to x, not z. This is achieved by a change
    of variable using dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to integrate

    Returns
    -------
    integral : z-series
        The indefinite integral

    Notes
    -----
    The zseries for x (ns) has been multiplied by two in order to avoid
    using floats that are incompatible with Decimal and likely other
    specialized scalar types. This scaling has been compensated by
    dividing the resulting zs by two.

    """
    n = 1 + len(zs)//2
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    zs = _zseries_mul(zs, ns)
    div = np.arange(-n, n+1)*2
    zs[:n] /= div[:n]
    zs[n+1:] /= div[n+1:]
    zs[n] = 0
    return zs

#
# Chebyshev series functions
#


def poly2cheb(pol) :
    """
    Convert a polynomial to a Chebyshev series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Chebyshev series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-d array containing the polynomial coefficients

    Returns
    -------
    cs : ndarray
        1-d array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    cheb2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> p = P.Polynomial(range(4))
    >>> p
    Polynomial([ 0.,  1.,  2.,  3.], [-1.,  1.])
    >>> c = p.convert(kind=P.Chebyshev)
    >>> c
    Chebyshev([ 1.  ,  3.25,  1.  ,  0.75], [-1.,  1.])
    >>> P.poly2cheb(range(4))
    array([ 1.  ,  3.25,  1.  ,  0.75])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1) :
        res = chebadd(chebmulx(res), pol[i])
    return res


def cheb2poly(cs) :
    """
    Convert a Chebyshev series to a polynomial.

    Convert an array representing the coefficients of a Chebyshev series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    cs : array_like
        1-d array containing the Chebyshev series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-d array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2cheb

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> c = P.Chebyshev(range(4))
    >>> c
    Chebyshev([ 0.,  1.,  2.,  3.], [-1.,  1.])
    >>> p = c.convert(kind=P.Polynomial)
    >>> p
    Polynomial([ -2.,  -8.,   4.,  12.], [-1.,  1.])
    >>> P.cheb2poly(range(4))
    array([ -2.,  -8.,   4.,  12.])

    """
    from polynomial import polyadd, polysub, polymulx

    [cs] = pu.as_series([cs])
    n = len(cs)
    if n < 3:
        return cs
    else:
        c0 = cs[-2]
        c1 = cs[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1) :
            tmp = c0
            c0 = polysub(cs[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1)*2)
        return polyadd(c0, polymulx(c1))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Chebyshev default domain.
chebdomain = np.array([-1,1])

# Chebyshev coefficients representing zero.
chebzero = np.array([0])

# Chebyshev coefficients representing one.
chebone = np.array([1])

# Chebyshev coefficients representing the identity x.
chebx = np.array([0,1])

def chebline(off, scl) :
    """
    Chebyshev series whose graph is a straight line.



    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Chebyshev series for
        ``off + scl*x``.

    See Also
    --------
    polyline

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebline(3,2)
    array([3, 2])
    >>> C.chebval(-3, C.chebline(3,2)) # should be -3
    -3.0

    """
    if scl != 0 :
        return np.array([off,scl])
    else :
        return np.array([off])

def chebfromroots(roots) :
    """
    Generate a Chebyshev series with the given roots.

    Return the array of coefficients for the C-series whose roots (a.k.a.
    "zeros") are given by *roots*.  The returned array of coefficients is
    ordered from lowest order "term" to highest, and zeros of multiplicity
    greater than one must be included in *roots* a number of times equal
    to their multiplicity (e.g., if `2` is a root of multiplicity three,
    then [2,2,2] must be in *roots*).

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-d array of the C-series' coefficients, ordered from low to
        high.  If all roots are real, ``out.dtype`` is a float type;
        otherwise, ``out.dtype`` is a complex type, even if all the
        coefficients in the result are real (see Examples below).

    See Also
    --------
    polyfromroots

    Notes
    -----
    What is returned are the :math:`c_i` such that:

    .. math::

        \\sum_{i=0}^{n} c_i*T_i(x) = \\prod_{i=0}^{n} (x - roots[i])

    where ``n == len(roots)`` and :math:`T_i(x)` is the `i`-th Chebyshev
    (basis) polynomial over the domain `[-1,1]`.  Note that, unlike
    `polyfromroots`, due to the nature of the C-series basis set, the
    above identity *does not* imply :math:`c_n = 1` identically (see
    Examples).

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebfromroots((-1,0,1)) # x^3 - x relative to the standard basis
    array([ 0.  , -0.25,  0.  ,  0.25])
    >>> j = complex(0,1)
    >>> C.chebfromroots((-j,j)) # x^2 + 1 relative to the standard basis
    array([ 1.5+0.j,  0.0+0.j,  0.5+0.j])

    """
    if len(roots) == 0 :
        return np.ones(1)
    else :
        [roots] = pu.as_series([roots], trim=False)
        roots.sort()
        p = [chebline(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            tmp = [chebmul(p[i], p[i+m]) for i in range(m)]
            if r:
                tmp[0] = chebmul(tmp[0], p[-1])
            p = tmp
            n = m
        return p[0]


def chebadd(c1, c2):
    """
    Add one Chebyshev series to another.

    Returns the sum of two Chebyshev series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Chebyshev series of their sum.

    See Also
    --------
    chebsub, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Chebyshev series
    is a Chebyshev series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebadd(c1,c2)
    array([ 4.,  4.,  4.])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if len(c1) > len(c2) :
        c1[:c2.size] += c2
        ret = c1
    else :
        c2[:c1.size] += c1
        ret = c2
    return pu.trimseq(ret)


def chebsub(c1, c2):
    """
    Subtract one Chebyshev series from another.

    Returns the difference of two Chebyshev series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their difference.

    See Also
    --------
    chebadd, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Chebyshev
    series is a Chebyshev series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebsub(c1,c2)
    array([-2.,  0.,  2.])
    >>> C.chebsub(c2,c1) # -C.chebsub(c1,c2)
    array([ 2.,  0., -2.])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if len(c1) > len(c2) :
        c1[:c2.size] -= c2
        ret = c1
    else :
        c2 = -c2
        c2[:c1.size] += c1
        ret = c2
    return pu.trimseq(ret)


def chebmulx(cs):
    """Multiply a Chebyshev series by x.

    Multiply the polynomial `cs` by x, where x is the independent
    variable.


    Parameters
    ----------
    cs : array_like
        1-d array of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    Notes
    -----
    .. versionadded:: 1.5.0

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    # The zero series needs special treatment
    if len(cs) == 1 and cs[0] == 0:
        return cs

    prd = np.empty(len(cs) + 1, dtype=cs.dtype)
    prd[0] = cs[0]*0
    prd[1] = cs[0]
    if len(cs) > 1:
        tmp = cs[1:]/2
        prd[2:] = tmp
        prd[0:-2] += tmp
    return prd


def chebmul(c1, c2):
    """
    Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their product.

    See Also
    --------
    chebadd, chebsub, chebdiv, chebpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Chebyshev polynomial basis set.  Thus, to express
    the product as a C-series, it is typically necessary to "re-project"
    the product onto said basis set, which typically produces
    "un-intuitive" (but correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebmul(c1,c2) # multiplication requires "reprojection"
    array([  6.5,  12. ,  12. ,   4. ,   1.5])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2)
    ret = _zseries_to_cseries(prd)
    return pu.trimseq(ret)


def chebdiv(c1, c2):
    """
    Divide one Chebyshev series by another.

    Returns the quotient-with-remainder of two Chebyshev series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Chebyshev series coefficients representing the quotient and
        remainder.

    See Also
    --------
    chebadd, chebsub, chebmul, chebpow

    Notes
    -----
    In general, the (polynomial) division of one C-series by another
    results in quotient and remainder terms that are not in the Chebyshev
    polynomial basis set.  Thus, to express these results as C-series, it
    is typically necessary to "re-project" the results onto said basis
    set, which typically produces "un-intuitive" (but correct) results;
    see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebdiv(c1,c2) # quotient "intuitive," remainder not
    (array([ 3.]), array([-8., -4.]))
    >>> c2 = (0,1,2,3)
    >>> C.chebdiv(c2,c1) # neither "intuitive"
    (array([ 0.,  2.]), array([-2., -4.]))

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0 :
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2 :
        return c1[:1]*0, c1
    elif lc2 == 1 :
        return c1/c2[-1], c1[:1]*0
    else :
        z1 = _cseries_to_zseries(c1)
        z2 = _cseries_to_zseries(c2)
        quo, rem = _zseries_div(z1, z2)
        quo = pu.trimseq(_zseries_to_cseries(quo))
        rem = pu.trimseq(_zseries_to_cseries(rem))
        return quo, rem

def chebpow(cs, pow, maxpower=16) :
    """Raise a Chebyshev series to a power.

    Returns the Chebyshev series `cs` raised to the power `pow`. The
    arguement `cs` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    cs : array_like
        1d array of chebyshev series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to umanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Chebyshev series of power.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    power = int(pow)
    if power != pow or power < 0 :
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower :
        raise ValueError("Power is too large")
    elif power == 0 :
        return np.array([1], dtype=cs.dtype)
    elif power == 1 :
        return cs
    else :
        # This can be made more efficient by using powers of two
        # in the usual way.
        zs = _cseries_to_zseries(cs)
        prd = zs
        for i in range(2, power + 1) :
            prd = np.convolve(prd, zs)
        return _zseries_to_cseries(prd)

def chebder(cs, m=1, scl=1) :
    """
    Differentiate a Chebyshev series.

    Returns the series `cs` differentiated `m` times.  At each iteration the
    result is multiplied by `scl` (the scaling factor is for use in a linear
    change of variable).  The argument `cs` is the sequence of coefficients
    from lowest order "term" to highest, e.g., [1,2,3] represents the series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    cs: array_like
        1-d array of Chebyshev series coefficients ordered from low to high.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)

    Returns
    -------
    der : ndarray
        Chebyshev series of the derivative.

    See Also
    --------
    chebint

    Notes
    -----
    In general, the result of differentiating a C-series needs to be
    "re-projected" onto the C-series basis set. Thus, typically, the
    result of this function is "un-intuitive," albeit correct; see Examples
    section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> cs = (1,2,3,4)
    >>> C.chebder(cs)
    array([ 14.,  12.,  24.])
    >>> C.chebder(cs,3)
    array([ 96.])
    >>> C.chebder(cs,scl=-1)
    array([-14., -12., -24.])
    >>> C.chebder(cs,2,-1)
    array([ 12.,  96.])

    """
    cnt = int(m)

    if cnt != m:
        raise ValueError, "The order of derivation must be integer"
    if cnt < 0 :
        raise ValueError, "The order of derivation must be non-negative"

    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if cnt == 0:
        return cs
    elif cnt >= len(cs):
        return cs[:1]*0
    else :
        zs = _cseries_to_zseries(cs)
        for i in range(cnt):
            zs = _zseries_der(zs)*scl
        return _zseries_to_cseries(zs)


def chebint(cs, m=1, k=[], lbnd=0, scl=1):
    """
    Integrate a Chebyshev series.

    Returns, as a C-series, the input C-series `cs`, integrated `m` times
    from `lbnd` to `x`.  At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `cs` is a sequence of
    coefficients, from lowest order C-series "term" to highest, e.g.,
    [1,2,3] represents the series :math:`T_0(x) + 2T_1(x) + 3T_2(x)`.

    Parameters
    ----------
    cs : array_like
        1-d array of C-series coefficients, ordered from low to high.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)

    Returns
    -------
    S : ndarray
        C-series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
        ``np.isscalar(scl) == False``.

    See Also
    --------
    chebder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to :math:`1/a`
    - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "re-projected" onto the C-series basis set.  Thus, typically,
    the result of this function is "un-intuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> cs = (1,2,3)
    >>> C.chebint(cs)
    array([ 0.5, -0.5,  0.5,  0.5])
    >>> C.chebint(cs,3)
    array([ 0.03125   , -0.1875    ,  0.04166667, -0.05208333,  0.01041667,
            0.00625   ])
    >>> C.chebint(cs, k=3)
    array([ 3.5, -0.5,  0.5,  0.5])
    >>> C.chebint(cs,lbnd=-2)
    array([ 8.5, -0.5,  0.5,  0.5])
    >>> C.chebint(cs,scl=-2)
    array([-1.,  1., -1., -1.])

    """
    cnt = int(m)
    if not np.iterable(k):
        k = [k]

    if cnt != m:
        raise ValueError, "The order of integration must be integer"
    if cnt < 0 :
        raise ValueError, "The order of integration must be non-negative"
    if len(k) > cnt :
        raise ValueError, "Too many integration constants"

    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if cnt == 0:
        return cs

    k = list(k) + [0]*(cnt - len(k))
    for i in range(cnt) :
        n = len(cs)
        cs *= scl
        if n == 1 and cs[0] == 0:
            cs[0] += k[i]
        else:
            zs = _cseries_to_zseries(cs)
            zs = _zseries_int(zs)
            cs = _zseries_to_cseries(zs)
            cs[0] += k[i] - chebval(lbnd, cs)
    return cs

def chebval(x, cs):
    """Evaluate a Chebyshev series.

    If `cs` is of length `n`, this function returns :

    ``p(x) = cs[0]*T_0(x) + cs[1]*T_1(x) + ... + cs[n-1]*T_{n-1}(x)``

    If x is a sequence or array then p(x) will have the same shape as x.
    If r is a ring_like object that supports multiplication and addition
    by the values in `cs`, then an object of the same type is returned.

    Parameters
    ----------
    x : array_like, ring_like
        Array of numbers or objects that support multiplication and
        addition with themselves and with the elements of `cs`.
    cs : array_like
        1-d array of Chebyshev coefficients ordered from low to high.

    Returns
    -------
    values : ndarray, ring_like
        If the return is an ndarray then it has the same shape as `x`.

    See Also
    --------
    chebfit

    Examples
    --------

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if isinstance(x, tuple) or isinstance(x, list) :
        x = np.asarray(x)

    if len(cs) == 1 :
        c0 = cs[0]
        c1 = 0
    elif len(cs) == 2 :
        c0 = cs[0]
        c1 = cs[1]
    else :
        x2 = 2*x
        c0 = cs[-2]
        c1 = cs[-1]
        for i in range(3, len(cs) + 1) :
            tmp = c0
            c0 = cs[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x

def chebvander(x, deg) :
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points `x`.
    This isn't a true Vandermonde matrix because `x` can be an arbitrary
    ndarray and the Chebyshev polynomials aren't powers. If ``V`` is the
    returned matrix and `x` is a 2d array, then the elements of ``V`` are
    ``V[i,j,k] = T_k(x[i,j])``, where ``T_k`` is the Chebyshev polynomial
    of degree ``k``.

    Parameters
    ----------
    x : array_like
        Array of points. The values are converted to double or complex
        doubles. If x is scalar it is converted to a 1D array.
    deg : integer
        Degree of the resulting matrix.

    Returns
    -------
    vander : Vandermonde matrix.
        The shape of the returned matrix is ``x.shape + (deg+1,)``. The last
        index is the degree.

    """
    ideg = int(deg)
    if ideg != deg:
        raise ValueError("deg must be integer")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=0, ndmin=1) + 0.0
    v = np.empty((ideg + 1,) + x.shape, dtype=x.dtype)
    # Use forward recursion to generate the entries.
    v[0] = x*0 + 1
    if ideg > 0 :
        x2 = 2*x
        v[1] = x
        for i in range(2, ideg + 1) :
            v[i] = v[i-1]*x2 - v[i-2]
    return np.rollaxis(v, 0, v.ndim)


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Chebyshev series to data.

    Return the coefficients of a Legendre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * T_1(x) + ... + c_n * T_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting series
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the contribution of each point
        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
        weights are chosen so that the errors of the products ``w[i]*y[i]``
        all have the same variance.  The default value is None.

        .. versionadded:: 1.5.0

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Chebyshev coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column k  of `y` are in column
        `k`.

    [residuals, rank, singular_values, rcond] : present when `full` = True
        Residuals of the least-squares fit, the effective rank of the
        scaled Vandermonde matrix and its singular values, and the
        specified value of `rcond`. For more details, see `linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', RankWarning)

    See Also
    --------
    polyfit, legfit, lagfit, hermfit, hermefit
    chebval : Evaluates a Chebyshev series.
    chebvander : Vandermonde matrix of Chebyshev series.
    chebweight : Chebyshev weight function.
    linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Chebyshev series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coeficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Chebyshev series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------

    """
    order = int(deg) + 1
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

    # check arguments.
    if deg < 0 :
        raise ValueError, "expected deg >= 0"
    if x.ndim != 1:
        raise TypeError, "expected 1D vector for x"
    if x.size == 0:
        raise TypeError, "expected non-empty vector for x"
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError, "expected 1D or 2D array for y"
    if len(x) != len(y):
        raise TypeError, "expected x and y to have same length"

    # set up the least squares matrices
    lhs = chebvander(x, deg)
    rhs = y
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError, "expected 1D vector for w"
        if len(x) != len(w):
            raise TypeError, "expected x and w to have same length"
        # apply weights
        if rhs.ndim == 2:
            lhs *= w[:, np.newaxis]
            rhs *= w[:, np.newaxis]
        else:
            lhs *= w[:, np.newaxis]
            rhs *= w

    # set rcond
    if rcond is None :
        rcond = len(x)*np.finfo(x.dtype).eps

    # scale the design matrix and solve the least squares equation
    scl = np.sqrt((lhs*lhs).sum(0))
    c, resids, rank, s = la.lstsq(lhs/scl, rhs, rcond)
    c = (c.T/scl).T

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, pu.RankWarning)

    if full :
        return c, [resids, rank, s, rcond]
    else :
        return c


def chebcompanion(cs):
    """Return the scaled companion matrix of cs.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `cs` represents a single Chebyshev polynomial. This
    provides better eigenvalue estimates than the unscaled case and in the
    single polynomial case the eigenvalues are guaranteed to be real if
    np.eigvalsh is used to obtain them.

    Parameters
    ----------
    cs : array_like
        1-d array of Legendre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if len(cs) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(cs) == 2:
        return np.array(-cs[0]/cs[1])

    n = len(cs) - 1
    mat = np.zeros((n, n), dtype=cs.dtype)
    scl = np.array([1.] + [np.sqrt(.5)]*(n-1))
    top = mat.reshape(-1)[1::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[0] = np.sqrt(.5)
    top[1:] = 1/2
    bot[...] = top
    mat[:,-1] -= (cs[:-1]/cs[-1])*(scl/scl[-1])*.5
    return mat


def chebroots(cs):
    """
    Compute the roots of a Chebyshev series.

    Return the roots (a.k.a "zeros") of the C-series represented by `cs`,
    which is the sequence of the C-series' coefficients from lowest order
    "term" to highest, e.g., [1,2,3] represents the C-series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    cs : array_like
        1-d array of C-series coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        Array of the roots.  If all the roots are real, then so is the
        dtype of ``out``; otherwise, ``out``'s dtype is complex.

    See Also
    --------
    polyroots

    Notes
    -----
    Algorithm(s) used:

    Remember: because the C-series basis set is different from the
    "standard" basis set, the results of this function *may* not be what
    one is expecting.

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as cheb
    >>> cheb.chebroots((-1, 1,-1, 1)) # T3 - T2 + T1 - T0 has real roots
    array([ -5.00000000e-01,   2.60860684e-17,   1.00000000e+00])

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if len(cs) < 2:
        return np.array([], dtype=cs.dtype)
    if len(cs) == 2:
        return np.array([-cs[0]/cs[1]])

    m = chebcompanion(cs)
    r = la.eigvals(m)
    r.sort()
    return r


def chebpts1(npts):
    """Chebyshev points of the first kind.

    Chebyshev points of the first kind are the set ``{cos(x_k)}``,
    where ``x_k = pi*(k + .5)/npts`` for k in ``range(npts}``.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the second kind.

    Notes
    -----
    .. versionadded:: 1.5.0

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = np.linspace(-np.pi, 0, _npts, endpoint=False) + np.pi/(2*_npts)
    return np.cos(x)


def chebpts2(npts):
    """Chebyshev points of the second kind.

    Chebyshev points of the second kind are the set ``{cos(x_k)}``,
    where ``x_k = pi*/(npts - 1)`` for k in ``range(npts}``.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the second kind.

    Notes
    -----
    .. versionadded:: 1.5.0

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = np.linspace(-np.pi, 0, _npts)
    return np.cos(x)


#
# Chebyshev series class
#

exec polytemplate.substitute(name='Chebyshev', nick='cheb', domain='[-1,1]')
