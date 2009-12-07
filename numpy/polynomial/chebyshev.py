"""Functions for dealing with Chebyshev series.

This module provide s a number of functions that are useful in dealing with
Chebyshev series as well as a ``Chebyshev`` class that encapsuletes the usual
arithmetic operations. All the Chebyshev series are assumed to be ordered
from low to high, thus ``array([1,2,3])`` will be treated as the series
``T_0 + 2*T_1 + 3*T_2``

Constants
---------
- chebdomain -- Chebyshev series default domain
- chebzero -- Chebyshev series that evaluates to 0.
- chebone -- Chebyshev series that evaluates to 1.
- chebx -- Chebyshev series of the identity map (x).

Arithmetic
----------
- chebadd -- add a Chebyshev series to another.
- chebsub -- subtract a Chebyshev series from another.
- chebmul -- multiply a Chebyshev series by another
- chebdiv -- divide one Chebyshev series by another.
- chebval -- evaluate a Chebyshev series at given points.

Calculus
--------
- chebder -- differentiate a Chebyshev series.
- chebint -- integrate a Chebyshev series.

Misc Functions
--------------
- chebfromroots -- create a Chebyshev series with specified roots.
- chebroots -- find the roots of a Chebyshev series.
- chebvander -- Vandermode like matrix for Chebyshev polynomials.
- chebfit -- least squares fit returning a Chebyshev series.
- chebtrim -- trim leading coefficients from a Chebyshev series.
- chebline -- Chebyshev series of given straight line
- cheb2poly -- convert a Chebyshev series to a polynomial.
- poly2cheb -- convert a polynomial to a Chebyshev series.

Classes
-------
- Chebyshev -- Chebyshev series class.

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities:

.. math ::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math :: x = \\frac{z + z^{-1}}{2}.

These identities allow a Chebyshev series to be expressed as a finite,
symmetric Laurent series. These sorts of Laurent series are referred to as
z-series in this module.

"""
from __future__ import division

__all__ = ['chebzero', 'chebone', 'chebx', 'chebdomain', 'chebline',
        'chebadd', 'chebsub', 'chebmul', 'chebdiv', 'chebval', 'chebder',
        'chebint', 'cheb2poly', 'poly2cheb', 'chebfromroots', 'chebvander',
        'chebfit', 'chebtrim', 'chebroots', 'Chebyshev']

import numpy as np
import numpy.linalg as la
import polyutils as pu
from polytemplate import polytemplate
from polyutils import RankWarning, PolyError, PolyDomainError

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
    """Convert a polynomial to a Chebyshev series.

    Convert a series containing polynomial coefficients ordered by degree
    from low to high to an equivalent Chebyshev series ordered from low to
    high.

    Inputs
    ------
    pol : array_like
        1-d array containing the polynomial coeffients

    Returns
    -------
    cseries : ndarray
        1-d array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    cheb2poly

    """
    [pol] = pu.as_series([pol])
    pol = pol[::-1]
    zs = pol[:1].copy()
    x = np.array([.5, 0, .5], dtype=pol.dtype)
    for i in range(1, len(pol)) :
        zs = _zseries_mul(zs, x)
        zs[i] += pol[i]
    return _zseries_to_cseries(zs)


def cheb2poly(cs) :
    """Convert a Chebyshev series to a polynomial.

    Covert a series containing Chebyshev series coefficients orderd from
    low to high to an equivalent polynomial ordered from low to
    high by degree.

    Inputs
    ------
    cs : array_like
        1-d array containing the Chebyshev series coeffients ordered from
        low to high.

    Returns
    -------
    pol : ndarray
        1-d array containing the coefficients of the equivalent polynomial
        ordered from low to high by degree.

    See Also
    --------
    poly2cheb

    """
    [cs] = pu.as_series([cs])
    pol = np.zeros(len(cs), dtype=cs.dtype)
    quo = _cseries_to_zseries(cs)
    x = np.array([.5, 0, .5], dtype=pol.dtype)
    for i in range(0, len(cs) - 1) :
        quo, rem = _zseries_div(quo, x)
        pol[i] = rem[0]
    pol[-1] = quo[0]
    return pol

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
    """Chebyshev series whose graph is a straight line

    The line has the formula ``off + scl*x``

    Parameters:
    -----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns:
    --------
    series : 1d ndarray
        The Chebyshev series representation of ``off + scl*x``.

    """
    if scl != 0 :
        return np.array([off,scl])
    else :
        return np.array([off])

def chebfromroots(roots) :
    """Generate a Chebyschev series with given roots.

    Generate a Chebyshev series whose roots are given by `roots`. The
    resulting series is the produet `(x - roots[0])*(x - roots[1])*...`

    Inputs
    ------
    roots : array_like
        1-d array containing the roots in sorted order.

    Returns
    -------
    series : ndarray
        1-d array containing the coefficients of the Chebeshev series
        ordered from low to high.

    See Also
    --------
    chebroots

    """
    if len(roots) == 0 :
        return np.ones(1)
    else :
        [roots] = pu.as_series([roots], trim=False)
        prd = np.array([1], dtype=roots.dtype)
        for r in roots :
            fac = np.array([.5, -r, .5], dtype=roots.dtype)
            prd = _zseries_mul(fac, prd)
        return _zseries_to_cseries(prd)


def chebadd(c1, c2):
    """Add one Chebyshev series to another.

    Returns the sum of two Chebyshev series `c1` + `c2`. The arguments are
    sequences of coefficients ordered from low to high, i.e., [1,2,3] is
    the series "T_0 + 2*T_1 + 3*T_2".

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Chebyshev series of the sum.

    See Also
    --------
    chebsub, chebmul, chebdiv, chebpow

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
    """Subtract one Chebyshev series from another.

    Returns the difference of two Chebyshev series `c1` - `c2`. The
    sequences of coefficients are ordered from low to high, i.e., [1,2,3]
    is the series ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Chebyshev series of the difference.

    See Also
    --------
    chebadd, chebmul, chebdiv, chebpow

    Examples
    --------

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


def chebmul(c1, c2):
    """Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`. The arguments
    are sequences of coefficients ordered from low to high, i.e., [1,2,3]
    is the series ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Chebyshev series of the product.

    See Also
    --------
    chebadd, chebsub, chebdiv, chebpow

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2)
    ret = _zseries_to_cseries(prd)
    return pu.trimseq(ret)


def chebdiv(c1, c2):
    """Divide one Chebyshev series by another.

    Returns the quotient of two Chebyshev series `c1` / `c2`. The arguments
    are sequences of coefficients ordered from low to high, i.e., [1,2,3]
    is the series ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarray
        Chebyshev series of the quotient and remainder.

    See Also
    --------
    chebadd, chebsub, chebmul, chebpow

    Examples
    --------

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
    """Differentiate a Chebyshev series.

    Returns the series `cs` differentiated `m` times. At each iteration the
    result is multiplied by `scl`. The scaling factor is for use in a
    linear change of variable. The argument `cs` is a sequence of
    coefficients ordered from low to high. i.e., [1,2,3] is the series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    cs: array_like
        1d array of chebyshev series coefficients ordered from low to high.
    m : int, optional
        Order of differentiation, must be non-negative. (default: 1)
    scl : scalar, optional
        The result of each derivation is multiplied by `scl`. The end
        result is multiplication by `scl`**`m`. This is for use in a linear
        change of variable. (default: 1)

    Returns
    -------
    der : ndarray
        Chebyshev series of the derivative.

    See Also
    --------
    chebint

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if m < 0 :
        raise ValueError, "The order of derivation must be positive"
    if not np.isscalar(scl) :
        raise ValueError, "The scl parameter must be a scalar"

    if m == 0 :
        return cs
    elif m >= len(cs) :
        return cs[:1]*0
    else :
        zs = _cseries_to_zseries(cs)
        for i in range(m) :
            zs = _zseries_der(zs)*scl
        return _zseries_to_cseries(zs)


def chebint(cs, m=1, k=[], lbnd=0, scl=1) :
    """Integrate a Chebyshev series.

    Returns the series integrated from `lbnd` to x `m` times. At each
    iteration the resulting series is multiplied by `scl` and an
    integration constant specified by `k` is added. The scaling factor is
    for use in a linear change of variable. The argument `cs` is a sequence
    of coefficients ordered from low to high. i.e., [1,2,3] is the series
    ``T_0 + 2*T_1 + 3*T_2``.


    Parameters
    ----------
    cs: array_like
        1d array of chebyshev series coefficients ordered from low to high.
    m : int, optional
        Order of integration, must be positeve. (default: 1)
    k : {[], list, scalar}, optional
        Integration constants. The value of the first integral at zero is
        the first value in the list, the value of the second integral at
        zero is the second value in the list, and so on. If ``[]``
        (default), all constants are set zero.  If `m = 1`, a single scalar
        can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (default: 0)
    scl : scalar, optional
        Following each integration the result is multiplied by `scl` before
        the integration constant is added. (default: 1)

    Returns
    -------
    der : ndarray
        Chebyshev series of the integral.

    Raises
    ------
    ValueError

    See Also
    --------
    chebder

    Examples
    --------

    """
    if np.isscalar(k) :
        k = [k]
    if m < 1 :
        raise ValueError, "The order of integration must be positive"
    if len(k) > m :
        raise ValueError, "Too many integration constants"
    if not np.isscalar(lbnd) :
        raise ValueError, "The lbnd parameter must be a scalar"
    if not np.isscalar(scl) :
        raise ValueError, "The scl parameter must be a scalar"

    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    k = list(k) + [0]*(m - len(k))
    for i in range(m) :
        zs = _cseries_to_zseries(cs)*scl
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
        Array of points. The values are converted to double or complex doubles.
    deg : integer
        Degree of the resulting matrix.

    Returns
    -------
    vander : Vandermonde matrix.
        The shape of the returned matrix is ``x.shape + (deg+1,)``. The last
        index is the degree.

    """
    x = np.asarray(x) + 0.0
    order = int(deg) + 1
    v = np.ones(x.shape + (order,), dtype=x.dtype)
    if order > 1 :
        x2 = 2*x
        v[...,1] = x
        for i in range(2, order) :
            v[...,i] = x2*v[...,i-1] - v[...,i-2]
    return v

def chebfit(x, y, deg, rcond=None, full=False):
    """Least squares fit of Chebyshev series to data.

    Fit a Chebyshev series ``p(x) = p[0] * T_{deq}(x) + ... + p[deg] *
    T_{0}(x)`` of degree `deg` to points `(x, y)`. Returns a vector of
    coefficients `p` that minimises the squared error.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.

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
    chebval : Evaluates a Chebyshev series.
    chebvander : Vandermonde matrix of Chebyshev series.
    polyfit : least squares fit using polynomials.
    linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution are the coefficients ``c[i]`` of the Chebyshev series
    ``T(x)`` that minimizes the squared error

    ``E = \sum_j |y_j - T(x_j)|^2``.

    This problem is solved by setting up as the overdetermined matrix
    equation

    ``V(x)*c = y``,

    where ``V`` is the Vandermonde matrix of `x`, the elements of ``c`` are
    the coefficients to be solved for, and the elements of `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of ``V``.

    If some of the singular values of ``V`` are so small that they are
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
    if x.shape[0] != y.shape[0] :
        raise TypeError, "expected x and y to have same length"

    # set rcond
    if rcond is None :
        rcond = len(x)*np.finfo(x.dtype).eps

    # set up the design matrix and solve the least squares equation
    A = chebvander(x, deg)
    scl = np.sqrt((A*A).sum(0))
    c, resids, rank, s = la.lstsq(A/scl, y, rcond)
    c = (c.T/scl).T

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, pu.RankWarning)

    if full :
        return c, [resids, rank, s, rcond]
    else :
        return c


def chebroots(cs):
    """Roots of a Chebyshev series.

    Compute the roots of the Chebyshev series `cs`. The argument `cs` is a
    sequence of coefficients ordered from low to high. i.e., [1,2,3] is the
    series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    cs : array_like
        1D array of Chebyshev coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        An array containing the complex roots of the chebyshev series.

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if len(cs) <= 1 :
        return np.array([], dtype=cs.dtype)
    if len(cs) == 2 :
        return np.array([-cs[0]/cs[1]])
    n = len(cs) - 1
    cmat = np.zeros((n,n), dtype=cs.dtype)
    cmat.flat[1::n+1] = .5
    cmat.flat[n::n+1] = .5
    cmat[1, 0] = 1
    cmat[:,-1] -= cs[:-1]*(.5/cs[-1])
    roots = la.eigvals(cmat)
    roots.sort()
    return roots


#
# Chebyshev series class
#

exec polytemplate.substitute(name='Chebyshev', nick='cheb', domain='[-1,1]')

