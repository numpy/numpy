"""
Objects for dealing with Laguerre series.

This module provides a number of objects (mostly functions) useful for
dealing with Laguerre series, including a `Laguerre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Constants
---------
- `lagdomain` -- Laguerre series default domain, [-1,1].
- `lagzero` -- Laguerre series that evaluates identically to 0.
- `lagone` -- Laguerre series that evaluates identically to 1.
- `lagx` -- Laguerre series for the identity map, ``f(x) = x``.

Arithmetic
----------
- `lagmulx` -- multiply a Laguerre series in ``P_i(x)`` by ``x``.
- `lagadd` -- add two Laguerre series.
- `lagsub` -- subtract one Laguerre series from another.
- `lagmul` -- multiply two Laguerre series.
- `lagdiv` -- divide one Laguerre series by another.
- `lagval` -- evaluate a Laguerre series at given points.

Calculus
--------
- `lagder` -- differentiate a Laguerre series.
- `lagint` -- integrate a Laguerre series.

Misc Functions
--------------
- `lagfromroots` -- create a Laguerre series with specified roots.
- `lagroots` -- find the roots of a Laguerre series.
- `lagvander` -- Vandermonde-like matrix for Laguerre polynomials.
- `lagfit` -- least-squares fit returning a Laguerre series.
- `lagtrim` -- trim leading coefficients from a Laguerre series.
- `lagline` -- Laguerre series of given straight line.
- `lag2poly` -- convert a Laguerre series to a polynomial.
- `poly2lag` -- convert a polynomial to a Laguerre series.

Classes
-------
- `Laguerre` -- A Laguerre series class.

See also
--------
`numpy.polynomial`

"""
from __future__ import division

__all__ = ['lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline',
        'lagadd', 'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagval',
        'lagder', 'lagint', 'lag2poly', 'poly2lag', 'lagfromroots',
        'lagvander', 'lagfit', 'lagtrim', 'lagroots', 'Laguerre']

import numpy as np
import numpy.linalg as la
import polyutils as pu
import warnings
from polytemplate import polytemplate

lagtrim = pu.trimcoef

def poly2lag(pol) :
    """
    poly2lag(pol)

    Convert a polynomial to a Laguerre series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Laguerre series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-d array containing the polynomial coefficients

    Returns
    -------
    cs : ndarray
        1-d array containing the coefficients of the equivalent Laguerre
        series.

    See Also
    --------
    lag2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import poly2lag
    >>> poly2lag(np.arange(4))
    array([ 23., -63.,  58., -18.])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1) :
        res = lagadd(lagmulx(res), pol[i])
    return res


def lag2poly(cs) :
    """
    Convert a Laguerre series to a polynomial.

    Convert an array representing the coefficients of a Laguerre series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    cs : array_like
        1-d array containing the Laguerre series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-d array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2lag

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lag2poly
    >>> lag2poly([ 23., -63.,  58., -18.])
    array([ 0.,  1.,  2.,  3.])

    """
    from polynomial import polyadd, polysub, polymulx

    [cs] = pu.as_series([cs])
    n = len(cs)
    if n == 1:
        return cs
    else:
        c0 = cs[-2]
        c1 = cs[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(cs[i - 2], (c1*(i - 1))/i)
            c1 = polyadd(tmp, polysub((2*i - 1)*c1, polymulx(c1))/i)
        return polyadd(c0, polysub(c1, polymulx(c1)))

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Laguerre
lagdomain = np.array([0,1])

# Laguerre coefficients representing zero.
lagzero = np.array([0])

# Laguerre coefficients representing one.
lagone = np.array([1])

# Laguerre coefficients representing the identity x.
lagx = np.array([1, -1])


def lagline(off, scl) :
    """
    Laguerre series whose graph is a straight line.



    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Laguerre series for
        ``off + scl*x``.

    See Also
    --------
    polyline, chebline

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagline, lagval
    >>> lagval(0,lagline(3, 2))
    3.0
    >>> lagval(1,lagline(3, 2))
    5.0

    """
    if scl != 0 :
        return np.array([off + scl, -scl])
    else :
        return np.array([off])


def lagfromroots(roots) :
    """
    Generate a Laguerre series with the given roots.

    Return the array of coefficients for the P-series whose roots (a.k.a.
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
        1-d array of the Laguerre series coefficients, ordered from low to
        high.  If all roots are real, ``out.dtype`` is a float type;
        otherwise, ``out.dtype`` is a complex type, even if all the
        coefficients in the result are real (see Examples below).

    See Also
    --------
    polyfromroots, chebfromroots

    Notes
    -----
    What is returned are the :math:`c_i` such that:

    .. math::

        \\sum_{i=0}^{n} c_i*P_i(x) = \\prod_{i=0}^{n} (x - roots[i])

    where ``n == len(roots)`` and :math:`P_i(x)` is the `i`-th Laguerre
    (basis) polynomial over the domain `[-1,1]`.  Note that, unlike
    `polyfromroots`, due to the nature of the Laguerre basis set, the
    above identity *does not* imply :math:`c_n = 1` identically (see
    Examples).

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagfromroots, lagval
    >>> coef = lagfromroots((-1, 0, 1))
    >>> lagval((-1, 0, 1), coef)
    array([ 0.,  0.,  0.])
    >>> coef = lagfromroots((-1j, 1j))
    >>> lagval((-1j, 1j), coef)
    array([ 0.+0.j,  0.+0.j])

    """
    if len(roots) == 0 :
        return np.ones(1)
    else :
        [roots] = pu.as_series([roots], trim=False)
        roots.sort()
        p = [lagline(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            tmp = [lagmul(p[i], p[i+m]) for i in range(m)]
            if r:
                tmp[0] = lagmul(tmp[0], p[-1])
            p = tmp
            n = m
        return p[0]


def lagadd(c1, c2):
    """
    Add one Laguerre series to another.

    Returns the sum of two Laguerre series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Laguerre series of their sum.

    See Also
    --------
    lagsub, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Laguerre series
    is a Laguerre series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagadd
    >>> lagadd([1, 2, 3], [1, 2, 3, 4])
    array([ 2.,  4.,  6.,  4.])


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


def lagsub(c1, c2):
    """
    Subtract one Laguerre series from another.

    Returns the difference of two Laguerre series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their difference.

    See Also
    --------
    lagadd, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Laguerre
    series is a Laguerre series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagsub
    >>> lagsub([1, 2, 3, 4], [1, 2, 3])
    array([ 0.,  0.,  0.,  4.])

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


def lagmulx(cs):
    """Multiply a Laguerre series by x.

    Multiply the Laguerre series `cs` by x, where x is the independent
    variable.


    Parameters
    ----------
    cs : array_like
        1-d array of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    Notes
    -----
    The multiplication uses the recursion relationship for Laguerre
    polynomials in the form

    .. math::

    xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmulx
    >>> lagmulx([1, 2, 3])
    array([ -1.,  -1.,  11.,  -9.])

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    # The zero series needs special treatment
    if len(cs) == 1 and cs[0] == 0:
        return cs

    prd = np.empty(len(cs) + 1, dtype=cs.dtype)
    prd[0] = cs[0]
    prd[1] = -cs[0]
    for i in range(1, len(cs)):
        prd[i + 1] = -cs[i]*(i + 1)
        prd[i] += cs[i]*(2*i + 1)
        prd[i - 1] -= cs[i]*i
    return prd


def lagmul(c1, c2):
    """
    Multiply one Laguerre series by another.

    Returns the product of two Laguerre series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their product.

    See Also
    --------
    lagadd, lagsub, lagdiv, lagpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Laguerre polynomial basis set.  Thus, to express
    the product as a Laguerre series, it is necessary to "re-project" the
    product onto said basis set, which may produce "un-intuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmul
    >>> lagmul([1, 2, 3], [0, 1, 2])
    array([  8., -13.,  38., -51.,  36.])

    """
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    if len(c1) > len(c2):
        cs = c2
        xs = c1
    else:
        cs = c1
        xs = c2

    if len(cs) == 1:
        c0 = cs[0]*xs
        c1 = 0
    elif len(cs) == 2:
        c0 = cs[0]*xs
        c1 = cs[1]*xs
    else :
        nd = len(cs)
        c0 = cs[-2]*xs
        c1 = cs[-1]*xs
        for i in range(3, len(cs) + 1) :
            tmp = c0
            nd =  nd - 1
            c0 = lagsub(cs[-i]*xs, (c1*(nd - 1))/nd)
            c1 = lagadd(tmp, lagsub((2*nd - 1)*c1, lagmulx(c1))/nd)
    return lagadd(c0, lagsub(c1, lagmulx(c1)))


def lagdiv(c1, c2):
    """
    Divide one Laguerre series by another.

    Returns the quotient-with-remainder of two Laguerre series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Laguerre series coefficients representing the quotient and
        remainder.

    See Also
    --------
    lagadd, lagsub, lagmul, lagpow

    Notes
    -----
    In general, the (polynomial) division of one Laguerre series by another
    results in quotient and remainder terms that are not in the Laguerre
    polynomial basis set.  Thus, to express these results as a Laguerre
    series, it is necessary to "re-project" the results onto the Laguerre
    basis set, which may produce "un-intuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagdiv
    >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])
    (array([ 1.,  2.,  3.]), array([ 0.]))
    >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])
    (array([ 1.,  2.,  3.]), array([ 1.,  1.]))

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
        quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
        rem = c1
        for i in range(lc1 - lc2, - 1, -1):
            p = lagmul([0]*i + [1], c2)
            q = rem[-1]/p[-1]
            rem = rem[:-1] - q*p[:-1]
            quo[i] = q
        return quo, pu.trimseq(rem)


def lagpow(cs, pow, maxpower=16) :
    """Raise a Laguerre series to a power.

    Returns the Laguerre series `cs` raised to the power `pow`. The
    arguement `cs` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    cs : array_like
        1d array of Laguerre series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to umanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Laguerre series of power.

    See Also
    --------
    lagadd, lagsub, lagmul, lagdiv

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagpow
    >>> lagpow([1, 2, 3], 2)
    array([ 14., -16.,  56., -72.,  54.])

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
        prd = cs
        for i in range(2, power + 1) :
            prd = lagmul(prd, cs)
        return prd


def lagder(cs, m=1, scl=1) :
    """
    Differentiate a Laguerre series.

    Returns the series `cs` differentiated `m` times.  At each iteration the
    result is multiplied by `scl` (the scaling factor is for use in a linear
    change of variable).  The argument `cs` is the sequence of coefficients
    from lowest order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    cs: array_like
        1-d array of Laguerre series coefficients ordered from low to high.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)

    Returns
    -------
    der : ndarray
        Laguerre series of the derivative.

    See Also
    --------
    lagint

    Notes
    -----
    In general, the result of differentiating a Laguerre series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "un-intuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagder
    >>> lagder([ 1.,  1.,  1., -3.])
    array([ 1.,  2.,  3.])
    >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)
    array([ 1.,  2.,  3.])

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
        for i in range(cnt):
            n = len(cs) - 1
            cs *= scl
            der = np.empty(n, dtype=cs.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = -cs[j]
                cs[j - 1] += cs[j]
            cs = der
        return cs


def lagint(cs, m=1, k=[], lbnd=0, scl=1):
    """
    Integrate a Laguerre series.

    Returns a Laguerre series that is the Laguerre series `cs`, integrated
    `m` times from `lbnd` to `x`.  At each iteration the resulting series
    is **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `cs` is a sequence of
    coefficients, from lowest order Laguerre series "term" to highest,
    e.g., [1,2,3] represents the series :math:`P_0(x) + 2P_1(x) + 3P_2(x)`.

    Parameters
    ----------
    cs : array_like
        1-d array of Laguerre series coefficients, ordered from low to high.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)

    Returns
    -------
    S : ndarray
        Laguerre series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
        ``np.isscalar(scl) == False``.

    See Also
    --------
    lagder

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
    >>> from numpy.polynomial.laguerre import lagint
    >>> lagint([1,2,3])
    array([ 1.,  1.,  1., -3.])
    >>> lagint([1,2,3], m=2)
    array([ 1.,  0.,  0., -4.,  3.])
    >>> lagint([1,2,3], k=1)
    array([ 2.,  1.,  1., -3.])
    >>> lagint([1,2,3], lbnd=-1)
    array([ 11.5,   1. ,   1. ,  -3. ])
    >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)
    array([ 11.16666667,  -5.        ,  -3.        ,   2.        ])

    """
    cnt = int(m)
    if np.isscalar(k) :
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
            tmp = np.empty(n + 1, dtype=cs.dtype)
            tmp[0] = cs[0]
            tmp[1] = -cs[0]
            for j in range(1, n):
                tmp[j] += cs[j]
                tmp[j + 1] = -cs[j]
            tmp[0] += k[i] - lagval(lbnd, tmp)
            cs = tmp
    return cs


def lagval(x, cs):
    """Evaluate a Laguerre series.

    If `cs` is of length `n`, this function returns :

    ``p(x) = cs[0]*P_0(x) + cs[1]*P_1(x) + ... + cs[n-1]*P_{n-1}(x)``

    If x is a sequence or array then p(x) will have the same shape as x.
    If r is a ring_like object that supports multiplication and addition
    by the values in `cs`, then an object of the same type is returned.

    Parameters
    ----------
    x : array_like, ring_like
        Array of numbers or objects that support multiplication and
        addition with themselves and with the elements of `cs`.
    cs : array_like
        1-d array of Laguerre coefficients ordered from low to high.

    Returns
    -------
    values : ndarray, ring_like
        If the return is an ndarray then it has the same shape as `x`.

    See Also
    --------
    lagfit

    Examples
    --------

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval
    >>> coef = [1,2,3]
    >>> lagval(1, coef)
    -0.5
    >>> lagval([[1,2],[3,4]], coef)
    array([[-0.5, -4. ],
           [-4.5, -2. ]])

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
        nd = len(cs)
        c0 = cs[-2]
        c1 = cs[-1]
        for i in range(3, len(cs) + 1) :
            tmp = c0
            nd =  nd - 1
            c0 = cs[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*((2*nd - 1) - x))/nd
    return c0 + c1*(1 - x)


def lagvander(x, deg) :
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points `x`.
    This isn't a true Vandermonde matrix because `x` can be an arbitrary
    ndarray and the Laguerre polynomials aren't powers. If ``V`` is the
    returned matrix and `x` is a 2d array, then the elements of ``V`` are
    ``V[i,j,k] = P_k(x[i,j])``, where ``P_k`` is the Laguerre polynomial
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

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagvander
    >>> x = np.array([0, 1, 2])
    >>> lagvander(x, 3)
    array([[ 1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  0.        , -0.5       , -0.66666667],
           [ 1.        , -1.        , -1.        , -0.33333333]])

    """
    ideg = int(deg)
    if ideg != deg:
        raise ValueError("deg must be integer")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=0, ndmin=1) + 0.0
    v = np.empty((ideg + 1,) + x.shape, dtype=x.dtype)
    v[0] = x*0 + 1
    if ideg > 0 :
        v[1] = 1 - x
        for i in range(2, ideg + 1) :
            v[i] = (v[i-1]*(2*i - 1 - x) - v[i-2]*(i - 1))/i
    return np.rollaxis(v, 0, v.ndim)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Laguerre series to data.

    Return the coefficients of a Laguerre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),

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
    w : array_like, shape (`M`,), optional
        Weights. If not None, the contribution of each point
        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
        weights are chosen so that the errors of the products ``w[i]*y[i]``
        all have the same variance.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Laguerre coefficients ordered from low to high. If `y` was 2-D,
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
    chebfit, legfit, polyfit, hermfit, hermefit
    lagval : Evaluates a Laguerre series.
    lagvander : pseudo Vandermonde matrix of Laguerre series.
    lagweight : Laguerre weight function.
    linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Laguerre series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up as the (typically) overdetermined matrix equation

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

    Fits using Laguerre series are probably most useful when the data can
    be approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the Laguerre
    weight. In that case the wieght ``sqrt(w(x[i])`` should be used
    together with data values ``y[i]/sqrt(w(x[i])``. The weight function is
    available as `lagweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagfit, lagval
    >>> x = np.linspace(0, 10)
    >>> err = np.random.randn(len(x))/10
    >>> y = lagval(x, [1, 2, 3]) + err
    >>> lagfit(x, y, 2)
    array([ 0.96971004,  2.00193749,  3.00288744])

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
    lhs = lagvander(x, deg)
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


def lagcompanion(cs):
    """Return the companion matrix of cs.

    The unscaled companion matrix of the Laguerre polynomials is already
    symmetric when `cs` represents a single Laguerre polynomial, so no
    further scaling is needed.

    Parameters
    ----------
    cs : array_like
        1-d array of Laguerre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Companion matrix of dimensions (deg, deg).

    """
    accprod = np.multiply.accumulate
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if len(cs) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(cs) == 2:
        return np.array(1 + cs[0]/cs[1])

    n = len(cs) - 1
    mat = np.zeros((n, n), dtype=cs.dtype)
    top = mat.reshape(-1)[1::n+1]
    mid = mat.reshape(-1)[0::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[...] = -np.arange(1,n)
    mid[...] = 2.*np.arange(n) + 1.
    bot[...] = top
    mat[:,-1] += (cs[:-1]/cs[-1])*n
    return mat


def lagroots(cs):
    """
    Compute the roots of a Laguerre series.

    Return the roots (a.k.a "zeros") of the Laguerre series represented by
    `cs`, which is the sequence of coefficients from lowest order "term"
    to highest, e.g., [1,2,3] is the series ``L_0 + 2*L_1 + 3*L_2``.

    Parameters
    ----------
    cs : array_like
        1-d array of Laguerre series coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        Array of the roots.  If all the roots are real, then so is the
        dtype of ``out``; otherwise, ``out``'s dtype is complex.

    See Also
    --------
    polyroots
    chebroots

    Notes
    -----
    Algorithm(s) used:

    Remember: because the Laguerre series basis set is different from the
    "standard" basis set, the results of this function *may* not be what
    one is expecting.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagroots, lagfromroots
    >>> coef = lagfromroots([0, 1, 2])
    >>> coef
    array([  2.,  -8.,  12.,  -6.])
    >>> lagroots(coef)
    array([ -4.44089210e-16,   1.00000000e+00,   2.00000000e+00])

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if len(cs) <= 1 :
        return np.array([], dtype=cs.dtype)
    if len(cs) == 2 :
        return np.array([1 + cs[0]/cs[1]])

    m = lagcompanion(cs)
    r = la.eigvals(m)
    r.sort()
    return r


#
# Laguerre series class
#

exec polytemplate.substitute(name='Laguerre', nick='lag', domain='[-1,1]')
