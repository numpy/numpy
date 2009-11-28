"""Utililty functions for polynomial modules.

This modules provides errors, warnings, and a polynomial base class along
with some common routines that are used in both the polynomial and
chebyshev modules.

Errors
------
- PolyError -- base class for errors
- PolyDomainError -- mismatched domains

Warnings
--------
- RankWarning -- issued by least squares fits to warn of deficient rank

Base Class
----------
- PolyBase -- Base class for the Polynomial and Chebyshev classes.

Functions
---------
- as_series -- turns list of array_like into 1d arrays of common type
- trimseq -- removes trailing zeros
- trimcoef -- removes trailing coefficients less than given magnitude
- getdomain -- finds appropriate domain for collection of points
- mapdomain -- maps points between domains
- mapparms -- parameters of the linear map between domains

"""
from __future__ import division

__all__ = ['RankWarning', 'PolyError', 'PolyDomainError', 'PolyBase',
           'as_series', 'trimseq', 'trimcoef', 'getdomain', 'mapdomain',
           'mapparms']

import warnings, exceptions
import numpy as np
import sys

#
# Warnings and Exceptions
#

class RankWarning(UserWarning) :
    """Issued by chebfit when the design matrix is rank deficient."""
    pass

class PolyError(Exception) :
    """Base class for errors in this module."""
    pass

class PolyDomainError(PolyError) :
    """Issued by the generic Poly class when two domains don't match.

    This is raised when an binary operation is passed Poly objects with
    different domains.

    """
    pass

#
# Base class for all polynomial types
#

class PolyBase(object) :
    pass

#
# We need the any function for python < 2.5
#
if sys.version_info[:2] < (2,5) :
    def any(iterable) :
        for element in iterable:
            if element :
                return True
        return False

#
# Helper functions to convert inputs to 1d arrays
#
def trimseq(seq) :
    """Remove small Poly series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Poly series coefficients. This routine fails for
        empty sequences.

    Returns
    -------
    series : sequence
        Subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Do not lose the type info if the sequence contains unknown objects.

    """
    if len(seq) == 0 :
        return seq
    else :
        for i in range(len(seq) - 1, -1, -1) :
            if seq[i] != 0 :
                break
        return seq[:i+1]


def as_series(alist, trim=True) :
    """Return arguments as a list of 1d arrays.

    The return type will always be an array of double, complex double. or
    object.

    Parameters
    ----------
    [a1, a2,...] : list of array_like.
        The arrays must have no more than one dimension when converted.
    trim : boolean
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    [a1, a2,...] : list of 1d-arrays
        A copy of the input data as a 1d-arrays.

    Raises
    ------
    ValueError :
        Raised when an input can not be coverted to 1-d array or the
        resulting array is empty.

    """
    arrays = [np.array(a, ndmin=1, copy=0) for a in alist]
    if min([a.size for a in arrays]) == 0 :
        raise ValueError("Coefficient array is empty")
    if any([a.ndim != 1 for a in arrays]) :
        raise ValueError("Coefficient array is not 1-d")
    if trim :
        arrays = [trimseq(a) for a in arrays]

    if any([a.dtype == np.dtype(object) for a in arrays]) :
        ret = []
        for a in arrays :
            if a.dtype != np.dtype(object) :
                tmp = np.empty(len(a), dtype=np.dtype(object))
                tmp[:] = a[:]
                ret.append(tmp)
            else :
                ret.append(a.copy())
    else :
        try :
            dtype = np.common_type(*arrays)
        except :
            raise ValueError("Coefficient arrays have no common type")
        ret = [np.array(a, copy=1, dtype=dtype) for a in arrays]
    return ret


def trimcoef(c, tol=0) :
    """Remove small trailing coefficients from a polynomial series.

    Parameters
    ----------
    c : array_like
        1-d array of coefficients, ordered from  low to high.
    tol : number
        Trailing elements with absolute value less than tol are removed.

    Returns
    -------
    trimmed : ndarray
        1_d array with tailing zeros removed. If the resulting series would
        be empty, a series containing a singel zero is returned.

    Raises
    ------
    ValueError : if tol < 0

    """
    if tol < 0 :
        raise ValueError("tol must be non-negative")

    [c] = as_series([c])
    [ind] = np.where(np.abs(c) > tol)
    if len(ind) == 0 :
        return c[:1]*0
    else :
        return c[:ind[-1] + 1].copy()

def getdomain(x) :
    """Determine suitable domain for given points.

    Find a suitable domain in which to fit a function defined at the points
    `x` with a polynomial or Chebyshev series.
    
    Parameters
    ----------
    x : array_like
        1D array of points whose domain will be determined.

    Returns
    -------
    domain : ndarray
        1D ndarray containing two values. If the inputs are complex, then
        the two points are the corners of the smallest rectangle alinged
        with the axes in the complex plane containing the points `x`. If
        the inputs are real, then the two points are the ends of the
        smallest interval containing the points `x`,

    See Also
    --------
    mapparms, mapdomain

    """
    [x] = as_series([x], trim=False)
    if x.dtype.char in np.typecodes['Complex'] :
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return np.array((complex(rmin, imin), complex(rmax, imax)))
    else :
        return np.array((x.min(), x.max()))

def mapparms(old, new) :
    """Linear map between domains.

    Return the parameters of the linear map ``off + scl*x`` that maps the
    `old` domain to the `new` domain. The map is defined by the requirement
    that the left end of the old domain map to the left end of the new
    domain, and similarly for the right ends.

    Parameters
    ----------
    old, new : array_like
        The two domains should convert as 1D arrays containing two values.

    Returns
    -------
    off, scl : scalars
        The map `=``off + scl*x`` maps the first domain to the second.

    See Also
    --------
    getdomain, mapdomain

    """
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1]*new[0] - old[0]*new[1])/oldlen
    scl = newlen/oldlen
    return off, scl

def mapdomain(x, old, new) :
    """Apply linear map to input points.
    
    The linear map of the form ``off + scl*x`` that takes the `old` domain
    to the `new` domain is applied to the points `x`.

    Parameters
    ----------
    x : array_like
        Points to  be mapped
    old, new : array_like
        The two domains that determin the map. They should both convert as
        1D arrays containing two values.


    Returns
    -------
    new_x : ndarray
        Array of points of the same shape as the input `x` after the linear
        map defined by the two domains is applied.

    See Also
    --------
    getdomain, mapparms

    """
    [x] = as_series([x], trim=False)
    off, scl = mapparms(old, new)
    return off + scl*x
