"""Miscellaneous functions for testing masked arrays and subclasses

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: testutils.py 3529 2007-11-13 08:01:14Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = "1.0"
__revision__ = "$Revision: 3529 $"
__date__ = "$Date: 2007-11-13 10:01:14 +0200 (Tue, 13 Nov 2007) $"


import numpy as N
from numpy.core import ndarray
from numpy.core.numerictypes import float_
import numpy.core.umath as umath
from numpy.testing import NumpyTest, NumpyTestCase
from numpy.testing.utils import build_err_msg, rand

import core
from core import mask_or, getmask, getmaskarray, masked_array, nomask, masked
from core import filled, equal, less

#------------------------------------------------------------------------------
def approx (a, b, fill_value=True, rtol=1.e-5, atol=1.e-8):
    """Returns true if all components of a and b are equal subject to given tolerances.

If fill_value is True, masked values considered equal. Otherwise, masked values
are considered unequal.
The relative error rtol should be positive and << 1.0
The absolute error atol comes into play for those elements of b that are very
small or zero; it says how small a must be also.
    """
    m = mask_or(getmask(a), getmask(b))
    d1 = filled(a)
    d2 = filled(b)
    if d1.dtype.char == "O" or d2.dtype.char == "O":
        return N.equal(d1,d2).ravel()
    x = filled(masked_array(d1, copy=False, mask=m), fill_value).astype(float_)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(float_)
    d = N.less_equal(umath.absolute(x-y), atol + rtol * umath.absolute(y))
    return d.ravel()
#................................................
def _assert_equal_on_sequences(actual, desired, err_msg=''):
    "Asserts the equality of two non-array sequences."
    assert_equal(len(actual),len(desired),err_msg)
    for k in range(len(desired)):
        assert_equal(actual[k], desired[k], 'item=%r\n%s' % (k,err_msg))
    return

def assert_equal_records(a,b):
    """Asserts that two records are equal. Pretty crude for now."""
    assert_equal(a.dtype, b.dtype)
    for f in a.dtype.names:
        (af, bf) = (getattr(a,f), getattr(b,f))
        if not (af is masked) and not (bf is masked):
            assert_equal(getattr(a,f), getattr(b,f))
    return

def assert_equal(actual,desired,err_msg=''):
    """Asserts that two items are equal.
    """
    # Case #1: dictionary .....
    if isinstance(desired, dict):
        assert isinstance(actual, dict), repr(type(actual))
        assert_equal(len(actual),len(desired),err_msg)
        for k,i in desired.items():
            assert k in actual, repr(k)
            assert_equal(actual[k], desired[k], 'key=%r\n%s' % (k,err_msg))
        return
    # Case #2: lists .....
    if isinstance(desired, (list,tuple)) and isinstance(actual, (list,tuple)):
        return _assert_equal_on_sequences(actual, desired, err_msg='')
    if not (isinstance(actual, ndarray) or isinstance(desired, ndarray)):
        msg = build_err_msg([actual, desired], err_msg,)
        assert desired == actual, msg
        return
    # Case #4. arrays or equivalent
    if ((actual is masked) and not (desired is masked)) or \
        ((desired is masked) and not (actual is masked)):
        msg = build_err_msg([actual, desired], err_msg, header='', names=('x', 'y'))
        raise ValueError(msg)
    actual = N.array(actual, copy=False, subok=True)
    desired = N.array(desired, copy=False, subok=True)
    if actual.dtype.char in "OS" and desired.dtype.char in "OS":
        return _assert_equal_on_sequences(actual.tolist(),
                                          desired.tolist(),
                                          err_msg='')
    return assert_array_equal(actual, desired, err_msg)
#.............................
def fail_if_equal(actual,desired,err_msg='',):
    """Raises an assertion error if two items are equal.
    """
    if isinstance(desired, dict):
        assert isinstance(actual, dict), repr(type(actual))
        fail_if_equal(len(actual),len(desired),err_msg)
        for k,i in desired.items():
            assert k in actual, repr(k)
            fail_if_equal(actual[k], desired[k], 'key=%r\n%s' % (k,err_msg))
        return
    if isinstance(desired, (list,tuple)) and isinstance(actual, (list,tuple)):
        fail_if_equal(len(actual),len(desired),err_msg)
        for k in range(len(desired)):
            fail_if_equal(actual[k], desired[k], 'item=%r\n%s' % (k,err_msg))
        return
    if isinstance(actual, N.ndarray) or isinstance(desired, N.ndarray):
        return fail_if_array_equal(actual, desired, err_msg)
    msg = build_err_msg([actual, desired], err_msg)
    assert desired != actual, msg
assert_not_equal = fail_if_equal
#............................
def assert_almost_equal(actual,desired,decimal=7,err_msg=''):
    """Asserts that two items are almost equal.
    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal)
    """
    if isinstance(actual, N.ndarray) or isinstance(desired, N.ndarray):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    msg = build_err_msg([actual, desired], err_msg)
    assert round(abs(desired - actual),decimal) == 0, msg
#............................
def assert_array_compare(comparison, x, y, err_msg='', header='',
                         fill_value=True):
    """Asserts that a comparison relation between two masked arrays is satisfied
    elementwise."""
    xf = filled(x)
    yf = filled(y)
    m = mask_or(getmask(x), getmask(y))

    x = masked_array(xf, copy=False, subok=False, mask=m).filled(fill_value)
    y = masked_array(yf, copy=False, subok=False, mask=m).filled(fill_value)

    if ((x is masked) and not (y is masked)) or \
        ((y is masked) and not (x is masked)):
        msg = build_err_msg([x, y], err_msg, header=header, names=('x', 'y'))
        raise ValueError(msg)

    if (x.dtype.char != "O") and (x.dtype.char != "S"):
        x = x.astype(float_)
        if isinstance(x, N.ndarray) and x.size > 1:
            x[N.isnan(x)] = 0
        elif N.isnan(x):
            x = 0
    if (y.dtype.char != "O") and (y.dtype.char != "S"):
        y = y.astype(float_)
        if isinstance(y, N.ndarray) and y.size > 1:
            y[N.isnan(y)] = 0
        elif N.isnan(y):
            y = 0
    try:
        cond = (x.shape==() or y.shape==()) or x.shape == y.shape
        if not cond:
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(shapes %s, %s mismatch)' % (x.shape,
                                                                  y.shape),
                                header=header,
                                names=('x', 'y'))
            assert cond, msg
        val = comparison(x,y)
        if m is not nomask and fill_value:
            val = masked_array(val, mask=m, copy=False)
        if isinstance(val, bool):
            cond = val
            reduced = [0]
        else:
            reduced = val.ravel()
            cond = reduced.all()
            reduced = reduced.tolist()
        if not cond:
            match = 100-100.0*reduced.count(1)/len(reduced)
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(mismatch %s%%)' % (match,),
                                header=header,
                                names=('x', 'y'))
            assert cond, msg
    except ValueError:
        msg = build_err_msg([x, y], err_msg, header=header, names=('x', 'y'))
        raise ValueError(msg)
#............................
def assert_array_equal(x, y, err_msg=''):
    """Checks the elementwise equality of two masked arrays."""
    assert_array_compare(equal, x, y, err_msg=err_msg,
                         header='Arrays are not equal')
##............................
def fail_if_array_equal(x, y, err_msg=''):
    "Raises an assertion error if two masked arrays are not equal (elementwise)."
    def compare(x,y):

        return (not N.alltrue(approx(x, y)))
    assert_array_compare(compare, x, y, err_msg=err_msg,
                         header='Arrays are not equal')
#............................
def assert_array_almost_equal(x, y, decimal=6, err_msg=''):
    """Checks the elementwise equality of two masked arrays, up to a given
    number of decimals."""
    def compare(x, y):
        "Returns the result of the loose comparison between x and y)."
        return approx(x,y, rtol=10.**-decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg,
                         header='Arrays are not almost equal')
#............................
def assert_array_less(x, y, err_msg=''):
    "Checks that x is smaller than y elementwise."
    assert_array_compare(less, x, y, err_msg=err_msg,
                         header='Arrays are not less-ordered')
#............................
assert_close = assert_almost_equal
#............................
def assert_mask_equal(m1, m2):
    """Asserts the equality of two masks."""
    if m1 is nomask:
        assert(m2 is nomask)
    if m2 is nomask:
        assert(m1 is nomask)
    assert_array_equal(m1, m2)

if __name__ == '__main__':
    pass
