import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import *

import warnings

def test_half_consistency():
    """Checks that all 16-bit values survive conversion
       to/from 32-bit and 64-bit float"""
    # Create an array of all possible 16-bit values.
    # Because the underlying routines preserve the NaN bits, every
    # value is preserved when converting to/from other floats.
    a = np.arange(0x10000, dtype=uint16)
    a_f16 = a.copy()
    a_f16.dtype = float16

    # Convert float16 to float32 and back
    a_f32 = np.array(a_f16, dtype=float32)
    b = np.array(a_f32, dtype=float16)
    b.dtype = uint16
    assert_equal(a,b)

    # Convert float16 to float64 and back
    a_f64 = np.array(a_f16, dtype=float64)
    b = np.array(a_f64, dtype=float16)
    b.dtype = uint16
    assert_equal(a,b)

    # Convert float16 to longdouble and back
    a_ld = np.array(a_f16, dtype=np.longdouble)
    b = np.array(a_f64, dtype=float16)
    b.dtype = uint16
    assert_equal(a,b)

    # Check the range for which all integers can be represented
    a = np.arange(-2048,2049)
    a_f16 = np.array(a, dtype=float16)
    b = np.array(a_f16, dtype=np.int)
    assert_equal(a,b)

    # Check comparisons with NaN
    nan = float16(np.nan)
    assert_(not (a == nan).any())
    assert_((a != nan).all())
    assert_(not (a < nan).any())
    assert_(not (a <= nan).any())
    assert_(not (a > nan).any())
    assert_(not (a >= nan).any())

def test_half_values():
    """Confirms a small number of known half values"""
    a = np.array([1.0, -1.0,
                  2.0, -2.0,
                  0.0999755859375, 0.333251953125, # 1/10, 1/3
                  65504, -65504,           # Maximum magnitude
                  2.0**(-14), -2.0**(-14), # Minimum normalized
                  2.0**(-24), -2.0**(-24), # Minimum subnormal
                  0, -1/1e1000,            # Signed zeros
                  np.inf, -np.inf])
    b = np.array([0x3c00, 0xbc00,
                  0x4000, 0xc000,
                  0x2e66, 0x3555,
                  0x7bff, 0xfbff,
                  0x0400, 0x8400,
                  0x0001, 0x8001,
                  0x0000, 0x8000,
                  0x7c00, 0xfc00], dtype=uint16)
    b.dtype = float16
    assert_equal(a, b)

def test_half_rounding():
    """Checks that rounding when converting to half is correct"""
    a = np.array([2.0**-25 + 2.0**-35,  # Rounds to minimum subnormal
                  2.0**-25,       # Underflows to zero (nearest even mode)
                  2.0**-26,       # Underflows to zero
                  1.0+2.0**-11 + 2.0**-16, # rounds to 1.0+2**(-10)
                  1.0+2.0**-11,   # rounds to 1.0 (nearest even mode)
                  1.0+2.0**-12,   # rounds to 1.0
                  65519,          # rounds to 65504
                  65520],         # rounds to inf
                  dtype=float64)
    rounded = [2.0**-24,
               0.0,
               0.0,
               1.0+2.0**(-10),
               1.0,
               1.0,
               65504,
               np.inf]

    # Check float64->float16 rounding
    b = np.array(a, dtype=float16)
    assert_equal(b, rounded)

    # Check float32->float16 rounding
    a = np.array(a, dtype=float32)
    b = np.array(a, dtype=float16)
    assert_equal(b, rounded)

def test_half_correctness():
    """Take every finite float16, and check the casting functions with
       a manual conversion."""

    # Create an array of all finite float16s
    a = np.arange(0x10000, dtype=uint16)
    a = a[np.nonzero((a&0x7c00) != 0x7c00)]
    a_f16 = a.view(dtype=float16)

    # Convert to 32-bit and 64-bit float with the numpy machinery
    a_f32 = np.array(a_f16, dtype=float32)
    a_f64 = np.array(a_f16, dtype=float64)

    # Convert to 64-bit float manually
    a_sgn = (-1.0)**((a&0x8000) >> 15)
    a_exp = np.array((a&0x7c00) >> 10, dtype=np.int32) - 15
    a_man = (a&0x03ff) * 2.0**(-10)
    # Implicit bit of normalized floats
    a_man[a_exp!=-15] += 1
    # Denormalized exponent is -14
    a_exp[a_exp==-15] = -14

    a_manual = a_sgn * a_man * 2.0**a_exp

    a32_fail = np.nonzero(a_f32 != a_manual)[0]
    if len(a32_fail) != 0:
        bad_index = a32_fail[0]
        assert_equal(a_f32, a_manual,
             "First non-equal is half value %x -> %g != %g" %
                        (a[bad_index], a_f32[bad_index], a_manual[bad_index]))

    a64_fail = np.nonzero(a_f64 != a_manual)[0]
    if len(a64_fail) != 0:
        bad_index = a64_fail[0]
        assert_equal(a_f64, a_manual,
             "First non-equal is half value %x -> %g != %g" %
                        (a[bad_index], a_f64[bad_index], a_manual[bad_index]))

def test_half_ordering():
    """Make sure comparisons are working right"""

    # Create an array of all non-NaN float16s
    a = np.arange(0x10000, dtype=uint16)
    a = a[np.nonzero(np.bitwise_or((a&0x7c00) != 0x7c00, (a&0x03ff) == 0x0000))]
    a.dtype = float16

    # 32-bit float copy
    b = np.array(a, dtype=float32)

    # Should sort the same
    a.sort()
    b.sort()
    assert_equal(a, b)

    # Comparisons should work
    assert_((a[:-1] <= a[1:]).all())
    assert_(not (a[:-1] > a[1:]).any())
    assert_((a[1:] >= a[:-1]).all())
    assert_(not (a[1:] < a[:-1]).any())
    # All != except for +/-0
    assert_equal(np.nonzero(a[:-1] < a[1:])[0].size, a.size-2)
    assert_equal(np.nonzero(a[1:] > a[:-1])[0].size, a.size-2)

def test_half_funcs():
    """Test the various ArrFuncs"""

    # fill
    assert_equal(np.arange(10, dtype=float16),
                 np.arange(10, dtype=float32))

    # fillwithscalar
    a = np.zeros((5,), dtype=float16)
    a.fill(1)
    assert_equal(a, np.ones((5,), dtype=float16))
    
    # nonzero and copyswap
    a = np.array([0,0,-1,-1/1e20,0,2.0**-24, 7.629e-6], dtype=float16)
    assert_equal(a.nonzero()[0],
                 [2,5,6])
    a = a.byteswap().newbyteorder()
    assert_equal(a.nonzero()[0],
                 [2,5,6])

    # dot
    a = np.arange(0, 10, 0.5, dtype=float16)
    b = np.ones((20,), dtype=float16)
    assert_equal(np.dot(a,b),
                 95)
    
    # argmax
    a = np.array([0, -np.inf, -2, 0.5, 12.55, 7.3, 2.1, 12.4], dtype=float16)
    assert_equal(a.argmax(),
                 4)
    a = np.array([0, -np.inf, -2, np.inf, 12.55, np.nan, 2.1, 12.4], dtype=float16)
    assert_equal(a.argmax(),
                 5)

    # getitem
    a = np.arange(10, dtype=float16)
    for i in range(10):
        assert_equal(a.item(i),i)
