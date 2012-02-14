import numpy as np
from numpy.compat import asbytes
from numpy.testing import *
import sys, warnings

def test_na_construction():
    # construct a new NA object
    v = np.NA()
    assert_(not v is np.NA)
    assert_equal(v.payload, None)
    assert_equal(v.dtype, None)

    # Construct with a payload
    v = np.NA(3)
    assert_equal(v.payload, 3)
    assert_equal(v.dtype, None)

    # Construct with a dtype
    v = np.NA(dtype='f4')
    assert_equal(v.payload, None)
    assert_equal(v.dtype, np.dtype('f4'))

    # Construct with both a payload and a dtype
    v = np.NA(5, dtype='f4,i2')
    assert_equal(v.payload, 5)
    assert_equal(v.dtype, np.dtype('f4,i2'))

    # min and max payload values
    v = np.NA(0)
    assert_equal(v.payload, 0)
    v = np.NA(127)
    assert_equal(v.payload, 127)

    # Out of bounds payload values
    assert_raises(ValueError, np.NA, -1)
    assert_raises(ValueError, np.NA, 128)

def test_na_str():
    # With no payload or dtype
    assert_equal(str(np.NA), 'NA')
    assert_equal(str(np.NA()), 'NA')

    # With a payload
    assert_equal(str(np.NA(10)), 'NA(10)')

    # With just a dtype
    assert_equal(str(np.NA(dtype='c16')), 'NA')

    # With a payload and a dtype
    assert_equal(str(np.NA(10, dtype='f4')), 'NA(10)')

def test_na_repr():
    # With no payload or dtype
    assert_equal(repr(np.NA), 'NA')
    assert_equal(repr(np.NA()), 'NA')

    # With a payload
    assert_equal(repr(np.NA(10)), 'NA(10)')

    # With just a dtype
    assert_equal(repr(np.NA(dtype='?')), "NA(dtype='bool')")
    if sys.byteorder == 'little':
        assert_equal(repr(np.NA(dtype='<c16')), "NA(dtype='complex128')")
        assert_equal(repr(np.NA(dtype='>c16')), "NA(dtype='>c16')")
    else:
        assert_equal(repr(np.NA(dtype='>c16')), "NA(dtype='complex128')")
        assert_equal(repr(np.NA(dtype='<c16')), "NA(dtype='<c16')")

    # With a payload and a dtype
    if sys.byteorder == 'little':
        assert_equal(repr(np.NA(10, dtype='<f4')), "NA(10, dtype='float32')")
        assert_equal(repr(np.NA(10, dtype='>f4')), "NA(10, dtype='>f4')")
    else:
        assert_equal(repr(np.NA(10, dtype='>f4')), "NA(10, dtype='float32')")
        assert_equal(repr(np.NA(10, dtype='<f4')), "NA(10, dtype='<f4')")

def test_na_comparison():
    # NA cannot be converted to a boolean
    assert_raises(ValueError, bool, np.NA)

    # Comparison results should be np.NA(dtype='bool')
    def check_comparison_result(res):
        assert_(np.isna(res))
        assert_(res.dtype == np.dtype('bool'))

    # Comparison with different objects produces an NA with boolean type
    check_comparison_result(np.NA < 3)
    check_comparison_result(np.NA <= 3)
    check_comparison_result(np.NA == 3)
    check_comparison_result(np.NA != 3)
    check_comparison_result(np.NA >= 3)
    check_comparison_result(np.NA > 3)

    # Should work with NA on the other side too
    check_comparison_result(3 < np.NA)
    check_comparison_result(3 <= np.NA)
    check_comparison_result(3 == np.NA)
    check_comparison_result(3 != np.NA)
    check_comparison_result(3 >= np.NA)
    check_comparison_result(3 > np.NA)

    # Comparison with an array should produce an array
    a = np.array([0,1,2]) < np.NA
    assert_equal(np.isna(a), [1,1,1])
    assert_equal(a.dtype, np.dtype('bool'))
    a = np.array([0,1,2]) == np.NA
    assert_equal(np.isna(a), [1,1,1])
    assert_equal(a.dtype, np.dtype('bool'))
    a = np.array([0,1,2]) != np.NA
    assert_equal(np.isna(a), [1,1,1])
    assert_equal(a.dtype, np.dtype('bool'))

    # Comparison with an array should work on the other side too
    a = np.NA > np.array([0,1,2])
    assert_equal(np.isna(a), [1,1,1])
    assert_equal(a.dtype, np.dtype('bool'))
    a = np.NA == np.array([0,1,2])
    assert_equal(np.isna(a), [1,1,1])
    assert_equal(a.dtype, np.dtype('bool'))
    a = np.NA != np.array([0,1,2])
    assert_equal(np.isna(a), [1,1,1])
    assert_equal(a.dtype, np.dtype('bool'))

def test_na_operations():
    # The minimum of the payload is taken
    assert_equal((np.NA + np.NA(3)).payload, None)
    assert_equal((np.NA(12) + np.NA()).payload, None)
    assert_equal((np.NA(2) - np.NA(6)).payload, 2)
    assert_equal((np.NA(5) - np.NA(1)).payload, 1)

    # The dtypes are promoted like np.promote_types
    assert_equal((np.NA(dtype='f4') * np.NA(dtype='f8')).dtype,
                 np.dtype('f8'))
    assert_equal((np.NA(dtype='c8') * np.NA(dtype='f8')).dtype,
                 np.dtype('c16'))
    assert_equal((np.NA * np.NA(dtype='i8')).dtype,
                 np.dtype('i8'))
    assert_equal((np.NA(dtype='i2') / np.NA).dtype,
                 np.dtype('i2'))

def test_na_other_operations():
    # Make sure we get NAs for all these operations
    assert_equal(type(np.NA + 3), np.NAType)
    assert_equal(type(3 + np.NA), np.NAType)
    assert_equal(type(np.NA - 3.0), np.NAType)
    assert_equal(type(3.0 - np.NA), np.NAType)
    assert_equal(type(np.NA * 2j), np.NAType)
    assert_equal(type(2j * np.NA), np.NAType)
    assert_equal(type(np.NA / 2j), np.NAType)
    assert_equal(type(2j / np.NA), np.NAType)
    assert_equal(type(np.NA // 2j), np.NAType)
    assert_equal(type(np.NA % 6), np.NAType)
    assert_equal(type(6 % np.NA), np.NAType)
    assert_equal(type(np.NA ** 2), np.NAType)
    assert_equal(type(2 ** np.NA), np.NAType)
    assert_equal(type(np.NA & 2), np.NAType)
    assert_equal(type(2 & np.NA), np.NAType)
    assert_equal(type(np.NA | 2), np.NAType)
    assert_equal(type(2 | np.NA), np.NAType)
    assert_equal(type(np.NA << 2), np.NAType)
    assert_equal(type(2 << np.NA), np.NAType)
    assert_equal(type(np.NA >> 2), np.NAType)
    assert_equal(type(2 >> np.NA), np.NAType)
    assert_(abs(np.NA) is np.NA)
    assert_((-np.NA) is np.NA)
    assert_((+np.NA) is np.NA)
    assert_((~np.NA) is np.NA)

    # The NA should get the dtype from the other operand
    assert_equal((np.NA + 3).dtype, np.array(3).dtype)
    assert_equal((np.NA - 3.0).dtype, np.array(3.0).dtype)
    assert_equal((np.NA * 2j).dtype, np.array(2j).dtype)

    # Should have type promotion if the NA already has a dtype
    assert_equal((np.NA(dtype='f4') ** 3.0).dtype, np.dtype('f8'))

    # Bitwise and/or are specialized slightly
    # NOTE: The keywords 'and' and 'or' coerce to boolean, so we cannot
    #       properly support them.
    assert_equal(np.NA & False, False)
    assert_equal(False & np.NA, False)
    assert_equal(np.NA | True, True)
    assert_equal(True | np.NA, True)
    assert_equal(type(np.NA | False), np.NAType)
    assert_equal(type(np.NA & True), np.NAType)
    assert_equal((np.NA | False).dtype, np.array(False).dtype)
    assert_equal((np.NA & True).dtype, np.array(True).dtype)


def test_na_writable_attributes_deletion():
    a = np.NA(2)
    attr =  ['payload', 'dtype']
    for s in attr:
        assert_raises(AttributeError, delattr, a, s)


if __name__ == "__main__":
    run_module_suite()
