"""
Test scalar buffer interface adheres to PEP 3118
"""
import sys
import numpy as np
from numpy.testing import run_module_suite, assert_, assert_equal, dec

# types
scalars = [
    np.bool_,
    np.byte, np.short, np.intc, np.int_, np.longlong,
    np.ubyte, np.ushort, np.uintc, np.uint, np.longlong,
    np.half, np.single, np.double, np.longdouble,
    np.csingle, np.cdouble, np.clongdouble,
]

scalars_set_code = [
    (np.bool_, '?'),
    (np.byte, 'b'),
    (np.short, 'h'),
    (np.intc, 'i'),
    (np.int_, 'l'),
    (np.longlong, 'q'),
    (np.ubyte, 'B'),
    (np.ushort, 'H'),
    (np.uintc, 'I'),
    (np.uint, 'L'),
    (np.ulonglong, 'Q'),
    (np.half, 'e'),
    (np.single, 'f'),
    (np.double, 'd'),
    (np.float_, 'd'),
    (np.longfloat, 'g'),
    (np.csingle, 'Zf'),
    (np.complex_, 'Zd'),
    (np.clongfloat, 'Zg'),
]


class TestScalarPEP3118(object):
    skip_if_no_buffer_interface = dec.skipif(sys.version_info.major < 3,
                "scalars do not implement buffer interface in Python 2")

    @skip_if_no_buffer_interface
    def test_scalar_match_array(self):
        for scalar in scalars:
            x = scalar()
            a = np.array([], dtype=np.dtype(scalar))
            mv_x = memoryview(x)
            mv_a = memoryview(a)
            assert_equal(mv_x.format, mv_a.format)

    @skip_if_no_buffer_interface
    def test_scalar_dim(self):
        for scalar in scalars:
            x = scalar()
            mv_x = memoryview(x)
            assert_equal(mv_x.itemsize, np.dtype(scalar).itemsize)
            assert_equal(mv_x.ndim, 0)
            assert_equal(mv_x.shape, ())
            assert_equal(mv_x.strides, ())
            assert_equal(mv_x.suboffsets, ())

    @skip_if_no_buffer_interface
    def test_scalar_known_code(self):
        for scalar, code in scalars_set_code:
            x = scalar()
            mv_x = memoryview(x)
            assert_equal(mv_x.format, code)

    @skip_if_no_buffer_interface
    def test_void_scalar_structured_data(self):
        dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
        x = np.array(('ndarray_scalar', (1.2, 3.0)), dtype=dt)[()]
        assert_(isinstance(x, np.void))
        mv_x = memoryview(x)
        expected_size = 16 * np.dtype((np.unicode_, 1)).itemsize
        expected_size += 2 * np.dtype((np.float64, 1)).itemsize
        assert_equal(mv_x.itemsize, expected_size)
        assert_equal(mv_x.ndim, 0)
        assert_equal(mv_x.shape, ())
        assert_equal(mv_x.strides, ())
        assert_equal(mv_x.suboffsets, ())

        # check scalar format string against ndarray format string
        a = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        assert_(isinstance(a, np.ndarray))
        mv_a = memoryview(a)
        assert_equal(mv_x.itemsize, mv_a.itemsize)
        assert_equal(mv_x.format, mv_a.format)

if __name__ == "__main__":
    run_module_suite()
