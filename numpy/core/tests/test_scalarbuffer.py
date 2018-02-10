"""
Test scalar buffer interface adheres to PEP 3118
"""
import sys
import numpy as np
from numpy.testing import run_module_suite, assert_, dec

# types
scalars = [
    np.bool_, np.bool8, np.byte, np.short, np.intc, np.int_,
    np.longlong, np.intp, np.int8, np.int16, np.int32, np.int64,
    np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong,
    np.uintp, np.uint8, np.uint16, np.uint32, np.uint64,
    np.half, np.single, np.double, np.float_, np.longfloat,
    np.float16, np.float32, np.float64, np.csingle, np.complex_,
    np.clongfloat, np.complex64, np.complex128,
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

# platform dependant dtypes
for dtype in ('float96', 'float128', 'complex192', 'complex256'):
    if hasattr(np, dtype):
        scalars.append(getattr(np, dtype))


class TestScalarPEP3118(object):
    @dec.skipif(sys.version_info.major < 3, "scalars do not implement buffer interface in Python 2")
    def test_scalar_match_array(self):
        for scalar in scalars:
            x = scalar()
            a = np.array([], dtype=np.dtype(scalar))
            mv_x = memoryview(x)
            mv_a = memoryview(a)
            assert_(mv_x.format == mv_a.format)

    @dec.skipif(sys.version_info.major < 3, "scalars do not implement buffer interface in Python 2")
    def test_scalar_dim(self):
        for scalar in scalars:
            x = scalar()
            mv_x = memoryview(x)
            assert_(mv_x.itemsize == np.dtype(scalar).itemsize)
            assert_(mv_x.ndim == 0)
            assert_(mv_x.shape == ())
            assert_(mv_x.strides == ())
            assert_(mv_x.suboffsets == ())

    @dec.skipif(sys.version_info.major < 3, "scalars do not implement buffer interface in Python 2")
    def test_scalar_known_code(self):
        for scalar, code in scalars_set_code:
            x = scalar()
            mv_x = memoryview(x)
            assert_(mv_x.format == code)

    @dec.skipif(sys.version_info.major < 3, "scalars do not implement buffer interface in Python 2")
    def test_void_scalar_structured_data(self):
        dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
        a = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        x = a[0]
        assert_(isinstance(x, np.void))

        mv_x = memoryview(x)
        expected_size = 16 * np.unicode_().__array__().itemsize
        expected_size += 2 * np.float64().__array__().itemsize
        assert_(mv_x.itemsize == expected_size)
        assert_(mv_x.ndim == 0)
        assert_(mv_x.shape == ())
        assert_(mv_x.strides == ())
        assert_(mv_x.suboffsets == ())

        mv_a = memoryview(a)
        assert_(mv_x.itemsize == mv_a.itemsize)
        assert_(mv_x.format == mv_a.format)

if __name__ == "__main__":
    run_module_suite()
