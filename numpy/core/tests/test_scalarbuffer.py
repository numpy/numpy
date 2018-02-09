"""
Test scalar buffer interface adheres to PEP 3118
"""
import numpy as np
from numpy.testing import run_module_suite, assert_

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

    # TODO: 'p', 'l', or 'q'?
    # (np.intp, 'p'),
    # (np.uintp, 'P'),
]

scalars_set_size = [
    (np.bool8, 1),
    (np.int8, 1),
    (np.int16, 2),
    (np.int32, 4),
    (np.int64, 8),
    (np.uint8, 1),
    (np.uint16, 2),
    (np.uint32, 4),
    (np.uint64, 8),
    (np.float16, 2),
    (np.float32, 4),
    (np.float64, 8),
    (np.complex64, 8),
    (np.complex128, 16),
]

# platform dependant dtypes
if hasattr(np, 'float96'):
    scalars.append(np.float96)
    scalars_set_size.append((np.float96, 12))
if hasattr(np, 'float128'):
    scalars.append(np.float128)
    scalars_set_size.append((np.float128, 16))
if hasattr(np, 'complex192'):
    scalars.append(np.complex192)
    scalars_set_size.append((np.complex192, 24))
if hasattr(np, 'complex256'):
    scalars.append(np.complex256)
    scalars_set_size.append((np.complex256, 32))


class TestScalarPEP3118(object):
    def test_scalar_match_array(self):
        for scalar in scalars:
            x = scalar()
            a = np.array([], dtype=scalar)
            assert_(x.data.format == a.data.format)
    
    def test_scalar_dim(self):
        for scalar in scalars:
            x = scalar()
            assert_(x.data.ndim == 0)
            assert_(x.data.shape == ())
            assert_(x.data.strides == ())
            assert_(x.data.suboffsets == ())
        
    def test_scalar_known_size(self):
        for scalar, size in scalars_set_size:
            x = scalar()
            assert_(x.data.nbytes == size)
    
    def test_scalar_known_code(self):
        for scalar, code in scalars_set_code:
            x = scalar()
            assert_(x.data.format == code)

if __name__ == "__main__":
    run_module_suite()
