""" This really needs some work...

    Calculate machine limits for Float32 and Float64.
    Actually, max and min are hard coded - and wrong!
    They are close, however.

"""

import Numeric
from type_check import cast
toFloat32=cast['f']
toFloat64=cast['d']

__all__ = ['epsilon','tiny','float_epsilon','float_tiny','float_min',
           'float_max','float_precision','float_resolution',
           'double_epsilon','double_tiny','double_min','double_max',
           'double_precision','double_resolution']


def epsilon(typecode):
    if typecode == Numeric.Float32: cast = toFloat32
    elif typecode == Numeric.Float64: cast = toFloat64
    one = cast(1.0)
    x = cast(1.0)
    while one+x > one:
        x = x * cast(.5)
    x = x * cast(2.0)
    return x

def tiny(typecode):
    if typecode == Numeric.Float32: cast = toFloat32
    if typecode == Numeric.Float64: cast = toFloat64
    zero = cast(0.0)
    d1 = cast(1.0)
    d2 = cast(1.0)
    while d1 > zero:
        d2 = d1
        d1 = d1 * cast(.5)
    return d2

float_epsilon = epsilon(Numeric.Float32)
float_tiny = tiny(Numeric.Float32)
#not correct
float_min = -3.402823e38
float_max = 3.402823e38
float_precision = 6
float_resolution = 10.0**(-float_precision)

# hard coded - taken from Norbert's Fortran code.
#      INTEGER, PARAMETER :: kind_DBLE = KIND(0D0)           ! 8 (HP-UX)
#      INTEGER, PARAMETER :: prec_DBLE = PRECISION(0D0)      ! 15
#      INTEGER, PARAMETER :: range_DBLE = RANGE(0D0)         ! 307
#      REAL(kind_DBLE), PARAMETER :: eps_DBLE = EPSILON(0D0) ! 2.22e-16
#      REAL(kind_DBLE), PARAMETER :: tiny_DBLE = TINY(0D0)   ! 2.23e-308
#      REAL(kind_DBLE), PARAMETER :: huge_DBLE = HUGE(0D0)   ! 1.80e+308
double_epsilon = epsilon(Numeric.Float64)
double_tiny = tiny(Numeric.Float64)

# not quite right...
double_min = -1.797683134862318e308
double_max = 1.797683134862318e308
double_precision = 15
double_resolution = 10.0**(-double_precision)

def test(level=10):
    from scipy_test.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_test.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)

if __name__ == '__main__':
    print 'float epsilon:',float_epsilon
    print 'float tiny:',float_tiny
    print 'double epsilon:',double_epsilon
    print 'double tiny:',double_tiny
