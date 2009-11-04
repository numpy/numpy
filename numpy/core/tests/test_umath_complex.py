from numpy.testing import *
import numpy.core.umath as ncu
import numpy as np

def assert_equal_spec(x, y):
    # Handles nan and inf
    if np.isnan(x):
        if np.isnan(y):
            pass
        else:
            raise AssertionError("""
Items are not almost equal:
 ACTUAL: %s
 DESIRED: %s""" % (str(x), str(y)))
    elif np.isinf(x) and np.isinf(y):
        if x * y > 0:
            pass
        else:
            raise AssertionError("""
Items are not equal:
 ACTUAL: %s
 DESIRED: %s""" % (str(x), str(y)))
    else:
        assert_equal(x, y)

def assert_almost_equal_spec(x, y):
    # Handles nan
    if np.isnan(x):
        if np.isnan(y):
            pass
        else:
            raise AssertionError("""
Items are not almost equal:
 ACTUAL: %s
 DESIRED: %s""" % (str(x), str(y)))
    elif np.isinf(x) and np.isinf(y):
        if x * y > 0:
            pass
        else:
            raise AssertionError("""
Items are not almost equal:
 ACTUAL: %s
 DESIRED: %s""" % (str(x), str(y)))
    else:
        assert_almost_equal(x, y)

class TestClog(TestCase):
    def test_simple(self):
        x = np.array([1+0j, 1+2j])
        y_r = np.log(np.abs(x)) + 1j * np.angle(x)
        y = np.log(x)
        for i in range(len(x)):
            assert_almost_equal_spec(y[i], y_r[i])

    def test_special_values(self):
        xl = []
        yl = []

        # From C99 std (Sec 6.3.2)
        # XXX: check exceptions raised

        # clog(-0 + i0) returns -inf + i pi and raises the 'divide-by-zero'
        # floating-point exception.
        x = np.array([np.NZERO], dtype=np.complex)
        y = np.complex(-np.inf, np.pi)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+0 + i0) returns -inf + i0 and raises the 'divide-by-zero'
        # floating-point exception.
        x = np.array([0], dtype=np.complex)
        y = np.complex(-np.inf, 0)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(x + i inf returns +inf + i pi /2, for finite x.
        x = np.array([complex(1, np.inf)], dtype=np.complex)
        y = np.complex(np.inf, 0.5 * np.pi)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(-1, np.inf)], dtype=np.complex)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(np.inf, np.inf)], dtype=np.complex)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(x + iNaN) returns NaN + iNaN and optionally raises the
        # 'invalid' floating- point exception, for finite x.
        x = np.array([complex(1., np.nan)], dtype=np.complex)
        y = np.complex(np.nan, np.nan)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([np.inf + np.nan * 1j], dtype=np.complex)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(- inf + iy) returns +inf + ipi , for finite positive-signed y.
        x = np.array([-np.inf + 1j], dtype=np.complex)
        y = np.complex(np.inf, np.pi)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+ inf + iy) returns +inf + i0, for finite positive-signed y.
        x = np.array([np.inf + 1j], dtype=np.complex)
        y = np.complex(np.inf, 0)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(- inf + i inf) returns +inf + i3pi /4.
        x = np.array([complex(-np.inf, np.inf)], dtype=np.complex)
        y = np.complex(np.inf, 0.75 * np.pi)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+ inf + i inf) returns +inf + ipi /4.
        x = np.array([complex(np.inf, np.inf)], dtype=np.complex)
        y = np.complex(np.inf, 0.25 * np.pi)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+/- inf + iNaN) returns +inf + iNaN.
        x = np.array([complex(np.inf, np.nan)], dtype=np.complex)
        y = np.complex(np.inf, np.nan)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(-np.inf, np.nan)], dtype=np.complex)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + iy) returns NaN + iNaN and optionally raises the
        # 'invalid' floating-point exception, for finite y.
        x = np.array([complex(np.nan, 1)], dtype=np.complex)
        y = np.complex(np.nan, np.nan)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + i inf) returns +inf + iNaN.
        x = np.array([complex(np.nan, np.inf)], dtype=np.complex)
        y = np.complex(np.inf, np.nan)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + iNaN) returns NaN + iNaN.
        x = np.array([complex(np.nan, np.nan)], dtype=np.complex)
        y = np.complex(np.nan, np.nan)
        assert_almost_equal_spec(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(conj(z)) = conj(clog(z)).
        xa = np.array(xl, dtype=np.complex)
        ya = np.array(yl, dtype=np.complex)
        for i in range(len(xa)):
            assert_almost_equal_spec(np.log(np.conj(xa[i])), np.conj(np.log(xa[i])))

class TestCpow(TestCase):
    def test_simple(self):
        x = np.array([1+1j, 0+2j, 1+2j, np.inf, np.nan])
        y_r = x ** 2
        y = np.power(x, 2)
        for i in range(len(x)):
            assert_almost_equal_spec(y[i], y_r[i])

class TestCabs(TestCase):
    def test_simple(self):
        x = np.array([1+1j, 0+2j, 1+2j, np.inf, np.nan])
        y_r = np.array([np.sqrt(2.), 2, np.sqrt(5), np.inf, np.nan])
        y = np.abs(x)
        for i in range(len(x)):
            assert_almost_equal_spec(y[i], y_r[i])

    def test_fabs(self):
        # Test that np.abs(x +- 0j) == np.abs(x) (as mandated by C99 for cabs)
        x = np.array([1+0j], dtype=np.complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(1, np.NZERO)], dtype=np.complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.inf, np.NZERO)], dtype=np.complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.nan, np.NZERO)], dtype=np.complex)
        assert_array_equal(np.abs(x), np.real(x))

    def test_cabs_inf_nan(self):
        xl = []
        yl = []

        x = np.array([complex(np.nan, np.nan)], np.complex)
        y = np.nan
        assert_almost_equal_spec(np.abs(x), y)
        xl.append(x)
        yl.append(y)

        # According to C99 standard, if exactly one of the real/part is inf and
        # the other nan, then cabs should return inf
        x = np.array([complex(np.inf, np.nan)], np.complex)
        y = np.inf
        assert_almost_equal_spec(np.abs(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(-np.inf, np.nan)], np.complex)
        y = np.inf
        assert_almost_equal_spec(np.abs(x), y)
        xl.append(x)
        yl.append(y)

        # cabs(conj(z)) = conj(cabs(z)).
        xa = np.array(xl, dtype=np.complex)
        ya = np.array(yl, dtype=np.complex)
        for i in range(len(xa)):
            assert_almost_equal_spec(np.abs(np.conj(xa[i])), np.conj(np.abs(xa[i])))

class TestCarg(object):
    def test_simple(self):
        check_complex_value(ncu._arg, 1, 0, 0, 0, False)
        check_complex_value(ncu._arg, 0, 1, 0.5*np.pi, 0, False)

        check_complex_value(ncu._arg, 1, 1, 0.25*np.pi, 0, False)
        check_complex_value(ncu._arg, np.PZERO, np.PZERO, np.PZERO, 0)

    @dec.knownfailureif(True,
        "Complex arithmetic with signed zero is buggy on most implementation")
    def test_zero(self):
        # carg(-0 +- 0i) returns +- pi
        yield check_complex_value, ncu._arg, np.NZERO, np.PZERO,  np.pi, 0, False
        yield check_complex_value, ncu._arg, np.NZERO, np.NZERO, -np.pi, 0, False

        # carg(+0 +- 0i) returns +- 0
        yield check_complex_value, ncu._arg, np.PZERO, np.PZERO, np.PZERO, 0
        yield check_complex_value, ncu._arg, np.PZERO, np.NZERO, np.NZERO, 0

        # carg(x +- 0i) returns +- 0 for x > 0
        yield check_complex_value, ncu._arg, 1, np.PZERO, np.PZERO, 0, False
        yield check_complex_value, ncu._arg, 1, np.NZERO, np.NZERO, 0, False

        # carg(x +- 0i) returns +- pi for x < 0
        yield check_complex_value, ncu._arg, -1, np.PZERO,  np.pi, 0, False
        yield check_complex_value, ncu._arg, -1, np.NZERO, -np.pi, 0, False

        # carg(+- 0 + yi) returns pi/2 for y > 0
        yield check_complex_value, ncu._arg, np.PZERO, 1, 0.5 * np.pi, 0, False
        yield check_complex_value, ncu._arg, np.NZERO, 1, 0.5 * np.pi, 0, False

        # carg(+- 0 + yi) returns -pi/2 for y < 0
        yield check_complex_value, ncu._arg, np.PZERO, -1, 0.5 * np.pi, 0, False
        yield check_complex_value, ncu._arg, np.NZERO, -1,-0.5 * np.pi, 0, False

    #def test_branch_cuts(self):
    #    _check_branch_cut(ncu._arg, -1, 1j, -1, 1)

    def test_special_values(self):
        # carg(-np.inf +- yi) returns +-pi for finite y > 0
        yield check_complex_value, ncu._arg, -np.inf,  1,  np.pi, 0, False
        yield check_complex_value, ncu._arg, -np.inf, -1, -np.pi, 0, False

        # carg(np.inf +- yi) returns +-0 for finite y > 0
        yield check_complex_value, ncu._arg, np.inf,  1, np.PZERO, 0, False
        yield check_complex_value, ncu._arg, np.inf, -1, np.NZERO, 0, False

        # carg(x +- np.infi) returns +-pi/2 for finite x
        yield check_complex_value, ncu._arg, 1,  np.inf,  0.5 * np.pi, 0, False
        yield check_complex_value, ncu._arg, 1, -np.inf, -0.5 * np.pi, 0, False

        # carg(-np.inf +- np.infi) returns +-3pi/4
        yield check_complex_value, ncu._arg, -np.inf,  np.inf,  0.75 * np.pi, 0, False
        yield check_complex_value, ncu._arg, -np.inf, -np.inf, -0.75 * np.pi, 0, False

        # carg(np.inf +- np.infi) returns +-pi/4
        yield check_complex_value, ncu._arg, np.inf,  np.inf,  0.25 * np.pi, 0, False
        yield check_complex_value, ncu._arg, np.inf, -np.inf, -0.25 * np.pi, 0, False

        # carg(x + yi) returns np.nan if x or y is nan
        yield check_complex_value, ncu._arg, np.nan,      0, np.nan, 0, False
        yield check_complex_value, ncu._arg,      0, np.nan, np.nan, 0, False

        yield check_complex_value, ncu._arg, np.nan, np.inf, np.nan, 0, False
        yield check_complex_value, ncu._arg, np.inf, np.nan, np.nan, 0, False

def check_complex_value(f, x1, y1, x2, y2, exact=True):
    z1 = np.array([complex(x1, y1)])
    z2 = np.complex(x2, y2)
    if exact:
        assert_equal_spec(f(z1), z2)
    else:
        assert_almost_equal_spec(f(z1), z2)

if __name__ == "__main__":
    run_module_suite()
