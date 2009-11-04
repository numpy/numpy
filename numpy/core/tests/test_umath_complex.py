from numpy.testing import *
import numpy.core.umath as ncu
import numpy as np

# TODO: branch cuts (use Pauli code)
# TODO: conj 'symmetry'
# TODO: FPU exceptions

def _are_equal(x, y):
    if x == 0:
        if y == 0:
            if not np.signbit(x) == np.signbit(y):
                return False
        else:
            return False

    # Handles nan and inf
    if np.isnan(x):
        if np.isnan(y):
            return True
        else:
            return False
    elif np.isinf(x) and np.isinf(y):
        if x * y > 0:
            return True
        else:
            return False
    else:
        try:
            assert_equal(x, y)
            return True
        except AssertionError:
            return False

def assert_equal_spec(x, y):
    # Barf if x and y are not both complex or real right away
    if (np.iscomplexobj(x) and not np.iscomplexobj(y)) \
            or (not np.iscomplexobj(x) and np.iscomplexobj(y)):
        raise AssertionError("Items are not equal:\n" \
                             "ACTUAL: %s\n" \
                             "DESIRED: %s\n" % (str(x), str(y)))

    if np.iscomplexobj(x):
        xr = np.real(x)
        yr = np.real(y)

        xi = np.imag(x)
        yi = np.imag(y)

        if not _are_equal(xr, xi) and _are_equal(yr, yi):
            raise AssertionError("Items are not equal:\n" \
                                 "ACTUAL: %s\n" \
                                 "DESIRED: %s\n" % (str(x), str(y)))
    else:
        if not _are_equal(x, y):
            raise AssertionError("Items are not equal:\n" \
                                 "ACTUAL: %s\n" \
                                 "DESIRED: %s\n" % (str(x), str(y)))

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

class TestCexp(object):
    def test_simple(self):
        check = check_complex_value 
        f = np.exp

        yield check, f, 1, 0, np.exp(1), 0, False
        yield check, f, 0, 1, np.cos(1), np.sin(1), False

        ref = np.exp(1) * np.complex(np.cos(1), np.sin(1))
        yield check, f, 1, 1, ref.real, ref.imag, False

    def test_special_values(self):
        # C99: Section G 6.3.1

        check = check_complex_value 
        f = np.exp

        # cexp(+-0 + 0i) is 1 + 0i
        yield check, f, np.PZERO, 0, 1, 0, False
        yield check, f, np.NZERO, 0, 1, 0, False

        # cexp(x + infi) is nan + nani for finite x and raises 'invalid' FPU
        # exception
        yield check, f,  1, np.inf, np.nan, np.nan
        yield check, f, -1, np.inf, np.nan, np.nan
        yield check, f,  0, np.inf, np.nan, np.nan

        # cexp(inf + 0i) is inf + 0i
        yield check, f,  np.inf, 0, np.inf, 0

        # cexp(-inf + yi) is +0 * (cos(y) + i sin(y)) for finite y
        ref = np.complex(np.cos(1.), np.sin(1.))
        yield check, f,  -np.inf, 1, np.PZERO, np.PZERO

        ref = np.complex(np.cos(np.pi * 0.75), np.sin(np.pi * 0.75))
        yield check, f,  -np.inf, 0.75 * np.pi, np.NZERO, np.PZERO

        # cexp(inf + yi) is +inf * (cos(y) + i sin(y)) for finite y
        ref = np.complex(np.cos(1.), np.sin(1.))
        yield check, f,  np.inf, 1, np.inf, np.inf

        ref = np.complex(np.cos(np.pi * 0.75), np.sin(np.pi * 0.75))
        yield check, f,  np.inf, 0.75 * np.pi, -np.inf, np.inf

        # cexp(-inf + inf i) is +-0 +- 0i (signs unspecified)
        def _check_ninf_inf(dummy):
            z = f(np.array(np.complex(-np.inf, np.inf)))
            if z.real != 0 or z.imag != 0:
                raise AssertionError(
                        "cexp(-inf, inf) is (%f, %f), expected (+-0, +-0)" \
                        % (z.real, z.imag))
        yield _check_ninf_inf, None

        # cexp(inf + inf i) is +-inf + NaNi and raised invalid FPU ex.
        def _check_inf_inf(dummy):
            z = f(np.array(np.complex(np.inf, np.inf)))
            if not np.isinf(z.real) or not np.isnan(z.imag):
                raise AssertionError(
                        "cexp(inf, inf) is (%f, %f), expected (+-inf, nan)" \
                        % (z.real, z.imag))
        yield _check_inf_inf, None

        # cexp(-inf + nan i) is +-0 +- 0i
        def _check_ninf_nan(dummy):
            z = f(np.array(np.complex(-np.inf, np.nan)))
            if z.real != 0 or z.imag != 0:
                raise AssertionError(
                        "cexp(-inf, nan) is (%f, %f), expected (+-0, +-0)" \
                        % (z.real, z.imag))
        yield _check_ninf_nan, None

        # cexp(inf + nan i) is +-inf + nan
        def _check_inf_nan(dummy):
            z = f(np.array(np.complex(np.inf, np.nan)))
            if not np.isinf(z.real) or not np.isnan(z.imag):
                raise AssertionError(
                        "cexp(-inf, nan) is (%f, %f), expected (+-inf, nan)" \
                        % (z.real, z.imag))
        yield _check_inf_nan, None

        # cexp(nan + 0i) is nan + 0i
        yield check, f, np.nan, 0, np.nan, 0

        # cexp(nan + yi) is nan + nani for y != 0 (optional: raises invalid FPU
        # ex)
        yield check, f, np.nan, 1, np.nan, np.nan
        yield check, f, np.nan, -1, np.nan, np.nan

        yield check, f, np.nan,  np.inf, np.nan, np.nan
        yield check, f, np.nan, -np.inf, np.nan, np.nan

        # cexp(nan + nani) is nan + nani
        yield check, f, np.nan, np.nan, np.nan, np.nan

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

class TestCsqrt(object):
    def test_simple(self):
        # sqrt(1)
        yield check_complex_value, np.sqrt, 1, 0, 1, 0

        # sqrt(1i)
        yield check_complex_value, np.sqrt, 0, 1, 0.5*np.sqrt(2), 0.5*np.sqrt(2), False

        # sqrt(-1)
        yield check_complex_value, np.sqrt, -1, 0, 0, 1

    def test_simple_conjugate(self):
        ref = np.conj(np.sqrt(np.complex(1, 1)))
        def f(z):
            return np.sqrt(np.conj(z))
        yield check_complex_value, f, 1, 1, ref.real, ref.imag, False

    #def test_branch_cut(self):
    #    _check_branch_cut(f, -1, 0, 1, -1)

    def test_special_values(self):
        check = check_complex_value 
        f = np.sqrt

        # C99: Sec G 6.4.2
        x, y = [], []

        # csqrt(+-0 + 0i) is 0 + 0i
        yield check, f, np.PZERO, 0, 0, 0
        yield check, f, np.NZERO, 0, 0, 0

        # csqrt(x + infi) is inf + infi for any x (including NaN)
        yield check, f,  1, np.inf, np.inf, np.inf
        yield check, f, -1, np.inf, np.inf, np.inf

        yield check, f, np.PZERO, np.inf, np.inf, np.inf
        yield check, f, np.NZERO, np.inf, np.inf, np.inf
        yield check, f,   np.inf, np.inf, np.inf, np.inf
        yield check, f,  -np.inf, np.inf, np.inf, np.inf
        yield check, f,  -np.nan, np.inf, np.inf, np.inf

        # csqrt(x + nani) is nan + nani for any finite x
        yield check, f,  1, np.nan, np.nan, np.nan
        yield check, f, -1, np.nan, np.nan, np.nan
        yield check, f,  0, np.nan, np.nan, np.nan

        # csqrt(-inf + yi) is +0 + infi for any finite y > 0
        yield check, f, -np.inf, 1, np.PZERO, np.inf

        # csqrt(inf + yi) is +inf + 0i for any finite y > 0
        yield check, f, np.inf, 1, np.inf, np.PZERO

        # csqrt(-inf + nani) is nan +- infi (both +i infi are valid)
        def _check_ninf_nan(dummy):
            z = np.sqrt(np.array(np.complex(-np.inf, np.nan)))
            if not np.isnan(z.real) or not np.isinf(z.imag):
                raise AssertionError(
                        "csqrt(-inf, nan) is (%f, %f), expected (nan, +-inf)" \
                        % (z.real, z.imag))

        yield _check_ninf_nan, None

        # csqrt(+inf + nani) is inf + nani
        yield check, f, np.inf, np.nan, np.inf, np.nan

        # csqrt(nan + yi) is nan + nani for any finite y (infinite handled in x
        # + nani)
        yield check, f, np.nan,       0, np.nan, np.nan
        yield check, f, np.nan,       1, np.nan, np.nan
        yield check, f, np.nan,  np.nan, np.nan, np.nan

        # XXX: check for conj(csqrt(z)) == csqrt(conj(z)) (need to fix branch
        # cuts first)

class TestCpow(TestCase):
    def test_simple(self):
        x = np.array([1+1j, 0+2j, 1+2j, np.inf, np.nan])
        y_r = x ** 2
        y = np.power(x, 2)
        for i in range(len(x)):
            assert_almost_equal_spec(y[i], y_r[i])

class TestCabs(object):
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
        x, y = [], []

        # cabs(+-nan + nani) returns nan
        x.append(np.nan)
        y.append(np.nan)
        yield check_real_value, np.abs,  np.nan, np.nan, np.nan

        x.append(np.nan)
        y.append(-np.nan)
        yield check_real_value, np.abs, -np.nan, np.nan, np.nan

        # According to C99 standard, if exactly one of the real/part is inf and
        # the other nan, then cabs should return inf
        x.append(np.inf)
        y.append(np.nan)
        yield check_real_value, np.abs,  np.inf, np.nan, np.inf

        x.append(-np.inf)
        y.append(np.nan)
        yield check_real_value, np.abs, -np.inf, np.nan, np.inf

        # cabs(conj(z)) == conj(cabs(z)) (= cabs(z))
        def f(a):
            return np.abs(np.conj(a))
        def g(a, b):
            return np.abs(np.complex(a, b))

        xa = np.array(x, dtype=np.complex)
        ya = np.array(x, dtype=np.complex)
        for i in range(len(xa)):
            ref = g(x[i], y[i])
            yield check_real_value, f, x[i], y[i], ref

class TestCarg(object):
    def test_simple(self):
        check_real_value(ncu._arg, 1, 0, 0, False)
        check_real_value(ncu._arg, 0, 1, 0.5*np.pi, False)

        check_real_value(ncu._arg, 1, 1, 0.25*np.pi, False)
        check_real_value(ncu._arg, np.PZERO, np.PZERO, np.PZERO)

    @dec.knownfailureif(True,
        "Complex arithmetic with signed zero is buggy on most implementation")
    def test_zero(self):
        # carg(-0 +- 0i) returns +- pi
        yield check_real_value, ncu._arg, np.NZERO, np.PZERO,  np.pi, False
        yield check_real_value, ncu._arg, np.NZERO, np.NZERO, -np.pi, False

        # carg(+0 +- 0i) returns +- 0
        yield check_real_value, ncu._arg, np.PZERO, np.PZERO, np.PZERO
        yield check_real_value, ncu._arg, np.PZERO, np.NZERO, np.NZERO

        # carg(x +- 0i) returns +- 0 for x > 0
        yield check_real_value, ncu._arg, 1, np.PZERO, np.PZERO, False
        yield check_real_value, ncu._arg, 1, np.NZERO, np.NZERO, False

        # carg(x +- 0i) returns +- pi for x < 0
        yield check_real_value, ncu._arg, -1, np.PZERO,  np.pi, False
        yield check_real_value, ncu._arg, -1, np.NZERO, -np.pi, False

        # carg(+- 0 + yi) returns pi/2 for y > 0
        yield check_real_value, ncu._arg, np.PZERO, 1, 0.5 * np.pi, False
        yield check_real_value, ncu._arg, np.NZERO, 1, 0.5 * np.pi, False

        # carg(+- 0 + yi) returns -pi/2 for y < 0
        yield check_real_value, ncu._arg, np.PZERO, -1, 0.5 * np.pi, False
        yield check_real_value, ncu._arg, np.NZERO, -1,-0.5 * np.pi, False

    #def test_branch_cuts(self):
    #    _check_branch_cut(ncu._arg, -1, 1j, -1, 1)

    def test_special_values(self):
        # carg(-np.inf +- yi) returns +-pi for finite y > 0
        yield check_real_value, ncu._arg, -np.inf,  1,  np.pi, False
        yield check_real_value, ncu._arg, -np.inf, -1, -np.pi, False

        # carg(np.inf +- yi) returns +-0 for finite y > 0
        yield check_real_value, ncu._arg, np.inf,  1, np.PZERO, False
        yield check_real_value, ncu._arg, np.inf, -1, np.NZERO, False

        # carg(x +- np.infi) returns +-pi/2 for finite x
        yield check_real_value, ncu._arg, 1,  np.inf,  0.5 * np.pi, False
        yield check_real_value, ncu._arg, 1, -np.inf, -0.5 * np.pi, False

        # carg(-np.inf +- np.infi) returns +-3pi/4
        yield check_real_value, ncu._arg, -np.inf,  np.inf,  0.75 * np.pi, False
        yield check_real_value, ncu._arg, -np.inf, -np.inf, -0.75 * np.pi, False

        # carg(np.inf +- np.infi) returns +-pi/4
        yield check_real_value, ncu._arg, np.inf,  np.inf,  0.25 * np.pi, False
        yield check_real_value, ncu._arg, np.inf, -np.inf, -0.25 * np.pi, False

        # carg(x + yi) returns np.nan if x or y is nan
        yield check_real_value, ncu._arg, np.nan,      0, np.nan, False
        yield check_real_value, ncu._arg,      0, np.nan, np.nan, False

        yield check_real_value, ncu._arg, np.nan, np.inf, np.nan, False
        yield check_real_value, ncu._arg, np.inf, np.nan, np.nan, False

def check_real_value(f, x1, y1, x, exact=True):
    z1 = np.array([complex(x1, y1)])
    if exact:
        assert_equal_spec(f(z1), x)
    else:
        assert_almost_equal_spec(f(z1), x)

def check_complex_value(f, x1, y1, x2, y2, exact=True):
    z1 = np.array([complex(x1, y1)])
    z2 = np.complex(x2, y2)
    if exact:
        assert_equal_spec(f(z1), z2)
    else:
        assert_almost_equal_spec(f(z1), z2)

if __name__ == "__main__":
    run_module_suite()
