import warnings

from numpy.testing import *
import numpy.lib
from numpy.lib import *
from numpy.core import *
from numpy import matrix, asmatrix

import numpy as np

class TestAny(TestCase):
    def test_basic(self):
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        assert_(any(y1))
        assert_(any(y3))
        assert_(not any(y2))

    def test_nd(self):
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        assert_(any(y1))
        assert_array_equal(sometrue(y1, axis=0), [1, 1, 0])
        assert_array_equal(sometrue(y1, axis=1), [0, 1, 1])


class TestAll(TestCase):
    def test_basic(self):
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        assert_(not all(y1))
        assert_(all(y3))
        assert_(not all(y2))
        assert_(all(~array(y2)))

    def test_nd(self):
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        assert_(not all(y1))
        assert_array_equal(alltrue(y1, axis=0), [0, 0, 1])
        assert_array_equal(alltrue(y1, axis=1), [0, 0, 1])


class TestAverage(TestCase):
    def test_basic(self):
        y1 = array([1, 2, 3])
        assert_(average(y1, axis=0) == 2.)
        y2 = array([1., 2., 3.])
        assert_(average(y2, axis=0) == 2.)
        y3 = [0., 0., 0.]
        assert_(average(y3, axis=0) == 0.)

        y4 = ones((4, 4))
        y4[0, 1] = 0
        y4[1, 0] = 2
        assert_almost_equal(y4.mean(0), average(y4, 0))
        assert_almost_equal(y4.mean(1), average(y4, 1))

        y5 = rand(5, 5)
        assert_almost_equal(y5.mean(0), average(y5, 0))
        assert_almost_equal(y5.mean(1), average(y5, 1))

        y6 = matrix(rand(5, 5))
        assert_array_equal(y6.mean(0), average(y6, 0))

    def test_weights(self):
        y = arange(10)
        w = arange(10)
        actual = average(y, weights=w)
        desired = (arange(10) ** 2).sum()*1. / arange(10).sum()
        assert_almost_equal(actual, desired)

        y1 = array([[1, 2, 3], [4, 5, 6]])
        w0 = [1, 2]
        actual = average(y1, weights=w0, axis=0)
        desired = array([3., 4., 5.])
        assert_almost_equal(actual, desired)

        w1 = [0, 0, 1]
        actual = average(y1, weights=w1, axis=1)
        desired = array([3., 6.])
        assert_almost_equal(actual, desired)

        # This should raise an error. Can we test for that ?
        # assert_equal(average(y1, weights=w1), 9./2.)

        # 2D Case
        w2 = [[0, 0, 1], [0, 0, 2]]
        desired = array([3., 6.])
        assert_array_equal(average(y1, weights=w2, axis=1), desired)
        assert_equal(average(y1, weights=w2), 5.)

    def test_returned(self):
        y = array([[1, 2, 3], [4, 5, 6]])

        # No weights
        avg, scl = average(y, returned=True)
        assert_equal(scl, 6.)

        avg, scl = average(y, 0, returned=True)
        assert_array_equal(scl, array([2., 2., 2.]))

        avg, scl = average(y, 1, returned=True)
        assert_array_equal(scl, array([3., 3.]))

        # With weights
        w0 = [1, 2]
        avg, scl = average(y, weights=w0, axis=0, returned=True)
        assert_array_equal(scl, array([3., 3., 3.]))

        w1 = [1, 2, 3]
        avg, scl = average(y, weights=w1, axis=1, returned=True)
        assert_array_equal(scl, array([6., 6.]))

        w2 = [[0, 0, 1], [1, 2, 3]]
        avg, scl = average(y, weights=w2, axis=1, returned=True)
        assert_array_equal(scl, array([1., 6.]))


class TestSelect(TestCase):
    def _select(self, cond, values, default=0):
        output = []
        for m in range(len(cond)):
            output += [V[m] for V, C in zip(values, cond) if C[m]] or [default]
        return output

    def test_basic(self):
        choices = [array([1, 2, 3]),
                   array([4, 5, 6]),
                   array([7, 8, 9])]
        conditions = [array([0, 0, 0]),
                      array([0, 1, 0]),
                      array([0, 0, 1])]
        assert_array_equal(select(conditions, choices, default=15),
                           self._select(conditions, choices, default=15))

        assert_equal(len(choices), 3)
        assert_equal(len(conditions), 3)


class TestInsert(TestCase):
    def test_basic(self):
        a = [1, 2, 3]
        assert_equal(insert(a, 0, 1), [1, 1, 2, 3])
        assert_equal(insert(a, 3, 1), [1, 2, 3, 1])
        assert_equal(insert(a, [1, 1, 1], [1, 2, 3]), [1, 1, 2, 3, 2, 3])


class TestAmax(TestCase):
    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(amax(a), 10.0)
        b = [[3, 6.0, 9.0],
             [4, 10.0, 5.0],
             [8, 3.0, 2.0]]
        assert_equal(amax(b, axis=0), [8.0, 10.0, 9.0])
        assert_equal(amax(b, axis=1), [9.0, 10.0, 8.0])


class TestAmin(TestCase):
    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(amin(a), -5.0)
        b = [[3, 6.0, 9.0],
             [4, 10.0, 5.0],
             [8, 3.0, 2.0]]
        assert_equal(amin(b, axis=0), [3.0, 3.0, 2.0])
        assert_equal(amin(b, axis=1), [3.0, 4.0, 2.0])


class TestPtp(TestCase):
    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(ptp(a, axis=0), 15.0)
        b = [[3, 6.0, 9.0],
             [4, 10.0, 5.0],
             [8, 3.0, 2.0]]
        assert_equal(ptp(b, axis=0), [5.0, 7.0, 7.0])
        assert_equal(ptp(b, axis= -1), [6.0, 6.0, 6.0])


class TestCumsum(TestCase):
    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [int8, uint8, int16, uint16, int32, uint32,
                      float32, float64, complex64, complex128]:
            a = array(ba, ctype)
            a2 = array(ba2, ctype)
            assert_array_equal(cumsum(a, axis=0), array([1, 3, 13, 24, 30, 35, 39], ctype))
            assert_array_equal(cumsum(a2, axis=0), array([[1, 2, 3, 4], [6, 8, 10, 13],
                                                         [16, 11, 14, 18]], ctype))
            assert_array_equal(cumsum(a2, axis=1),
                               array([[1, 3, 6, 10],
                                      [5, 11, 18, 27],
                                      [10, 13, 17, 22]], ctype))


class TestProd(TestCase):
    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [int16, uint16, int32, uint32,
                      float32, float64, complex64, complex128]:
            a = array(ba, ctype)
            a2 = array(ba2, ctype)
            if ctype in ['1', 'b']:
                self.assertRaises(ArithmeticError, prod, a)
                self.assertRaises(ArithmeticError, prod, a2, 1)
                self.assertRaises(ArithmeticError, prod, a)
            else:
                assert_equal(prod(a, axis=0), 26400)
                assert_array_equal(prod(a2, axis=0),
                                   array([50, 36, 84, 180], ctype))
                assert_array_equal(prod(a2, axis= -1), array([24, 1890, 600], ctype))


class TestCumprod(TestCase):
    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [int16, uint16, int32, uint32,
                      float32, float64, complex64, complex128]:
            a = array(ba, ctype)
            a2 = array(ba2, ctype)
            if ctype in ['1', 'b']:
                self.assertRaises(ArithmeticError, cumprod, a)
                self.assertRaises(ArithmeticError, cumprod, a2, 1)
                self.assertRaises(ArithmeticError, cumprod, a)
            else:
                assert_array_equal(cumprod(a, axis= -1),
                                   array([1, 2, 20, 220,
                                          1320, 6600, 26400], ctype))
                assert_array_equal(cumprod(a2, axis=0),
                                   array([[ 1, 2, 3, 4],
                                          [ 5, 12, 21, 36],
                                          [50, 36, 84, 180]], ctype))
                assert_array_equal(cumprod(a2, axis= -1),
                                   array([[ 1, 2, 6, 24],
                                          [ 5, 30, 210, 1890],
                                          [10, 30, 120, 600]], ctype))


class TestDiff(TestCase):
    def test_basic(self):
        x = [1, 4, 6, 7, 12]
        out = array([3, 2, 1, 5])
        out2 = array([-1, -1, 4])
        out3 = array([0, 5])
        assert_array_equal(diff(x), out)
        assert_array_equal(diff(x, n=2), out2)
        assert_array_equal(diff(x, n=3), out3)

    def test_nd(self):
        x = 20 * rand(10, 20, 30)
        out1 = x[:, :, 1:] - x[:, :, :-1]
        out2 = out1[:, :, 1:] - out1[:, :, :-1]
        out3 = x[1:, :, :] - x[:-1, :, :]
        out4 = out3[1:, :, :] - out3[:-1, :, :]
        assert_array_equal(diff(x), out1)
        assert_array_equal(diff(x, n=2), out2)
        assert_array_equal(diff(x, axis=0), out3)
        assert_array_equal(diff(x, n=2, axis=0), out4)


class TestGradient(TestCase):
    def test_basic(self):
        x = array([[1, 1], [3, 4]])
        dx = [array([[2., 3.], [2., 3.]]),
              array([[0., 0.], [1., 1.]])]
        assert_array_equal(gradient(x), dx)

    def test_badargs(self):
        # for 2D array, gradient can take 0,1, or 2 extra args
        x = array([[1, 1], [3, 4]])
        assert_raises(SyntaxError, gradient, x, array([1., 1.]),
                      array([1., 1.]), array([1., 1.]))

    def test_masked(self):
        # Make sure that gradient supports subclasses like masked arrays
        x = np.ma.array([[1, 1], [3, 4]])
        assert_equal(type(gradient(x)[0]), type(x))


class TestAngle(TestCase):
    def test_basic(self):
        x = [1 + 3j, sqrt(2) / 2.0 + 1j * sqrt(2) / 2, 1, 1j, -1, -1j, 1 - 3j, -1 + 3j]
        y = angle(x)
        yo = [arctan(3.0 / 1.0), arctan(1.0), 0, pi / 2, pi, -pi / 2.0,
              - arctan(3.0 / 1.0), pi - arctan(3.0 / 1.0)]
        z = angle(x, deg=1)
        zo = array(yo) * 180 / pi
        assert_array_almost_equal(y, yo, 11)
        assert_array_almost_equal(z, zo, 11)


class TestTrimZeros(TestCase):
    """ only testing for integer splits.
    """
    def test_basic(self):
        a = array([0, 0, 1, 2, 3, 4, 0])
        res = trim_zeros(a)
        assert_array_equal(res, array([1, 2, 3, 4]))

    def test_leading_skip(self):
        a = array([0, 0, 1, 0, 2, 3, 4, 0])
        res = trim_zeros(a)
        assert_array_equal(res, array([1, 0, 2, 3, 4]))

    def test_trailing_skip(self):
        a = array([0, 0, 1, 0, 2, 3, 0, 4, 0])
        res = trim_zeros(a)
        assert_array_equal(res, array([1, 0, 2, 3, 0, 4]))


class TestExtins(TestCase):
    def test_basic(self):
        a = array([1, 3, 2, 1, 2, 3, 3])
        b = extract(a > 1, a)
        assert_array_equal(b, [3, 2, 2, 3, 3])

    def test_place(self):
        a = array([1, 4, 3, 2, 5, 8, 7])
        place(a, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
        assert_array_equal(a, [1, 2, 3, 4, 5, 6, 7])

    def test_both(self):
        a = rand(10)
        mask = a > 0.5
        ac = a.copy()
        c = extract(mask, a)
        place(a, mask, 0)
        place(a, mask, c)
        assert_array_equal(a, ac)


class TestVectorize(TestCase):
    def test_simple(self):
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b
        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_scalar(self):
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b
        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], 5)
        assert_array_equal(r, [5, 8, 1, 4])

    def test_large(self):
        x = linspace(-3, 2, 10000)
        f = vectorize(lambda x: x)
        y = f(x)
        assert_array_equal(y, x)

    def test_ufunc(self):
        import math
        f = vectorize(math.cos)
        args = array([0, 0.5*pi, pi, 1.5*pi, 2*pi])
        r1 = f(args)
        r2 = cos(args)
        assert_array_equal(r1, r2)

    def test_keywords(self):
        import math
        def foo(a, b=1):
            return a + b
        f = vectorize(foo)
        args = array([1,2,3])
        r1 = f(args)
        r2 = array([2,3,4])
        assert_array_equal(r1, r2)
        r1 = f(args, 2)
        r2 = array([3,4,5])
        assert_array_equal(r1, r2)

    def test_keywords_no_func_code(self):
        # This needs to test a function that has keywords but
        # no func_code attribute, since otherwise vectorize will
        # inspect the func_code.
        import random
        try:
            f = vectorize(random.randrange)
        except:
            raise AssertionError()


class TestDigitize(TestCase):
    def test_forward(self):
        x = arange(-6, 5)
        bins = arange(-5, 5)
        assert_array_equal(digitize(x, bins), arange(11))

    def test_reverse(self):
        x = arange(5, -6, -1)
        bins = arange(5, -5, -1)
        assert_array_equal(digitize(x, bins), arange(11))

    def test_random(self):
        x = rand(10)
        bin = linspace(x.min(), x.max(), 10)
        assert_(all(digitize(x, bin) != 0))


class TestUnwrap(TestCase):
    def test_simple(self):
                #check that unwrap removes jumps greather that 2*pi
        assert_array_equal(unwrap([1, 1 + 2 * pi]), [1, 1])
        #check that unwrap maintans continuity
        assert_(all(diff(unwrap(rand(10) * 100)) < pi))


class TestFilterwindows(TestCase):
    def test_hanning(self):
        #check symmetry
        w = hanning(10)
        assert_array_almost_equal(w, flipud(w), 7)
        #check known value
        assert_almost_equal(sum(w, axis=0), 4.500, 4)

    def test_hamming(self):
        #check symmetry
        w = hamming(10)
        assert_array_almost_equal(w, flipud(w), 7)
        #check known value
        assert_almost_equal(sum(w, axis=0), 4.9400, 4)

    def test_bartlett(self):
        #check symmetry
        w = bartlett(10)
        assert_array_almost_equal(w, flipud(w), 7)
        #check known value
        assert_almost_equal(sum(w, axis=0), 4.4444, 4)

    def test_blackman(self):
        #check symmetry
        w = blackman(10)
        assert_array_almost_equal(w, flipud(w), 7)
        #check known value
        assert_almost_equal(sum(w, axis=0), 3.7800, 4)


class TestTrapz(TestCase):
    def test_simple(self):
        r = trapz(exp(-1.0 / 2 * (arange(-10, 10, .1)) ** 2) / sqrt(2 * pi), dx=0.1)
        #check integral of normal equals 1
        assert_almost_equal(sum(r, axis=0), 1, 7)

    def test_ndim(self):
        x = linspace(0, 1, 3)
        y = linspace(0, 2, 8)
        z = linspace(0, 3, 13)

        wx = ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        q = x[:, None, None] + y[None, :, None] + z[None, None, :]

        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # n-d `x`
        r = trapz(q, x=x[:, None, None], axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y[None, :, None], axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z[None, None, :], axis=2)
        assert_almost_equal(r, qz)

        # 1-d `x`
        r = trapz(q, x=x, axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y, axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z, axis=2)
        assert_almost_equal(r, qz)

    def test_masked(self):
        #Testing that masked arrays behave as if the function is 0 where
        #masked
        x = arange(5)
        y = x * x
        mask = x == 2
        ym = np.ma.array(y, mask=mask)
        r = 13.0 # sum(0.5 * (0 + 1) * 1.0 + 0.5 * (9 + 16))
        assert_almost_equal(trapz(ym, x), r)

        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapz(ym, xm), r)

        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapz(y, xm), r)

    def test_matrix(self):
        #Test to make sure matrices give the same answer as ndarrays
        x = linspace(0, 5)
        y = x * x
        r = trapz(y, x)
        mx = matrix(x)
        my = matrix(y)
        mr = trapz(my, mx)
        assert_almost_equal(mr, r)


class TestSinc(TestCase):
    def test_simple(self):
        assert_(sinc(0) == 1)
        w = sinc(linspace(-1, 1, 100))
        #check symmetry
        assert_array_almost_equal(w, flipud(w), 7)

    def test_array_like(self):
        x = [0, 0.5]
        y1 = sinc(array(x))
        y2 = sinc(list(x))
        y3 = sinc(tuple(x))
        assert_array_equal(y1, y2)
        assert_array_equal(y1, y3)

class TestHistogram(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple(self):
        n = 100
        v = rand(n)
        (a, b) = histogram(v)
        #check if the sum of the bins equals the number of samples
        assert_equal(sum(a, axis=0), n)
        #check that the bin counts are evenly spaced when the data is from a
        # linear function
        (a, b) = histogram(linspace(0, 10, 100))
        assert_array_equal(a, 10)

    def test_one_bin(self):
        # Ticket 632
        hist, edges = histogram([1, 2, 3, 4], [1, 2])
        assert_array_equal(hist, [2, ])
        assert_array_equal(edges, [1, 2])
        assert_raises(ValueError, histogram, [1, 2], bins=0)
        h, e = histogram([1,2], bins=1)
        assert_equal(h, array([2]))
        assert_allclose(e, array([1., 2.]))

    def test_normed(self):
        # Check that the integral of the density equals 1.
        n = 100
        v = rand(n)
        a, b = histogram(v, normed=True)
        area = sum(a * diff(b))
        assert_almost_equal(area, 1)

        # Check with non-constant bin widths (buggy but backwards compatible)
        v = np.arange(10)
        bins = [0, 1, 5, 9, 10]
        a, b = histogram(v, bins, normed=True)
        area = sum(a * diff(b))
        assert_almost_equal(area, 1)

    def test_density(self):
        # Check that the integral of the density equals 1.
        n = 100
        v = rand(n)
        a, b = histogram(v, density=True)
        area = sum(a * diff(b))
        assert_almost_equal(area, 1)

        # Check with non-constant bin widths
        v = np.arange(10)
        bins = [0,1,3,6,10]
        a, b = histogram(v, bins, density=True)
        assert_array_equal(a, .1)
        assert_equal(sum(a*diff(b)), 1)

        # Variale bin widths are especially useful to deal with
        # infinities.
        v = np.arange(10)
        bins = [0,1,3,6,np.inf]
        a, b = histogram(v, bins, density=True)
        assert_array_equal(a, [.1,.1,.1,0.])

        # Taken from a bug report from N. Becker on the numpy-discussion
        # mailing list Aug. 6, 2010.
        counts, dmy = np.histogram([1,2,3,4], [0.5,1.5,np.inf], density=True)
        assert_equal(counts, [.25, 0])

    def test_outliers(self):
        # Check that outliers are not tallied
        a = arange(10) + .5

        # Lower outliers
        h, b = histogram(a, range=[0, 9])
        assert_equal(h.sum(), 9)

        # Upper outliers
        h, b = histogram(a, range=[1, 10])
        assert_equal(h.sum(), 9)

        # Normalization
        h, b = histogram(a, range=[1, 9], normed=True)
        assert_equal((h * diff(b)).sum(), 1)

        # Weights
        w = arange(10) + .5
        h, b = histogram(a, range=[1, 9], weights=w, normed=True)
        assert_equal((h * diff(b)).sum(), 1)

        h, b = histogram(a, bins=8, range=[1, 9], weights=w)
        assert_equal(h, w[1:-1])

    def test_type(self):
        # Check the type of the returned histogram
        a = arange(10) + .5
        h, b = histogram(a)
        assert_(issubdtype(h.dtype, int))

        h, b = histogram(a, normed=True)
        assert_(issubdtype(h.dtype, float))

        h, b = histogram(a, weights=ones(10, int))
        assert_(issubdtype(h.dtype, int))

        h, b = histogram(a, weights=ones(10, float))
        assert_(issubdtype(h.dtype, float))

    def test_weights(self):
        v = rand(100)
        w = ones(100) * 5
        a, b = histogram(v)
        na, nb = histogram(v, normed=True)
        wa, wb = histogram(v, weights=w)
        nwa, nwb = histogram(v, weights=w, normed=True)
        assert_array_almost_equal(a * 5, wa)
        assert_array_almost_equal(na, nwa)

        # Check weights are properly applied.
        v = linspace(0, 10, 10)
        w = concatenate((zeros(5), ones(5)))
        wa, wb = histogram(v, bins=arange(11), weights=w)
        assert_array_almost_equal(wa, w)

        # Check with integer weights
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1])
        assert_array_equal(wa, [4, 5, 0, 1])
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1], normed=True)
        assert_array_almost_equal(wa, array([4, 5, 0, 1]) / 10. / 3. * 4)

        # Check weights with non-uniform bin widths
        a,b = histogram(np.arange(9), [0,1,3,6,10], \
                        weights=[2,1,1,1,1,1,1,1,1], density=True)
        assert_almost_equal(a, [.2, .1, .1, .075])

    def test_empty(self):
        a, b = histogram([], bins=([0,1]))
        assert_array_equal(a, array([0]))
        assert_array_equal(b, array([0, 1]))


class TestHistogramdd(TestCase):
    def test_simple(self):
        x = array([[-.5, .5, 1.5], [-.5, 1.5, 2.5], [-.5, 2.5, .5], \
        [.5, .5, 1.5], [.5, 1.5, 2.5], [.5, 2.5, 2.5]])
        H, edges = histogramdd(x, (2, 3, 3), range=[[-1, 1], [0, 3], [0, 3]])
        answer = asarray([[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1],
            [0, 0, 1]]])
        assert_array_equal(H, answer)
        # Check normalization
        ed = [[-2, 0, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
        H, edges = histogramdd(x, bins=ed, normed=True)
        assert_(all(H == answer / 12.))
        # Check that H has the correct shape.
        H, edges = histogramdd(x, (2, 3, 4), range=[[-1, 1], [0, 3], [0, 4]],
            normed=True)
        answer = asarray([[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], [[0, 1, 0, 0],
            [0, 0, 1, 0], [0, 0, 1, 0]]])
        assert_array_almost_equal(H, answer / 6., 4)
        # Check that a sequence of arrays is accepted and H has the correct
        # shape.
        z = [squeeze(y) for y in split(x, 3, axis=1)]
        H, edges = histogramdd(z, bins=(4, 3, 2), range=[[-2, 2], [0, 3], [0, 2]])
        answer = asarray([[[0, 0], [0, 0], [0, 0]],
                          [[0, 1], [0, 0], [1, 0]],
                          [[0, 1], [0, 0], [0, 0]],
                          [[0, 0], [0, 0], [0, 0]]])
        assert_array_equal(H, answer)

        Z = zeros((5, 5, 5))
        Z[range(5), range(5), range(5)] = 1.
        H, edges = histogramdd([arange(5), arange(5), arange(5)], 5)
        assert_array_equal(H, Z)

    def test_shape_3d(self):
        # All possible permutations for bins of different lengths in 3D.
        bins = ((5, 4, 6), (6, 4, 5), (5, 6, 4), (4, 6, 5), (6, 5, 4),
            (4, 5, 6))
        r = rand(10, 3)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

    def test_shape_4d(self):
        # All possible permutations for bins of different lengths in 4D.
        bins = ((7, 4, 5, 6), (4, 5, 7, 6), (5, 6, 4, 7), (7, 6, 5, 4),
            (5, 7, 6, 4), (4, 6, 7, 5), (6, 5, 7, 4), (7, 5, 4, 6),
            (7, 4, 6, 5), (6, 4, 7, 5), (6, 7, 5, 4), (4, 6, 5, 7),
            (4, 7, 5, 6), (5, 4, 6, 7), (5, 7, 4, 6), (6, 7, 4, 5),
            (6, 5, 4, 7), (4, 7, 6, 5), (4, 5, 6, 7), (7, 6, 4, 5),
            (5, 4, 7, 6), (5, 6, 7, 4), (6, 4, 5, 7), (7, 5, 6, 4))

        r = rand(10, 4)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

    def test_weights(self):
        v = rand(100, 2)
        hist, edges = histogramdd(v)
        n_hist, edges = histogramdd(v, normed=True)
        w_hist, edges = histogramdd(v, weights=ones(100))
        assert_array_equal(w_hist, hist)
        w_hist, edges = histogramdd(v, weights=ones(100) * 2, normed=True)
        assert_array_equal(w_hist, n_hist)
        w_hist, edges = histogramdd(v, weights=ones(100, int) * 2)
        assert_array_equal(w_hist, 2 * hist)

    def test_identical_samples(self):
        x = zeros((10, 2), int)
        hist, edges = histogramdd(x, bins=2)
        assert_array_equal(edges[0], array([-0.5, 0. , 0.5]))

    def test_empty(self):
        a, b = histogramdd([[], []], bins=([0,1], [0,1]))
        assert_array_max_ulp(a, array([ 0., 0.]))

    def test_bins_errors(self):
        """There are two ways to specify bins. Check for the right errors when
        mixing those."""
        x = np.arange(8).reshape(2, 4)
        assert_raises(ValueError, np.histogramdd, x, bins=[-1, 2, 4, 5])
        assert_raises(ValueError, np.histogramdd, x, bins=[1, 0.99, 1, 1])
        assert_raises(ValueError, np.histogramdd, x, bins=[1, 1, 1, [1, 2, 2, 3]])
        assert_raises(ValueError, np.histogramdd, x, bins=[1, 1, 1, [1, 2, 3, -3]])
        assert_(np.histogramdd(x, bins=[1, 1, 1, [1, 2, 3, 4]]))

    def test_inf_edges(self):
        """Test using +/-inf bin edges works. See #1788."""
        x = np.arange(6).reshape(3, 2)
        expected = np.array([[1, 0], [0, 1], [0, 1]])
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 2, 10]])
        assert_allclose(h, expected)
        h, e = np.histogramdd(x, bins=[3, np.array([-1, 2, np.inf])])
        assert_allclose(h, expected)
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 3, np.inf]])
        assert_allclose(h, expected)


class TestUnique(TestCase):
    def test_simple(self):
        x = array([4, 3, 2, 1, 1, 2, 3, 4, 0])
        assert_(all(unique(x) == [0, 1, 2, 3, 4]))
        assert_(unique(array([1, 1, 1, 1, 1])) == array([1]))
        x = ['widget', 'ham', 'foo', 'bar', 'foo', 'ham']
        assert_(all(unique(x) == ['bar', 'foo', 'ham', 'widget']))
        x = array([5 + 6j, 1 + 1j, 1 + 10j, 10, 5 + 6j])
        assert_(all(unique(x) == [1 + 1j, 1 + 10j, 5 + 6j, 10]))


class TestCheckFinite(TestCase):
    def test_simple(self):
        a = [1, 2, 3]
        b = [1, 2, inf]
        c = [1, 2, nan]
        numpy.lib.asarray_chkfinite(a)
        assert_raises(ValueError, numpy.lib.asarray_chkfinite, b)
        assert_raises(ValueError, numpy.lib.asarray_chkfinite, c)


class TestNaNFuncts(TestCase):
    def setUp(self):
        self.A = array([[[ nan, 0.01319214, 0.01620964],
                         [ 0.11704017, nan, 0.75157887],
                         [ 0.28333658, 0.1630199 , nan       ]],
                        [[ 0.59541557, nan, 0.37910852],
                         [ nan, 0.87964135, nan       ],
                         [ 0.70543747, nan, 0.34306596]],
                        [[ 0.72687499, 0.91084584, nan       ],
                         [ 0.84386844, 0.38944762, 0.23913896],
                         [ nan, 0.37068164, 0.33850425]]])

    def test_nansum(self):
        assert_almost_equal(nansum(self.A), 8.0664079100000006)
        assert_almost_equal(nansum(self.A, 0),
                            array([[ 1.32229056, 0.92403798, 0.39531816],
                                   [ 0.96090861, 1.26908897, 0.99071783],
                                   [ 0.98877405, 0.53370154, 0.68157021]]))
        assert_almost_equal(nansum(self.A, 1),
                            array([[ 0.40037675, 0.17621204, 0.76778851],
                                   [ 1.30085304, 0.87964135, 0.72217448],
                                   [ 1.57074343, 1.6709751 , 0.57764321]]))
        assert_almost_equal(nansum(self.A, 2),
                            array([[ 0.02940178, 0.86861904, 0.44635648],
                                   [ 0.97452409, 0.87964135, 1.04850343],
                                   [ 1.63772083, 1.47245502, 0.70918589]]))

    def test_nanmin(self):
        assert_almost_equal(nanmin(self.A), 0.01319214)
        assert_almost_equal(nanmin(self.A, 0),
                            array([[ 0.59541557, 0.01319214, 0.01620964],
                                   [ 0.11704017, 0.38944762, 0.23913896],
                                   [ 0.28333658, 0.1630199 , 0.33850425]]))
        assert_almost_equal(nanmin(self.A, 1),
                            array([[ 0.11704017, 0.01319214, 0.01620964],
                                   [ 0.59541557, 0.87964135, 0.34306596],
                                   [ 0.72687499, 0.37068164, 0.23913896]]))
        assert_almost_equal(nanmin(self.A, 2),
                            array([[ 0.01319214, 0.11704017, 0.1630199 ],
                                   [ 0.37910852, 0.87964135, 0.34306596],
                                   [ 0.72687499, 0.23913896, 0.33850425]]))
        assert_(np.isnan(nanmin([nan, nan])))

    def test_nanargmin(self):
        assert_almost_equal(nanargmin(self.A), 1)
        assert_almost_equal(nanargmin(self.A, 0),
                            array([[1, 0, 0],
                                   [0, 2, 2],
                                   [0, 0, 2]]))
        assert_almost_equal(nanargmin(self.A, 1),
                            array([[1, 0, 0],
                                   [0, 1, 2],
                                   [0, 2, 1]]))
        assert_almost_equal(nanargmin(self.A, 2),
                            array([[1, 0, 1],
                                   [2, 1, 2],
                                   [0, 2, 2]]))

    def test_nanmax(self):
        assert_almost_equal(nanmax(self.A), 0.91084584000000002)
        assert_almost_equal(nanmax(self.A, 0),
                            array([[ 0.72687499, 0.91084584, 0.37910852],
                                   [ 0.84386844, 0.87964135, 0.75157887],
                                   [ 0.70543747, 0.37068164, 0.34306596]]))
        assert_almost_equal(nanmax(self.A, 1),
                            array([[ 0.28333658, 0.1630199 , 0.75157887],
                                   [ 0.70543747, 0.87964135, 0.37910852],
                                   [ 0.84386844, 0.91084584, 0.33850425]]))
        assert_almost_equal(nanmax(self.A, 2),
                            array([[ 0.01620964, 0.75157887, 0.28333658],
                                   [ 0.59541557, 0.87964135, 0.70543747],
                                   [ 0.91084584, 0.84386844, 0.37068164]]))
        assert_(np.isnan(nanmax([nan, nan])))

    def test_nanmin_allnan_on_axis(self):
        assert_array_equal(isnan(nanmin([[nan] * 2] * 3, axis=1)),
                     [True, True, True])

    def test_nanmin_masked(self):
        a = np.ma.fix_invalid([[2, 1, 3, nan], [5, 2, 3, nan]])
        ctrl_mask = a._mask.copy()
        test = np.nanmin(a, axis=1)
        assert_equal(test, [1, 2])
        assert_equal(a._mask, ctrl_mask)
        assert_equal(np.isinf(a), np.zeros((2, 4), dtype=bool))


class TestNanFunctsIntTypes(TestCase):

    int_types = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)

    def setUp(self, *args, **kwargs):
        self.A = array([127, 39,  93,  87, 46])

    def integer_arrays(self):
        for dtype in self.int_types:
            yield self.A.astype(dtype)

    def test_nanmin(self):
        min_value = min(self.A)
        for A in self.integer_arrays():
            assert_equal(nanmin(A), min_value)

    def test_nanmax(self):
        max_value = max(self.A)
        for A in self.integer_arrays():
            assert_equal(nanmax(A), max_value)

    def test_nanargmin(self):
        min_arg = argmin(self.A)
        for A in self.integer_arrays():
            assert_equal(nanargmin(A), min_arg)

    def test_nanargmax(self):
        max_arg = argmax(self.A)
        for A in self.integer_arrays():
            assert_equal(nanargmax(A), max_arg)


class TestCorrCoef(TestCase):
    A = array([[ 0.15391142, 0.18045767, 0.14197213],
               [ 0.70461506, 0.96474128, 0.27906989],
               [ 0.9297531 , 0.32296769, 0.19267156]])
    B = array([[ 0.10377691, 0.5417086 , 0.49807457],
               [ 0.82872117, 0.77801674, 0.39226705],
               [ 0.9314666 , 0.66800209, 0.03538394]])
    res1 = array([[ 1.        , 0.9379533 , -0.04931983],
               [ 0.9379533 , 1.        , 0.30007991],
               [-0.04931983, 0.30007991, 1.        ]])
    res2 = array([[ 1.        , 0.9379533 , -0.04931983,
                 0.30151751, 0.66318558, 0.51532523],
               [ 0.9379533 , 1.        , 0.30007991,
                 - 0.04781421, 0.88157256, 0.78052386],
               [-0.04931983, 0.30007991, 1.        ,
                 - 0.96717111, 0.71483595, 0.83053601],
               [ 0.30151751, -0.04781421, -0.96717111,
                 1.        , -0.51366032, -0.66173113],
               [ 0.66318558, 0.88157256, 0.71483595,
                 - 0.51366032, 1.        , 0.98317823],
               [ 0.51532523, 0.78052386, 0.83053601,
                 - 0.66173113, 0.98317823, 1.        ]])

    def test_simple(self):
        assert_almost_equal(corrcoef(self.A), self.res1)
        assert_almost_equal(corrcoef(self.A, self.B), self.res2)

    def test_ddof(self):
        assert_almost_equal(corrcoef(self.A, ddof=-1), self.res1)
        assert_almost_equal(corrcoef(self.A, self.B, ddof=-1), self.res2)

    def test_empty(self):
        assert_equal(corrcoef(np.array([])).size, 0)
        assert_equal(corrcoef(np.array([]).reshape(0, 2)).shape, (0, 2))


class TestCov(TestCase):
    def test_basic(self):
        x = np.array([[0, 2], [1, 1], [2, 0]]).T
        assert_allclose(np.cov(x), np.array([[ 1.,-1.], [-1.,1.]]))

    def test_empty(self):
        assert_equal(cov(np.array([])).size, 0)
        assert_equal(cov(np.array([]).reshape(0, 2)).shape, (0, 2))


class Test_i0(TestCase):
    def test_simple(self):
        assert_almost_equal(i0(0.5), array(1.0634833707413234))
        A = array([ 0.49842636, 0.6969809 , 0.22011976, 0.0155549])
        assert_almost_equal(i0(A),
                            array([ 1.06307822, 1.12518299, 1.01214991, 1.00006049]))
        B = array([[ 0.827002  , 0.99959078],
                   [ 0.89694769, 0.39298162],
                   [ 0.37954418, 0.05206293],
                   [ 0.36465447, 0.72446427],
                   [ 0.48164949, 0.50324519]])
        assert_almost_equal(i0(B),
                            array([[ 1.17843223, 1.26583466],
                                   [ 1.21147086, 1.0389829 ],
                                   [ 1.03633899, 1.00067775],
                                   [ 1.03352052, 1.13557954],
                                   [ 1.0588429 , 1.06432317]]))


class TestKaiser(TestCase):
    def test_simple(self):
        assert_almost_equal(kaiser(0, 1.0), array([]))
        assert_(isfinite(kaiser(1, 1.0)))
        assert_almost_equal(kaiser(2, 1.0), array([ 0.78984831, 0.78984831]))
        assert_almost_equal(kaiser(5, 1.0),
                            array([ 0.78984831, 0.94503323, 1.        ,
                                    0.94503323, 0.78984831]))
        assert_almost_equal(kaiser(5, 1.56789),
                            array([ 0.58285404, 0.88409679, 1.        ,
                                    0.88409679, 0.58285404]))

    def test_int_beta(self):
        kaiser(3, 4)


class TestMsort(TestCase):
    def test_simple(self):
        A = array([[ 0.44567325, 0.79115165, 0.5490053 ],
                   [ 0.36844147, 0.37325583, 0.96098397],
                   [ 0.64864341, 0.52929049, 0.39172155]])
        assert_almost_equal(msort(A),
                            array([[ 0.36844147, 0.37325583, 0.39172155],
                                   [ 0.44567325, 0.52929049, 0.5490053 ],
                                   [ 0.64864341, 0.79115165, 0.96098397]]))


class TestMeshgrid(TestCase):
    def test_simple(self):
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7])
        assert_(all(X == array([[1, 2, 3],
                               [1, 2, 3],
                               [1, 2, 3],
                               [1, 2, 3]])))
        assert_(all(Y == array([[4, 4, 4],
                               [5, 5, 5],
                               [6, 6, 6],
                               [7, 7, 7]])))


class TestPiecewise(TestCase):
    def test_simple(self):
        # Condition is single bool list
        x = piecewise([0, 0], [True, False], [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: single bool list
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])

        # Conditions is single bool array
        x = piecewise([0, 0], array([True, False]), [1])
        assert_array_equal(x, [1, 0])

        # Condition is single int array
        x = piecewise([0, 0], array([1, 0]), [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: int array
        x = piecewise([0, 0], [array([1, 0])], [1])
        assert_array_equal(x, [1, 0])


        x = piecewise([0, 0], [[False, True]], [lambda x:-1])
        assert_array_equal(x, [0, -1])

        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_default(self):
        # No value specified for x[1], should be 0
        x = piecewise([1, 2], [True, False], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = piecewise([1, 2], [True, False], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = array(3)
        y = piecewise(x, x > 3, [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)


class TestBincount(TestCase):
    def test_simple(self):
        y = np.bincount(np.arange(4))
        assert_array_equal(y, np.ones(4))

    def test_simple2(self):
        y = np.bincount(np.array([1, 5, 2, 4, 1]))
        assert_array_equal(y, np.array([0, 2, 1, 0, 1, 1]))

    def test_simple_weight(self):
        x = np.arange(4)
        w = np.array([0.2, 0.3, 0.5, 0.1])
        y = np.bincount(x, w)
        assert_array_equal(y, w)

    def test_simple_weight2(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1]))

    def test_with_minlength(self):
        x = np.array([0, 1, 0, 1, 1])
        y = np.bincount(x, minlength=3)
        assert_array_equal(y, np.array([2, 3, 0]))

    def test_with_minlength_smaller_than_maxvalue(self):
        x = np.array([0, 1, 1, 2, 2, 3, 3])
        y = np.bincount(x, minlength=2)
        assert_array_equal(y, np.array([1, 2, 2, 2]))

    def test_with_minlength_and_weights(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w, 8)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0]))

class TestInterp(TestCase):
    def test_exceptions(self):
        assert_raises(ValueError, interp, 0, [], [])
        assert_raises(ValueError, interp, 0, [0], [1, 2])

    def test_basic(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.linspace(0, 1, 50)
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_right_left_behavior(self):
        assert_equal(interp([-1, 0, 1], [0], [1]), [1,1,1])
        assert_equal(interp([-1, 0, 1], [0], [1], left=0), [0,1,1])
        assert_equal(interp([-1, 0, 1], [0], [1], right=0), [1,1,0])
        assert_equal(interp([-1, 0, 1], [0], [1], left=0, right=0), [0,1,0])

    def test_scalar_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = 0
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = .3
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.float32(.3)
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.float64(.3)
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_zero_dimensional_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.array(.3)
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.array(.3, dtype=object)
        assert_almost_equal(np.interp(x0, x, y), .3)


def compare_results(res, desired):
    for i in range(len(desired)):
        assert_array_equal(res[i], desired[i])


def test_percentile_list():
    assert_equal(np.percentile([1, 2, 3], 0), 1)

def test_percentile_out():
    x = np.array([1, 2, 3])
    y = np.zeros((3,))
    p = (1, 2, 3)
    np.percentile(x, p, out=y)
    assert_equal(y, np.percentile(x, p))

    x = np.array([[1, 2, 3],
                  [4, 5, 6]])

    y = np.zeros((3, 3))
    np.percentile(x, p, axis=0, out=y)
    assert_equal(y, np.percentile(x, p, axis=0))

    y = np.zeros((3, 2))
    np.percentile(x, p, axis=1, out=y)
    assert_equal(y, np.percentile(x, p, axis=1))


if __name__ == "__main__":
    run_module_suite()
