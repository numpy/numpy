from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (
        TestCase, run_module_suite, assert_, assert_raises, assert_equal,
        assert_warns)
from numpy.testing.decorators import skipif
from numpy import random_intel as rnd
from numpy.compat import asbytes
import sys
import warnings

from numpy.distutils.system_info import get_info
if get_info('mkl'):
    skip_test = False
else:
    skip_test = True


class TestSeed_Intel(TestCase):
    @skipif(skip_test)
    def test_scalar(self):
        evs_zero_seed = {
            'MT19937' : 844, 'SFMT19937' : 857,
            'WH' : 0,        'MT2203' : 890,
            'MCG31' : 0,     'R250' : 229,
            'MRG32K3A' : 0,  'MCG59' : 0      }
        for brng_algo in evs_zero_seed:
            s = rnd.RandomState(0, brng = brng_algo)
            assert_equal(s.get_state()[0], brng_algo)
            assert_equal(s.randint(1000), evs_zero_seed[brng_algo])
        evs_max_seed = {
            'MT19937' : 635,  'SFMT19937' : 25,
            'WH' : 100,       'MT2203' : 527,
            'MCG31' : 0,      'R250' : 229,
            'MRG32K3A' : 961, 'MCG59' : 0     }
        for brng_algo in evs_max_seed:
            s = rnd.RandomState(4294967295, brng = brng_algo)
            assert_equal(s.get_state()[0], brng_algo)
            assert_equal(s.randint(1000), evs_max_seed[brng_algo])

    @skipif(skip_test)
    def test_array(self):
        s = rnd.RandomState(range(10), brng='MT19937')
        assert_equal(s.randint(1000), 410)
        s = rnd.RandomState(np.arange(10), brng='MT19937')
        assert_equal(s.randint(1000), 410)
        s = rnd.RandomState([0], brng='MT19937')
        assert_equal(s.randint(1000), 844)
        s = rnd.RandomState([4294967295], brng='MT19937')
        assert_equal(s.randint(1000), 635)

    @skipif(skip_test)
    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integers
        assert_raises(TypeError, rnd.RandomState, -0.5)
        assert_raises(ValueError, rnd.RandomState, -1)

    @skipif(skip_test)
    def test_invalid_array(self):
        # seed must be an unsigned 32 bit integers
        assert_raises(TypeError, rnd.RandomState, [-0.5])
        assert_raises(ValueError, rnd.RandomState, [-1])
        assert_raises(ValueError, rnd.RandomState, [4294967296])
        assert_raises(ValueError, rnd.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, rnd.RandomState, [1, -2, 4294967296])

class TestBinomial_Intel(TestCase):
    @skipif(skip_test)
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        # This test addresses issue #3480.
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            assert_(rnd.binomial(0, p) == 0)
            np.testing.assert_array_equal(rnd.binomial(zeros, p), zeros)

    @skipif(skip_test)
    def test_p_is_nan(self):
        # Issue #4571.
        assert_raises(ValueError, rnd.binomial, 1, np.nan)


class TestMultinomial_Intel(TestCase):
    @skipif(skip_test)
    def test_basic(self):
        rnd.multinomial(100, [0.2, 0.8])

    @skipif(skip_test)
    def test_zero_probability(self):
        rnd.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    @skipif(skip_test)
    def test_int_negative_interval(self):
        assert_(-5 <= rnd.randint(-5, -1) < -1)
        x = rnd.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    @skipif(skip_test)
    def test_size(self):
        # gh-3173
        p = [0.5, 0.5]
        assert_equal(rnd.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(rnd.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(rnd.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(rnd.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(rnd.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(rnd.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, rnd.multinomial, 1, p,
                      np.float(1))


class TestSetState_Intel(TestCase):
    @skipif(skip_test)
    def setUp(self):
        self.seed = 1234567890
        self.prng = rnd.RandomState(self.seed)
        self.state = self.prng.get_state()

    @skipif(skip_test)
    def test_basic(self):
        old = self.prng.tomaxint(16)
        self.prng.set_state(self.state)
        new = self.prng.tomaxint(16)
        assert_(np.all(old == new))

    @skipif(skip_test)
    def test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(self.state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    @skipif(skip_test)
    def test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.
        self.prng.standard_normal()
        state = self.prng.get_state()
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    @skipif(skip_test)
    def test_backwards_compatibility(self):
        # Make sure we can accept old state tuples that do not have the
        # cached Gaussian value.
        if len(self.state) == 5:
            old_state = self.state[:-2]
            x1 = self.prng.standard_normal(size=16)
            self.prng.set_state(old_state)
            x2 = self.prng.standard_normal(size=16)
            self.prng.set_state(self.state)
            x3 = self.prng.standard_normal(size=16)
            assert_(np.all(x1 == x2))
            assert_(np.all(x1 == x3))

    @skipif(skip_test)
    def test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.prng.negative_binomial(0.5, 0.5)

class TestRandint_Intel(TestCase):
    @skipif(skip_test)
    def setUp(self):
        self.rfunc = rnd.randint
        # valid integer/boolean types
        self.itype = [np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                      np.int32, np.uint32, np.int64, np.uint64]

    @skipif(skip_test)
    def test_unsupported_type(self):
        assert_raises(TypeError, self.rfunc, 1, dtype=np.float)

    @skipif(skip_test)
    def test_bounds_checking(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)

    @skipif(skip_test)
    def test_rng_zero_and_extremes(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
            tgt = (lbnd + ubnd)//2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    @skipif(skip_test)
    def test_in_bounds_fuzz(self):
        # Don't use fixed seed
        rnd.seed()
        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd, size=2**16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)
        vals = self.rfunc(0, 2, size=2**16, dtype=np.bool)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    @skipif(skip_test)
    def test_repeatability(self):
        import hashlib
        # We use a md5 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but np.bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool': '4fee98a6885457da67c39331a9ec336f',
               'int16': '80a5ff69c315ab6f80b03da1d570b656',
               'int32': '15a3c379b6c7b0f296b162194eab68bc',
               'int64': 'ea9875f9334c2775b00d4976b85a1458',
               'int8': '0f56333af47de94930c799806158a274',
               'uint16': '80a5ff69c315ab6f80b03da1d570b656',
               'uint32': '15a3c379b6c7b0f296b162194eab68bc',
               'uint64': 'ea9875f9334c2775b00d4976b85a1458',
               'uint8': '0f56333af47de94930c799806158a274'}

        for dt in self.itype[1:]:
            rnd.seed(1234, brng='MT19937')

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()

            res = hashlib.md5(val.view(np.int8)).hexdigest()
            print("")
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianess
        rnd.seed(1234, brng='MT19937')
        val = self.rfunc(0, 2, size=1000, dtype=np.bool).view(np.int8)
        res = hashlib.md5(val).hexdigest()
        assert_(tgt[np.dtype(np.bool).name] == res)

    @skipif(skip_test)
    def test_respect_dtype_singleton(self):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1

            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            self.assertEqual(sample.dtype, np.dtype(dt))

        for dt in (np.bool, np.int, np.long):
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            # gh-7284: Ensure that we get Python data types
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            self.assertFalse(hasattr(sample, 'dtype'))
            self.assertEqual(type(sample), dt)


class TestRandomDist_Intel(TestCase):
    # Make sure the random distribution returns the correct value for a
    # given seed. Low value of decimal argument is intended, since functional
    # transformations's implementation or approximations thereof used to produce non-uniform
    # random variates can vary across platforms, yet be statistically indistinguishable to the end user,
    # that is no computationally feasible statistical experiment can detect the difference.

    @skipif(skip_test)
    def setUp(self):
        self.seed = 1234567890
        self.brng = 'SFMT19937'

    @skipif(skip_test)
    def test_rand(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.rand(3, 2)
        desired = np.array([[0.9838694715872407, 0.019142669625580311],
                            [0.1767608025111258, 0.70966427633538842],
                            [0.518550637178123, 0.98780936631374061]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)

    @skipif(skip_test)
    def test_randn(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.randn(3, 2)
        desired = np.array([[2.1411609928913298, -2.0717866791744819],
                            [-0.92778018318550248, 0.55240420724917727],
                            [0.04651632135517459, 2.2510674226058036]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)

    @skipif(skip_test)
    def test_randint(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.randint(-99, 99, size=(3, 2))
        desired = np.array([[95, -96], [-65, 41], [3, 96]])
        np.testing.assert_array_equal(actual, desired)

    @skipif(skip_test)
    def test_random_integers(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.random_integers(-99, 99, size=(3, 2))
        desired = np.array([[96, -96], [-64, 42], [4, 97]])
        np.testing.assert_array_equal(actual, desired)

    @skipif(skip_test)
    def test_random_integers_max_int(self):
        # Tests whether random_integers can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        actual = rnd.random_integers(np.iinfo('l').max,
                                           np.iinfo('l').max)
        desired = np.iinfo('l').max
        np.testing.assert_equal(actual, desired)

    @skipif(skip_test)
    def test_random_integers_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # DeprecationWarning raised with high == None
            assert_raises(DeprecationWarning,
                          rnd.random_integers,
                          np.iinfo('l').max)

            # DeprecationWarning raised with high != None
            assert_raises(DeprecationWarning,
                          rnd.random_integers,
                          np.iinfo('l').max, np.iinfo('l').max)

    @skipif(skip_test)
    def test_random_sample(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.random_sample((3, 2))
        desired = np.array([[0.9838694715872407, 0.01914266962558031],
                            [0.1767608025111258, 0.7096642763353884],
                            [0.518550637178123, 0.9878093663137406]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)

    @skipif(skip_test)
    def test_choice_uniform_replace(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.choice(4, 4)
        desired = np.array([3, 0, 0, 2])
        np.testing.assert_array_equal(actual, desired)

    @skipif(skip_test)
    def test_choice_nonuniform_replace(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([3, 0, 0, 1])
        np.testing.assert_array_equal(actual, desired)

    @skipif(skip_test)
    def test_choice_uniform_noreplace(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.choice(4, 3, replace=False)
        desired = np.array([2, 1, 3])
        np.testing.assert_array_equal(actual, desired)

    @skipif(skip_test)
    def test_choice_nonuniform_noreplace(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.choice(4, 3, replace=False,
                                  p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([3, 0, 1])
        np.testing.assert_array_equal(actual, desired)

    @skipif(skip_test)
    def test_choice_noninteger(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['d', 'a', 'a', 'c'])
        np.testing.assert_array_equal(actual, desired)

    @skipif(skip_test)
    def test_choice_exceptions(self):
        sample = rnd.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                                          p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2, replace=False,
                                          p=[1, 0, 0])

    @skipif(skip_test)
    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        assert_(np.isscalar(rnd.choice(2, replace=True)))
        assert_(np.isscalar(rnd.choice(2, replace=False)))
        assert_(np.isscalar(rnd.choice(2, replace=True, p=p)))
        assert_(np.isscalar(rnd.choice(2, replace=False, p=p)))
        assert_(np.isscalar(rnd.choice([1, 2], replace=True)))
        assert_(rnd.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(rnd.choice(arr, replace=True) is a)

        # Check 0-d array
        s = tuple()
        assert_(not np.isscalar(rnd.choice(2, s, replace=True)))
        assert_(not np.isscalar(rnd.choice(2, s, replace=False)))
        assert_(not np.isscalar(rnd.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(rnd.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(rnd.choice([1, 2], s, replace=True)))
        assert_(rnd.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(rnd.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_(rnd.choice(6, s, replace=True).shape, s)
        assert_(rnd.choice(6, s, replace=False).shape, s)
        assert_(rnd.choice(6, s, replace=True, p=p).shape, s)
        assert_(rnd.choice(6, s, replace=False, p=p).shape, s)
        assert_(rnd.choice(np.arange(6), s, replace=True).shape, s)

    @skipif(skip_test)
    def test_bytes(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.bytes(10)
        desired = asbytes('\xa4\xde\xde{\xb4\x88\xe6\x84*2')
        np.testing.assert_equal(actual, desired)

    @skipif(skip_test)
    def test_shuffle(self):
        # Test lists, arrays (of various dtypes), and multidimensional versions
        # of both, c-contiguous or not:
        for conv in [lambda x: np.array([]),
                     lambda x: x,
                     lambda x: np.asarray(x).astype(np.int8),
                     lambda x: np.asarray(x).astype(np.float32),
                     lambda x: np.asarray(x).astype(np.complex64),
                     lambda x: np.asarray(x).astype(object),
                     lambda x: [(i, i) for i in x],
                     lambda x: np.asarray([[i, i] for i in x]),
                     lambda x: np.vstack([x, x]).T,
                     # gh-4270
                     lambda x: np.asarray([(i, i) for i in x],
                                          [("a", object, 1),
                                           ("b", np.int32, 1)])]:
            rnd.seed(self.seed, self.brng)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            rnd.shuffle(alist)
            actual = alist
            desired = conv([9, 8, 5, 1, 6, 4, 7, 2, 3, 0])
            np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_shuffle_masked(self):
        # gh-3263
        a = np.ma.masked_values(np.reshape(range(20), (5,4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            rnd.shuffle(a)
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            rnd.shuffle(b)
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))


    @skipif(skip_test)
    def test_beta(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.beta(.1, .9, size=(3, 2))
        desired = np.array(
            [[0.9856952034381025, 4.35869375658114e-08],
             [0.0014230232791189966, 1.4981856288121975e-06],
             [1.426135763875603e-06, 4.5801786040477326e-07]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_binomial(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.binomial(100.123, .456, size=(3, 2))
        desired = np.array([[43, 48], [55, 48], [46, 53]])
        np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_chisquare(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.chisquare(50, size=(3, 2))
        desired = np.array([[50.955833609920589, 50.133178918244099],
                    [61.513615847062013, 50.757127871422448],
                    [52.79816819717081, 49.973023331993552]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_dirichlet(self):
        rnd.seed(self.seed, self.brng)
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = rnd.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.6332947001908874, 0.36670529980911254],
                             [0.5376828907571894, 0.4623171092428107]],
                            [[0.6835615930093024, 0.3164384069906976],
                             [0.5452378139016114, 0.45476218609838875]],
                            [[0.6498494402738553, 0.3501505597261446],
                             [0.5622024400324822, 0.43779755996751785]]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_dirichlet_size(self):
        # gh-3173
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(rnd.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(rnd.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(rnd.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(rnd.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(rnd.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(rnd.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, rnd.dirichlet, p, np.float(1))


    @skipif(skip_test)
    def test_exponential(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.exponential(1.1234, size=(3, 2))
        desired = np.array([[0.01826877748252199, 4.4439855151117005],
                            [1.9468048583654507, 0.38528493864979607],
                            [0.7377565464231758, 0.013779117663987912]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_f(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.f(12, 77, size=(3, 2))
        desired = np.array([[1.325076177478387, 0.8670927327120197],
                            [2.1190792007836827, 0.9095296301824258],
                            [1.4953697422236187, 0.9547125618834837]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=9)


    @skipif(skip_test)
    def test_gamma(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.gamma(5, 3, size=(3, 2))
        desired = np.array([[15.073510060334929, 14.525495858042685],
                            [22.73897210140115, 14.94044782480266],
                            [16.327929995271095, 14.419692564592896]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_geometric(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.geometric(.123456789, size=(3, 2))
        desired = np.array([[0, 30], [13, 2], [4, 0]])
        np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_gumbel(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-8.114386462751979, 2.873840411460178],
                            [1.2231161758452016, -2.0168070493213532],
                            [-0.7175455966332102, -8.678464904504784]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_hypergeometric(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[10, 9], [9, 10], [9, 10]])
        np.testing.assert_array_equal(actual, desired)

        # Test nbad = 0
        actual = rnd.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        np.testing.assert_array_equal(actual, desired)

        actual = rnd.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        np.testing.assert_array_equal(actual, desired)

        # Test ngood = 0
        actual = rnd.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        np.testing.assert_array_equal(actual, desired)

        actual = rnd.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_laplace(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.15598087210935016, -3.3424589282252994],
                            [-1.189978401356375, 3.0607925598732253],
                            [0.0030946589024587745, 3.14795824463997]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_logistic(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[8.345015961402696, -7.749557532940552],
                            [-2.9534419690278444, 1.910964962531448],
                            [0.2719300361499433, 8.913100396613983]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_lognormal(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[81.92291750917155, 0.01795087229603931],
                            [0.1769118704670423, 3.415299544410577],
                            [1.2417099625339398, 102.0631392685238]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=6)
        actual = rnd.lognormal(mean=.123456789, sigma=2.0, size=(3,2),
                                     method='Box-Muller2')
        desired = np.array([[0.2585388231094821, 0.43734953048924663],
                            [26.050836228611697, 26.76266237820882],
                            [0.24216420175675096, 0.2481945765083541]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_logseries(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[18, 1], [1, 1], [5, 19]])
        np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_multinomial(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.multinomial(20, [1/6.]*6, size=(3, 2))
        desired = np.array([[[7, 0, 2, 4, 4, 3], [7, 1, 2, 6, 4, 0]],
                            [[2, 2, 3, 4, 6, 3], [1, 2, 5, 5, 6, 1]],
                            [[2, 7, 4, 1, 2, 4], [3, 5, 2, 5, 4, 1]]])
        np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_multivariate_normal(self):
        rnd.seed(self.seed, self.brng)
        mean = (.123456789, 10)
        # Hmm... not even symmetric.
        cov = [[1, 0], [1, 0]]
        size = (3, 2)
        actual = rnd.multivariate_normal(mean, cov, size)
        desired = np.array([[[-2.42282709811266, 10.0],
                             [1.2267795840027274, 10.0]],
                            [[0.06813924868067336, 10.0],
                             [1.001190462507746, 10.0]],
                            [[-1.74157261455869, 10.0],
                             [1.0400952859037553, 10.0]]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)

        # Check for default size, was raising deprecation warning
        actual = rnd.multivariate_normal(mean, cov)
        desired = np.array([1.0579899448949994, 10.0])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)

        # Check that non positive-semidefinite covariance raises warning
        mean = [0, 0]
        cov = [[1, 1 + 1e-10], [1 + 1e-10, 1]]
        assert_warns(RuntimeWarning, rnd.multivariate_normal, mean, cov)


    @skipif(skip_test)
    def test_multinormal_cholesky(self):
        rnd.seed(self.seed, self.brng)
        mean = (.123456789, 10)
        # lower-triangular cholesky matrix
        chol_mat = [[1, 0], [-0.5, 1]]
        size = (3, 2)
        actual = rnd.multinormal_cholesky(mean, chol_mat, size, method='ICDF')
        desired = np.array([[[2.26461778189133, 6.857632824379853],
                             [-0.8043233941855025, 11.01629429884193]],
                            [[0.1699731103551746, 12.227809261928217],
                             [-0.6146263106001378, 9.893801873973892]],
                            [[1.691753328795276, 10.797627196240155],
                             [-0.647341237129921, 9.626899489691816]]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_negative_binomial(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = np.array([[667, 679], [677, 676], [779, 648]])
        np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_noncentral_chisquare(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[5.871334619375055, 8.756238913383225],
                            [17.29576535176833, 3.9028417087862177],
                            [5.1315133729432505, 9.942717979531027]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)

        actual = rnd.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = np.array([[0.0008971007339949436, 0.08948578998156566],
                            [0.6721835871997511, 2.8892645287699352],
                            [5.0858149962761007e-05, 1.7315797643658821]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_noncentral_f(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                        size=(3, 2))
        desired = np.array([[0.2216297348371284, 0.7632696724492449],
                            [98.67664232828238, 0.9500319825372799],
                            [0.3489618249246971, 1.5035633972571092]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_normal(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.normal(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[4.405778774782659, -4.020116569348963],
                            [-1.732103577371005, 1.2282652034983546],
                            [0.21648943171034918, 4.625591634211608]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)

        rnd.seed(self.seed, self.brng)
        actual = rnd.normal(loc=.123456789, scale=2.0, size=(3, 2), method="BoxMuller")
        desired = np.array([[0.16673479781277187, -3.4809986872165952],
                            [-0.05193761082535492, 3.249201213154922],
                            [-0.11915582299214138, 3.555636100927892]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=8)

        rnd.seed(self.seed, self.brng)
        actual = rnd.normal(loc=.123456789, scale=2.0, size=(3, 2), method="BoxMuller2")
        desired = np.array([[0.16673479781277187, 0.48153966449249175],
                            [-3.4809986872165952, -0.8101190082826486],
                            [-0.051937610825354905, 2.4088402362484342]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_pareto(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.pareto(a=.123456789, size=(3, 2))
        desired = np.array(
            [[0.14079174875385214, 82372044085468.92],
             [1247881.6368437486, 15.086855668610944],
             [203.2638558933401, 0.10445383654349749]])
        # For some reason on 32-bit x86 Ubuntu 12.10 the [1, 0] entry in this
        # matrix differs by 24 nulps. Discussion:
        #   http://mail.scipy.org/pipermail/numpy-discussion/2012-September/063801.html
        # Consensus is that this is probably some gcc quirk that affects
        # rounding but not in any important way, so we just use a looser
        # tolerance on this test:
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)


    @skipif(skip_test)
    def test_poisson(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.poisson(lam=.123456789, size=(3, 2))
        desired = np.array([[1, 0], [0, 0], [0, 1]])
        np.testing.assert_array_equal(actual, desired)

        rnd.seed(self.seed, self.brng)
        actual = rnd.poisson(lam=1234.56789, size=(3, 2))
        desired = np.array([[1310, 1162], [1202, 1254], [1236, 1314]])
        np.testing.assert_array_equal(actual, desired)


    @skipif(skip_test)
    def test_poisson_exceptions(self):
        lambig = np.iinfo('l').max
        lamneg = -1
        assert_raises(ValueError, rnd.poisson, lamneg)
        assert_raises(ValueError, rnd.poisson, [lamneg]*10)
        assert_raises(ValueError, rnd.poisson, lambig)
        assert_raises(ValueError, rnd.poisson, [lambig]*10)


    @skipif(skip_test)
    def test_power(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.power(a=.123456789, size=(3, 2))
        desired = np.array([[0.8765841803224415, 1.2140041091640163e-14],
                            [8.013574117268635e-07, 0.06216255187464781],
                            [0.004895628723087296, 0.9054248959192386]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_rayleigh(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[1.80344345931194, 28.127692489122378],
                            [18.6169699930609, 8.282068232120208],
                            [11.460520015934597, 1.5662406536967712]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_standard_cauchy(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.standard_cauchy(size=(3, 2))
        desired = np.array([[19.716487700629912, -16.608240276131227],
                            [-1.6117703817332278, 0.7739915895826882],
                            [0.058344614106131, 26.09825325697747]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=9)


    @skipif(skip_test)
    def test_standard_exponential(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.standard_exponential(size=(3, 2))
        desired = np.array([[0.016262041554675085, 3.955835423813157],
                            [1.7329578586126497, 0.3429632710074738],
                            [0.6567175951781875, 0.012265548926462446]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_standard_gamma(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[2.939330965027084, 2.799606052259993],
                            [4.988193705918075, 2.905305108691164],
                            [3.2630929395548147, 2.772756340265377]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_standard_normal(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.standard_normal(size=(3, 2))
        desired = np.array([[2.1411609928913298, -2.071786679174482],
                            [-0.9277801831855025, 0.5524042072491773],
                            [0.04651632135517459, 2.2510674226058036]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)

        rnd.seed(self.seed, self.brng)
        actual = rnd.standard_normal(size=(3, 2), method='BoxMuller2')
        desired = np.array([[0.021639004406385935, 0.17904143774624587],
                            [-1.8022277381082976, -0.4667878986413243],
                            [-0.08769719991267745, 1.1426917236242171]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=7)


    @skipif(skip_test)
    def test_standard_t(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.standard_t(df=10, size=(3, 2))
        desired = np.array([[-0.783927044239963, 0.04762883516531178],
                            [0.7624597987725193, -1.8045540288955506],
                            [-1.2657694296239195, 0.307870906117017]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_triangular(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.triangular(left=5.12, mode=10.23, right=20.34,
                                      size=(3, 2))
        desired = np.array([[18.764540652669638, 6.340166306695037],
                            [8.827752689522429, 13.65605077739865],
                            [11.732872979633328, 18.970392754850423]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_uniform(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[10.38982478047721, 1.408218254214153],
                            [2.8756430713785814, 7.836974412682466],
                            [6.057706432128325, 10.426505200380925]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_uniform_range_bounds(self):
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        func = rnd.uniform
        np.testing.assert_raises(OverflowError, func, -np.inf, 0)
        np.testing.assert_raises(OverflowError, func,  0,      np.inf)
        # this should not throw any error, since rng can be sampled as fmin*u + fmax*(1-u)
        # for 0<u<1 and it stays completely in range
        rnd.uniform(fmin, fmax)

        # (fmax / 1e17) - fmin is within range, so this should not throw
        rnd.uniform(low=fmin, high=fmax / 1e17)


    @skipif(skip_test)
    def test_vonmises(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[1.1027657269593822, 1.2539311427727782],
                            [2.0281801137277764, 1.3262040229028056],
                            [0.9510301598100863, 2.0284972823322818]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_vonmises_small(self):
        # check infinite loop, gh-4720
        rnd.seed(self.seed, self.brng)
        r = rnd.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        np.testing.assert_(np.isfinite(r).all())


    @skipif(skip_test)
    def test_wald(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[0.3465678392232347, 0.3594497155916536],
                            [2.192908727996422, 1.7408141717513501],
                            [1.1943709105062374, 0.3273455943060196]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)


    @skipif(skip_test)
    def test_weibull(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.weibull(a=1.23, size=(3, 2))
        desired = np.array([[0.035129404330214734, 3.058859465984936],
                            [1.5636393343788513, 0.4189406773709585],
                            [0.710439924774508, 0.02793103204502023]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=10)

    @skipif(skip_test)
    def test_zipf(self):
        rnd.seed(self.seed, self.brng)
        actual = rnd.zipf(a=1.23, size=(3, 2))
        desired = np.array([[62062919, 1], [24, 209712763], [2, 24]])
        np.testing.assert_array_equal(actual, desired)


class TestThread_Intel(object):
    # make sure each state produces the same sequence even in threads
    @skipif(skip_test)
    def setUp(self):
        self.seeds = range(4)


    @skipif(skip_test)
    def check_function(self, function, sz):
        from threading import Thread

        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # threaded generation
        t = [Thread(target=function, args=(rnd.RandomState(s), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(rnd.RandomState(s), o)

        # these platforms change x87 fpu precision mode in threads
        if (np.intp().dtype.itemsize == 4 and sys.platform == "win32"):
            np.testing.assert_array_almost_equal(out1, out2)
        else:
            np.testing.assert_array_equal(out1, out2)


    @skipif(skip_test)
    def test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)
        self.check_function(gen_random, sz=(10000,))


    @skipif(skip_test)
    def test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))
        self.check_function(gen_random, sz=(100, 1000))


    @skipif(skip_test)
    def test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1/6.]*6, size=10000)
        self.check_function(gen_random, sz=(10000,6))


if __name__ == "__main__":
    run_module_suite()
