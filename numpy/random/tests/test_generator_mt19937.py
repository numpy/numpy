import sys

import pytest

import numpy as np
from numpy.testing import (
    assert_, assert_raises, assert_equal,
    assert_warns, assert_no_warnings, assert_array_equal,
    assert_array_almost_equal, suppress_warnings)

from numpy.random import Generator, MT19937

random = Generator(MT19937())


@pytest.fixture(scope='module', params=[True, False])
def endpoint(request):
    return request.param


class TestSeed(object):
    def test_scalar(self):
        s = Generator(MT19937(0))
        assert_equal(s.integers(1000), 548)
        s = Generator(MT19937(4294967295))
        assert_equal(s.integers(1000), 97)

    def test_array(self):
        s = Generator(MT19937(range(10)))
        assert_equal(s.integers(1000), 410)
        s = Generator(MT19937(np.arange(10)))
        assert_equal(s.integers(1000), 410)
        s = Generator(MT19937([0]))
        assert_equal(s.integers(1000), 844)
        s = Generator(MT19937([4294967295]))
        assert_equal(s.integers(1000), 635)

    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, MT19937, -0.5)
        assert_raises(ValueError, MT19937, -1)

    def test_invalid_array(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, MT19937, [-0.5])
        assert_raises(ValueError, MT19937, [-1])
        assert_raises(ValueError, MT19937, [4294967296])
        assert_raises(ValueError, MT19937, [1, 2, 4294967296])
        assert_raises(ValueError, MT19937, [1, -2, 4294967296])

    def test_noninstantized_bitgen(self):
        assert_raises(ValueError, Generator, MT19937)


class TestBinomial(object):
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        # This test addresses issue #3480.
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            assert_(random.binomial(0, p) == 0)
            assert_array_equal(random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        # Issue #4571.
        assert_raises(ValueError, random.binomial, 1, np.nan)


class TestMultinomial(object):
    def test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert_(-5 <= random.integers(-5, -1) < -1)
        x = random.integers(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def test_size(self):
        # gh-3173
        p = [0.5, 0.5]
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, random.multinomial, 1, p,
                      float(1))

    def test_invalid_prob(self):
        assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
        assert_raises(ValueError, random.multinomial, 100, [-.1, 0.9])

    def test_invalid_n(self):
        assert_raises(ValueError, random.multinomial, -1, [0.8, 0.2])
        assert_raises(ValueError, random.multinomial, [-1] * 10, [0.8, 0.2])
    
    def test_p_non_contiguous(self):
        p = np.arange(15.)
        p /= np.sum(p[1::3])
        pvals = p[1::3]
        random.bit_generator.seed(1432985819)
        non_contig = random.multinomial(100, pvals=pvals)
        random.bit_generator.seed(1432985819)
        contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
        assert_array_equal(non_contig, contig)


class TestSetState(object):
    def setup(self):
        self.seed = 1234567890
        self.rg = Generator(MT19937(self.seed))
        self.bit_generator = self.rg.bit_generator
        self.state = self.bit_generator.state
        self.legacy_state = (self.state['bit_generator'],
                             self.state['state']['key'],
                             self.state['state']['pos'])

    def test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = self.state
        new = self.rg.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.

        self.rg.standard_normal()
        state = self.bit_generator.state
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = state
        new = self.rg.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.rg.negative_binomial(0.5, 0.5)


class TestIntegers(object):
    rfunc = random.integers

    # valid integer/boolean types
    itype = [bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self, endpoint):
        assert_raises(TypeError, self.rfunc, 1, endpoint=endpoint, dtype=float)

    def test_bounds_checking(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, endpoint=endpoint,
                          dtype=dt)

            assert_raises(ValueError, self.rfunc, [lbnd - 1], ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd], [ubnd + 1],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd], [lbnd],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, [0],
                          endpoint=endpoint, dtype=dt)

    def test_bounds_checking_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + (not endpoint)

            assert_raises(ValueError, self.rfunc, [lbnd - 1] * 2, [ubnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd] * 2,
                          [ubnd + 1] * 2, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, [lbnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [1] * 2, 0,
                          endpoint=endpoint, dtype=dt)

    def test_rng_zero_and_extremes(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            is_open = not endpoint

            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc(tgt, [tgt + is_open], size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], [tgt + is_open],
                                    size=1000, endpoint=endpoint, dtype=dt),
                         tgt)

    def test_rng_zero_and_extremes_array(self, endpoint):
        size = 1000
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            tgt = ubnd - 1
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

    def test_full_range(self, endpoint):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_full_range_array(self, endpoint):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc([lbnd] * 2, [ubnd], endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_in_bounds_fuzz(self, endpoint):
        # Don't use fixed seed
        random.bit_generator.seed()

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd - endpoint, size=2 ** 16,
                                  endpoint=endpoint, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = self.rfunc(0, 2 - endpoint, size=2 ** 16, endpoint=endpoint,
                          dtype=bool)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_scalar_array_equiv(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            size = 1000
            random.bit_generator.seed(1234)
            scalar = self.rfunc(lbnd, ubnd, size=size, endpoint=endpoint,
                                dtype=dt)

            random.bit_generator.seed(1234)
            scalar_array = self.rfunc([lbnd], [ubnd], size=size,
                                      endpoint=endpoint, dtype=dt)

            random.bit_generator.seed(1234)
            array = self.rfunc([lbnd] * size, [ubnd] *
                               size, size=size, endpoint=endpoint, dtype=dt)
            assert_array_equal(scalar, scalar_array)
            assert_array_equal(scalar, array)

    def test_repeatability(self, endpoint):
        import hashlib
        # We use a md5 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool': '7dd3170d7aa461d201a65f8bcf3944b0',
               'int16': '2d26cafb53cb0f5acbb9b3fe86b36991',
               'int32': '54f153d6ae944ce0dde49a66602959bb',
               'int64': '47a068f62fda47f6034aa745e39a1b0d',
               'int8': '1d71d3947cd98598b4f00a77c117d62a',
               'uint16': '2d26cafb53cb0f5acbb9b3fe86b36991',
               'uint32': '54f153d6ae944ce0dde49a66602959bb',
               'uint64': '47a068f62fda47f6034aa745e39a1b0d',
               'uint8': '1d71d3947cd98598b4f00a77c117d62a'}

        for dt in self.itype[1:]:
            random.bit_generator.seed(1234)

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                 dtype=dt)
            else:
                val = self.rfunc(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                 dtype=dt).byteswap()

            res = hashlib.md5(val.view(np.int8)).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianness
        random.bit_generator.seed(1234)
        val = self.rfunc(0, 2 - endpoint, size=1000, endpoint=endpoint,
                         dtype=bool).view(np.int8)
        res = hashlib.md5(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    def test_repeatability_broadcasting(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt in (np.bool, bool, np.bool_) else np.iinfo(dt).min
            ubnd = 2 if dt in (
                np.bool, bool, np.bool_) else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # view as little endian for hash
            random.bit_generator.seed(1234)
            val = self.rfunc(lbnd, ubnd, size=1000, endpoint=endpoint,
                             dtype=dt)

            random.bit_generator.seed(1234)
            val_bc = self.rfunc([lbnd] * 1000, ubnd, endpoint=endpoint,
                                dtype=dt)

            assert_array_equal(val, val_bc)

            random.bit_generator.seed(1234)
            val_bc = self.rfunc([lbnd] * 1000, [ubnd] * 1000,
                                endpoint=endpoint, dtype=dt)

            assert_array_equal(val, val_bc)

    def test_int64_uint64_broadcast_exceptions(self, endpoint):
        configs = {np.uint64: ((0, 2**65), (-1, 2**62), (10, 9), (0, 0)),
                   np.int64: ((0, 2**64), (-(2**64), 2**62), (10, 9), (0, 0),
                              (-2**63-1, -2**63-1))}
        for dtype in configs:
            for config in configs[dtype]:
                low, high = config
                high = high - endpoint
                low_a = np.array([[low]*10])
                high_a = np.array([high] * 10)
                assert_raises(ValueError, random.integers, low, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_a,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high_a,
                              endpoint=endpoint, dtype=dtype)

                low_o = np.array([[low]*10], dtype=np.object)
                high_o = np.array([high] * 10, dtype=np.object)
                assert_raises(ValueError, random.integers, low_o, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_o,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_o, high_o,
                              endpoint=endpoint, dtype=dtype)

    def test_int64_uint64_corner_case(self, endpoint):
        # When stored in Numpy arrays, `lbnd` is casted
        # as np.int64, and `ubnd` is casted as np.uint64.
        # Checking whether `lbnd` >= `ubnd` used to be
        # done solely via direct comparison, which is incorrect
        # because when Numpy tries to compare both numbers,
        # it casts both to np.float64 because there is
        # no integer superset of np.int64 and np.uint64. However,
        # `ubnd` is too large to be represented in np.float64,
        # causing it be round down to np.iinfo(np.int64).max,
        # leading to a ValueError because `lbnd` now equals
        # the new `ubnd`.

        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1 - endpoint)

        # None of these function calls should
        # generate a ValueError now.
        actual = random.integers(lbnd, ubnd, endpoint=endpoint, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool_ if dt is bool else dt

            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)

        for dt in (bool, int, np.long):
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # gh-7284: Ensure that we get Python data types
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert not hasattr(sample, 'dtype')
            assert_equal(type(sample), dt)

    def test_respect_dtype_array(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool_ if dt is bool else dt

            sample = self.rfunc([lbnd], [ubnd], endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)
            sample = self.rfunc([lbnd] * 2, [ubnd] * 2, endpoint=endpoint,
                                dtype=dt)
            assert_equal(sample.dtype, dt)

    def test_zero_size(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            sample = self.rfunc(0, 0, (3, 0, 4), endpoint=endpoint, dtype=dt)
            assert sample.shape == (3, 0, 4)
            assert sample.dtype == dt
            assert self.rfunc(0, -10, 0, endpoint=endpoint,
                              dtype=dt).shape == (0,)
            assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape,
                         (3, 0, 4))
            assert_equal(random.integers(0, -10, size=0).shape, (0,))
            assert_equal(random.integers(10, 10, size=0).shape, (0,))

    def test_error_byteorder(self):
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        with pytest.raises(ValueError):
            random.integers(0, 200, size=10, dtype=other_byteord_dt)


class TestRandomDist(object):
    # Make sure the random distribution returns the correct value for a
    # given seed

    def setup(self):
        self.seed = 1234567890

    def test_integers(self):
        random.bit_generator.seed(self.seed)
        actual = random.integers(-99, 99, size=(3, 2))
        desired = np.array([[23, -32],
                            [18, -70],
                            [76, -53]])
        assert_array_equal(actual, desired)

    def test_integers_masked(self):
        # Test masked rejection sampling algorithm to generate array of
        # uint32 in an interval.
        random.bit_generator.seed(self.seed)
        actual = random.integers(0, 99, size=(3, 2), dtype=np.uint32)
        desired = np.array([[61, 33],
                            [58, 14],
                            [87, 23]], dtype=np.uint32)
        assert_array_equal(actual, desired)

    def test_integers_closed(self):
        random.bit_generator.seed(self.seed)
        actual = random.integers(-99, 99, size=(3, 2), endpoint=True)
        desired = np.array([[24, -32], [18, -70], [77, -53]])
        assert_array_equal(actual, desired)

    def test_integers_max_int(self):
        # Tests whether integers with closed=True can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        actual = random.integers(np.iinfo('l').max, np.iinfo('l').max,
                                 endpoint=True)

        desired = np.iinfo('l').max
        assert_equal(actual, desired)

    def test_random(self):
        random.bit_generator.seed(self.seed)
        actual = random.random((3, 2))
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

        random.bit_generator.seed(self.seed)
        actual = random.random()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_random_float(self):
        random.bit_generator.seed(self.seed)
        actual = random.random((3, 2))
        desired = np.array([[0.6187948, 0.5916236],
                            [0.8886836, 0.8916548],
                            [0.4575675, 0.7781881]])
        assert_array_almost_equal(actual, desired, decimal=7)

    def test_random_float_scalar(self):
        random.bit_generator.seed(self.seed)
        actual = random.random(dtype=np.float32)
        desired = 0.6187948
        assert_array_almost_equal(actual, desired, decimal=7)

    def test_random_unsupported_type(self):
        assert_raises(TypeError, random.random, dtype='int32')

    def test_choice_uniform_replace(self):
        random.bit_generator.seed(self.seed)
        actual = random.choice(4, 4)
        desired = np.array([2, 1, 2, 0], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_replace(self):
        random.bit_generator.seed(self.seed)
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([1, 1, 2, 2], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_uniform_noreplace(self):
        random.bit_generator.seed(self.seed)
        actual = random.choice(4, 3, replace=False)
        desired = np.array([0, 2, 3], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_noreplace(self):
        random.bit_generator.seed(self.seed)
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([2, 3, 1], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_noninteger(self):
        random.bit_generator.seed(self.seed)
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['c', 'b', 'c', 'a'])
        assert_array_equal(actual, desired)

    def test_choice_multidimensional_default_axis(self):
        random.bit_generator.seed(self.seed)
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 3)
        desired = np.array([[4, 5], [2, 3], [4, 5]])
        assert_array_equal(actual, desired)

    def test_choice_multidimensional_custom_axis(self):
        random.bit_generator.seed(self.seed)
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 1, axis=1)
        desired = np.array([[1], [3], [5], [7]])
        assert_array_equal(actual, desired)

    def test_choice_exceptions(self):
        sample = random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # gh-13087
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])

    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        assert_(np.isscalar(random.choice(2, replace=True)))
        assert_(np.isscalar(random.choice(2, replace=False)))
        assert_(np.isscalar(random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(random.choice([1, 2], replace=True)))
        assert_(random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)

        # Check 0-d array
        s = tuple()
        assert_(not np.isscalar(random.choice(2, s, replace=True)))
        assert_(not np.isscalar(random.choice(2, s, replace=False)))
        assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))
        assert_(random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)
        assert_equal(random.choice(6, s, replace=False).shape, s)
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)

        # Check zero-size
        assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.integers(0, -10, size=0).shape, (0,))
        assert_equal(random.integers(10, 10, size=0).shape, (0,))
        assert_equal(random.choice(0, size=0).shape, (0,))
        assert_equal(random.choice([], size=(0,)).shape, (0,))
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)
    
    def test_choice_p_non_contiguous(self):
        p = np.ones(10) / 5
        p[1::2] = 3.0
        random.bit_generator.seed(self.seed)
        non_contig = random.choice(5, 3, p=p[::2])
        random.bit_generator.seed(self.seed)
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)

    def test_choice_return_type(self):
        # gh 9867
        p = np.ones(4) / 4.
        actual = random.choice(4, 2)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, replace=False)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, p=p)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, p=p, replace=False)
        assert actual.dtype == np.int64

    def test_choice_large_sample(self):
        import hashlib

        choice_hash = '6395868be877d27518c832213c17977c'
        random.bit_generator.seed(self.seed)
        actual = random.choice(10000, 5000, replace=False)
        if sys.byteorder != 'little':
            actual = actual.byteswap()
        res = hashlib.md5(actual.view(np.int8)).hexdigest()
        assert_(choice_hash == res)

    def test_bytes(self):
        random.bit_generator.seed(self.seed)
        actual = random.bytes(10)
        desired = b'\x82Ui\x9e\xff\x97+Wf\xa5'
        assert_equal(actual, desired)

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
                     # gh-11442
                     lambda x: (np.asarray([(i, i) for i in x],
                                           [("a", int), ("b", int)])
                                .view(np.recarray)),
                     # gh-4270
                     lambda x: np.asarray([(i, i) for i in x],
                                          [("a", object, (1,)),
                                           ("b", np.int32, (1,))])]:
            random.bit_generator.seed(self.seed)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            random.shuffle(alist)
            actual = alist
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
            assert_array_equal(actual, desired)

    def test_shuffle_masked(self):
        # gh-3263
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            random.shuffle(a)
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            random.shuffle(b)
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    def test_permutation(self):
        random.bit_generator.seed(self.seed)
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actual = random.permutation(alist)
        desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
        assert_array_equal(actual, desired)

        random.bit_generator.seed(self.seed)
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        actual = random.permutation(arr_2d)
        assert_array_equal(actual, np.atleast_2d(desired).T)

    def test_beta(self):
        random.bit_generator.seed(self.seed)
        actual = random.beta(.1, .9, size=(3, 2))
        desired = np.array(
            [[1.45341850513746058e-02, 5.31297615662868145e-04],
             [1.85366619058432324e-06, 4.19214516800110563e-03],
             [1.58405155108498093e-04, 1.26252891949397652e-04]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        random.bit_generator.seed(self.seed)
        actual = random.binomial(100.123, .456, size=(3, 2))
        desired = np.array([[37, 43],
                            [42, 48],
                            [46, 45]])
        assert_array_equal(actual, desired)

        random.bit_generator.seed(self.seed)
        actual = random.binomial(100.123, .456)
        desired = 37
        assert_array_equal(actual, desired)

    def test_chisquare(self):
        random.bit_generator.seed(self.seed)
        actual = random.chisquare(50, size=(3, 2))
        desired = np.array([[22.2534560369812, 46.9302393710074],
                            [52.9974164611614, 85.3559029505718],
                            [46.1580841240719, 36.1933148548090]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        random.bit_generator.seed(self.seed)
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.444382290764855, 0.555617709235145],
                             [0.468440809291970, 0.531559190708030]],
                            [[0.613461427360549, 0.386538572639451],
                             [0.529103072088183, 0.470896927911817]],
                            [[0.513490650101800, 0.486509349898200],
                             [0.558550925712797, 0.441449074287203]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        bad_alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)

        random.bit_generator.seed(self.seed)
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha)
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_dirichlet_size(self):
        # gh-3173
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        # gh-2089
        alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, alpha)
    
    def test_dirichlet_alpha_non_contiguous(self):
        a = np.array([51.72840233779265162, -1.0, 39.74494232180943953])
        alpha = a[::2]
        random.bit_generator.seed(self.seed)
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random.bit_generator.seed(self.seed)
        contig = random.dirichlet(np.ascontiguousarray(alpha),
                                  size=(3, 2))
        assert_array_almost_equal(non_contig, contig)

    def test_exponential(self):
        random.bit_generator.seed(self.seed)
        actual = random.exponential(1.1234, size=(3, 2))
        desired = np.array([[5.350682337747634, 1.152307441755771],
                            [3.867015473358779, 1.538765912839396],
                            [0.347846818048527, 2.715656549872026]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        assert_equal(random.exponential(scale=0), 0)
        assert_raises(ValueError, random.exponential, scale=-0.)

    def test_f(self):
        random.bit_generator.seed(self.seed)
        actual = random.f(12, 77, size=(3, 2))
        desired = np.array([[0.809498839488467, 2.867222762455471],
                            [0.588036831639353, 1.012185639664636],
                            [1.147554281917365, 1.150886518432105]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        random.bit_generator.seed(self.seed)
        actual = random.gamma(5, 3, size=(3, 2))
        desired = np.array([[12.46569350177219, 16.46580642087044],
                            [43.65744473309084, 11.98722785682592],
                            [6.50371499559955, 7.48465689751638]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        assert_equal(random.gamma(shape=0, scale=0), 0)
        assert_raises(ValueError, random.gamma, shape=-0., scale=-0.)

    def test_geometric(self):
        random.bit_generator.seed(self.seed)
        actual = random.geometric(.123456789, size=(3, 2))
        desired = np.array([[8, 7],
                            [17, 17],
                            [5, 12]])
        assert_array_equal(actual, desired)

    def test_geometric_exceptions(self):
        assert_raises(ValueError, random.geometric, 1.1)
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        assert_raises(ValueError, random.geometric, -0.1)
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.geometric, np.nan)
            assert_raises(ValueError, random.geometric, [np.nan] * 10)

    def test_gumbel(self):
        random.bit_generator.seed(self.seed)
        actual = random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.19591898743416816, 0.34405539668096674],
                            [-1.4492522252274278, -1.47374816298446865],
                            [1.10651090478803416, -0.69535848626236174]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        assert_equal(random.gumbel(scale=0), 0)
        assert_raises(ValueError, random.gumbel, scale=-0.)

    def test_hypergeometric(self):
        random.bit_generator.seed(self.seed)
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[10, 10],
                            [10, 10],
                            [9, 9]])
        assert_array_equal(actual, desired)

        # Test nbad = 0
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # Test ngood = 0
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def test_laplace(self):
        random.bit_generator.seed(self.seed)
        actual = random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.66599721112760157, 0.52829452552221945],
                            [3.12791959514407125, 3.18202813572992005],
                            [-0.05391065675859356, 1.74901336242837324]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        assert_equal(random.laplace(scale=0), 0)
        assert_raises(ValueError, random.laplace, scale=-0.)

    def test_logistic(self):
        random.bit_generator.seed(self.seed)
        actual = random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[1.09232835305011444, 0.8648196662399954],
                            [4.27818590694950185, 4.33897006346929714],
                            [-0.21682183359214885, 2.63373365386060332]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        random.bit_generator.seed(self.seed)
        actual = random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[1.0894838661036e-03, 9.0990021488311e-01],
                            [6.9178869932225e-01, 2.7672077560016e-01],
                            [2.3248645126975e+00, 1.4609997951330e+00]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_lognormal_0(self):
        assert_equal(random.lognormal(sigma=0), 1)
        assert_raises(ValueError, random.lognormal, sigma=-0.)

    def test_logseries(self):
        random.bit_generator.seed(self.seed)
        actual = random.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[2, 2],
                            [6, 17],
                            [3, 6]])
        assert_array_equal(actual, desired)

    def test_logseries_exceptions(self):
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.logseries, np.nan)
            assert_raises(ValueError, random.logseries, [np.nan] * 10)

    def test_multinomial(self):
        random.bit_generator.seed(self.seed)
        actual = random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[4, 3, 5, 4, 2, 2],
                             [5, 2, 8, 2, 2, 1]],
                            [[3, 4, 3, 6, 0, 4],
                             [2, 1, 4, 3, 6, 4]],
                            [[4, 4, 2, 5, 2, 3],
                             [4, 3, 4, 2, 3, 4]]])
        assert_array_equal(actual, desired)

    def test_multivariate_normal(self):
        random.bit_generator.seed(self.seed)
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = random.multivariate_normal(mean, cov, size)
        desired = np.array([[[-3.34929721161096100, 9.891061435770858],
                             [-0.12250896439641100, 9.295898449738300]],
                            [[0.48355927611635563, 10.127832101772366],
                             [3.11093021424924300, 10.283109168794352]],
                            [[-0.20332082341774727, 9.868532121697195],
                             [-1.33806889550667330, 9.813657233804179]]])

        assert_array_almost_equal(actual, desired, decimal=15)

        # Check for default size, was raising deprecation warning
        actual = random.multivariate_normal(mean, cov)
        desired = np.array([-1.097443117192574, 10.535787051184261])
        assert_array_almost_equal(actual, desired, decimal=15)

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')

        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0

        mu = np.zeros(2)
        cov = np.eye(2)
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='other')
        assert_raises(ValueError, random.multivariate_normal,
                      np.zeros((2, 1, 1)), cov)
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.empty((3, 2)))
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.eye(3))

    def test_negative_binomial(self):
        random.bit_generator.seed(self.seed)
        actual = random.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = np.array([[521, 736],
                            [665, 690],
                            [723, 751]])
        assert_array_equal(actual, desired)

    def test_negative_binomial_exceptions(self):
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.negative_binomial, 100, np.nan)
            assert_raises(ValueError, random.negative_binomial, 100,
                          [np.nan] * 10)

    def test_noncentral_chisquare(self):
        random.bit_generator.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[9.47783251920357, 10.02066178260461],
                            [3.15869984192364, 10.5581565031544],
                            [5.01652540543548, 13.7689551218441]])
        assert_array_almost_equal(actual, desired, decimal=14)

        actual = random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = np.array([[0.00145153051285, 0.22432468724778],
                            [0.02956713468556, 0.00207192946898],
                            [1.41985055641800, 0.15451287602753]])
        assert_array_almost_equal(actual, desired, decimal=14)

        random.bit_generator.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[3.64881368071039, 5.48224544747803],
                            [20.41999842025404, 3.44075915187367],
                            [1.29765160605552, 1.64125033268606]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        random.bit_generator.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                     size=(3, 2))
        desired = np.array([[1.22680230963236, 2.56457837623956],
                            [2.7653304499494, 7.4336268865443],
                            [1.16362730891403, 2.54104276581491]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f_nan(self):
        random.bit_generator.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        assert np.isnan(actual)

    def test_normal(self):
        random.bit_generator.seed(self.seed)
        actual = random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-6.822051212221923, -0.094420339458285],
                            [-0.368474717792823, -1.284746311523402],
                            [0.843661763232711, 0.379120992544734]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_normal_0(self):
        assert_equal(random.normal(scale=0), 0)
        assert_raises(ValueError, random.normal, scale=-0.)

    def test_pareto(self):
        random.bit_generator.seed(self.seed)
        actual = random.pareto(a=.123456789, size=(3, 2))
        desired = np.array([[5.6883528121891552e+16, 4.0569373841667057e+03],
                            [1.2854967019379475e+12, 6.5833156486851483e+04],
                            [1.1281132447159091e+01, 3.1895968171107006e+08]])
        # For some reason on 32-bit x86 Ubuntu 12.10 the [1, 0] entry in this
        # matrix differs by 24 nulps. Discussion:
        #   https://mail.python.org/pipermail/numpy-discussion/2012-September/063801.html
        # Consensus is that this is probably some gcc quirk that affects
        # rounding but not in any important way, so we just use a looser
        # tolerance on this test:
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        random.bit_generator.seed(self.seed)
        actual = random.poisson(lam=.123456789, size=(3, 2))
        desired = np.array([[0, 0],
                            [1, 0],
                            [0, 0]])
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        lambig = np.iinfo('int64').max
        lamneg = -1
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)

    def test_power(self):
        random.bit_generator.seed(self.seed)
        actual = random.power(a=.123456789, size=(3, 2))
        desired = np.array([[9.328833342693975e-01, 2.742250409261003e-02],
                            [7.684513237993961e-01, 9.297548209160028e-02],
                            [2.214811188828573e-05, 4.693448360603472e-01]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        random.bit_generator.seed(self.seed)
        actual = random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[13.8882496494248393, 13.383318339044731],
                            [20.95413364294492098, 21.08285015800712614],
                            [11.06066537006854311, 17.35468505778271009]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        assert_equal(random.rayleigh(scale=0), 0)
        assert_raises(ValueError, random.rayleigh, scale=-0.)

    def test_standard_cauchy(self):
        random.bit_generator.seed(self.seed)
        actual = random.standard_cauchy(size=(3, 2))
        desired = np.array([[31.87809592667601, 0.349332782046838],
                            [2.816995747731641, 10.552372563459114],
                            [2.485608017991235, 7.843211273201831]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        random.bit_generator.seed(self.seed)
        actual = random.standard_exponential(size=(3, 2), method='inv')
        desired = np.array([[0.96441739162374596, 0.89556604882105506],
                            [2.1953785836319808, 2.22243285392490542],
                            [0.6116915921431676, 1.50592546727413201]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_expoential_type_error(self):
        assert_raises(TypeError, random.standard_exponential, dtype=np.int32)

    def test_standard_gamma(self):
        random.bit_generator.seed(self.seed)
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[2.28483515569645,  3.29899524967824],
                            [11.12492298902645,  2.16784417297277],
                            [0.92121813690910,  1.12853552328470]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_gammma_scalar_float(self):
        random.bit_generator.seed(self.seed)
        actual = random.standard_gamma(3, dtype=np.float32)
        desired = 1.3877466
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_standard_gamma_float(self):
        random.bit_generator.seed(self.seed)
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[2.2848352, 3.2989952],
                            [11.124923, 2.1678442],
                            [0.9212181, 1.1285355]])
        assert_array_almost_equal(actual, desired, decimal=5)

    def test_standard_gammma_float_out(self):
        actual = np.zeros((3, 2), dtype=np.float32)
        random.bit_generator.seed(self.seed)
        random.standard_gamma(10.0, out=actual, dtype=np.float32)
        desired = np.array([[6.9824033, 7.3731737],
                            [14.860578, 7.5327270],
                            [11.767487, 6.2320185]], dtype=np.float32)
        assert_array_almost_equal(actual, desired, decimal=5)

        random.bit_generator.seed(self.seed)
        random.standard_gamma(10.0, out=actual, size=(3, 2), dtype=np.float32)
        assert_array_almost_equal(actual, desired, decimal=5)

    def test_standard_gamma_unknown_type(self):
        assert_raises(TypeError, random.standard_gamma, 1.,
                      dtype='int32')

    def test_out_size_mismatch(self):
        out = np.zeros(10)
        assert_raises(ValueError, random.standard_gamma, 10.0, size=20,
                      out=out)
        assert_raises(ValueError, random.standard_gamma, 10.0, size=(10, 1),
                      out=out)

    def test_standard_gamma_0(self):
        assert_equal(random.standard_gamma(shape=0), 0)
        assert_raises(ValueError, random.standard_gamma, shape=-0.)

    def test_standard_normal(self):
        random.bit_generator.seed(self.seed)
        actual = random.standard_normal(size=(3, 2))
        desired = np.array([[-3.472754000610961, -0.108938564229143],
                            [-0.245965753396411, -0.704101550261701],
                            [0.360102487116356, 0.127832101772367]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_normal_unsupported_type(self):
        assert_raises(TypeError, random.standard_normal, dtype=np.int32)

    def test_standard_t(self):
        random.bit_generator.seed(self.seed)
        actual = random.standard_t(df=10, size=(3, 2))
        desired = np.array([[-3.68722108185508, -0.672031186266171],
                            [2.900224996448669, -0.199656996187739],
                            [-1.12179956985969, 1.85668262342106]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        random.bit_generator.seed(self.seed)
        actual = random.triangular(left=5.12, mode=10.23, right=20.34,
                                   size=(3, 2))
        desired = np.array([[12.68117178949215784, 12.4129206149193152],
                            [16.20131377335158263, 16.25692138747600524],
                            [11.20400690911820263, 14.4978144835829923]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        random.bit_generator.seed(self.seed)
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[6.99097932346268003, 6.73801597444323974],
                            [9.50364421400426274, 9.53130618907631089],
                            [5.48995325769805476, 8.47493103280052118]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_uniform_range_bounds(self):
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        func = random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # (fmax / 1e17) - fmin is within range, so this should not throw
        # account for i386 extended precision DBL_MAX / 1e17 + DBL_MAX >
        # DBL_MAX by increasing fmin a bit
        random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    def test_scalar_exception_propagation(self):
        # Tests that exceptions are correctly propagated in distributions
        # when called with objects that throw exceptions when converted to
        # scalars.
        #
        # Regression test for gh: 8865

        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, random.uniform, throwing_float,
                      throwing_float)

        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        random.bit_generator.seed(self.seed)
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[2.28567572673902042, 2.89163838442285037],
                            [0.38198375564286025, 2.57638023113890746],
                            [1.19153771588353052, 1.83509849681825354]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        # check infinite loop, gh-4720
        random.bit_generator.seed(self.seed)
        r = random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        assert_(np.isfinite(r).all())

    def test_vonmises_nan(self):
        random.bit_generator.seed(self.seed)
        r = random.vonmises(mu=0., kappa=np.nan)
        assert_(np.isnan(r))

    def test_wald(self):
        random.bit_generator.seed(self.seed)
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[0.10653278160339, 0.98771068102461],
                            [0.89276055317879, 0.13640126419923],
                            [0.9194319091599, 0.36037816317472]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        random.bit_generator.seed(self.seed)
        actual = random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[3.557276979846361, 1.020870580998542],
                            [2.731847777612348, 1.29148068905082],
                            [0.385531483942839, 2.049551716717254]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_weibull_0(self):
        random.bit_generator.seed(self.seed)
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        assert_raises(ValueError, random.weibull, a=-0.)

    def test_zipf(self):
        random.bit_generator.seed(self.seed)
        actual = random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[66, 29],
                            [1, 1],
                            [3, 13]])
        assert_array_equal(actual, desired)


class TestBroadcast(object):
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments
    def setup(self):
        self.seed = 123456789

    def set_seed(self):
        random.bit_generator.seed(self.seed)

    def test_uniform(self):
        low = [0]
        high = [1]
        uniform = random.uniform
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.set_seed()
        actual = uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        self.set_seed()
        actual = uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        normal = random.normal
        desired = np.array([0.454879818179180,
                            -0.62749179463661,
                            -0.06063266769872])

        self.set_seed()
        actual = normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc * 3, bad_scale)
        assert_raises(ValueError, random.normal, loc * 3, bad_scale)

        self.set_seed()
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)
        assert_raises(ValueError, random.normal, loc, bad_scale * 3)

    def test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        beta = random.beta
        desired = np.array([0.63222080311226,
                            0.33310522220774,
                            0.64494078460190])

        self.set_seed()
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        self.set_seed()
        actual = beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        exponential = random.exponential
        desired = np.array([1.68591211640990,
                            3.14186859487914,
                            0.67717375919228])

        self.set_seed()
        actual = exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, exponential, bad_scale * 3)

    def test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        std_gamma = random.standard_gamma
        desired = np.array([1.68591211640990,
                            3.14186859487914,
                            0.67717375919228])

        self.set_seed()
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        gamma = random.gamma
        desired = np.array([3.37182423281980,
                            6.28373718975827,
                            1.35434751838456])

        self.set_seed()
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        self.set_seed()
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        f = random.f
        desired = np.array([0.84207044881810,
                            3.08607209903483,
                            3.12823105933169])

        self.set_seed()
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        self.set_seed()
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)

    def test_noncentral_f(self):
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        nonc_f = random.noncentral_f
        desired = np.array([3.83710578542563,
                            8.74926819712029,
                            0.48892943835401])

        self.set_seed()
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))

        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        self.set_seed()
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        self.set_seed()
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def test_noncentral_f_small_df(self):
        self.set_seed()
        desired = np.array([21.57878070681719,  1.17110217503908])
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_chisquare(self):
        df = [1]
        bad_df = [-1]
        chisquare = random.chisquare
        desired = np.array([0.57022801133088286,
                            0.51947702108840776,
                            0.1320969254923558])

        self.set_seed()
        actual = chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        nonc_chi = random.noncentral_chisquare
        desired = np.array([2.20478739452297,
                            1.45177405755115,
                            1.00418921695354])

        self.set_seed()
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        self.set_seed()
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def test_standard_t(self):
        df = [1]
        bad_df = [-1]
        t = random.standard_t
        desired = np.array([0.60081050724244,
                            -0.90380889829210,
                            -0.64499590504117])

        self.set_seed()
        actual = t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, t, bad_df * 3)
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        vonmises = random.vonmises
        desired = np.array([2.9883443664201312,
                            -2.7064099483995943,
                            -1.8672476700665914])

        self.set_seed()
        actual = vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu * 3, bad_kappa)

        self.set_seed()
        actual = vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        a = [1]
        bad_a = [-1]
        pareto = random.pareto
        desired = np.array([4.397371719158540,
                            22.14707898642946,
                            0.968306954322200])

        self.set_seed()
        actual = pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, pareto, bad_a * 3)
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def test_weibull(self):
        a = [1]
        bad_a = [-1]
        weibull = random.weibull
        desired = np.array([1.68591211640990,
                            3.14186859487914,
                            0.67717375919228])

        self.set_seed()
        actual = weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, weibull, bad_a * 3)
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def test_power(self):
        a = [1]
        bad_a = [-1]
        power = random.power
        desired = np.array([0.81472463783615,
                            0.95679800459547,
                            0.49194916077287])

        self.set_seed()
        actual = power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, power, bad_a * 3)
        assert_raises(ValueError, random.power, bad_a * 3)

    def test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        laplace = random.laplace
        desired = np.array([0.067921356028507157,
                            0.070715642226971326,
                            0.019290950698972624])

        self.set_seed()
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        self.set_seed()
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        gumbel = random.gumbel
        desired = np.array([0.2730318639556768,
                            0.26936705726291116,
                            0.33906220393037939])

        self.set_seed()
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        self.set_seed()
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        logistic = random.logistic
        desired = np.array([0.13152135837586171,
                            0.13675915696285773,
                            0.038216792802833396])

        self.set_seed()
        actual = logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc * 3, bad_scale)

        self.set_seed()
        actual = logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc, bad_scale * 3)
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        lognormal = random.lognormal
        desired = np.array([1.57598396702930,
                            0.53392932731280,
                            0.94116889802361])

        self.set_seed()
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)
        assert_raises(ValueError, random.lognormal, mean * 3, bad_sigma)

        self.set_seed()
        actual = lognormal(mean, sigma * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        rayleigh = random.rayleigh
        desired = np.array([1.2337491937897689,
                            1.2360119924878694,
                            1.1936818095781789])

        self.set_seed()
        actual = rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, rayleigh, bad_scale * 3)

    def test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        wald = random.wald
        desired = np.array([0.36297361471752,
                            0.52190135028254,
                            0.55111022040727])

        self.set_seed()
        actual = wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        assert_raises(ValueError, wald, mean * 3, bad_scale)
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)

        self.set_seed()
        actual = wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        assert_raises(ValueError, random.wald, bad_mean, scale * 3)
        assert_raises(ValueError, random.wald, mean, bad_scale * 3)

    def test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        triangular = random.triangular
        desired = np.array([2.03339048710429,
                            2.0347400359389356,
                            2.0095991069536208])

        self.set_seed()
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        self.set_seed()
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        self.set_seed()
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)

        assert_raises(ValueError, triangular, 10., 0., 20.)
        assert_raises(ValueError, triangular, 10., 25., 20.)
        assert_raises(ValueError, triangular, 10., 10., 10.)

    def test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        binom = random.binomial
        desired = np.array([1, 1, 1])

        self.set_seed()
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        self.set_seed()
        actual = binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)

    def test_negative_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        neg_binom = random.negative_binomial
        desired = np.array([3, 1, 2], dtype=np.int64)

        self.set_seed()
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        self.set_seed()
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def test_poisson(self):
        max_lam = random._poisson_lam_max

        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        desired = np.array([1, 1, 0])

        self.set_seed()
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def test_zipf(self):
        a = [2]
        bad_a = [0]
        zipf = random.zipf
        desired = np.array([2, 2, 1])

        self.set_seed()
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    def test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        geom = random.geometric
        desired = np.array([2, 2, 2])

        self.set_seed()
        actual = geom(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geom, bad_p_one * 3)
        assert_raises(ValueError, geom, bad_p_two * 3)

    def test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [-1]
        bad_nsample_two = [4]
        hypergeom = random.hypergeometric
        desired = np.array([1, 1, 1])

        self.set_seed()
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)

        self.set_seed()
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)

        self.set_seed()
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, -1)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

    def test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        logseries = random.logseries
        desired = np.array([1, 1, 1])

        self.set_seed()
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)

    def test_multinomial(self):
        random.bit_generator.seed(self.seed)
        actual = random.multinomial([5, 20], [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[1, 1, 1, 1, 0, 1],
                             [4, 5, 1, 4, 3, 3]],
                            [[1, 1, 1, 0, 0, 2],
                             [2, 0, 4, 3, 7, 4]],
                            [[1, 2, 0, 0, 2, 0],
                             [3, 2, 3, 4, 2, 6]]], dtype=np.int64)
        assert_array_equal(actual, desired)

        random.bit_generator.seed(self.seed)
        actual = random.multinomial([5, 20], [1 / 6.] * 6)
        desired = np.array([[1, 1, 1, 1, 0, 1],
                            [4, 5, 1, 4, 3, 3]], dtype=np.int64)
        assert_array_equal(actual, desired)


class TestThread(object):
    # make sure each state produces the same sequence even in threads
    def setup(self):
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # threaded generation
        t = [Thread(target=function, args=(Generator(MT19937(s)), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(Generator(MT19937(s)), o)

        # these platforms change x87 fpu precision mode in threads
        if np.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)

        self.check_function(gen_random, sz=(10000,))

    def test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))

        self.check_function(gen_random, sz=(100, 1000))

    def test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1 / 6.] * 6, size=10000)

        self.check_function(gen_random, sz=(10000, 6))


# See Issue #4263
class TestSingleEltArrayInput(object):
    def setup(self):
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        funcs = (random.exponential, random.standard_gamma,
                 random.chisquare, random.standard_t,
                 random.pareto, random.weibull,
                 random.power, random.rayleigh,
                 random.poisson, random.zipf,
                 random.geometric, random.logseries)

        probfuncs = (random.geometric, random.logseries)

        for func in funcs:
            if func in probfuncs:  # p < 1.0
                out = func(np.array([0.5]))

            else:
                out = func(self.argOne)

            assert_equal(out.shape, self.tgtShape)

    def test_two_arg_funcs(self):
        funcs = (random.uniform, random.normal,
                 random.beta, random.gamma,
                 random.f, random.noncentral_chisquare,
                 random.vonmises, random.laplace,
                 random.gumbel, random.logistic,
                 random.lognormal, random.wald,
                 random.binomial, random.negative_binomial)

        probfuncs = (random.binomial, random.negative_binomial)

        for func in funcs:
            if func in probfuncs:  # p <= 1
                argTwo = np.array([0.5])

            else:
                argTwo = self.argTwo

            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def test_integers(self, endpoint):
        itype = [np.bool, np.int8, np.uint8, np.int16, np.uint16,
                 np.int32, np.uint32, np.int64, np.uint64]
        func = random.integers
        high = np.array([1])
        low = np.array([0])

        for dt in itype:
            out = func(low, high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low[0], high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low, high[0], endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

    def test_three_arg_funcs(self):
        funcs = [random.noncentral_f, random.triangular,
                 random.hypergeometric]

        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)
