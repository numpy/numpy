import os
import pickle
import sys
import time

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_, \
    assert_array_equal

from randomgen._testing import suppress_warnings
from randomgen import RandomGenerator, MT19937, DSFMT, ThreeFry32, ThreeFry, \
    PCG32, PCG64, Philox, Xoroshiro128, Xorshift1024
from randomgen import entropy


@pytest.fixture(scope='module',
                params=(np.bool, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64))
def dtype(request):
    return request.param


def params_0(f):
    val = f()
    assert_(np.isscalar(val))
    val = f(10)
    assert_(val.shape == (10,))
    val = f((10, 10))
    assert_(val.shape == (10, 10))
    val = f((10, 10, 10))
    assert_(val.shape == (10, 10, 10))
    val = f(size=(5, 5))
    assert_(val.shape == (5, 5))


def params_1(f, bounded=False):
    a = 5.0
    b = np.arange(2.0, 12.0)
    c = np.arange(2.0, 102.0).reshape(10, 10)
    d = np.arange(2.0, 1002.0).reshape(10, 10, 10)
    e = np.array([2.0, 3.0])
    g = np.arange(2.0, 12.0).reshape(1, 10, 1)
    if bounded:
        a = 0.5
        b = b / (1.5 * b.max())
        c = c / (1.5 * c.max())
        d = d / (1.5 * d.max())
        e = e / (1.5 * e.max())
        g = g / (1.5 * g.max())

    # Scalar
    f(a)
    # Scalar - size
    f(a, size=(10, 10))
    # 1d
    f(b)
    # 2d
    f(c)
    # 3d
    f(d)
    # 1d size
    f(b, size=10)
    # 2d - size - broadcast
    f(e, size=(10, 2))
    # 3d - size
    f(g, size=(10, 10, 10))


def comp_state(state1, state2):
    identical = True
    if isinstance(state1, dict):
        for key in state1:
            identical &= comp_state(state1[key], state2[key])
    elif type(state1) != type(state2):
        identical &= type(state1) == type(state2)
    else:
        if (isinstance(state1, (list, tuple, np.ndarray)) and isinstance(
                state2, (list, tuple, np.ndarray))):
            for s1, s2 in zip(state1, state2):
                identical &= comp_state(s1, s2)
        else:
            identical &= state1 == state2
    return identical


def warmup(rg, n=None):
    if n is None:
        n = 11 + np.random.randint(0, 20)
    rg.standard_normal(n)
    rg.standard_normal(n)
    rg.standard_normal(n, dtype=np.float32)
    rg.standard_normal(n, dtype=np.float32)
    rg.randint(0, 2 ** 24, n, dtype=np.uint64)
    rg.randint(0, 2 ** 48, n, dtype=np.uint64)
    rg.standard_gamma(11.0, n)
    rg.standard_gamma(11.0, n, dtype=np.float32)
    rg.random_sample(n, dtype=np.float64)
    rg.random_sample(n, dtype=np.float32)


class RNG(object):
    @classmethod
    def _extra_setup(cls):
        cls.vec_1d = np.arange(2.0, 102.0)
        cls.vec_2d = np.arange(2.0, 102.0)[None, :]
        cls.mat = np.arange(2.0, 102.0, 0.01).reshape((100, 100))
        cls.seed_error = TypeError

    def _reset_state(self):
        self.rg.state = self.initial_state

    def test_init(self):
        rg = RandomGenerator(self.brng())
        state = rg.state
        rg.standard_normal(1)
        rg.standard_normal(1)
        rg.state = state
        new_state = rg.state
        assert_(comp_state(state, new_state))

    def test_advance(self):
        state = self.rg.state
        if hasattr(self.rg, 'advance'):
            self.rg.advance(self.advance)
            assert_(not comp_state(state, self.rg.state))
        else:
            pytest.skip()

    def test_jump(self):
        state = self.rg.state
        if hasattr(self.rg, 'jump'):
            self.rg.jump()
            jumped_state = self.rg.state
            assert_(not comp_state(state, jumped_state))
            self.rg.random_sample(2 * 3 * 5 * 7 * 11 * 13 * 17)
            self.rg.state = state
            self.rg.jump()
            rejumped_state = self.rg.state
            assert_(comp_state(jumped_state, rejumped_state))
        else:
            pytest.skip()

    def test_random_uintegers(self):
        assert_(len(self.rg.random_uintegers(10)) == 10)

    def test_random_raw(self):
        assert_(len(self.rg.random_raw(10)) == 10)
        assert_(self.rg.random_raw((10, 10)).shape == (10, 10))

    def test_uniform(self):
        r = self.rg.uniform(-1.0, 0.0, size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())

    def test_uniform_array(self):
        r = self.rg.uniform(np.array([-1.0] * 10), 0.0, size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())
        r = self.rg.uniform(np.array([-1.0] * 10),
                            np.array([0.0] * 10), size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())
        r = self.rg.uniform(-1.0, np.array([0.0] * 10), size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())

    def test_random_sample(self):
        assert_(len(self.rg.random_sample(10)) == 10)
        params_0(self.rg.random_sample)

    def test_standard_normal_zig(self):
        assert_(len(self.rg.standard_normal(10)) == 10)

    def test_standard_normal(self):
        assert_(len(self.rg.standard_normal(10)) == 10)
        params_0(self.rg.standard_normal)

    def test_standard_gamma(self):
        assert_(len(self.rg.standard_gamma(10, 10)) == 10)
        assert_(len(self.rg.standard_gamma(np.array([10] * 10), 10)) == 10)
        params_1(self.rg.standard_gamma)

    def test_standard_exponential(self):
        assert_(len(self.rg.standard_exponential(10)) == 10)
        params_0(self.rg.standard_exponential)

    def test_standard_cauchy(self):
        assert_(len(self.rg.standard_cauchy(10)) == 10)
        params_0(self.rg.standard_cauchy)

    def test_standard_t(self):
        assert_(len(self.rg.standard_t(10, 10)) == 10)
        params_1(self.rg.standard_t)

    def test_binomial(self):
        assert_(self.rg.binomial(10, .5) >= 0)
        assert_(self.rg.binomial(1000, .5) >= 0)

    def test_reset_state(self):
        state = self.rg.state
        int_1 = self.rg.random_raw(1)
        self.rg.state = state
        int_2 = self.rg.random_raw(1)
        assert_(int_1 == int_2)

    def test_entropy_init(self):
        rg = RandomGenerator(self.brng())
        rg2 = RandomGenerator(self.brng())
        assert_(not comp_state(rg.state, rg2.state))

    def test_seed(self):
        rg = RandomGenerator(self.brng(*self.seed))
        rg2 = RandomGenerator(self.brng(*self.seed))
        rg.random_sample()
        rg2.random_sample()
        if not comp_state(rg.state, rg2.state):
            for key in rg.state:
                print(key)
                print(rg.state[key])
                print(rg2.state[key])
        assert_(comp_state(rg.state, rg2.state))

    def test_reset_state_gauss(self):
        rg = RandomGenerator(self.brng(*self.seed))
        rg.standard_normal()
        state = rg.state
        n1 = rg.standard_normal(size=10)
        rg2 = RandomGenerator(self.brng())
        rg2.state = state
        n2 = rg2.standard_normal(size=10)
        assert_array_equal(n1, n2)

    def test_reset_state_uint32(self):
        rg = RandomGenerator(self.brng(*self.seed))
        rg.randint(0, 2 ** 24, 120, dtype=np.uint32)
        state = rg.state
        n1 = rg.randint(0, 2 ** 24, 10, dtype=np.uint32)
        rg2 = RandomGenerator(self.brng())
        rg2.state = state
        n2 = rg2.randint(0, 2 ** 24, 10, dtype=np.uint32)
        assert_array_equal(n1, n2)

    def test_reset_state_uintegers(self):
        rg = RandomGenerator(self.brng(*self.seed))
        rg.random_uintegers(bits=32)
        state = rg.state
        n1 = rg.random_uintegers(bits=32, size=10)
        rg2 = RandomGenerator(self.brng())
        rg2.state = state
        n2 = rg2.random_uintegers(bits=32, size=10)
        assert_((n1 == n2).all())

    def test_shuffle(self):
        original = np.arange(200, 0, -1)
        permuted = self.rg.permutation(original)
        assert_((original != permuted).any())

    def test_permutation(self):
        original = np.arange(200, 0, -1)
        permuted = self.rg.permutation(original)
        assert_((original != permuted).any())

    def test_tomaxint(self):
        vals = self.rg.tomaxint(size=100000)
        maxsize = 0
        if os.name == 'nt':
            maxsize = 2 ** 31 - 1
        else:
            try:
                maxsize = sys.maxint
            except AttributeError:
                maxsize = sys.maxsize
        if maxsize < 2 ** 32:
            assert_((vals < sys.maxsize).all())
        else:
            assert_((vals >= 2 ** 32).any())

    def test_beta(self):
        vals = self.rg.beta(2.0, 2.0, 10)
        assert_(len(vals) == 10)
        vals = self.rg.beta(np.array([2.0] * 10), 2.0)
        assert_(len(vals) == 10)
        vals = self.rg.beta(2.0, np.array([2.0] * 10))
        assert_(len(vals) == 10)
        vals = self.rg.beta(np.array([2.0] * 10), np.array([2.0] * 10))
        assert_(len(vals) == 10)
        vals = self.rg.beta(np.array([2.0] * 10), np.array([[2.0]] * 10))
        assert_(vals.shape == (10, 10))

    def test_bytes(self):
        vals = self.rg.bytes(10)
        assert_(len(vals) == 10)

    def test_chisquare(self):
        vals = self.rg.chisquare(2.0, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.chisquare)

    def test_complex_normal(self):
        st = self.rg.state
        vals = self.rg.complex_normal(
            2.0 + 7.0j, 10.0, 5.0 - 5.0j, size=10)
        assert_(len(vals) == 10)

        self.rg.state = st
        vals2 = [self.rg.complex_normal(
            2.0 + 7.0j, 10.0, 5.0 - 5.0j) for _ in range(10)]
        np.testing.assert_allclose(vals, vals2)

        self.rg.state = st
        vals3 = self.rg.complex_normal(
            2.0 + 7.0j * np.ones(10), 10.0 * np.ones(1), 5.0 - 5.0j)
        np.testing.assert_allclose(vals, vals3)

        self.rg.state = st
        norms = self.rg.standard_normal(size=20)
        norms = np.reshape(norms, (10, 2))
        cov = 0.5 * (-5.0)
        v_real = 7.5
        v_imag = 2.5
        rho = cov / np.sqrt(v_real * v_imag)
        imag = 7 + np.sqrt(v_imag) * (rho *
                                      norms[:, 0] + np.sqrt(1 - rho ** 2) *
                                      norms[:, 1])
        real = 2 + np.sqrt(v_real) * norms[:, 0]
        vals4 = [re + im * (0 + 1.0j) for re, im in zip(real, imag)]

        np.testing.assert_allclose(vals4, vals)

    def test_complex_normal_bm(self):
        st = self.rg.state
        vals = self.rg.complex_normal(
            2.0 + 7.0j, 10.0, 5.0 - 5.0j, size=10)
        assert_(len(vals) == 10)

        self.rg.state = st
        vals2 = [self.rg.complex_normal(
            2.0 + 7.0j, 10.0, 5.0 - 5.0j) for _ in range(10)]
        np.testing.assert_allclose(vals, vals2)

        self.rg.state = st
        vals3 = self.rg.complex_normal(
            2.0 + 7.0j * np.ones(10), 10.0 * np.ones(1), 5.0 - 5.0j)
        np.testing.assert_allclose(vals, vals3)

    def test_complex_normal_zero_variance(self):
        st = self.rg.state
        c = self.rg.complex_normal(0, 1.0, 1.0)
        assert_almost_equal(c.imag, 0.0)
        self.rg.state = st
        n = self.rg.standard_normal()
        np.testing.assert_allclose(c, n, atol=1e-8)

        st = self.rg.state
        c = self.rg.complex_normal(0, 1.0, -1.0)
        assert_almost_equal(c.real, 0.0)
        self.rg.state = st
        self.rg.standard_normal()
        n = self.rg.standard_normal()
        assert_almost_equal(c.real, 0.0)
        np.testing.assert_allclose(c.imag, n, atol=1e-8)

    def test_exponential(self):
        vals = self.rg.exponential(2.0, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.exponential)

    def test_f(self):
        vals = self.rg.f(3, 1000, 10)
        assert_(len(vals) == 10)

    def test_gamma(self):
        vals = self.rg.gamma(3, 2, 10)
        assert_(len(vals) == 10)

    def test_geometric(self):
        vals = self.rg.geometric(0.5, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.exponential, bounded=True)

    def test_gumbel(self):
        vals = self.rg.gumbel(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_laplace(self):
        vals = self.rg.laplace(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_logitic(self):
        vals = self.rg.logistic(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_logseries(self):
        vals = self.rg.logseries(0.5, 10)
        assert_(len(vals) == 10)

    def test_negative_binomial(self):
        vals = self.rg.negative_binomial(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_rand(self):
        state = self.rg.state
        vals = self.rg.rand(10, 10, 10)
        self.rg.state = state
        assert_((vals == self.rg.random_sample((10, 10, 10))).all())
        assert_(vals.shape == (10, 10, 10))
        vals = self.rg.rand(10, 10, 10, dtype=np.float32)
        assert_(vals.shape == (10, 10, 10))

    def test_randn(self):
        state = self.rg.state
        vals = self.rg.randn(10, 10, 10)
        self.rg.state = state
        assert_equal(vals, self.rg.standard_normal((10, 10, 10)))
        assert_equal(vals.shape, (10, 10, 10))

        state = self.rg.state
        vals = self.rg.randn(10, 10, 10)
        self.rg.state = state
        assert_equal(vals, self.rg.standard_normal((10, 10, 10)))

        state = self.rg.state
        self.rg.randn(10, 10, 10)
        self.rg.state = state
        vals = self.rg.randn(10, 10, 10, dtype=np.float32)
        assert_(vals.shape == (10, 10, 10))

    def test_noncentral_chisquare(self):
        vals = self.rg.noncentral_chisquare(10, 2, 10)
        assert_(len(vals) == 10)

    def test_noncentral_f(self):
        vals = self.rg.noncentral_f(3, 1000, 2, 10)
        assert_(len(vals) == 10)
        vals = self.rg.noncentral_f(np.array([3] * 10), 1000, 2)
        assert_(len(vals) == 10)
        vals = self.rg.noncentral_f(3, np.array([1000] * 10), 2)
        assert_(len(vals) == 10)
        vals = self.rg.noncentral_f(3, 1000, np.array([2] * 10))
        assert_(len(vals) == 10)

    def test_normal(self):
        vals = self.rg.normal(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_pareto(self):
        vals = self.rg.pareto(3.0, 10)
        assert_(len(vals) == 10)

    def test_poisson(self):
        vals = self.rg.poisson(10, 10)
        assert_(len(vals) == 10)
        vals = self.rg.poisson(np.array([10] * 10))
        assert_(len(vals) == 10)
        params_1(self.rg.poisson)

    def test_power(self):
        vals = self.rg.power(0.2, 10)
        assert_(len(vals) == 10)

    def test_randint(self):
        vals = self.rg.randint(10, 20, 10)
        assert_(len(vals) == 10)

    def test_random_integers(self):
        with suppress_warnings() as sup:
            sup.record(DeprecationWarning)
            vals = self.rg.random_integers(10, 20, 10)
        assert_(len(vals) == 10)

    def test_rayleigh(self):
        vals = self.rg.rayleigh(0.2, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.rayleigh, bounded=True)

    def test_vonmises(self):
        vals = self.rg.vonmises(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_wald(self):
        vals = self.rg.wald(1.0, 1.0, 10)
        assert_(len(vals) == 10)

    def test_weibull(self):
        vals = self.rg.weibull(1.0, 10)
        assert_(len(vals) == 10)

    def test_zipf(self):
        vals = self.rg.zipf(10, 10)
        assert_(len(vals) == 10)
        vals = self.rg.zipf(self.vec_1d)
        assert_(len(vals) == 100)
        vals = self.rg.zipf(self.vec_2d)
        assert_(vals.shape == (1, 100))
        vals = self.rg.zipf(self.mat)
        assert_(vals.shape == (100, 100))

    def test_hypergeometric(self):
        vals = self.rg.hypergeometric(25, 25, 20)
        assert_(np.isscalar(vals))
        vals = self.rg.hypergeometric(np.array([25] * 10), 25, 20)
        assert_(vals.shape == (10,))

    def test_triangular(self):
        vals = self.rg.triangular(-5, 0, 5)
        assert_(np.isscalar(vals))
        vals = self.rg.triangular(-5, np.array([0] * 10), 5)
        assert_(vals.shape == (10,))

    def test_multivariate_normal(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # diagonal covariance
        x = self.rg.multivariate_normal(mean, cov, 5000)
        assert_(x.shape == (5000, 2))
        x_zig = self.rg.multivariate_normal(mean, cov, 5000)
        assert_(x.shape == (5000, 2))
        x_inv = self.rg.multivariate_normal(mean, cov, 5000)
        assert_(x.shape == (5000, 2))
        assert_((x_zig != x_inv).any())

    def test_multinomial(self):
        vals = self.rg.multinomial(100, [1.0 / 3, 2.0 / 3])
        assert_(vals.shape == (2,))
        vals = self.rg.multinomial(100, [1.0 / 3, 2.0 / 3], size=10)
        assert_(vals.shape == (10, 2))

    def test_dirichlet(self):
        s = self.rg.dirichlet((10, 5, 3), 20)
        assert_(s.shape == (20, 3))

    def test_pickle(self):
        pick = pickle.dumps(self.rg)
        unpick = pickle.loads(pick)
        assert_((type(self.rg) == type(unpick)))
        assert_(comp_state(self.rg.state, unpick.state))

        pick = pickle.dumps(self.rg)
        unpick = pickle.loads(pick)
        assert_((type(self.rg) == type(unpick)))
        assert_(comp_state(self.rg.state, unpick.state))

    def test_seed_array(self):
        if self.seed_vector_bits is None:
            pytest.skip()

        if self.seed_vector_bits == 32:
            dtype = np.uint32
        else:
            dtype = np.uint64
        seed = np.array([1], dtype=dtype)
        self.rg.seed(seed)
        state1 = self.rg.state
        self.rg.seed(1)
        state2 = self.rg.state
        assert_(comp_state(state1, state2))

        seed = np.arange(4, dtype=dtype)
        self.rg.seed(seed)
        state1 = self.rg.state
        self.rg.seed(seed[0])
        state2 = self.rg.state
        assert_(not comp_state(state1, state2))

        seed = np.arange(1500, dtype=dtype)
        self.rg.seed(seed)
        state1 = self.rg.state
        self.rg.seed(seed[0])
        state2 = self.rg.state
        assert_(not comp_state(state1, state2))

        seed = 2 ** np.mod(np.arange(1500, dtype=dtype),
                           self.seed_vector_bits - 1) + 1
        self.rg.seed(seed)
        state1 = self.rg.state
        self.rg.seed(seed[0])
        state2 = self.rg.state
        assert_(not comp_state(state1, state2))

    def test_seed_array_error(self):
        if self.seed_vector_bits == 32:
            out_of_bounds = 2 ** 32
        else:
            out_of_bounds = 2 ** 64

        seed = -1
        with pytest.raises(ValueError):
            self.rg.seed(seed)

        seed = np.array([-1], dtype=np.int32)
        with pytest.raises(ValueError):
            self.rg.seed(seed)

        seed = np.array([1, 2, 3, -5], dtype=np.int32)
        with pytest.raises(ValueError):
            self.rg.seed(seed)

        seed = np.array([1, 2, 3, out_of_bounds])
        with pytest.raises(ValueError):
            self.rg.seed(seed)

    def test_uniform_float(self):
        rg = RandomGenerator(self.brng(12345))
        warmup(rg)
        state = rg.state
        r1 = rg.random_sample(11, dtype=np.float32)
        rg2 = RandomGenerator(self.brng())
        warmup(rg2)
        rg2.state = state
        r2 = rg2.random_sample(11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.state, rg2.state))

    def test_gamma_floats(self):
        rg = RandomGenerator(self.brng())
        warmup(rg)
        state = rg.state
        r1 = rg.standard_gamma(4.0, 11, dtype=np.float32)
        rg2 = RandomGenerator(self.brng())
        warmup(rg2)
        rg2.state = state
        r2 = rg2.standard_gamma(4.0, 11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.state, rg2.state))

    def test_normal_floats(self):
        rg = RandomGenerator(self.brng())
        warmup(rg)
        state = rg.state
        r1 = rg.standard_normal(11, dtype=np.float32)
        rg2 = RandomGenerator(self.brng())
        warmup(rg2)
        rg2.state = state
        r2 = rg2.standard_normal(11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.state, rg2.state))

    def test_normal_zig_floats(self):
        rg = RandomGenerator(self.brng())
        warmup(rg)
        state = rg.state
        r1 = rg.standard_normal(11, dtype=np.float32)
        rg2 = RandomGenerator(self.brng())
        warmup(rg2)
        rg2.state = state
        r2 = rg2.standard_normal(11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.state, rg2.state))

    def test_output_fill(self):
        rg = self.rg
        state = rg.state
        size = (31, 7, 97)
        existing = np.empty(size)
        rg.state = state
        rg.standard_normal(out=existing)
        rg.state = state
        direct = rg.standard_normal(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rg.state = state
        rg.standard_normal(out=existing, dtype=np.float32)
        rg.state = state
        direct = rg.standard_normal(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_uniform(self):
        rg = self.rg
        state = rg.state
        size = (31, 7, 97)
        existing = np.empty(size)
        rg.state = state
        rg.random_sample(out=existing)
        rg.state = state
        direct = rg.random_sample(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rg.state = state
        rg.random_sample(out=existing, dtype=np.float32)
        rg.state = state
        direct = rg.random_sample(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_exponential(self):
        rg = self.rg
        state = rg.state
        size = (31, 7, 97)
        existing = np.empty(size)
        rg.state = state
        rg.standard_exponential(out=existing)
        rg.state = state
        direct = rg.standard_exponential(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rg.state = state
        rg.standard_exponential(out=existing, dtype=np.float32)
        rg.state = state
        direct = rg.standard_exponential(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_gamma(self):
        rg = self.rg
        state = rg.state
        size = (31, 7, 97)
        existing = np.zeros(size)
        rg.state = state
        rg.standard_gamma(1.0, out=existing)
        rg.state = state
        direct = rg.standard_gamma(1.0, size=size)
        assert_equal(direct, existing)

        existing = np.zeros(size, dtype=np.float32)
        rg.state = state
        rg.standard_gamma(1.0, out=existing, dtype=np.float32)
        rg.state = state
        direct = rg.standard_gamma(1.0, size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_gamma_broadcast(self):
        rg = self.rg
        state = rg.state
        size = (31, 7, 97)
        mu = np.arange(97.0) + 1.0
        existing = np.zeros(size)
        rg.state = state
        rg.standard_gamma(mu, out=existing)
        rg.state = state
        direct = rg.standard_gamma(mu, size=size)
        assert_equal(direct, existing)

        existing = np.zeros(size, dtype=np.float32)
        rg.state = state
        rg.standard_gamma(mu, out=existing, dtype=np.float32)
        rg.state = state
        direct = rg.standard_gamma(mu, size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_fill_error(self):
        rg = self.rg
        size = (31, 7, 97)
        existing = np.empty(size)
        with pytest.raises(TypeError):
            rg.standard_normal(out=existing, dtype=np.float32)
        with pytest.raises(ValueError):
            rg.standard_normal(out=existing[::3])
        existing = np.empty(size, dtype=np.float32)
        with pytest.raises(TypeError):
            rg.standard_normal(out=existing, dtype=np.float64)

        existing = np.zeros(size, dtype=np.float32)
        with pytest.raises(TypeError):
            rg.standard_gamma(1.0, out=existing, dtype=np.float64)
        with pytest.raises(ValueError):
            rg.standard_gamma(1.0, out=existing[::3], dtype=np.float32)
        existing = np.zeros(size, dtype=np.float64)
        with pytest.raises(TypeError):
            rg.standard_gamma(1.0, out=existing, dtype=np.float32)
        with pytest.raises(ValueError):
            rg.standard_gamma(1.0, out=existing[::3])

    def test_randint_broadcast(self, dtype):
        if dtype == np.bool:
            upper = 2
            lower = 0
        else:
            info = np.iinfo(dtype)
            upper = int(info.max) + 1
            lower = info.min
        self._reset_state()
        a = self.rg.randint(lower, [upper] * 10, dtype=dtype)
        self._reset_state()
        b = self.rg.randint([lower] * 10, upper, dtype=dtype)
        assert_equal(a, b)
        self._reset_state()
        c = self.rg.randint(lower, upper, size=10, dtype=dtype)
        assert_equal(a, c)
        self._reset_state()
        d = self.rg.randint(np.array(
            [lower] * 10), np.array([upper], dtype=np.object), size=10,
            dtype=dtype)
        assert_equal(a, d)
        self._reset_state()
        e = self.rg.randint(
            np.array([lower] * 10), np.array([upper] * 10), size=10,
            dtype=dtype)
        assert_equal(a, e)

        self._reset_state()
        a = self.rg.randint(0, upper, size=10, dtype=dtype)
        self._reset_state()
        b = self.rg.randint([upper] * 10, dtype=dtype)
        assert_equal(a, b)

    def test_randint_numpy(self, dtype):
        high = np.array([1])
        low = np.array([0])

        out = self.rg.randint(low, high, dtype=dtype)
        assert out.shape == (1,)

        out = self.rg.randint(low[0], high, dtype=dtype)
        assert out.shape == (1,)

        out = self.rg.randint(low, high[0], dtype=dtype)
        assert out.shape == (1,)

    def test_randint_broadcast_errors(self, dtype):
        if dtype == np.bool:
            upper = 2
            lower = 0
        else:
            info = np.iinfo(dtype)
            upper = int(info.max) + 1
            lower = info.min
        with pytest.raises(ValueError):
            self.rg.randint(lower, [upper + 1] * 10, dtype=dtype)
        with pytest.raises(ValueError):
            self.rg.randint(lower - 1, [upper] * 10, dtype=dtype)
        with pytest.raises(ValueError):
            self.rg.randint([lower - 1], [upper] * 10, dtype=dtype)
        with pytest.raises(ValueError):
            self.rg.randint([0], [0], dtype=dtype)


class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = MT19937
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = 32
        cls._extra_setup()
        cls.seed_error = ValueError

    def test_numpy_state(self):
        nprg = np.random.RandomState()
        nprg.standard_normal(99)
        state = nprg.get_state()
        self.rg.state = state
        state2 = self.rg.state
        assert_((state[1] == state2['state']['key']).all())
        assert_((state[2] == state2['state']['pos']))


class TestPCG64(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = PCG64
        cls.advance = 2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1,
                    2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = None
        cls._extra_setup()

    def test_seed_array_error(self):
        # GH #82 for error type changes
        if self.seed_vector_bits == 32:
            out_of_bounds = 2 ** 32
        else:
            out_of_bounds = 2 ** 64

        seed = -1
        with pytest.raises(ValueError):
            self.rg.seed(seed)

        error_type = ValueError if self.seed_vector_bits else TypeError
        seed = np.array([-1], dtype=np.int32)
        with pytest.raises(error_type):
            self.rg.seed(seed)

        seed = np.array([1, 2, 3, -5], dtype=np.int32)
        with pytest.raises(error_type):
            self.rg.seed(seed)

        seed = np.array([1, 2, 3, out_of_bounds])
        with pytest.raises(error_type):
            self.rg.seed(seed)


class TestPhilox(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = Philox
        cls.advance = None
        cls.seed = [12345]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = 64
        cls._extra_setup()


class TestThreeFry(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = ThreeFry
        cls.advance = None
        cls.seed = [12345]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = 64
        cls._extra_setup()


class TestXoroshiro128(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = Xoroshiro128
        cls.advance = None
        cls.seed = [12345]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = 64
        cls._extra_setup()


class TestXorshift1024(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = Xorshift1024
        cls.advance = None
        cls.seed = [12345]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = 64
        cls._extra_setup()


class TestDSFMT(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = DSFMT
        cls.advance = None
        cls.seed = [12345]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls._extra_setup()
        cls.seed_vector_bits = 32


class TestThreeFry32(RNG):
    @classmethod
    def setup_class(cls):
        cls.brng = ThreeFry32
        cls.advance = [2 ** 96 + 2 ** 16 + 2 ** 5 + 1]
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = 64
        cls._extra_setup()
        cls.seed_error = ValueError


class TestEntropy(object):
    def test_entropy(self):
        e1 = entropy.random_entropy()
        e2 = entropy.random_entropy()
        assert_((e1 != e2))
        e1 = entropy.random_entropy(10)
        e2 = entropy.random_entropy(10)
        assert_((e1 != e2).all())
        e1 = entropy.random_entropy(10, source='system')
        e2 = entropy.random_entropy(10, source='system')
        assert_((e1 != e2).all())

    def test_fallback(self):
        e1 = entropy.random_entropy(source='fallback')
        time.sleep(0.1)
        e2 = entropy.random_entropy(source='fallback')
        assert_((e1 != e2))


class TestPCG32(TestPCG64):
    @classmethod
    def setup_class(cls):
        cls.brng = PCG32
        cls.advance = 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1,
                    2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.initial_state = cls.rg.state
        cls.seed_vector_bits = None
        cls._extra_setup()
