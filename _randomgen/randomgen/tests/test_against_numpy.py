import numpy as np
import numpy.random
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

import randomgen
from randomgen import RandomGenerator, MT19937


def compare_0_input(f1, f2):
    inputs = [(tuple([]), {}), (tuple([]), {'size': 10}),
              (tuple([]), {'size': (20, 31)}),
              (tuple([]), {'size': (20, 31, 5)})]


def compare_1_input(f1, f2, is_small=False):
    a = 0.3 if is_small else 10
    inputs = [((a,), {}),
              ((a,), {'size': 10}),
              ((np.array([a] * 10),), {}),
              ((np.array([a] * 10),), {'size': 10}),
              ((np.array([a] * 10),), {'size': (100, 10)})]
    for i in inputs:
        v1 = f1(*i[0], **i[1])
        v2 = f2(*i[0], **i[1])
        assert_allclose(v1, v2)


def compare_2_input(f1, f2, is_np=False, is_scalar=False):
    if is_np:
        a, b = 10, 0.3
        dtype = np.int
    else:
        a, b = 2, 3
        dtype = np.double
    inputs = [((a, b), {}),
              ((a, b), {'size': 10}),
              ((a, b), {'size': (23, 7)}),
              ((np.array([a] * 10), b), {}),
              ((a, np.array([b] * 10)), {}),
              ((a, np.array([b] * 10)), {'size': 10}),
              ((np.reshape(np.array([[a] * 100]), (100, 1)),
                np.array([b] * 10)), {'size': (100, 10)}),
              ((np.ones((7, 31), dtype=dtype) * a,
                np.array([b] * 31)), {'size': (7, 31)}),
              ((np.ones((7, 31), dtype=dtype) * a, np.array([b] * 31)),
               {'size': (10, 7, 31)})]

    if is_scalar:
        inputs = inputs[:3]

    for i in inputs:
        v1 = f1(*i[0], **i[1])
        v2 = f2(*i[0], **i[1])
        assert_allclose(v1, v2)


def compare_3_input(f1, f2, is_np=False):
    a, b, c = 10, 20, 25
    inputs = [((a, b, c), {}),
              ((a, b, c), {'size': 10}),
              ((a, b, c), {'size': (23, 7)}),
              ((np.array([a] * 10), b, c), {}),
              ((a, np.array([b] * 10), c), {}),
              ((a, b, np.array([c] * 10)), {}),
              ((a, np.array([b] * 10), np.array([c] * 10)), {}),
              ((a, np.array([b] * 10), c), {'size': 10}),
              ((np.ones((1, 37), dtype=np.int) * a,
                np.ones((23, 1), dtype=np.int) * [b],
                c * np.ones((7, 1, 1), dtype=np.int)),
               {}),
              ((np.ones((1, 37), dtype=np.int) * a,
                np.ones((23, 1), dtype=np.int) * [b],
                c * np.ones((7, 1, 1), dtype=np.int)),
               {'size': (7, 23, 37)})
              ]

    for i in inputs:
        v1 = f1(*i[0], **i[1])
        v2 = f2(*i[0], **i[1])
        assert_allclose(v1, v2)


class TestAgainstNumPy(object):
    @classmethod
    def setup_class(cls):
        cls.np = numpy.random
        cls.brng = MT19937
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rg = RandomGenerator(cls.brng(*cls.seed))
        cls.nprs = cls.np.RandomState(*cls.seed)
        cls.initial_state = cls.rg.state
        cls._set_common_state()

    @classmethod
    def _set_common_state(cls):
        state = cls.rg.state
        st = [[]] * 5
        st[0] = 'MT19937'
        st[1] = state['state']['key']
        st[2] = state['state']['pos']
        st[3] = 0
        st[4] = 0.0
        cls.nprs.set_state(st)

    def _is_state_common(self):
        state = self.nprs.get_state()
        state2 = self.rg.state
        assert (state[1] == state2['state']['key']).all()
        assert (state[2] == state2['state']['pos'])

    def test_common_seed(self):
        self.rg.seed(1234)
        self.nprs.seed(1234)
        self._is_state_common()
        self.rg.seed(23456)
        self.nprs.seed(23456)
        self._is_state_common()

    def test_numpy_state(self):
        nprs = np.random.RandomState()
        nprs.standard_normal(99)
        state = nprs.get_state()
        self.rg.state = state
        state2 = self.rg.state
        assert (state[1] == state2['state']['key']).all()
        assert (state[2] == state2['state']['pos'])

    def test_random_sample(self):
        self._set_common_state()
        self._is_state_common()
        v1 = self.nprs.random_sample(10)
        v2 = self.rg.random_sample(10)

        assert_array_equal(v1, v2)

    def test_standard_normal(self):
        self._set_common_state()
        self._is_state_common()
        compare_0_input(self.nprs.standard_normal,
                        self.rg.standard_normal)
        self._is_state_common()

    def test_standard_cauchy(self):
        self._set_common_state()
        self._is_state_common()
        compare_0_input(self.nprs.standard_cauchy,
                        self.rg.standard_cauchy)
        self._is_state_common()

    def test_standard_exponential(self):
        self._set_common_state()
        self._is_state_common()
        compare_0_input(self.nprs.standard_exponential,
                        self.rg.standard_exponential)
        self._is_state_common()

    def test_tomaxint(self):
        self._set_common_state()
        self._is_state_common()
        compare_0_input(self.nprs.tomaxint,
                        self.rg.tomaxint)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_chisquare(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.chisquare,
                        self.rg.chisquare)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_standard_gamma(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.standard_gamma,
                        self.rg.standard_gamma)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_standard_t(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.standard_t,
                        self.rg.standard_t)
        self._is_state_common()

    def test_pareto(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.pareto,
                        self.rg.pareto)
        self._is_state_common()

    def test_poisson(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.poisson,
                        self.rg.poisson)
        self._is_state_common()

    def test_power(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.power,
                        self.rg.power)
        self._is_state_common()

    def test_rayleigh(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.rayleigh,
                        self.rg.rayleigh)
        self._is_state_common()

    def test_weibull(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.weibull,
                        self.rg.weibull)
        self._is_state_common()

    def test_zipf(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.zipf,
                        self.rg.zipf)
        self._is_state_common()

    def test_logseries(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.logseries,
                        self.rg.logseries,
                        is_small=True)
        self._is_state_common()

    def test_geometric(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.geometric,
                        self.rg.geometric,
                        is_small=True)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_beta(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.beta,
                        self.rg.beta)
        self._is_state_common()

    def test_exponential(self):
        self._set_common_state()
        self._is_state_common()
        compare_1_input(self.nprs.exponential,
                        self.rg.exponential)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_f(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.f,
                        self.rg.f)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_gamma(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.gamma,
                        self.rg.gamma)
        self._is_state_common()

    def test_logistic(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.logistic,
                        self.rg.logistic)
        self._is_state_common()

    def test_gumbel(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.gumbel,
                        self.rg.gumbel)
        self._is_state_common()

    def test_laplace(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.laplace,
                        self.rg.laplace)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_lognormal(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.lognormal,
                        self.rg.lognormal)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_noncentral_chisquare(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.noncentral_chisquare,
                        self.rg.noncentral_chisquare)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_normal(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.normal,
                        self.rg.normal)
        self._is_state_common()

    def test_uniform(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.uniform,
                        self.rg.uniform)
        self._is_state_common()

    def test_vonmises(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.vonmises,
                        self.rg.vonmises)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_wald(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.wald,
                        self.rg.wald)
        self._is_state_common()

    def test_random_integers(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.random_integers,
                        self.rg.random_integers,
                        is_scalar=True)
        self._is_state_common()

    def test_binomial(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.binomial,
                        self.rg.binomial,
                        is_np=True)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_negative_binomial(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.nprs.negative_binomial,
                        self.rg.negative_binomial,
                        is_np=True)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_randn(self):
        f = self.rg.randn
        g = self.nprs.randn
        assert_allclose(f(10), g(10))
        assert_allclose(f(3, 4, 5), g(3, 4, 5))

    def test_rand(self):
        self._set_common_state()
        self._is_state_common()
        f = self.rg.rand
        g = self.nprs.rand
        assert_allclose(f(10), g(10))
        assert_allclose(f(3, 4, 5), g(3, 4, 5))

    def test_poisson_lam_max(self):
        assert_allclose(self.rg.poisson_lam_max, self.nprs.poisson_lam_max)

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_dirichlet(self):
        f = self.rg.dirichlet
        g = self.nprs.dirichlet
        a = [3, 4, 5, 6, 7, 10]
        assert_allclose(f(a), g(a))
        assert_allclose(f(np.array(a), 10), g(np.array(a), 10))
        assert_allclose(f(np.array(a), (3, 37)), g(np.array(a), (3, 37)))

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_noncentral_f(self):
        self._set_common_state()
        self._is_state_common()
        compare_3_input(self.nprs.noncentral_f,
                        self.rg.noncentral_f)
        self._is_state_common()

    def test_triangular(self):
        self._set_common_state()
        self._is_state_common()
        compare_3_input(self.nprs.triangular,
                        self.rg.triangular)
        self._is_state_common()

    def test_hypergeometric(self):
        self._set_common_state()
        self._is_state_common()
        compare_3_input(self.nprs.hypergeometric,
                        self.rg.hypergeometric)
        self._is_state_common()

    def test_bytes(self):
        self._set_common_state()
        self._is_state_common()
        assert_equal(self.rg.bytes(8), self.nprs.bytes(8))
        self._is_state_common()
        assert_equal(self.rg.bytes(103), self.nprs.bytes(103))
        self._is_state_common()
        assert_equal(self.rg.bytes(8), self.nprs.bytes(8))
        self._is_state_common()

    def test_multinomial(self):
        self._set_common_state()
        self._is_state_common()
        f = self.rg.multinomial
        g = self.nprs.multinomial
        p = [.1, .3, .4, .2]
        assert_equal(f(100, p), g(100, p))
        assert_equal(f(100, np.array(p)), g(100, np.array(p)))
        assert_equal(f(100, np.array(p), size=(7, 23)),
                     g(100, np.array(p), size=(7, 23)))
        self._is_state_common()

    def test_choice(self):
        self._set_common_state()
        self._is_state_common()
        f = self.rg.choice
        g = self.nprs.choice
        a = np.arange(100)
        size = 25
        for replace in (True, False):
            assert_equal(f(a, size, replace), g(a, size, replace))
            assert_equal(f(100, size, replace), g(100, size, replace))
        self._is_state_common()

    def test_permutation(self):
        self._set_common_state()
        self._is_state_common()
        f = self.rg.permutation
        g = self.nprs.permutation
        a = np.arange(100)
        assert_equal(f(a), g(a))
        assert_equal(f(23), g(23))
        self._is_state_common()

    def test_shuffle(self):
        self._set_common_state()
        self._is_state_common()
        f = self.rg.shuffle
        g = self.nprs.shuffle
        a = np.arange(100)
        fa = a.copy()
        ga = a.copy()
        g(ga)
        f(fa)
        assert_equal(fa, ga)
        self._is_state_common()

    @pytest.mark.skip(reason='Box-Muller no longer supported')
    def test_multivariate_normal(self):
        self._set_common_state()
        self._is_state_common()
        mu = [1, 2, 3]
        cov = [[1, .2, .3], [.2, 4, 1], [.3, 1, 10]]
        f = self.rg.multivariate_normal
        g = self.nprs.multivariate_normal
        assert_allclose(f(mu, cov), g(mu, cov))
        assert_allclose(f(np.array(mu), cov), g(np.array(mu), cov))
        assert_allclose(f(np.array(mu), np.array(cov)),
                        g(np.array(mu), np.array(cov)))
        assert_allclose(f(np.array(mu), np.array(cov), size=(7, 31)),
                        g(np.array(mu), np.array(cov), size=(7, 31)))
        self._is_state_common()

    def test_randint(self):
        self._set_common_state()
        self._is_state_common()
        compare_2_input(self.rg.randint,
                        self.nprs.randint,
                        is_scalar=True)
        self._is_state_common()

    def test_scalar(self):
        s = RandomGenerator(MT19937(0))
        assert_equal(s.randint(1000), 684)
        s1 = np.random.RandomState(0)
        assert_equal(s1.randint(1000), 684)
        assert_equal(s1.randint(1000), s.randint(1000))

        s = RandomGenerator(MT19937(4294967295))
        assert_equal(s.randint(1000), 419)
        s1 = np.random.RandomState(4294967295)
        assert_equal(s1.randint(1000), 419)
        assert_equal(s1.randint(1000), s.randint(1000))

        self.rg.seed(4294967295)
        self.nprs.seed(4294967295)
        self._is_state_common()

    def test_array(self):
        s = RandomGenerator(MT19937(range(10)))
        assert_equal(s.randint(1000), 468)
        s = np.random.RandomState(range(10))
        assert_equal(s.randint(1000), 468)

        s = RandomGenerator(MT19937(np.arange(10)))
        assert_equal(s.randint(1000), 468)
        s = RandomGenerator(MT19937([0]))
        assert_equal(s.randint(1000), 973)
        s = RandomGenerator(MT19937([4294967295]))
        assert_equal(s.randint(1000), 265)

    def test_dir(self):
        nprs_d = set(dir(self.nprs))
        rs_d = dir(self.rg)
        excluded = {'get_state', 'set_state'}
        nprs_d.difference_update(excluded)
        assert (len(nprs_d.difference(rs_d)) == 0)

        npmod = dir(numpy.random)
        mod = dir(randomgen.generator)
        known_exlcuded = ['__all__', '__cached__', '__path__', 'Tester',
                          'info', 'bench', '__RandomState_ctor', 'mtrand',
                          'test', '__warningregistry__', '_numpy_tester',
                          'division', 'get_state', 'set_state', 'seed',
                          'ranf', 'random', 'sample', 'absolute_import',
                          'print_function', 'RandomState']
        mod += known_exlcuded
        diff = set(npmod).difference(mod)
        print(diff)
        assert_equal(len(diff), 0)
