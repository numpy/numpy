from numpy.testing import *
from numpy import random
import numpy as np

class TestRegression(TestCase):

    def test_VonMises_range(self):
        """Make sure generated random variables are in [-pi, pi].

        Regression test for ticket #986.
        """
        for mu in np.linspace(-7., 7., 5):
            r = random.mtrand.vonmises(mu,1,50)
            assert np.all(r > -np.pi) and np.all(r <= np.pi)

    def test_hypergeometric_range(self) :
        """Test for ticket #921"""
        assert_(np.all(np.random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(np.random.hypergeometric(18, 3, 11, size=10) > 0))

    def test_logseries_convergence(self) :
        """Test for ticket #923"""
        N = 1000
        np.random.seed(0)
        rvsn = np.random.logseries(0.8, size=N)
        # these two frequency counts should be close to theoretical
        # numbers with this large sample
        # theoretical large N result is 0.49706795
        freq = np.sum(rvsn == 1) / float(N)
        msg = "Frequency was %f, should be > 0.45" % freq
        assert_(freq > 0.45, msg)
        # theoretical large N result is 0.19882718
        freq = np.sum(rvsn == 2) / float(N)
        msg = "Frequency was %f, should be < 0.23" % freq
        assert_(freq < 0.23, msg)


class TestMultinomial(TestCase):
    def test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert -5 <= random.randint(-5,-1) < -1
        x = random.randint(-5,-1,5)
        assert np.all(-5 <= x)
        assert np.all(x < -1)



class TestSetState(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.prng = random.RandomState(self.seed)
        self.state = self.prng.get_state()

    def test_basic(self):
        old = self.prng.tomaxint(16)
        self.prng.set_state(self.state)
        new = self.prng.tomaxint(16)
        assert np.all(old == new)

    def test_gaussian_reset(self):
        """ Make sure the cached every-other-Gaussian is reset.
        """
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(self.state)
        new = self.prng.standard_normal(size=3)
        assert np.all(old == new)

    def test_gaussian_reset_in_media_res(self):
        """ When the state is saved with a cached Gaussian, make sure the cached
        Gaussian is restored.
        """
        self.prng.standard_normal()
        state = self.prng.get_state()
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(state)
        new = self.prng.standard_normal(size=3)
        assert np.all(old == new)

    def test_backwards_compatibility(self):
        """ Make sure we can accept old state tuples that do not have the cached
        Gaussian value.
        """
        old_state = self.state[:-2]
        x1 = self.prng.standard_normal(size=16)
        self.prng.set_state(old_state)
        x2 = self.prng.standard_normal(size=16)
        self.prng.set_state(self.state)
        x3 = self.prng.standard_normal(size=16)
        assert np.all(x1 == x2)
        assert np.all(x1 == x3)

    def test_negative_binomial(self):
        """ Ensure that the negative binomial results take floating point
        arguments without truncation.
        """
        self.prng.negative_binomial(0.5, 0.5)


if __name__ == "__main__":
    run_module_suite()
