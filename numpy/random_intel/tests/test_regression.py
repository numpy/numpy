from __future__ import division, absolute_import, print_function

import sys
from numpy.testing import (TestCase, run_module_suite, assert_,
                           assert_array_equal, assert_raises)
from numpy import random_intel as rnd
from numpy.compat import long
import numpy as np


class TestRegression_Intel(TestCase):

    def test_VonMises_range(self):
        # Make sure generated random variables are in [-pi, pi].
        # Regression test for ticket #986.
        for mu in np.linspace(-7., 7., 5):
            r = rnd.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        # Test for ticket #921
        assert_(np.all(rnd.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(rnd.hypergeometric(18, 3, 11, size=10) > 0))

        # Test for ticket #5623
        args = [
            (2**20 - 2, 2**20 - 2, 2**20 - 2),  # Check for 32-bit systems
            (2 ** 30 - 1, 2 ** 30 - 2, 2 ** 30 - 1)
        ]
        for arg in args:
            assert_(rnd.hypergeometric(*arg) > 0)

    def test_logseries_convergence(self):
        # Test for ticket #923
        N = 1000
        rnd.seed(0)
        rvsn = rnd.logseries(0.8, size=N)
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

    def test_permutation_longs(self):
        rnd.seed(1234)
        a = rnd.permutation(12)
        rnd.seed(1234)
        b = rnd.permutation(long(12))
        assert_array_equal(a, b)

    def test_randint_range(self):
        # Test for ticket #1690
        lmax = np.iinfo('l').max
        lmin = np.iinfo('l').min
        try:
            rnd.randint(lmin, lmax)
        except:
            raise AssertionError

    def test_shuffle_mixed_dimension(self):
        # Test for trac ticket #2074
        for t in [[1, 2, 3, None],
                  [(1, 1), (2, 2), (3, 3), None],
                  [1, (2, 2), (3, 3), None],
                  [(1, 1), 2, 3, None]]:
            rnd.seed(12345, brng='MT2203')
            shuffled = list(t)
            rnd.shuffle(shuffled)
            assert_array_equal(shuffled, [t[0], t[2], t[1], t[3]])

    def test_call_within_randomstate(self):
        # Check that custom RandomState does not call into global state
        m = rnd.RandomState()
        res = np.array([5, 7, 5, 4, 5, 5, 6, 9, 6, 1])
        for i in range(3):
            rnd.seed(i)
            m.seed(4321, brng='SFMT19937')
            # If m.state is not honored, the result will change
            assert_array_equal(m.choice(10, size=10, p=np.ones(10)/10.), res)

    def test_multivariate_normal_size_types(self):
        # Test for multivariate_normal issue with 'size' argument.
        # Check that the multivariate_normal size argument can be a
        # numpy integer.
        rnd.multivariate_normal([0], [[0]], size=1)
        rnd.multivariate_normal([0], [[0]], size=np.int_(1))
        rnd.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # Test that beta with small a and b parameters does not produce
        # NaNs due to roundoff errors causing 0 / 0, gh-5851
        rnd.seed(1234567890)
        x = rnd.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in rnd.beta')

    def test_choice_sum_of_probs_tolerance(self):
        # The sum of probs should be 1.0 with some tolerance.
        # For low precision dtypes the tolerance was too tight.
        # See numpy github issue 6123.
        rnd.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = rnd.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, rnd.choice, a, p=probs*0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # Test that permuting an array of different length strings
        # will not cause a segfault on garbage collection
        # Tests gh-7710
        rnd.seed(1234)

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            rnd.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()


    def test_shuffle_of_array_of_objects(self):
        # Test that permuting an array of objects will not cause
        # a segfault on garbage collection.
        # See gh-7719
        rnd.seed(1234)
        a = np.array([np.arange(1), np.arange(4)])

        for _ in range(1000):
            rnd.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()


if __name__ == "__main__":
    run_module_suite()
