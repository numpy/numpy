from numpy.testing import *
from numpy import random
import numpy as N

class TestMultinomial(NumpyTestCase):
    def test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert -5 <= random.randint(-5,-1) < -1
        x = random.randint(-5,-1,5)
        assert N.all(-5 <= x)
        assert N.all(x < -1)

if __name__ == "__main__":
    NumpyTest().run()
