from __future__ import division, absolute_import, print_function

from numpy.testing import *
from numpy import logspace, linspace

class TestLogspace(TestCase):
    def test_basic(self):
        y = logspace(0, 6)
        assert_(len(y)==50)
        y = logspace(0, 6, num=100)
        assert_(y[-1] == 10**6)
        y = logspace(0, 6, endpoint=0)
        assert_(y[-1] < 10**6)
        y = logspace(0, 6, num=7)
        assert_array_equal(y, [1, 10, 100, 1e3, 1e4, 1e5, 1e6])

class TestLinspace(TestCase):
    def test_basic(self):
        y = linspace(0, 10)
        assert_(len(y)==50)
        y = linspace(2, 10, num=100)
        assert_(y[-1] == 10)
        y = linspace(2, 10, endpoint=0)
        assert_(y[-1] < 10)

    def test_corner(self):
        y = list(linspace(0, 1, 1))
        assert_(y == [0.0], y)
        y = list(linspace(0, 1, 2.5))
        assert_(y == [0.0, 1.0])

    def test_type(self):
        t1 = linspace(0, 1, 0).dtype
        t2 = linspace(0, 1, 1).dtype
        t3 = linspace(0, 1, 2).dtype
        assert_equal(t1, t2)
        assert_equal(t2, t3)
