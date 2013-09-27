from __future__ import division, absolute_import, print_function

from numpy.testing import *

rlevel = 1

class TestRegression(TestCase):
    def test_numeric_random(self, level=rlevel):
        """Ticket #552"""
        from numpy.oldnumeric.random_array import randint
        randint(0, 50, [2, 3])

    def test_mlab_import(self):
        """gh-3803"""
        try:
            from numpy.oldnumeric import mlab
            import numpy.oldnumeric.mlab as mlab
        except:
            raise AssertionError("mlab import failed")

