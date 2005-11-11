""" Test functions for limits module.

    Currently empty -- not sure how to test these values
    and routines as they are machine dependent.  Suggestions?
"""

import unittest

import sys
from scipy_test.testing import *
set_package_path()
import scipy_base;reload(scipy_base)
from scipy_base import *
del sys.path[0]

##################################################
### Test for sum

class test_float(unittest.TestCase):
    def check_nothing(self):
        pass

class test_double(unittest.TestCase):
    def check_nothing(self):
        pass

if __name__ == "__main__":
    ScipyTest('scipy_base.limits').run()

