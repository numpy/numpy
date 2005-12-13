import unittest
import sys

from scipy.test.testing import *
set_package_path()
import scipy.base;reload(scipy.base)
from scipy.base import *
from scipy.base import records as rec

class test_fromrecords(ScipyTestCase):
    def check_fromrecords(self):
        r = rec.fromrecords([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
        assert_equal(r[0],(456, 'dbe', 1.2))

del sys.path[0]
if __name__ == "__main__":
    ScipyTest().run()
