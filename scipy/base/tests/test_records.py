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
        assert_equal(r[0].toscalar(),(456, 'dbe', 1.2))

    def check_method_array(ScipyTestCase):
        r = rec.array('abcdefg'*100,formats='i2,a3,i4',shape=3,byteorder='big')
        assert_equal(r[1].toscalar(),(25444, 'efg', 1633837924)) 

del sys.path[0]
if __name__ == "__main__":
    ScipyTest().run()
