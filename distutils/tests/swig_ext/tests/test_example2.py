import sys
from scipy_test.testing import *
set_package_path()
from swig_ext import example2
del sys.path[0]

class test_example2(ScipyTestCase):

    def check_zoo(self):
        z = example2.Zoo()
        z.shut_up('Tiger')
        z.shut_up('Lion')
        z.display()


if __name__ == "__main__":
    ScipyTest('swig_ext.example2').run()
