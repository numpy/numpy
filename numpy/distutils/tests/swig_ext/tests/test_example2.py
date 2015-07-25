from __future__ import division, absolute_import, print_function

from numpy.testing import TestCase, run_module_suite
from swig_ext import example2

class TestExample2(TestCase):
    def test_zoo(self):
        z = example2.Zoo()
        z.shut_up('Tiger')
        z.shut_up('Lion')
        z.display()


if __name__ == "__main__":
    run_module_suite()
