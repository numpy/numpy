import unittest
from Numeric import *
from fastumath import *
import RandomArray
import time

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path
from scipy_distutils.misc_util import add_local_to_path

add_grandparent_to_path(__name__)
import ast_tools
restore_path()

add_local_to_path(__name__)
from weave_test_utils import *
restore_path()

class test_harvest_variables(unittest.TestCase):
    """ Not much testing going on here, but 
        at least it is a flame test.
    """    
    def generic_test(self,expr,desired):
        import parser
        ast_list = parser.suite(expr).tolist()
        actual = ast_tools.harvest_variables(ast_list)
        print_assert_equal(expr,actual,desired)

    def check_simple_expr(self):
        """convert simple expr to blitz
           
           a[:1:2] = b[:1+i+2:]
        """
        expr = "a[:1:2] = b[:1+i+2:]"        
        desired = ['a','b','i']        
        self.generic_test(expr,desired)


def test_suite():
    suites = []
    suites.append( unittest.makeSuite(test_harvest_variables,'check_') )
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test():
    all_tests = test_suite()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
