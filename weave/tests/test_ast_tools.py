import unittest
from Numeric import *
# The following try/except so that non-SciPy users can still use blitz
try:
    from scipy_base.fastumath import *
except:
    pass # scipy_base.fastumath not available    
import RandomArray
import time

from scipy_test.testing import *
set_package_path()
from weave import ast_tools
restore_path()

set_local_path()
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

if __name__ == "__main__":
    ScipyTest('weave.ast_tools').run()
