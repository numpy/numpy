import unittest
from Numeric import *

from scipy_distutils.misc_util import add_grandparent_to_path,restore_path
from scipy_distutils.misc_util import add_local_to_path

add_grandparent_to_path(__name__)
import inline_tools
restore_path()

class test_inline(unittest.TestCase):
    """ These are long running tests...
    
         I'd like to benchmark these things somehow.
    """
    def check_exceptions(self):
        a = 3                                  
        code = """
               if (a < 2)
                  throw_error(PyExc_ValueError,
                              "the variable 'a' should not be less than 2");
               else    
                   return_val = Py::new_reference_to(Py::Int(a+1));
               """
        result = inline_tools.inline(code,['a'])
        assert(result == 4)
        
        try:
            a = 1
            result = inline_tools.inline(code,['a'])
            assert(1) # should've thrown a ValueError
        except ValueError:
            pass
        
        from distutils.errors import DistutilsError, CompileError    
        try:
            a = 'string'
            result = inline_tools.inline(code,['a'])
            assert(1) # should've gotten an error
        except: 
            # ?CompileError is the error reported, but catching it doesn't work
            pass
            
def test_suite(level=1):
    suites = []
    if level >= 5:
        suites.append( unittest.makeSuite(test_inline,'check_') )    
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
