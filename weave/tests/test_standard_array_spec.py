import unittest
from Numeric import *
# The following try/except so that non-SciPy users can still use blitz
try:
    from scipy_base.fastumath import *
except:
    pass # fastumath not available    
import RandomArray
import time

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import standard_array_spec
restore_path()

def remove_whitespace(in_str):
    import string
    out = string.replace(in_str," ","")
    out = string.replace(out,"\t","")
    out = string.replace(out,"\n","")
    return out
    
def print_assert_equal(test_string,actual,desired):
    """this should probably be in scipy_base.testing
    """
    import pprint
    try:
        assert(actual == desired)
    except AssertionError:
        import cStringIO
        msg = cStringIO.StringIO()
        msg.write(test_string)
        msg.write(' failed\nACTUAL: \n')
        pprint.pprint(actual,msg)
        msg.write('DESIRED: \n')
        pprint.pprint(desired,msg)
        raise AssertionError, msg.getvalue()

class test_array_converter(unittest.TestCase):    
    def check_type_match_string(self):
        s = standard_array_spec.array_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = standard_array_spec.array_converter()        
        assert(not s.type_match(5))
    def check_type_match_array(self):
        s = standard_array_spec.array_converter()        
        assert(s.type_match(arange(4)))

def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(test_array_converter,'check_'))

    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
