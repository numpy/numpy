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
from weave import standard_array_spec
restore_path()

def remove_whitespace(in_str):
    import string
    out = string.replace(in_str," ","")
    out = string.replace(out,"\t","")
    out = string.replace(out,"\n","")
    return out
    
def print_assert_equal(test_string,actual,desired):
    """this should probably be in scipy_test.testing
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

if __name__ == "__main__":
    ScipyTest('weave.standard_array_spec').run()

