import unittest

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import inline_tools
restore_path()
    
class test_sequence_converter(unittest.TestCase):    
    def check_convert_to_dict(self):
        d = {}
        inline_tools.inline("",['d']) 
    def check_convert_to_list(self):        
        l = []
        inline_tools.inline("",['l']) 
    def check_convert_to_string(self):        
        s = 'hello'
        inline_tools.inline("",['s']) 
    def check_convert_to_tuple(self):        
        t = ()
        inline_tools.inline("",['t']) 
        
def test_suite():
    suites = []   
    suites.append( unittest.makeSuite(test_sequence_converter,'check_'))
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test():
    all_tests = test_suite()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
