""" compiler provides several tools:

        1. inline() -- a function for including C/C++ code within Python
        2. blitz()  -- a function for compiling Numeric expressions to C++
        3. ext_tools-- a module that helps construct C/C++ extension modules.
        4. accelerate -- a module that inline accelerates Python functions
"""

try:
    from blitz_tools import blitz
except ImportError:
    pass # Numeric wasn't available    
    
from inline_tools import inline
import ext_tools
from ext_tools import ext_module, ext_function
try:
    from accelerate_tools import accelerate
except:
    pass

#---- testing ----#

def test():
    import unittest
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
    return runner

def test_suite():
    import scipy_test
    import scipy.weave
    this_mod = scipy.weave
    return scipy_test.harvest_test_suites(this_mod)
