#
# weave - C/C++ integration
#

from pre___init__ import __doc__
from weave_version import weave_version as __version__

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

def test(level=10):
    import unittest
    runner = unittest.TextTestRunner()
    runner.run(test_suite(level))
    return runner

def test_suite(level=1):
    import scipy_test.testing
    import weave
    return scipy_test.testing.harvest_test_suites(weave,level=level)
