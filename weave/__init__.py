#
# weave - C/C++ integration
#

from info_weave import __doc__
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

from scipy_test.testing import ScipyTest
test = ScipyTest('weave').test
