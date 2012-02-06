#
#

# Get documentation string:
from info import __doc__

# Import symbols from sub-module:
from pad import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

