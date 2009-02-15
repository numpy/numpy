"""Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.
"""

from unittest import TestCase

import decorators as dec
from utils import *
from numpytest import *
from nosetester import NoseTester as Tester
from nosetester import run_module_suite
test = Tester().test
