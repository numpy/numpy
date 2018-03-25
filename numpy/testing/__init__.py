"""Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.

"""
from __future__ import division, absolute_import, print_function

from unittest import TestCase

from .utils import *
from . import decorators as dec

# Note: remove import of _numpy_tester from this import to avoid a nose
# dependency. All the others have lazy imports.
from .nosetester import run_module_suite, NoseTester as Tester, _numpy_tester
test = _numpy_tester().test
