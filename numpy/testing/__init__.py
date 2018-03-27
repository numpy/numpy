"""Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.

"""
from __future__ import division, absolute_import, print_function

from unittest import TestCase

from ._private.utils import *
from ._private import decorators as dec
from ._private.nosetester import (
    run_module_suite, NoseTester as Tester, _numpy_tester,
    )

__all__ = _private.utils.__all__ + ['TestCase', 'run_module_suite']

test = _numpy_tester().test
