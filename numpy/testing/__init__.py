"""Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.
"""

#import unittest
from unittest import TestCase

import decorators as dec
from utils import *
from numpytest import *
from pkgtester import Tester
test = Tester().test
