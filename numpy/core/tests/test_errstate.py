# The following exec statement (or something like it) is needed to
# prevent SyntaxError on Python < 2.5. Even though this is a test,
# SyntaxErrors are not acceptable; on Debian systems, they block
# byte-compilation during install and thus cause the package to fail
# to install.

import sys
if sys.version_info[:2] >= (2, 5):
    exec """
from __future__ import with_statement
import platform

from numpy.core import *
from numpy.random import rand, randint
from numpy.testing import *


class TestErrstate(TestCase):
    @dec.skipif(platform.machine() == "armv5tel", "See gh-413.")
    def test_invalid(self):
        with errstate(all='raise', under='ignore'):
            a = -arange(3)
            # This should work
            with errstate(invalid='ignore'):
                sqrt(a)
            # While this should fail!
            try:
                sqrt(a)
            except FloatingPointError:
                pass
            else:
                self.fail("Did not raise an invalid error")

    def test_divide(self):
        with errstate(all='raise', under='ignore'):
            a = -arange(3)
            # This should work
            with errstate(divide='ignore'):
                a // 0
            # While this should fail!
            try:
                a // 0
            except FloatingPointError:
                pass
            else:
                self.fail("Did not raise divide by zero error")

    def test_errcall(self):
        def foo(*args):
            print(args)
        olderrcall = geterrcall()
        with errstate(call=foo):
            assert_(geterrcall() is foo, 'call is not foo')
            with errstate(call=None):
                assert_(geterrcall() is None, 'call is not None')
        assert_(geterrcall() is olderrcall, 'call is not olderrcall')

"""

if __name__ == "__main__":
    run_module_suite()
