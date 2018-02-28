from __future__ import division, absolute_import, print_function

from . import util

from numpy.testing import run_module_suite, assert_equal, dec

import sys

class TestQuotedCharacter(util.F2PyTest):
    code = """
      SUBROUTINE FOO(OUT1, OUT2, OUT3, OUT4, OUT5, OUT6)
      CHARACTER SINGLE, DOUBLE, SEMICOL, EXCLA, OPENPAR, CLOSEPAR
      PARAMETER (SINGLE="'", DOUBLE='"', SEMICOL=';', EXCLA="!", 
     1           OPENPAR="(", CLOSEPAR=")")
      CHARACTER OUT1, OUT2, OUT3, OUT4, OUT5, OUT6
Cf2py intent(out) OUT1, OUT2, OUT3, OUT4, OUT5, OUT6
      OUT1 = SINGLE
      OUT2 = DOUBLE
      OUT3 = SEMICOL
      OUT4 = EXCLA
      OUT5 = OPENPAR
      OUT6 = CLOSEPAR
      RETURN
      END
    """

    @dec.knownfailureif(sys.platform=='win32', msg='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_quoted_character(self):
        assert_equal(self.module.foo(), (b"'", b'"', b';', b'!', b'(', b')'))

if __name__ == "__main__":
    run_module_suite()
