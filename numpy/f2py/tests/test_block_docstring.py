from __future__ import division as _, absolute_import as _, print_function as _

import textwrap
import sys
import pytest
from . import util

from numpy.testing import assert_equal

class TestBlockDocString(util.F2PyTest):
    code = """
      SUBROUTINE FOO()
      INTEGER BAR(2, 3)

      COMMON  /BLOCK/ BAR
      RETURN
      END
    """

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_block_docstring(self):
        expected = "'i'-array(2,3)\n"
        assert_equal(self.module.block.__doc__, expected)
