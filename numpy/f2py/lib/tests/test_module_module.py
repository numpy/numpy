#!/usr/bin/env python
"""
Tests for module with scalar derived types and subprograms.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

import os
import sys
from numpy.testing import *

set_package_path()
from lib.main import build_extension, compile
restore_path()

fortran_code = '''
module test_module_module_ext2
  type rat
    integer n,d
  end type rat
  contains
    subroutine foo2()
      print*,"In foo2"
    end subroutine foo2
end module
module test_module_module_ext
  contains
    subroutine foo
      use test_module_module_ext2
      print*,"In foo"
      call foo2
    end subroutine foo
    subroutine bar(a)
      use test_module_module_ext2
      type(rat) a
      print*,"In bar,a=",a
    end subroutine bar
end module test_module_module_ext
'''

m,m2 = compile(fortran_code, modulenames=['test_module_module_ext',
                                          'test_module_module_ext2',
                                          ])

from numpy import *

class TestM(NumpyTestCase):

    def check_foo_simple(self, level=1):
        foo = m.foo
        foo()

if __name__ == "__main__":
    NumpyTest().run()
