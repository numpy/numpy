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
from numpy.f2py.lib.main import build_extension, compile

fortran_code = '''
module test_module_scalar_ext

  contains
    subroutine foo(a)
    integer a
!f2py intent(in,out) a
    a = a + 1
    end subroutine foo
    function foo2(a)
    integer a
    integer foo2
    foo2 = a + 2
    end function foo2
end module test_module_scalar_ext
'''

m, = compile(fortran_code, modulenames = ['test_module_scalar_ext'])

from numpy import *

class TestM(TestCase):

    def test_foo_simple(self, level=1):
        foo = m.foo
        r = foo(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,3)

    def test_foo2_simple(self, level=1):
        foo2 = m.foo2
        r = foo2(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,4)

if __name__ == "__main__":
    run_module_suite()
