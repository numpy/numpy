#!/usr/bin/env python
"""
Tests for intent(in,out) derived type arguments in Fortran subroutine's.

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
subroutine foo(a)
  type myt
    integer flag
  end type myt
  type(myt) a
!f2py intent(in,out) a
  a % flag = a % flag + 1
end
function foo2(a)
  type myt
    integer flag
  end type myt
  type(myt) a
  type(myt) foo2
  foo2 % flag = a % flag + 2
end
'''

m, = compile(fortran_code, 'test_derived_scalar_ext')

from numpy import *

class TestM(TestCase):

    def test_foo_simple(self, level=1):
        a = m.myt(2)
        assert_equal(a.flag,2)
        assert isinstance(a,m.myt),`a`
        r = m.foo(a)
        assert isinstance(r,m.myt),`r`
        assert r is a
        assert_equal(r.flag,3)
        assert_equal(a.flag,3)

        a.flag = 5
        assert_equal(r.flag,5)

        #s = m.foo((5,))

    def test_foo2_simple(self, level=1):
        a = m.myt(2)
        assert_equal(a.flag,2)
        assert isinstance(a,m.myt),`a`
        r = m.foo2(a)
        assert isinstance(r,m.myt),`r`
        assert r is not a
        assert_equal(a.flag,2)
        assert_equal(r.flag,4)


if __name__ == "__main__":
    run_module_suite()
