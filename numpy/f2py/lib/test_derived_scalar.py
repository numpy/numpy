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

def build(fortran_code, rebuild=True):
    modulename = os.path.splitext(os.path.basename(__file__))[0]+'_ext'
    try:
        exec ('import %s as m' % (modulename))
        if rebuild and os.stat(m.__file__)[8] < os.stat(__file__)[8]:
            del sys.modules[m.__name__] # soft unload extension module
            os.remove(m.__file__)
            raise ImportError,'%s is newer than %s' % (__file__, m.__file__)
    except ImportError,msg:
        print msg, ', recompiling %s.' % (modulename)
        import tempfile
        fname = tempfile.mktemp() + '.f90'
        f = open(fname,'w')
        f.write(fortran_code)
        f.close()
        sys_argv = []
        sys_argv.extend(['--build-dir','dsctmp'])
        #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
        from main import build_extension
        sys_argv.extend(['-m',modulename, fname])
        build_extension(sys_argv)
        os.remove(fname)
        os.system(' '.join([sys.executable] + sys.argv))
        sys.exit(0)
    return m

fortran_code = '''
subroutine foo(a)
  type myt
    integer flag
  end type myt
  type(myt) a
!f2py intent(in,out) a
  a % flag = a % flag + 1
end
'''

# tester note: set rebuild=True when changing fortan_code and for SVN
m = build(fortran_code, rebuild=True)

from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        a = m.myt(2)
        assert_equal(a.flag,2)
        assert isinstance(a,m.myt),`a`
        r = m.foo(a)
        assert isinstance(r,m.myt),`r`
        assert_equal(r.flag,3)
        assert_equal(a.flag,2)
        
if __name__ == "__main__":
    NumpyTest().run()
