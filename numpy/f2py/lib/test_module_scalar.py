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

def build(fortran_code, rebuild=True):
    modulename = os.path.splitext(os.path.basename(__file__))[0] + '_ext'
    try:
        exec ('import %s as m' % (modulename))
        if rebuild and os.stat(m.__file__)[8] < os.stat(__file__)[8]:
            del sys.modules[m.__name__] # soft unload extension module
            os.remove(m.__file__)
            raise ImportError,'%s is newer than %s' % (__file__, m.__file__)
    except ImportError,msg:
        assert str(msg)==('No module named %s' % (modulename)),str(msg)
        print msg, ', recompiling %s.' % (modulename)
        import tempfile
        fname = tempfile.mktemp() + '.f90'
        f = open(fname,'w')
        f.write(fortran_code)
        f.close()
        sys_argv = []
        sys_argv.extend(['--build-dir','tmp'])
        #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
        from main import build_extension
        sys_argv.extend(['-m',modulename, fname])
        build_extension(sys_argv)
        os.remove(fname)
        status = os.system(' '.join([sys.executable] + sys.argv))
        sys.exit(status)
    return m

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

# tester note: set rebuild=True when changing fortan_code and for SVN
m = build(fortran_code, rebuild=True)

from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        foo = m.foo
        r = foo(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,3)

    def check_foo2_simple(self, level=1):
        foo2 = m.foo2
        r = foo2(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,4)

if __name__ == "__main__":
    NumpyTest().run()
