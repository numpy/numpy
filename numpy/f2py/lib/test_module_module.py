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

def build(fortran_code, rebuild=True, build_dir = 'tmp'):
    modulename = os.path.splitext(os.path.basename(__file__))[0] + '_ext'
    try:
        exec ('import %s as m' % (modulename))
        if rebuild and os.stat(m.__file__)[8] < os.stat(__file__)[8]:
            del sys.modules[m.__name__] # soft unload extension module
            os.remove(m.__file__)
            raise ImportError,'%s is newer than %s' % (__file__, m.__file__)
    except ImportError,msg:
        assert str(msg)==('No module named %s' % (modulename)) \
               or str(msg).startswith('%s is newer than' % (__file__)),str(msg)
        print msg, ', recompiling %s.' % (modulename)
        if not os.path.isdir(build_dir): os.makedirs(build_dir)
        fname = os.path.join(build_dir, modulename + '_source.f90')
        f = open(fname,'w')
        f.write(fortran_code)
        f.close()
        sys_argv = []
        sys_argv.extend(['--build-dir',build_dir])
        #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
        from main import build_extension
        sys_argv.extend(['-m',modulename, fname])
        build_extension(sys_argv)
        status = os.system(' '.join([sys.executable] + sys.argv))
        sys.exit(status)
    return m

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

# tester note: set rebuild=True when changing fortan_code and for SVN
m = build(fortran_code, rebuild=True)

from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        foo = m.foo
        foo()

if __name__ == "__main__":
    NumpyTest().run()
