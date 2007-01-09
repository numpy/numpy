#!/usr/bin/env python
"""
Tests for intent(in) arguments in subroutine-wrapped Fortran functions.

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
        assert str(msg).startswith('No module named'),str(msg)
        print msg, ', recompiling %s.' % (modulename)
        import tempfile
        fname = tempfile.mktemp() + '.f'
        f = open(fname,'w')
        f.write(fortran_code)
        f.close()
        sys_argv = ['--build-dir','tmp']
        #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
        from main import build_extension
        sys_argv.extend(['-m',modulename, fname])
        build_extension(sys_argv)
        os.remove(fname)
        os.system(' '.join([sys.executable] + sys.argv))
        sys.exit(0)
    return m

fortran_code = '''
      function fooint1(a)
      integer*1 a
      integer*1 fooint1
      fooint1 = a + 1
      end
      function fooint2(a)
      integer*2 a
      integer*2 fooint2
      fooint2 = a + 1
      end
      function fooint4(a)
      integer*4 a
      integer*4 fooint4
      fooint4 = a + 1
      end
      function fooint8(a)
      integer*8 a
      integer*8 fooint8
      fooint8 = a + 1
      end
      function foofloat4(a)
      real*4 a
      real*4 foofloat4
      foofloat4 = a + 1.0e0
      end
      function foofloat8(a)
      real*8 a
      real*8 foofloat8
      foofloat8 = a + 1.0d0
      end
      function foocomplex8(a)
      complex*8 a
      complex*8 foocomplex8
      foocomplex8 = a + 1.0e0
      end
      function foocomplex16(a)
      complex*16 a
      complex*16 foocomplex16
      foocomplex16 = a + 1.0d0
      end
      function foobool1(a)
      logical*1 a
      logical*1 foobool1
      foobool1 = .not. a
      end
      function foobool2(a)
      logical*2 a
      logical*2 foobool2
      foobool2 = .not. a
      end
      function foobool4(a)
      logical*4 a
      logical*4 foobool4
      foobool4 = .not. a
      end
      function foobool8(a)
      logical*8 a
      logical*8 foobool8
      foobool8 = .not. a
      end
      function foostring1(a)
      character*1 a
      character*1 foostring1
      foostring1 = "1"
      end
      function foostring5(a)
      character*5 a
      character*5 foostring5
      foostring5 = a
      foostring5(1:2) = "12"
      end
!      function foostringstar(a)
!      character*(*) a
!      character*(*) foostringstar
!      if (len(a).gt.0) then
!        foostringstar = a
!        foostringstar(1:1) = "1"
!      endif
!      end
'''

# tester note: set rebuild=True when changing fortan_code and for SVN
m = build(fortran_code, rebuild=True)

from numpy import *

class test_m(NumpyTestCase):

    def check_foo_integer1(self, level=1):
        i = int8(2)
        e = int8(3)
        func = m.fooint1
        assert isinstance(i,int8),`type(i)`
        r = func(i)
        assert isinstance(r,int8),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,int8),`type(r)`
        assert_equal(r,e)

        for intx in [int64,int16,int32]:
            r = func(intx(2))
            assert isinstance(r,int8),`type(r)`
            assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,int8),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,int8),`type(r)`
        assert_equal(r,e)

        r = func([2])
        assert isinstance(r,int8),`type(r)`
        assert_equal(r,e)

        self.assertRaises(TypeError,lambda :func(2.2j))
        self.assertRaises(TypeError,lambda :func([2,1]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_integer2(self, level=1):
        i = int16(2)
        e = int16(3)
        func = m.fooint2
        assert isinstance(i,int16),`type(i)`
        r = func(i)
        assert isinstance(r,int16),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,int16),`type(r)`
        assert_equal(r,e)

        for intx in [int8,int64,int32]:
            r = func(intx(2))
            assert isinstance(r,int16),`type(r)`
            assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,int16),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,int16),`type(r)`
        assert_equal(r,e)

        r = func([2])
        assert isinstance(r,int16),`type(r)`
        assert_equal(r,e)

        self.assertRaises(TypeError,lambda :func(2.2j))
        self.assertRaises(TypeError,lambda :func([2,1]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_integer4(self, level=1):
        i = int32(2)
        e = int32(3)
        func = m.fooint4
        assert isinstance(i,int32),`type(i)`
        r = func(i)
        assert isinstance(r,int32),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,e)

        for intx in [int8,int16,int64]:
            r = func(intx(2))
            assert isinstance(r,int32),`type(r)`
            assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,e)

        r = func([2])
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,e)

        self.assertRaises(TypeError,lambda :func(2.2j))
        self.assertRaises(TypeError,lambda :func([2,1]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_integer8(self, level=1):
        i = int64(2)
        e = int64(3)
        func = m.fooint8
        assert isinstance(i,int64),`type(i)`
        r = func(i)
        assert isinstance(r,int64),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,int64),`type(r)`
        assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,int64),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,int64),`type(r)`
        assert_equal(r,e)

        for intx in [int8,int16,int32]:
            r = func(intx(2))
            assert isinstance(r,int64),`type(r)`
            assert_equal(r,e)

        r = func([2])
        assert isinstance(r,int64),`type(r)`
        assert_equal(r,e)

        self.assertRaises(TypeError,lambda :func(2.2j))
        self.assertRaises(TypeError,lambda :func([2,1]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_real4(self, level=1):
        i = float32(2)
        e = float32(3)
        func = m.foofloat4
        assert isinstance(i,float32),`type(i)`
        r = func(i)
        assert isinstance(r,float32),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,float32),`type(r)`
        assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,float32),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,float32),`type(r)`
        assert_equal(r,e+float32(0.2))

        r = func(float64(2.0))
        assert isinstance(r,float32),`type(r)`
        assert_equal(r,e)

        r = func([2])
        assert isinstance(r,float32),`type(r)`
        assert_equal(r,e)

        self.assertRaises(TypeError,lambda :func(2.2j))
        self.assertRaises(TypeError,lambda :func([2,1]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_real8(self, level=1):
        i = float64(2)
        e = float64(3)
        func = m.foofloat8
        assert isinstance(i,float64),`type(i)`
        r = func(i)
        assert isinstance(r,float64),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,float64),`type(r)`
        assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,float64),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,float64),`type(r)`
        assert_equal(r,e+float64(0.2))

        r = func(float32(2.0))
        assert isinstance(r,float64),`type(r)`
        assert_equal(r,e)

        r = func([2])
        assert isinstance(r,float64),`type(r)`
        assert_equal(r,e)

        self.assertRaises(TypeError,lambda :func(2.2j))
        self.assertRaises(TypeError,lambda :func([2,1]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_complex8(self, level=1):
        i = complex64(2)
        e = complex64(3)
        func = m.foocomplex8
        assert isinstance(i,complex64),`type(i)`
        r = func(i)
        assert isinstance(r,complex64),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,complex64),`type(r)`
        assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,complex64),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,complex64),`type(r)`
        assert_equal(r,e+complex64(0.2))

        r = func(2+1j)
        assert isinstance(r,complex64),`type(r)`
        assert_equal(r,e+complex64(1j))

        r = func(complex128(2.0))
        assert isinstance(r,complex64),`type(r)`
        assert_equal(r,e)

        r = func([2])
        assert isinstance(r,complex64),`type(r)`
        assert_equal(r,e)

        r = func([2,3])
        assert isinstance(r,complex64),`type(r)`
        assert_equal(r,e+complex64(3j))

        self.assertRaises(TypeError,lambda :func([2,1,3]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_complex16(self, level=1):
        i = complex128(2)
        e = complex128(3)
        func = m.foocomplex16
        assert isinstance(i,complex128),`type(i)`
        r = func(i)
        assert isinstance(r,complex128),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func(2)
        assert isinstance(r,complex128),`type(r)`
        assert_equal(r,e)

        r = func(2.0)
        assert isinstance(r,complex128),`type(r)`
        assert_equal(r,e)

        r = func(2.2)
        assert isinstance(r,complex128),`type(r)`
        assert_equal(r,e+complex128(0.2))

        r = func(2+1j)
        assert isinstance(r,complex128),`type(r)`
        assert_equal(r,e+complex128(1j))

        r = func([2])
        assert isinstance(r,complex128),`type(r)`
        assert_equal(r,e)

        r = func([2,3])
        assert isinstance(r,complex128),`type(r)`
        assert_equal(r,e+complex128(3j))

        r = func(complex64(2.0))
        assert isinstance(r,complex128),`type(r)`
        assert_equal(r,e)

        self.assertRaises(TypeError,lambda :func([2,1,3]))
        self.assertRaises(TypeError,lambda :func({}))

    def check_foo_bool1(self, level=1):
        i = bool8(True)
        e = bool8(False)
        func = m.foobool1
        assert isinstance(i,bool8),`type(i)`
        r = func(i)
        assert isinstance(r,bool8),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        for tv in [1,2,2.1,-1j,[0],True]:
            r = func(tv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,e)

        for fv in [0,0.0,0j,False,(),{},[]]:
            r = func(fv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,not e)

    def check_foo_bool2(self, level=1):
        i = bool8(True)
        e = bool8(False)
        func = m.foobool2
        assert isinstance(i,bool8),`type(i)`
        r = func(i)
        assert isinstance(r,bool8),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        for tv in [1,2,2.1,-1j,[0],True]:
            r = func(tv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,e)

        for fv in [0,0.0,0j,False,(),{},[]]:
            r = func(fv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,not e)

    def check_foo_bool4(self, level=1):
        i = bool8(True)
        e = bool8(False)
        func = m.foobool4
        assert isinstance(i,bool8),`type(i)`
        r = func(i)
        assert isinstance(r,bool8),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        for tv in [1,2,2.1,-1j,[0],True]:
            r = func(tv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,e)

        for fv in [0,0.0,0j,False,(),{},[]]:
            r = func(fv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,not e)

    def check_foo_bool8(self, level=1):
        i = bool8(True)
        e = bool8(False)
        func = m.foobool8
        assert isinstance(i,bool8),`type(i)`
        r = func(i)
        assert isinstance(r,bool8),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        for tv in [1,2,2.1,-1j,[0],True]:
            r = func(tv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,e)

        for fv in [0,0.0,0j,False,(),{},[]]:
            r = func(fv)
            assert isinstance(r,bool8),`type(r)`
            assert_equal(r,not e)

    def check_foo_string1(self, level=1):
        i = string0('a')
        e = string0('1')
        func = m.foostring1
        assert isinstance(i,string0),`type(i)`
        r = func(i)
        assert isinstance(r,string0),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func('ab')
        assert isinstance(r,string0),`type(r)`
        assert_equal(r,e)

        r = func('')
        assert isinstance(r,string0),`type(r)`
        assert_equal(r,e)

    def check_foo_string5(self, level=1):
        i = string0('abcde')
        e = string0('12cde')
        func = m.foostring5
        assert isinstance(i,string0),`type(i)`
        r = func(i)
        assert isinstance(r,string0),`type(r)`
        assert i is not r,`id(i),id(r)`
        assert_equal(r,e)

        r = func('abc')
        assert isinstance(r,string0),`type(r)`
        assert_equal(r,'12c  ')

        r = func('abcdefghi')
        assert isinstance(r,string0),`type(r)`
        assert_equal(r,'12cde')

        r = func([1])
        assert isinstance(r,string0),`type(r)`
        assert_equal(r,'12]  ')

    def _check_foo_string0(self, level=1):
        i = string0('abcde')
        e = string0('12cde')
        func = m.foostringstar
        r = func('abcde')
        assert_equal(r,'1bcde')
        r = func('')
        assert_equal(r,'')

if __name__ == "__main__":
    NumpyTest().run()
