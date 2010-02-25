# XXX: investigate cases that are disabled under win32
#

__usage__ = """
Run:
  python return_integer.py [<f2py options>]
Examples:
  python return_integer.py --quiet
"""

import sys
import numpy.f2py as f2py
from numpy import array

def build(f2py_opts):
    try:
        import f90_ext_return_integer
    except ImportError:
        assert not f2py.compile('''\
module f90_return_integer
  contains
       function t0(value)
         integer :: value
         integer :: t0
         t0 = value
       end function t0
       function t1(value)
         integer(kind=1) :: value
         integer(kind=1) :: t1
         t1 = value
       end function t1
       function t2(value)
         integer(kind=2) :: value
         integer(kind=2) :: t2
         t2 = value
       end function t2
       function t4(value)
         integer(kind=4) :: value
         integer(kind=4) :: t4
         t4 = value
       end function t4
       function t8(value)
         integer(kind=8) :: value
         integer(kind=8) :: t8
         t8 = value
       end function t8

       subroutine s0(t0,value)
         integer :: value
         integer :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s1(t1,value)
         integer(kind=1) :: value
         integer(kind=1) :: t1
!f2py    intent(out) t1
         t1 = value
       end subroutine s1
       subroutine s2(t2,value)
         integer(kind=2) :: value
         integer(kind=2) :: t2
!f2py    intent(out) t2
         t2 = value
       end subroutine s2
       subroutine s4(t4,value)
         integer(kind=4) :: value
         integer(kind=4) :: t4
!f2py    intent(out) t4
         t4 = value
       end subroutine s4
       subroutine s8(t8,value)
         integer(kind=8) :: value
         integer(kind=8) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
end module f90_return_integer
''','f90_ext_return_integer',f2py_opts,source_fn='f90_ret_int.f90')

    from f90_ext_return_integer import f90_return_integer as m
    test_functions = [m.t0,m.t1,m.t2,m.t4,m.t8,m.s0,m.s1,m.s2,m.s4,m.s8]
    return test_functions

def runtest(t):
    tname = t.__doc__.split()[0]
    assert t(123)==123
    assert t(123.6)==123
    assert t(123l)==123
    if sys.version[:3]<='2.2':
        assert t(123.6+3j)==123
    assert t('123')==123
    assert t(-123)==-123
    assert t([123])==123
    assert t((123,))==123
    assert t(array(123))==123
    assert t(array([123]))==123
    assert t(array([[123]]))==123
    assert t(array([123],'b'))==123
    assert t(array([123],'h'))==123
    assert t(array([123],'i'))==123
    assert t(array([123],'l'))==123
    assert t(array([123],'q'))==123
    assert t(array([123],'f'))==123
    assert t(array([123],'d'))==123
    if sys.version[:3]<='2.2':
        assert t(array([123+3j],'F'))==123
        assert t(array([123],'D'))==123

    try: raise RuntimeError,`t(array([123],'c'))`
    except ValueError: pass
    except RuntimeError: print "Failed Error"
    except: print "Wrong Error Type 1"

    try: raise RuntimeError,`t('abc')`
    except ValueError: pass
    except RuntimeError: print "Failed Error"
    except: print "Wrong Error Type 2"

    try: raise RuntimeError,`t([])`
    except IndexError: pass
    except RuntimeError: print "Failed Error"
    except: print "Wrong Error Type 3"

    try: raise RuntimeError,`t(())`
    except IndexError: pass
    except RuntimeError: print "Failed Error"
    except: print "Wrong Error Type 4"

    try: raise RuntimeError,`t(t)`
    except TypeError: pass
    except RuntimeError: print "Failed Error"
    except: print "Wrong Error Type 5"

    try: raise RuntimeError,`t({})`
    except TypeError: pass
    except RuntimeError: print "Failed Error"
    except: print "Wrong Error Type 6"

    if tname in ['t8','s8']:
        try: raise RuntimeError,`t(100000000000000000000000l)`
        except OverflowError: pass
        try: raise RuntimeError,`t(10000000011111111111111.23)`
        except OverflowError: pass
    else:
        if sys.version[:3]<='2.2':
            try: raise RuntimeError,`t(10000000000000l)`
            except OverflowError: pass
            try: raise RuntimeError,`t(10000000000.23)`
            except OverflowError: pass

if __name__=='__main__':
    #import libwadpy
    status = 1
    try:
        repeat,f2py_opts = f2py.f2py_testing.cmdline()
        test_functions = build(f2py_opts)
        f2py.f2py_testing.run(runtest,test_functions,repeat)
        print 'ok'
        status = 0
    finally:
        if status:
            print '*'*20
            print 'Running f2py.diagnose'
            f2py.diagnose.run()
