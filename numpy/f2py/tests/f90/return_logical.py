__usage__ = """
Run:
  python return_logical.py [<f2py options>]
Examples:
  python return_logical.py --quiet
"""

import numpy.f2py as f2py
from numpy import array

try: True
except NameError:
    True = 1
    False = 0

def build(f2py_opts):
    try:
        import f90_ext_return_logical
    except ImportError:
        assert not f2py.compile('''\
module f90_return_logical
  contains
       function t0(value)
         logical :: value
         logical :: t0
         t0 = value
       end function t0
       function t1(value)
         logical(kind=1) :: value
         logical(kind=1) :: t1
         t1 = value
       end function t1
       function t2(value)
         logical(kind=2) :: value
         logical(kind=2) :: t2
         t2 = value
       end function t2
       function t4(value)
         logical(kind=4) :: value
         logical(kind=4) :: t4
         t4 = value
       end function t4
       function t8(value)
         logical(kind=8) :: value
         logical(kind=8) :: t8
         t8 = value
       end function t8

       subroutine s0(t0,value)
         logical :: value
         logical :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s1(t1,value)
         logical(kind=1) :: value
         logical(kind=1) :: t1
!f2py    intent(out) t1
         t1 = value
       end subroutine s1
       subroutine s2(t2,value)
         logical(kind=2) :: value
         logical(kind=2) :: t2
!f2py    intent(out) t2
         t2 = value
       end subroutine s2
       subroutine s4(t4,value)
         logical(kind=4) :: value
         logical(kind=4) :: t4
!f2py    intent(out) t4
         t4 = value
       end subroutine s4
       subroutine s8(t8,value)
         logical(kind=8) :: value
         logical(kind=8) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
end module f90_return_logical
''','f90_ext_return_logical',f2py_opts,source_fn='f90_ret_log.f90')

    from f90_ext_return_logical import f90_return_logical as m
    test_functions = [m.t0,m.t1,m.t2,m.t4,m.t8,m.s0,m.s1,m.s2,m.s4,m.s8]
    return test_functions




def runtest(t):
    assert t(True)==1,`t(True)`
    assert t(False)==0,`t(False)`
    assert t(0)==0
    assert t(None)==0
    assert t(0.0)==0
    assert t(0j)==0
    assert t(1j)==1
    assert t(234)==1
    assert t(234.6)==1
    assert t(234l)==1
    assert t(234.6+3j)==1
    assert t('234')==1
    assert t('aaa')==1
    assert t('')==0
    assert t([])==0
    assert t(())==0
    assert t({})==0
    assert t(t)==1
    assert t(-234)==1
    assert t(10l**100)==1
    assert t([234])==1
    assert t((234,))==1
    assert t(array(234))==1
    assert t(array([234]))==1
    assert t(array([[234]]))==1
    assert t(array([234],'b'))==1
    assert t(array([234],'h'))==1
    assert t(array([234],'i'))==1
    assert t(array([234],'l'))==1
    assert t(array([234],'q'))==1
    assert t(array([234],'f'))==1
    assert t(array([234],'d'))==1
    assert t(array([234+3j],'F'))==1
    assert t(array([234],'D'))==1
    assert t(array(0))==0
    assert t(array([0]))==0
    assert t(array([[0]]))==0
    assert t(array([0j]))==0
    assert t(array([1]))==1
    # The call itself raises an error.
    #assert t(array([0,0])) == 0 # fails
    #assert t(array([1,1])) == 1 # fails

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'
