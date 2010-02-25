__usage__ = """
Run:
  python return_character.py [<f2py options>]
Examples:
  python return_character.py --fcompiler=Gnu --no-wrap-functions
  python return_character.py --quiet
"""

import numpy.f2py as f2py
from numpy import array

def build(f2py_opts):
    try:
        import f90_ext_return_character
    except ImportError:
        assert not f2py.compile('''\
module f90_return_char
  contains
       function t0(value)
         character :: value
         character :: t0
         t0 = value
       end function t0
       function t1(value)
         character(len=1) :: value
         character(len=1) :: t1
         t1 = value
       end function t1
       function t5(value)
         character(len=5) :: value
         character(len=5) :: t5
         t5 = value
       end function t5
       function ts(value)
         character(len=*) :: value
         character(len=10) :: ts
         ts = value
       end function ts

       subroutine s0(t0,value)
         character :: value
         character :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s1(t1,value)
         character(len=1) :: value
         character(len=1) :: t1
!f2py    intent(out) t1
         t1 = value
       end subroutine s1
       subroutine s5(t5,value)
         character(len=5) :: value
         character(len=5) :: t5
!f2py    intent(out) t5
         t5 = value
       end subroutine s5
       subroutine ss(ts,value)
         character(len=*) :: value
         character(len=10) :: ts
!f2py    intent(out) ts
         ts = value
       end subroutine ss
end module f90_return_char
''','f90_ext_return_character',f2py_opts,source_fn='f90_ret_char.f90')

    from f90_ext_return_character import f90_return_char as m
    test_functions = [m.t0,m.t1,m.t5,m.ts,m.s0,m.s1,m.s5,m.ss]
    return test_functions


def runtest(t):
    tname = t.__doc__.split()[0]
    if tname in ['t0','t1','s0','s1']:
        assert t(23)=='2'
        r = t('ab');assert r=='a',`r`
        r = t(array('ab'));assert r=='a',`r`
        r = t(array(77,'l'));assert r=='M',`r`

        try: raise RuntimeError,`t(array([77,87]))`
        except ValueError: pass
        except RuntimeError: print "Failed Error"
        except: print "Wrong Error Type 1"

        try: raise RuntimeError,`t(array(77))`
        except ValueError: pass
        except RuntimeError: print "Failed Error"
        except: print "Wrong Error Type 2"

    elif tname in ['ts','ss']:
        assert t(23)=='23        ',`t(23)`
        assert t('123456789abcdef')=='123456789a',`t('123456789abcdef')`
    elif tname in ['t5','s5']:
        assert t(23)=='23   '
        assert t('ab')=='ab   '
        assert t('123456789abcdef')=='12345'
    else:
        raise NotImplementedError

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'
