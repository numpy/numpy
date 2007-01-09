__usage__ = """
Run:
  python return_character.py [<f2py options>]
Examples:
  python return_character.py --fcompiler=Gnu --no-wrap-functions
  python return_character.py --quiet
"""

import sys
import f2py2e
from Numeric import array

def build(f2py_opts):
    try:
        import f77_ext_return_character
    except ImportError:
        assert not f2py2e.compile('''\
       function t0(value)
         character value
         character t0
         t0 = value
       end
       function t1(value)
         character*1 value
         character*1 t1
         t1 = value
       end
       function t5(value)
         character*5 value
         character*5 t5
         t5 = value
       end
       function ts(value)
         character*(*) value
         character*(*) ts
         ts = value
       end

       subroutine s0(t0,value)
         character value
         character t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s1(t1,value)
         character*1 value
         character*1 t1
cf2py    intent(out) t1
         t1 = value
       end
       subroutine s5(t5,value)
         character*5 value
         character*5 t5
cf2py    intent(out) t5
         t5 = value
       end
       subroutine ss(ts,value)
         character*(*) value
         character*10 ts
cf2py    intent(out) ts
         ts = value
       end
''','f77_ext_return_character',f2py_opts,source_fn='f77_ret_char.f')

    from f77_ext_return_character import t0,t1,t5,s0,s1,s5,ss
    test_functions = [t0,t1,t5,s0,s1,s5,ss]
    if sys.platform!='win32': # this is acctually compiler dependent case
        from f77_ext_return_character import ts
        test_functions.append(ts)

    return test_functions

def runtest(t):
    tname = t.__doc__.split()[0]
    if tname in ['t0','t1','s0','s1']:
        assert t(23)=='2'
        r = t('ab');assert r=='a',`r`
        r = t(array('ab'));assert r=='a',`r`
        r = t(array(77,'1'));assert r=='M',`r`
        try: raise RuntimeError,`t(array([77,87]))`
        except ValueError: pass
        try: raise RuntimeError,`t(array(77))`
        except ValueError: pass
    elif tname in ['ts','ss']:
        assert t(23)=='23        ',`t(23)`
        assert t('123456789abcdef')=='123456789a'
    elif tname in ['t5','s5']:
        assert t(23)=='23   ',`t(23)`
        assert t('ab')=='ab   ',`t('ab')`
        assert t('123456789abcdef')=='12345'
    else:
        raise NotImplementedError

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py2e.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'
