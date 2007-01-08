__usage__ = """
Run:
  python return_complex.py [<f2py options>]
Examples:
  python return_complex.py --fcompiler=Gnu --no-wrap-functions
  python return_complex.py --quiet
"""

import f2py2e
from Numeric import array

def build(f2py_opts):
    try:
        import f77_ext_return_complex
    except ImportError:
        assert not f2py2e.compile('''\
       function t0(value)
         complex value
         complex t0
         t0 = value
       end
       function t8(value)
         complex*8 value
         complex*8 t8
         t8 = value
       end
       function t16(value)
         complex*16 value
         complex*16 t16
         t16 = value
       end
       function td(value)
         double complex value
         double complex td
         td = value
       end

       subroutine s0(t0,value)
         complex value
         complex t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s8(t8,value)
         complex*8 value
         complex*8 t8
cf2py    intent(out) t8
         t8 = value
       end
       subroutine s16(t16,value)
         complex*16 value
         complex*16 t16
cf2py    intent(out) t16
         t16 = value
       end
       subroutine sd(td,value)
         double complex value
         double complex td
cf2py    intent(out) td
         td = value
       end
''','f77_ext_return_complex',f2py_opts)

    from f77_ext_return_complex import t0,t8,t16,td,s0,s8,s16,sd
    test_functions = [t0,t8,t16,td,s0,s8,s16,sd]
    return test_functions


def runtest(t):
    tname = t.__doc__.split()[0]
    if tname in ['t0','t8','s0','s8']:
        err = 1e-5
    else:
        err = 0.0
    assert abs(t(234j)-234.0j)<=err
    assert abs(t(234.6)-234.6)<=err
    assert abs(t(234l)-234.0)<=err
    assert abs(t(234.6+3j)-(234.6+3j))<=err
    #assert abs(t('234')-234.)<=err
    #assert abs(t('234.6')-234.6)<=err
    assert abs(t(-234)+234.)<=err
    assert abs(t([234])-234.)<=err
    assert abs(t((234,))-234.)<=err
    assert abs(t(array(234))-234.)<=err
    assert abs(t(array(23+4j,'F'))-(23+4j))<=err
    assert abs(t(array([234]))-234.)<=err
    assert abs(t(array([[234]]))-234.)<=err
    assert abs(t(array([234],'1'))+22.)<=err
    assert abs(t(array([234],'s'))-234.)<=err
    assert abs(t(array([234],'i'))-234.)<=err
    assert abs(t(array([234],'l'))-234.)<=err
    assert abs(t(array([234],'b'))-234.)<=err
    assert abs(t(array([234],'f'))-234.)<=err
    assert abs(t(array([234],'d'))-234.)<=err
    assert abs(t(array([234+3j],'F'))-(234+3j))<=err
    assert abs(t(array([234],'D'))-234.)<=err

    try: raise RuntimeError,`t(array([234],'c'))`
    except TypeError: pass
    try: raise RuntimeError,`t('abc')`
    except TypeError: pass

    try: raise RuntimeError,`t([])`
    except IndexError: pass
    try: raise RuntimeError,`t(())`
    except IndexError: pass

    try: raise RuntimeError,`t(t)`
    except TypeError: pass
    try: raise RuntimeError,`t({})`
    except TypeError: pass

    try:
        try: raise RuntimeError,`t(10l**400)`
        except OverflowError: pass
    except RuntimeError:
        r = t(10l**400); assert `r` in ['(inf+0j)','(Infinity+0j)'],`r`

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py2e.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'
