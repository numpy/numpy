__usage__ = """
Run:
  python return_real.py [<f2py options>]
Examples:
  python return_real.py --fcompiler=Gnu --no-wrap-functions
  python return_real.py --quiet
"""

import numpy.f2py as f2py2e
from numpy import array

def build(f2py_opts):
    try:
        import f77_ext_return_real
    except ImportError:
        assert not f2py2e.compile('''\
       function t0(value)
         real value
         real t0
         t0 = value
       end
       function t4(value)
         real*4 value
         real*4 t4
         t4 = value
       end
       function t8(value)
         real*8 value
         real*8 t8
         t8 = value
       end
       function td(value)
         double precision value
         double precision td
         td = value
       end

       subroutine s0(t0,value)
         real value
         real t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s4(t4,value)
         real*4 value
         real*4 t4
cf2py    intent(out) t4
         t4 = value
       end
       subroutine s8(t8,value)
         real*8 value
         real*8 t8
cf2py    intent(out) t8
         t8 = value
       end
       subroutine sd(td,value)
         double precision value
         double precision td
cf2py    intent(out) td
         td = value
       end
''','f77_ext_return_real',f2py_opts,source_fn='f77_ret_real.f')

    from f77_ext_return_real import t0,t4,t8,td,s0,s4,s8,sd
    test_functions = [t0,t4,t8,td,s0,s4,s8,sd]
    return test_functions

def runtest(t):
    import sys
    if t.__doc__.split()[0] in ['t0','t4','s0','s4']:
        err = 1e-5
    else:
        err = 0.0
    assert abs(t(234)-234.0)<=err
    assert abs(t(234.6)-234.6)<=err
    assert abs(t(234l)-234.0)<=err
    if sys.version[:3]<'2.3':
        assert abs(t(234.6+3j)-234.6)<=err
    assert abs(t('234')-234)<=err
    assert abs(t('234.6')-234.6)<=err
    assert abs(t(-234)+234)<=err
    assert abs(t([234])-234)<=err
    assert abs(t((234,))-234.)<=err
    assert abs(t(array(234))-234.)<=err
    assert abs(t(array([234]))-234.)<=err
    assert abs(t(array([[234]]))-234.)<=err
    assert abs(t(array([234],'b'))+22)<=err
    assert abs(t(array([234],'h'))-234.)<=err
    assert abs(t(array([234],'i'))-234.)<=err
    assert abs(t(array([234],'l'))-234.)<=err
    assert abs(t(array([234],'B'))-234.)<=err
    assert abs(t(array([234],'f'))-234.)<=err
    assert abs(t(array([234],'d'))-234.)<=err
    if sys.version[:3]<'2.3':
        assert abs(t(array([234+3j],'F'))-234.)<=err
        assert abs(t(array([234],'D'))-234.)<=err
    if t.__doc__.split()[0] in ['t0','t4','s0','s4']:
        assert t(1e200)==t(1e300) # inf

    try: raise RuntimeError,`t(array([234],'c'))`
    except ValueError: pass
    try: raise RuntimeError,`t('abc')`
    except ValueError: pass

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
        r = t(10l**400); assert `r` in ['inf','Infinity'],`r`

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py2e.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'
