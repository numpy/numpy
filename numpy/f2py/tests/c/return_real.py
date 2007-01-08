__usage__ = """
Run:
  python return_real.py [<f2py options>]
Examples:
  python return_real.py --fcompiler=Gnu --no-wrap-functions
  python return_real.py --quiet
"""

import f2py2e
from Numeric import array

def build(f2py_opts):
    try:
        import c_ext_return_real
    except ImportError:
        assert not f2py2e.compile('''\
python module c_ext_return_real
usercode \'\'\'
float t4(float value) { return value; }
void s4(float *t4, float value) { *t4 = value; }
double t8(double value) { return value; }
void s8(double *t8, double value) { *t8 = value; }
\'\'\'
interface
  function t4(value)
    real*4 intent(c) :: t4,value
  end
  function t8(value)
    real*8 intent(c) :: t8,value
  end
  subroutine s4(t4,value)
    intent(c) s4
    real*4 intent(out) :: t4
    real*4 intent(c) :: value
  end
  subroutine s8(t8,value)
    intent(c) s8
    real*8 intent(out) :: t8
    real*8 intent(c) :: value
  end
end interface
end python module c_ext_return_real
''','c_ext_return_real',f2py_opts,source_fn='c_ret_real.pyf')

    from c_ext_return_real import t4,t8,s4,s8
    test_functions = [t4,t8,s4,s8]
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
    assert abs(t(array([234],'1'))+22)<=err
    assert abs(t(array([234],'s'))-234.)<=err
    assert abs(t(array([234],'i'))-234.)<=err
    assert abs(t(array([234],'l'))-234.)<=err
    assert abs(t(array([234],'b'))-234.)<=err
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
