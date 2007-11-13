#!/usr/bin/env python
__usage__ = """
Run:
  python run.py [<f2py options>]
Examples:
  python run.py --quiet
"""

import os
import sys
import f2py2e
from Numeric import array

def build(f2py_opts):
    try:
        import mixed_f77_f90
    except:
        d,b=os.path.split(sys.argv[0])
        files = ['foo.f','foo_fixed.f90','foo_free.f90']
        files = [os.path.join(d,f) for f in files]
        files = ' '.join(files)
        args = ' -c -m mixed_f77_f90 %s %s'%(files,f2py_opts)
        c = '%s -c "import f2py2e;f2py2e.main()" %s' %(sys.executable,args)
        s = os.system(c)
        assert not s
    from mixed_f77_f90 import bar11
    test_functions = [bar11]
    from mixed_f77_f90 import foo_fixed as m
    test_functions.append(m.bar12)
    from mixed_f77_f90 import foo_free as m
    test_functions.append(m.bar13)
    return test_functions

def runtest(t):
    tname = t.__doc__.split()[0]
    if tname=='bar11':
        assert t()==11
    elif tname=='bar12':
        assert t()==12
    elif tname=='bar13':
        assert t()==13
    else:
        raise NotImplementedError

if __name__=='__main__':
    repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py2e.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'
