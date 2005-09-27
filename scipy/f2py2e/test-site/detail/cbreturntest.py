all = 1
tests = []
skip=1
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Callback function returning integer%s'%s
        test['depends']=['fncall','cb']
        test['f']="""\
      function f(g)
      integer%s f,g
      external g
      f=g()
      end
"""%s
        test['py']="""\
import f2pytest,sys
def g():
    return 4
r = f2pytest.f(g)
if not r==4:
    sys.stderr.write('expected 4 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Callback function returning integer*8'
    test['depends']=['fncall','cb']
    test['f']="""\
      function f(g)
      integer*8 f,g
      external g
      f=g()
      end
"""
    test['py']="""\
import f2pytest,sys
def g():
    return 222222222L
r = f2pytest.f(g)
if not r==222222222L:
    sys.stderr.write('expected 222222222 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*4','*8','*16']:
        if s=='*16' and skip: continue
        test={}
        test['name']='Callback function returning real%s'%s
        test['depends']=['fncall','cb']
        test['f']="""\
      function f(g)
      real%s f,g
      external g
      f=g()
      end
"""%s
        test['py']="""\
import f2pytest,sys
def g():
    return 34.56
r = f2pytest.f(g)
if abs(r-34.56)>1e-5:
    sys.stderr.write('expected 34.56 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*8','*16','*32']:
        if s=='*32' and skip: continue
        test={}
        test['trydefine']=['-DF2PY_CB_RETURNCOMPLEX']
        test['name']='Callback function returning complex%s'%s
        test['depends']=['fncall','cb']
        test['f']="""\
      function f(g)
      complex%s f,g
      external g
      f=g()
      end
"""%s
        test['py']="""\
import f2pytest,sys
def g():
    return 1+2j
r = f2pytest.f(g)
if abs(r-(1+2j))>1e-5:
    sys.stderr.write('expected (1+2j) but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4','*8']:
        test={}
        test['name']='Callback function returning logical%s'%s
        test['depends']=['fncall','cb']
        test['f']="""\
      function f(g,h)
      logical%s f,g,h,a,b
      external g,h
      a=g()
      a=.not. a
      b=h()
      f = a .and. b
      end
"""%s
        test['py']="""\
import f2pytest,sys
def g():
    return 0
def h():
    return 1
r = f2pytest.f(g,h)
if not r:
    sys.stderr.write('expected .true. but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    test={}
    #test['f2pyflags']=['--debug-capi']
    test['name']='Callback function returning character'
    test['depends']=['fncall','cb']
    test['f']="""\
      function f(g)
      character f
      character g
      external g
      f = g()
      end
"""
    test['py']="""\
import f2pytest,sys
def g():
    return 't'
r = f2pytest.f(g)
if not r=='t':
    sys.stderr.write('expected "t" but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Callback function returning character*9'
    test['depends']=['fncall','cb']
    test['f']="""\
      function f(g)
      character*9 f,g
      external g
      f = g()
      end
"""
    test['py']="""\
import f2pytest,sys
def g():
    return 'abcdefghi'
r = f2pytest.f(g)
if not r=='abcdefghi':
    sys.stderr.write('expected "abcdefghi" but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 and (all and not skip): 
    test={}
    test['name']='Callback function returning character*(*)' # not possible
    test['depends']=['fncall','cb']
    test['f']="""\
      function f(g)
      external g
      character*(*) g
      character*(*) f
      f = g()
      end
"""
    test['py']="""\
import f2pytest,sys
def g():
    return 'abcdefgh'
r = f2pytest.f(g)
if not r=='abcdefgh':
    sys.stderr.write('expected "abcdefgh" but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
