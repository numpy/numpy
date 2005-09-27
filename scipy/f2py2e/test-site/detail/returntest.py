all = 1
tests = []
skip=1
################################################################
if 0 or all:
    for s in ['','*1']:
        test={}
        test['name']='Fortran function returning character%s'%s
        test['depends']=['fncall']
        test['f']="""\
      function f()
      character%s f
      f = "Y"
      end
"""%s
        test['py']="""\
import f2pytest,sys
e = 'Y'
r = f2pytest.f()
if not e==r:
    sys.stderr.write('expected %s but got %s\\n'%(`e`,`r`))
    sys.exit()
print 'ok'
"""
        tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Fortran function returning character*9'
    test['depends']=['fncall']
    test['f']="""\
      function f()
      character*9 f
      f = "abcdefgh"
      end
"""
    test['py']="""\
import f2pytest,sys
e = 'abcdefgh '
r = f2pytest.f()
if not e==r:
    sys.stderr.write('expected %s but got %s\\n'%(`e`,`r`))
    sys.exit()
print 'ok'
"""
    tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Fortran function returning character*(*)'
    test['depends']=['fncall']
    test['f']="""\
      function f()
      character*(*) f
      f = "abcdefgh"
      end
"""
    test['py']="""\
import f2pytest,sys
e = 'abcdefgh'
r = f2pytest.f()
if not e==r:
    sys.stderr.write('expected %s but got %s (o?k)\\n'%(`e`,`r`))
    sys.exit()
print 'ok'
"""
    tests.append(test)
################################################################
if 0 or all:
    for s in ['','*4','*8','*16']:
        if s=='*16' and skip: continue
        test={}
        #test['f2pyflags']=['--debug-capi']
        test['name']='Fortran function returning real%s'%s
        test['depends']=['fncall']
        test['f']="""\
      function f()
      real%s f
      f = 13.45
      end
"""%s
        test['py']="""\
import f2pytest,sys
e = 13.45
r = f2pytest.f()
if abs(r-e) > 1e-6:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
        break
################################################################
if 0 or all:
    for s in ['','*8','*16','*32']:
        if s=='*32' and skip: continue
        test={}
        #test['f2pyflags']=['--debug-capi']
        test['name']='Fortran function returning complex%s'%s
        test['depends']=['fncall']
        test['f']="""\
      function f()
      complex%s f
      f = (1,2)
      end
"""%s
        test['py']="""\
import f2pytest,sys
e = 1+2j
r = f2pytest.f()
if abs(r-e) > 1e-6:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
################################################################
if 0 or all:
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Fortran function returning integer%s'%s
        test['depends']=['fncall']
        test['f']="""\
      function f()
      integer%s f
      f = 2
      end
"""%s
        test['py']="""\
import f2pytest,sys
e = 2
r = f2pytest.f()
if not r==e:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Fortran function returning integer*8'
    test['depends']=['fncall']
    test['f']="""\
      function f()
      integer*8 f
      f = 20
      f = f * 222222222
      end
"""
    test['py']="""\
import f2pytest,sys
e = 20L*222222222L
r = f2pytest.f()
if not r==e:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
    tests.append(test)
################################################################
if 0 or all:
    for s in ['','*1','*2','*4','*8']:
        test={}
        test['name']='Fortran function returning logical%s'%s
        test['depends']=['fncall']
        #test['f2pyflags']=['--debug-capi']
        test['f']="""\
      function f()
      logical%s f
      f = .false.
      end
      function f2()
      logical%s f2
      f2 = .true.
      end
"""%(s,s)
        test['py']="""\
import f2pytest,sys
r = f2pytest.f()
if r:
    sys.stderr.write('expected .false. but got %s\\n'%(r))
r = f2pytest.f2()
if not r:
    sys.stderr.write('expected .true. but got %s\\n'%(r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
