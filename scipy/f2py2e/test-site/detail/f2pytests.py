#!/usr/bin/env python
"""

This file contains the following tests:

    Argument passing to Fortran function(<typespec>)
    Fortran function returning <typespec>
    Simple callback from Fortran
    Callback function returning <typespec>

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2000/05/01 17:10:43 $
Pearu Peterson
"""

__version__ = "$Revision: 1.12 $"[10:-1]


tests=[]
all=1 # run all tests
skip=1
#################################################################
if 0: #Template
    test={}
    test['name']=''
    test['f']="""\
      function f()
      end
"""
    test['py']="""\
import f2pytest,sys
e = # expected
r = f2pytest.f()
if abs(r-e) > 1e-4:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Trivial call to Fortran subroutine'
    test['f']="""\
      subroutine f()
      end
"""
    test['py']="""\
import f2pytest,sys
f2pytest.f()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Trivial call to Fortran function'
    test['f']="""\
      integer function f()
      f = 3
      end
"""
    test['py']="""\
import f2pytest,sys
r = f2pytest.f()
if r==3:
    print 'ok'
else:
    sys.stderr.write('expected 3 but got %s'%r)
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(character)'
    test['f']="""\
      function f(a)
      integer f
      character a
      if (a .eq. 'w') then
          f = 3
      else
          write(*,*) "Fortran: expected 'w' but got '",a,"'"
          f = 4
      end if
      end
"""
    test['py']="""\
import f2pytest,sys
r = f2pytest.f('w')
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
    tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(character*9)'
    test['f']="""\
      function f(a)
      integer f
      character*9 a
      if (a .eq. 'abcdefgh ') then
          f = 3
      else
          write(*,*) "Fortran: expected 'abcdefgh ' but got '",a,"'"
          f = 4
      end if
      end
"""
    test['py']="""\
import f2pytest,sys
r = f2pytest.f('abcdefgh ')
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
    tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(character*(*))'
    test['f']="""\
      function f(a)
      integer f
      character*(*) a
      if (a .eq. 'abcdef5') then
          f = 3
      else
          write(*,*) "Fortran: expected 'abcdef5' but got '",a,"'"
          f = 4
      end if
      end
"""
    test['py']="""\
import f2pytest,sys
r = f2pytest.f('abcdef5')
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
    tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Argument passing to Fortran function(integer%s)'%s
        test['f']="""\
      function f(a)
      integer f
      integer%s a
      if (a .eq. 34) then
          f = 3
      else
          write(*,*) "Fortran: expected 34 but got",a
          f = 4
      end if
      end
"""%s
        test['py']="""\
import f2pytest,sys
r = f2pytest.f(34)
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
        tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(integer*8)'
    test['f']="""\
      function f(a)
      integer f
      integer*8 a,e
      e = 20
      e = e*222222222
      if (a .eq. e) then
          f = 3
      else
          write(*,*) "Fortran: expected ",e," but got",a
          f = 4
      end if
      end
"""
    test['py']="""\
import f2pytest,sys
r = f2pytest.f(20L*222222222L)
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
    tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*4','*8','*16']:
        if s=='*16' and skip: continue
        test={}
        test['name']='Argument passing to Fortran function(real%s)'%s
        test['f']="""\
      function f(a)
      integer f
      real%s a,e
      e = abs(a-34.56)
      if (e .lt. 1e-5) then
          f = 3
      else
          write(*,*) "Fortran: expected 34.56 but got",a
          f = 4
      end if
      end
"""%s
        test['py']="""\
import f2pytest,sys
r = f2pytest.f(34.56)
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
        tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*8','*16','*32']:
        if s=='*32' and skip: continue
        test={}
        test['name']='Argument passing to Fortran function(complex%s)'%s
        test['f']="""\
      function f(a)
      integer f
      complex%s a
      real*8 e
      e = abs(a-(1,2))
      if (e .lt. 1e-5) then
          f = 3
      else
          write(*,*) "Fortran: expected (1.,2.) but got",a
          f = 4
      end if
      end
"""%s
        test['py']="""\
import f2pytest,sys
r = f2pytest.f(1+2j)
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
        tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4','*8']:
        if s=='*8' and skip: continue
        test={}
        test['name']='Argument passing to Fortran function(logical%s)'%s
        test['f']="""\
      function f(a)
      integer f
      logical%s a
      if (a) then
          f = 3
      else
          write(*,*) "Fortran: expected .true. but got",a
          f = 4
      end if
      end
      function f2(a)
      integer f2
      logical%s a
      if (a) then
          write(*,*) "Fortran: expected .false. but got",a
          f2 = 4
      else
          f2 = 3
      end if
      end
"""%(s,s)
        test['py']="""\
import f2pytest,sys
r = f2pytest.f(1)
if r==4:
    sys.stderr.write('incorrect value received')
    sys.exit()
elif not r==3:
    sys.stderr.write('incorrect return value')
    sys.exit()
r = f2pytest.f2(0)
if r==4:
    sys.stderr.write('incorrect value received')
    sys.exit()
elif not r==3:
    sys.stderr.write('incorrect return value')
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*1']:
        test={}
        test['name']='Fortran function returning character%s'%s
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
    sys.stderr.write('expected %s but got %s (ok)\\n'%(`e`,`r`))
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
################################################################
if 0 or all:
    test={}
    test['name']='Simple callback from Fortran'
    test['f']="""\
      subroutine f(g)
      external g
      call g()
      end
"""
    test['py']="""\
import f2pytest,sys
r = 3
def g():
    global r
    r = 4
f2pytest.f(g)
if not r==4:
    sys.stderr.write('expected 4 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Callback function returning integer%s'%s
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
if 0 or (all and not skip):
    test={}
    test['name']='Callback function returning character*(*)'
    test['f']="""\
      function f(g)
      character*(*) f,g
      external g
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
