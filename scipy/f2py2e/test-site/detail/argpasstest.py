#!/usr/bin/env python
"""

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2000/09/10 12:35:44 $
Pearu Peterson
"""

__version__ = "$Revision: 1.2 $"[10:-1]

tests=[]
all = 1
skip = 1
################################################################
if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(character)'
    test['depends']=['fncall']
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

if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(character*9)'
    test['depends']=['fncall']
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
    test['depends']=['fncall']
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
        test['depends']=['fncall']
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
    test['depends']=['fncall']
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
        test['depends']=['fncall']
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
        test['depends']=['fncall']
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
        test['depends']=['fncall']
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
