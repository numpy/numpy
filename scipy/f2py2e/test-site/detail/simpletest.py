tests=[]
all=1
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
    test['id']='call'
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
    test['id']='fncall'
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

################################################################

if 0 or all:
    test={}
    test['name']='Simple callback from Fortran'
    test['id']='cb'
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

