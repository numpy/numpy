#!/usr/bin/env python

import sys,os
import Numeric,string,f2py2e
from scipy_distutils.core import setup,Extension

if __name__ == "__main__":
    __file__ = sys.argv[0]

failure_str = 'FAILURE:'+os.path.abspath(__file__)+':\n\texpected %s but got %s'

define_macros=[('DMALLOC',None)]
libraries=['dmalloc']
define_macros=[]
libraries=[]

def check(result,expected):
    if result==expected:
        return 1
    print failure_str % (expected,result)

def main():
    #return test_callback() and test_array()
    return test_array() and test_callback() and test_marray() \
           and test_scalar() and test_complex()

def test_array():
    print 'test_array'

    # Creating signature file
    if not os.path.exists('fooa.pyf'):
        f2py2e.run_main(string.split('-m fooa -h fooa.pyf arraytest.f'))

    # Build extension module
    ext = Extension('fooa',sources=['fooa.pyf','arraytest.f'],
                    define_macros=define_macros,
                    libraries=libraries,)
    setup(ext_modules = [ext])

    # Import extension module
    import fooa

    # See doc string
    print fooa.foo.__doc__
    
    # Let's try it
    x=Numeric.array(range(-10,10,1),"i")
    r = fooa.foo(x)
    
    return check(r,9.0) and check(x[0],-10)

def test_callback():
    print 'test_callback'
    if not os.path.exists('foocb.pyf'):
        f2py2e.run_main(string.split('callbacktest.f -m foocb -h foocb.pyf'))
    ext = Extension('foocb',sources=['foocb.pyf','callbacktest.f'],
                    define_macros=define_macros,
                    libraries=libraries,
                    )
    setup(ext_modules = [ext])
    import foocb
    for f in dir(foocb):
        if not f[0]=="_":
            print eval("foocb.%s.__doc__"%f)
    def fun(x):
        print "In fun: x",x
        x[:]=x*x
        return 7j
    x=Numeric.array([-1,-2,-3],"d")
    n=Numeric.array(len(x),"i")
    print "x=",x
    print foocb.foo(fun,x,n)
    print "x=",x
    print "n=",n
    def fun(x,e):
        print "In fun: x,e=",x,e
        x[:]=x*x
        return 7j
    print "x=",x
    res = foocb.foo(fun,x,f_extra_args=(57,))
    print res
    print "x=",x
    print "n=",n
    print "ok"
    return check(n,3) and check(x[1],16.0) and check(res,7j)

def test_marray():
    print 'test_marray'
    if not os.path.exists('foom.pyf'):
        f2py2e.run_main(string.split('marraytest.f -m foom -h foom.pyf'))
    ext = Extension('foom',sources=['marraytest.f','foom.pyf'],
                    define_macros=define_macros,
                    libraries=libraries,)
    setup(ext_modules = [ext])
    import foom
    for f in dir(foom):
        if not f[0]=="_":
            print eval("foom.%s.__doc__"%f)
    x=Numeric.transpose(Numeric.array([[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5]],"d"))
    #x=0+x # this would make x contiguous but we don't want that.
    print "x=",x,Numeric.asarray(x).shape,x.iscontiguous()
    r,x1 = foom.foo(x)
    print "x1",x1
    print "x=",x # note that because transpose(x).iscontiguous() and with
                 # a proper type, it gets changed in situ.
    return check(x[2][0],0.0) and check(x[2][1],-3.0)

def test_scalar():
    print 'test_scalar'
    if not os.path.exists('foo.pyf'):
        f2py2e.run_main(string.split('scalar.f -m foo -h foo.pyf'))
    ext = Extension('foo',sources=['scalar.f','foo.pyf'],
                    define_macros=define_macros,
                    libraries=libraries,)
    setup(ext_modules = [ext])
    import foo
    for f in dir(foo):
        if not f[0]=="_":
            print eval("foo.%s.__doc__"%f)
    x=Numeric.array(5,"i")
    print x,foo.dfoo(1,2,x),x
    x=Numeric.array([3.0j,5])
    print x,foo.cfoo(1+2j,2+7j,x),x
    x=Numeric.array(3)
    print x
    r1 = foo.ifoo(1,2,x)
    print foo.sfoo("1111","222222222","3333")
    return check(r1,6) and check(x,3)

def test_complex():
    print 'test_complex'
    if not os.path.exists('ctest.pyf'):
        f2py2e.run_main(string.split('complextest.f -m ctest -h ctest.pyf'))
    ext = Extension('ctest',sources=['complextest.f','ctest.pyf'],
                    define_macros=define_macros,
                    libraries=libraries,)
    setup(ext_modules = [ext])
    def f(c):
        print "In Python f: c=",c
        return
    def g():
        c=5+7j
        print "In Python g: returning c=",c
        return c
    import ctest
    print ctest.foo.__doc__
    print ctest.foo(f,g)
    return 1

if __name__ == "__main__":
    sys.argv.extend(string.split('build --build-platlib .'))
    main()
