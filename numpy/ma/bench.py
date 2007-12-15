#! python

import timeit
#import IPython.ipapi
#ip = IPython.ipapi.get()
#from IPython import ipmagic
import numpy
import maskedarray
from maskedarray import filled
from maskedarray.testutils import assert_equal


#####---------------------------------------------------------------------------
#---- --- Global variables ---
#####---------------------------------------------------------------------------

# Small arrays ..................................
xs = numpy.random.uniform(-1,1,6).reshape(2,3)
ys = numpy.random.uniform(-1,1,6).reshape(2,3)
zs = xs + 1j * ys
m1 = [[True, False, False], [False, False, True]]
m2 = [[True, False, True], [False, False, True]]
nmxs = numpy.ma.array(xs, mask=m1)
nmys = numpy.ma.array(ys, mask=m2)
nmzs = numpy.ma.array(zs, mask=m1)
mmxs = maskedarray.array(xs, mask=m1)
mmys = maskedarray.array(ys, mask=m2)
mmzs = maskedarray.array(zs, mask=m1)
# Big arrays ....................................
xl = numpy.random.uniform(-1,1,100*100).reshape(100,100)
yl = numpy.random.uniform(-1,1,100*100).reshape(100,100)
zl = xl + 1j * yl
maskx = xl > 0.8
masky = yl < -0.8
nmxl = numpy.ma.array(xl, mask=maskx)
nmyl = numpy.ma.array(yl, mask=masky)
nmzl = numpy.ma.array(zl, mask=maskx)
mmxl = maskedarray.array(xl, mask=maskx, shrink=True)
mmyl = maskedarray.array(yl, mask=masky, shrink=True)
mmzl = maskedarray.array(zl, mask=maskx, shrink=True)

#####---------------------------------------------------------------------------
#---- --- Functions ---
#####---------------------------------------------------------------------------

def timer(s, v='', nloop=500, nrep=3):
    units = ["s", "ms", "\xb5s", "ns"]
    scaling = [1, 1e3, 1e6, 1e9]
    print "%s : %-50s : " % (v,s),
    varnames = ["%ss,nm%ss,mm%ss,%sl,nm%sl,mm%sl" % tuple(x*6) for x in 'xyz']
    setup = 'from __main__ import numpy, maskedarray, %s' % ','.join(varnames)
    Timer = timeit.Timer(stmt=s, setup=setup)
    best = min(Timer.repeat(nrep, nloop)) / nloop
    if best > 0.0:
        order = min(-int(numpy.floor(numpy.log10(best)) // 3), 3)
    else:
        order = 3
    print "%d loops, best of %d: %.*g %s per loop" % (nloop, nrep,
                                                      3,
                                                      best * scaling[order],
                                                      units[order])
#    ip.magic('timeit -n%i %s' % (nloop,s))



def compare_functions_1v(func, nloop=500, test=True,
                       xs=xs, nmxs=nmxs, mmxs=mmxs,
                       xl=xl, nmxl=nmxl, mmxl=mmxl):
    funcname = func.__name__
    print "-"*50
    print "%s on small arrays" % funcname
    if test:
        assert_equal(filled(eval("numpy.ma.%s(nmxs)" % funcname),0),
                     filled(eval("maskedarray.%s(mmxs)" % funcname),0))
    for (module, data) in zip(("numpy", "numpy.ma","maskedarray"),
                              ("xs","nmxs","mmxs")):
        timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
    #
    print "%s on large arrays" % funcname
    if test:
        assert_equal(filled(eval("numpy.ma.%s(nmxl)" % funcname),0),
                     filled(eval("maskedarray.%s(mmxl)" % funcname),0))
    for (module, data) in zip(("numpy", "numpy.ma","maskedarray"),
                              ("xl","nmxl","mmxl")):
        timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
    return

def compare_methods(methodname, args, vars='x', nloop=500, test=True,
                    xs=xs, nmxs=nmxs, mmxs=mmxs,
                    xl=xl, nmxl=nmxl, mmxl=mmxl):
    print "-"*50
    print "%s on small arrays" % methodname
    if test:
        assert_equal(filled(eval("nm%ss.%s(%s)" % (vars,methodname,args)),0),
                     filled(eval("mm%ss.%s(%s)" % (vars,methodname,args)),0))
    for (data, ver) in zip(["nm%ss" % vars, "mm%ss" % vars], ('numpy.ma   ','maskedarray')):
        timer("%(data)s.%(methodname)s(%(args)s)" % locals(), v=ver, nloop=nloop)
    #
    print "%s on large arrays" % methodname
    if test:
        assert_equal(filled(eval("nm%sl.%s(%s)" % (vars,methodname,args)),0),
                     filled(eval("mm%sl.%s(%s)" % (vars,methodname,args)),0))
    for (data, ver) in zip(["nm%sl" % vars, "mm%sl" % vars], ('numpy.ma   ','maskedarray')):
        timer("%(data)s.%(methodname)s(%(args)s)" % locals(), v=ver, nloop=nloop)
    return

def compare_functions_2v(func, nloop=500, test=True,
                       xs=xs, nmxs=nmxs, mmxs=mmxs,
                       ys=ys, nmys=nmys, mmys=mmys,
                       xl=xl, nmxl=nmxl, mmxl=mmxl,
                       yl=yl, nmyl=nmyl, mmyl=mmyl):
    funcname = func.__name__
    print "-"*50
    print "%s on small arrays" % funcname
    if test:
        assert_equal(filled(eval("numpy.ma.%s(nmxs,nmys)" % funcname),0),
                     filled(eval("maskedarray.%s(mmxs,mmys)" % funcname),0))
    for (module, data) in zip(("numpy", "numpy.ma","maskedarray"),
                              ("xs,ys","nmxs,nmys","mmxs,mmys")):
        timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
    #
    print "%s on large arrays" % funcname
    if test:
        assert_equal(filled(eval("numpy.ma.%s(nmxl, nmyl)" % funcname),0),
                     filled(eval("maskedarray.%s(mmxl, mmyl)" % funcname),0))
    for (module, data) in zip(("numpy", "numpy.ma","maskedarray"),
                              ("xl,yl","nmxl,nmyl","mmxl,mmyl")):
        timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
    return


###############################################################################


################################################################################
if __name__ == '__main__':
#    # Small arrays ..................................
#    xs = numpy.random.uniform(-1,1,6).reshape(2,3)
#    ys = numpy.random.uniform(-1,1,6).reshape(2,3)
#    zs = xs + 1j * ys
#    m1 = [[True, False, False], [False, False, True]]
#    m2 = [[True, False, True], [False, False, True]]
#    nmxs = numpy.ma.array(xs, mask=m1)
#    nmys = numpy.ma.array(ys, mask=m2)
#    nmzs = numpy.ma.array(zs, mask=m1)
#    mmxs = maskedarray.array(xs, mask=m1)
#    mmys = maskedarray.array(ys, mask=m2)
#    mmzs = maskedarray.array(zs, mask=m1)
#    # Big arrays ....................................
#    xl = numpy.random.uniform(-1,1,100*100).reshape(100,100)
#    yl = numpy.random.uniform(-1,1,100*100).reshape(100,100)
#    zl = xl + 1j * yl
#    maskx = xl > 0.8
#    masky = yl < -0.8
#    nmxl = numpy.ma.array(xl, mask=maskx)
#    nmyl = numpy.ma.array(yl, mask=masky)
#    nmzl = numpy.ma.array(zl, mask=maskx)
#    mmxl = maskedarray.array(xl, mask=maskx, shrink=True)
#    mmyl = maskedarray.array(yl, mask=masky, shrink=True)
#    mmzl = maskedarray.array(zl, mask=maskx, shrink=True)
#
    compare_functions_1v(numpy.sin)
    compare_functions_1v(numpy.log)
    compare_functions_1v(numpy.sqrt)
    #....................................................................
    compare_functions_2v(numpy.multiply)
    compare_functions_2v(numpy.divide)
    compare_functions_2v(numpy.power)
    #....................................................................
    compare_methods('ravel','', nloop=1000)
    compare_methods('conjugate','','z', nloop=1000)
    compare_methods('transpose','', nloop=1000)
    compare_methods('compressed','', nloop=1000)
    compare_methods('__getitem__','0', nloop=1000)
    compare_methods('__getitem__','(0,0)', nloop=1000)
    compare_methods('__getitem__','[0,-1]', nloop=1000)
    compare_methods('__setitem__','0, 17', nloop=1000, test=False)
    compare_methods('__setitem__','(0,0), 17', nloop=1000, test=False)
    #....................................................................
    print "-"*50
    print "__setitem__ on small arrays"
    timer('nmxs.__setitem__((-1,0),numpy.ma.masked)', 'numpy.ma   ',nloop=10000)
    timer('mmxs.__setitem__((-1,0),maskedarray.masked)', 'maskedarray',nloop=10000)
    print "-"*50
    print "__setitem__ on large arrays"
    timer('nmxl.__setitem__((-1,0),numpy.ma.masked)', 'numpy.ma   ',nloop=10000)
    timer('mmxl.__setitem__((-1,0),maskedarray.masked)', 'maskedarray',nloop=10000)
    #....................................................................
    print "-"*50
    print "where on small arrays"
    assert_equal(eval("numpy.ma.where(nmxs>2,nmxs,nmys)"),
                 eval("maskedarray.where(mmxs>2, mmxs,mmys)"))
    timer('numpy.ma.where(nmxs>2,nmxs,nmys)', 'numpy.ma   ',nloop=1000)
    timer('maskedarray.where(mmxs>2, mmxs,mmys)', 'maskedarray',nloop=1000)
    print "-"*50
    print "where on large arrays"
    timer('numpy.ma.where(nmxl>2,nmxl,nmyl)', 'numpy.ma   ',nloop=100)
    timer('maskedarray.where(mmxl>2, mmxl,mmyl)', 'maskedarray',nloop=100)
