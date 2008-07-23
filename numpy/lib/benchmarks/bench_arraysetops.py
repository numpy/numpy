import numpy as np
import time
from numpy.lib.arraysetops import *

def bench_unique1d( plot_results = False ):
    exponents = np.linspace( 2, 7, 9 )
    ratios = []
    nItems = []
    dt1s = []
    dt2s = []
    for ii in exponents:

        nItem = 10 ** ii
        print 'using %d items:' % nItem
        a = np.fix( nItem / 10 * np.random.random( nItem ) )

        print 'unique:'
        tt = time.clock()
        b = np.unique( a )
        dt1 = time.clock() - tt
        print dt1

        print 'unique1d:'
        tt = time.clock()
        c = unique1d( a )
        dt2 = time.clock() - tt
        print dt2


        if dt1 < 1e-8:
            ratio = 'ND'
        else:
            ratio = dt2 / dt1
        print 'ratio:', ratio
        print 'nUnique: %d == %d\n' % (len( b ), len( c ))

        nItems.append( nItem )
        ratios.append( ratio )
        dt1s.append( dt1 )
        dt2s.append( dt2 )

        assert np.alltrue( b == c )

    print nItems
    print dt1s
    print dt2s
    print ratios

    if plot_results:
        import pylab

        def plotMe( fig, fun, nItems, dt1s, dt2s ):
            pylab.figure( fig )
            fun( nItems, dt1s, 'g-o', linewidth = 2, markersize = 8 )
            fun( nItems, dt2s, 'b-x', linewidth = 2, markersize = 8 )
            pylab.legend( ('unique', 'unique1d' ) )
            pylab.xlabel( 'nItem' )
            pylab.ylabel( 'time [s]' )

        plotMe( 1, pylab.loglog, nItems, dt1s, dt2s )
        plotMe( 2, pylab.plot, nItems, dt1s, dt2s )
        pylab.show()

if __name__ == '__main__':
    bench_unique1d( plot_results = True )
