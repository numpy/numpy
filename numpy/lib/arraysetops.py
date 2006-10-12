"""
Set operations for 1D numeric arrays based on sort() function.

Contains:
  ediff1d,
  unique1d,
  intersect1d,
  intersect1d_nu,
  setxor1d,
  setmember1d,
  union1d,
  setdiff1d

All functions work best with integer numerical arrays on input
(e.g. indices). For floating point arrays, innacurate results may appear due to
usual round-off and floating point comparison issues.

Except unique1d, union1d and intersect1d_nu, all functions expect inputs with
unique elements. Speed could be gained in some operations by an implementaion
of sort(), that can provide directly the permutation vectors, avoiding thus
calls to argsort().

Run test_unique1d_speed() to compare performance of numpy.unique1d() and
numpy.unique() - it should be the same.

To do: Optionally return indices analogously to unique1d for all functions.

Author: Robert Cimrman

created:       01.11.2005
last revision: 12.10.2006
"""
__all__ = ['ediff1d', 'unique1d', 'intersect1d', 'intersect1d_nu', 'setxor1d',
           'setmember1d', 'union1d', 'setdiff1d']

import time
import numpy as nm

def ediff1d(ary, to_end = None, to_begin = None):
    """Array difference with prefixed and/or appended value.

    See also: unique1d, intersect1d, intersect1d_nu, setxor1d,
    setmember1d, union1d, setdiff1d
    """
    ary = nm.asarray(ary).flat
    ed = ary[1:] - ary[:-1]
    if to_begin is not None:
        if to_end is not None:
            ed = nm.r_[to_begin, ed, to_end]
        else:
            ed = nm.insert(ed, 0, to_begin)
    elif to_end is not None:
        ed = nm.append(ed, to_end)
        
    return ed

def unique1d(ar1, return_index=False):
    """Unique elements of 1D array. When return_index is True, return
    also the indices indx such that ar1.flat[indx] is the resulting
    array of unique elements.
    
    See also: ediff1d, intersect1d, intersect1d_nu, setxor1d,
    setmember1d, union1d, setdiff1d
    """
    ar = nm.asarray(ar1).flatten()
    if ar.size == 0:
        if return_index: return nm.empty(0, nm.bool), ar
        else: return ar
    
    if return_index:
        perm = ar.argsort()
        aux = ar[perm]
        flag = nm.concatenate( ([True], aux[1:] != aux[:-1]) )
        return perm[flag], aux[flag]
    
    else:
        ar.sort()
        flag = nm.concatenate( ([True], ar[1:] != ar[:-1]) )
        return ar[flag]

def intersect1d( ar1, ar2 ):
    """Intersection of 1D arrays with unique elements.

    See also: ediff1d, unique1d, intersect1d_nu, setxor1d,
    setmember1d, union1d, setdiff1d
    """
    aux = nm.concatenate((ar1,ar2))
    aux.sort()
    return aux[aux[1:] == aux[:-1]]

def intersect1d_nu( ar1, ar2 ):
    """Intersection of 1D arrays with any elements.

    See also: ediff1d, unique1d, intersect1d, setxor1d,
    setmember1d, union1d, setdiff1d
    """
    # Might be faster then unique1d( intersect1d( ar1, ar2 ) )?
    aux = nm.concatenate((unique1d(ar1), unique1d(ar2)))
    aux.sort()
    return aux[aux[1:] == aux[:-1]]

def setxor1d( ar1, ar2 ):
    """Set exclusive-or of 1D arrays with unique elements.

    See also: ediff1d, unique1d, intersect1d, intersect1d_nu,
    setmember1d, union1d, setdiff1d
    """
    aux = nm.concatenate((ar1, ar2))
    if aux.size == 0:
        return aux
    
    aux.sort()
#    flag = ediff1d( aux, to_end = 1, to_begin = 1 ) == 0
    flag = nm.concatenate( ([True], aux[1:] != aux[:-1], [True] ) )
#    flag2 = ediff1d( flag ) == 0
    flag2 = flag[1:] == flag[:-1]
    return aux[flag2]

def setmember1d( ar1, ar2 ):
    """Return an array of shape of ar1 containing 1 where the elements of
    ar1 are in ar2 and 0 otherwise.

    See also: ediff1d, unique1d, intersect1d, intersect1d_nu, setxor1d,
    union1d, setdiff1d
    """
    zlike = nm.zeros_like
    ar = nm.concatenate( (ar1, ar2 ) )
    tt = nm.concatenate( (zlike( ar1 ), zlike( ar2 ) + 1) )
    perm = ar.argsort()
    aux = ar[perm]
    aux2 = tt[perm]
#    flag = ediff1d( aux, 1 ) == 0
    flag = nm.concatenate( (aux[1:] == aux[:-1], [False] ) )

    ii = nm.where( flag * aux2 )[0]
    aux = perm[ii+1]
    perm[ii+1] = perm[ii]
    perm[ii] = aux

    indx = perm.argsort()[:len( ar1 )]

    return flag[indx]

def union1d( ar1, ar2 ):
    """Union of 1D arrays with unique elements.

    See also: ediff1d, unique1d, intersect1d, intersect1d_nu, setxor1d,
    setmember1d, setdiff1d
    """
    return unique1d( nm.concatenate( (ar1, ar2) ) )

def setdiff1d( ar1, ar2 ):
    """Set difference of 1D arrays with unique elements.

    See also: ediff1d, unique1d, intersect1d, intersect1d_nu, setxor1d,
    setmember1d, union1d
    """
    aux = setmember1d(ar1,ar2)
    if aux.size == 0:
        return aux
    else:
        return nm.asarray(ar1)[aux == 0]

def test_unique1d_speed( plot_results = False ):
#    exponents = nm.linspace( 2, 7, 9 )
    exponents = nm.linspace( 2, 7, 9 )
    ratios = []
    nItems = []
    dt1s = []
    dt2s = []
    for ii in exponents:

        nItem = 10 ** ii
        print 'using %d items:' % nItem
        a = nm.fix( nItem / 10 * nm.random.random( nItem ) )

        print 'unique:'
        tt = time.clock()
        b = nm.unique( a )
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

        assert nm.alltrue( b == c )

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

if (__name__ == '__main__'):
    test_unique1d_speed( plot_results = True )
