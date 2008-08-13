""" Test functions for 1D array set operations.

"""

from numpy.testing import *
import numpy as np
from numpy.lib.arraysetops import *

import warnings

class TestAso(TestCase):
    def test_unique1d( self ):
        a = np.array( [5, 7, 1, 2, 1, 5, 7] )

        ec = np.array( [1, 2, 5, 7] )
        c = unique1d( a )
        assert_array_equal( c, ec )

        warnings.simplefilter('ignore', Warning)
        unique, indices = unique1d( a, return_index=True )
        warnings.resetwarnings()

        ed = np.array( [2, 3, 0, 1] )
        assert_array_equal(unique, ec)
        assert_array_equal(indices, ed)

        assert_array_equal([], unique1d([]))

    def test_intersect1d( self ):
        a = np.array( [5, 7, 1, 2] )
        b = np.array( [2, 4, 3, 1, 5] )

        ec = np.array( [1, 2, 5] )
        c = intersect1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], intersect1d([],[]))

    def test_intersect1d_nu( self ):
        a = np.array( [5, 5, 7, 1, 2] )
        b = np.array( [2, 1, 4, 3, 3, 1, 5] )

        ec = np.array( [1, 2, 5] )
        c = intersect1d_nu( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], intersect1d_nu([],[]))

    def test_setxor1d( self ):
        a = np.array( [5, 7, 1, 2] )
        b = np.array( [2, 4, 3, 1, 5] )

        ec = np.array( [3, 4, 7] )
        c = setxor1d( a, b )
        assert_array_equal( c, ec )

        a = np.array( [1, 2, 3] )
        b = np.array( [6, 5, 4] )

        ec = np.array( [1, 2, 3, 4, 5, 6] )
        c = setxor1d( a, b )
        assert_array_equal( c, ec )

        a = np.array( [1, 8, 2, 3] )
        b = np.array( [6, 5, 4, 8] )

        ec = np.array( [1, 2, 3, 4, 5, 6] )
        c = setxor1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], setxor1d([],[]))

    def test_ediff1d(self):
        zero_elem = np.array([])
        one_elem = np.array([1])
        two_elem = np.array([1,2])

        assert_array_equal([],ediff1d(zero_elem))
        assert_array_equal([0],ediff1d(zero_elem,to_begin=0))
        assert_array_equal([0],ediff1d(zero_elem,to_end=0))
        assert_array_equal([-1,0],ediff1d(zero_elem,to_begin=-1,to_end=0))
        assert_array_equal([],ediff1d(one_elem))
        assert_array_equal([1],ediff1d(two_elem))

    def test_setmember1d( self ):
        a = np.array( [5, 7, 1, 2] )
        b = np.array( [2, 4, 3, 1, 5] )

        ec = np.array( [True, False, True, True] )
        c = setmember1d( a, b )
        assert_array_equal( c, ec )

        a[0] = 8
        ec = np.array( [False, False, True, True] )
        c = setmember1d( a, b )
        assert_array_equal( c, ec )

        a[0], a[3] = 4, 8
        ec = np.array( [True, False, True, False] )
        c = setmember1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], setmember1d([],[]))

    def test_union1d( self ):
        a = np.array( [5, 4, 7, 1, 2] )
        b = np.array( [2, 4, 3, 3, 2, 1, 5] )

        ec = np.array( [1, 2, 3, 4, 5, 7] )
        c = union1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], union1d([],[]))

    def test_setdiff1d( self ):
        a = np.array( [6, 5, 4, 7, 1, 2] )
        b = np.array( [2, 4, 3, 3, 2, 1, 5] )

        ec = np.array( [6, 7] )
        c = setdiff1d( a, b )
        assert_array_equal( c, ec )

        a = np.arange( 21 )
        b = np.arange( 19 )
        ec = np.array( [19, 20] )
        c = setdiff1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], setdiff1d([],[]))

    def test_setdiff1d_char_array(self):
        a = np.array(['a','b','c'])
        b = np.array(['a','b','s'])
        assert_array_equal(setdiff1d(a,b),np.array(['c']))

    def test_manyways( self ):
        nItem = 100
        a = np.fix( nItem / 10 * np.random.random( nItem ) )
        b = np.fix( nItem / 10 * np.random.random( nItem ) )

        c1 = intersect1d_nu( a, b )
        c2 = unique1d( intersect1d( a, b ) )
        assert_array_equal( c1, c2 )

        a = np.array( [5, 7, 1, 2, 8] )
        b = np.array( [9, 8, 2, 4, 3, 1, 5] )

        c1 = setxor1d( a, b )
        aux1 = intersect1d( a, b )
        aux2 = union1d( a, b )
        c2 = setdiff1d( aux2, aux1 )
        assert_array_equal( c1, c2 )


if __name__ == "__main__":
    run_module_suite()
