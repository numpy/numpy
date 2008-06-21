""" Test functions for 1D array set operations.

"""

from numpy.testing import *
set_package_path()
import numpy
from numpy.lib.arraysetops import *
restore_path()

##################################################

class TestAso(TestCase):
    ##
    # 03.11.2005, c
    def test_unique1d( self ):

        a = numpy.array( [5, 7, 1, 2, 1, 5, 7] )

        ec = numpy.array( [1, 2, 5, 7] )
        c = unique1d( a )
        assert_array_equal( c, ec )

        assert_array_equal([], unique1d([]))

    ##
    # 03.11.2005, c
    def test_intersect1d( self ):

        a = numpy.array( [5, 7, 1, 2] )
        b = numpy.array( [2, 4, 3, 1, 5] )

        ec = numpy.array( [1, 2, 5] )
        c = intersect1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], intersect1d([],[]))

    ##
    # 03.11.2005, c
    def test_intersect1d_nu( self ):

        a = numpy.array( [5, 5, 7, 1, 2] )
        b = numpy.array( [2, 1, 4, 3, 3, 1, 5] )

        ec = numpy.array( [1, 2, 5] )
        c = intersect1d_nu( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], intersect1d_nu([],[]))

    ##
    # 03.11.2005, c
    def test_setxor1d( self ):

        a = numpy.array( [5, 7, 1, 2] )
        b = numpy.array( [2, 4, 3, 1, 5] )

        ec = numpy.array( [3, 4, 7] )
        c = setxor1d( a, b )
        assert_array_equal( c, ec )

        a = numpy.array( [1, 2, 3] )
        b = numpy.array( [6, 5, 4] )

        ec = numpy.array( [1, 2, 3, 4, 5, 6] )
        c = setxor1d( a, b )
        assert_array_equal( c, ec )

        a = numpy.array( [1, 8, 2, 3] )
        b = numpy.array( [6, 5, 4, 8] )

        ec = numpy.array( [1, 2, 3, 4, 5, 6] )
        c = setxor1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], setxor1d([],[]))

    def test_ediff1d(self):
        zero_elem = numpy.array([])
        one_elem = numpy.array([1])
        two_elem = numpy.array([1,2])

        assert_array_equal([],ediff1d(zero_elem))
        assert_array_equal([0],ediff1d(zero_elem,to_begin=0))
        assert_array_equal([0],ediff1d(zero_elem,to_end=0))
        assert_array_equal([-1,0],ediff1d(zero_elem,to_begin=-1,to_end=0))
        assert_array_equal([],ediff1d(one_elem))
        assert_array_equal([1],ediff1d(two_elem))

    ##
    # 03.11.2005, c
    def test_setmember1d( self ):

        a = numpy.array( [5, 7, 1, 2] )
        b = numpy.array( [2, 4, 3, 1, 5] )

        ec = numpy.array( [True, False, True, True] )
        c = setmember1d( a, b )
        assert_array_equal( c, ec )

        a[0] = 8
        ec = numpy.array( [False, False, True, True] )
        c = setmember1d( a, b )
        assert_array_equal( c, ec )

        a[0], a[3] = 4, 8
        ec = numpy.array( [True, False, True, False] )
        c = setmember1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], setmember1d([],[]))

    ##
    # 03.11.2005, c
    def test_union1d( self ):

        a = numpy.array( [5, 4, 7, 1, 2] )
        b = numpy.array( [2, 4, 3, 3, 2, 1, 5] )

        ec = numpy.array( [1, 2, 3, 4, 5, 7] )
        c = union1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], union1d([],[]))

    ##
    # 03.11.2005, c
    # 09.01.2006
    def test_setdiff1d( self ):

        a = numpy.array( [6, 5, 4, 7, 1, 2] )
        b = numpy.array( [2, 4, 3, 3, 2, 1, 5] )

        ec = numpy.array( [6, 7] )
        c = setdiff1d( a, b )
        assert_array_equal( c, ec )

        a = numpy.arange( 21 )
        b = numpy.arange( 19 )
        ec = numpy.array( [19, 20] )
        c = setdiff1d( a, b )
        assert_array_equal( c, ec )

        assert_array_equal([], setdiff1d([],[]))

    def test_setdiff1d_char_array(self):
        a = numpy.array(['a','b','c'])
        b = numpy.array(['a','b','s'])
        assert_array_equal(setdiff1d(a,b),numpy.array(['c']))

    ##
    # 03.11.2005, c
    def test_manyways( self ):

        nItem = 100
        a = numpy.fix( nItem / 10 * numpy.random.random( nItem ) )
        b = numpy.fix( nItem / 10 * numpy.random.random( nItem ) )

        c1 = intersect1d_nu( a, b )
        c2 = unique1d( intersect1d( a, b ) )
        assert_array_equal( c1, c2 )

        a = numpy.array( [5, 7, 1, 2, 8] )
        b = numpy.array( [9, 8, 2, 4, 3, 1, 5] )

        c1 = setxor1d( a, b )
        aux1 = intersect1d( a, b )
        aux2 = union1d( a, b )
        c2 = setdiff1d( aux2, aux1 )
        assert_array_equal( c1, c2 )


if __name__ == "__main__":
    run_module_suite()
