#! /usr/bin/env python

# System imports
from   distutils.util import get_platform
import os
import sys
import unittest

import numpy as N

# Add the distutils-generated build directory to the python search path and then
# import the extension module
libDir = "lib.%s-%s" % (get_platform(), sys.version[:3])
sys.path.insert(0,os.path.join("build", libDir))
import Series

######################################################################

class SeriesTestCase(unittest.TestCase):

    ########################################################
    ### Test functions that take 1D arrays of type SHORT ###
    def testShortSum(self):
        "Test the shortSum function"
        self.assertEquals(Series.shortSum([1,2,3,4]), 10)

    def testShortProd(self):
        "Test the shortProd function"
        self.assertEquals(Series.shortProd([1,2,3,4]), 24)

    ######################################################
    ### Test functions that take 1D arrays of type INT ###
    def testIntSum(self):
        "Test the intSum function"
        self.assertEquals(Series.intSum([1,2,3,4]), 10)

    def testIntSumBadContainer(self):
        "Test the intSum function with an invalid list"
        self.assertRaises(TypeError, Series.intSum, [1,2,"junk"])

    def testIntProd(self):
        "Test the intProd function"
        self.assertEquals(Series.intProd([1,2,3,4]), 24)

    def testIntProdNonContainer(self):
        "Test the intProd function with None"
        self.assertRaises(TypeError, Series.intProd, None)

    def testIntZeros(self):
        "Test the intZeros function"
        myArray = N.ones(5,'i')
        Series.intZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testIntZerosNonArray(self):
        "Test the intZeros function with an integer"
        self.assertRaises(TypeError, Series.intZeros, 5)

    def testIntOnes(self):
        "Test the intOnes function"
        myArray = N.zeros(5,'i')
        Series.intOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testIntNegate(self):
        "Test the intNegate function"
        myArray = N.arange(5,dtype='i')
        Series.intNegate(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,-1,-2,-3,-4]))

    #######################################################
    ### Test functions that take 1D arrays of type LONG ###
    def testLongSum(self):
        "Test the longSum function"
        self.assertEquals(Series.longSum([1,2,3,4]), 10)

    def testLongSumMultiDimensional(self):
        "Test the longSum function with multi-dimensional array"
        self.assertRaises(TypeError, Series.longSum, [[1,2,3],[9,8,7]])

    def testLongProd(self):
        "Test the longProd function"
        self.assertEquals(Series.longProd([1,2,3,4]), 24)

    ########################################################
    ### Test functions that take 1D arrays of type FLOAT ###
    def testFloatSum(self):
        "Test the floatSum function (to 5 decimal places)"
        self.assertAlmostEquals(Series.floatSum([1,2,3.14,4]), 10.14, 5)

    def testFloatSumBadContainer(self):
        "Test the floatSum function with a dictionary"
        self.assertRaises(TypeError, Series.floatSum, {"key":"value"})

    def testFloatProd(self):
        "Test the floatProd function (to 5 decimal places)"
        self.assertAlmostEquals(Series.floatProd((1,2.718,3,4)), 32.616, 5)

    def testFloatProdBadContainer(self):
        "Test the floatProd function with an invalid list"
        self.assertRaises(TypeError, Series.floatProd, [3.14, "pi"])

    #########################################################
    ### Test functions that take 1D arrays of type DOUBLE ###
    def testDoubleSum(self):
        "Test the doubleSum function"
        self.assertEquals(Series.doubleSum([1,2,3.14,4]), 10.14)

    def testDoubleSumNonContainer(self):
        "Test the doubleSum function with None"
        self.assertRaises(TypeError, Series.doubleSum, None)

    def testDoubleProd(self):
        "Test the doubleProd function"
        self.assertEquals(Series.doubleProd((1,2.718,3,4)), 32.616)

    def testDoubleProdBadContainer(self):
        "Test the doubleProd function with an invalid list"
        self.assertRaises(TypeError, Series.doubleProd, [3.14, "pi"])

    def testDoubleZeros(self):
        "Test the doubleZeros function"
        myArray = N.ones(5,'d')
        Series.doubleZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0.,0.,0.,0.,0.]))

    def testDoubleOnes(self):
        "Test the doubleOnes function"
        myArray = N.zeros(5,'d')
        Series.doubleOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    def testDoubleOnesNonArray(self):
        "Test the doubleOnes function with a list"
        self.assertRaises(TypeError, Series.doubleOnes, [True, 0, 2.718, "pi"])

    def testDoubleNegate(self):
        "Test the doubleNegate function"
        myArray = N.arange(5) * 1.0
        Series.doubleNegate(myArray)
        N.testing.assert_array_equal(myArray, N.array([0.,-1.,-2.,-3.,-4.]))

    #########################################################
    ### Test functions that take 2D arrays of type DOUBLE ###
    def testIntMax(self):
        "Test the intMax function"
        matrix = [[6,-5,4],[-3,2,-1]]
        self.assertEquals(Series.intMax(matrix), 6)

    def testIntMaxNonContainer(self):
        "Test the intMax function with None"
        self.assertRaises(TypeError, Series.intMax, None)

    def testIntFloor(self):
        "Test the intFloor function"
        matrix = N.array([[10,-2],[-6,7]])
        Series.intFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testDoubleMax(self):
        "Test the doubleMax function"
        matrix = [[-6,5,-4],[3.14,-2.718,1]]
        self.assertEquals(Series.doubleMax(matrix), 5.0)

    def testDoubleMaxWrongDim(self):
        "Test the doubleMax function with a 1D array"
        self.assertRaises(TypeError, Series.doubleMax, [0.0, -1, 2.718, -3.14])

    def testDoubleFloor(self):
        "Test the doubleFloor function"
        matrix = N.array([[10,-2.718],[-6,3.14]])
        Series.doubleFloor(matrix,5.0)
        N.testing.assert_array_equal(matrix, N.array([[10.0,0],[0,0]]))

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SeriesTestCase))

    # Execute the test suite
    print "Testing Module Series\n"
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
