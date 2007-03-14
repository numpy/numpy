#! /usr/bin/env python

# System imports
from   distutils.util import get_platform
import os
import sys
import unittest

# Import NumPy
import numpy as N
major, minor = [ int(d) for d in N.__version__.split(".")[:2] ]
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

# Add the distutils-generated build directory to the python search path and then
# import the extension module
libDir = "lib.%s-%s" % (get_platform(), sys.version[:3])
sys.path.insert(0,os.path.join("build", libDir))
import Series

######################################################################

class SeriesTestCase(unittest.TestCase):

    ####################################################
    ### Test functions that take arrays of type BYTE ###
    def testScharProd(self):
        "Test the scharProd function"
        self.assertEquals(Series.scharProd([1,2,3,4]), 24)

    def testScharProdNonContainer(self):
        "Test the scharProd function with None"
        self.assertRaises(TypeError, Series.scharProd, None)

    def testScharOnes(self):
        "Test the scharOnes function"
        myArray = N.zeros(5,'b')
        Series.scharOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testScharMax(self):
        "Test the scharMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.scharMax(matrix), 5)

    def testScharMaxNonContainer(self):
        "Test the scharMax function with None"
        self.assertRaises(TypeError, Series.scharMax, None)

    def testScharMaxWrongDim(self):
        "Test the scharMax function with a 1D array"
        self.assertRaises(TypeError, Series.scharMax, [0, -1, 2, -3])

    def testScharFloor(self):
        "Test the scharFloor function"
        matrix = N.array([[10,-2],[-6,7]],'b')
        Series.scharFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    #####################################################
    ### Test functions that take arrays of type UBYTE ###
    def testUcharProd(self):
        "Test the ucharProd function"
        self.assertEquals(Series.ucharProd([1,2,3,4]), 24)

    def testUcharProdNonContainer(self):
        "Test the ucharProd function with None"
        self.assertRaises(TypeError, Series.ucharProd, None)

    def testUcharOnes(self):
        "Test the ucharOnes function"
        myArray = N.zeros(5,'B')
        Series.ucharOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUcharMax(self):
        "Test the ucharMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ucharMax(matrix), 6)

    def testUcharMaxNonContainer(self):
        "Test the ucharMax function with None"
        self.assertRaises(TypeError, Series.ucharMax, None)

    def testUcharMaxWrongDim(self):
        "Test the ucharMax function with a 1D array"
        self.assertRaises(TypeError, Series.ucharMax, [0, 1, 2, 3])

    def testUcharFloor(self):
        "Test the ucharFloor function"
        matrix = N.array([[10,2],[6,7]],'B')
        Series.ucharFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    #####################################################
    ### Test functions that take arrays of type SHORT ###
    def testShortProd(self):
        "Test the shortProd function"
        self.assertEquals(Series.shortProd([1,2,3,4]), 24)

    def testShortProdNonContainer(self):
        "Test the shortProd function with None"
        self.assertRaises(TypeError, Series.shortProd, None)

    def testShortOnes(self):
        "Test the shortOnes function"
        myArray = N.zeros(5,'h')
        Series.shortOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testShortMax(self):
        "Test the shortMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.shortMax(matrix), 5)

    def testShortMaxNonContainer(self):
        "Test the shortMax function with None"
        self.assertRaises(TypeError, Series.shortMax, None)

    def testShortMaxWrongDim(self):
        "Test the shortMax function with a 1D array"
        self.assertRaises(TypeError, Series.shortMax, [0, -1, 2, -3])

    def testShortFloor(self):
        "Test the shortFloor function"
        matrix = N.array([[10,-2],[-6,7]],'h')
        Series.shortFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    ######################################################
    ### Test functions that take arrays of type USHORT ###
    def testUshortProd(self):
        "Test the ushortProd function"
        self.assertEquals(Series.ushortProd([1,2,3,4]), 24)

    def testUshortProdNonContainer(self):
        "Test the ushortProd function with None"
        self.assertRaises(TypeError, Series.ushortProd, None)

    def testUshortOnes(self):
        "Test the ushortOnes function"
        myArray = N.zeros(5,'H')
        Series.ushortOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUshortMax(self):
        "Test the ushortMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ushortMax(matrix), 6)

    def testUshortMaxNonContainer(self):
        "Test the ushortMax function with None"
        self.assertRaises(TypeError, Series.ushortMax, None)

    def testUshortMaxWrongDim(self):
        "Test the ushortMax function with a 1D array"
        self.assertRaises(TypeError, Series.ushortMax, [0, 1, 2, 3])

    def testUshortFloor(self):
        "Test the ushortFloor function"
        matrix = N.array([[10,2],[6,7]],'H')
        Series.ushortFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    ###################################################
    ### Test functions that take arrays of type INT ###
    def testIntProd(self):
        "Test the intProd function"
        self.assertEquals(Series.intProd([1,2,3,4]), 24)

    def testIntProdNonContainer(self):
        "Test the intProd function with None"
        self.assertRaises(TypeError, Series.intProd, None)

    def testIntOnes(self):
        "Test the intOnes function"
        myArray = N.zeros(5,'i')
        Series.intOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testIntMax(self):
        "Test the intMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.intMax(matrix), 5)

    def testIntMaxNonContainer(self):
        "Test the intMax function with None"
        self.assertRaises(TypeError, Series.intMax, None)

    def testIntMaxWrongDim(self):
        "Test the intMax function with a 1D array"
        self.assertRaises(TypeError, Series.intMax, [0, -1, 2, -3])

    def testIntFloor(self):
        "Test the intFloor function"
        matrix = N.array([[10,-2],[-6,7]],'i')
        Series.intFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    ####################################################
    ### Test functions that take arrays of type UINT ###
    def testUintProd(self):
        "Test the uintProd function"
        self.assertEquals(Series.uintProd([1,2,3,4]), 24)

    def testUintProdNonContainer(self):
        "Test the uintProd function with None"
        self.assertRaises(TypeError, Series.uintProd, None)

    def testUintOnes(self):
        "Test the uintOnes function"
        myArray = N.zeros(5,'I')
        Series.uintOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUintMax(self):
        "Test the uintMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.uintMax(matrix), 6)

    def testUintMaxNonContainer(self):
        "Test the uintMax function with None"
        self.assertRaises(TypeError, Series.uintMax, None)

    def testUintMaxWrongDim(self):
        "Test the uintMax function with a 1D array"
        self.assertRaises(TypeError, Series.uintMax, [0, 1, 2, 3])

    def testUintFloor(self):
        "Test the uintFloor function"
        matrix = N.array([[10,2],[6,7]],'I')
        Series.uintFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    ####################################################
    ### Test functions that take arrays of type LONG ###
    def testLongProd(self):
        "Test the longProd function"
        self.assertEquals(Series.longProd([1,2,3,4]), 24)

    def testLongProdNonContainer(self):
        "Test the longProd function with None"
        self.assertRaises(TypeError, Series.longProd, None)

    def testLongOnes(self):
        "Test the longOnes function"
        myArray = N.zeros(5,'l')
        Series.longOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testLongMax(self):
        "Test the longMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.longMax(matrix), 5)

    def testLongMaxNonContainer(self):
        "Test the longMax function with None"
        self.assertRaises(TypeError, Series.longMax, None)

    def testLongMaxWrongDim(self):
        "Test the longMax function with a 1D array"
        self.assertRaises(TypeError, Series.longMax, [0, -1, 2, -3])

    def testLongFloor(self):
        "Test the longFloor function"
        matrix = N.array([[10,-2],[-6,7]],'l')
        Series.longFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    #####################################################
    ### Test functions that take arrays of type ULONG ###
    def testUlongProd(self):
        "Test the ulongProd function"
        self.assertEquals(Series.ulongProd([1,2,3,4]), 24)

    def testUlongProdNonContainer(self):
        "Test the ulongProd function with None"
        self.assertRaises(TypeError, Series.ulongProd, None)

    def testUlongOnes(self):
        "Test the ulongOnes function"
        myArray = N.zeros(5,'L')
        Series.ulongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUlongMax(self):
        "Test the ulongMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ulongMax(matrix), 6)

    def testUlongMaxNonContainer(self):
        "Test the ulongMax function with None"
        self.assertRaises(TypeError, Series.ulongMax, None)

    def testUlongMaxWrongDim(self):
        "Test the ulongMax function with a 1D array"
        self.assertRaises(TypeError, Series.ulongMax, [0, 1, 2, 3])

    def testUlongFloor(self):
        "Test the ulongFloor function"
        matrix = N.array([[10,2],[6,7]],'L')
        Series.ulongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    ########################################################
    ### Test functions that take arrays of type LONGLONG ###
    def testLongLongProd(self):
        "Test the longLongProd function"
        self.assertEquals(Series.longLongProd([1,2,3,4]), 24)

    def testLongLongProdNonContainer(self):
        "Test the longLongProd function with None"
        self.assertRaises(TypeError, Series.longLongProd, None)

    def testLongLongOnes(self):
        "Test the longLongOnes function"
        myArray = N.zeros(5,'q')
        Series.longLongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testLongLongMax(self):
        "Test the longLongMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.longLongMax(matrix), 5)

    def testLongLongMaxNonContainer(self):
        "Test the longLongMax function with None"
        self.assertRaises(TypeError, Series.longLongMax, None)

    def testLongLongMaxWrongDim(self):
        "Test the longLongMax function with a 1D array"
        self.assertRaises(TypeError, Series.longLongMax, [0, -1, 2, -3])

    def testLongLongFloor(self):
        "Test the longLongFloor function"
        matrix = N.array([[10,-2],[-6,7]],'q')
        Series.longLongFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    #########################################################
    ### Test functions that take arrays of type ULONGLONG ###
    def testUlonglongProd(self):
        "Test the ulongLongProd function"
        self.assertEquals(Series.ulongLongProd([1,2,3,4]), 24)

    def testUlongLongProdNonContainer(self):
        "Test the ulongLongProd function with None"
        self.assertRaises(TypeError, Series.ulongLongProd, None)

    def testUlongLongOnes(self):
        "Test the ulongLongOnes function"
        myArray = N.zeros(5,'Q')
        Series.ulongLongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUlongLongMax(self):
        "Test the ulongLongMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ulongLongMax(matrix), 6)

    def testUlongLongMaxNonContainer(self):
        "Test the ulongLongMax function with None"
        self.assertRaises(TypeError, Series.ulongLongMax, None)

    def testUlongLongMaxWrongDim(self):
        "Test the ulongLongMax function with a 1D array"
        self.assertRaises(TypeError, Series.ulongLongMax, [0, 1, 2, 3])

    def testUlongLongFloor(self):
        "Test the ulongLongFloor function"
        matrix = N.array([[10,2],[6,7]],'Q')
        Series.ulongLongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    #####################################################
    ### Test functions that take arrays of type FLOAT ###
    def testFloatProd(self):
        "Test the floatProd function (to 5 decimal places)"
        self.assertAlmostEquals(Series.floatProd((1,2.718,3,4)), 32.616, 5)

    def testFloatProdBadContainer(self):
        "Test the floatProd function with an invalid list"
        self.assertRaises(BadListError, Series.floatProd, [3.14, "pi"])

    def testFloatOnes(self):
        "Test the floatOnes function"
        myArray = N.zeros(5,'f')
        Series.floatOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    def testFloatOnesNonArray(self):
        "Test the floatOnes function with a list"
        self.assertRaises(TypeError, Series.floatOnes, [True, 0, 2.718, "pi"])

    def testFloatMax(self):
        "Test the floatMax function"
        matrix = [[-6,5,-4],[3.14,-2.718,1]]
        self.assertEquals(Series.floatMax(matrix), 5.0)

    def testFloatMaxNonContainer(self):
        "Test the floatMax function with None"
        self.assertRaises(TypeError, Series.floatMax, None)

    def testFloatMaxWrongDim(self):
        "Test the floatMax function with a 1D array"
        self.assertRaises(TypeError, Series.floatMax, [0.0, -1, 2.718, -3.14])

    def testFloatFloor(self):
        "Test the floatFloor function"
        matrix = N.array([[10,-2],[-6,7]],'f')
        Series.floatFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    ######################################################
    ### Test functions that take arrays of type DOUBLE ###
    def testDoubleProd(self):
        "Test the doubleProd function"
        self.assertEquals(Series.doubleProd((1,2.718,3,4)), 32.616)

    def testDoubleProdBadContainer(self):
        "Test the doubleProd function with an invalid list"
        self.assertRaises(BadListError, Series.doubleProd, [3.14, "pi"])

    def testDoubleOnes(self):
        "Test the doubleOnes function"
        myArray = N.zeros(5,'d')
        Series.doubleOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    def testDoubleOnesNonArray(self):
        "Test the doubleOnes function with a list"
        self.assertRaises(TypeError, Series.doubleOnes, [True, 0, 2.718, "pi"])

    def testDoubleMax(self):
        "Test the doubleMax function"
        matrix = [[-6,5,-4],[3.14,-2.718,1]]
        self.assertEquals(Series.doubleMax(matrix), 5.0)

    def testDoubleMaxNonContainer(self):
        "Test the doubleMax function with None"
        self.assertRaises(TypeError, Series.doubleMax, None)

    def testDoubleMaxWrongDim(self):
        "Test the doubleMax function with a 1D array"
        self.assertRaises(TypeError, Series.doubleMax, [0.0, -1, 2.718, -3.14])

    def testDoubleFloor(self):
        "Test the doubleFloor function"
        matrix = N.array([[10,-2],[-6,7]],'d')
        Series.doubleFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    # ##########################################################
    # ### Test functions that take arrays of type LONGDOUBLE ###
    # def testLongDoubleProd(self):
    #     "Test the longDoubleProd function"
    #     self.assertEquals(Series.longDoubleProd((1,2.718,3,4)), 32.616)

    # def testLongDoubleProdBadContainer(self):
    #     "Test the longDoubleProd function with an invalid list"
    #     self.assertRaises(BadListError, Series.longDoubleProd, [3.14, "pi"])

    # def testLongDoubleOnes(self):
    #     "Test the longDoubleOnes function"
    #     myArray = N.zeros(5,'g')
    #     Series.longDoubleOnes(myArray)
    #     N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    # def testLongDoubleOnesNonArray(self):
    #     "Test the longDoubleOnes function with a list"
    #     self.assertRaises(TypeError, Series.longDoubleOnes, [True, 0, 2.718, "pi"])

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SeriesTestCase))

    # Execute the test suite
    print "Testing Module Series\n"
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
