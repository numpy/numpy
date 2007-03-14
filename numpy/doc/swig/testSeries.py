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
        "Test scharProd function"
        self.assertEquals(Series.scharProd([1,2,3,4]), 24)

    def testScharProdNonContainer(self):
        "Test scharProd function with None"
        self.assertRaises(TypeError, Series.scharProd, None)

    def testScharOnes(self):
        "Test scharOnes function"
        myArray = N.zeros(5,'b')
        Series.scharOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testScharMax(self):
        "Test scharMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.scharMax(matrix), 5)

    def testScharMaxNonContainer(self):
        "Test scharMax function with None"
        self.assertRaises(TypeError, Series.scharMax, None)

    def testScharMaxWrongDim(self):
        "Test scharMax function with a 1D array"
        self.assertRaises(TypeError, Series.scharMax, [0, -1, 2, -3])

    def testScharFloor(self):
        "Test scharFloor function"
        matrix = N.array([[10,-2],[-6,7]],'b')
        Series.scharFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testScharSum(self):
        "Test scharSum function"
        self.assertEquals(Series.scharSum([-5,6,-7,8]), 2)

    def testScharZeros(self):
        "Test scharZeros function"
        myArray = N.ones(5,'b')
        Series.scharZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testScharMin(self):
        "Test scharMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.scharMin(matrix), 4)

    def testScharCeil(self):
        "Test scharCeil function"
        matrix = N.array([[10,-2],[-6,7]],'b')
        Series.scharCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    #####################################################
    ### Test functions that take arrays of type UBYTE ###
    def testUcharProd(self):
        "Test ucharProd function"
        self.assertEquals(Series.ucharProd([1,2,3,4]), 24)

    def testUcharProdNonContainer(self):
        "Test ucharProd function with None"
        self.assertRaises(TypeError, Series.ucharProd, None)

    def testUcharOnes(self):
        "Test ucharOnes function"
        myArray = N.zeros(5,'B')
        Series.ucharOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUcharMax(self):
        "Test ucharMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ucharMax(matrix), 6)

    def testUcharMaxNonContainer(self):
        "Test ucharMax function with None"
        self.assertRaises(TypeError, Series.ucharMax, None)

    def testUcharMaxWrongDim(self):
        "Test ucharMax function with a 1D array"
        self.assertRaises(TypeError, Series.ucharMax, [0, 1, 2, 3])

    def testUcharFloor(self):
        "Test ucharFloor function"
        matrix = N.array([[10,2],[6,7]],'B')
        Series.ucharFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUcharSum(self):
        "Test ucharSum function"
        self.assertEquals(Series.ucharSum([5,6,7,8]), 26)

    def testUcharZeros(self):
        "Test ucharZeros function"
        myArray = N.ones(5,'B')
        Series.ucharZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUcharMin(self):
        "Test ucharMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ucharMin(matrix), 4)

    def testUcharCeil(self):
        "Test ucharCeil function"
        matrix = N.array([[10,2],[6,7]],'B')
        Series.ucharCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    #####################################################
    ### Test functions that take arrays of type SHORT ###
    def testShortProd(self):
        "Test shortProd function"
        self.assertEquals(Series.shortProd([1,2,3,4]), 24)

    def testShortProdNonContainer(self):
        "Test shortProd function with None"
        self.assertRaises(TypeError, Series.shortProd, None)

    def testShortOnes(self):
        "Test shortOnes function"
        myArray = N.zeros(5,'h')
        Series.shortOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testShortMax(self):
        "Test shortMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.shortMax(matrix), 5)

    def testShortMaxNonContainer(self):
        "Test shortMax function with None"
        self.assertRaises(TypeError, Series.shortMax, None)

    def testShortMaxWrongDim(self):
        "Test shortMax function with a 1D array"
        self.assertRaises(TypeError, Series.shortMax, [0, -1, 2, -3])

    def testShortFloor(self):
        "Test shortFloor function"
        matrix = N.array([[10,-2],[-6,7]],'h')
        Series.shortFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testShortSum(self):
        "Test shortSum function"
        self.assertEquals(Series.shortSum([-5,6,-7,8]), 2)

    def testShortZeros(self):
        "Test shortZeros function"
        myArray = N.ones(5,'h')
        Series.shortZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testShortMin(self):
        "Test shortMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.shortMin(matrix), 4)

    def testShortCeil(self):
        "Test shortCeil function"
        matrix = N.array([[10,-2],[-6,7]],'h')
        Series.shortCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    ######################################################
    ### Test functions that take arrays of type USHORT ###
    def testUshortProd(self):
        "Test ushortProd function"
        self.assertEquals(Series.ushortProd([1,2,3,4]), 24)

    def testUshortProdNonContainer(self):
        "Test ushortProd function with None"
        self.assertRaises(TypeError, Series.ushortProd, None)

    def testUshortOnes(self):
        "Test ushortOnes function"
        myArray = N.zeros(5,'H')
        Series.ushortOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUshortMax(self):
        "Test ushortMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ushortMax(matrix), 6)

    def testUshortMaxNonContainer(self):
        "Test ushortMax function with None"
        self.assertRaises(TypeError, Series.ushortMax, None)

    def testUshortMaxWrongDim(self):
        "Test ushortMax function with a 1D array"
        self.assertRaises(TypeError, Series.ushortMax, [0, 1, 2, 3])

    def testUshortFloor(self):
        "Test ushortFloor function"
        matrix = N.array([[10,2],[6,7]],'H')
        Series.ushortFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUshortSum(self):
        "Test ushortSum function"
        self.assertEquals(Series.ushortSum([5,6,7,8]), 26)

    def testUshortZeros(self):
        "Test ushortZeros function"
        myArray = N.ones(5,'H')
        Series.ushortZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUshortMin(self):
        "Test ushortMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ushortMin(matrix), 4)

    def testUshortCeil(self):
        "Test ushortCeil function"
        matrix = N.array([[10,2],[6,7]],'H')
        Series.ushortCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    ###################################################
    ### Test functions that take arrays of type INT ###
    def testIntProd(self):
        "Test intProd function"
        self.assertEquals(Series.intProd([1,2,3,4]), 24)

    def testIntProdNonContainer(self):
        "Test intProd function with None"
        self.assertRaises(TypeError, Series.intProd, None)

    def testIntOnes(self):
        "Test intOnes function"
        myArray = N.zeros(5,'i')
        Series.intOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testIntMax(self):
        "Test intMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.intMax(matrix), 5)

    def testIntMaxNonContainer(self):
        "Test intMax function with None"
        self.assertRaises(TypeError, Series.intMax, None)

    def testIntMaxWrongDim(self):
        "Test intMax function with a 1D array"
        self.assertRaises(TypeError, Series.intMax, [0, -1, 2, -3])

    def testIntFloor(self):
        "Test intFloor function"
        matrix = N.array([[10,-2],[-6,7]],'i')
        Series.intFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testIntSum(self):
        "Test intSum function"
        self.assertEquals(Series.intSum([-5,6,-7,8]), 2)

    def testIntZeros(self):
        "Test intZeros function"
        myArray = N.ones(5,'i')
        Series.intZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testIntMin(self):
        "Test intMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.intMin(matrix), 4)

    def testIntCeil(self):
        "Test intCeil function"
        matrix = N.array([[10,-2],[-6,7]],'i')
        Series.intCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    ####################################################
    ### Test functions that take arrays of type UINT ###
    def testUintProd(self):
        "Test uintProd function"
        self.assertEquals(Series.uintProd([1,2,3,4]), 24)

    def testUintProdNonContainer(self):
        "Test uintProd function with None"
        self.assertRaises(TypeError, Series.uintProd, None)

    def testUintOnes(self):
        "Test uintOnes function"
        myArray = N.zeros(5,'I')
        Series.uintOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUintMax(self):
        "Test uintMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.uintMax(matrix), 6)

    def testUintMaxNonContainer(self):
        "Test uintMax function with None"
        self.assertRaises(TypeError, Series.uintMax, None)

    def testUintMaxWrongDim(self):
        "Test uintMax function with a 1D array"
        self.assertRaises(TypeError, Series.uintMax, [0, 1, 2, 3])

    def testUintFloor(self):
        "Test uintFloor function"
        matrix = N.array([[10,2],[6,7]],'I')
        Series.uintFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUintSum(self):
        "Test uintSum function"
        self.assertEquals(Series.uintSum([5,6,7,8]), 26)

    def testUintZeros(self):
        "Test uintZeros function"
        myArray = N.ones(5,'I')
        Series.uintZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUintMin(self):
        "Test uintMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.uintMin(matrix), 4)

    def testUintCeil(self):
        "Test uintCeil function"
        matrix = N.array([[10,2],[6,7]],'I')
        Series.uintCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    ####################################################
    ### Test functions that take arrays of type LONG ###
    def testLongProd(self):
        "Test longProd function"
        self.assertEquals(Series.longProd([1,2,3,4]), 24)

    def testLongProdNonContainer(self):
        "Test longProd function with None"
        self.assertRaises(TypeError, Series.longProd, None)

    def testLongOnes(self):
        "Test longOnes function"
        myArray = N.zeros(5,'l')
        Series.longOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testLongMax(self):
        "Test longMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.longMax(matrix), 5)

    def testLongMaxNonContainer(self):
        "Test longMax function with None"
        self.assertRaises(TypeError, Series.longMax, None)

    def testLongMaxWrongDim(self):
        "Test longMax function with a 1D array"
        self.assertRaises(TypeError, Series.longMax, [0, -1, 2, -3])

    def testLongFloor(self):
        "Test longFloor function"
        matrix = N.array([[10,-2],[-6,7]],'l')
        Series.longFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testLongSum(self):
        "Test longSum function"
        self.assertEquals(Series.longSum([-5,6,-7,8]), 2)

    def testLongZeros(self):
        "Test longZeros function"
        myArray = N.ones(5,'l')
        Series.longZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testLongMin(self):
        "Test longMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.longMin(matrix), 4)

    def testLongCeil(self):
        "Test longCeil function"
        matrix = N.array([[10,-2],[-6,7]],'l')
        Series.longCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    #####################################################
    ### Test functions that take arrays of type ULONG ###
    def testUlongProd(self):
        "Test ulongProd function"
        self.assertEquals(Series.ulongProd([1,2,3,4]), 24)

    def testUlongProdNonContainer(self):
        "Test ulongProd function with None"
        self.assertRaises(TypeError, Series.ulongProd, None)

    def testUlongOnes(self):
        "Test ulongOnes function"
        myArray = N.zeros(5,'L')
        Series.ulongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUlongMax(self):
        "Test ulongMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ulongMax(matrix), 6)

    def testUlongMaxNonContainer(self):
        "Test ulongMax function with None"
        self.assertRaises(TypeError, Series.ulongMax, None)

    def testUlongMaxWrongDim(self):
        "Test ulongMax function with a 1D array"
        self.assertRaises(TypeError, Series.ulongMax, [0, 1, 2, 3])

    def testUlongFloor(self):
        "Test ulongFloor function"
        matrix = N.array([[10,2],[6,7]],'L')
        Series.ulongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUlongSum(self):
        "Test ulongSum function"
        self.assertEquals(Series.ulongSum([5,6,7,8]), 26)

    def testUlongZeros(self):
        "Test ulongZeros function"
        myArray = N.ones(5,'L')
        Series.ulongZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUlongMin(self):
        "Test ulongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ulongMin(matrix), 4)

    def testUlongCeil(self):
        "Test ulongCeil function"
        matrix = N.array([[10,2],[6,7]],'L')
        Series.ulongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    ########################################################
    ### Test functions that take arrays of type LONGLONG ###
    def testLongLongProd(self):
        "Test longLongProd function"
        self.assertEquals(Series.longLongProd([1,2,3,4]), 24)

    def testLongLongProdNonContainer(self):
        "Test longLongProd function with None"
        self.assertRaises(TypeError, Series.longLongProd, None)

    def testLongLongOnes(self):
        "Test longLongOnes function"
        myArray = N.zeros(5,'q')
        Series.longLongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testLongLongMax(self):
        "Test longLongMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Series.longLongMax(matrix), 5)

    def testLongLongMaxNonContainer(self):
        "Test longLongMax function with None"
        self.assertRaises(TypeError, Series.longLongMax, None)

    def testLongLongMaxWrongDim(self):
        "Test longLongMax function with a 1D array"
        self.assertRaises(TypeError, Series.longLongMax, [0, -1, 2, -3])

    def testLongLongFloor(self):
        "Test longLongFloor function"
        matrix = N.array([[10,-2],[-6,7]],'q')
        Series.longLongFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testLongLongSum(self):
        "Test longLongSum function"
        self.assertEquals(Series.longLongSum([-5,6,-7,8]), 2)

    def testLongLongZeros(self):
        "Test longLongZeros function"
        myArray = N.ones(5,'q')
        Series.longLongZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testLongLongMin(self):
        "Test longLongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.longLongMin(matrix), 4)

    def testLongLongCeil(self):
        "Test longLongCeil function"
        matrix = N.array([[10,-2],[-6,7]],'q')
        Series.longLongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    #########################################################
    ### Test functions that take arrays of type ULONGLONG ###
    def testUlonglongProd(self):
        "Test ulongLongProd function"
        self.assertEquals(Series.ulongLongProd([1,2,3,4]), 24)

    def testUlongLongProdNonContainer(self):
        "Test ulongLongProd function with None"
        self.assertRaises(TypeError, Series.ulongLongProd, None)

    def testUlongLongOnes(self):
        "Test ulongLongOnes function"
        myArray = N.zeros(5,'Q')
        Series.ulongLongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUlongLongMax(self):
        "Test ulongLongMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Series.ulongLongMax(matrix), 6)

    def testUlongLongMaxNonContainer(self):
        "Test ulongLongMax function with None"
        self.assertRaises(TypeError, Series.ulongLongMax, None)

    def testUlongLongMaxWrongDim(self):
        "Test ulongLongMax function with a 1D array"
        self.assertRaises(TypeError, Series.ulongLongMax, [0, 1, 2, 3])

    def testUlongLongFloor(self):
        "Test ulongLongFloor function"
        matrix = N.array([[10,2],[6,7]],'Q')
        Series.ulongLongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUlongLongSum(self):
        "Test ulongLongSum function"
        self.assertEquals(Series.ulongLongSum([5,6,7,8]), 26)

    def testUlongLongZeros(self):
        "Test ulongLongZeros function"
        myArray = N.ones(5,'Q')
        Series.ulongLongZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUlongLongMin(self):
        "Test ulongLongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ulongLongMin(matrix), 4)

    def testUlongLongCeil(self):
        "Test ulongLongCeil function"
        matrix = N.array([[10,2],[6,7]],'Q')
        Series.ulongLongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    #####################################################
    ### Test functions that take arrays of type FLOAT ###
    def testFloatProd(self):
        "Test floatProd function (to 5 decimal places)"
        self.assertAlmostEquals(Series.floatProd((1,2.718,3,4)), 32.616, 5)

    def testFloatProdBadContainer(self):
        "Test floatProd function with an invalid list"
        self.assertRaises(BadListError, Series.floatProd, [3.14, "pi"])

    def testFloatOnes(self):
        "Test floatOnes function"
        myArray = N.zeros(5,'f')
        Series.floatOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    def testFloatOnesNonArray(self):
        "Test floatOnes function with a list"
        self.assertRaises(TypeError, Series.floatOnes, [True, 0, 2.718, "pi"])

    def testFloatMax(self):
        "Test floatMax function"
        matrix = [[-6,5,-4],[3.14,-2.718,1]]
        self.assertEquals(Series.floatMax(matrix), 5.0)

    def testFloatMaxNonContainer(self):
        "Test floatMax function with None"
        self.assertRaises(TypeError, Series.floatMax, None)

    def testFloatMaxWrongDim(self):
        "Test floatMax function with a 1D array"
        self.assertRaises(TypeError, Series.floatMax, [0.0, -1, 2.718, -3.14])

    def testFloatFloor(self):
        "Test floatFloor function"
        matrix = N.array([[10,-2],[-6,7]],'f')
        Series.floatFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testFloatSum(self):
        "Test floatSum function"
        self.assertEquals(Series.floatSum([-5,6,-7,8]), 2)

    def testFloatZeros(self):
        "Test floatZeros function"
        myArray = N.ones(5,'f')
        Series.floatZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testFloatMin(self):
        "Test floatMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.floatMin(matrix), 4)

    def testFloatCeil(self):
        "Test floatCeil function"
        matrix = N.array([[10,-2],[-6,7]],'f')
        Series.floatCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    ######################################################
    ### Test functions that take arrays of type DOUBLE ###
    def testDoubleProd(self):
        "Test doubleProd function"
        self.assertEquals(Series.doubleProd((1,2.718,3,4)), 32.616)

    def testDoubleProdBadContainer(self):
        "Test doubleProd function with an invalid list"
        self.assertRaises(BadListError, Series.doubleProd, [3.14, "pi"])

    def testDoubleOnes(self):
        "Test doubleOnes function"
        myArray = N.zeros(5,'d')
        Series.doubleOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    def testDoubleOnesNonArray(self):
        "Test doubleOnes function with a list"
        self.assertRaises(TypeError, Series.doubleOnes, [True, 0, 2.718, "pi"])

    def testDoubleMax(self):
        "Test doubleMax function"
        matrix = [[-6,5,-4],[3.14,-2.718,1]]
        self.assertEquals(Series.doubleMax(matrix), 5.0)

    def testDoubleMaxNonContainer(self):
        "Test doubleMax function with None"
        self.assertRaises(TypeError, Series.doubleMax, None)

    def testDoubleMaxWrongDim(self):
        "Test doubleMax function with a 1D array"
        self.assertRaises(TypeError, Series.doubleMax, [0.0, -1, 2.718, -3.14])

    def testDoubleFloor(self):
        "Test doubleFloor function"
        matrix = N.array([[10,-2],[-6,7]],'d')
        Series.doubleFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testDoubleSum(self):
        "Test doubleSum function"
        self.assertEquals(Series.doubleSum([-5,6,-7,8]), 2)

    def testDoubleZeros(self):
        "Test doubleZeros function"
        myArray = N.ones(5,'d')
        Series.doubleZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testDoubleMin(self):
        "Test doubleMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.doubleMin(matrix), 4)

    def testDoubleCeil(self):
        "Test doubleCeil function"
        matrix = N.array([[10,-2],[-6,7]],'d')
        Series.doubleCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SeriesTestCase))

    # Execute the test suite
    print "Testing Module Series\n"
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
