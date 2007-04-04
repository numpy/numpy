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
import Vector

######################################################################

class VectorTestCase(unittest.TestCase):

    ####################################################
    ### Test functions that take arrays of type BYTE ###
    def testScharLength(self):
        "Test scharLength function"
        self.assertEquals(Vector.scharLength([5, 12, 0]), 13)

    def testScharLengthBad(self):
        "Test scharLength function for wrong size"
        self.assertRaises(TypeError, Vector.scharLength, [5, 12])

    def testScharProd(self):
        "Test scharProd function"
        self.assertEquals(Vector.scharProd([1,2,3,4]), 24)

    def testScharProdNonContainer(self):
        "Test scharProd function with None"
        self.assertRaises(TypeError, Vector.scharProd, None)

    def testScharSum(self):
        "Test scharSum function"
        self.assertEquals(Vector.scharSum([-5,6,-7,8]), 2)

    def testScharReverse(self):
        "Test scharReverse function"
        vector = N.array([1,2,4],'b')
        Vector.scharReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testScharOnes(self):
        "Test scharOnes function"
        myArray = N.zeros(5,'b')
        Vector.scharOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testScharZeros(self):
        "Test scharZeros function"
        myArray = N.ones(5,'b')
        Vector.scharZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testScharEOSplit(self):
        "Test scharEOSplit function"
        even, odd = Vector.scharEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testScharTwos(self):
        "Test scharTwos function"
        twos = Vector.scharTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testScharThrees(self):
        "Test scharThrees function"
        threes = Vector.scharThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    #####################################################
    ### Test functions that take arrays of type UBYTE ###
    def testUcharLength(self):
        "Test ucharLength function"
        self.assertEquals(Vector.ucharLength([5, 12, 0]), 13)

    def testUcharLengthBad(self):
        "Test ucharLength function for wrong size"
        self.assertRaises(TypeError, Vector.ucharLength, [5, 12])

    def testUcharProd(self):
        "Test ucharProd function"
        self.assertEquals(Vector.ucharProd([1,2,3,4]), 24)

    def testUcharProdNonContainer(self):
        "Test ucharProd function with None"
        self.assertRaises(TypeError, Vector.ucharProd, None)

    def testUcharSum(self):
        "Test ucharSum function"
        self.assertEquals(Vector.ucharSum([5,6,7,8]), 26)

    def testUcharReverse(self):
        "Test ucharReverse function"
        vector = N.array([1,2,4],'B')
        Vector.ucharReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testUcharOnes(self):
        "Test ucharOnes function"
        myArray = N.zeros(5,'B')
        Vector.ucharOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUcharZeros(self):
        "Test ucharZeros function"
        myArray = N.ones(5,'B')
        Vector.ucharZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUcharEOSplit(self):
        "Test ucharEOSplit function"
        even, odd = Vector.ucharEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testUcharTwos(self):
        "Test ucharTwos function"
        twos = Vector.ucharTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testUcharThrees(self):
        "Test ucharThrees function"
        threes = Vector.ucharThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    #####################################################
    ### Test functions that take arrays of type SHORT ###
    def testShortLength(self):
        "Test shortLength function"
        self.assertEquals(Vector.shortLength([5, 12, 0]), 13)

    def testShortLengthBad(self):
        "Test shortLength function for wrong size"
        self.assertRaises(TypeError, Vector.shortLength, [5, 12])

    def testShortProd(self):
        "Test shortProd function"
        self.assertEquals(Vector.shortProd([1,2,3,4]), 24)

    def testShortProdNonContainer(self):
        "Test shortProd function with None"
        self.assertRaises(TypeError, Vector.shortProd, None)

    def testShortSum(self):
        "Test shortSum function"
        self.assertEquals(Vector.shortSum([-5,6,-7,8]), 2)

    def testShortReverse(self):
        "Test shortReverse function"
        vector = N.array([1,2,4],'h')
        Vector.shortReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testShortOnes(self):
        "Test shortOnes function"
        myArray = N.zeros(5,'h')
        Vector.shortOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testShortZeros(self):
        "Test shortZeros function"
        myArray = N.ones(5,'h')
        Vector.shortZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testShortEOSplit(self):
        "Test shortEOSplit function"
        even, odd = Vector.shortEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testShortTwos(self):
        "Test shortTwos function"
        twos = Vector.shortTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testShortThrees(self):
        "Test shortThrees function"
        threes = Vector.shortThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    ######################################################
    ### Test functions that take arrays of type USHORT ###
    def testUshortLength(self):
        "Test ushortLength function"
        self.assertEquals(Vector.ushortLength([5, 12, 0]), 13)

    def testUshortLengthBad(self):
        "Test ushortLength function for wrong size"
        self.assertRaises(TypeError, Vector.ushortLength, [5, 12])

    def testUshortProd(self):
        "Test ushortProd function"
        self.assertEquals(Vector.ushortProd([1,2,3,4]), 24)

    def testUshortProdNonContainer(self):
        "Test ushortProd function with None"
        self.assertRaises(TypeError, Vector.ushortProd, None)

    def testUshortSum(self):
        "Test ushortSum function"
        self.assertEquals(Vector.ushortSum([5,6,7,8]), 26)

    def testUshortReverse(self):
        "Test ushortReverse function"
        vector = N.array([1,2,4],'H')
        Vector.ushortReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testUshortOnes(self):
        "Test ushortOnes function"
        myArray = N.zeros(5,'H')
        Vector.ushortOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUshortZeros(self):
        "Test ushortZeros function"
        myArray = N.ones(5,'H')
        Vector.ushortZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUshortEOSplit(self):
        "Test ushortEOSplit function"
        even, odd = Vector.ushortEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testUshortTwos(self):
        "Test ushortTwos function"
        twos = Vector.ushortTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testUshortThrees(self):
        "Test ushortThrees function"
        threes = Vector.ushortThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    ###################################################
    ### Test functions that take arrays of type INT ###
    def testIntLength(self):
        "Test intLength function"
        self.assertEquals(Vector.intLength([5, 12, 0]), 13)

    def testIntLengthBad(self):
        "Test intLength function for wrong size"
        self.assertRaises(TypeError, Vector.intLength, [5, 12])

    def testIntProd(self):
        "Test intProd function"
        self.assertEquals(Vector.intProd([1,2,3,4]), 24)

    def testIntProdNonContainer(self):
        "Test intProd function with None"
        self.assertRaises(TypeError, Vector.intProd, None)

    def testIntSum(self):
        "Test intSum function"
        self.assertEquals(Vector.intSum([-5,6,-7,8]), 2)

    def testIntOnes(self):
        "Test intOnes function"
        myArray = N.zeros(5,'i')
        Vector.intOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testIntReverse(self):
        "Test intReverse function"
        vector = N.array([1,2,4],'i')
        Vector.intReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testIntZeros(self):
        "Test intZeros function"
        myArray = N.ones(5,'i')
        Vector.intZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testIntEOSplit(self):
        "Test intEOSplit function"
        even, odd = Vector.intEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testIntTwos(self):
        "Test intTwos function"
        twos = Vector.intTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testIntThrees(self):
        "Test intThrees function"
        threes = Vector.intThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    ####################################################
    ### Test functions that take arrays of type UINT ###
    def testUintLength(self):
        "Test uintLength function"
        self.assertEquals(Vector.uintLength([5, 12, 0]), 13)

    def testUintLengthBad(self):
        "Test uintLength function for wrong size"

        self.assertRaises(TypeError, Vector.uintLength, [5, 12])

    def testUintProd(self):
        "Test uintProd function"
        self.assertEquals(Vector.uintProd([1,2,3,4]), 24)

    def testUintProdNonContainer(self):
        "Test uintProd function with None"
        self.assertRaises(TypeError, Vector.uintProd, None)

    def testUintSum(self):
        "Test uintSum function"
        self.assertEquals(Vector.uintSum([5,6,7,8]), 26)

    def testUintReverse(self):
        "Test uintReverse function"
        vector = N.array([1,2,4],'I')
        Vector.uintReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testUintOnes(self):
        "Test uintOnes function"
        myArray = N.zeros(5,'I')
        Vector.uintOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUintZeros(self):
        "Test uintZeros function"
        myArray = N.ones(5,'I')
        Vector.uintZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUintEOSplit(self):
        "Test uintEOSplit function"
        even, odd = Vector.uintEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testUintTwos(self):
        "Test uintTwos function"
        twos = Vector.uintTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testUintThrees(self):
        "Test uintThrees function"
        threes = Vector.uintThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    ####################################################
    ### Test functions that take arrays of type LONG ###
    def testLongLength(self):
        "Test longLength function"
        self.assertEquals(Vector.longLength([5, 12, 0]), 13)

    def testLongLengthBad(self):
        "Test longLength function for wrong size"
        self.assertRaises(TypeError, Vector.longLength, [5, 12])

    def testLongProd(self):
        "Test longProd function"
        self.assertEquals(Vector.longProd([1,2,3,4]), 24)

    def testLongProdNonContainer(self):
        "Test longProd function with None"
        self.assertRaises(TypeError, Vector.longProd, None)

    def testLongSum(self):
        "Test longSum function"
        self.assertEquals(Vector.longSum([-5,6,-7,8]), 2)

    def testLongReverse(self):
        "Test longReverse function"
        vector = N.array([1,2,4],'l')
        Vector.longReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testLongOnes(self):
        "Test longOnes function"
        myArray = N.zeros(5,'l')
        Vector.longOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testLongZeros(self):
        "Test longZeros function"
        myArray = N.ones(5,'l')
        Vector.longZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testLongEOSplit(self):
        "Test longEOSplit function"
        even, odd = Vector.longEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testLongTwos(self):
        "Test longTwos function"
        twos = Vector.longTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testLongThrees(self):
        "Test longThrees function"
        threes = Vector.longThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    #####################################################
    ### Test functions that take arrays of type ULONG ###
    def testUlongLength(self):
        "Test ulongLength function"
        self.assertEquals(Vector.ulongLength([5, 12, 0]), 13)

    def testUlongLengthBad(self):
        "Test ulongLength function for wrong size"
        self.assertRaises(TypeError, Vector.ulongLength, [5, 12])

    def testUlongProd(self):
        "Test ulongProd function"
        self.assertEquals(Vector.ulongProd([1,2,3,4]), 24)

    def testUlongProdNonContainer(self):
        "Test ulongProd function with None"
        self.assertRaises(TypeError, Vector.ulongProd, None)

    def testUlongSum(self):
        "Test ulongSum function"
        self.assertEquals(Vector.ulongSum([5,6,7,8]), 26)

    def testUlongReverse(self):
        "Test ulongReverse function"
        vector = N.array([1,2,4],'L')
        Vector.ulongReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testUlongOnes(self):
        "Test ulongOnes function"
        myArray = N.zeros(5,'L')
        Vector.ulongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUlongZeros(self):
        "Test ulongZeros function"
        myArray = N.ones(5,'L')
        Vector.ulongZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUlongEOSplit(self):
        "Test ulongEOSplit function"
        even, odd = Vector.ulongEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testUlongTwos(self):
        "Test ulongTwos function"
        twos = Vector.ulongTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testUlongThrees(self):
        "Test ulongThrees function"
        threes = Vector.ulongThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    ########################################################
    ### Test functions that take arrays of type LONGLONG ###
    def testLongLongLength(self):
        "Test longLongLength function"
        self.assertEquals(Vector.longLongLength([5, 12, 0]), 13)

    def testLongLongLengthBad(self):
        "Test longLongLength function for wrong size"
        self.assertRaises(TypeError, Vector.longLongLength, [5, 12])

    def testLongLongProd(self):
        "Test longLongProd function"
        self.assertEquals(Vector.longLongProd([1,2,3,4]), 24)

    def testLongLongProdNonContainer(self):
        "Test longLongProd function with None"
        self.assertRaises(TypeError, Vector.longLongProd, None)

    def testLongLongSum(self):
        "Test longLongSum function"
        self.assertEquals(Vector.longLongSum([-5,6,-7,8]), 2)

    def testLongLongReverse(self):
        "Test longLongReverse function"
        vector = N.array([1,2,4],'q')
        Vector.longLongReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testLongLongOnes(self):
        "Test longLongOnes function"
        myArray = N.zeros(5,'q')
        Vector.longLongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testLongLongZeros(self):
        "Test longLongZeros function"
        myArray = N.ones(5,'q')
        Vector.longLongZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testLongLongEOSplit(self):
        "Test longLongEOSplit function"
        even, odd = Vector.longLongEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testLongLongTwos(self):
        "Test longLongTwos function"
        twos = Vector.longLongTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testLongLongThrees(self):
        "Test longLongThrees function"
        threes = Vector.longLongThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    #########################################################
    ### Test functions that take arrays of type ULONGLONG ###
    def testUlongLongLength(self):
        "Test ulongLongLength function"
        self.assertEquals(Vector.ulongLongLength([5, 12, 0]), 13)

    def testUlongLongLengthBad(self):
        "Test ulongLongLength function for wrong size"
        self.assertRaises(TypeError, Vector.ulongLongLength, [5, 12])

    def testUlonglongProd(self):
        "Test ulongLongProd function"
        self.assertEquals(Vector.ulongLongProd([1,2,3,4]), 24)

    def testUlongLongProdNonContainer(self):
        "Test ulongLongProd function with None"
        self.assertRaises(TypeError, Vector.ulongLongProd, None)

    def testUlongLongSum(self):
        "Test ulongLongSum function"
        self.assertEquals(Vector.ulongLongSum([5,6,7,8]), 26)

    def testUlongLongReverse(self):
        "Test ulongLongReverse function"
        vector = N.array([1,2,4],'Q')
        Vector.ulongLongReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testUlongLongOnes(self):
        "Test ulongLongOnes function"
        myArray = N.zeros(5,'Q')
        Vector.ulongLongOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testUlongLongZeros(self):
        "Test ulongLongZeros function"
        myArray = N.ones(5,'Q')
        Vector.ulongLongZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testUlongLongEOSplit(self):
        "Test ulongLongEOSplit function"
        even, odd = Vector.ulongLongEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testUlongLongTwos(self):
        "Test ulongLongTwos function"
        twos = Vector.ulongLongTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testUlongLongThrees(self):
        "Test ulongLongThrees function"
        threes = Vector.ulongLongThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    #####################################################
    ### Test functions that take arrays of type FLOAT ###
    def testFloatLength(self):
        "Test floatLength function"
        self.assertEquals(Vector.floatLength([5, 12, 0]), 13)

    def testFloatLengthBad(self):
        "Test floatLength function for wrong size"
        self.assertRaises(TypeError, Vector.floatLength, [5, 12])

    def testFloatProd(self):
        "Test floatProd function (to 5 decimal places)"
        self.assertAlmostEquals(Vector.floatProd((1,2.718,3,4)), 32.616, 5)

    def testFloatProdBadContainer(self):
        "Test floatProd function with an invalid list"
        self.assertRaises(BadListError, Vector.floatProd, [3.14, "pi"])

    def testFloatSum(self):
        "Test floatSum function"
        self.assertEquals(Vector.floatSum([-5,6,-7,8]), 2)

    def testFloatReverse(self):
        "Test floatReverse function"
        vector = N.array([1,2,4],'f')
        Vector.floatReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testFloatOnes(self):
        "Test floatOnes function"
        myArray = N.zeros(5,'f')
        Vector.floatOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    def testFloatOnesNonArray(self):
        "Test floatOnes function with a list"
        self.assertRaises(TypeError, Vector.floatOnes, [True, 0, 2.718, "pi"])

    def testFloatZeros(self):
        "Test floatZeros function"
        myArray = N.ones(5,'f')
        Vector.floatZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testFloatEOSplit(self):
        "Test floatEOSplit function"
        even, odd = Vector.floatEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testFloatTwos(self):
        "Test floatTwos function"
        twos = Vector.floatTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testFloatThrees(self):
        "Test floatThrees function"
        threes = Vector.floatThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

    ######################################################
    ### Test functions that take arrays of type DOUBLE ###
    def testDoubleLength(self):
        "Test doubleLength function"
        self.assertEquals(Vector.doubleLength([5, 12, 0]), 13)

    def testDoubleLengthBad(self):
        "Test doubleLength function for wrong size"
        self.assertRaises(TypeError, Vector.doubleLength, [5, 12])

    def testDoubleProd(self):
        "Test doubleProd function"
        self.assertEquals(Vector.doubleProd((1,2.718,3,4)), 32.616)

    def testDoubleProdBadContainer(self):
        "Test doubleProd function with an invalid list"
        self.assertRaises(BadListError, Vector.doubleProd, [3.14, "pi"])

    def testDoubleSum(self):
        "Test doubleSum function"
        self.assertEquals(Vector.doubleSum([-5,6,-7,8]), 2)

    def testDoubleReverse(self):
        "Test doubleReverse function"
        vector = N.array([1,2,4],'d')
        Vector.doubleReverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testDoubleOnes(self):
        "Test doubleOnes function"
        myArray = N.zeros(5,'d')
        Vector.doubleOnes(myArray)
        N.testing.assert_array_equal(myArray, N.array([1.,1.,1.,1.,1.]))

    def testDoubleOnesNonArray(self):
        "Test doubleOnes function with a list"
        self.assertRaises(TypeError, Vector.doubleOnes, [True, 0, 2.718, "pi"])

    def testDoubleZeros(self):
        "Test doubleZeros function"
        myArray = N.ones(5,'d')
        Vector.doubleZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testDoubleEOSplit(self):
        "Test doubleEOSplit function"
        even, odd = Vector.doubleEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testDoubleTwos(self):
        "Test doubleTwos function"
        twos = Vector.doubleTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testDoubleThrees(self):
        "Test doubleThrees function"
        threes = Vector.doubleThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(VectorTestCase))

    # Execute the test suite
    print "Testing 1D Functions of Module Vector"
    print "NumPy version", N.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
