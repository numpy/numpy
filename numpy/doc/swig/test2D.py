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
    def testScharDet(self):
        "Test scharDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Series.scharDet(matrix), -2)

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

    def testScharMin(self):
        "Test scharMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.scharMin(matrix), 4)

    def testScharScale(self):
        "Test scharScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'b')
        Series.scharScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testScharFloor(self):
        "Test scharFloor function"
        matrix = N.array([[10,-2],[-6,7]],'b')
        Series.scharFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testScharCeil(self):
        "Test scharCeil function"
        matrix = N.array([[10,-2],[-6,7]],'b')
        Series.scharCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testScharLUSplit(self):
        "Test scharLUSplit function"
        lower, upper = Series.scharLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type UBYTE ###
    def testUcharDet(self):
        "Test ucharDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Series.ucharDet(matrix), 2)

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

    def testUcharMin(self):
        "Test ucharMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ucharMin(matrix), 4)

    def testUcharScale(self):
        "Test ucharScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'B')
        Series.ucharScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUcharFloor(self):
        "Test ucharFloor function"
        matrix = N.array([[10,2],[6,7]],'B')
        Series.ucharFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUcharCeil(self):
        "Test ucharCeil function"
        matrix = N.array([[10,2],[6,7]],'B')
        Series.ucharCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUcharLUSplit(self):
        "Test ucharLUSplit function"
        lower, upper = Series.ucharLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type SHORT ###
    def testShortDet(self):
        "Test shortDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Series.shortDet(matrix), -2)

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

    def testShortMin(self):
        "Test shortMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.shortMin(matrix), 4)

    def testShortScale(self):
        "Test shortScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'h')
        Series.shortScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testShortFloor(self):
        "Test shortFloor function"
        matrix = N.array([[10,-2],[-6,7]],'h')
        Series.shortFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testShortCeil(self):
        "Test shortCeil function"
        matrix = N.array([[10,-2],[-6,7]],'h')
        Series.shortCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testShortLUSplit(self):
        "Test shortLUSplit function"
        lower, upper = Series.shortLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ######################################################
    ### Test functions that take arrays of type USHORT ###
    def testUshortDet(self):
        "Test ushortDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Series.ushortDet(matrix), 2)

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

    def testUshortMin(self):
        "Test ushortMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ushortMin(matrix), 4)

    def testUshortScale(self):
        "Test ushortScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'H')
        Series.ushortScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUshortFloor(self):
        "Test ushortFloor function"
        matrix = N.array([[10,2],[6,7]],'H')
        Series.ushortFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUshortCeil(self):
        "Test ushortCeil function"
        matrix = N.array([[10,2],[6,7]],'H')
        Series.ushortCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUshortLUSplit(self):
        "Test ushortLUSplit function"
        lower, upper = Series.ushortLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ###################################################
    ### Test functions that take arrays of type INT ###
    def testIntDet(self):
        "Test intDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Series.intDet(matrix), -2)

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

    def testIntMin(self):
        "Test intMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.intMin(matrix), 4)

    def testIntScale(self):
        "Test intScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'i')
        Series.intScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testIntFloor(self):
        "Test intFloor function"
        matrix = N.array([[10,-2],[-6,7]],'i')
        Series.intFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testIntCeil(self):
        "Test intCeil function"
        matrix = N.array([[10,-2],[-6,7]],'i')
        Series.intCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testIntLUSplit(self):
        "Test intLUSplit function"
        lower, upper = Series.intLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ####################################################
    ### Test functions that take arrays of type UINT ###
    def testUintDet(self):
        "Test uintDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Series.uintDet(matrix), 2)

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

    def testUintMin(self):
        "Test uintMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.uintMin(matrix), 4)

    def testUintScale(self):
        "Test uintScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'I')
        Series.uintScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUintFloor(self):
        "Test uintFloor function"
        matrix = N.array([[10,2],[6,7]],'I')
        Series.uintFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUintCeil(self):
        "Test uintCeil function"
        matrix = N.array([[10,2],[6,7]],'I')
        Series.uintCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUintLUSplit(self):
        "Test uintLUSplit function"
        lower, upper = Series.uintLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ####################################################
    ### Test functions that take arrays of type LONG ###
    def testLongDet(self):
        "Test longDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Series.longDet(matrix), -2)

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

    def testLongMin(self):
        "Test longMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.longMin(matrix), 4)

    def testLongScale(self):
        "Test longScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'l')
        Series.longScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testLongFloor(self):
        "Test longFloor function"
        matrix = N.array([[10,-2],[-6,7]],'l')
        Series.longFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testLongCeil(self):
        "Test longCeil function"
        matrix = N.array([[10,-2],[-6,7]],'l')
        Series.longCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testLongLUSplit(self):
        "Test longLUSplit function"
        lower, upper = Series.longLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type ULONG ###
    def testUlongDet(self):
        "Test ulongDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Series.ulongDet(matrix), 2)

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

    def testUlongMin(self):
        "Test ulongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ulongMin(matrix), 4)

    def testUlongScale(self):
        "Test ulongScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'L')
        Series.ulongScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUlongFloor(self):
        "Test ulongFloor function"
        matrix = N.array([[10,2],[6,7]],'L')
        Series.ulongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUlongCeil(self):
        "Test ulongCeil function"
        matrix = N.array([[10,2],[6,7]],'L')
        Series.ulongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUlongLUSplit(self):
        "Test ulongLUSplit function"
        lower, upper = Series.ulongLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ########################################################
    ### Test functions that take arrays of type LONGLONG ###
    def testLongLongDet(self):
        "Test longLongDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Series.longLongDet(matrix), -2)

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

    def testLongLongMin(self):
        "Test longLongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.longLongMin(matrix), 4)

    def testLongLongScale(self):
        "Test longLongScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'q')
        Series.longLongScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testLongLongFloor(self):
        "Test longLongFloor function"
        matrix = N.array([[10,-2],[-6,7]],'q')
        Series.longLongFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testLongLongCeil(self):
        "Test longLongCeil function"
        matrix = N.array([[10,-2],[-6,7]],'q')
        Series.longLongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testLongLongLUSplit(self):
        "Test longLongLUSplit function"
        lower, upper = Series.longLongLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #########################################################
    ### Test functions that take arrays of type ULONGLONG ###
    def testUlongLongDet(self):
        "Test ulongLongDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Series.ulongLongDet(matrix), 2)

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

    def testUlongLongMin(self):
        "Test ulongLongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.ulongLongMin(matrix), 4)

    def testUlongLongScale(self):
        "Test ulongLongScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'Q')
        Series.ulongLongScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUlongLongFloor(self):
        "Test ulongLongFloor function"
        matrix = N.array([[10,2],[6,7]],'Q')
        Series.ulongLongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUlongLongCeil(self):
        "Test ulongLongCeil function"
        matrix = N.array([[10,2],[6,7]],'Q')
        Series.ulongLongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUlongLongLUSplit(self):
        "Test ulongLongLUSplit function"
        lower, upper = Series.ulongLongLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type FLOAT ###
    def testFloatDet(self):
        "Test floatDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Series.floatDet(matrix), -2)

    def testFloatZeros(self):
        "Test floatZeros function"
        myArray = N.ones(5,'f')
        Series.floatZeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testFloatEOSplit(self):
        "Test floatEOSplit function"
        even, odd = Series.floatEOSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testFloatTwos(self):
        "Test floatTwos function"
        twos = Series.floatTwos(5)
        self.assertEquals((twos == [2,2,2,2,2]).all(), True)

    def testFloatThrees(self):
        "Test floatThrees function"
        threes = Series.floatThrees(6)
        self.assertEquals((threes == [3,3,3,3,3,3]).all(), True)

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

    def testFloatMin(self):
        "Test floatMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.floatMin(matrix), 4)

    def testFloatScale(self):
        "Test floatScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'f')
        Series.floatScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testFloatFloor(self):
        "Test floatFloor function"
        matrix = N.array([[10,-2],[-6,7]],'f')
        Series.floatFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testFloatCeil(self):
        "Test floatCeil function"
        matrix = N.array([[10,-2],[-6,7]],'f')
        Series.floatCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testFloatLUSplit(self):
        "Test floatLUSplit function"
        lower, upper = Series.floatLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ######################################################
    ### Test functions that take arrays of type DOUBLE ###
    def testDoubleDet(self):
        "Test doubleDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Series.doubleDet(matrix), -2)

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

    def testDoubleMin(self):
        "Test doubleMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Series.doubleMin(matrix), 4)

    def testDoubleScale(self):
        "Test doubleScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'d')
        Series.doubleScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testDoubleFloor(self):
        "Test doubleFloor function"
        matrix = N.array([[10,-2],[-6,7]],'d')
        Series.doubleFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testDoubleCeil(self):
        "Test doubleCeil function"
        matrix = N.array([[10,-2],[-6,7]],'d')
        Series.doubleCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testDoubleLUSplit(self):
        "Test doubleLUSplit function"
        lower, upper = Series.doubleLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SeriesTestCase))

    # Execute the test suite
    print "Testing 2D Functions of Module Series"
    print "NumPy version", N.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
