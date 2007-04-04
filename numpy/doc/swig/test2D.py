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
import Matrix

######################################################################

class MatrixTestCase(unittest.TestCase):

    ####################################################
    ### Test functions that take arrays of type BYTE ###
    def testScharDet(self):
        "Test scharDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Matrix.scharDet(matrix), -2)

    def testScharMax(self):
        "Test scharMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Matrix.scharMax(matrix), 5)

    def testScharMaxNonContainer(self):
        "Test scharMax function with None"
        self.assertRaises(TypeError, Matrix.scharMax, None)

    def testScharMaxWrongDim(self):
        "Test scharMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.scharMax, [0, -1, 2, -3])

    def testScharMin(self):
        "Test scharMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.scharMin(matrix), 4)

    def testScharScale(self):
        "Test scharScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'b')
        Matrix.scharScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testScharFloor(self):
        "Test scharFloor function"
        matrix = N.array([[10,-2],[-6,7]],'b')
        Matrix.scharFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testScharCeil(self):
        "Test scharCeil function"
        matrix = N.array([[10,-2],[-6,7]],'b')
        Matrix.scharCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testScharLUSplit(self):
        "Test scharLUSplit function"
        lower, upper = Matrix.scharLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type UBYTE ###
    def testUcharDet(self):
        "Test ucharDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Matrix.ucharDet(matrix), 2)

    def testUcharMax(self):
        "Test ucharMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Matrix.ucharMax(matrix), 6)

    def testUcharMaxNonContainer(self):
        "Test ucharMax function with None"
        self.assertRaises(TypeError, Matrix.ucharMax, None)

    def testUcharMaxWrongDim(self):
        "Test ucharMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.ucharMax, [0, 1, 2, 3])

    def testUcharMin(self):
        "Test ucharMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.ucharMin(matrix), 4)

    def testUcharScale(self):
        "Test ucharScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'B')
        Matrix.ucharScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUcharFloor(self):
        "Test ucharFloor function"
        matrix = N.array([[10,2],[6,7]],'B')
        Matrix.ucharFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUcharCeil(self):
        "Test ucharCeil function"
        matrix = N.array([[10,2],[6,7]],'B')
        Matrix.ucharCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUcharLUSplit(self):
        "Test ucharLUSplit function"
        lower, upper = Matrix.ucharLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type SHORT ###
    def testShortDet(self):
        "Test shortDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Matrix.shortDet(matrix), -2)

    def testShortMax(self):
        "Test shortMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Matrix.shortMax(matrix), 5)

    def testShortMaxNonContainer(self):
        "Test shortMax function with None"
        self.assertRaises(TypeError, Matrix.shortMax, None)

    def testShortMaxWrongDim(self):
        "Test shortMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.shortMax, [0, -1, 2, -3])

    def testShortMin(self):
        "Test shortMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.shortMin(matrix), 4)

    def testShortScale(self):
        "Test shortScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'h')
        Matrix.shortScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testShortFloor(self):
        "Test shortFloor function"
        matrix = N.array([[10,-2],[-6,7]],'h')
        Matrix.shortFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testShortCeil(self):
        "Test shortCeil function"
        matrix = N.array([[10,-2],[-6,7]],'h')
        Matrix.shortCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testShortLUSplit(self):
        "Test shortLUSplit function"
        lower, upper = Matrix.shortLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ######################################################
    ### Test functions that take arrays of type USHORT ###
    def testUshortDet(self):
        "Test ushortDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Matrix.ushortDet(matrix), 2)

    def testUshortMax(self):
        "Test ushortMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Matrix.ushortMax(matrix), 6)

    def testUshortMaxNonContainer(self):
        "Test ushortMax function with None"
        self.assertRaises(TypeError, Matrix.ushortMax, None)

    def testUshortMaxWrongDim(self):
        "Test ushortMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.ushortMax, [0, 1, 2, 3])

    def testUshortMin(self):
        "Test ushortMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.ushortMin(matrix), 4)

    def testUshortScale(self):
        "Test ushortScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'H')
        Matrix.ushortScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUshortFloor(self):
        "Test ushortFloor function"
        matrix = N.array([[10,2],[6,7]],'H')
        Matrix.ushortFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUshortCeil(self):
        "Test ushortCeil function"
        matrix = N.array([[10,2],[6,7]],'H')
        Matrix.ushortCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUshortLUSplit(self):
        "Test ushortLUSplit function"
        lower, upper = Matrix.ushortLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ###################################################
    ### Test functions that take arrays of type INT ###
    def testIntDet(self):
        "Test intDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Matrix.intDet(matrix), -2)

    def testIntMax(self):
        "Test intMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Matrix.intMax(matrix), 5)

    def testIntMaxNonContainer(self):
        "Test intMax function with None"
        self.assertRaises(TypeError, Matrix.intMax, None)

    def testIntMaxWrongDim(self):
        "Test intMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.intMax, [0, -1, 2, -3])

    def testIntMin(self):
        "Test intMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.intMin(matrix), 4)

    def testIntScale(self):
        "Test intScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'i')
        Matrix.intScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testIntFloor(self):
        "Test intFloor function"
        matrix = N.array([[10,-2],[-6,7]],'i')
        Matrix.intFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testIntCeil(self):
        "Test intCeil function"
        matrix = N.array([[10,-2],[-6,7]],'i')
        Matrix.intCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testIntLUSplit(self):
        "Test intLUSplit function"
        lower, upper = Matrix.intLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ####################################################
    ### Test functions that take arrays of type UINT ###
    def testUintDet(self):
        "Test uintDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Matrix.uintDet(matrix), 2)

    def testUintMax(self):
        "Test uintMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Matrix.uintMax(matrix), 6)

    def testUintMaxNonContainer(self):
        "Test uintMax function with None"
        self.assertRaises(TypeError, Matrix.uintMax, None)

    def testUintMaxWrongDim(self):
        "Test uintMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.uintMax, [0, 1, 2, 3])

    def testUintMin(self):
        "Test uintMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.uintMin(matrix), 4)

    def testUintScale(self):
        "Test uintScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'I')
        Matrix.uintScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUintFloor(self):
        "Test uintFloor function"
        matrix = N.array([[10,2],[6,7]],'I')
        Matrix.uintFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUintCeil(self):
        "Test uintCeil function"
        matrix = N.array([[10,2],[6,7]],'I')
        Matrix.uintCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUintLUSplit(self):
        "Test uintLUSplit function"
        lower, upper = Matrix.uintLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ####################################################
    ### Test functions that take arrays of type LONG ###
    def testLongDet(self):
        "Test longDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Matrix.longDet(matrix), -2)

    def testLongMax(self):
        "Test longMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Matrix.longMax(matrix), 5)

    def testLongMaxNonContainer(self):
        "Test longMax function with None"
        self.assertRaises(TypeError, Matrix.longMax, None)

    def testLongMaxWrongDim(self):
        "Test longMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.longMax, [0, -1, 2, -3])

    def testLongMin(self):
        "Test longMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.longMin(matrix), 4)

    def testLongScale(self):
        "Test longScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'l')
        Matrix.longScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testLongFloor(self):
        "Test longFloor function"
        matrix = N.array([[10,-2],[-6,7]],'l')
        Matrix.longFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testLongCeil(self):
        "Test longCeil function"
        matrix = N.array([[10,-2],[-6,7]],'l')
        Matrix.longCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testLongLUSplit(self):
        "Test longLUSplit function"
        lower, upper = Matrix.longLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type ULONG ###
    def testUlongDet(self):
        "Test ulongDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Matrix.ulongDet(matrix), 2)

    def testUlongMax(self):
        "Test ulongMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Matrix.ulongMax(matrix), 6)

    def testUlongMaxNonContainer(self):
        "Test ulongMax function with None"
        self.assertRaises(TypeError, Matrix.ulongMax, None)

    def testUlongMaxWrongDim(self):
        "Test ulongMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.ulongMax, [0, 1, 2, 3])

    def testUlongMin(self):
        "Test ulongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.ulongMin(matrix), 4)

    def testUlongScale(self):
        "Test ulongScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'L')
        Matrix.ulongScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUlongFloor(self):
        "Test ulongFloor function"
        matrix = N.array([[10,2],[6,7]],'L')
        Matrix.ulongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUlongCeil(self):
        "Test ulongCeil function"
        matrix = N.array([[10,2],[6,7]],'L')
        Matrix.ulongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUlongLUSplit(self):
        "Test ulongLUSplit function"
        lower, upper = Matrix.ulongLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ########################################################
    ### Test functions that take arrays of type LONGLONG ###
    def testLongLongDet(self):
        "Test longLongDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Matrix.longLongDet(matrix), -2)

    def testLongLongMax(self):
        "Test longLongMax function"
        matrix = [[-6,5,-4],[3,-2,1]]
        self.assertEquals(Matrix.longLongMax(matrix), 5)

    def testLongLongMaxNonContainer(self):
        "Test longLongMax function with None"
        self.assertRaises(TypeError, Matrix.longLongMax, None)

    def testLongLongMaxWrongDim(self):
        "Test longLongMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.longLongMax, [0, -1, 2, -3])

    def testLongLongMin(self):
        "Test longLongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.longLongMin(matrix), 4)

    def testLongLongScale(self):
        "Test longLongScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'q')
        Matrix.longLongScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testLongLongFloor(self):
        "Test longLongFloor function"
        matrix = N.array([[10,-2],[-6,7]],'q')
        Matrix.longLongFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testLongLongCeil(self):
        "Test longLongCeil function"
        matrix = N.array([[10,-2],[-6,7]],'q')
        Matrix.longLongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testLongLongLUSplit(self):
        "Test longLongLUSplit function"
        lower, upper = Matrix.longLongLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #########################################################
    ### Test functions that take arrays of type ULONGLONG ###
    def testUlongLongDet(self):
        "Test ulongLongDet function"
        matrix = [[7,6],[9,8]]
        self.assertEquals(Matrix.ulongLongDet(matrix), 2)

    def testUlongLongMax(self):
        "Test ulongLongMax function"
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(Matrix.ulongLongMax(matrix), 6)

    def testUlongLongMaxNonContainer(self):
        "Test ulongLongMax function with None"
        self.assertRaises(TypeError, Matrix.ulongLongMax, None)

    def testUlongLongMaxWrongDim(self):
        "Test ulongLongMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.ulongLongMax, [0, 1, 2, 3])

    def testUlongLongMin(self):
        "Test ulongLongMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.ulongLongMin(matrix), 4)

    def testUlongLongScale(self):
        "Test ulongLongScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'Q')
        Matrix.ulongLongScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testUlongLongFloor(self):
        "Test ulongLongFloor function"
        matrix = N.array([[10,2],[6,7]],'Q')
        Matrix.ulongLongFloor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[10,7],[7,7]]))

    def testUlongLongCeil(self):
        "Test ulongLongCeil function"
        matrix = N.array([[10,2],[6,7]],'Q')
        Matrix.ulongLongCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,2],[5,5]]))

    def testUlongLongLUSplit(self):
        "Test ulongLongLUSplit function"
        lower, upper = Matrix.ulongLongLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type FLOAT ###
    def testFloatDet(self):
        "Test floatDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Matrix.floatDet(matrix), -2)

    def testFloatMax(self):
        "Test floatMax function"
        matrix = [[-6,5,-4],[3.14,-2.718,1]]
        self.assertEquals(Matrix.floatMax(matrix), 5.0)

    def testFloatMaxNonContainer(self):
        "Test floatMax function with None"
        self.assertRaises(TypeError, Matrix.floatMax, None)

    def testFloatMaxWrongDim(self):
        "Test floatMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.floatMax, [0.0, -1, 2.718, -3.14])

    def testFloatMin(self):
        "Test floatMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.floatMin(matrix), 4)

    def testFloatScale(self):
        "Test floatScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'f')
        Matrix.floatScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testFloatFloor(self):
        "Test floatFloor function"
        matrix = N.array([[10,-2],[-6,7]],'f')
        Matrix.floatFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testFloatCeil(self):
        "Test floatCeil function"
        matrix = N.array([[10,-2],[-6,7]],'f')
        Matrix.floatCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testFloatLUSplit(self):
        "Test floatLUSplit function"
        lower, upper = Matrix.floatLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

    ######################################################
    ### Test functions that take arrays of type DOUBLE ###
    def testDoubleDet(self):
        "Test doubleDet function"
        matrix = [[6,7],[8,9]]
        self.assertEquals(Matrix.doubleDet(matrix), -2)

    def testDoubleMax(self):
        "Test doubleMax function"
        matrix = [[-6,5,-4],[3.14,-2.718,1]]
        self.assertEquals(Matrix.doubleMax(matrix), 5.0)

    def testDoubleMaxNonContainer(self):
        "Test doubleMax function with None"
        self.assertRaises(TypeError, Matrix.doubleMax, None)

    def testDoubleMaxWrongDim(self):
        "Test doubleMax function with a 1D array"
        self.assertRaises(TypeError, Matrix.doubleMax, [0.0, -1, 2.718, -3.14])

    def testDoubleMin(self):
        "Test doubleMin function"
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(Matrix.doubleMin(matrix), 4)

    def testDoubleScale(self):
        "Test doubleScale function"
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'d')
        Matrix.doubleScale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    def testDoubleFloor(self):
        "Test doubleFloor function"
        matrix = N.array([[10,-2],[-6,7]],'d')
        Matrix.doubleFloor(matrix,0)
        N.testing.assert_array_equal(matrix, N.array([[10,0],[0,7]]))

    def testDoubleCeil(self):
        "Test doubleCeil function"
        matrix = N.array([[10,-2],[-6,7]],'d')
        Matrix.doubleCeil(matrix,5)
        N.testing.assert_array_equal(matrix, N.array([[5,-2],[-6,5]]))

    def testDoubleLUSplit(self):
        "Test doubleLUSplit function"
        lower, upper = Matrix.doubleLUSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MatrixTestCase))

    # Execute the test suite
    print "Testing 2D Functions of Module Matrix"
    print "NumPy version", N.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
