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

    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"

    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDet(self):
        "Test det function"
        print >>sys.stderr, self.typeStr, "... ",
        det = Matrix.__dict__[self.typeStr + "Det"]
        matrix = [[8,7],[6,9]]
        self.assertEquals(det(matrix), 30)

    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDetWrongDim(self):
        "Test det function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        det = Matrix.__dict__[self.typeStr + "Det"]
        matrix = [8,7]
        self.assertRaises(TypeError, det, matrix)

    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDetWrongSize(self):
        "Test det function with wrong size"
        print >>sys.stderr, self.typeStr, "... ",
        det = Matrix.__dict__[self.typeStr + "Det"]
        matrix = [[8,7,6], [5,4,3], [2,1,0]]
        self.assertRaises(TypeError, det, matrix)

    # Test (type IN_ARRAY2[ANY][ANY]) typemap
    def testDetNonContainer(self):
        "Test det function with non-container"
        print >>sys.stderr, self.typeStr, "... ",
        det = Matrix.__dict__[self.typeStr + "Det"]
        self.assertRaises(TypeError, det, None)

    # Test (type* IN_ARRAY2, int DIM1, int DIM2) typemap
    def testMax(self):
        "Test max function"
        print >>sys.stderr, self.typeStr, "... ",
        max = Matrix.__dict__[self.typeStr + "Max"]
        matrix = [[6,5,4],[3,2,1]]
        self.assertEquals(max(matrix), 6)

    # Test (type* IN_ARRAY2, int DIM1, int DIM2) typemap
    def testMaxNonContainer(self):
        "Test max function with non-container"
        print >>sys.stderr, self.typeStr, "... ",
        max = Matrix.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, None)

    # Test (type* IN_ARRAY2, int DIM1, int DIM2) typemap
    def testMaxWrongDim(self):
        "Test max function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        max = Matrix.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, [0, 1, 2, 3])

    # Test (int DIM1, int DIM2, type* IN_ARRAY2) typemap
    def testMin(self):
        "Test min function"
        print >>sys.stderr, self.typeStr, "... ",
        min = Matrix.__dict__[self.typeStr + "Min"]
        matrix = [[9,8],[7,6],[5,4]]
        self.assertEquals(min(matrix), 4)

    # Test (int DIM1, int DIM2, type* IN_ARRAY2) typemap
    def testMinWrongDim(self):
        "Test min function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        min = Matrix.__dict__[self.typeStr + "Min"]
        self.assertRaises(TypeError, min, [1,3,5,7,9])

    # Test (int DIM1, int DIM2, type* IN_ARRAY2) typemap
    def testMinNonContainer(self):
        "Test min function with non-container"
        print >>sys.stderr, self.typeStr, "... ",
        min = Matrix.__dict__[self.typeStr + "Min"]
        self.assertRaises(TypeError, min, False)

    # Test (type INPLACE_ARRAY2[ANY][ANY]) typemap
    def testScale(self):
        "Test scale function"
        print >>sys.stderr, self.typeStr, "... ",
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],self.typeCode)
        scale(matrix,4)
        self.assertEquals((matrix == [[4,8,12],[8,4,8],[12,8,4]]).all(), True)

    # Test (type INPLACE_ARRAY2[ANY][ANY]) typemap
    def testScaleWrongDim(self):
        "Test scale function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        matrix = N.array([1,2,2,1],self.typeCode)
        self.assertRaises(TypeError, scale, matrix)

    # Test (type INPLACE_ARRAY2[ANY][ANY]) typemap
    def testScaleWrongSize(self):
        "Test scale function with wrong size"
        print >>sys.stderr, self.typeStr, "... ",
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        matrix = N.array([[1,2],[2,1]],self.typeCode)
        self.assertRaises(TypeError, scale, matrix)

    # Test (type INPLACE_ARRAY2[ANY][ANY]) typemap
    def testScaleWrongType(self):
        "Test scale function with wrong type"
        print >>sys.stderr, self.typeStr, "... ",
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        matrix = N.array([[1,2,3],[2,1,2],[3,2,1]],'c')
        self.assertRaises(TypeError, scale, matrix)

    # Test (type INPLACE_ARRAY2[ANY][ANY]) typemap
    def testScaleNonArray(self):
        "Test scale function with non-array"
        print >>sys.stderr, self.typeStr, "... ",
        scale = Matrix.__dict__[self.typeStr + "Scale"]
        matrix = [[1,2,3],[2,1,2],[3,2,1]]
        self.assertRaises(TypeError, scale, matrix)

    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloor(self):
        "Test floor function"
        print >>sys.stderr, self.typeStr, "... ",
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        matrix = N.array([[6,7],[8,9]],self.typeCode)
        floor(matrix,7)
        N.testing.assert_array_equal(matrix, N.array([[7,7],[8,9]]))

    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloorWrongDim(self):
        "Test floor function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        matrix = N.array([6,7,8,9],self.typeCode)
        self.assertRaises(TypeError, floor, matrix)

    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloorWrongType(self):
        "Test floor function with wrong type"
        print >>sys.stderr, self.typeStr, "... ",
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        matrix = N.array([[6,7], [8,9]],'c')
        self.assertRaises(TypeError, floor, matrix)

    # Test (type* INPLACE_ARRAY2, int DIM1, int DIM2) typemap
    def testFloorNonArray(self):
        "Test floor function with non-array"
        print >>sys.stderr, self.typeStr, "... ",
        floor = Matrix.__dict__[self.typeStr + "Floor"]
        matrix = [[6,7], [8,9]]
        self.assertRaises(TypeError, floor, matrix)

    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeil(self):
        "Test ceil function"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        matrix = N.array([[1,2],[3,4]],self.typeCode)
        ceil(matrix,3)
        N.testing.assert_array_equal(matrix, N.array([[1,2],[3,3]]))

    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeilWrongDim(self):
        "Test ceil function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        matrix = N.array([1,2,3,4],self.typeCode)
        self.assertRaises(TypeError, ceil, matrix)

    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeilWrongType(self):
        "Test ceil function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        matrix = N.array([[1,2], [3,4]],'c')
        self.assertRaises(TypeError, ceil, matrix)

    # Test (int DIM1, int DIM2, type* INPLACE_ARRAY2) typemap
    def testCeilNonArray(self):
        "Test ceil function with non-array"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = Matrix.__dict__[self.typeStr + "Ceil"]
        matrix = [[1,2], [3,4]]
        self.assertRaises(TypeError, ceil, matrix)

    # Test (type ARGOUT_ARRAY2[ANY][ANY]) typemap
    def testLUSplit(self):
        "Test luSplit function"
        print >>sys.stderr, self.typeStr, "... ",
        luSplit = Matrix.__dict__[self.typeStr + "LUSplit"]
        lower, upper = luSplit([[1,2,3],[4,5,6],[7,8,9]])
        self.assertEquals((lower == [[1,0,0],[4,5,0],[7,8,9]]).all(), True)
        self.assertEquals((upper == [[0,2,3],[0,0,6],[0,0,0]]).all(), True)

######################################################################

class scharTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "schar"
        self.typeCode = "b"

######################################################################

class ucharTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "uchar"
        self.typeCode = "B"

######################################################################

class shortTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "short"
        self.typeCode = "h"

######################################################################

class ushortTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "ushort"
        self.typeCode = "H"

######################################################################

class intTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "int"
        self.typeCode = "i"

######################################################################

class uintTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "uint"
        self.typeCode = "I"

######################################################################

class longTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "long"
        self.typeCode = "l"

######################################################################

class ulongTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "ulong"
        self.typeCode = "L"

######################################################################

class longLongTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "longLong"
        self.typeCode = "q"

######################################################################

class ulongLongTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "ulongLong"
        self.typeCode = "Q"

######################################################################

class floatTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "float"
        self.typeCode = "f"

######################################################################

class doubleTestCase(MatrixTestCase):
    def __init__(self, methodName="runTest"):
        MatrixTestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(    scharTestCase))
    suite.addTest(unittest.makeSuite(    ucharTestCase))
    suite.addTest(unittest.makeSuite(    shortTestCase))
    suite.addTest(unittest.makeSuite(   ushortTestCase))
    suite.addTest(unittest.makeSuite(      intTestCase))
    suite.addTest(unittest.makeSuite(     uintTestCase))
    suite.addTest(unittest.makeSuite(     longTestCase))
    suite.addTest(unittest.makeSuite(    ulongTestCase))
    suite.addTest(unittest.makeSuite( longLongTestCase))
    suite.addTest(unittest.makeSuite(ulongLongTestCase))
    suite.addTest(unittest.makeSuite(    floatTestCase))
    suite.addTest(unittest.makeSuite(   doubleTestCase))

    # Execute the test suite
    print "Testing 2D Functions of Module Matrix"
    print "NumPy version", N.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
