#! /usr/bin/env python
from __future__ import division, absolute_import, print_function

# System imports
from   distutils.util import get_platform
from   math           import sqrt
import os
import sys
import unittest

# Import NumPy
import numpy as np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

import Tensor

######################################################################

class TensorTestCase(unittest.TestCase):

    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"
        self.result   = sqrt(28.0/8)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNorm(self):
        "Test norm function"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        tensor = [[[0, 1], [2, 3]],
                  [[3, 2], [1, 0]]]
        if isinstance(self.result, int):
            self.assertEquals(norm(tensor), self.result)
        else:
            self.assertAlmostEqual(norm(tensor), self.result, 6)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormBadList(self):
        "Test norm function with bad list"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        tensor = [[[0, "one"], [2, 3]],
                  [[3, "two"], [1, 0]]]
        self.assertRaises(BadListError, norm, tensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormWrongDim(self):
        "Test norm function with wrong dimensions"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        tensor = [[0, 1, 2, 3],
                  [3, 2, 1, 0]]
        self.assertRaises(TypeError, norm, tensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormWrongSize(self):
        "Test norm function with wrong size"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        tensor = [[[0, 1, 0], [2, 3, 2]],
                  [[3, 2, 3], [1, 0, 1]]]
        self.assertRaises(TypeError, norm, tensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormNonContainer(self):
        "Test norm function with non-container"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        self.assertRaises(TypeError, norm, None)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMax(self):
        "Test max function"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + "Max"]
        tensor = [[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]]
        self.assertEquals(max(tensor), 8)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxBadList(self):
        "Test max function with bad list"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + "Max"]
        tensor = [[[1, "two"], [3, 4]],
                  [[5, "six"], [7, 8]]]
        self.assertRaises(BadListError, max, tensor)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxNonContainer(self):
        "Test max function with non-container"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, None)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxWrongDim(self):
        "Test max function with wrong dimensions"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, [0, -1, 2, -3])

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMin(self):
        "Test min function"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + "Min"]
        tensor = [[[9, 8], [7, 6]],
                  [[5, 4], [3, 2]]]
        self.assertEquals(min(tensor), 2)

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinBadList(self):
        "Test min function with bad list"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + "Min"]
        tensor = [[["nine", 8], [7, 6]],
                  [["five", 4], [3, 2]]]
        self.assertRaises(BadListError, min, tensor)

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinNonContainer(self):
        "Test min function with non-container"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + "Min"]
        self.assertRaises(TypeError, min, True)

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinWrongDim(self):
        "Test min function with wrong dimensions"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + "Min"]
        self.assertRaises(TypeError, min, [[1, 3], [5, 7]])

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScale(self):
        "Test scale function"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        tensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                          [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                          [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], self.typeCode)
        scale(tensor, 4)
        self.assertEquals((tensor == [[[4, 0, 4], [0, 4, 0], [4, 0, 4]],
                                      [[0, 4, 0], [4, 0, 4], [0, 4, 0]],
                                      [[4, 0, 4], [0, 4, 0], [4, 0, 4]]]).all(), True)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleWrongType(self):
        "Test scale function with wrong type"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        tensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                          [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                          [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], 'c')
        self.assertRaises(TypeError, scale, tensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleWrongDim(self):
        "Test scale function with wrong dimensions"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        tensor = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1],
                          [0, 1, 0], [1, 0, 1], [0, 1, 0]], self.typeCode)
        self.assertRaises(TypeError, scale, tensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleWrongSize(self):
        "Test scale function with wrong size"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        tensor = np.array([[[1, 0], [0, 1], [1, 0]],
                          [[0, 1], [1, 0], [0, 1]],
                          [[1, 0], [0, 1], [1, 0]]], self.typeCode)
        self.assertRaises(TypeError, scale, tensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleNonArray(self):
        "Test scale function with non-array"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        self.assertRaises(TypeError, scale, True)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloor(self):
        "Test floor function"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        tensor = np.array([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]], self.typeCode)
        floor(tensor, 4)
        np.testing.assert_array_equal(tensor, np.array([[[4, 4], [4, 4]],
                                                      [[5, 6], [7, 8]]]))

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloorWrongType(self):
        "Test floor function with wrong type"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        tensor = np.array([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]], 'c')
        self.assertRaises(TypeError, floor, tensor)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloorWrongDim(self):
        "Test floor function with wrong type"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        tensor = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], self.typeCode)
        self.assertRaises(TypeError, floor, tensor)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloorNonArray(self):
        "Test floor function with non-array"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        self.assertRaises(TypeError, floor, object)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeil(self):
        "Test ceil function"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        tensor = np.array([[[9, 8], [7, 6]],
                          [[5, 4], [3, 2]]], self.typeCode)
        ceil(tensor, 5)
        np.testing.assert_array_equal(tensor, np.array([[[5, 5], [5, 5]],
                                                      [[5, 4], [3, 2]]]))

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilWrongType(self):
        "Test ceil function with wrong type"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        tensor = np.array([[[9, 8], [7, 6]],
                          [[5, 4], [3, 2]]], 'c')
        self.assertRaises(TypeError, ceil, tensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilWrongDim(self):
        "Test ceil function with wrong dimensions"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        tensor = np.array([[9, 8], [7, 6], [5, 4], [3, 2]], self.typeCode)
        self.assertRaises(TypeError, ceil, tensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilNonArray(self):
        "Test ceil function with non-array"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        tensor = [[[9, 8], [7, 6]],
                  [[5, 4], [3, 2]]]
        self.assertRaises(TypeError, ceil, tensor)

    # Test (type ARGOUT_ARRAY3[ANY][ANY][ANY]) typemap
    def testLUSplit(self):
        "Test luSplit function"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        luSplit = Tensor.__dict__[self.typeStr + "LUSplit"]
        lower, upper = luSplit([[[1, 1], [1, 1]],
                                [[1, 1], [1, 1]]])
        self.assertEquals((lower == [[[1, 1], [1, 0]],
                                     [[1, 0], [0, 0]]]).all(), True)
        self.assertEquals((upper == [[[0, 0], [0, 1]],
                                     [[0, 1], [1, 1]]]).all(), True)

######################################################################

class scharTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "schar"
        self.typeCode = "b"
        self.result   = int(self.result)

######################################################################

class ucharTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "uchar"
        self.typeCode = "B"
        self.result   = int(self.result)

######################################################################

class shortTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "short"
        self.typeCode = "h"
        self.result   = int(self.result)

######################################################################

class ushortTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "ushort"
        self.typeCode = "H"
        self.result   = int(self.result)

######################################################################

class intTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "int"
        self.typeCode = "i"
        self.result   = int(self.result)

######################################################################

class uintTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "uint"
        self.typeCode = "I"
        self.result   = int(self.result)

######################################################################

class longTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "long"
        self.typeCode = "l"
        self.result   = int(self.result)

######################################################################

class ulongTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "ulong"
        self.typeCode = "L"
        self.result   = int(self.result)

######################################################################

class longLongTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "longLong"
        self.typeCode = "q"
        self.result   = int(self.result)

######################################################################

class ulongLongTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "ulongLong"
        self.typeCode = "Q"
        self.result   = int(self.result)

######################################################################

class floatTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
        self.typeStr  = "float"
        self.typeCode = "f"

######################################################################

class doubleTestCase(TensorTestCase):
    def __init__(self, methodName="runTest"):
        TensorTestCase.__init__(self, methodName)
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
    print("Testing 3D Functions of Module Tensor")
    print("NumPy version", np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
