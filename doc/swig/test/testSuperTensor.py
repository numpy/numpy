#! /usr/bin/env python
from __future__ import division

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

import SuperTensor

######################################################################

class SuperTensorTestCase(unittest.TestCase):

    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNorm(self):
        "Test norm function"
        print >>sys.stderr, self.typeStr, "... ",
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]
        supertensor = np.arange(2*2*2*2,dtype=self.typeCode).reshape((2,2,2,2))
        #Note: cludge to get an answer of the same type as supertensor.
        #Answer is simply sqrt(sum(supertensor*supertensor)/16)
        answer = np.array([np.sqrt(np.sum(supertensor.astype('d')*supertensor)/16.)],dtype=self.typeCode)[0]
        self.assertAlmostEqual(norm(supertensor), answer, 6)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormBadList(self):
        "Test norm function with bad list"
        print >>sys.stderr, self.typeStr, "... ",
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]
        supertensor = [[[[0,"one"],[2,3]], [[3,"two"],[1,0]]],[[[0,"one"],[2,3]], [[3,"two"],[1,0]]]]
        self.assertRaises(BadListError, norm, supertensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormWrongDim(self):
        "Test norm function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]
        supertensor = np.arange(2*2*2,dtype=self.typeCode).reshape((2,2,2))
        self.assertRaises(TypeError, norm, supertensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormWrongSize(self):
        "Test norm function with wrong size"
        print >>sys.stderr, self.typeStr, "... ",
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]
        supertensor = np.arange(3*2*2,dtype=self.typeCode).reshape((3,2,2))
        self.assertRaises(TypeError, norm, supertensor)

    # Test (type IN_ARRAY3[ANY][ANY][ANY]) typemap
    def testNormNonContainer(self):
        "Test norm function with non-container"
        print >>sys.stderr, self.typeStr, "... ",
        norm = SuperTensor.__dict__[self.typeStr + "Norm"]
        self.assertRaises(TypeError, norm, None)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMax(self):
        "Test max function"
        print >>sys.stderr, self.typeStr, "... ",
        max = SuperTensor.__dict__[self.typeStr + "Max"]
        supertensor = [[[[1,2], [3,4]], [[5,6], [7,8]]],[[[1,2], [3,4]], [[5,6], [7,8]]]]
        self.assertEquals(max(supertensor), 8)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxBadList(self):
        "Test max function with bad list"
        print >>sys.stderr, self.typeStr, "... ",
        max = SuperTensor.__dict__[self.typeStr + "Max"]
        supertensor = [[[[1,"two"], [3,4]], [[5,"six"], [7,8]]],[[[1,"two"], [3,4]], [[5,"six"], [7,8]]]]
        self.assertRaises(BadListError, max, supertensor)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxNonContainer(self):
        "Test max function with non-container"
        print >>sys.stderr, self.typeStr, "... ",
        max = SuperTensor.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, None)

    # Test (type* IN_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testMaxWrongDim(self):
        "Test max function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        max = SuperTensor.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, [0, -1, 2, -3])

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMin(self):
        "Test min function"
        print >>sys.stderr, self.typeStr, "... ",
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        supertensor = [[[[9,8], [7,6]], [[5,4], [3,2]]],[[[9,8], [7,6]], [[5,4], [3,2]]]]
        self.assertEquals(min(supertensor), 2)

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinBadList(self):
        "Test min function with bad list"
        print >>sys.stderr, self.typeStr, "... ",
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        supertensor = [[[["nine",8], [7,6]], [["five",4], [3,2]]],[[["nine",8], [7,6]], [["five",4], [3,2]]]]
        self.assertRaises(BadListError, min, supertensor)

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinNonContainer(self):
        "Test min function with non-container"
        print >>sys.stderr, self.typeStr, "... ",
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        self.assertRaises(TypeError, min, True)

    # Test (int DIM1, int DIM2, int DIM3, type* IN_ARRAY3) typemap
    def testMinWrongDim(self):
        "Test min function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        min = SuperTensor.__dict__[self.typeStr + "Min"]
        self.assertRaises(TypeError, min, [[1,3],[5,7]])

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScale(self):
        "Test scale function"
        print >>sys.stderr, self.typeStr, "... ",
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        supertensor = np.arange(3*3*3*3,dtype=self.typeCode).reshape((3,3,3,3))
        answer = supertensor.copy()*4
        scale(supertensor,4)
        self.assertEquals((supertensor == answer).all(), True)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleWrongType(self):
        "Test scale function with wrong type"
        print >>sys.stderr, self.typeStr, "... ",
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        supertensor = np.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'c')
        self.assertRaises(TypeError, scale, supertensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleWrongDim(self):
        "Test scale function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        supertensor = np.array([[1,0,1], [0,1,0], [1,0,1],
                          [0,1,0], [1,0,1], [0,1,0]],self.typeCode)
        self.assertRaises(TypeError, scale, supertensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleWrongSize(self):
        "Test scale function with wrong size"
        print >>sys.stderr, self.typeStr, "... ",
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        supertensor = np.array([[[1,0], [0,1], [1,0]],
                          [[0,1], [1,0], [0,1]],
                          [[1,0], [0,1], [1,0]]],self.typeCode)
        self.assertRaises(TypeError, scale, supertensor)

    # Test (type INPLACE_ARRAY3[ANY][ANY][ANY]) typemap
    def testScaleNonArray(self):
        "Test scale function with non-array"
        print >>sys.stderr, self.typeStr, "... ",
        scale = SuperTensor.__dict__[self.typeStr + "Scale"]
        self.assertRaises(TypeError, scale, True)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloor(self):
        "Test floor function"
        print >>sys.stderr, self.typeStr, "... ",
        supertensor = np.arange(2*2*2*2,dtype=self.typeCode).reshape((2,2,2,2))
        answer = supertensor.copy()
        answer[answer < 4] = 4

        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        floor(supertensor,4)
        np.testing.assert_array_equal(supertensor, answer)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloorWrongType(self):
        "Test floor function with wrong type"
        print >>sys.stderr, self.typeStr, "... ",
        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        supertensor = np.ones(2*2*2*2,dtype='c').reshape((2,2,2,2))
        self.assertRaises(TypeError, floor, supertensor)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloorWrongDim(self):
        "Test floor function with wrong type"
        print >>sys.stderr, self.typeStr, "... ",
        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        supertensor = np.arange(2*2*2,dtype=self.typeCode).reshape((2,2,2))
        self.assertRaises(TypeError, floor, supertensor)

    # Test (type* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) typemap
    def testFloorNonArray(self):
        "Test floor function with non-array"
        print >>sys.stderr, self.typeStr, "... ",
        floor = SuperTensor.__dict__[self.typeStr + "Floor"]
        self.assertRaises(TypeError, floor, object)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeil(self):
        "Test ceil function"
        print >>sys.stderr, self.typeStr, "... ",
        supertensor = np.arange(2*2*2*2,dtype=self.typeCode).reshape((2,2,2,2))
        answer = supertensor.copy()
        answer[answer > 5] = 5
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        ceil(supertensor,5)
        np.testing.assert_array_equal(supertensor, answer)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilWrongType(self):
        "Test ceil function with wrong type"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        supertensor = np.ones(2*2*2*2,'c').reshape((2,2,2,2))
        self.assertRaises(TypeError, ceil, supertensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilWrongDim(self):
        "Test ceil function with wrong dimensions"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        supertensor = np.arange(2*2*2,dtype=self.typeCode).reshape((2,2,2))
        self.assertRaises(TypeError, ceil, supertensor)

    # Test (int DIM1, int DIM2, int DIM3, type* INPLACE_ARRAY3) typemap
    def testCeilNonArray(self):
        "Test ceil function with non-array"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = SuperTensor.__dict__[self.typeStr + "Ceil"]
        supertensor = np.arange(2*2*2*2,dtype=self.typeCode).reshape((2,2,2,2)).tolist()
        self.assertRaises(TypeError, ceil, supertensor)

    # Test (type ARGOUT_ARRAY3[ANY][ANY][ANY]) typemap
    def testLUSplit(self):
        "Test luSplit function"
        print >>sys.stderr, self.typeStr, "... ",
        luSplit = SuperTensor.__dict__[self.typeStr + "LUSplit"]
        supertensor = np.ones(2*2*2*2,dtype=self.typeCode).reshape((2,2,2,2))
        answer_upper = [[[[0, 0], [0, 1]], [[0, 1], [1, 1]]], [[[0, 1], [1, 1]], [[1, 1], [1, 1]]]]
        answer_lower = [[[[1, 1], [1, 0]], [[1, 0], [0, 0]]], [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]
        lower, upper = luSplit(supertensor)
        self.assertEquals((lower == answer_lower).all(), True)
        self.assertEquals((upper == answer_upper).all(), True)

######################################################################

class scharTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "schar"
        self.typeCode = "b"
        #self.result   = int(self.result)

######################################################################

class ucharTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "uchar"
        self.typeCode = "B"
        #self.result   = int(self.result)

######################################################################

class shortTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "short"
        self.typeCode = "h"
        #self.result   = int(self.result)

######################################################################

class ushortTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "ushort"
        self.typeCode = "H"
        #self.result   = int(self.result)

######################################################################

class intTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "int"
        self.typeCode = "i"
        #self.result   = int(self.result)

######################################################################

class uintTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "uint"
        self.typeCode = "I"
        #self.result   = int(self.result)

######################################################################

class longTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "long"
        self.typeCode = "l"
        #self.result   = int(self.result)

######################################################################

class ulongTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "ulong"
        self.typeCode = "L"
        #self.result   = int(self.result)

######################################################################

class longLongTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "longLong"
        self.typeCode = "q"
        #self.result   = int(self.result)

######################################################################

class ulongLongTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "ulongLong"
        self.typeCode = "Q"
        #self.result   = int(self.result)

######################################################################

class floatTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr  = "float"
        self.typeCode = "f"

######################################################################

class doubleTestCase(SuperTensorTestCase):
    def __init__(self, methodName="runTest"):
        SuperTensorTestCase.__init__(self, methodName)
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
    print "Testing 4D Functions of Module SuperTensor"
    print "NumPy version", np.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
