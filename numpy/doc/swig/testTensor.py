#! /usr/bin/env python

# System imports
from   distutils.util import get_platform
from   math           import sqrt
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
import Tensor

######################################################################

class TensorTestCase(unittest.TestCase):

    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"
        self.result   = sqrt(28.0/8)

    def testNorm(self):
        "Test norm function"
        print >>sys.stderr, self.typeStr, "... ",
        norm = Tensor.__dict__[self.typeStr + "Norm"]
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        if isinstance(self.result, int):
            self.assertEquals(norm(tensor), self.result)
        else:
            self.assertAlmostEqual(norm(tensor), self.result, 6)

    def testMax(self):
        "Test max function"
        print >>sys.stderr, self.typeStr, "... ",
        max = Tensor.__dict__[self.typeStr + "Max"]
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(max(tensor), 8)

    def testMaxNonContainer(self):
        "Test max function with None"
        print >>sys.stderr, self.typeStr, "... ",
        max = Tensor.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, None)

    def testMaxWrongDim(self):
        "Test max function with a 1D array"
        print >>sys.stderr, self.typeStr, "... ",
        max = Tensor.__dict__[self.typeStr + "Max"]
        self.assertRaises(TypeError, max, [0, -1, 2, -3])

    def testMin(self):
        "Test min function"
        print >>sys.stderr, self.typeStr, "... ",
        min = Tensor.__dict__[self.typeStr + "Min"]
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(min(tensor), 2)

    def testScale(self):
        "Test scale function"
        print >>sys.stderr, self.typeStr, "... ",
        scale = Tensor.__dict__[self.typeStr + "Scale"]
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],self.typeCode)
        scale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testFloor(self):
        "Test floor function"
        print >>sys.stderr, self.typeStr, "... ",
        floor = Tensor.__dict__[self.typeStr + "Floor"]
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],self.typeCode)
        floor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testCeil(self):
        "Test ceil function"
        print >>sys.stderr, self.typeStr, "... ",
        ceil = Tensor.__dict__[self.typeStr + "Ceil"]
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],self.typeCode)
        ceil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testLUSplit(self):
        "Test luSplit function"
        print >>sys.stderr, self.typeStr, "... ",
        luSplit = Tensor.__dict__[self.typeStr + "LUSplit"]
        lower, upper = luSplit([[[1,1], [1,1]],
                                [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

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
    print "Testing 3D Functions of Module Tensor"
    print "NumPy version", N.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
