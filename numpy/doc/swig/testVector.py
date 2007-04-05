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

    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"

    def testLength(self):
        "Test length function"
        print >>sys.stderr, self.typeStr, "... ",
        length = Vector.__dict__[self.typeStr + "Length"]
        self.assertEquals(length([5, 12, 0]), 13)

    def testLengthBad(self):
        "Test length function for wrong size"
        print >>sys.stderr, self.typeStr, "... ",
        length = Vector.__dict__[self.typeStr + "Length"]
        self.assertRaises(TypeError, length, [5, 12])

    def testProd(self):
        "Test prod function"
        print >>sys.stderr, self.typeStr, "... ",
        prod = Vector.__dict__[self.typeStr + "Prod"]
        self.assertEquals(prod([1,2,3,4]), 24)

    def testProdNonContainer(self):
        "Test prod function with None"
        print >>sys.stderr, self.typeStr, "... ",
        prod = Vector.__dict__[self.typeStr + "Prod"]
        self.assertRaises(TypeError, prod, None)

    def testSum(self):
        "Test sum function"
        print >>sys.stderr, self.typeStr, "... ",
        sum = Vector.__dict__[self.typeStr + "Sum"]
        self.assertEquals(sum([5,6,7,8]), 26)

    def testReverse(self):
        "Test reverse function"
        print >>sys.stderr, self.typeStr, "... ",
        reverse = Vector.__dict__[self.typeStr + "Reverse"]
        vector = N.array([1,2,4],self.typeCode)
        reverse(vector)
        self.assertEquals((vector == [4,2,1]).all(), True)

    def testOnes(self):
        "Test ones function"
        print >>sys.stderr, self.typeStr, "... ",
        ones = Vector.__dict__[self.typeStr + "Ones"]
        myArray = N.zeros(5,self.typeCode)
        ones(myArray)
        N.testing.assert_array_equal(myArray, N.array([1,1,1,1,1]))

    def testZeros(self):
        "Test zeros function"
        print >>sys.stderr, self.typeStr, "... ",
        zeros = Vector.__dict__[self.typeStr + "Zeros"]
        myArray = N.ones(5,self.typeCode)
        zeros(myArray)
        N.testing.assert_array_equal(myArray, N.array([0,0,0,0,0]))

    def testEOSplit(self):
        "Test eoSplit function"
        print >>sys.stderr, self.typeStr, "... ",
        eoSplit = Vector.__dict__[self.typeStr + "EOSplit"]
        even, odd = eoSplit([1,2,3])
        self.assertEquals((even == [1,0,3]).all(), True)
        self.assertEquals((odd  == [0,2,0]).all(), True)

    def testTwos(self):
        "Test twos function"
        print >>sys.stderr, self.typeStr, "... ",
        twos = Vector.__dict__[self.typeStr + "Twos"]
        vector = twos(5)
        self.assertEquals((vector == [2,2,2,2,2]).all(), True)

    def testThrees(self):
        "Test threes function"
        print >>sys.stderr, self.typeStr, "... ",
        threes = Vector.__dict__[self.typeStr + "Threes"]
        vector = threes(6)
        self.assertEquals((vector == [3,3,3,3,3,3]).all(), True)

######################################################################

class scharTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "schar"
        self.typeCode = "b"

######################################################################

class ucharTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "uchar"
        self.typeCode = "B"

######################################################################

class shortTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "short"
        self.typeCode = "h"

######################################################################

class ushortTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "ushort"
        self.typeCode = "H"

######################################################################

class intTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "int"
        self.typeCode = "i"

######################################################################

class uintTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "uint"
        self.typeCode = "I"

######################################################################

class longTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "long"
        self.typeCode = "l"

######################################################################

class ulongTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "ulong"
        self.typeCode = "L"

######################################################################

class longLongTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "longLong"
        self.typeCode = "q"

######################################################################

class ulongLongTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "ulongLong"
        self.typeCode = "Q"

######################################################################

class floatTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "float"
        self.typeCode = "f"

######################################################################

class doubleTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
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
    print "Testing 1D Functions of Module Vector"
    print "NumPy version", N.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
