#!/usr/bin/env python3
import struct
import sys
import unittest

import numpy as np

major, minor = [int(d) for d in np.__version__.split(".")[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError

import Flat

######################################################################

class FlatTestCase(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr = "double"
        self.typeCode = "d"

    # Test the (type* INPLACE_ARRAY_FLAT, int DIM_FLAT) typemap
    def testProcess1D(self):
        "Test Process function 1D array"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        process = Flat.__dict__[self.typeStr + "Process"]
        pack_output = b''
        for i in range(10):
            pack_output += struct.pack(self.typeCode, i)
        x = np.frombuffer(pack_output, dtype=self.typeCode)
        y = x.copy()
        process(y)
        self.assertEqual(np.all((x + 1) == y), True)

    def testProcess3D(self):
        "Test Process function 3D array"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        process = Flat.__dict__[self.typeStr + "Process"]
        pack_output = b''
        for i in range(24):
            pack_output += struct.pack(self.typeCode, i)
        x = np.frombuffer(pack_output, dtype=self.typeCode)
        x.shape = (2, 3, 4)
        y = x.copy()
        process(y)
        self.assertEqual(np.all((x + 1) == y), True)

    def testProcess3DTranspose(self):
        "Test Process function 3D array, FORTRAN order"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        process = Flat.__dict__[self.typeStr + "Process"]
        pack_output = b''
        for i in range(24):
            pack_output += struct.pack(self.typeCode, i)
        x = np.frombuffer(pack_output, dtype=self.typeCode)
        x.shape = (2, 3, 4)
        y = x.copy()
        process(y.T)
        self.assertEqual(np.all((x.T + 1) == y.T), True)

    def testProcessNoncontiguous(self):
        "Test Process function with non-contiguous array, which should raise an error"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        process = Flat.__dict__[self.typeStr + "Process"]
        pack_output = b''
        for i in range(24):
            pack_output += struct.pack(self.typeCode, i)
        x = np.frombuffer(pack_output, dtype=self.typeCode)
        x.shape = (2, 3, 4)
        self.assertRaises(TypeError, process, x[:, :, 0])


######################################################################

class scharTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "schar"
        self.typeCode = "b"

######################################################################

class ucharTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "uchar"
        self.typeCode = "B"

######################################################################

class shortTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "short"
        self.typeCode = "h"

######################################################################

class ushortTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "ushort"
        self.typeCode = "H"

######################################################################

class intTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "int"
        self.typeCode = "i"

######################################################################

class uintTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "uint"
        self.typeCode = "I"

######################################################################

class longTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "long"
        self.typeCode = "l"

######################################################################

class ulongTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "ulong"
        self.typeCode = "L"

######################################################################

class longLongTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "longLong"
        self.typeCode = "q"

######################################################################

class ulongLongTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "ulongLong"
        self.typeCode = "Q"

######################################################################

class floatTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "float"
        self.typeCode = "f"

######################################################################

class doubleTestCase(FlatTestCase):
    def __init__(self, methodName="runTest"):
        FlatTestCase.__init__(self, methodName)
        self.typeStr = "double"
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
    print("Testing 1D Functions of Module Flat")
    print("NumPy version", np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))
