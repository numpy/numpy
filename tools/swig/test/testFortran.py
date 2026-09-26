#!/usr/bin/env python3
import sys
import unittest

import numpy as np

major, minor = [int(d) for d in np.__version__.split(".")[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError

import Fortran

######################################################################

class FortranTestCase(unittest.TestCase):

    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr = "double"
        self.typeCode = "d"

    # Test (type* IN_FARRAY2, int DIM1, int DIM2) typemap
    def testSecondElementFortran(self):
        "Test Fortran matrix initialized from reshaped NumPy fortranarray"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        second = Fortran.__dict__[self.typeStr + "SecondElement"]
        matrix = np.asfortranarray(np.arange(9).reshape(3, 3),
                                   self.typeCode)
        self.assertEqual(second(matrix), 3)

    def testSecondElementObject(self):
        "Test Fortran matrix initialized from nested list fortranarray"
        print(self.typeStr, "... ", end=' ', file=sys.stderr)
        second = Fortran.__dict__[self.typeStr + "SecondElement"]
        matrix = np.asfortranarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], self.typeCode)
        self.assertEqual(second(matrix), 3)

######################################################################

class scharTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "schar"
        self.typeCode = "b"

######################################################################

class ucharTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "uchar"
        self.typeCode = "B"

######################################################################

class shortTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "short"
        self.typeCode = "h"

######################################################################

class ushortTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "ushort"
        self.typeCode = "H"

######################################################################

class intTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "int"
        self.typeCode = "i"

######################################################################

class uintTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "uint"
        self.typeCode = "I"

######################################################################

class longTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "long"
        self.typeCode = "l"

######################################################################

class ulongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "ulong"
        self.typeCode = "L"

######################################################################

class longLongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "longLong"
        self.typeCode = "q"

######################################################################

class ulongLongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "ulongLong"
        self.typeCode = "Q"

######################################################################

class floatTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "float"
        self.typeCode = "f"

######################################################################

class doubleTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr = "double"
        self.typeCode = "d"

######################################################################


if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(    scharTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(    ucharTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(    shortTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(   ushortTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(      intTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(     uintTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(     longTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(    ulongTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase( longLongTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ulongLongTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(    floatTestCase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(   doubleTestCase))

    # Execute the test suite
    print("Testing 2D Functions of Module Matrix")
    print("NumPy version", np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))
