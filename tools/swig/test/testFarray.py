#! /usr/bin/env python
from __future__ import division, absolute_import, print_function

# System imports
from   distutils.util import get_platform
import os
import sys
import unittest

# Import NumPy
import numpy as np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

# Add the distutils-generated build directory to the python search path and then
# import the extension module
libDir = "lib.{}-{}.{}".format(get_platform(), *sys.version_info[:2])
sys.path.insert(0, os.path.join("build", libDir))
import Farray

######################################################################

class FarrayTestCase(unittest.TestCase):

    def setUp(self):
        self.nrows = 5
        self.ncols = 4
        self.array = Farray.Farray(self.nrows, self.ncols)

    def testConstructor1(self):
        "Test Farray size constructor"
        self.failUnless(isinstance(self.array, Farray.Farray))

    def testConstructor2(self):
        "Test Farray copy constructor"
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i + j
        arrayCopy = Farray.Farray(self.array)
        self.failUnless(arrayCopy == self.array)

    def testConstructorBad1(self):
        "Test Farray size constructor, negative nrows"
        self.assertRaises(ValueError, Farray.Farray, -4, 4)

    def testConstructorBad2(self):
        "Test Farray size constructor, negative ncols"
        self.assertRaises(ValueError, Farray.Farray, 4, -4)

    def testNrows(self):
        "Test Farray nrows method"
        self.failUnless(self.array.nrows() == self.nrows)

    def testNcols(self):
        "Test Farray ncols method"
        self.failUnless(self.array.ncols() == self.ncols)

    def testLen(self):
        "Test Farray __len__ method"
        self.failUnless(len(self.array) == self.nrows*self.ncols)

    def testSetGet(self):
        "Test Farray __setitem__, __getitem__ methods"
        m = self.nrows
        n = self.ncols
        for i in range(m):
            for j in range(n):
                self.array[i, j] = i*j
        for i in range(m):
            for j in range(n):
                self.failUnless(self.array[i, j] == i*j)

    def testSetBad1(self):
        "Test Farray __setitem__ method, negative row"
        self.assertRaises(IndexError, self.array.__setitem__, (-1, 3), 0)

    def testSetBad2(self):
        "Test Farray __setitem__ method, negative col"
        self.assertRaises(IndexError, self.array.__setitem__, (1, -3), 0)

    def testSetBad3(self):
        "Test Farray __setitem__ method, out-of-range row"
        self.assertRaises(IndexError, self.array.__setitem__, (self.nrows+1, 0), 0)

    def testSetBad4(self):
        "Test Farray __setitem__ method, out-of-range col"
        self.assertRaises(IndexError, self.array.__setitem__, (0, self.ncols+1), 0)

    def testGetBad1(self):
        "Test Farray __getitem__ method, negative row"
        self.assertRaises(IndexError, self.array.__getitem__, (-1, 3))

    def testGetBad2(self):
        "Test Farray __getitem__ method, negative col"
        self.assertRaises(IndexError, self.array.__getitem__, (1, -3))

    def testGetBad3(self):
        "Test Farray __getitem__ method, out-of-range row"
        self.assertRaises(IndexError, self.array.__getitem__, (self.nrows+1, 0))

    def testGetBad4(self):
        "Test Farray __getitem__ method, out-of-range col"
        self.assertRaises(IndexError, self.array.__getitem__, (0, self.ncols+1))

    def testAsString(self):
        "Test Farray asString method"
        result = """\
[ [ 0, 1, 2, 3 ],
  [ 1, 2, 3, 4 ],
  [ 2, 3, 4, 5 ],
  [ 3, 4, 5, 6 ],
  [ 4, 5, 6, 7 ] ]
"""
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i+j
        self.failUnless(self.array.asString() == result)

    def testStr(self):
        "Test Farray __str__ method"
        result = """\
[ [ 0, -1, -2, -3 ],
  [ 1, 0, -1, -2 ],
  [ 2, 1, 0, -1 ],
  [ 3, 2, 1, 0 ],
  [ 4, 3, 2, 1 ] ]
"""
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i-j
        self.failUnless(str(self.array) == result)

    def testView(self):
        "Test Farray view method"
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i+j
        a = self.array.view()
        self.failUnless(isinstance(a, np.ndarray))
        self.failUnless(a.flags.f_contiguous)
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.failUnless(a[i, j] == i+j)

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FarrayTestCase))

    # Execute the test suite
    print("Testing Classes of Module Farray")
    print("NumPy version", np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))
