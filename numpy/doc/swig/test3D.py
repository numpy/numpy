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
import Tensor

######################################################################

class TensorTestCase(unittest.TestCase):

    ####################################################
    ### Test functions that take arrays of type BYTE ###
    def testScharNorm(self):
        "Test scharNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.scharNorm(tensor), 1)

    def testScharMax(self):
        "Test scharMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.scharMax(tensor), 8)

    def testScharMaxNonContainer(self):
        "Test scharMax function with None"
        self.assertRaises(TypeError, Tensor.scharMax, None)

    def testScharMaxWrongDim(self):
        "Test scharMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.scharMax, [0, -1, 2, -3])

    def testScharMin(self):
        "Test scharMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.scharMin(tensor), 2)

    def testScharScale(self):
        "Test scharScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'b')
        Tensor.scharScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testScharFloor(self):
        "Test scharFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'b')
        Tensor.scharFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testScharCeil(self):
        "Test scharCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'b')
        Tensor.scharCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testScharLUSplit(self):
        "Test scharLUSplit function"
        lower, upper = Tensor.scharLUSplit([[[1,1], [1,1]],
                                            [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type UBYTE ###
    def testUcharNorm(self):
        "Test ucharNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.ucharNorm(tensor), 1)

    def testUcharMax(self):
        "Test ucharMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.ucharMax(tensor), 8)

    def testUcharMaxNonContainer(self):
        "Test ucharMax function with None"
        self.assertRaises(TypeError, Tensor.ucharMax, None)

    def testUcharMaxWrongDim(self):
        "Test ucharMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.ucharMax, [0, -1, 2, -3])

    def testUcharMin(self):
        "Test ucharMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.ucharMin(tensor), 2)

    def testUcharScale(self):
        "Test ucharScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'B')
        Tensor.ucharScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testUcharFloor(self):
        "Test ucharFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'B')
        Tensor.ucharFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testUcharCeil(self):
        "Test ucharCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'B')
        Tensor.ucharCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testUcharLUSplit(self):
        "Test ucharLUSplit function"
        lower, upper = Tensor.ucharLUSplit([[[1,1], [1,1]],
                                            [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type SHORT ###
    def testShortNorm(self):
        "Test shortNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.shortNorm(tensor), 1)

    def testShortMax(self):
        "Test shortMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.shortMax(tensor), 8)

    def testShortMaxNonContainer(self):
        "Test shortMax function with None"
        self.assertRaises(TypeError, Tensor.shortMax, None)

    def testShortMaxWrongDim(self):
        "Test shortMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.shortMax, [0, -1, 2, -3])

    def testShortMin(self):
        "Test shortMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.shortMin(tensor), 2)

    def testShortScale(self):
        "Test shortScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'h')
        Tensor.shortScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testShortFloor(self):
        "Test shortFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'h')
        Tensor.shortFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testShortCeil(self):
        "Test shortCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'h')
        Tensor.shortCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testShortLUSplit(self):
        "Test shortLUSplit function"
        lower, upper = Tensor.shortLUSplit([[[1,1], [1,1]],
                                            [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    ######################################################
    ### Test functions that take arrays of type USHORT ###
    def testUshortNorm(self):
        "Test ushortNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.ushortNorm(tensor), 1)

    def testUshortMax(self):
        "Test ushortMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.ushortMax(tensor), 8)

    def testUshortMaxNonContainer(self):
        "Test ushortMax function with None"
        self.assertRaises(TypeError, Tensor.ushortMax, None)

    def testUshortMaxWrongDim(self):
        "Test ushortMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.ushortMax, [0, -1, 2, -3])

    def testUshortMin(self):
        "Test ushortMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.ushortMin(tensor), 2)

    def testUshortScale(self):
        "Test ushortScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'H')
        Tensor.ushortScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testUshortFloor(self):
        "Test ushortFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'H')
        Tensor.ushortFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testUshortCeil(self):
        "Test ushortCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'H')
        Tensor.ushortCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testUshortLUSplit(self):
        "Test ushortLUSplit function"
        lower, upper = Tensor.ushortLUSplit([[[1,1], [1,1]],
                                             [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    ###################################################
    ### Test functions that take arrays of type INT ###
    def testIntNorm(self):
        "Test intNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.intNorm(tensor), 1)

    def testIntMax(self):
        "Test intMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.intMax(tensor), 8)

    def testIntMaxNonContainer(self):
        "Test intMax function with None"
        self.assertRaises(TypeError, Tensor.intMax, None)

    def testIntMaxWrongDim(self):
        "Test intMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.intMax, [0, -1, 2, -3])

    def testIntMin(self):
        "Test intMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.intMin(tensor), 2)

    def testIntScale(self):
        "Test intScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'i')
        Tensor.intScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testIntFloor(self):
        "Test intFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'i')
        Tensor.intFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testIntCeil(self):
        "Test intCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'i')
        Tensor.intCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testIntLUSplit(self):
        "Test intLUSplit function"
        lower, upper = Tensor.intLUSplit([[[1,1], [1,1]],
                                          [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    ####################################################
    ### Test functions that take arrays of type UINT ###
    def testUintNorm(self):
        "Test uintNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.uintNorm(tensor), 1)

    def testUintMax(self):
        "Test uintMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.uintMax(tensor), 8)

    def testUintMaxNonContainer(self):
        "Test uintMax function with None"
        self.assertRaises(TypeError, Tensor.uintMax, None)

    def testUintMaxWrongDim(self):
        "Test uintMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.uintMax, [0, -1, 2, -3])

    def testUintMin(self):
        "Test uintMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.uintMin(tensor), 2)

    def testUintScale(self):
        "Test uintScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'I')
        Tensor.uintScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testUintFloor(self):
        "Test uintFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'I')
        Tensor.uintFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testUintCeil(self):
        "Test uintCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'I')
        Tensor.uintCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testUintLUSplit(self):
        "Test uintLUSplit function"
        lower, upper = Tensor.uintLUSplit([[[1,1], [1,1]],
                                           [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    ####################################################
    ### Test functions that take arrays of type LONG ###
    def testLongNorm(self):
        "Test longNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.longNorm(tensor), 1)

    def testLongMax(self):
        "Test longMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.longMax(tensor), 8)

    def testLongMaxNonContainer(self):
        "Test longMax function with None"
        self.assertRaises(TypeError, Tensor.longMax, None)

    def testLongMaxWrongDim(self):
        "Test longMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.longMax, [0, -1, 2, -3])

    def testLongMin(self):
        "Test longMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.longMin(tensor), 2)

    def testLongScale(self):
        "Test longScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'l')
        Tensor.longScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testLongFloor(self):
        "Test longFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'l')
        Tensor.longFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testLongCeil(self):
        "Test longCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'l')
        Tensor.longCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testLongLUSplit(self):
        "Test longLUSplit function"
        lower, upper = Tensor.longLUSplit([[[1,1], [1,1]],
                                           [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type ULONG ###
    def testUlongNorm(self):
        "Test ulongNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.ulongNorm(tensor), 1)

    def testUlongMax(self):
        "Test ulongMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.ulongMax(tensor), 8)

    def testUlongMaxNonContainer(self):
        "Test ulongMax function with None"
        self.assertRaises(TypeError, Tensor.ulongMax, None)

    def testUlongMaxWrongDim(self):
        "Test ulongMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.ulongMax, [0, -1, 2, -3])

    def testUlongMin(self):
        "Test ulongMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.ulongMin(tensor), 2)

    def testUlongScale(self):
        "Test ulongScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'L')
        Tensor.ulongScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testUlongFloor(self):
        "Test ulongFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'L')
        Tensor.ulongFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testUlongCeil(self):
        "Test ulongCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'L')
        Tensor.ulongCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testUlongLUSplit(self):
        "Test ulongLUSplit function"
        lower, upper = Tensor.ulongLUSplit([[[1,1], [1,1]],
                                            [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    ########################################################
    ### Test functions that take arrays of type LONGLONG ###
    def testLongLongNorm(self):
        "Test longLongNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.longLongNorm(tensor), 1)

    def testLongLongMax(self):
        "Test longLongMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.longLongMax(tensor), 8)

    def testLongLongMaxNonContainer(self):
        "Test longLongMax function with None"
        self.assertRaises(TypeError, Tensor.longLongMax, None)

    def testLongLongMaxWrongDim(self):
        "Test longLongMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.longLongMax, [0, -1, 2, -3])

    def testLongLongMin(self):
        "Test longLongMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.longLongMin(tensor), 2)

    def testLongLongScale(self):
        "Test longLongScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'q')
        Tensor.longLongScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testLongLongFloor(self):
        "Test longLongFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'q')
        Tensor.longLongFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testLongLongCeil(self):
        "Test longLongCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'q')
        Tensor.longLongCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testLongLongLUSplit(self):
        "Test longLongLUSplit function"
        lower, upper = Tensor.longLongLUSplit([[[1,1], [1,1]],
                                               [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    #########################################################
    ### Test functions that take arrays of type ULONGLONG ###
    def testUlongLongNorm(self):
        "Test ulongLongNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertEquals(Tensor.ulongLongNorm(tensor), 1)

    def testUlongLongMax(self):
        "Test ulongLongMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.ulongLongMax(tensor), 8)

    def testUlongLongMaxNonContainer(self):
        "Test ulongLongMax function with None"
        self.assertRaises(TypeError, Tensor.ulongLongMax, None)

    def testUlongLongMaxWrongDim(self):
        "Test ulongLongMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.ulongLongMax, [0, -1, 2, -3])

    def testUlongLongMin(self):
        "Test ulongLongMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.ulongLongMin(tensor), 2)

    def testUlongLongScale(self):
        "Test ulongLongScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'Q')
        Tensor.ulongLongScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testUlongLongFloor(self):
        "Test ulongLongFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'Q')
        Tensor.ulongLongFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testUlongLongCeil(self):
        "Test ulongLongCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'Q')
        Tensor.ulongLongCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testUlongLongLUSplit(self):
        "Test ulongLongLUSplit function"
        lower, upper = Tensor.ulongLongLUSplit([[[1,1], [1,1]],
                                                [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    #####################################################
    ### Test functions that take arrays of type FLOAT ###
    def testFloatNorm(self):
        "Test floatNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertAlmostEquals(Tensor.floatNorm(tensor), 1.870828, 5)
        # 1.8708286933869707

    def testFloatMax(self):
        "Test floatMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.floatMax(tensor), 8)

    def testFloatMaxNonContainer(self):
        "Test floatMax function with None"
        self.assertRaises(TypeError, Tensor.floatMax, None)

    def testFloatMaxWrongDim(self):
        "Test floatMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.floatMax, [0, -1, 2, -3])

    def testFloatMin(self):
        "Test floatMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.floatMin(tensor), 2)

    def testFloatScale(self):
        "Test floatScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'f')
        Tensor.floatScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testFloatFloor(self):
        "Test floatFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'f')
        Tensor.floatFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testFloatCeil(self):
        "Test floatCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'f')
        Tensor.floatCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testFloatLUSplit(self):
        "Test floatLUSplit function"
        lower, upper = Tensor.floatLUSplit([[[1,1], [1,1]],
                                            [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

    ######################################################
    ### Test functions that take arrays of type DOUBLE ###
    def testDoubleNorm(self):
        "Test doubleNorm function"
        tensor = [[[0,1], [2,3]],
                  [[3,2], [1,0]]]
        self.assertAlmostEquals(Tensor.doubleNorm(tensor), 1.8708286933869)

    def testDoubleMax(self):
        "Test doubleMax function"
        tensor = [[[1,2], [3,4]],
                  [[5,6], [7,8]]]
        self.assertEquals(Tensor.doubleMax(tensor), 8)

    def testDoubleMaxNonContainer(self):
        "Test doubleMax function with None"
        self.assertRaises(TypeError, Tensor.doubleMax, None)

    def testDoubleMaxWrongDim(self):
        "Test doubleMax function with a 1D array"
        self.assertRaises(TypeError, Tensor.doubleMax, [0, -1, 2, -3])

    def testDoubleMin(self):
        "Test doubleMin function"
        tensor = [[[9,8], [7,6]],
                  [[5,4], [3,2]]]
        self.assertEquals(Tensor.doubleMin(tensor), 2)

    def testDoubleScale(self):
        "Test doubleScale function"
        tensor = N.array([[[1,0,1], [0,1,0], [1,0,1]],
                          [[0,1,0], [1,0,1], [0,1,0]],
                          [[1,0,1], [0,1,0], [1,0,1]]],'d')
        Tensor.doubleScale(tensor,4)
        self.assertEquals((tensor == [[[4,0,4], [0,4,0], [4,0,4]],
                                      [[0,4,0], [4,0,4], [0,4,0]],
                                      [[4,0,4], [0,4,0], [4,0,4]]]).all(), True)

    def testDoubleFloor(self):
        "Test doubleFloor function"
        tensor = N.array([[[1,2], [3,4]],
                          [[5,6], [7,8]]],'d')
        Tensor.doubleFloor(tensor,4)
        N.testing.assert_array_equal(tensor, N.array([[[4,4], [4,4]],
                                                      [[5,6], [7,8]]]))

    def testDoubleCeil(self):
        "Test doubleCeil function"
        tensor = N.array([[[9,8], [7,6]],
                          [[5,4], [3,2]]],'d')
        Tensor.doubleCeil(tensor,5)
        N.testing.assert_array_equal(tensor, N.array([[[5,5], [5,5]],
                                                      [[5,4], [3,2]]]))

    def testDoubleLUSplit(self):
        "Test doubleLUSplit function"
        lower, upper = Tensor.doubleLUSplit([[[1,1], [1,1]],
                                             [[1,1], [1,1]]])
        self.assertEquals((lower == [[[1,1], [1,0]],
                                     [[1,0], [0,0]]]).all(), True)
        self.assertEquals((upper == [[[0,0], [0,1]],
                                     [[0,1], [1,1]]]).all(), True)

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TensorTestCase))

    # Execute the test suite
    print "Testing 3D Functions of Module Tensor"
    print "NumPy version", N.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
