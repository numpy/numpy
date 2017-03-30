from numpy.testing import assert_, assert_raises
from numpy import genfromtxt
from io import BytesIO

class TestGenfromtxt:
    def test_genfromtxt_1number(self):
        data = b"1"
        myfile = genfromtxt(BytesIO(data), delimiter=",", ndmin=1)
        shape = myfile.shape
        correct_shape = (1,)
        assert_(shape == correct_shape)

    def test_genfromtxt_1col(self):
        data = b"1\n2\n3\n4"
        myfile = genfromtxt(BytesIO(data), delimiter=",", ndmin=2)
        shape = myfile.shape
        correct_shape = (4, 1)
        assert_(shape == correct_shape)

    def test_genfromtxt_1row(self):
        data = b"1,2,3,4"
        myfile = genfromtxt(BytesIO(data), delimiter=",", ndmin=2)
        shape = myfile.shape
        correct_shape = (1, 4)
        assert_(shape == correct_shape)

    def test_genfromtxt_1colT(self):
        data = b"1\n2\n3\n4"
        myfile = genfromtxt(BytesIO(data), delimiter=",", unpack=True, ndmin=2)
        shape = myfile.shape
        correct_shape = (1, 4)
        assert_(shape == correct_shape)

    def test_genfromtxt_1rowT(self):
        data = b"1,2,3,4"
        myfile = genfromtxt(BytesIO(data), delimiter=",", unpack=True, ndmin=2)
        shape = myfile.shape
        correct_shape = (4, 1)
        assert_(shape == correct_shape)

    def test_genfromtxt_2d(self):
        data = b"1,2,3,4\n5,6,7,8"
        myfile = genfromtxt(BytesIO(data), delimiter=",")
        shape = myfile.shape
        correct_shape = (2, 4)
        assert_(shape == correct_shape)

    def test_genfromtxt_2dT(self):
        data = b"1,2,3,4\n5,6,7,8"
        myfile = genfromtxt(BytesIO(data), delimiter=",", unpack=True)
        shape = myfile.shape
        correct_shape = (4, 2)
        assert_(shape == correct_shape)
