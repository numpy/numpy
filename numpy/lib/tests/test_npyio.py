from numpy.testing import assert_equal
from numpy import genfromtxt
from io import BytesIO

class TestGenfromtxt:
    def test_genfromtxt_1number(self):
        myfile = BytesIO(b"1")
        data = genfromtxt(myfile, delimiter=",", ndmin=1)
        assert_equal(data.shape, (1,))

    def test_genfromtxt_1col(self):
        myfile = BytesIO(b"1\n2\n3\n4")
        data = genfromtxt(myfile, delimiter=",", ndmin=2)
        assert_equal(data.shape, (4, 1))

    def test_genfromtxt_1row(self):
        myfile = BytesIO(b"1,2,3,4")
        data = genfromtxt(myfile, delimiter=",", ndmin=2)
        assert_equal(data.shape, (1, 4))

    def test_genfromtxt_1colT(self):
        myfile = BytesIO(b"1\n2\n3\n4")
        data = genfromtxt(myfile, delimiter=",", unpack=True, ndmin=2)
        assert_equal(data.shape, (1, 4))

    def test_genfromtxt_1rowT(self):
        myfile = BytesIO(b"1,2,3,4")
        data = genfromtxt(myfile, delimiter=",", unpack=True, ndmin=2)
        assert_equal(data.shape, (4, 1))

    def test_genfromtxt_2d(self):
        myfile = BytesIO(b"1,2,3,4\n5,6,7,8")
        data = genfromtxt(myfile, delimiter=",")
        assert_equal(data.shape, (2, 4))

    def test_genfromtxt_2dT(self):
        myfile = BytesIO(b"1,2,3,4\n5,6,7,8")
        data = genfromtxt(myfile, delimiter=",", unpack=True)
        assert_equal(data.shape, (4, 2))
        
        
