from numpy.core import *
from numpy.random import rand
from numpy.testing import *
from numpy.core.multiarray import dot as dot_


class test_dot(ScipyTestCase):
    def setUp(self):
        self.A = rand(10,10)
        self.b1 = rand(10,1)
        self.b2 = rand(10)
        self.b3 = rand(1,10)
        self.N = 14

    def check_matmat(self):
        A = self.A
        c1 = dot(A, A)
        c2 = dot_(A, A)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_matvec(self):
        A, b1 = self.A, self.b1
        c1 = dot(A, b1)
        c2 = dot_(A, b1)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_matvec2(self):
        A, b2 = self.A, self.b2 
        c1 = dot(A, b2)
        c2 = dot_(A, b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecmat(self):
        A, b2 = self.A, self.b2
        c1 = dot(b2, A)
        c2 = dot_(b2, A)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecmat2(self):
        b3, A = self.b3, self.A
        c1 = dot(b3, A)
        c2 = dot_(b3, A)
        assert_almost_equal(c1, c2, decimal=self.N)        

    def check_vecvecouter(self):
        b1, b3 = self.b1, self.b3
        c1 = dot(b1, b3)
        c2 = dot_(b1, b3)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecvecinner(self):
        b1, b3 = self.b1, self.b3
        c1 = dot(b3, b1)
        c2 = dot_(b3, b1)
        assert_almost_equal(c1, c2, decimal=self.N)

        
        
        
