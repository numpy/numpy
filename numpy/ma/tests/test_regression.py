from numpy.testing import *
import numpy as np

rlevel = 1

class TestRegression(TestCase):
    def test_masked_array_create(self,level=rlevel):
        """Ticket #17"""
        x = np.ma.masked_array([0,1,2,3,0,4,5,6],mask=[0,0,0,1,1,1,0,0])
        assert_array_equal(np.ma.nonzero(x),[[1,2,6,7]])

    def test_masked_array(self,level=rlevel):
        """Ticket #61"""
        x = np.ma.array(1,mask=[1])

    def test_mem_masked_where(self,level=rlevel):
        """Ticket #62"""
        from numpy.ma import masked_where, MaskType
        a = np.zeros((1,1))
        b = np.zeros(a.shape, MaskType)
        c = masked_where(b,a)
        a-c

    def test_masked_array_multiply(self,level=rlevel):
        """Ticket #254"""
        a = np.ma.zeros((4,1))
        a[2,0] = np.ma.masked
        b = np.zeros((4,2))
        a*b
        b*a

    def test_masked_array_repeat(self, level=rlevel):
        """Ticket #271"""
        np.ma.array([1],mask=False).repeat(10)

