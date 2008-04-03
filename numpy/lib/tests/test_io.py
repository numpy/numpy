from numpy.testing import *
import numpy as np
import StringIO

class Testsavetxt(NumpyTestCase):
    def test_array(self):
        a =np.array( [[1,2],[3,4]], float)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        assert(c.readlines(), ['1.000000000000000000e+00 2.000000000000000000e+00\n', '3.000000000000000000e+00 4.000000000000000000e+00\n'])

        a =np.array( [[1,2],[3,4]], int)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        assert(c.readlines(), ['1 2\n', '3 4\n'])
        
    def test_1D(self):
        a = np.array([1,2,3,4], int)
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert(c.readlines(), ['1\n', '2\n', '3\n', '4\n'])
    
    def test_record(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', '<i4'), ('y', '<i4')])
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert(c.readlines(), ['1 2\n', '3 4\n'])

        

class Testloadtxt(NumpyTestCase):
    def test_record(self):
        c = StringIO.StringIO()
        c.write('1 2\n3 4')
        c.seek(0)
        x = np.loadtxt(c, dtype=[('x', np.int32), ('y', np.int32)])
        a = np.array([(1, 2), (3, 4)], dtype=[('x', '<i4'), ('y', '<i4')])
        assert_array_equal(x, a)
        
    def test_array(self):
        c = StringIO.StringIO()
        c.write('1 2\n3 4')
        
        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([[1,2],[3,4]], int)
        assert_array_equal(x, a)
        
        c.seek(0)
        x = np.loadtxt(c, dtype=float)
        a = np.array([[1,2],[3,4]], float)
        assert_array_equal(x, a)
        
    def test_1D(self):
        c = StringIO.StringIO()
        c.write('1\n2\n3\n4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([1,2,3,4], int)
        assert_array_equal(x, a)
    
if __name__ == "__main__":
    NumpyTest().run()
