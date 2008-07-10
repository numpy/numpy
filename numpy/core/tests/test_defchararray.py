from numpy.testing import *
from numpy.core import *
import numpy as np

class TestBasic(TestCase):
    def test_construction(self):
        A = np.array([['abc', '123'],
                      ['789', 'xyz']])
        A1 = A.view(np.chararray)
        A2 = np.chararray.__new__(np.chararray, A.shape, itemsize=A.itemsize,
                                  buffer=A)
        assert all(A1 == A2)


class TestWhitespace(TestCase):
    def setUp(self):
        self.A = np.array([['abc ', '123  '],
                           ['789 ', 'xyz ']]).view(np.chararray)
        self.B = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.chararray)

    def test1(self):
        assert all(self.A == self.B)


class TestOperations(TestCase):
    def setUp(self):
        self.A = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.chararray)
        self.B = np.array([['efg', '456'],
                           ['051', 'tuv']]).view(np.chararray)

    def test_add(self):
        AB = np.array([['abcefg', '123456'],
                       ['789051', 'xyztuv']]).view(np.chararray)
        assert all(AB == (self.A + self.B))

    def test_radd(self):
        QA = np.array([['qabc', 'q123'],
                       ['q789', 'qxyz']]).view(np.chararray)
        assert all(QA == ('q' + self.A))

    def test_mul(self):
        A = self.A
#        for r in (2,3,5,7,197):
#            Ar = np.array([[A[0,0]*r, A[0,1]*r],
#                           [A[1,0]*r, A[1,1]*r]]).view(np.chararray)
#
#            assert all(Ar == (self.A * r))
        
        for ob in [object(), 'qrs']:
            try:
                A * ob
            except ValueError:
                pass
            else:
                self.fail("chararray can only be multiplied by integers")

    def test_rmul(self):
        A = self.A
#        for r in (2,3,5,7,197):
#            Ar = np.array([[A[0,0]*r, A[0,1]*r],
#                           [A[1,0]*r, A[1,1]*r]]).view(np.chararray)
#
#            assert all(Ar == (r * self.A))
        
        for ob in [object(), 'qrs']:
            try:
                ob * A
            except ValueError:
                pass
            else:
                self.fail("chararray can only be multiplied by integers")

    def test_mod(self):
        pass

    def test_rmod(self):
        assert ("%s" % self.A) == str(self.A)
        assert ("%r" % self.A) == repr(self.A)
        
        for ob in [42, object()]:
            try:
                ob % self.A
            except TypeError:
                pass
            else:
                self.fail("chararray __rmod__ should fail with " \
                          "non-string objects")


if __name__ == "__main__":
    run_module_suite()
