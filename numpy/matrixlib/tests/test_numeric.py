from __future__ import division, absolute_import, print_function

from numpy.testing import assert_equal, TestCase, run_module_suite
from numpy.core import ones
from numpy import matrix, diagonal, diag

class TestDot(TestCase):
    def test_matscalar(self):
        b1 = matrix(ones((3, 3), dtype=complex))
        assert_equal(b1*1.0, b1)


def test_diagonal():
    b1 = matrix([[1,2],[3,4]])
    diag_b1 = matrix([[1, 4]])

    assert_equal(b1.diagonal(), diag_b1)
    assert_equal(diagonal(b1), diag_b1)
    assert_equal(diag(b1), diag_b1)


if __name__ == "__main__":
    run_module_suite()
