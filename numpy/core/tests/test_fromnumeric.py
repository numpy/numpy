from __future__ import division, absolute_import, print_function

from numpy import put
from numpy.testing import TestCase, assert_raises


class TestPut(TestCase):

    def test_bad_array(self):
        # We want to raise a TypeError in the
        # case that a non-ndarray object is passed
        # in since `np.put` modifies in place and
        # hence would do nothing to a non-ndarray
        v = 5
        indx = [0, 2]
        bad_array = [1, 2, 3]
        assert_raises(TypeError, put, bad_array, indx, v)
