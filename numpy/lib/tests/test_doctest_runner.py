import doctest
import numpy as np
# the next line will register the NPY_FLEX_NUM option too
from numpy.lib.npy_doctest import FlexNumOutputChecker
import os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

class TestFlexNum_Doctest(object):
    def __init__(self):
        checker = FlexNumOutputChecker()
        self.parser = doctest.DocTestParser()
        self.runner = doctest.DocTestRunner(checker, False, 0)

    def _test(self, s):
        test = self.parser.get_doctest(s, {'np': np}, 'test', None, None)
        output = StringIO()
        res = self.runner.run(test, 0, output.write)
        output.seek(0)
        return output, res

    def _testdoc(self, s):
        output, res = self._test(s)
        if res.failed > 0:
            raise Exception("Flex Doctest Mismatch:\n" + output.read())

    def _testdoc_fails(self, s):
        output, res = self._test(s)
        if res.failed != 0:
            raise Exception("Flex Doctest Expected to fail but didn't:\n" + output.read())

    def test_flexnum_doctests(self):
        # test with some spacing and precision differences
        self._testdoc("""
            >>> np.arange(6.0) # doctest: +NPY_FLEX_NUMS
            array([0., 1.,  2.,     3.,  4.000001,  5.])
        """)

        # precision change is too big
        self._testdoc_fails("""
            >>> np.arange(6.0) # doctest: +NPY_FLEX_NUMS
            array([0., 1.,  2.,     3.,  4.001,  5.])
        """)

        # check ellipsis trailing float
        self._testdoc("""
            >>> np.arange(6.0) # doctest: +NPY_FLEX_NUMS
            array([ 0.,  1.000...,  2.,  3.,  4.000001,  5.])
        """)

        # check ellipsis outside a number is still caught by ELLIPSIS option
        self._testdoc("""
            >>> np.arange(6.0) # doctest: +NPY_FLEX_NUMS, +ELLIPSIS
            ar...y([ 0.,  1.00...,  2.,  3.,  4.000001,  5.])
        """)

        # check complex
        self._testdoc("""
            >>> np.array([1+1j, 1j, np.nan*1j]) # doctest: +NPY_FLEX_NUMS
            array([  1. +1.0j,   0.+1.j,  nan+nanj])
        """)
