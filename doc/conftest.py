"""
Pytest configuration and fixtures for the Numpy test suite.
"""
import pytest
import numpy
import matplotlib
import doctest

matplotlib.use('agg', force=True)

# Ignore matplotlib output such as `<matplotlib.image.AxesImage at
# 0x7f956908c280>`. doctest monkeypatching inspired by
# https://github.com/wooyek/pytest-doctest-ellipsis-markers (MIT license)
OutputChecker = doctest.OutputChecker

class SkipMatplotlibOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        if '<matplotlib.' in got:
            got = ''
        return OutputChecker.check_output(self, want, got, optionflags)

doctest.OutputChecker = SkipMatplotlibOutputChecker

@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    numpy.random.seed(1)
    doctest_namespace['np'] = numpy

