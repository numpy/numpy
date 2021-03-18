import os
import pytest

from numpy.testing import assert_array_equal, assert_raises
import numpy as np
from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

@pytest.fixture
def string():
    return np.array(b'1234')

@pytest.fixture
def toolarge():
    return np.array(b'12345')

@pytest.fixture
def toosmall():
    return np.array(b'123')

class TestString(util.F2PyTest):
    sources = [_path('src', 'string', 'char.f90')]

    @pytest.mark.slow
    def test_char(self):
        strings = np.array(['ab', 'cd', 'ef'], dtype='c').T
        inp, out = self.module.char_test.change_strings(strings,
                                                        strings.shape[1])
        assert_array_equal(inp, strings)
        expected = strings.copy()
        expected[1, :] = 'AAA'
        assert_array_equal(out, expected)

    @pytest.mark.xfail
    # Returns on less character than expected - see gh-18431
    def test_char_intent_inout(self, string):
        self.module.char_test.string_size_inout(string)
        assert_array_equal(string, np.array(b'A234'))

    def test_char_intent_inout_toosmall(self, toosmall):
        assert_raises(ValueError, self.module.char_test.string_size_inout,
                      toosmall)

    def test_char_intent_inout_toolarge(self, toolarge):
        self.module.char_test.string_size_inout(toolarge)
        assert_array_equal(toolarge, np.array(b'A234'))

    def test_char_intent_cache(self, string):
        self.module.char_test.string_size_cache(string)
        assert_array_equal(string, np.array(b'1234'))

    def test_char_intent_cache_toosmall(self, toosmall):
        assert_raises(ValueError, self.module.char_test.string_size_cache,
                      toosmall)

    def test_char_intent_cache_toolarge(self, toolarge):
        self.module.char_test.string_size_cache(toolarge)
        assert_array_equal(toolarge, np.array(b'12345'))

    @pytest.mark.xfail
    # string is unchanged even with intent(in, overwrite)
    # this may be expected?
    def test_char_intent_overwrite(self, string):
        self.module.char_test.string_size_overwrite(string)
        assert_array_equal(string, np.array(b'A234'))

    def test_char_intent_overwrite_toosmall(self, toosmall):
        assert_raises(ValueError, self.module.char_test.string_size_overwrite,
                      toosmall)

    @pytest.mark.xfail
    # string is unchanged even with intent(in, overwrite)
    # this may be expected?
    def test_char_intent_overwrite_toolarge(self, toolarge):
        self.module.char_test.string_size_overwrite(toolarge)
        assert_array_equal(toolarge, np.array(b'A234'))

    def test_char_intent_copy(self, string):
        result = self.module.char_test.string_size_copy(string)
        assert_array_equal(string, np.array(b'1234'))
        assert_array_equal(result, np.array(b'A234'))

    @pytest.mark.xfail
    # Raises value error but should create a temporary variable with appropriate
    # size
    def test_char_intent_copy_toosmall(self, toosmall):
        result = self.module.char_test.string_size_copy(toosmall)
        assert_array_equal(toosmall, np.array(b'123'))
        assert_array_equal(result, np.array(b'A23'))

    def test_char_intent_copy_toolarge(self, toolarge):
        result = self.module.char_test.string_size_copy(toolarge)
        assert_array_equal(toolarge, np.array(b'12345'))
        assert_array_equal(result, np.array(b'A234'))

    @pytest.mark.xfail
    # string is unchanged even with intent(inplace)
    # this may be expected?
    def test_char_intent_inplace(self, string):
        self.module.char_test.string_size_inplace(string)
        assert_array_equal(string, np.array(b'A234'))

    @pytest.mark.xfail
    # Raises value error but should create a temporary variable with appropriate
    # size
    def test_char_intent_inplace_toosmall(self, toosmall):
        self.module.char_test.string_size_inplace(toosmall)
        assert_array_equal(toosmall, np.array(b'A23'))

    @pytest.mark.xfail
    # string is unchanged even with intent(inplace)
    # this may be expected?
    def test_char_intent_inplace_toolarge(self, toolarge):
        self.module.char_test.string_size_inplace(toolarge)
        assert_array_equal(toolarge, np.array(b'A234'))

    def test_char_intent_out(self, string):
        result = self.module.char_test.string_size_out(string)
        assert_array_equal(string, np.array(b'1234'))
        assert_array_equal(result, np.array(b'A234'))

    @pytest.mark.xfail
    # Raises value error but should create a temporary variable with appropriate
    # size
    def test_char_intent_out_toosmall(self, toosmall):
        result = self.module.char_test.string_size_out(toosmall)
        assert_array_equal(toosmall, np.array(b'123'))
        assert_array_equal(result, np.array(b'A23'))

    def test_char_intent_out_toolarge(self, toolarge):
        result = self.module.char_test.string_size_out(toolarge)
        assert_array_equal(toolarge, np.array(b'12345'))
        assert_array_equal(result, np.array(b'A234'))
