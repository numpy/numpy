import sys
import pytest

import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
import numpy.lib.utils as utils

from io import StringIO


@pytest.mark.filterwarnings("ignore:.*safe_eval.*:DeprecationWarning")
def test_safe_eval_nameconstant():
    # Test if safe_eval supports Python 3.4 _ast.NameConstant
    utils.safe_eval('None')


class TestByteBounds:

    def test_byte_bounds(self):
        # pointer difference matches size * itemsize
        # due to contiguity
        a = arange(12).reshape(3, 4)
        low, high = utils.byte_bounds(a)
        assert_equal(high - low, a.size * a.itemsize)

    def test_unusual_order_positive_stride(self):
        a = arange(12).reshape(3, 4)
        b = a.T
        low, high = utils.byte_bounds(b)
        assert_equal(high - low, b.size * b.itemsize)

    def test_unusual_order_negative_stride(self):
        a = arange(12).reshape(3, 4)
        b = a.T[::-1]
        low, high = utils.byte_bounds(b)
        assert_equal(high - low, b.size * b.itemsize)

    def test_strided(self):
        a = arange(12)
        b = a[::2]
        low, high = utils.byte_bounds(b)
        # the largest pointer address is lost (even numbers only in the
        # stride), and compensate addresses for striding by 2
        assert_equal(high - low, b.size * 2 * b.itemsize - b.itemsize)


def test_assert_raises_regex_context_manager():
    with assert_raises_regex(ValueError, 'no deprecation warning'):
        raise ValueError('no deprecation warning')


def test_info_method_heading():
    # info(class) should only print "Methods:" heading if methods exist

    class NoPublicMethods:
        pass

    class WithPublicMethods:
        def first_method():
            pass

    def _has_method_heading(cls):
        out = StringIO()
        utils.info(cls, output=out)
        return 'Methods:' in out.getvalue()

    assert _has_method_heading(WithPublicMethods)
    assert not _has_method_heading(NoPublicMethods)


def test_drop_metadata():
    def _compare_dtypes(dt1, dt2):
        return np.can_cast(dt1, dt2, casting='no')

    # structured dtype
    dt = np.dtype([('l1', [('l2', np.dtype('S8', metadata={'msg': 'toto'}))])],
                  metadata={'msg': 'titi'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None
    assert dt_m['l1'].metadata is None
    assert dt_m['l1']['l2'].metadata is None
    
    # alignment
    dt = np.dtype([('x', '<f8'), ('y', '<i4')],
                  align=True,
                  metadata={'msg': 'toto'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None

    # subdtype
    dt = np.dtype('8f',
                  metadata={'msg': 'toto'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None

    # scalar
    dt = np.dtype('uint32',
                  metadata={'msg': 'toto'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None


@pytest.mark.parametrize("dtype",
        [np.dtype("i,i,i,i")[["f1", "f3"]],
        np.dtype("f8"),
        np.dtype("10i")])
def test_drop_metadata_identity_and_copy(dtype):
    # If there is no metadata, the identity is preserved:
    assert utils.drop_metadata(dtype) is dtype

    # If there is any, it is dropped (subforms are checked above)
    dtype = np.dtype(dtype, metadata={1: 2})
    assert utils.drop_metadata(dtype).metadata is None
