# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_subclassing.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

import numpy as np
from numpy.testing import *
from numpy.ma.testutils import *
from numpy.ma.core import *

class SubArray(np.ndarray):
    """Defines a generic np.ndarray subclass, that stores some metadata
    in the  dictionary `info`."""
    def __new__(cls,arr,info={}):
        x = np.asanyarray(arr).view(cls)
        x.info = info
        return x
    def __array_finalize__(self, obj):
        self.info = getattr(obj,'info',{})
        return
    def __add__(self, other):
        result = np.ndarray.__add__(self, other)
        result.info.update({'added':result.info.pop('added',0)+1})
        return result

subarray = SubArray

class MSubArray(SubArray,MaskedArray):
    def __new__(cls, data, info={}, mask=nomask):
        subarr = SubArray(data, info)
        _data = MaskedArray.__new__(cls, data=subarr, mask=mask)
        _data.info = subarr.info
        return _data
    def __array_finalize__(self,obj):
        MaskedArray.__array_finalize__(self,obj)
        SubArray.__array_finalize__(self, obj)
        return
    def _get_series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view
    _series = property(fget=_get_series)

msubarray = MSubArray

class MMatrix(MaskedArray, np.matrix,):
    def __new__(cls, data, mask=nomask):
        mat = np.matrix(data)
        _data = MaskedArray.__new__(cls, data=mat, mask=mask)
        return _data
    def __array_finalize__(self,obj):
        np.matrix.__array_finalize__(self, obj)
        MaskedArray.__array_finalize__(self,obj)
        return
    def _get_series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view
    _series = property(fget=_get_series)

mmatrix = MMatrix

class TestSubclassing(TestCase):
    """Test suite for masked subclasses of ndarray."""

    def test_data_subclassing(self):
        "Tests whether the subclass is kept."
        x = np.arange(5)
        m = [0,0,1,0,0]
        xsub = SubArray(x)
        xmsub = masked_array(xsub, mask=m)
        self.failUnless(isinstance(xmsub, MaskedArray))
        assert_equal(xmsub._data, xsub)
        self.failUnless(isinstance(xmsub._data, SubArray))

    def test_maskedarray_subclassing(self):
        "Tests subclassing MaskedArray"
        x = np.arange(5)
        mx = mmatrix(x,mask=[0,1,0,0,0])
        self.failUnless(isinstance(mx._data, np.matrix))
        "Tests masked_unary_operation"
        self.failUnless(isinstance(add(mx,mx), mmatrix))
        self.failUnless(isinstance(add(mx,x), mmatrix))
        assert_equal(add(mx,x), mx+x)
        self.failUnless(isinstance(add(mx,mx)._data, np.matrix))
        self.failUnless(isinstance(add.outer(mx,mx), mmatrix))
        "Tests masked_binary_operation"
        self.failUnless(isinstance(hypot(mx,mx), mmatrix))
        self.failUnless(isinstance(hypot(mx,x), mmatrix))

    def test_attributepropagation(self):
        x = array(arange(5), mask=[0]+[1]*4)
        my = masked_array(subarray(x))
        ym = msubarray(x)
        #
        z = (my+1)
        self.failUnless(isinstance(z,MaskedArray))
        self.failUnless(not isinstance(z, MSubArray))
        self.failUnless(isinstance(z._data, SubArray))
        assert_equal(z._data.info, {})
        #
        z = (ym+1)
        self.failUnless(isinstance(z, MaskedArray))
        self.failUnless(isinstance(z, MSubArray))
        self.failUnless(isinstance(z._data, SubArray))
        self.failUnless(z._data.info['added'] > 0)
        #
        ym._set_mask([1,0,0,0,1])
        assert_equal(ym._mask, [1,0,0,0,1])
        ym._series._set_mask([0,0,0,0,1])
        assert_equal(ym._mask, [0,0,0,0,1])
        #
        xsub = subarray(x, info={'name':'x'})
        mxsub = masked_array(xsub)
        self.failUnless(hasattr(mxsub, 'info'))
        assert_equal(mxsub.info, xsub.info)

    def test_subclasspreservation(self):
        "Checks that masked_array(...,subok=True) preserves the class."
        x = np.arange(5)
        m = [0,0,1,0,0]
        xinfo = [(i,j) for (i,j) in zip(x,m)]
        xsub = MSubArray(x, mask=m, info={'xsub':xinfo})
        #
        mxsub = masked_array(xsub, subok=False)
        self.failUnless(not isinstance(mxsub, MSubArray))
        self.failUnless(isinstance(mxsub, MaskedArray))
        assert_equal(mxsub._mask, m)
        #
        mxsub = asarray(xsub)
        self.failUnless(not isinstance(mxsub, MSubArray))
        self.failUnless(isinstance(mxsub, MaskedArray))
        assert_equal(mxsub._mask, m)
        #
        mxsub = masked_array(xsub, subok=True)
        self.failUnless(isinstance(mxsub, MSubArray))
        assert_equal(mxsub.info, xsub.info)
        assert_equal(mxsub._mask, xsub._mask)
        #
        mxsub = asanyarray(xsub)
        self.failUnless(isinstance(mxsub, MSubArray))
        assert_equal(mxsub.info, xsub.info)
        assert_equal(mxsub._mask, m)


################################################################################
if __name__ == '__main__':
    run_module_suite()


