# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for maskedArray statistics.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_mstats.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

import numpy

import numpy.ma
from numpy.ma import masked, masked_array

import numpy.ma.testutils
from numpy.ma.testutils import *

from numpy.ma.mstats import *

#..............................................................................
class TestQuantiles(NumpyTestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        self.a = numpy.ma.arange(1,101)
    #
    def test_1d_nomask(self):
        "Test quantiles 1D - w/o mask."
        a = self.a
        assert_almost_equal(mquantiles(a, alphap=1., betap=1.),
                            [25.75, 50.5, 75.25])
        assert_almost_equal(mquantiles(a, alphap=0, betap=1.),
                            [25., 50., 75.])
        assert_almost_equal(mquantiles(a, alphap=0.5, betap=0.5),
                            [25.5, 50.5, 75.5])
        assert_almost_equal(mquantiles(a, alphap=0., betap=0.),
                            [25.25, 50.5, 75.75])
        assert_almost_equal(mquantiles(a, alphap=1./3, betap=1./3),
                            [25.41666667, 50.5, 75.5833333])
        assert_almost_equal(mquantiles(a, alphap=3./8, betap=3./8),
                            [25.4375, 50.5, 75.5625])
        assert_almost_equal(mquantiles(a), [25.45, 50.5, 75.55])#
    #
    def test_1d_mask(self):
        "Test quantiles 1D - w/ mask."
        a = self.a
        a[1::2] = masked
        assert_almost_equal(mquantiles(a, alphap=1., betap=1.),
                            [25.5, 50.0, 74.5])
        assert_almost_equal(mquantiles(a, alphap=0, betap=1.),
                            [24., 49., 74.])
        assert_almost_equal(mquantiles(a, alphap=0.5, betap=0.5),
                            [25., 50., 75.])
        assert_almost_equal(mquantiles(a, alphap=0., betap=0.),
                            [24.5, 50.0, 75.5])
        assert_almost_equal(mquantiles(a, alphap=1./3, betap=1./3),
                            [24.833333, 50.0, 75.166666])
        assert_almost_equal(mquantiles(a, alphap=3./8, betap=3./8),
                            [24.875, 50., 75.125])
        assert_almost_equal(mquantiles(a), [24.9, 50., 75.1])
    #
    def test_2d_nomask(self):
        "Test quantiles 2D - w/o mask."
        a = self.a
        b = numpy.ma.resize(a, (100,100))
        assert_almost_equal(mquantiles(b), [25.45, 50.5, 75.55])
        assert_almost_equal(mquantiles(b, axis=0), numpy.ma.resize(a,(3,100)))
        assert_almost_equal(mquantiles(b, axis=1),
                            numpy.ma.resize([25.45, 50.5, 75.55], (100,3)))
    #
    def test_2d_mask(self):
        "Test quantiles 2D - w/ mask."
        a = self.a
        a[1::2] = masked
        b = numpy.ma.resize(a, (100,100))
        assert_almost_equal(mquantiles(b), [25., 50., 75.])
        assert_almost_equal(mquantiles(b, axis=0), numpy.ma.resize(a,(3,100)))
        assert_almost_equal(mquantiles(b, axis=1),
                            numpy.ma.resize([24.9, 50., 75.1], (100,3)))

class TestMedian(NumpyTestCase):
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)

    def test_2d(self):
        "Tests median w/ 2D"
        (n,p) = (101,30)
        x = masked_array(numpy.linspace(-1.,1.,n),)
        x[:10] = x[-10:] = masked
        z = masked_array(numpy.empty((n,p), dtype=numpy.float_))
        z[:,0] = x[:]
        idx = numpy.arange(len(x))
        for i in range(1,p):
            numpy.random.shuffle(idx)
            z[:,i] = x[idx]
        assert_equal(mmedian(z[:,0]), 0)
        assert_equal(mmedian(z), numpy.zeros((p,)))

    def test_3d(self):
        "Tests median w/ 3D"
        x = numpy.ma.arange(24).reshape(3,4,2)
        x[x%3==0] = masked
        assert_equal(mmedian(x,0), [[12,9],[6,15],[12,9],[18,15]])
        x.shape = (4,3,2)
        assert_equal(mmedian(x,0),[[99,10],[11,99],[13,14]])
        x = numpy.ma.arange(24).reshape(4,3,2)
        x[x%5==0] = masked
        assert_equal(mmedian(x,0), [[12,10],[8,9],[16,17]])

#..............................................................................
class TestTrimming(NumpyTestCase):
    #
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
    #
    def test_trim(self):
        "Tests trimming."
        x = numpy.ma.arange(100)
        assert_equal(trim_both(x).count(), 60)
        assert_equal(trim_tail(x,tail='r').count(), 80)
        x[50:70] = masked
        trimx = trim_both(x)
        assert_equal(trimx.count(), 48)
        assert_equal(trimx._mask, [1]*16 + [0]*34 + [1]*20 + [0]*14 + [1]*16)
        x._mask = nomask
        x.shape = (10,10)
        assert_equal(trim_both(x).count(), 60)
        assert_equal(trim_tail(x).count(), 80)
    #
    def test_trimmedmean(self):
        "Tests the trimmed mean."
        data = masked_array([ 77, 87, 88,114,151,210,219,246,253,262,
                             296,299,306,376,428,515,666,1310,2611])
        assert_almost_equal(trimmed_mean(data,0.1), 343, 0)
        assert_almost_equal(trimmed_mean(data,0.2), 283, 0)
    #
    def test_trimmed_stde(self):
        "Tests the trimmed mean standard error."
        data = masked_array([ 77, 87, 88,114,151,210,219,246,253,262,
                             296,299,306,376,428,515,666,1310,2611])
        assert_almost_equal(trimmed_stde(data,0.2), 56.1, 1)
    #
    def test_winsorization(self):
        "Tests the Winsorization of the data."
        data = masked_array([ 77, 87, 88,114,151,210,219,246,253,262,
                             296,299,306,376,428,515,666,1310,2611])
        assert_almost_equal(winsorize(data).varu(), 21551.4, 1)
        data[5] = masked
        winsorized = winsorize(data)
        assert_equal(winsorized.mask, data.mask)
#..............................................................................

class TestMisc(NumpyTestCase):
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)

    def check_cov(self):
        "Tests the cov function."
        x = masked_array([[1,2,3],[4,5,6]], mask=[[1,0,0],[0,0,0]])
        c = cov(x[0])
        assert_equal(c, (x[0].anom()**2).sum())
        c = cov(x[1])
        assert_equal(c, (x[1].anom()**2).sum()/2.)
        c = cov(x)
        assert_equal(c[1,0], (x[0].anom()*x[1].anom()).sum())


###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()
