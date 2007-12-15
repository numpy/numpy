# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for maskedArray statistics.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_morestats.py 317 2007-10-04 19:31:14Z backtopop $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: backtopop $)"
__version__ = '1.0'
__revision__ = "$Revision: 317 $"
__date__     = '$Date: 2007-10-04 15:31:14 -0400 (Thu, 04 Oct 2007) $'

import numpy

import maskedarray
from maskedarray import masked, masked_array

import maskedarray.mstats
from maskedarray.mstats import *
import maskedarray.morestats
from maskedarray.morestats import *

import maskedarray.testutils
from maskedarray.testutils import *


class TestMisc(NumpyTestCase):
    #
    def __init__(self, *args, **kwargs):
        NumpyTestCase.__init__(self, *args, **kwargs)
    #
    def test_mjci(self):
        "Tests the Marits-Jarrett estimator"
        data = masked_array([ 77, 87, 88,114,151,210,219,246,253,262,
                             296,299,306,376,428,515,666,1310,2611])
        assert_almost_equal(mjci(data),[55.76819,45.84028,198.8788],5)
    #
    def test_trimmedmeanci(self):
        "Tests the confidence intervals of the trimmed mean."
        data = masked_array([545,555,558,572,575,576,578,580,
                             594,605,635,651,653,661,666])
        assert_almost_equal(trimmed_mean(data,0.2), 596.2, 1)
        assert_equal(numpy.round(trimmed_mean_ci(data,0.2),1), [561.8, 630.6])

#..............................................................................
class TestRanking(NumpyTestCase):
    #
    def __init__(self, *args, **kwargs):
        NumpyTestCase.__init__(self, *args, **kwargs)
    #
    def test_ranking(self):
        x = masked_array([0,1,1,1,2,3,4,5,5,6,])
        assert_almost_equal(rank_data(x),[1,3,3,3,5,6,7,8.5,8.5,10])
        x[[3,4]] = masked
        assert_almost_equal(rank_data(x),[1,2.5,2.5,0,0,4,5,6.5,6.5,8])
        assert_almost_equal(rank_data(x,use_missing=True),
                            [1,2.5,2.5,4.5,4.5,4,5,6.5,6.5,8])
        x = masked_array([0,1,5,1,2,4,3,5,1,6,])
        assert_almost_equal(rank_data(x),[1,3,8.5,3,5,7,6,8.5,3,10])
        x = masked_array([[0,1,1,1,2], [3,4,5,5,6,]])
        assert_almost_equal(rank_data(x),[[1,3,3,3,5],[6,7,8.5,8.5,10]])
        assert_almost_equal(rank_data(x,axis=1),[[1,3,3,3,5],[1,2,3.5,3.5,5]])
        assert_almost_equal(rank_data(x,axis=0),[[1,1,1,1,1],[2,2,2,2,2,]])

#..............................................................................
class TestQuantiles(NumpyTestCase):
    #
    def __init__(self, *args, **kwargs):
        NumpyTestCase.__init__(self, *args, **kwargs)
    #
    def test_hdquantiles(self):
        data = [0.706560797,0.727229578,0.990399276,0.927065621,0.158953014,
            0.887764025,0.239407086,0.349638551,0.972791145,0.149789972,
            0.936947700,0.132359948,0.046041972,0.641675031,0.945530547,
            0.224218684,0.771450991,0.820257774,0.336458052,0.589113496,
            0.509736129,0.696838829,0.491323573,0.622767425,0.775189248,
            0.641461450,0.118455200,0.773029450,0.319280007,0.752229111,
            0.047841438,0.466295911,0.583850781,0.840581845,0.550086491,
            0.466470062,0.504765074,0.226855960,0.362641207,0.891620942,
            0.127898691,0.490094097,0.044882048,0.041441695,0.317976349,
            0.504135618,0.567353033,0.434617473,0.636243375,0.231803616,
            0.230154113,0.160011327,0.819464108,0.854706985,0.438809221,
            0.487427267,0.786907310,0.408367937,0.405534192,0.250444460,
            0.995309248,0.144389588,0.739947527,0.953543606,0.680051621,
            0.388382017,0.863530727,0.006514031,0.118007779,0.924024803,
            0.384236354,0.893687694,0.626534881,0.473051932,0.750134705,
            0.241843555,0.432947602,0.689538104,0.136934797,0.150206859,
            0.474335206,0.907775349,0.525869295,0.189184225,0.854284286,
            0.831089744,0.251637345,0.587038213,0.254475554,0.237781276,
            0.827928620,0.480283781,0.594514455,0.213641488,0.024194386,
            0.536668589,0.699497811,0.892804071,0.093835427,0.731107772]
        #
        assert_almost_equal(hdquantiles(data,[0., 1.]),
                            [0.006514031, 0.995309248])
        hdq = hdquantiles(data,[0.25, 0.5, 0.75])
        assert_almost_equal(hdq, [0.253210762, 0.512847491, 0.762232442,])
        hdq = hdquantiles_sd(data,[0.25, 0.5, 0.75])
        assert_almost_equal(hdq, [0.03786954, 0.03805389, 0.03800152,], 4)
        #
        data = numpy.array(data).reshape(10,10)
        hdq = hdquantiles(data,[0.25,0.5,0.75],axis=0)
        assert_almost_equal(hdq[:,0], hdquantiles(data[:,0],[0.25,0.5,0.75]))
        assert_almost_equal(hdq[:,-1], hdquantiles(data[:,-1],[0.25,0.5,0.75]))
        hdq = hdquantiles(data,[0.25,0.5,0.75],axis=0,var=True)
        assert_almost_equal(hdq[...,0],
                            hdquantiles(data[:,0],[0.25,0.5,0.75],var=True))
        assert_almost_equal(hdq[...,-1],
                            hdquantiles(data[:,-1],[0.25,0.5,0.75], var=True))


###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()
