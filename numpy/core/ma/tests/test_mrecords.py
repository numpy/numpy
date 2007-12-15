# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for mrecarray.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_mrecords.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

import types

import numpy as N
import numpy.core.fromnumeric  as fromnumeric
from numpy.testing import NumpyTest, NumpyTestCase
from numpy.testing.utils import build_err_msg

import maskedarray.testutils
from maskedarray.testutils import *

import maskedarray
from maskedarray import masked_array, masked, nomask

#import maskedarray.mrecords
#from maskedarray.mrecords import mrecarray, fromarrays, fromtextfile, fromrecords
import maskedarray.mrecords
from maskedarray.mrecords import MaskedRecords, \
    fromarrays, fromtextfile, fromrecords, addfield

#..............................................................................
class TestMRecords(NumpyTestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        self.setup()

    def setup(self):
        "Generic setup"
        d = N.arange(5)
        m = maskedarray.make_mask([1,0,0,1,1])
        base_d = N.r_[d,d[::-1]].reshape(2,-1).T
        base_m = N.r_[[m, m[::-1]]].T
        base = masked_array(base_d, mask=base_m)
        mrecord = fromarrays(base.T, dtype=[('a',N.float_),('b',N.float_)])
        self.data = [d, m, mrecord]

    def test_get(self):
        "Tests fields retrieval"
        [d, m, mrec] = self.data
        mrec = mrec.copy()
        assert_equal(mrec.a, masked_array(d,mask=m))
        assert_equal(mrec.b, masked_array(d[::-1],mask=m[::-1]))
        assert((mrec._fieldmask == N.core.records.fromarrays([m, m[::-1]], dtype=mrec._fieldmask.dtype)).all())
        assert_equal(mrec._mask, N.r_[[m,m[::-1]]].all(0))
        assert_equal(mrec.a[1], mrec[1].a)
        #
        assert(isinstance(mrec[:2], MaskedRecords))
        assert_equal(mrec[:2]['a'], d[:2])

    def test_set(self):
        "Tests setting fields/attributes."
        [d, m, mrecord] = self.data
        mrecord.a._data[:] = 5
        assert_equal(mrecord['a']._data, [5,5,5,5,5])
        mrecord.a = 1
        assert_equal(mrecord['a']._data, [1]*5)
        assert_equal(getmaskarray(mrecord['a']), [0]*5)
        mrecord.b = masked
        assert_equal(mrecord.b.mask, [1]*5)
        assert_equal(getmaskarray(mrecord['b']), [1]*5)
        mrecord._mask = masked
        assert_equal(getmaskarray(mrecord['b']), [1]*5)
        assert_equal(mrecord['a']._mask, mrecord['b']._mask)
        mrecord._mask = nomask
        assert_equal(getmaskarray(mrecord['b']), [0]*5)
        assert_equal(mrecord['a']._mask, mrecord['b']._mask)
        #
    def test_setfields(self):
        "Tests setting fields."
        [d, m, mrecord] = self.data
        mrecord.a[3:] = 5
        assert_equal(mrecord.a, [0,1,2,5,5])
        assert_equal(mrecord.a._mask, [1,0,0,0,0])
        #
        mrecord.b[3:] = masked
        assert_equal(mrecord.b, [4,3,2,1,0])
        assert_equal(mrecord.b._mask, [1,1,0,1,1])

    def test_setslices(self):
        "Tests setting slices."
        [d, m, mrec] = self.data
        mrec[:2] = 5
        assert_equal(mrec.a._data, [5,5,2,3,4])
        assert_equal(mrec.b._data, [5,5,2,1,0])
        assert_equal(mrec.a._mask, [0,0,0,1,1])
        assert_equal(mrec.b._mask, [0,0,0,0,1])
        #
        mrec[:2] = masked
        assert_equal(mrec._mask, [1,1,0,0,1])
        mrec[-2] = masked
        assert_equal(mrec._mask, [1,1,0,1,1])
        #
    def test_setslices_hardmask(self):
        "Tests setting slices w/ hardmask."
        [d, m, mrec] = self.data
        mrec.harden_mask()
        mrec[-2:] = 5
        assert_equal(mrec.a._data, [0,1,2,3,4])
        assert_equal(mrec.b._data, [4,3,2,5,0])
        assert_equal(mrec.a._mask, [1,0,0,1,1])
        assert_equal(mrec.b._mask, [1,1,0,0,1])

    def test_hardmask(self):
        "Test hardmask"
        [d, m, mrec] = self.data
        mrec = mrec.copy()
        mrec.harden_mask()
        assert(mrec._hardmask)
        mrec._mask = nomask
        assert_equal(mrec._mask, N.r_[[m,m[::-1]]].all(0))
        mrec.soften_mask()
        assert(not mrec._hardmask)
        mrec._mask = nomask
        assert(mrec['b']._mask is nomask)
        assert_equal(mrec['a']._mask,mrec['b']._mask)

    def test_fromrecords(self):
        "Test from recarray."
        [d, m, mrec] = self.data
        nrec = N.core.records.fromarrays(N.r_[[d,d[::-1]]],
                                         dtype=[('a',N.float_),('b',N.float_)])
        #....................
        mrecfr = fromrecords(nrec)
        assert_equal(mrecfr.a, mrec.a)
        assert_equal(mrecfr.dtype, mrec.dtype)
        #....................
        tmp = mrec[::-1] #.tolist()
        mrecfr = fromrecords(tmp)
        assert_equal(mrecfr.a, mrec.a[::-1])
        #....................
        mrecfr = fromrecords(nrec.tolist(), names=nrec.dtype.names)
        assert_equal(mrecfr.a, mrec.a)
        assert_equal(mrecfr.dtype, mrec.dtype)

    def test_fromtextfile(self):
        "Tests reading from a text file."
        fcontent = """#
'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
'strings',1,1.0,'mixed column',,1
'with embedded "double quotes"',2,2.0,1.0,,1
'strings',3,3.0E5,3,,1
'strings',4,-1e-10,,,1
"""
        import os
        from datetime import datetime
        fname = 'tmp%s' % datetime.now().strftime("%y%m%d%H%M%S%s")
        f = open(fname, 'w')
        f.write(fcontent)
        f.close()
        mrectxt = fromtextfile(fname,delimitor=',',varnames='ABCDEFG')
        os.unlink(fname)
        #
        assert(isinstance(mrectxt, MaskedRecords))
        assert_equal(mrectxt.F, [1,1,1,1])
        assert_equal(mrectxt.E._mask, [1,1,1,1])
        assert_equal(mrectxt.C, [1,2,3.e+5,-1e-10])

    def test_addfield(self):
        "Tests addfield"
        [d, m, mrec] = self.data
        mrec = addfield(mrec, masked_array(d+10, mask=m[::-1]))
        assert_equal(mrec.f2, d+10)
        assert_equal(mrec.f2._mask, m[::-1])

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()
