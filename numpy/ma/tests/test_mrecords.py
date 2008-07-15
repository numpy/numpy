# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for mrecords.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

import types

import numpy as np
from numpy import recarray
from numpy.core.records import fromrecords as recfromrecords, \
    fromarrays as recfromarrays

import numpy.ma.testutils
from numpy.ma.testutils import *

import numpy.ma as ma
from numpy.ma import masked, nomask, getdata, getmaskarray

import numpy.ma.mrecords

from numpy.ma.mrecords import MaskedRecords, mrecarray,\
    fromarrays, fromtextfile, fromrecords, addfield

#..............................................................................
class TestMRecords(NumpyTestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        self.setup()

    def setup(self):
        "Generic setup"
        ilist = [1,2,3,4,5]
        flist = [1.1,2.2,3.3,4.4,5.5]
        slist = ['one','two','three','four','five']
        ddtype = [('a',int),('b',float),('c','|S8')]
        mask = [0,1,0,0,1]
        self.base = ma.array(zip(ilist,flist,slist), mask=mask, dtype=ddtype)

    def test_byview(self):
        "Test creation by view"
        base = self.base
        mbase = base.view(mrecarray)
        assert_equal(mbase._mask, base._mask)
        assert isinstance(mbase._data, recarray)
        assert_equal_records(mbase._data, base._data.view(recarray))
        for field in ('a','b','c'):
            assert_equal(base[field], mbase[field])
        assert_equal_records(mbase.view(mrecarray), mbase)

    def test_get(self):
        "Tests fields retrieval"
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # As fields..........
        for field in ('a','b','c'):
            assert_equal(getattr(mbase,field), mbase[field])
            assert_equal(base[field], mbase[field])
        # as elements .......
        mbase_first = mbase[0]
        assert isinstance(mbase_first, mrecarray)
        assert_equal(mbase_first.dtype, mbase.dtype)
        assert_equal(mbase_first.tolist(), (1,1.1,'one'))
        assert_equal(mbase_first.mask, nomask)
        assert_equal(mbase_first._fieldmask.item(), (False, False, False))
        assert_equal(mbase_first['a'], mbase['a'][0])
        mbase_last = mbase[-1]
        assert isinstance(mbase_last, mrecarray)
        assert_equal(mbase_last.dtype, mbase.dtype)
        assert_equal(mbase_last.tolist(), (None,None,None))
        assert_equal(mbase_last.mask, True)
        assert_equal(mbase_last._fieldmask.item(), (True, True, True))
        assert_equal(mbase_last['a'], mbase['a'][-1])
        assert (mbase_last['a'] is masked)
        # as slice ..........
        mbase_sl = mbase[:2]
        assert isinstance(mbase_sl, mrecarray)
        assert_equal(mbase_sl.dtype, mbase.dtype)
        assert_equal(mbase_sl._mask, [0,1])
        assert_equal_records(mbase_sl, base[:2].view(mrecarray))
        for field in ('a','b','c'):
            assert_equal(getattr(mbase_sl,field), base[:2][field])

    def test_set_fields(self):
        "Tests setting fields."
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase = mbase.copy()
        mbase.fill_value = (999999,1e20,'N/A')
        # Change the data, the mask should be conserved
        mbase.a._data[:] = 5
        assert_equal(mbase['a']._data, [5,5,5,5,5])
        assert_equal(mbase['a']._mask, [0,1,0,0,1])
        # Change the elements, and the mask will follow
        mbase.a = 1
        assert_equal(mbase['a']._data, [1]*5)
        assert_equal(ma.getmaskarray(mbase['a']), [0]*5)
        assert_equal(mbase._mask, [False]*5)
        assert_equal(mbase._fieldmask.tolist(),
                     np.array([(0,0,0),(0,1,1),(0,0,0),(0,0,0),(0,1,1)],
                              dtype=bool))
        # Set a field to mask ........................
        mbase.c = masked
        assert_equal(mbase.c.mask, [1]*5)
        assert_equal(ma.getmaskarray(mbase['c']), [1]*5)
        assert_equal(ma.getdata(mbase['c']), ['N/A']*5)
        assert_equal(mbase._fieldmask.tolist(),
                     np.array([(0,0,1),(0,1,1),(0,0,1),(0,0,1),(0,1,1)],
                              dtype=bool))
        # Set fields by slices .......................
        mbase = base.view(mrecarray).copy()
        mbase.a[3:] = 5
        assert_equal(mbase.a, [1,2,3,5,5])
        assert_equal(mbase.a._mask, [0,1,0,0,0])
        mbase.b[3:] = masked
        assert_equal(mbase.b, base['b'])
        assert_equal(mbase.b._mask, [0,1,0,1,1])
        # Set fields globally..........................
        ndtype = [('alpha','|S1'),('num',int)]
        data = ma.array([('a',1),('b',2),('c',3)], dtype=ndtype)
        rdata = data.view(MaskedRecords)
        val = ma.array([10,20,30], mask=[1,0,0])
        #
        import warnings
        warnings.simplefilter("ignore")
        rdata['num'] = val
        assert_equal(rdata.num, val)
        assert_equal(rdata.num.mask, [1,0,0])

    #
    def test_set_mask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # Set the mask to True .......................
        mbase._mask = masked
        assert_equal(ma.getmaskarray(mbase['b']), [1]*5)
        assert_equal(mbase['a']._mask, mbase['b']._mask)
        assert_equal(mbase['a']._mask, mbase['c']._mask)
        assert_equal(mbase._fieldmask.tolist(),
                     np.array([(1,1,1)]*5, dtype=bool))
        # Delete the mask ............................
        mbase._mask = nomask
        assert_equal(ma.getmaskarray(mbase['c']), [0]*5)
        assert_equal(mbase._fieldmask.tolist(),
                     np.array([(0,0,0)]*5, dtype=bool))
    #
    def test_set_mask_fromarray(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # Sets the mask w/ an array
        mbase._mask = [1,0,0,0,1]
        assert_equal(mbase.a.mask, [1,0,0,0,1])
        assert_equal(mbase.b.mask, [1,0,0,0,1])
        assert_equal(mbase.c.mask, [1,0,0,0,1])
        # Yay, once more !
        mbase.mask = [0,0,0,0,1]
        assert_equal(mbase.a.mask, [0,0,0,0,1])
        assert_equal(mbase.b.mask, [0,0,0,0,1])
        assert_equal(mbase.c.mask, [0,0,0,0,1])
    #
    def test_set_mask_fromfields(self):
        mbase = self.base.copy().view(mrecarray)
        #
        nmask = np.array([(0,1,0),(0,1,0),(1,0,1),(1,0,1),(0,0,0)],
                         dtype=[('a',bool),('b',bool),('c',bool)])
        mbase.mask = nmask
        assert_equal(mbase.a.mask, [0,0,1,1,0])
        assert_equal(mbase.b.mask, [1,1,0,0,0])
        assert_equal(mbase.c.mask, [0,0,1,1,0])
        # Reinitalizes and redo
        mbase.mask = False
        mbase.fieldmask = nmask
        assert_equal(mbase.a.mask, [0,0,1,1,0])
        assert_equal(mbase.b.mask, [1,1,0,0,0])
        assert_equal(mbase.c.mask, [0,0,1,1,0])
    #
    def test_set_elements(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # Set an element to mask .....................
        mbase[-2] = masked
        assert_equal(mbase._fieldmask.tolist(),
                     np.array([(0,0,0),(1,1,1),(0,0,0),(1,1,1),(1,1,1)],
                              dtype=bool))
        assert_equal(mbase._mask, [0,1,0,1,1])
        # Set slices .................................
        mbase = base.view(mrecarray).copy()
        mbase[:2] = 5
        assert_equal(mbase.a._data, [5,5,3,4,5])
        assert_equal(mbase.a._mask, [0,0,0,0,1])
        assert_equal(mbase.b._data, [5.,5.,3.3,4.4,5.5])
        assert_equal(mbase.b._mask, [0,0,0,0,1])
        assert_equal(mbase.c._data, ['5','5','three','four','five'])
        assert_equal(mbase.b._mask, [0,0,0,0,1])
        #
        mbase = base.view(mrecarray).copy()
        mbase[:2] = masked
        assert_equal(mbase.a._data, [1,2,3,4,5])
        assert_equal(mbase.a._mask, [1,1,0,0,1])
        assert_equal(mbase.b._data, [1.1,2.2,3.3,4.4,5.5])
        assert_equal(mbase.b._mask, [1,1,0,0,1])
        assert_equal(mbase.c._data, ['one','two','three','four','five'])
        assert_equal(mbase.b._mask, [1,1,0,0,1])
    #
    def test_setslices_hardmask(self):
        "Tests setting slices w/ hardmask."
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        mbase[-2:] = 5
        assert_equal(mbase.a._data, [1,2,3,5,5])
        assert_equal(mbase.b._data, [1.1,2.2,3.3,5,5.5])
        assert_equal(mbase.c._data, ['one','two','three','5','five'])
        assert_equal(mbase.a._mask, [0,1,0,0,1])
        assert_equal(mbase.b._mask, mbase.a._mask)
        assert_equal(mbase.b._mask, mbase.c._mask)

    def test_hardmask(self):
        "Test hardmask"
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        assert(mbase._hardmask)
        mbase._mask = nomask
        assert_equal(mbase._mask, [0,1,0,0,1])
        mbase.soften_mask()
        assert(not mbase._hardmask)
        mbase._mask = nomask
        assert(mbase['b']._mask is nomask)
        assert_equal(mbase['a']._mask,mbase['b']._mask)
    #
    def test_pickling(self):
        "Test pickling"
        import cPickle
        base = self.base.copy()
        mrec = base.view(mrecarray)
        _ = cPickle.dumps(mrec)
        mrec_ = cPickle.loads(_)
        assert_equal(mrec_.dtype, mrec.dtype)
        assert_equal_records(mrec_._data, mrec._data)
        assert_equal(mrec_._mask, mrec._mask)
        assert_equal_records(mrec_._fieldmask, mrec._fieldmask)
    #
    def test_filled(self):
        "Test filling the array"
        _a = ma.array([1,2,3],mask=[0,0,1],dtype=int)
        _b = ma.array([1.1,2.2,3.3],mask=[0,0,1],dtype=float)
        _c = ma.array(['one','two','three'],mask=[0,0,1],dtype='|S8')
        ddtype = [('a',int),('b',float),('c','|S8')]
        mrec = fromarrays([_a,_b,_c], dtype=ddtype,
                          fill_value=(99999,99999.,'N/A'))
        mrecfilled = mrec.filled()
        assert_equal(mrecfilled['a'], np.array((1,2,99999), dtype=int))
        assert_equal(mrecfilled['b'], np.array((1.1,2.2,99999.), dtype=float))
        assert_equal(mrecfilled['c'], np.array(('one','two','N/A'), dtype='|S8'))
    #
    def test_tolist(self):
        "Test tolist."
        _a = ma.array([1,2,3],mask=[0,0,1],dtype=int)
        _b = ma.array([1.1,2.2,3.3],mask=[0,0,1],dtype=float)
        _c = ma.array(['one','two','three'],mask=[1,0,0],dtype='|S8')
        ddtype = [('a',int),('b',float),('c','|S8')]
        mrec = fromarrays([_a,_b,_c], dtype=ddtype,
                          fill_value=(99999,99999.,'N/A'))
        #
        assert_equal(mrec.tolist(),
                     [(1,1.1,None),(2,2.2,'two'),(None,None,'three')])
    #
    def test_withnames(self):
        "Test the creation w/ format and names"
        x = mrecarray(1, formats=float, names='base')
        x[0]['base'] = 10
        assert_equal(x['base'][0], 10)
    #
    def test_exotic_formats(self):
        "Test that 'exotic' formats are processed properly"
        easy = mrecarray(1, dtype=[('i',int), ('s','|S3'), ('f',float)])
        easy[0] = masked
        easy.filled(1)
        assert_equal(easy.filled(1).item(), (1,'1',1.))
        #
        solo = mrecarray(1, dtype=[('f0', '<f8', (2, 2))])
        solo[0] = masked
        assert_equal(solo.filled(1).item(), 
                     np.array((1,), dtype=solo.dtype).item())
        #
        mult = mrecarray(2, dtype= "i4, (2,3)float, float")
        mult[0] = masked
        mult[1] = (1, 1, 1)
        mult.filled(0)
        assert_equal(mult.filled(0),
                     np.array([(0,0,0),(1,1,1)], dtype=mult.dtype))

################################################################################
class TestMRecordsImport(NumpyTestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        self.setup()

    def setup(self):
        "Generic setup"
        _a = ma.array([1,2,3],mask=[0,0,1],dtype=int)
        _b = ma.array([1.1,2.2,3.3],mask=[0,0,1],dtype=float)
        _c = ma.array(['one','two','three'],mask=[0,0,1],dtype='|S8')
        ddtype = [('a',int),('b',float),('c','|S8')]
        mrec = fromarrays([_a,_b,_c], dtype=ddtype,
                          fill_value=(99999,99999.,'N/A'))
        nrec = recfromarrays((_a.data,_b.data,_c.data), dtype=ddtype)
        self.data = (mrec, nrec, ddtype)

    def test_fromarrays(self):
        _a = ma.array([1,2,3],mask=[0,0,1],dtype=int)
        _b = ma.array([1.1,2.2,3.3],mask=[0,0,1],dtype=float)
        _c = ma.array(['one','two','three'],mask=[0,0,1],dtype='|S8')
        (mrec, nrec, _) = self.data
        for (f,l) in zip(('a','b','c'),(_a,_b,_c)):
            assert_equal(getattr(mrec,f)._mask, l._mask)
        # One record only
        _x = ma.array([1,1.1,'one'], mask=[1,0,0],)
        assert_equal_records(fromarrays(_x, dtype=mrec.dtype), mrec[0])



    def test_fromrecords(self):
        "Test construction from records."
        (mrec, nrec, ddtype) = self.data
        #......
        palist = [(1, 'abc', 3.7000002861022949, 0),
                  (2, 'xy', 6.6999998092651367, 1),
                  (0, ' ', 0.40000000596046448, 0)]
        pa = recfromrecords(palist, names='c1, c2, c3, c4')
        mpa = fromrecords(palist, names='c1, c2, c3, c4')
        assert_equal_records(pa,mpa)
        #.....
        _mrec = fromrecords(nrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        for field in _mrec.dtype.names:
            assert_equal(getattr(_mrec, field), getattr(mrec._data, field))
        #
        _mrec = fromrecords(nrec.tolist(), names='c1,c2,c3')
        assert_equal(_mrec.dtype, [('c1',int),('c2',float),('c3','|S5')])
        for (f,n) in zip(('c1','c2','c3'), ('a','b','c')):
            assert_equal(getattr(_mrec,f), getattr(mrec._data, n))
        #
        _mrec = fromrecords(mrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        assert_equal_records(_mrec._data, mrec.filled())
        assert_equal_records(_mrec._fieldmask, mrec._fieldmask)

    def test_fromrecords_wmask(self):
        "Tests construction from records w/ mask."
        (mrec, nrec, ddtype) = self.data
        #
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=[0,1,0,])
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._fieldmask.tolist(), [(0,0,0),(1,1,1),(0,0,0)])
        #
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=True)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._fieldmask.tolist(), [(1,1,1),(1,1,1),(1,1,1)])
        #
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._fieldmask)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._fieldmask.tolist(), mrec._fieldmask.tolist())
        #
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype,
                            mask=mrec._fieldmask.tolist())
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._fieldmask.tolist(), mrec._fieldmask.tolist())

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
        (mrec, nrec, ddtype) = self.data
        (d,m) = ([100,200,300], [1,0,0])
        mrec = addfield(mrec, ma.array(d, mask=m))
        assert_equal(mrec.f3, d)
        assert_equal(mrec.f3._mask, m)

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()
