import sys
from numpy.testing import *
import numpy
from numpy import zeros, ones, array


# This is the structure of the table used for plain objects:
#
# +-+-+-+
# |x|y|z|
# +-+-+-+

# Structure of a plain array description:
Pdescr = [
    ('x', 'i4', (2,)),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

# A plain list of tuples with values for testing:
PbufferT = [
    # x     y                  z
    ([3,2], [[6.,4.],[6.,4.]], 8),
    ([4,3], [[7.,5.],[7.,5.]], 9),
    ]


# This is the structure of the table used for nested objects (DON'T PANIC!):
#
# +-+---------------------------------+-----+----------+-+-+
# |x|Info                             |color|info      |y|z|
# | +-----+--+----------------+----+--+     +----+-----+ | |
# | |value|y2|Info2           |name|z2|     |Name|Value| | |
# | |     |  +----+-----+--+--+    |  |     |    |     | | |
# | |     |  |name|value|y3|z3|    |  |     |    |     | | |
# +-+-----+--+----+-----+--+--+----+--+-----+----+-----+-+-+
#

# The corresponding nested array description:
Ndescr = [
    ('x', 'i4', (2,)),
    ('Info', [
        ('value', 'c16'),
        ('y2', 'f8'),
        ('Info2', [
            ('name', 'S2'),
            ('value', 'c16', (2,)),
            ('y3', 'f8', (2,)),
            ('z3', 'u4', (2,))]),
        ('name', 'S2'),
        ('z2', 'b1')]),
    ('color', 'S2'),
    ('info', [
        ('Name', 'U8'),
        ('Value', 'c16')]),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

NbufferT = [
    # x     Info                                                color info        y                  z
    #       value y2 Info2                            name z2         Name Value
    #                name   value    y3       z3
    ([3,2], (6j, 6., ('nn', [6j,4j], [6.,4.], [1,2]), 'NN', True), 'cc', ('NN', 6j), [[6.,4.],[6.,4.]], 8),
    ([4,3], (7j, 7., ('oo', [7j,5j], [7.,5.], [2,1]), 'OO', False), 'dd', ('OO', 7j), [[7.,5.],[7.,5.]], 9),
    ]


byteorder = {'little':'<', 'big':'>'}[sys.byteorder]

def normalize_descr(descr):
    "Normalize a description adding the platform byteorder."

    out = []
    for item in descr:
        dtype = item[1]
        if isinstance(dtype, str):
            if dtype[0] not in ['|','<','>']:
                onebyte = dtype[1:] == "1"
                if onebyte or dtype[0] in ['S', 'V', 'b']:
                    dtype = "|" + dtype
                else:
                    dtype = byteorder + dtype
            if len(item) > 2 and item[2] > 1:
                nitem = (item[0], dtype, item[2])
            else:
                nitem = (item[0], dtype)
            out.append(nitem)
        elif isinstance(item[1], list):
            l = []
            for j in normalize_descr(item[1]):
                l.append(j)
            out.append((item[0], l))
        else:
            raise ValueError("Expected a str or list and got %s" % \
                             (type(item)))
    return out


############################################################
#    Creation tests
############################################################

class create_zeros:
    """Check the creation of heterogeneous arrays zero-valued"""

    def check_zeros0D(self):
        """Check creation of 0-dimensional objects"""
        h = zeros((), dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        self.assert_(h.dtype.fields['x'][0].name[:4] == 'void')
        self.assert_(h.dtype.fields['x'][0].char == 'V')
        self.assert_(h.dtype.fields['x'][0].type == numpy.void)
        # A small check that data is ok
        assert_equal(h['z'], zeros((), dtype='u1'))

    def check_zerosSD(self):
        """Check creation of single-dimensional objects"""
        h = zeros((2,), dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        self.assert_(h.dtype['y'].name[:4] == 'void')
        self.assert_(h.dtype['y'].char == 'V')
        self.assert_(h.dtype['y'].type == numpy.void)
        # A small check that data is ok
        assert_equal(h['z'], zeros((2,), dtype='u1'))

    def check_zerosMD(self):
        """Check creation of multi-dimensional objects"""
        h = zeros((2,3), dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        self.assert_(h.dtype['z'].name == 'uint8')
        self.assert_(h.dtype['z'].char == 'B')
        self.assert_(h.dtype['z'].type == numpy.uint8)
        # A small check that data is ok
        assert_equal(h['z'], zeros((2,3), dtype='u1'))


class test_create_zeros_plain(create_zeros, NumpyTestCase):
    """Check the creation of heterogeneous arrays zero-valued (plain)"""
    _descr = Pdescr

class test_create_zeros_nested(create_zeros, NumpyTestCase):
    """Check the creation of heterogeneous arrays zero-valued (nested)"""
    _descr = Ndescr


class create_values:
    """Check the creation of heterogeneous arrays with values"""

    def check_tuple(self):
        """Check creation from tuples"""
        h = array(self._buffer, dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            self.assert_(h.shape == (2,))
        else:
            self.assert_(h.shape == ())

    def check_list_of_tuple(self):
        """Check creation from list of tuples"""
        h = array([self._buffer], dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            self.assert_(h.shape == (1,2))
        else:
            self.assert_(h.shape == (1,))

    def check_list_of_list_of_tuple(self):
        """Check creation from list of list of tuples"""
        h = array([[self._buffer]], dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            self.assert_(h.shape == (1,1,2))
        else:
            self.assert_(h.shape == (1,1))


class test_create_values_plain_single(create_values, NumpyTestCase):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    _descr = Pdescr
    multiple_rows = 0
    _buffer = PbufferT[0]

class test_create_values_plain_multiple(create_values, NumpyTestCase):
    """Check the creation of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr
    multiple_rows = 1
    _buffer = PbufferT

class test_create_values_nested_single(create_values, NumpyTestCase):
    """Check the creation of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr
    multiple_rows = 0
    _buffer = NbufferT[0]

class test_create_values_nested_multiple(create_values, NumpyTestCase):
    """Check the creation of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr
    multiple_rows = 1
    _buffer = NbufferT


############################################################
#    Reading tests
############################################################

class read_values_plain:
    """Check the reading of values in heterogeneous arrays (plain)"""

    def check_access_fields(self):
        h = array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            self.assert_(h.shape == ())
            assert_equal(h['x'], array(self._buffer[0], dtype='i4'))
            assert_equal(h['y'], array(self._buffer[1], dtype='f8'))
            assert_equal(h['z'], array(self._buffer[2], dtype='u1'))
        else:
            self.assert_(len(h) == 2)
            assert_equal(h['x'], array([self._buffer[0][0],
                                             self._buffer[1][0]], dtype='i4'))
            assert_equal(h['y'], array([self._buffer[0][1],
                                             self._buffer[1][1]], dtype='f8'))
            assert_equal(h['z'], array([self._buffer[0][2],
                                             self._buffer[1][2]], dtype='u1'))


class test_read_values_plain_single(read_values_plain, NumpyTestCase):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    _descr = Pdescr
    multiple_rows = 0
    _buffer = PbufferT[0]

class test_read_values_plain_multiple(read_values_plain, NumpyTestCase):
    """Check the values of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr
    multiple_rows = 1
    _buffer = PbufferT

class read_values_nested:
    """Check the reading of values in heterogeneous arrays (nested)"""


    def check_access_top_fields(self):
        """Check reading the top fields of a nested array"""
        h = array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            self.assert_(h.shape == ())
            assert_equal(h['x'], array(self._buffer[0], dtype='i4'))
            assert_equal(h['y'], array(self._buffer[4], dtype='f8'))
            assert_equal(h['z'], array(self._buffer[5], dtype='u1'))
        else:
            self.assert_(len(h) == 2)
            assert_equal(h['x'], array([self._buffer[0][0],
                                             self._buffer[1][0]], dtype='i4'))
            assert_equal(h['y'], array([self._buffer[0][4],
                                             self._buffer[1][4]], dtype='f8'))
            assert_equal(h['z'], array([self._buffer[0][5],
                                             self._buffer[1][5]], dtype='u1'))


    def check_nested1_acessors(self):
        """Check reading the nested fields of a nested array (1st level)"""
        h = array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_equal(h['Info']['value'],
                         array(self._buffer[1][0], dtype='c16'))
            assert_equal(h['Info']['y2'],
                         array(self._buffer[1][1], dtype='f8'))
            assert_equal(h['info']['Name'],
                         array(self._buffer[3][0], dtype='U2'))
            assert_equal(h['info']['Value'],
                         array(self._buffer[3][1], dtype='c16'))
        else:
            assert_equal(h['Info']['value'],
                         array([self._buffer[0][1][0],
                                self._buffer[1][1][0]],
                                dtype='c16'))
            assert_equal(h['Info']['y2'],
                         array([self._buffer[0][1][1],
                                self._buffer[1][1][1]],
                                dtype='f8'))
            assert_equal(h['info']['Name'],
                         array([self._buffer[0][3][0],
                                self._buffer[1][3][0]],
                               dtype='U2'))
            assert_equal(h['info']['Value'],
                         array([self._buffer[0][3][1],
                                self._buffer[1][3][1]],
                               dtype='c16'))

    def check_nested2_acessors(self):
        """Check reading the nested fields of a nested array (2nd level)"""
        h = array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_equal(h['Info']['Info2']['value'],
                         array(self._buffer[1][2][1], dtype='c16'))
            assert_equal(h['Info']['Info2']['z3'],
                         array(self._buffer[1][2][3], dtype='u4'))
        else:
            assert_equal(h['Info']['Info2']['value'],
                         array([self._buffer[0][1][2][1],
                                self._buffer[1][1][2][1]],
                               dtype='c16'))
            assert_equal(h['Info']['Info2']['z3'],
                         array([self._buffer[0][1][2][3],
                                self._buffer[1][1][2][3]],
                               dtype='u4'))

    def check_nested1_descriptor(self):
        """Check access nested descriptors of a nested array (1st level)"""
        h = array(self._buffer, dtype=self._descr)
        self.assert_(h.dtype['Info']['value'].name == 'complex128')
        self.assert_(h.dtype['Info']['y2'].name == 'float64')
        self.assert_(h.dtype['info']['Name'].name == 'unicode256')
        self.assert_(h.dtype['info']['Value'].name == 'complex128')

    def check_nested2_descriptor(self):
        """Check access nested descriptors of a nested array (2nd level)"""
        h = array(self._buffer, dtype=self._descr)
        self.assert_(h.dtype['Info']['Info2']['value'].name == 'void256')
        self.assert_(h.dtype['Info']['Info2']['z3'].name == 'void64')


class test_read_values_nested_single(read_values_nested, NumpyTestCase):
    """Check the values of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr
    multiple_rows = False
    _buffer = NbufferT[0]

class test_read_values_nested_multiple(read_values_nested, NumpyTestCase):
    """Check the values of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr
    multiple_rows = True
    _buffer = NbufferT

class TestEmptyField(NumpyTestCase):
    def check_assign(self):
        a = numpy.arange(10, dtype=numpy.float32)
        a.dtype = [("int",   "<0i4"),("float", "<2f4")]
        assert(a['int'].shape == (5,0))
        assert(a['float'].shape == (5,2))

class TestCommonType(NumpyTestCase):
    def check_scalar_loses1(self):
        res = numpy.find_common_type(['f4','f4','i4'],['f8'])
        assert(res == 'f4')
    def check_scalar_loses2(self):
        res = numpy.find_common_type(['f4','f4'],['i8'])
        assert(res == 'f4')
    def check_scalar_wins(self):
        res = numpy.find_common_type(['f4','f4','i4'],['c8'])
        assert(res == 'c8')
    def check_scalar_wins2(self):
        res = numpy.find_common_type(['u4','i4','i4'],['f4'])
        assert(res == 'f8')
    def check_scalar_wins3(self): # doesn't go up to 'f16' on purpose
        res = numpy.find_common_type(['u8','i8','i8'],['f8'])
        assert(res == 'f8')

        

        
        

if __name__ == "__main__":
    NumpyTest().run()
