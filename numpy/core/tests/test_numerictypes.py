import sys
from numpy.testing import *
set_package_path()
from numpy import zeros, ones, array
restore_path()


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
        ('Name', 'S2'),
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
    "Normalize a description adding the addient byteorder."

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
#    Creating tests
############################################################

class create_zeros(ScipyTestCase):
    """Check the creation of heterogeneous arrays zero-valued"""

    def check_zerosScalar(self):
        """Check creation of multirow objects"""
        h = zeros((), dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        # A small check that data is ok
        assert_equal(h['z'], zeros((), dtype='u1'))

    def check_zerosSD(self):
        """Check creation of multirow objects"""
        h = zeros((2,), dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        # A small check that data is ok
        assert_equal(h['z'], zeros((2,), dtype='u1'))

    def check_zerosMD(self):
        """Check creation of multidimensional objects"""
        h = zeros((2,3), dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        # A small check that data is ok
        assert_equal(h['z'], zeros((2,3), dtype='u1'))


class test_create_zeros_plain(create_zeros):
    """Check the creation of heterogeneous arrays zero-valued (plain)"""
    _descr = Pdescr

class test_create_zeros_nested(create_zeros):
    """Check the creation of heterogeneous arrays zero-valued (nested)"""
    _descr = Ndescr


class create_values(ScipyTestCase):
    """Check the creation of heterogeneous arrays with values"""


    def check_tuple(self):
        """Check creation from tuples"""
        h = array(self._bufferT, dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            self.assert_(h.shape == (2,))
        else:
            self.assert_(h.shape == ())

    def check_list_of_tuple(self):
        """Check creation from list of tuples"""
        h = array([self._bufferT], dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            self.assert_(h.shape == (1,2))
        else:
            self.assert_(h.shape == (1,))

    def check_list_of_list_of_tuple(self):
        """Check creation from list of list of tuples"""
        h = array([[self._bufferT]], dtype=self._descr)
        self.assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            self.assert_(h.shape == (1,1,2))
        else:
            self.assert_(h.shape == (1,1))


class test_create_values_plain_single(create_values):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    _descr = Pdescr
    multiple_rows = 0
    _bufferT = PbufferT[0]

class test_create_values_plain_multiple(create_values):
    """Check the creation of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr
    multiple_rows = 1
    _bufferT = PbufferT

class test_create_values_nested_single(create_values):
    """Check the creation of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr
    multiple_rows = 0
    _bufferT = NbufferT[0]

class test_create_values_nested_multiple(create_values):
    """Check the creation of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr
    multiple_rows = 1
    _bufferT = NbufferT


############################################################
#    Reading tests
############################################################

class read_values_plain(ScipyTestCase):
    """Check the reading of values in heterogeneous arrays (plain)"""

    def is_correct(self):
        if self.multiple_rows:
            assert_equal(self.h['x'], array(self._buffer[0][0], dtype='i4'))
            assert_equal(self.h['y'], array(self._buffer[0][1], dtype='f8'))
            assert_equal(self.h['z'], array(self._buffer[0][2], dtype='u1'))
        else:
            assert_equal(self.h['x'], array([self._buffer[0][0],
                                             self._buffer[1][0]], dtype='i4'))
            assert_equal(self.h['y'], array([self._buffer[0][1],
                                             self._buffer[1][1]], dtype='f8'))
            assert_equal(self.h['z'], array([self._buffer[0][2],
                                             self._buffer[1][2]], dtype='u1'))


    def check_read_full_tuples(self):
        """Check reading from objects created from tuples"""
        self._buffer = self._bufferT
        self.h = array(self._buffer, dtype=self._descr)


class test_read_values_plain_single(read_values_plain):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    _descr = Pdescr
    multiple_rows = 0
    _bufferT = PbufferT[0]

class test_read_values_plain_multiple(read_values_plain):
    """Check the values of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr
    multiple_rows = 1
    _bufferT = PbufferT

class read_values_nested(read_values_plain):
    """Check the reading of values in heterogeneous arrays (nested)"""

    
    # Uncomment this when numpy will eventually support lists as inputs.
    def _check_read_full_list(self):
        """Check reading from objects created from list"""
        h = array(self._bufferT, dtype=self._descr)
        # Add here more code to check this...

    def check_read_full_tuple(self):
        """Check reading from objects created from tuples"""
        h = array(self._bufferT, dtype=self._descr)
        # Add here more code to check this...

# The next test classes are not finished yet...
class _test_read_values_nested_single(read_values_nested):
    """Check the values of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr
    multiple_rows = 0
    _bufferT = NbufferT[0]

class _test_read_values_nested_multiple(read_values_nested):
    """Check the values of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr
    multiple_rows = 1
    _bufferT = NbufferT


if __name__ == "__main__":
    ScipyTest().run()
