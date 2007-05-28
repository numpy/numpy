import sys
from numpy.testing import *
from numpy.core import *

# Guess the UCS length for this python interpreter
if len(buffer(u'u')) == 4:
    ucs4 = True
else:
    ucs4 = False

# Value that can be represented in UCS2 interpreters
ucs2_value = u'\uFFFF'
# Value that cannot be represented in UCS2 interpreters (but can in UCS4)
ucs4_value = u'\U0010FFFF'


############################################################
#    Creation tests
############################################################

class create_zeros(NumpyTestCase):
    """Check the creation of zero-valued arrays"""

    def content_test(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        self.assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        self.assert_(len(ua.data) == nbytes)
        # Small check that data in array element is ok
        self.assert_(ua_scalar == u'')
        # Encode to ascii and double check
        self.assert_(ua_scalar.encode('ascii') == '')
        # Check buffer lengths for scalars
        if ucs4:
            self.assert_(len(buffer(ua_scalar)) == 0)
        else:
            self.assert_(len(buffer(ua_scalar)) == 0)

    def check_zeros0D(self):
        """Check creation of 0-dimensional objects"""
        ua = zeros((), dtype='U%s' % self.ulen)
        self.content_test(ua, ua[()], 4*self.ulen)

    def check_zerosSD(self):
        """Check creation of single-dimensional objects"""
        ua = zeros((2,), dtype='U%s' % self.ulen)
        self.content_test(ua, ua[0], 4*self.ulen*2)
        self.content_test(ua, ua[1], 4*self.ulen*2)

    def check_zerosMD(self):
        """Check creation of multi-dimensional objects"""
        ua = zeros((2,3,4), dtype='U%s' % self.ulen)
        self.content_test(ua, ua[0,0,0], 4*self.ulen*2*3*4)
        self.content_test(ua, ua[-1,-1,-1], 4*self.ulen*2*3*4)


class test_create_zeros_1(create_zeros):
    """Check the creation of zero-valued arrays (size 1)"""
    ulen = 1

class test_create_zeros_2(create_zeros):
    """Check the creation of zero-valued arrays (size 2)"""
    ulen = 2

class test_create_zeros_1009(create_zeros):
    """Check the creation of zero-valued arrays (size 1009)"""
    ulen = 1009


class create_values(NumpyTestCase):
    """Check the creation of unicode arrays with values"""

    def content_test(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        self.assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        self.assert_(len(ua.data) == nbytes)
        # Small check that data in array element is ok
        self.assert_(ua_scalar == self.ucs_value*self.ulen)
        # Encode to UTF-8 and double check
        self.assert_(ua_scalar.encode('utf-8') == \
                     (self.ucs_value*self.ulen).encode('utf-8'))
        # Check buffer lengths for scalars
        if ucs4:
            self.assert_(len(buffer(ua_scalar)) == 4*self.ulen)
        else:
            if self.ucs_value == ucs4_value:
                # In UCS2, the \U0010FFFF will be represented using a
                # surrogate *pair*
                self.assert_(len(buffer(ua_scalar)) == 2*2*self.ulen)
            else:
                # In UCS2, the \uFFFF will be represented using a
                # regular 2-byte word
                self.assert_(len(buffer(ua_scalar)) == 2*self.ulen)

    def check_values0D(self):
        """Check creation of 0-dimensional objects with values"""
        ua = array(self.ucs_value*self.ulen, dtype='U%s' % self.ulen)
        self.content_test(ua, ua[()], 4*self.ulen)

    def check_valuesSD(self):
        """Check creation of single-dimensional objects with values"""
        ua = array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        self.content_test(ua, ua[0], 4*self.ulen*2)
        self.content_test(ua, ua[1], 4*self.ulen*2)

    def check_valuesMD(self):
        """Check creation of multi-dimensional objects with values"""
        ua = array([[[self.ucs_value*self.ulen]*2]*3]*4, dtype='U%s' % self.ulen)
        self.content_test(ua, ua[0,0,0], 4*self.ulen*2*3*4)
        self.content_test(ua, ua[-1,-1,-1], 4*self.ulen*2*3*4)


class test_create_values_1_ucs2(create_values):
    """Check the creation of valued arrays (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value

class test_create_values_1_ucs4(create_values):
    """Check the creation of valued arrays (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value

class test_create_values_2_ucs2(create_values):
    """Check the creation of valued arrays (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value

class test_create_values_2_ucs4(create_values):
    """Check the creation of valued arrays (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value

class test_create_values_1009_ucs2(create_values):
    """Check the creation of valued arrays (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value

class test_create_values_1009_ucs4(create_values):
    """Check the creation of valued arrays (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value


############################################################
#    Assignment tests
############################################################

class assign_values(NumpyTestCase):
    """Check the assignment of unicode arrays with values"""

    def content_test(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        self.assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        self.assert_(len(ua.data) == nbytes)
        # Small check that data in array element is ok
        self.assert_(ua_scalar == self.ucs_value*self.ulen)
        # Encode to UTF-8 and double check
        self.assert_(ua_scalar.encode('utf-8') == \
                     (self.ucs_value*self.ulen).encode('utf-8'))
        # Check buffer lengths for scalars
        if ucs4:
            self.assert_(len(buffer(ua_scalar)) == 4*self.ulen)
        else:
            if self.ucs_value == ucs4_value:
                # In UCS2, the \U0010FFFF will be represented using a
                # surrogate *pair*
                self.assert_(len(buffer(ua_scalar)) == 2*2*self.ulen)
            else:
                # In UCS2, the \uFFFF will be represented using a
                # regular 2-byte word
                self.assert_(len(buffer(ua_scalar)) == 2*self.ulen)

    def check_values0D(self):
        """Check assignment of 0-dimensional objects with values"""
        ua = zeros((), dtype='U%s' % self.ulen)
        ua[()] = self.ucs_value*self.ulen
        self.content_test(ua, ua[()], 4*self.ulen)

    def check_valuesSD(self):
        """Check assignment of single-dimensional objects with values"""
        ua = zeros((2,), dtype='U%s' % self.ulen)
        ua[0] = self.ucs_value*self.ulen
        self.content_test(ua, ua[0], 4*self.ulen*2)
        ua[1] = self.ucs_value*self.ulen
        self.content_test(ua, ua[1], 4*self.ulen*2)

    def check_valuesMD(self):
        """Check assignment of multi-dimensional objects with values"""
        ua = zeros((2,3,4), dtype='U%s' % self.ulen)
        ua[0,0,0] = self.ucs_value*self.ulen
        self.content_test(ua, ua[0,0,0], 4*self.ulen*2*3*4)
        ua[-1,-1,-1] = self.ucs_value*self.ulen
        self.content_test(ua, ua[-1,-1,-1], 4*self.ulen*2*3*4)


class test_assign_values_1_ucs2(assign_values):
    """Check the assignment of valued arrays (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value

class test_assign_values_1_ucs4(assign_values):
    """Check the assignment of valued arrays (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value

class test_assign_values_2_ucs2(assign_values):
    """Check the assignment of valued arrays (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value

class test_assign_values_2_ucs4(assign_values):
    """Check the assignment of valued arrays (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value

class test_assign_values_1009_ucs2(assign_values):
    """Check the assignment of valued arrays (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value

class test_assign_values_1009_ucs4(assign_values):
    """Check the assignment of valued arrays (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value


############################################################
#    Byteorder tests
############################################################

class byteorder_values(NumpyTestCase):
    """Check the byteorder of unicode arrays in round-trip conversions"""

    def check_values0D(self):
        """Check byteorder of 0-dimensional objects"""
        ua = array(self.ucs_value*self.ulen, dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        # This changes the interpretation of the data region (but not the
        #  actual data), therefore the returned scalars are not
        #  the same (they are byte-swapped versions of each other).
        self.assert_(ua[()] != ua2[()])
        ua3 = ua2.newbyteorder()
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def check_valuesSD(self):
        """Check byteorder of single-dimensional objects"""
        ua = array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        self.assert_(ua[0] != ua2[0])
        self.assert_(ua[-1] != ua2[-1])
        ua3 = ua2.newbyteorder()
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def check_valuesMD(self):
        """Check byteorder of multi-dimensional objects"""
        ua = array([[[self.ucs_value*self.ulen]*2]*3]*4,
                   dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        self.assert_(ua[0,0,0] != ua2[0,0,0])
        self.assert_(ua[-1,-1,-1] != ua2[-1,-1,-1])
        ua3 = ua2.newbyteorder()
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

class test_byteorder_1_ucs2(byteorder_values):
    """Check the byteorder in unicode (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value

class test_byteorder_1_ucs4(byteorder_values):
    """Check the byteorder in unicode (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value

class test_byteorder_2_ucs2(byteorder_values):
    """Check the byteorder in unicode (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value

class test_byteorder_2_ucs4(byteorder_values):
    """Check the byteorder in unicode (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value

class test_byteorder_1009_ucs2(byteorder_values):
    """Check the byteorder in unicode (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value

class test_byteorder_1009_ucs4(byteorder_values):
    """Check the byteorder in unicode (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value


if __name__ == "__main__":
    NumpyTest().run()
