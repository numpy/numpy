
import StringIO

import numpy as np
from numpy.lib._iotools import LineSplitter, NameValidator, StringConverter,\
                               has_nested_fields
from numpy.testing import *

class TestLineSplitter(TestCase):
    "Tests the LineSplitter class."
    #
    def test_no_delimiter(self):
        "Test LineSplitter w/o delimiter"
        strg = " 1 2 3 4  5 # test"
        test = LineSplitter()(strg)
        assert_equal(test, ['1', '2', '3', '4', '5'])
        test = LineSplitter('')(strg)
        assert_equal(test, ['1', '2', '3', '4', '5'])

    def test_space_delimiter(self):
        "Test space delimiter"
        strg = " 1 2 3 4  5 # test"
        test = LineSplitter(' ')(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5'])
        test = LineSplitter('  ')(strg)
        assert_equal(test, ['1 2 3 4', '5'])

    def test_tab_delimiter(self):
        "Test tab delimiter"
        strg= " 1\t 2\t 3\t 4\t 5  6"
        test = LineSplitter('\t')(strg)
        assert_equal(test, ['1', '2', '3', '4', '5  6'])
        strg= " 1  2\t 3  4\t 5  6"
        test = LineSplitter('\t')(strg)
        assert_equal(test, ['1  2', '3  4', '5  6'])

    def test_other_delimiter(self):
        "Test LineSplitter on delimiter"
        strg = "1,2,3,4,,5"
        test = LineSplitter(',')(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5'])
        #
        strg = " 1,2,3,4,,5 # test"
        test = LineSplitter(',')(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5'])

    def test_constant_fixed_width(self):
        "Test LineSplitter w/ fixed-width fields"
        strg = "  1  2  3  4     5   # test"
        test = LineSplitter(3)(strg)
        assert_equal(test, ['1', '2', '3', '4', '', '5', ''])
        #
        strg = "  1     3  4  5  6# test"
        test = LineSplitter(20)(strg)
        assert_equal(test, ['1     3  4  5  6'])
        #
        strg = "  1     3  4  5  6# test"
        test = LineSplitter(30)(strg)
        assert_equal(test, ['1     3  4  5  6'])

    def test_variable_fixed_width(self):
        strg = "  1     3  4  5  6# test"
        test = LineSplitter((3,6,6,3))(strg)
        assert_equal(test, ['1', '3', '4  5', '6'])
        #
        strg = "  1     3  4  5  6# test"
        test = LineSplitter((6,6,9))(strg)
        assert_equal(test, ['1', '3  4', '5  6'])


#-------------------------------------------------------------------------------

class TestNameValidator(TestCase):
    #
    def test_case_sensitivity(self):
        "Test case sensitivity"
        names = ['A', 'a', 'b', 'c']
        test = NameValidator().validate(names)
        assert_equal(test, ['A', 'a', 'b', 'c'])
        test = NameValidator(case_sensitive=False).validate(names)
        assert_equal(test, ['A', 'A_1', 'B', 'C'])
        test = NameValidator(case_sensitive='upper').validate(names)
        assert_equal(test, ['A', 'A_1', 'B', 'C'])
        test = NameValidator(case_sensitive='lower').validate(names)
        assert_equal(test, ['a', 'a_1', 'b', 'c'])
    #
    def test_excludelist(self):
        "Test excludelist"
        names = ['dates', 'data', 'Other Data', 'mask']
        validator = NameValidator(excludelist = ['dates', 'data', 'mask'])
        test = validator.validate(names)
        assert_equal(test, ['dates_', 'data_', 'Other_Data', 'mask_'])


#-------------------------------------------------------------------------------

class TestStringConverter(TestCase):
    "Test StringConverter"
    #
    def test_creation(self):
        "Test creation of a StringConverter"
        converter = StringConverter(int, -99999)
        assert_equal(converter._status, 1)
        assert_equal(converter.default, -99999)
    #
    def test_upgrade(self):
        "Tests the upgrade method."
        converter = StringConverter()
        assert_equal(converter._status, 0)
        converter.upgrade('0')
        assert_equal(converter._status, 1)
        converter.upgrade('0.')
        assert_equal(converter._status, 2)
        converter.upgrade('0j')
        assert_equal(converter._status, 3)
        converter.upgrade('a')
        assert_equal(converter._status, len(converter._mapper)-1)
    #
    def test_missing(self):
        "Tests the use of missing values."
        converter = StringConverter(missing_values=('missing','missed'))
        converter.upgrade('0')
        assert_equal(converter('0'), 0)
        assert_equal(converter(''), converter.default)
        assert_equal(converter('missing'), converter.default)
        assert_equal(converter('missed'), converter.default)
        try:
            converter('miss')
        except ValueError:
            pass
    #
    def test_upgrademapper(self):
        "Tests updatemapper"
        from datetime import date
        import time
        dateparser = lambda s : date(*time.strptime(s, "%Y-%m-%d")[:3])
        StringConverter.upgrade_mapper(dateparser, date(2000,1,1))
        convert = StringConverter(dateparser, date(2000, 1, 1))
        test = convert('2001-01-01')
        assert_equal(test, date(2001, 01, 01))
        test = convert('2009-01-01')
        assert_equal(test, date(2009, 01, 01))
        test = convert('')
        assert_equal(test, date(2000, 01, 01))


#-------------------------------------------------------------------------------

class TestMiscFunctions(TestCase):
    #
    def test_has_nested_dtype(self):
        "Test has_nested_dtype"
        ndtype = np.dtype(np.float)
        assert_equal(has_nested_fields(ndtype), False)
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        assert_equal(has_nested_fields(ndtype), False)
        ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        assert_equal(has_nested_fields(ndtype), True)

