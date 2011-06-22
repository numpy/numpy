from numpy.testing import *
import numpy as np
import os, re
from os import path
from nose.plugins.attrib import attr
import datetime

def check_datafile(txtfile, pyfile, *args, **kwargs):
    """
    Called in most tests to verify loadtable's output matches
    the expected output, which is stored in a .py file.
    """
    data_dir = path.join(path.dirname(__file__), 'loadtable_data')

    loaded = np.loadtable(path.join(data_dir,txtfile), *args, **kwargs)
    ref = eval(open(path.join(data_dir,pyfile)).read())
    assert_equal(loaded, ref)
    assert_equal(loaded.dtype, ref.dtype)

def test_int():
    check_datafile("int.txt", "int.py")
    check_datafile("int.txt", "int_masked.py", force_mask=True)

def test_float_string_bool():
    check_datafile("float_string_bool.txt",
                    "float_string_bool.py")
    check_datafile("float_string_bool.txt",
                    "float_string_bool_masked.py", force_mask=True)

def test_comment_string():
    check_datafile("comment_string.txt",
                    "comment_string.py")
    check_datafile("comment_string.txt",
                    "comment_string_masked.py", force_mask=True)

def test_header():
    check_datafile("header.txt",
                    "header.py", header=True)
    check_datafile("header.txt",
                    "header_masked.py", force_mask=True, header=True)

def test_unnamed_columns():
    check_datafile("unnamed_columns.txt", "unnamed_columns.py",
                        header=True)
    check_datafile("unnamed_columns.txt", "unnamed_columns_masked.py",
                        force_mask=True, header=True)

def test_exponential_notation():
    check_datafile("exponential_notation.txt",
                    "exponential_notation.py")
    check_datafile("exponential_notation.txt",
                    "exponential_notation_masked.py",
                    force_mask=True)

def test_semicolon_delimited():
    check_datafile("semicolon_delimited.txt",
                    "semicolon_delimited.py", delimiter=';')
    check_datafile("semicolon_delimited.txt",
                    "semicolon_delimited_masked.py",
                        force_mask=True, delimiter=';')

def test_type_search_order():
    check_datafile("type_search_order.txt", "type_search_order.py",
                     type_search_order=['i4','f4'])
    check_datafile("type_search_order.txt", "type_search_order_masked.py",
                     force_mask=True,
                     type_search_order=['i4','f4'])

def test_complex():
    check_datafile("complex.txt", "complex.py",
                     type_search_order=['b1','i4','f4','c8'])
    check_datafile("complex.txt", "complex_masked.py",
                     force_mask=True,
                     type_search_order=['b1','i4','f4','c8'])

def test_num_lines_search():
    check_datafile("num_lines_search.txt", "num_lines_search.py",
                        num_lines_search=2)
    check_datafile("num_lines_search.txt", "num_lines_search_masked.py",
                        force_mask=True, num_lines_search=2)

def test_minstrsize():
    check_datafile("min_strsize.txt", "min_strsize.py",
                        min_string_size=20)
    check_datafile("min_strsize.txt", "min_strsize_masked.py",
                        force_mask=True, min_string_size=20)

def test_rowname():
    check_datafile("rowname.txt", "rowname.py",
                     delimiter=' ', header=True)
    check_datafile("rowname.txt", "rowname_masked.py",
                     force_mask=True,
                     delimiter=' ', header=True)

def test_rowname2():
    def mismatched_header():
        data_dir = path.join(path.dirname(__file__), 'loadtable_data')
        np.loadtable(path.join(data_dir,'rowname2.txt'),
                        header=True,
                        delimiter=' ')
    assert_raises(ValueError, mismatched_header)

def test_comment_re():
    check_datafile("comment_re.txt", "comment_re.py", delimiter=' ' ,
                        comments='#|24+|\'I\'m a comment\'')
    check_datafile("comment_re.txt", "comment_re_masked.py",
                        delimiter=' ' ,
                        force_mask=True,
                        comments='#|24+|\'I\'m a comment\'')

def test_float():
    data_dir = path.join(path.dirname(__file__), 'loadtable_data')
    pyfile = 'float.py'
    loaded = np.loadtable(path.join(data_dir,'float.txt'))
    ref = eval(open(path.join(data_dir,pyfile)).read())
    assert_equal(loaded.dtype, ref.dtype)
    assert_equal(loaded[0],ref[0])
    assert_equal(loaded[1], ref[1])
    assert_equal(loaded[2][0], ref[2][0])
    assert_equal(loaded[2][3], ref[2][3])
    assert np.isnan(loaded[2][1])
    assert np.isnan(loaded[2][2])
    assert_equal(loaded[3], ref[3])

def test_no_entry():
    check_datafile("no_entry.txt", "no_entry.py",
                        NA_re=None, header=False)
    check_datafile("no_entry.txt", "no_entry_masked.py",
                        force_mask=True, header=False)

def test_na():
    check_datafile("na.txt", "na.py", NA_re=None)
    check_datafile("na.txt", "na_masked.py")

def test_bool_str_size():
    check_datafile("bool_str_size.txt", "bool_str_size.py", delimiter=' ')
    check_datafile("bool_str_size.txt", "bool_str_size_masked.py",
                        force_mask=True, delimiter=' ')

def test_skip_lines():
    check_datafile("skip_lines.txt", "skip_lines.py",
                        header=True, skip_lines=3)
    check_datafile("skip_lines.txt", "skip_lines_masked.py",
                        force_mask=True,
                        header=True, skip_lines=3)

def test_date_basic():
    check_datafile("date.txt", "date.py",
                        header=True)
    check_datafile("date.txt", "date_masked.py",
                        header=True,
                        force_mask=True)

def test_date_na():
    check_datafile("date_na.txt", "date_na_masked.py",
                        header=True)

def test_date_non_iso():
    check_datafile("date_non_iso.txt", "date_non_iso1.py",
                    header=True,
                    date_re=r'\d{4}/\d{2}/\d{2}',
                    date_strp='%Y/%m/%d')
    check_datafile("date_non_iso.txt", "date_non_iso1_masked.py",
                    header=True,
                    date_re=r'\d{4}/\d{2}/\d{2}',
                    date_strp='%Y/%m/%d',
                    force_mask=True)

def test_date_non_iso2():
    check_datafile("date_non_iso.txt", "date_non_iso2.py",
                    header=True,
                    date_re=r'\d{2}-\d{3}',
                    date_strp='%y-%j')
    check_datafile("date_non_iso.txt", "date_non_iso2_masked.py",
                    header=True,
                    date_re=r'\d{2}-\d{3}',
                    date_strp='%y-%j',
                    force_mask=True)

def test_date_fail():
    data_dir = path.join(path.dirname(__file__), 'loadtable_data')
    def bad_month():
        np.loadtable(path.join(data_dir, "date_bad_month.txt"),
                        header=True)
    def bad_day():
        np.loadtable(path.join(data_dir, "date_bad_day.txt"),
                        header=True)
    def bad_combo():
        np.loadtable(path.join(data_dir, "date_non_iso.txt"),
                        header=True,
                        date_re=r"\d{4}/\d{2}/\d{2}",
                        date_strp='%Y-%m-%d')

    assert_raises(ValueError, bad_month)
    assert_raises(ValueError, bad_day)
    assert_raises(ValueError, bad_combo)

def test_quoted():
    check_datafile("quoted.txt", "quoted.py",
                        header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True)
    check_datafile("quoted.txt", "quoted_masked.py",
                        header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True,
                        force_mask=True)

def test_quoted_mixed():
    check_datafile("quoted_mixed.txt", "quoted_mixed.py",
                        header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True)
    check_datafile("quoted_mixed.txt", "quoted_mixed_masked.py",
                        header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True,
                        force_mask=True)

def test_quoted_na():
    check_datafile("quoted_na.txt", "quoted_na.py",
                        header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True,
                        NA_re=None)
    check_datafile("quoted_na.txt", "quoted_na_masked.py",
                        header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True)

def test_comma_float():
    check_datafile("comma_float.txt", "comma_float.py",
                        quoted=True,
                        comma_decimals=True)
    check_datafile("comma_float.txt", "comma_float_masked.py",
                        quoted=True,
                        comma_decimals=True,
                        force_mask=True)

def test_comma_float_mixed():
    check_datafile("comma_float_mixed.txt", "comma_float_mixed.py",
                        quoted=True,
                        comma_decimals=True)
    check_datafile("comma_float_mixed.txt", "comma_float_mixed_masked.py",
                        quoted=True,
                        comma_decimals=True,
                        force_mask=True)


def test_real_dataset1():
    check_datafile("real_dataset1.txt", "real_dataset1.py",
                        delimiter=' ',
                        header=True)
    check_datafile("real_dataset1.txt", "real_dataset1_masked.py",
                        delimiter=' ',
                        force_mask=True,
                        header=True)

def test_real_dataset2():
    check_datafile("real_dataset2.txt", "real_dataset2.py", header=True)
    check_datafile("real_dataset2.txt", "real_dataset2_masked.py",
                        force_mask=True, header=True)

if __name__ == "__main__":
    run_module_suite()
