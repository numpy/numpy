from numpy.testing import *
import numpy as np
import os, re
from os import path
from nose.plugins.attrib import attr
import datetime
from StringIO import StringIO

def check_mats(loaded, ref):
    """
    Checks the array data and the dtypes. 
    """
    assert_equal(loaded, ref)
    assert_equal(loaded.dtype, ref.dtype)

def test_int():
    s = "1,1,1"
    f = StringIO(s)
    
    loaded = np.loadtable(f, delimiter=',')
    ref = np.array([(1L, 1L, 1L)], 
            dtype=[('f0', '<i8'), ('f1', '<i8'), ('f2', '<i8')])
    check_mats(loaded, ref)
    
    f.seek(0)
    loaded = np.loadtable(f, force_mask=True, delimiter=',')
    ref = np.ma.MaskedArray([(1L, 1L, 1L)], 
            dtype=[('f0', '<i8'), ('f1', '<i8'), ('f2', '<i8')])
    check_mats(loaded, ref)

def test_float_string_bool():
    s = '\n'.join(['3, hello, True', 
                   '1.5, 2.4, False'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',')
    ref = np.array([(3, "hello", True), (1.5, "2.4", False)],
            dtype=[("f0", "f8"), ("f1", "S5"), ("f2", "?")])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, force_mask=True, delimiter=',')
    ref = np.ma.MaskedArray([(3.0, 'hello', True), (1.5, '2.4', False)], 
            dtype=[('f0', '<f8'), ('f1', '|S5'), ('f2', '|b1')])
    check_mats(loaded, ref)

def test_comment_string():
    s = '\n'.join(['#comment ', 
                   '   #comment',
                   '        #comment ',
                   '\t \t',
                   '1, 1.2, fad',
                   '2, 2.4, grr',
                   '  \t  ',
                   '3, 7.4, 5g4'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',')
    ref = np.array([(1L, 1.2, 'fad'), 
                    (2L, 2.3999999999999999, 'grr'),
                    (3L, 7.4000000000000004, '5g4')], 
            dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '|S3')])
    check_mats

    f.seek(0)
    loaded = np.loadtable(f, force_mask=True, delimiter=',')
    ref = np.ma.MaskedArray([(1L, 1.2, 'fad'), 
                             (2L, 2.3999999999999999, 'grr'),
                             (3L, 7.4000000000000004, '5g4')], 
            dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '|S3')])
    check_mats(loaded, ref)

def test_header():
    s = '\n'.join(['column1, column2, column3, column4',
                   '1, 2,3,4',
                   '5,6,7,8'])
    f = StringIO(s)

    loaded = np.loadtable(f, header=True, delimiter=',')
    ref = np.array([(1L, 2L, 3L, 4L), 
                    (5L, 6L, 7L, 8L)], 
            dtype=[('column1', '<i8'), ('column2', '<i8'), 
                   ('column3', '<i8'), ('column4', '<i8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, header=True, force_mask=True, delimiter=',')
    ref = np.ma.MaskedArray([(1L, 2L, 3L, 4L), 
                             (5L, 6L, 7L, 8L)], 
            dtype=[('column1', '<i8'), ('column2', '<i8'), 
                   ('column3', '<i8'), ('column4', '<i8')]) 
    check_mats(loaded, ref)

def test_unnamed_columns():
    s = '\n'.join(['field1, field2, ,',
                   '1, 2, 3, 4 ',
                   '5, 6, 7,8'])
    f = StringIO(s)

    loaded = np.loadtable(f, header=True, delimiter=',')
    ref = np.array([(1L, 2L, 3L, 4L), 
                    (5L, 6L, 7L, 8L)], 
            dtype=[('field1', '<i8'), ('field2', '<i8'), 
                   ('f2', '<i8'), ('f3', '<i8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, header=True, force_mask=True, delimiter=',')
    ref = np.ma.MaskedArray([(1L, 2L, 3L, 4L), 
                             (5L, 6L, 7L, 8L)], 
            dtype=[('field1', '<i8'), ('field2', '<i8'), 
                   ('f2', '<i8'), ('f3', '<i8')])
    check_mats(loaded, ref)

def test_exponential_notation():
    s = '\n'.join(['1.2e5, -15e-3, 4E-3, 5.E3',
                   '2.4e-13, 41e24, -1E-1, 12e4'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',')
    ref = np.array([(120000.0, -0.014999999999999999, 
                            0.0040000000000000001, 5000.0),
                    (2.3999999999999999e-13, 4.0999999999999997e+25, 
                            -0.10000000000000001, 120000.0)], 
            dtype=[('f0', '<f8'), ('f1', '<f8'), 
                   ('f2', '<f8'), ('f3', '<f8')]) 
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, force_mask=True, delimiter=',')
    ref = np.ma.array([(120000.0, -0.014999999999999999, 
                            0.0040000000000000001, 5000.0),
                       (2.3999999999999999e-13, 4.0999999999999997e+25, 
                            -0.10000000000000001, 120000.0)], 
            dtype=[('f0', '<f8'), ('f1', '<f8'), 
                   ('f2', '<f8'), ('f3', '<f8')])
    check_mats(loaded, ref)

def test_semicolon_delimited():
    s = '\n'.join(['1; abc ;True',
                   '4;def;False',
                   '8 ; ghi ; True'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=';')
    ref = np.array([(1L, 'abc', True), 
                    (4L, 'def', False), 
                    (8L, 'ghi', True)], 
            dtype=[('f0', '<i8'), ('f1', '|S3'), ('f2', '|b1')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, delimiter=';', force_mask=True)
    ref = np.ma.MaskedArray([(1L, 'abc', True), 
                             (4L, 'def', False), 
                             (8L, 'ghi', True)], 
            dtype=[('f0', '<i8'), ('f1', '|S3'), ('f2', '|b1')])
    check_mats(loaded, ref)

def test_space_delimited(): 
    s = '\n'.join(['1 2 a false',
                   '3 4 b true',
                   '5 6 c false'])
    f = StringIO(s)

    loaded = np.loadtable(f)
    ref = np.array([(1L, 2L, 'a', False),
                    (3L, 4L, 'b', True),
                    (5L, 6L, 'c', False)],
            dtype=[('f0','i8'), ('f1','i8'), ('f2','|S1'), ('f3','|b1')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, force_mask=True)
    ref = np.ma.MaskedArray([(1L, 2L, 'a', False),
                              (3L, 4L, 'b', True),
                              (5L, 6L, 'c', False)],
            dtype=[('f0','i8'), ('f1','i8'), ('f2','|S1'), ('f3','|b1')])
    check_mats(loaded, ref)


def test_type_search_order():
    s = '\n'.join(['12, 3.2, True',
                   '4, -3.4, False'])
    f = StringIO(s)

    loaded = np.loadtable(f, type_search_order=['i4', 'f4'], delimiter=',')
    ref = np.array([(12, 3.2000000476837158, 'True'), 
                    (4, -3.4000000953674316, 'False')], 
            dtype=[('f0', '<i4'), ('f1', '<f4'), ('f2', '|S5')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, type_search_order=['i4','f4'], 
                            force_mask=True, delimiter=',')
    ref = np.ma.MaskedArray([(12, 3.2000000476837158, 'True'), 
                             (4, -3.4000000953674316, 'False')], 
            dtype=[('f0', '<i4'), ('f1', '<f4'), ('f2', '|S5')])
    check_mats(loaded, ref)

def test_complex():
    s = '\n'.join(['1, 1+2j, 5.2, abc',
                   '2, 5.2, 1+3j, 1+5j'])
    f = StringIO(s)

    loaded = np.loadtable(f, type_search_order=['b1','i4','f4','c8'],
                            delimiter=',')
 
    ref = np.array([(1, (1+2j), (5.1999998092651367+0j), 'abc'),
                    (2, (5.1999998092651367+0j), (1+3j), '1+5j')], 
            dtype=[('f0', '<i4'), ('f1', '<c8'), 
                   ('f2', '<c8'), ('f3', '|S4')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, type_search_order=['b1','i4','f4','c8'],
                            delimiter=',', force_mask=True)
    ref = np.ma.MaskedArray([(1, (1+2j), (5.1999998092651367+0j), 'abc'),
                             (2, (5.1999998092651367+0j), (1+3j), '1+5j')], 
            dtype=[('f0', '<i4'), ('f1', '<c8'), 
                   ('f2', '<c8'), ('f3', '|S4')])
    check_mats(loaded, ref)

def test_num_lines_search():
    s = '\n'.join(['a,b,c',
                   'aa,bb,cc',
                   'aaa,bbb,ccc'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',', num_lines_search=2)
    ref = np.array([('a','b','c'),
                    ('aa','bb','cc'),
                    ('aa','bb','cc')],
            dtype=[('f0','|S2'), ('f1', '|S2'), ('f2', '|S2')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, delimiter=',', num_lines_search=2,
                            force_mask=True)
    ref = np.ma.MaskedArray([('a','b','c'),
                             ('aa','bb','cc'),
                             ('aa','bb','cc')],
            dtype=[('f0','|S2'), ('f1', '|S2'), ('f2', '|S2')])
    check_mats(loaded, ref)

def test_string_sizes_size():
    s = '\n'.join(['abc, def, ghi',
                  'a, b,  c '])
    f = StringIO(s)

    loaded = np.loadtable(f, string_sizes=20, delimiter=',')
    ref = np.array([('abc', 'def', 'ghi'), ('a', 'b', 'c')], 
            dtype=[('f0', '|S20'), ('f1', '|S20'), ('f2', '|S20')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, string_sizes=20, force_mask=True, 
                            delimiter=',')
    ref = np.ma.MaskedArray([('abc', 'def', 'ghi'), ('a', 'b', 'c')], 
            dtype=[('f0', '|S20'), ('f1', '|S20'), ('f2', '|S20')])
    check_mats(loaded, ref)
      
    def wrong_sizes():
        np.loadtable(f, string_sizes = [3,3], delimiter=',')
    f.seek(0)
    assert_raises(ValueError, wrong_sizes)

    f.seek(0)
    loaded = np.loadtable(f, string_sizes=[1,10,20], delimiter=',')
    ref = np.array([('abc', 'def', 'ghi'), ('a', 'b', 'c')], 
            dtype=[('f0', '|S3'), ('f1', '|S10'), ('f2', '|S20')])
    check_mats(loaded, ref)

def test_check_sizes():
    s = '\n'.join(['a, aa, aaa',
                   'ab, aabb, aaabbb',
                   'abcc, aabbcccc, aaabbbcccccc'])
    f = StringIO(s)

    loaded = np.loadtable(f, check_sizes=False, delimiter=',')
    ref = np.array([('a','a','a'), ('a','a','a'),('a','a','a')],
            dtype=[('f0','S1'), ('f1','S1'), ('f2','S1')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, check_sizes=1, delimiter=',')
    ref = np.array([('a','aa','aaa'), ('a','aa','aaa'),('a','aa','aaa')],
            dtype=[('f0','S1'), ('f1','S2'), ('f2','S3')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, check_sizes=2, delimiter=',')
    ref = np.array([('a','aa','aaa'), ('ab','aabb','aaabbb'),
                    ('ab','aabb','aaabbb')],
            dtype=[('f0','S2'), ('f1','S4'), ('f2','S6')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, check_sizes=1, string_sizes=[7,1,1],
                            delimiter=',')
    ref = np.array([('a','aa','aaa'), ('ab','aa','aaa'),
                    ('abcc','aa','aaa')],
            dtype=[('f0','S7'), ('f1','S2'), ('f2','S3')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, check_sizes=False, string_sizes=[2, 4, 1],
                            delimiter=',')
    ref = np.array([('a', 'aa', 'a'),
                    ('ab', 'aabb', 'a'),
                    ('ab', 'aabb', 'a')],
            dtype=[('f0','|S2'), ('f1', '|S4'), ('f2', '|S1')])
    check_mats(loaded, ref)


def test_rowname():
    s = '\n'.join(['col1 col2 col3',
                   'meat 1 2 3',
                   'potatoes 4 5 6'])
    f = StringIO(s)

    loaded = np.loadtable(f, header=True)
    ref = np.array([('meat', 1L, 2L, 3L), 
                    ('potatoes', 4L, 5L, 6L)], 
            dtype=[('row_names', '|S8'), ('col1', '<i8'), 
                   ('col2', '<i8'), ('col3', '<i8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, header=True, force_mask=True)
    ref = np.ma.MaskedArray([('meat', 1L, 2L, 3L), 
                             ('potatoes', 4L, 5L, 6L)], 
            dtype=[('row_names', '|S8'), ('col1', '<i8'), 
                   ('col2', '<i8'), ('col3', '<i8')])
    check_mats(loaded, ref)

def test_rowname2():
    s = '\n'.join(['col1 col2',
                   '\t \t ',
                   'a b c d',
                   'e f g h'])
    f = StringIO(s)

    def mismatched_header():
        np.loadtable(f, header=True)
    assert_raises(ValueError, mismatched_header)

def test_comment_re():
    s = '\n'.join(['#comment ',
                   "2444 comment",
                   "'I'm a comment' blah blah",
                   "1 2 3", 
                   "244 still a comment",
                   "4 5 6",
                   "24 a final comment"])
    f = StringIO(s)

    loaded = np.loadtable(f, comments='#|24+|\'I\'m a comment\'')
    ref = np.array([(1L, 2L, 3L), (4L, 5L, 6L)], 
            dtype=[('f0', '<i8'), ('f1', '<i8'), ('f2', '<i8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, comments='#|24+|\'I\'m a comment\'',
                            force_mask=True)
    ref = np.ma.MaskedArray([(1L, 2L, 3L), (4L, 5L, 6L)], 
            dtype=[('f0', '<i8'), ('f1', '<i8'), ('f2', '<i8')])
    check_mats(loaded, ref)

def test_float():
    s = '\n'.join(['Inf, 2, INF, nan',
                   '-inf, 2 ,-INF ,inf',
                   '3.5 ,NAN ,NaN ,12.5',
                   '12, 3.76E22 ,3.1415, infn'])
    f = StringIO(s)
    
    loaded = np.loadtable(f, delimiter=',')
    ref = np.array([(np.inf, 2.0, np.inf, 'nan'), 
                    (-np.inf, 2.0, -np.inf, 'inf'),
                    (3.5, np.nan, np.nan, '12.5'),
                    (12.0, 3.76e+22, 3.1415000000000002, 'infn')], 
            dtype=[('f0', '<f8'), ('f1', '<f8'), 
                   ('f2', '<f8'), ('f3', '|S4')])
    assert_equal(loaded.dtype, ref.dtype)
    assert_equal(loaded[0],ref[0])
    assert_equal(loaded[1], ref[1])
    assert_equal(loaded[2][0], ref[2][0])
    assert_equal(loaded[2][3], ref[2][3])
    assert np.isnan(loaded[2][1])
    assert np.isnan(loaded[2][2])
    assert_equal(loaded[3], ref[3])

def test_no_entry():
    s = '\n'.join(["A, AA, AAA,",
                   ", A, AAAA, B",
                   "D, AA, B, C"])
    f = StringIO(s)

    loaded = np.loadtable(f, NA_re=None, header=False, delimiter=',')
    ref = np.array([('A', 'AA', 'AAA', ''),
                    ('', 'A', 'AAAA', 'B'),
                    ('D', 'AA', 'B', 'C')],
        dtype='S1,S2,S4,S1') 
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, header=False, delimiter=',')
    ref = np.ma.MaskedArray([('A', 'AA', 'AAA', 'N/A'), 
                             ('N/A', 'A', 'AAAA', 'B'), 
                             ('D', 'AA', 'B', 'C')], 
                        mask = [(False, False, False, True), 
                                (True, False, False, False), 
                                (False, False, False, False)],  
                       dtype=[('f0', '|S1'), ('f1', '|S2'), 
                              ('f2', '|S4'), ('f3', '|S1')]) 
    check_mats(loaded, ref)

def test_na_column():
    s = '\n'.join(["NA, , 1",
                   ", NA, 2 ",
                   "na, , 3"])
    f = StringIO(s)

    loaded = np.loadtable(f, NA_re=None, delimiter=',')
    ref = np.array([('NA', '', 1L), ('', 'NA', 2L), ('na', '', 3L)], 
            dtype=[('f0', 'S2'), ('f1', 'S2'), ('f2', '<i8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, delimiter=',', NA_re=r'NA|na|')
    ref = np.ma.array([('N', 'N', 1L), 
                       ('N', 'N', 2L), 
                       ('N', 'N', 3L)],
                  mask=[(True, True, False),
                        (True, True,False),
                        (True,True,False)],
                  dtype=[('f0', 'S1'), ('f1', 'S1'), ('f2', '<i8')])
    check_mats(loaded, ref)

def test_na_column_string():
    s = '\n'.join(['NA, 1',
                   ', 2',
                   'na, 3'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',')
    ref = np.ma.MaskedArray([('N/', 1L), ('N/', 2L), ('na', 3L)],
                mask=[(True,False), (True,False), (False, False)],
                dtype=[('f0', 'S2'), ('f1', '<i8')])
    check_mats(loaded, ref)

def test_na():
    s = '\n'.join(["NA, 3.4, 3e2, blah, sing, a",
                   "5, NA, 24.6, NA, zoomzoom,",
                   "NA, 42, , posies, , a",
                   ", , NA, little, lamb, ab"])
    f = StringIO(s)

    loaded = np.loadtable(f, NA_re=None, delimiter=',')
    ref = np.array([('NA', '3.4', '3e2', 'blah', 'sing', 'a'),
                    ('5', 'NA', '24.6', 'NA', 'zoomzoom', ''),
                    ('NA', '42', '', 'posies', '', 'a'),
                    ('', '', 'NA', 'little', 'lamb', 'ab')], 
            dtype=[('f0', 'S2'), ('f1', 'S3'), 
                   ('f2', 'S4'), ('f3', 'S6'), 
                   ('f4', 'S8'), ('f5', 'S2')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, delimiter=',')
    ref = np.ma.MaskedArray([(999999, 3.4, 300.0, 'blah', 'sing', 'a'), 
                             (5, 1e+20, 24.6, 'N/A', 'zoomzoom', 'N/'),
                             (999999, 42.0, 1e+20, 'posies', 'N/A', 'a'), 
                             (999999, 1e+20, 1e+20, 'little', 'lamb', 'ab')], 
				    mask = [(True, False, False, False, False, False),
				            (False, True, False, True, False, True),
							(True, False, True, False, True, False),
				            (True, True, True, False, False, False)],
			        dtype=[('f0', '<i8'), ('f1', '<f8'), 
                           ('f2', '<f8'), ('f3', 'S6'), 
                           ('f4', 'S8'), ('f5', 'S2')])
    check_mats(loaded, ref)

def test_not_na():
    s = '\n'.join(["A, B, NA",
                   "NAC, NAB, NAC",
                   "D, NA, NA ",
                   "NAD, NAE, NAF"])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',')
    ref = np.ma.MaskedArray([('A', 'B', 'N/A'), 
                             ('NAC', 'NAB', 'NAC'),
                             ('D', 'N/A', 'N/A'), 
                             ('NAD', 'NAE', 'NAF')],
                     mask = [(False, False, True), 
                             (False, False, False),
                             (False, True, True), 
                             (False, False, False)],
                     dtype=[('f0', 'S3'), ('f1', 'S3'), ('f2', 'S3')])
    check_mats(loaded, ref)

def test_bool_str_size():
    s = '\n'.join(["True false True False",
                   "X Y false TRUE"])
    f = StringIO(s)

    loaded = np.loadtable(f)
    ref = np.array([('True', 'false', True, False),
                    ('X', 'Y', False, True)],
            dtype = 'S4, S5, b1, b1')
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, force_mask=True)
    ref = np.ma.MaskedArray([('True', 'false', True, False), 
                             ('X', 'Y', False, True)], 
            dtype=[('f0', '|S4'), ('f1', '|S5'), 
                   ('f2', '|b1'), ('f3', '|b1')])
    check_mats(loaded, ref) 

def test_skip_lines():
    s = '\n'.join(["Shouldn't pay any attention to this. ",
                   "Or to this.",
                   "Not this either. ",
                   "field1, field2, field3",
                   "1, 2, 3",
                   "4, 5, 6"])
    f = StringIO(s)

    loaded = np.loadtable(f, header=True, skip_lines=3, delimiter=',')
    ref = np.array([(1L, 2L, 3L), (4L, 5L, 6L)], 
            dtype=[('field1', '<i8'), ('field2', '<i8'), ('field3', '<i8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, header=True, skip_lines=3, force_mask=True,
                                delimiter=',')
    ref = np.ma.MaskedArray([(1L, 2L, 3L), (4L, 5L, 6L)], 
            dtype=[('field1', '<i8'), ('field2', '<i8'), 
                   ('field3', '<i8')])
    check_mats(loaded, ref)

def test_date_basic():
    s = '\n'.join(["date1, date2",
                   "1991-11-03, 1887-01-01",
                   "1991-02-17, 1021-12-25",
                   "2001-03-10, 3013-06-06"])
    f = StringIO(s)

    loaded = np.loadtable(f, header=True, delimiter=',')
    ref = np.array([(datetime.date(1991, 11, 3), 
                        datetime.date(1887, 1, 1)),
                    (datetime.date(1991, 2, 17), 
                        datetime.date(1021, 12, 25)),
                    (datetime.date(2001, 3, 10), 
                        datetime.date(3013, 6, 6))], 
            dtype=[('date1', '<M8[D]'), ('date2', '<M8[D]')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, header=True, force_mask=True,
                            delimiter=',')
    ref = np.ma.MaskedArray([(datetime.date(1991, 11, 3), 
                                datetime.date(1887, 1, 1)),
                             (datetime.date(1991, 2, 17), 
                                datetime.date(1021, 12, 25)),
                             (datetime.date(2001, 3, 10), 
                                datetime.date(3013, 6, 6))], 
            dtype=[('date1', '<M8[D]'), ('date2', '<M8[D]')]) 
    check_mats(loaded, ref)

def test_date_na():
    s = '\n'.join(["date1, date2",
                   "1991-11-03, NA",
                   "NA, 1021-12-25",
                   "2001-03-10, 3013-06-06"])
    f = StringIO(s)
    
    loaded = np.loadtable(f, header=True, delimiter=',')
    ref = np.ma.MaskedArray([(datetime.date(1991,11,3), 
                                np.datetime64('NaT') ), 
                             (np.datetime64('NaT'), 
                                datetime.date(1021,12,25)),
                             (datetime.date(2001, 3, 10), 
                                datetime.date(3013, 6, 6))],
                     mask = [(False, True), 
                             (True, False), 
                             (False, False)],
                     dtype=[('date1', '<M8[D]'), ('date2', '<M8[D]')])
    check_mats(loaded, ref)

def test_date_non_iso():
    s = '\n'.join(["date1, date2, date3",
                   "1991-1-03, 87-125, 1935/04/01",
                   "1991-02-17, 21-346, 0174/05/02",
                   "2001-03-10, 13-007, 2344/02/26"])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',', header=True,
                            date_re=r'\d{4}/\d{2}/\d{2}',
                            date_strp='%Y/%m/%d')
    ref = np.array([('1991-1-03', '87-125', datetime.date(1935, 4, 1)),
                    ('1991-02-17', '21-346', datetime.date(174, 5, 2)),
                    ('2001-03-10', '13-007', datetime.date(2344, 2, 26))], 
            dtype=[('date1', 'S10'), ('date2', 'S6'), 
                        ('date3', '<M8[D]')])
    check_mats(loaded, ref)
   
    f.seek(0)
    loaded = np.loadtable(f, delimiter=',', header=True,
                            date_re=r'\d{4}/\d{2}/\d{2}',
                            date_strp='%Y/%m/%d',
                            force_mask=True)
    ref = np.ma.MaskedArray([('1991-1-03', '87-125', 
                                datetime.date(1935, 4, 1)),
                             ('1991-02-17', '21-346', 
                                datetime.date(174, 5, 2)),
                             ('2001-03-10', '13-007', 
                                datetime.date(2344, 2, 26))], 
            dtype=[('date1', 'S10'), ('date2', 'S6'), 
                        ('date3', '<M8[D]')])
    check_mats(loaded, ref)

def test_date_non_iso2():
    s = '\n'.join(["date1, date2, date3",
                   "1991-1-03, 87-125, 1935/04/01",
                   "1991-02-17, 21-346, 0174/05/02",
                   "2001-03-10, 13-007, 2344/02/26"])
    f = StringIO(s)

    loaded = np.loadtable(f, header=True, delimiter=',',
                            date_re=r'\d{2}-\d{3}',
                            date_strp='%y-%j')
    ref = np.array([('1991-1-03', datetime.date(1987, 5, 5), 
                        '1935/04/01'),
                    ('1991-02-17', datetime.date(2021, 12, 12), 
                        '0174/05/02'),
                    ('2001-03-10', datetime.date(2013, 1, 7), 
                        '2344/02/26')], 
            dtype=[('date1', 'S10'), ('date2', '<M8[D]'), 
                        ('date3', 'S10')])
    check_mats(loaded, ref)
           
    f.seek(0)
    loaded = np.loadtable(f, header=True, delimiter=',',
                            date_re=r'\d{2}-\d{3}',
                            date_strp='%y-%j',
                            force_mask=True)
    ref = np.ma.MaskedArray([('1991-1-03', datetime.date(1987, 5, 5), 
                                '1935/04/01'),
                             ('1991-02-17', datetime.date(2021, 12, 12), 
                                '0174/05/02'),
                             ('2001-03-10', datetime.date(2013, 1, 7), 
                                '2344/02/26')], 
            dtype=[('date1', 'S10'), ('date2', '<M8[D]'), 
                        ('date3', 'S10')])
    check_mats(loaded, ref)

def test_date_fail():
    def bad_month():
        s = '\n'.join(['date1, date2',
                       '1991-13-03, 1887-01-01'
                       '1991-02-17, 1021-12-25'
                       '2001-03-10, 3013-06-06'])
        f = StringIO(s)    
        np.loadtable(f, delimiter=',', header=True)
    def bad_day():
        s = '\n'.join(["date1, date2",
                       "1991-11-42, 1887-01-01",
                       "1991-02-17, 1021-12-25",
                       "2001-03-10, 3013-06-06"])
        f = StringIO(s)
        np.loadtable(f, delimiter=',', header=True)
    def bad_combo():
        s = '\n'.join(["date1, date2, date3",
                       "1991-1-03, 87-125, 1935/04/01",
                       "1991-02-17, 21-346, 0174/05/02",
                       "2001-03-10, 13-007, 2344/02/26"])
        f = StringIO(s)
        np.loadtable(f, delimiter=',', header=True,
                        date_re=r"\d{4}/\d{2}/\d{2}",
                        date_strp='%Y-%m-%d')

    assert_raises(ValueError, bad_month)
    assert_raises(ValueError, bad_day)
    assert_raises(ValueError, bad_combo)

def test_quoted():
    s = '\n'.join(['bool, int, float, string, complex, datetime',
                   '"True", "6", "3", "2", "6", "1994-04-01"',
                   '"FALSE", "-6", "5.24", "BAH!", "2j", "1992-04-29"',
                   '"false", "-400", "4.e4", "Yellow Helicopter", "5.2+7.3J", "3003-04-01"',
                   '"true", "323", "-3e-2", "Righteous Rabbit", "-3.4-4.2e2j", "1676-05-07"'])
    f = StringIO(s)

    loaded = np.loadtable(f, header=True, delimiter=',',
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True)
    ref = np.array([(True, 6L, 3.0, '2', (6+0j), datetime.date(1994, 4, 1)),
                    (False, -6L, 5.2400000000000002, 'BAH!', 2j, 
                        datetime.date(1992, 4, 29)),
                    (False, -400L, 40000.0, 'Yellow Helicopter', 
                        (5.1999998092651367+7.3000001907348633j), 
                        datetime.date(3003, 4, 1)),
                    (True, 323L, -0.029999999999999999, 'Righteous Rabbit', 
                        (-3.4000000953674316-420j), 
                        datetime.date(1676, 5, 7))], 
            dtype=[('bool', '?'), ('int', '<i8'), 
                   ('float', '<f8'), ('string', 'S19'), 
                   ('complex', '<c8'), ('datetime', '<M8[D]')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, header=True, delimiter=',',
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True,
                        force_mask=True)
    ref = np.ma.MaskedArray([(True, 6L, 3.0, '2', (6+0j), 
                                 datetime.date(1994, 4, 1)),
                             (False, -6L, 5.2400000000000002, 'BAH!', 
                                 2j, datetime.date(1992, 4, 29)),
                             (False, -400L, 40000.0, 'Yellow Helicopter', 
                                 (5.1999998092651367+7.3000001907348633j), 
                                 datetime.date(3003, 4, 1)),
                             (True, 323L, -0.029999999999999999, 
                                 'Righteous Rabbit', 
                                 (-3.4000000953674316-420j), 
                                 datetime.date(1676, 5, 7))], 
            dtype=[('bool', '?'), ('int', '<i8'), 
                   ('float', '<f8'), ('string', 'S19'), 
                   ('complex', '<c8'), ('datetime', '<M8[D]')])
    check_mats(loaded, ref)

def test_quoted_mixed():
    s = '\n'.join(['bool, int, float, string, complex, datetime',
                   'True, 6, 3, 2, 6, 1994-04-01',
                   '"FALSE", "-6", "5.24", "BAH!", "2j", "1992-04-29"',
                   'false, -400, 4.e4, Yellow Helicopter, 5.2+7.3J, 3003-04-01',
                   '"true", "323", "-3e-2", "Righteous Rabbit", "-3.4-4.2e2j", "1676-05-07"'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',', header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True)
    ref = np.array([(True, 6L, 3.0, '2', (6+0j), datetime.date(1994, 4, 1)),
                    (False, -6L, 5.2400000000000002, 'BAH!', 2j, 
                        datetime.date(1992, 4, 29)),
                    (False, -400L, 40000.0, 'Yellow Helicopter', 
                        (5.1999998092651367+7.3000001907348633j), 
                        datetime.date(3003, 4, 1)),
                    (True, 323L, -0.029999999999999999, 'Righteous Rabbit', 
                        (-3.4000000953674316-420j), 
                        datetime.date(1676, 5, 7))], 
            dtype=[('bool', '?'), ('int', '<i8'), 
                   ('float', '<f8'), ('string', 'S18'), 
                   ('complex', '<c8'), ('datetime', '<M8[D]')])
    check_mats(loaded, ref) 

    f.seek(0)
    loaded = np.loadtable(f, delimiter=',', header=True, 
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True,
                        force_mask=True)
    ref = np.ma.MaskedArray([(True, 6L, 3.0, '2', (6+0j), 
                                 datetime.date(1994, 4, 1)),
                             (False, -6L, 5.2400000000000002, 'BAH!', 2j,
                                 datetime.date(1992, 4, 29)),
                             (False, -400L, 40000.0, 'Yellow Helicopter',
                                 (5.1999998092651367+7.3000001907348633j), 
                                 datetime.date(3003, 4, 1)),
                             (True, 323L, -0.029999999999999999, 
                                 'Righteous Rabbit', 
                                 (-3.4000000953674316-420j), 
                                 datetime.date(1676, 5, 7))], 
            dtype=[('bool', '?'), ('int', '<i8'), 
                   ('float', '<f8'), ('string', 'S18'), 
                   ('complex', '<c8'), ('datetime', '<M8[D]')])
    check_mats(loaded, ref)

def test_quoted_na():
    s = '\n'.join(['bool, int, float, string, complex, datetime',
                   '"True", "6", NA, "2", "6", NA',
                   'NA, "-6", "5.24", "BAH!", "2j", "1992-04-29"',
                   '"false", NA, "4.e4", "Yellow Helicopter", NA, "3003-04-01"',
                   '"true", "323", "-3e-2", NA, "-3.4-4.2e2j", "1676-05-07"'])
    f = StringIO(s)    

    loaded = np.loadtable(f, delimiter=',', header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True,
                        NA_re=None)
    ref = np.array([('True', '6', 'NA', '2', '6', 'NA'),
                    ('NA', '-6', '5.24', 'BAH!', '2j', '1992-04-29'),
                    ('false', 'NA', '4.e4', 'Yellow Helicopter', 'NA', 
                        '3003-04-01'),
                    ('true', '323', '-3e-2', 'NA', '-3.4-4.2e2j', 
                        '1676-05-07')], 
                dtype=[('bool', 'S7'), ('int', 'S5'), 
                       ('float', 'S7'), ('string', 'S19'), 
                       ('complex', 'S13'), ('datetime', 'S12')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, delimiter=',', header=True,
                        type_search_order=['b1', 'i8', 'f8', 'c8', 'M8[D]'],
                        quoted=True)
    ref = np.ma.MaskedArray(data=[(True, 6L, 1e+20, '2', (6+0j), None),
                                  (True, -6L, 5.2400000000000002, 'BAH!', 
                                      2j, datetime.date(1992, 4, 29)),
                                  (False, 999999L, 40000.0, 
                                      'Yellow Helicopter', 
                                      (1.0000000200408773e+20+0j), 
                                      datetime.date(3003, 4, 1)),
                                  (True, 323L, -0.029999999999999999, 
                                      'N/A', (-3.4000000953674316-420j), 
                                      datetime.date(1676, 5, 7))],
                          mask = [(False, False, True, False, False, True),
                                  (True, False, False, False, False, False),
                                  (False, True, False, False, True, False),
                                  (False, False, False, True, False, False)],
                        dtype=[('bool', '?'), ('int', '<i8'), 
                               ('float', '<f8'), ('string', 'S19'), 
                               ('complex', '<c8'), ('datetime', '<M8[D]')])
    check_mats(loaded, ref)

def test_comma_float():
    s = '\n'.join(['"4", "-7,64"',
                   '"2,42", "4,24"',
                   '"62,24", "73,24"',
                   '"1,63E3", "5.14e-5"'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',', quoted=True, 
                        comma_decimals=True)
    ref = np.array([(4.0, -7.6399999999999997),
                    (2.4199999999999999, 4.2400000000000002),
                    (62.240000000000002, 73.239999999999995),
                    (1630.0, 5.1400000000000003e-05)], 
            dtype=[('f0', '<f8'), ('f1', '<f8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, delimiter=',', quoted=True, 
                        comma_decimals=True, force_mask=True)
    ref = np.ma.MaskedArray([(4.0, -7.6399999999999997),
                             (2.4199999999999999, 4.2400000000000002),
                             (62.240000000000002, 73.239999999999995),
                             (1630.0, 5.1400000000000003e-05)], 
            dtype=[('f0', '<f8'), ('f1', '<f8')])
    check_mats(loaded, ref)

def test_comma_float_mixed():
    s = '\n'.join(['3.6, 6.8',
                   '"4", "-7,64"',
                   '"2,42", "4,24"',
                   '14.23, 5.2',
                   '"62,24", "73,24"',
                   '"1,63E3", "5.14e-5"'])
    f = StringIO(s)

    loaded = np.loadtable(f, delimiter=',', quoted=True, 
                            comma_decimals=True)
    ref = np.array([(3.6000000000000001, 6.7999999999999998),
                    (4.0, -7.6399999999999997),
                    (2.4199999999999999, 4.2400000000000002),
                    (14.23, 5.2000000000000002),
                    (62.240000000000002, 73.239999999999995),
                    (1630.0, 5.1400000000000003e-05)], 
            dtype=[('f0', '<f8'), ('f1', '<f8')])
    check_mats(loaded, ref)

    f.seek(0)
    loaded = np.loadtable(f, delimiter=',', quoted=True, force_mask=True,
                            comma_decimals=True)
    ref = np.ma.MaskedArray([(3.6000000000000001, 6.7999999999999998),
                             (4.0, -7.6399999999999997),
                             (2.4199999999999999, 4.2400000000000002),
                             (14.23, 5.2000000000000002),
                             (62.240000000000002, 73.239999999999995),
                             (1630.0, 5.1400000000000003e-05)], 
            dtype=[('f0', '<f8'), ('f1', '<f8')])
    check_mats(loaded, ref)

if __name__ == "__main__":
    run_module_suite()
