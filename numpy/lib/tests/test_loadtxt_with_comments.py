"""
Tests specific to `np.loadtxt_w_comm`
"""

import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO

import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY


def test_scientific_notation_wc():
    """Test that both 'e' and 'E' are parsed correctly."""
    data = StringIO(

            "# Header\n"
            "1.0e-1,2.0E1,3.0\n"
            "4.0e-2,5.0E-1,6.0\n"
            "7.0e-3,8.0E1,9.0\n"
            "0.0e-4,1.0E-1,2.0"

    )
    expected_array = np.array(
        [[0.1, 20., 3.0], [0.04, 0.5, 6], [0.007, 80., 9], [0, 0.1, 2]]
    )
    expected_comments = np.array(["Header"])
    a, c = np.loadtxt_w_comm(data, delimiter=",")
    
    assert_array_equal(a, expected_array)
    assert_array_equal(c, expected_comments)


@pytest.mark.parametrize("comment", ["..", "//", "@-", "this is a comment:"])
def test_comment_multiple_chars_wc(comment):
    content = "# IGNORE\n1.5, 2.5# ABC\n3.0,4.0# XXX\n5.5,6.0\n"
    txt = StringIO(content.replace("#", comment))
    a, c = np.loadtxt_w_comm(txt, delimiter=",", comments=comment)
    
    assert_equal(a, [[1.5, 2.5], [3.0, 4.0], [5.5, 6.0]])
    assert_equal(c, np.array([" IGNORE", " ABC", " XXX"]))


@pytest.fixture
def mixed_types_structured_wc():
    """
    Fixture providing heterogeneous input data with a structured dtype, along
    with the associated structured array.
    """
    data = StringIO(

            "# Introduction\n"
            "1000;2.4;alpha;-34\n"
            "2000;3.1;beta;29\n"
            "# Body Paragraph\n"
            "3500;9.9;gamma;120\n"
            "4090;8.1;delta;0\n"
            "# Conclusion\n"
            "5001;4.4;epsilon;-99\n"
            "6543;7.8;omega;-1\n"

    )
    dtype = np.dtype(
        [('f0', np.uint16), ('f1', np.float64), ('f2', 'S7'), ('f3', np.int8)]
    )
    expected_arr = np.array(
        [
            (1000, 2.4, "alpha", -34),
            (2000, 3.1, "beta", 29),
            (3500, 9.9, "gamma", 120),
            (4090, 8.1, "delta", 0),
            (5001, 4.4, "epsilon", -99),
            (6543, 7.8, "omega", -1)
        ],
        dtype=dtype
    )
    expectec_comm = np.array([" Introduction", " Body Paragraph", " Conclusion"])
    return data, dtype, expected_arr, expectec_comm


@pytest.mark.parametrize('skiprows', [0, 1, 2, 3])
def test_structured_dtype_and_skiprows_no_empty_lines_wc(
        skiprows, mixed_types_structured_wc):
    data, dtype, expected_arr, expected_comm = mixed_types_structured_wc
    a, c = np.loadtxt_w_comm(data, dtype=dtype, delimiter=";", skiprows=skiprows)
    
    assert_array_equal(a, expected_arr[skiprows:])
    assert_array_equal(c, expected_comm)


def test_unpack_structured_wc(mixed_types_structured_wc):
    data, dtype, expected_arr, expected_comm = mixed_types_structured_wc

    a, b, c, d, comm = np.loadtxt_w_comm(data, dtype=dtype, delimiter=";", unpack=True)
    assert_array_equal(a, expected_arr["f0"])
    assert_array_equal(b, expected_arr["f1"])
    assert_array_equal(c, expected_arr["f2"])
    assert_array_equal(d, expected_arr["f3"])
    assert_array_equal(comm, expected_comm)


def test_structured_dtype_with_shape_wc():
    dtype = np.dtype([("a", "u1", 2), ("b", "u1", 2)])
    data = StringIO("# Toothless\n0,1,2,3\n6,7,8,9\n")
    expected_arr = np.array([((0, 1), (2, 3)), ((6, 7), (8, 9))], dtype=dtype)
    expected_comm = np.array([ " Toothless"])
    a, c = np.loadtxt_w_comm(data, delimiter=",", dtype=dtype)
    
    assert_array_equal(a, expected_arr)
    assert_array_equal(c, expected_comm)

def test_structured_dtype_with_multi_shape_wc():
    dtype = np.dtype([("a", "u1", (2, 2))])
    data = StringIO("0 1 2 3\n# Hiccup\n")
    expected_arr = np.array([(((0, 1), (2, 3)),)], dtype=dtype)
    expected_comm = np.array([" Hiccup"])
    a, c = np.loadtxt_w_comm(data, dtype=dtype)
    
    assert_array_equal(a, expected_arr)
    assert_array_equal(c, expected_comm)

def test_no_comments_wc():
    data = StringIO(
        "1\n"
        "2\n"
        "3\n"
        )
    expected_arr = np.array([1., 2., 3.])
    expected_comm = np.array([])

    a, c = np.load_txt_with_comments(data)
    
    assert_array_equal(a, expected_arr)
    assert_array_equal(c, expected_comm)

def test_multiple_comment_chars_wc():
    data = StringIO(
        "# Snotlout\n"
        "// Fishlegs\n"
        "2\n"
        "Foo Ruffnut and Tuffnut\n"
        "Bar Astrid\n"
    )
    expected_arr = np.array([2.])
    expected_comm = np.array([" Snotlout", " Fishlegs", " Ruffnut and Tuffnut", " Astrid"])
    a, c = np.load_txt_with_comments(data, comments=["#", "Bar", "Foo", "//"])
    
    assert_array_equal(a, expected_arr)
    assert_array_equal(c, expected_comm)