from __future__ import division, absolute_import, print_function

import locale
from tempfile import NamedTemporaryFile
import unittest

import numpy as np
from numpy.testing import (
    run_module_suite, assert_, assert_equal, dec, assert_raises,
    assert_array_equal
)
from numpy.compat import sixu
from test_print import in_foreign_locale

longdouble_longer_than_double = (np.finfo(np.longdouble).eps
                                 < np.finfo(np.double).eps)


def test_scalar_extraction():
    """Confirm that extracting a value doesn't convert to python float"""
    o = 1 + np.finfo(np.longdouble).eps
    a = np.array([o, o, o])
    assert_equal(a[1], o)


# Conversions string -> long double


def test_repr_roundtrip():
    o = 1 + np.finfo(np.longdouble).eps
    assert_equal(np.longdouble(repr(o)), o,
                 "repr was %s" % repr(o))


def test_unicode():
    np.longdouble(sixu("1.2"))


def test_string():
    np.longdouble("1.2")


def test_bytes():
    np.longdouble(b"1.2")


@in_foreign_locale
def test_fromstring_foreign():
    f = 1.234
    a = np.fromstring(repr(f), dtype=float, sep=" ")
    assert_equal(a[0], f)


def test_repr_roundtrip_bytes():
    o = 1 + np.finfo(np.longdouble).eps
    assert_equal(np.longdouble(repr(o).encode("ascii")), o)


@in_foreign_locale
def test_repr_roundtrip_foreign():
    o = 1.5
    assert_equal(o, np.longdouble(repr(o)))


def test_bogus_string():
    assert_raises(ValueError, np.longdouble, "spam")
    assert_raises(ValueError, np.longdouble, "1.0 flub")


def test_underflow_zero():
    z = np.finfo


def test_fromstring():
    o = 1 + np.finfo(np.longdouble).eps
    s = (" " + repr(o))*5
    a = np.array([o]*5)
    assert_equal(np.fromstring(s, sep=" ", dtype=np.longdouble), a,
                 err_msg="reading '%s'" % s)


def test_loadtxt():
    o = 1 + np.finfo(np.longdouble).eps
    f = NamedTemporaryFile(mode="wt")
    for i in range(5):
        f.write(repr(o) + "\n")
    f.flush()
    a = np.array([o]*5)
    assert_equal(np.loadtxt(f.name, dtype=np.longdouble), a)
    f.close()


def test_genfromtxt():
    o = 1 + np.finfo(np.longdouble).eps
    f = NamedTemporaryFile(mode="wt")
    for i in range(5):
        f.write(repr(o) + "\n")
    f.flush()
    a = np.array([o]*5)
    assert_equal(np.genfromtxt(f.name, dtype=np.longdouble), a)
    f.close()


def test_fromfile():
    o = 1 + np.finfo(np.longdouble).eps
    f = NamedTemporaryFile(mode="wt")
    for i in range(5):
        f.write(repr(o) + "\n")
    f.flush()
    a = np.array([o]*5)
    F = open(f.name, "rt")
    b = np.fromfile(F,
                    dtype=np.longdouble,
                    sep="\n")
    F.close()
    F = open(f.name, "rt")
    s = F.read()
    F.close()
    f.close()
    assert_equal(b, a, err_msg="decoded %s as %s" % (repr(s), repr(b)))


def test_fromstring_bogus():
    assert_equal(np.fromstring("1. 2. 3. flop 4.", dtype=float, sep=" "),
                 np.array([1., 2., 3.]))


def test_fromstring_empty():
    assert_equal(np.fromstring("xxxxx", sep="x"),
                 np.array([]))


def test_fromstring_missing():
    assert_equal(np.fromstring("1xx3x4x5x6", sep="x"),
                 np.array([1]))


class FileBased(unittest.TestCase):
    def setUp(self):
        self.o = 1 + np.finfo(np.longdouble).eps
        self.f = NamedTemporaryFile(mode="wt")

    def tearDown(self):
        self.f.close()
        del self.f

    def test_fromfile_bogus(self):
        self.f.write("1. 2. 3. flop 4.\n")
        self.f.flush()
        F = open(self.f.name, "rt")
        try:
            assert_equal(np.fromfile(F, dtype=float, sep=" "),
                         np.array([1., 2., 3.]))
        finally:
            F.close()

    def test_tofile_roundtrip(self):
        a = np.array([self.o]*3)
        a.tofile(self.f.name, sep=" ")
        F = open(self.f.name, "rt")
        try:
            assert_equal(np.fromfile(F, dtype=np.longdouble, sep=" "),
                         a)
        finally:
            F.close()


@in_foreign_locale
def test_fromstring_foreign():
    s = "1.234"
    a = np.fromstring(s, dtype=np.longdouble, sep=" ")
    assert_equal(a[0], np.longdouble(s))


@in_foreign_locale
def test_fromstring_foreign_sep():
    a = np.array([1, 2, 3, 4])
    b = np.fromstring("1,2,3,4,", dtype=np.longdouble, sep=",")
    assert_array_equal(a, b)


@in_foreign_locale
def test_fromstring_foreign_value():
    b = np.fromstring("1,234", dtype=np.longdouble, sep=" ")
    assert_array_equal(b[0], 1)


# Conversions long double -> string


def test_repr_exact():
    o = 1 + np.finfo(np.longdouble).eps
    assert_(repr(o) != '1')


@dec.knownfailureif(longdouble_longer_than_double, "BUG #2376")
def test_format():
    o = 1 + np.finfo(np.longdouble).eps
    assert_("{0:.40g}".format(o) != '1')


@dec.knownfailureif(longdouble_longer_than_double, "BUG #2376")
def test_percent():
    o = 1 + np.finfo(np.longdouble).eps
    assert_("%.40g" % o != '1')


@dec.knownfailureif(longdouble_longer_than_double, "array repr problem")
def test_array_repr():
    o = 1 + np.finfo(np.longdouble).eps
    a = np.array([o])
    b = np.array([1], dtype=np.longdouble)
    if not np.all(a != b):
        raise ValueError("precision loss creating arrays")
    assert_(repr(a) != repr(b))


if __name__ == "__main__":
    run_module_suite()
