#cython: language_level=3

"""
Functions in this module give python-space wrappers for cython functions
exposed in numpy/__init__.pxd, so they can be tested in test_cython.py
"""
cimport numpy as cnp
cnp.import_array()


def is_td64(obj):
    return cnp.is_timedelta64_object(obj)


def is_dt64(obj):
    return cnp.is_datetime64_object(obj)


def get_dt64_value(obj):
    return cnp.get_datetime64_value(obj)


def get_td64_value(obj):
    return cnp.get_timedelta64_value(obj)


def get_dt64_unit(obj):
    return cnp.get_datetime64_unit(obj)


def is_integer(obj):
    return isinstance(obj, (cnp.integer, int))


def get_datetime_iso_8601_strlen():
    return cnp.get_datetime_iso_8601_strlen(0, cnp.NPY_FR_ns)


def convert_datetime64_to_datetimestruct():
    cdef:
        cnp.npy_datetimestruct dts
        cnp.PyArray_DatetimeMetaData meta
        cnp.int64_t value = 1647374515260292
        # i.e. (time.time() * 10**6) at 2022-03-15 20:01:55.260292 UTC

    meta.base = cnp.NPY_FR_us
    meta.num = 1
    cnp.convert_datetime64_to_datetimestruct(&meta, value, &dts)
    return dts


def make_iso_8601_datetime(dt: "datetime"):
    cdef:
        cnp.npy_datetimestruct dts
        char result[36]  # 36 corresponds to NPY_FR_s passed below
        int local = 0
        int utc = 0
        int tzoffset = 0

    dts.year = dt.year
    dts.month = dt.month
    dts.day = dt.day
    dts.hour = dt.hour
    dts.min = dt.minute
    dts.sec = dt.second
    dts.us = dt.microsecond
    dts.ps = dts.as = 0

    cnp.make_iso_8601_datetime(
        &dts,
        result,
        sizeof(result),
        local,
        utc,
        cnp.NPY_FR_s,
        tzoffset,
        cnp.NPY_NO_CASTING,
    )
    return result
