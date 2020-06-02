"""
Functions in this module give python-space wrappers for cython functions
exposed in numpy/__init__.pxd, so they can be tested in test_cython.py
"""
from datetime import datetime

import numpy as np
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


def make_iso_8601_datetime(dt: datetime):
    cdef:
        cnp.npy_datetimestruct dts
        char* result
        cnp.npy_intp outlen
        int local = 0
        int utc = 0
        int tzoffset = 0

    dts.year = dt.year
    dts.month = dt.month
    dts.day = dt.day
    dts.hour = dt.hour
    dts.sec = dt.second
    dts.us = dt.microsecond

    cnp.make_iso_8601_datetime(
        &dts,
        result,
        outlen,
        local,
        utc,
        cnp.NPY_FR_s,
        tzoffset,
        cnp.NPY_NO_CASTING,
    )
    return result


def get_datetime_iso_8601_strlen():
    return cnp.get_datetime_iso_8601_strlen(0, cnp.NPY_FR_Y)


def parse_iso_8601_datetime(obj):
    # cnp.parse_iso_8601_datetime(...)
    raise NotImplementedError

def convert_datetime_to_datetimestruct(dt: datetime):
    cdef:
        cnp.PyArray_DatetimeMetaData meta
        cnp.npy_datetimestruct dts

    meta.base = cnp.NPY_FR_us
    meta.num = 1

    dt64 = np.datetime64(dt)
    cnp.convert_datetime_to_datetimestruct(&meta, dt64.view("i8"), &dts)
    return dts  # gets converted to a dict in python-space


def convert_datetimestruct_to_datetime(obj):
    # cnp.convert_datetimestruct_to_datetime(...)
    raise NotImplementedError
