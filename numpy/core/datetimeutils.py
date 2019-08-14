"""
Utilities for extracting datetime64 information.
"""

import functools
import sys
import warnings
import sys

from . import overrides
import numpy as np

__all__ = ['datetime_year', 'datetime_month', 'datetime_day', 'datetime_hour',
           'datetime_minute', 'datetime_second', 'datetime_microsecond',
           'datetime_yday', 'datetime_wday']

@overrides.set_module('numpy')
def datetime_year(arr):
    """
    Unpack the year from a datetime64 object.

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array-like of integers representing the year.
        Note for fine resolutions (i.e. picoseconds, this may return
        something close to the internal epoch (1970).

    Examples
    --------
    >>> d = np.arange('2015-12-31', 12*3, 12, dtype='datetime64[M]')
    >>> d
    array(['2015-12', '2016-12', '2017-12'], dtype='datetime64[M]')
    >>> y = np.datetime_year(d); y
    array([2015, 2016, 2017])
    """
    return arr.astype('datetime64[Y]').astype(int) + 1970

@overrides.set_module('numpy')
def datetime_month(arr):
    """
    Unpack the month from a datetime64 object.

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array-like of integers representing the month of year (1-12).

    Examples
    --------
    >>> d = np.arange('2015-12-20', 31*3, 15, dtype='datetime64[D]')
    >>> d
    array(['2015-12-20', '2016-01-04', '2016-01-19', '2016-02-03',
           '2016-02-18', '2016-03-04', '2016-03-19'], dtype='datetime64[D]')
    >>> m = np.datetime_month(d); m
    array([12,  1,  1,  2,  2,  3,  3])
    """
    return arr.astype('datetime64[M]').astype(int) % 12 + 1

@overrides.set_module('numpy')
def datetime_day(arr):
    """
    Unpack the day of month from a datetime64 object.

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array-like of integers representing the day of thee month (1-31).

    Examples
    --------
    >>> d = np.arange('2015-12-20', 31*3, 15, dtype='datetime64[D]')
    >>> d
    array(['2015-12-20', '2016-01-04', '2016-01-19', '2016-02-03',
           '2016-02-18', '2016-03-04', '2016-03-19'], dtype='datetime64[D]')
    >>> day = np.datetime_day(d); day
    array([20,  4, 19,  3, 18,  4, 19])
    """
    return (arr.astype('datetime64[D]') -
            arr.astype('datetime64[M]')).astype(int) + 1

@overrides.set_module('numpy')
def datetime_hour(arr):
    """
    Unpack the hour of the day from a datetime64 object.

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array-like of integers representing the hour of the day (0-23).

    Examples
    --------
    >>> d = np.arange('2015-10-03T20:10:13', 60*4, 60, dtype='datetime64[m]')
    >>> d
    array(['2015-10-03T20:10', '2015-10-03T21:10', '2015-10-03T22:10',
           '2015-10-03T23:10'], dtype='datetime64[m]')
    >>> h = np.datetime_hour(d); h
    array([20, 21, 22, 23])
    """
    return arr.astype('datetime64[h]').astype(int) % 24

@overrides.set_module('numpy')
def datetime_minute(arr):
    """
    Unpack the mintes of hour from a datetime64 object.

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array-like of integers representing the minute of the hour (0-59).

    Examples
    --------
    >>> d = np.arange('2015-10-03T20:10:13', 60*4, 60, dtype='datetime64[s]')
    >>> d
    array(['2015-10-03T20:10:13', '2015-10-03T20:11:13',
           '2015-10-03T20:12:13', '2015-10-03T20:13:13'],
          dtype='datetime64[s]')
    >>> m = np.datetime_minute(d); m
    array([10, 11, 12, 13])
    """
    return (arr.astype('datetime64[m]') -
            arr.astype('datetime64[h]')).astype(int)

@overrides.set_module('numpy')
def datetime_second(arr):
    """
    Unpack the seconds from a datetime64 object.

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array-like of integers representing the second of the minute (0-59).

    Examples
    --------
    >>> d = np.arange('2015-10-03T20:10:13.1', 1000*3, 700,
    ...               dtype='datetime64[ms]')
    >>> d
    array(['2015-10-03T20:10:13.100', '2015-10-03T20:10:13.800',
           '2015-10-03T20:10:14.500', '2015-10-03T20:10:15.200',
           '2015-10-03T20:10:15.900'], dtype='datetime64[ms]')
    >>> s = np.datetime_second(d); s
    array([13, 13, 14, 15, 15])
    """
    return (arr.astype('datetime64[s]') -
            arr.astype('datetime64[m]')).astype(int)

@overrides.set_module('numpy')
def datetime_microsecond(arr):
    """
    Unpack the microseconds from a datetime64 object.

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array-like of integers representing the microseconds of the second.
        These range from 0 to 999999, so half a second = 500000.

    Examples
    --------
    >>> d = np.arange('2015-10-03T20:10:13.125', 1000*3, 700,
    ...                dtype='datetime64[ns]')
    >>> d
    array(['2015-10-03T20:10:13.125000000', '2015-10-03T20:10:13.125000700',
           '2015-10-03T20:10:13.125001400', '2015-10-03T20:10:13.125002100',
           '2015-10-03T20:10:13.125002800'], dtype='datetime64[ns]')
    >>> m = np.datetime_microsecond(d); m
    array([125000, 125000, 125001, 125002, 125002])
    """
    return (arr.astype('datetime64[us]') -
            arr.astype('datetime64[s]')).astype(int)

@overrides.set_module('numpy')
def datetime_yday(arr):
    """
    Unpack day of year from datetime64 as floating point

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array like of floats in units of days since 1 Jan 00:00. Note
    1 Jan 12:00 is 0.5 (some yearday conventions will call this 1.5)

    Examples
    --------
    >>> d = np.datetime64('2012-01-01T12:00')
    >>> np.datetime_yday(d)
    0.5
    >>> d = np.datetime64('2012-12-31T12:00')
    >>> np.datetime_yday(d)  # note this is a leap year
    365.5
    """

    td = (arr - arr.astype('datetime64[Y]'))
    return (td).astype('timedelta64[ns]').astype(float) / 1.e9 / 24 / 3600

@overrides.set_module('numpy')
def datetime_wday(arr):
    """
    Unpack days since Sunday from datetime64 as integer

    Parameters
    ----------
    arr : array_like of datetime64

    Returns
    -------
    array like of ints in units of days since Sunday [0-6].

    Examples
    --------
    >>> d = np.datetime64('2012-12-31T23:30')
    >>> np.datetime_wday(d)  # a Monday
    1
    """

    randomsunday = np.datetime64('1969-12-28')  # actually sunday nearest epoch
    return (arr - randomsunday).astype('timedelta64[D]').astype(int) % 7
