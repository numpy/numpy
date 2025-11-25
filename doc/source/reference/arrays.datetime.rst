.. currentmodule:: numpy

.. _arrays.datetime:

************************
Datetimes and timedeltas
************************

Starting in NumPy 1.7, there are core array data types which natively
support datetime functionality. The data type is called :class:`datetime64`,
so named because :class:`~datetime.datetime` is already taken by the Python standard library.

Datetime64 conventions and assumptions
======================================

Similar to the Python `~datetime.date` class, dates are expressed in the current
Gregorian Calendar, indefinitely extended both in the future and in the past.
[#]_ Contrary to Python `~datetime.date`, which supports only years in the 1 AD — 9999
AD range, `datetime64` allows also for dates BC; years BC follow the `Astronomical
year numbering <https://en.wikipedia.org/wiki/Astronomical_year_numbering>`_
convention, i.e. year 2 BC is numbered −1, year 1 BC is numbered 0, year 1 AD is
numbered 1.

Time instants, say 16:23:32.234, are represented counting hours, minutes,
seconds and fractions from midnight: i.e. 00:00:00.000 is midnight, 12:00:00.000
is noon, etc. Each calendar day has exactly 86400 seconds. This is a "naive"
time, with no explicit notion of timezones or specific time scales (UT1, UTC, TAI,
etc.). [#]_

.. [#] The calendar obtained by extending the Gregorian calendar before its
       official adoption on Oct. 15, 1582 is called `Proleptic Gregorian Calendar
       <https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar>`_

.. [#] The assumption of 86400 seconds per calendar day is not valid for UTC,
       the present day civil time scale. In fact due to the presence of
       `leap seconds <https://en.wikipedia.org/wiki/Leap_second>`_ on rare occasions
       a day may be 86401 or 86399 seconds long. On the contrary the 86400s day
       assumption holds for the TAI timescale. An explicit support for TAI and
       TAI to UTC conversion, accounting for leap seconds, is proposed but not
       yet implemented. See also the `shortcomings`_ section below.


Basic datetimes
===============

The most basic way to create datetimes is from strings in ISO 8601 date
or datetime format. It is also possible to create datetimes from an integer by
offset relative to the Unix epoch (00:00:00 UTC on 1 January 1970).
The unit for internal storage is automatically selected from the
form of the string, and can be either a :ref:`date unit <arrays.dtypes.dateunits>` or a
:ref:`time unit <arrays.dtypes.timeunits>`. The date units are years ('Y'),
months ('M'), weeks ('W'), and days ('D'), while the time units are
hours ('h'), minutes ('m'), seconds ('s'), milliseconds ('ms'), and
some additional SI-prefix seconds-based units. The `datetime64` data type
also accepts the string "NAT", in any combination of lowercase/uppercase
letters, for a "Not A Time" value. The string "now" is also supported and
returns the current UTC time. By default, it uses second ('s') precision, but
you can specify a different unit (e.g., 'M', 'D', 'h') to truncate the result
to that precision. Units finer than seconds (such as 'ms' or 'ns') are
supported but will show fractional parts as zeros, effectively truncating to
whole seconds. The string "today" is also supported and returns the current UTC
date with day precision. It also supports the same precision specifiers
as ``now``.

.. admonition:: Example

  .. try_examples::

    A simple ISO date:

    >>> import numpy as np

    >>> np.datetime64('2005-02-25')
    np.datetime64('2005-02-25')

    From an integer and a date unit, 1 year since the UNIX epoch:

    >>> np.datetime64(1, 'Y')
    np.datetime64('1971')

    Using months for the unit:

    >>> np.datetime64('2005-02')
    np.datetime64('2005-02')

    Specifying just the month, but forcing a 'days' unit:

    >>> np.datetime64('2005-02', 'D')
    np.datetime64('2005-02-01')

    From a date and time:

    >>> np.datetime64('2005-02-25T03:30')
    np.datetime64('2005-02-25T03:30')

    NAT (not a time):

    >>> np.datetime64('nat')
    np.datetime64('NaT')

    The current time (UTC, default second precision):

    >>> np.datetime64('now')
    np.datetime64('2025-08-05T02:22:14')  # result will depend on the current time

    >>> np.datetime64('now', 'D')
    np.datetime64('2025-08-05')
    
    >>> np.datetime64('now', 'ms')
    np.datetime64('2025-08-05T02:22:14.000')

    The current date:

    >>> np.datetime64('today')
    np.datetime64('2025-08-05')  # result will depend on the current date

When creating an array of datetimes from a string, it is still possible
to automatically select the unit from the inputs, by using the
datetime type with generic units.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
    array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64[D]')

    >>> np.array(['2001-01-01T12:00', '2002-02-03T13:56:03.172'], dtype='datetime64')
    array(['2001-01-01T12:00:00.000', '2002-02-03T13:56:03.172'],
          dtype='datetime64[ms]')

An array of datetimes can be constructed from integers representing
POSIX timestamps with the given unit.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.array([0, 1577836800], dtype='datetime64[s]')
    array(['1970-01-01T00:00:00', '2020-01-01T00:00:00'],
          dtype='datetime64[s]')

    >>> np.array([0, 1577836800000]).astype('datetime64[ms]')
    array(['1970-01-01T00:00:00.000', '2020-01-01T00:00:00.000'],
          dtype='datetime64[ms]')

The datetime type works with many common NumPy functions, for
example :func:`arange` can be used to generate ranges of dates.

.. admonition:: Example

  .. try_examples::

    All the dates for one month:

    >>> import numpy as np

    >>> np.arange('2005-02', '2005-03', dtype='datetime64[D]')
    array(['2005-02-01', '2005-02-02', '2005-02-03', '2005-02-04',
           '2005-02-05', '2005-02-06', '2005-02-07', '2005-02-08',
           '2005-02-09', '2005-02-10', '2005-02-11', '2005-02-12',
           '2005-02-13', '2005-02-14', '2005-02-15', '2005-02-16',
           '2005-02-17', '2005-02-18', '2005-02-19', '2005-02-20',
           '2005-02-21', '2005-02-22', '2005-02-23', '2005-02-24',
           '2005-02-25', '2005-02-26', '2005-02-27', '2005-02-28'],
          dtype='datetime64[D]')

The datetime object represents a single moment in time. If two
datetimes have different units, they may still be representing
the same moment of time, and converting from a bigger unit like
months to a smaller unit like days is considered a 'safe' cast
because the moment of time is still being represented exactly.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.datetime64('2005') == np.datetime64('2005-01-01')
    True

    >>> np.datetime64('2010-03-14T15') == np.datetime64('2010-03-14T15:00:00.00')
    True

.. deprecated:: 1.11.0

  NumPy does not store timezone information. For backwards compatibility, datetime64
  still parses timezone offsets, which it handles by converting to
  UTC±00:00 (Zulu time). This behaviour is deprecated and will raise an error in the
  future.


Datetime and timedelta arithmetic
=================================

NumPy allows the subtraction of two datetime values, an operation which
produces a number with a time unit. Because NumPy doesn't have a physical
quantities system in its core, the `timedelta64` data type was created
to complement `datetime64`. The arguments for `timedelta64` are a number,
to represent the number of units, and a date/time unit, such as
(D)ay, (M)onth, (Y)ear, (h)ours, (m)inutes, or (s)econds. The `timedelta64`
data type also accepts the string "NAT" in place of the number for a "Not A Time" value.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.timedelta64(1, 'D')
    np.timedelta64(1,'D')

    >>> np.timedelta64(4, 'h')
    np.timedelta64(4,'h')

    >>> np.timedelta64('nAt')
    np.timedelta64('NaT')

Datetimes and Timedeltas work together to provide ways for
simple datetime calculations.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.datetime64('2009-01-01') - np.datetime64('2008-01-01')
    np.timedelta64(366,'D')

    >>> np.datetime64('2009') + np.timedelta64(20, 'D')
    np.datetime64('2009-01-21')

    >>> np.datetime64('2011-06-15T00:00') + np.timedelta64(12, 'h')
    np.datetime64('2011-06-15T12:00')

    >>> np.timedelta64(1,'W') / np.timedelta64(1,'D')
    7.0

    >>> np.timedelta64(1,'W') % np.timedelta64(10,'D')
    np.timedelta64(7,'D')

    >>> np.datetime64('nat') - np.datetime64('2009-01-01')
    np.timedelta64('NaT','D')

    >>> np.datetime64('2009-01-01') + np.timedelta64('nat')
    np.datetime64('NaT')

There are two Timedelta units ('Y', years and 'M', months) which are treated
specially, because how much time they represent changes depending
on when they are used. While a timedelta day unit is equivalent to
24 hours, month and year units cannot be converted directly into days
without using 'unsafe' casting.

The `numpy.ndarray.astype` method can be used for unsafe
conversion of months/years to days. The conversion follows
calculating the averaged values from the 400 year leap-year cycle.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> a = np.timedelta64(1, 'Y')

    >>> np.timedelta64(a, 'M')
    numpy.timedelta64(12,'M')

    >>> np.timedelta64(a, 'D')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Cannot cast NumPy timedelta64 scalar from metadata [Y] to [D] according to the rule 'same_kind'


Datetime units
==============

The Datetime and Timedelta data types support a large number of time
units, as well as generic units which can be coerced into any of the
other units based on input data.

Datetimes are always stored with
an epoch of 1970-01-01T00:00. This means the supported dates are
always a symmetric interval around the epoch, called "time span" in the
table below.

The length of the span is the range of a 64-bit integer times the length
of the date or unit.  For example, the time span for 'W' (week) is exactly
7 times longer than the time span for 'D' (day), and the time span for
'D' (day) is exactly 24 times longer than the time span for 'h' (hour).

Here are the date units:

.. _arrays.dtypes.dateunits:

======== ================ ======================= ==========================
  Code       Meaning       Time span (relative)    Time span (absolute)
======== ================ ======================= ==========================
   Y       year             +/- 9.2e18 years        [9.2e18 BC, 9.2e18 AD]
   M       month            +/- 7.6e17 years        [7.6e17 BC, 7.6e17 AD]
   W       week             +/- 1.7e17 years        [1.7e17 BC, 1.7e17 AD]
   D       day              +/- 2.5e16 years        [2.5e16 BC, 2.5e16 AD]
======== ================ ======================= ==========================

And here are the time units:

.. _arrays.dtypes.timeunits:

======== ================ ======================= ==========================
  Code       Meaning       Time span (relative)    Time span (absolute)
======== ================ ======================= ==========================
   h       hour             +/- 1.0e15 years        [1.0e15 BC, 1.0e15 AD]
   m       minute           +/- 1.7e13 years        [1.7e13 BC, 1.7e13 AD]
   s       second           +/- 2.9e11 years        [2.9e11 BC, 2.9e11 AD]
   ms      millisecond      +/- 2.9e8 years         [ 2.9e8 BC,  2.9e8 AD]
us / μs    microsecond      +/- 2.9e5 years         [290301 BC, 294241 AD]
   ns      nanosecond       +/- 292 years           [  1678 AD,   2262 AD]
   ps      picosecond       +/- 106 days            [  1969 AD,   1970 AD]
   fs      femtosecond      +/- 2.6 hours           [  1969 AD,   1970 AD]
   as      attosecond       +/- 9.2 seconds         [  1969 AD,   1970 AD]
======== ================ ======================= ==========================


Converting datetime and timedelta to Python Object
==================================================

NumPy follows a strict protocol when converting `datetime64` and/or `timedelta64` to Python Objects (e.g., ``tuple``, ``list``, `datetime.datetime`). 

The protocol is described in the following table:

================================ ================================= ==================================
         Input Type                         for `datetime64`               for `timedelta64`
================================ ================================= ==================================
          ``NaT``                             ``None``                           ``None``
        ns/ps/fs/as                           ``int``                            ``int``
        μs/ms/s/m/h                      `datetime.datetime`               `datetime.timedelta`
      D/W (Linear units)                   `datetime.date`                 `datetime.timedelta` 
    Y/M (Non-linear units)                 `datetime.date`                       ``int``
        Generic units                      `datetime.date`                       ``int``
================================ ================================= ==================================

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> type(np.datetime64('NaT').item())
    <class 'NoneType'>

    >>> type(np.timedelta64('NaT').item())
    <class 'NoneType'>

    >>> type(np.timedelta64(123, 'ns').item())
    <class 'int'>

    >>> type(np.datetime64('2025-01-01T12:00:00.123456').item())
    <class 'datetime.datetime'>

    >>> type(np.timedelta64(10, 'D').item())
    <class 'datetime.timedelta'>


In the case where conversion of `datetime64` and/or `timedelta64` is done against Python types like ``int``, ``float``, and ``str`` the corresponding return types will be ``np.str_``, ``np.int64`` and ``np.float64``.


.. admonition:: Example

  .. try_examples::

    >>> import numpy as np
    
    >>> type(np.timedelta64(1, 'D').astype(int))
    <class 'numpy.int64'>

    >>> type(np.datetime64('2025-01-01T12:00:00.123456').astype(float))
    <class 'numpy.float64'>

    >>> type(np.timedelta64(123, 'ns').astype(str))
    <class 'numpy.str_'>


Business day functionality
==========================

To allow the datetime to be used in contexts where only certain days of
the week are valid, NumPy includes a set of "busday" (business day)
functions.

The default for busday functions is that the only valid days are Monday
through Friday (the usual business days).  The implementation is based on
a "weekmask" containing 7 Boolean flags to indicate valid days; custom
weekmasks are possible that specify other sets of valid days.

The "busday" functions can additionally check a list of "holiday" dates,
specific dates that are not valid days.

The function :func:`busday_offset` allows you to apply offsets
specified in business days to datetimes with a unit of 'D' (day).

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.busday_offset('2011-06-23', 1)
    np.datetime64('2011-06-24')

    >>> np.busday_offset('2011-06-23', 2)
    np.datetime64('2011-06-27')

When an input date falls on the weekend or a holiday,
:func:`busday_offset` first applies a rule to roll the
date to a valid business day, then applies the offset. The
default rule is 'raise', which simply raises an exception.
The rules most typically used are 'forward' and 'backward'.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.busday_offset('2011-06-25', 2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: Non-business day date in busday_offset

    >>> np.busday_offset('2011-06-25', 0, roll='forward')
    np.datetime64('2011-06-27')

    >>> np.busday_offset('2011-06-25', 2, roll='forward')
    np.datetime64('2011-06-29')

    >>> np.busday_offset('2011-06-25', 0, roll='backward')
    np.datetime64('2011-06-24')

    >>> np.busday_offset('2011-06-25', 2, roll='backward')
    np.datetime64('2011-06-28')

In some cases, an appropriate use of the roll and the offset
is necessary to get a desired answer.

.. admonition:: Example

  .. try_examples::

    The first business day on or after a date:

    >>> import numpy as np

    >>> np.busday_offset('2011-03-20', 0, roll='forward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 0, roll='forward')
    np.datetime64('2011-03-22')

    The first business day strictly after a date:

    >>> np.busday_offset('2011-03-20', 1, roll='backward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 1, roll='backward')
    np.datetime64('2011-03-23')

The function is also useful for computing some kinds of days
like holidays. In Canada and the U.S., Mother's day is on
the second Sunday in May, which can be computed with a custom
weekmask.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
    np.datetime64('2012-05-13')

When performance is important for manipulating many business dates
with one particular choice of weekmask and holidays, there is
an object :class:`busdaycalendar` which stores the data necessary
in an optimized form.

np.is_busday():
---------------
To test a `datetime64` value to see if it is a valid day, use :func:`is_busday`.

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.is_busday(np.datetime64('2011-07-15'))  # a Friday
    True
    >>> np.is_busday(np.datetime64('2011-07-16')) # a Saturday
    False
    >>> np.is_busday(np.datetime64('2011-07-16'), weekmask="Sat Sun")
    True
    >>> a = np.arange(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
    >>> np.is_busday(a)
    array([ True,  True,  True,  True,  True, False, False])

np.busday_count():
------------------
To find how many valid days there are in a specified range of datetime64
dates, use :func:`busday_count`:

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> np.busday_count(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
    5
    >>> np.busday_count(np.datetime64('2011-07-18'), np.datetime64('2011-07-11'))
    -5

If you have an array of datetime64 day values, and you want a count of
how many of them are valid dates, you can do this:

.. admonition:: Example

  .. try_examples::

    >>> import numpy as np

    >>> a = np.arange(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
    >>> np.count_nonzero(np.is_busday(a))
    5



Custom weekmasks
----------------

Here are several examples of custom weekmask values.  These examples
specify the "busday" default of Monday through Friday being valid days.

Some examples::

    # Positional sequences; positions are Monday through Sunday.
    # Length of the sequence must be exactly 7.
    weekmask = [1, 1, 1, 1, 1, 0, 0]
    # list or other sequence; 0 == invalid day, 1 == valid day
    weekmask = "1111100"
    # string '0' == invalid day, '1' == valid day

    # string abbreviations from this list: Mon Tue Wed Thu Fri Sat Sun
    weekmask = "Mon Tue Wed Thu Fri"
    # any amount of whitespace is allowed; abbreviations are case-sensitive.
    weekmask = "MonTue Wed  Thu\tFri"


.. _shortcomings:

Datetime64 shortcomings
=======================

The assumption that all days are exactly 86400 seconds long makes `datetime64`
largely compatible with Python `datetime` and "POSIX time" semantics; therefore
they all share the same well known shortcomings with respect to the UTC
timescale and historical time determination. A brief non exhaustive summary is
given below.

- It is impossible to parse valid UTC timestamps occurring during a positive
  leap second.

  .. admonition:: Example

    "2016-12-31 23:59:60 UTC" was a leap second, therefore "2016-12-31
    23:59:60.450 UTC" is a valid timestamp which is not parseable by
    `datetime64`:

    .. try_examples::

      >>> import numpy as np

      >>> np.datetime64("2016-12-31 23:59:60.450")
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      ValueError: Seconds out of range in datetime string "2016-12-31 23:59:60.450"

- Timedelta64 computations between two UTC dates can be wrong by an integer
  number of SI seconds.

  .. admonition:: Example

    Compute the number of SI seconds between "2021-01-01 12:56:23.423 UTC" and
    "2001-01-01 00:00:00.000 UTC":

    .. try_examples::

      >>> import numpy as np

      >>> (
      ...   np.datetime64("2021-01-01 12:56:23.423")
      ...   - np.datetime64("2001-01-01")
      ... ) / np.timedelta64(1, "s")
      631198583.423

      However, the correct answer is `631198588.423` SI seconds, because there were
      5 leap seconds between 2001 and 2021.

- Timedelta64 computations for dates in the past do not return SI seconds, as
  one would expect.

  .. admonition:: Example

     Compute the number of seconds between "000-01-01 UT" and "1600-01-01 UT",
     where UT is `universal time
     <https://en.wikipedia.org/wiki/Universal_Time>`_:

    .. try_examples::

      >>> import numpy as np

      >>> a = np.datetime64("0000-01-01", "us")
      >>> b = np.datetime64("1600-01-01", "us")
      >>> b - a
      numpy.timedelta64(50491123200000000,'us')

      The computed results, `50491123200` seconds, are obtained as the elapsed
      number of days (`584388`) times `86400` seconds; this is the number of
      seconds of a clock in sync with the Earth's rotation. The exact value in SI
      seconds can only be estimated, e.g., using data published in `Measurement of
      the Earth's rotation: 720 BC to AD 2015, 2016, Royal Society's Proceedings
      A 472, by Stephenson et.al. <https://doi.org/10.1098/rspa.2016.0404>`_. A
      sensible estimate is `50491112870 ± 90` seconds, with a difference of 10330
      seconds.
