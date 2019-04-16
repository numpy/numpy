.. currentmodule:: numpy

.. _arrays.datetime:

************************
Datetimes and Timedeltas
************************

.. versionadded:: 1.7.0

Starting in NumPy 1.7, there are core array data types which natively
support datetime functionality. The data type is called "datetime64",
so named because "datetime" is already taken by the datetime library
included in Python.

.. note:: The datetime API is *experimental* in 1.7.0, and may undergo changes
   in future versions of NumPy.

Basic Datetimes
===============

The most basic way to create datetimes is from strings in
ISO 8601 date or datetime format. The unit for internal storage
is automatically selected from the form of the string, and can
be either a :ref:`date unit <arrays.dtypes.dateunits>` or a
:ref:`time unit <arrays.dtypes.timeunits>`. The date units are years ('Y'),
months ('M'), weeks ('W'), and days ('D'), while the time units are
hours ('h'), minutes ('m'), seconds ('s'), milliseconds ('ms'), and
some additional SI-prefix seconds-based units.

.. admonition:: Example

    A simple ISO date:

    >>> np.datetime64('2005-02-25')
    numpy.datetime64('2005-02-25')

    Using months for the unit:

    >>> np.datetime64('2005-02')
    numpy.datetime64('2005-02')

    Specifying just the month, but forcing a 'days' unit:

    >>> np.datetime64('2005-02', 'D')
    numpy.datetime64('2005-02-01')

    From a date and time:

    >>> np.datetime64('2005-02-25T03:30')
    numpy.datetime64('2005-02-25T03:30')

When creating an array of datetimes from a string, it is still possible
to automatically select the unit from the inputs, by using the
datetime type with generic units.

.. admonition:: Example

    >>> np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
    array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64[D]')

    >>> np.array(['2001-01-01T12:00', '2002-02-03T13:56:03.172'], dtype='datetime64')
    array(['2001-01-01T12:00:00.000-0600', '2002-02-03T13:56:03.172-0600'], dtype='datetime64[ms]')


The datetime type works with many common NumPy functions, for
example :func:`arange` can be used to generate ranges of dates.

.. admonition:: Example

    All the dates for one month:

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

    >>> np.datetime64('2005') == np.datetime64('2005-01-01')
    True

    >>> np.datetime64('2010-03-14T15Z') == np.datetime64('2010-03-14T15:00:00.00Z')
    True

Datetime and Timedelta Arithmetic
=================================

NumPy allows the subtraction of two Datetime values, an operation which
produces a number with a time unit. Because NumPy doesn't have a physical
quantities system in its core, the timedelta64 data type was created
to complement datetime64.

Datetimes and Timedeltas work together to provide ways for
simple datetime calculations.

.. admonition:: Example

    >>> np.datetime64('2009-01-01') - np.datetime64('2008-01-01')
    numpy.timedelta64(366,'D')

    >>> np.datetime64('2009') + np.timedelta64(20, 'D')
    numpy.datetime64('2009-01-21')

    >>> np.datetime64('2011-06-15T00:00') + np.timedelta64(12, 'h')
    numpy.datetime64('2011-06-15T12:00-0500')

    >>> np.timedelta64(1,'W') / np.timedelta64(1,'D')
    7.0

    >>> np.timedelta64(1,'W') % np.timedelta64(10,'D')
    numpy.timedelta64(7,'D')

There are two Timedelta units ('Y', years and 'M', months) which are treated
specially, because how much time they represent changes depending
on when they are used. While a timedelta day unit is equivalent to
24 hours, there is no way to convert a month unit into days, because
different months have different numbers of days.

.. admonition:: Example

    >>> a = np.timedelta64(1, 'Y')

    >>> np.timedelta64(a, 'M')
    numpy.timedelta64(12,'M')

    >>> np.timedelta64(a, 'D')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Cannot cast NumPy timedelta64 scalar from metadata [Y] to [D] according to the rule 'same_kind'

Datetime Units
==============

The Datetime and Timedelta data types support a large number of time
units, as well as generic units which can be coerced into any of the
other units based on input data.

Datetimes are always stored based on POSIX time (though having a TAI
mode which allows for accounting of leap-seconds is proposed), with
an epoch of 1970-01-01T00:00Z. This means the supported dates are
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
   us      microsecond      +/- 2.9e5 years         [290301 BC, 294241 AD]
   ns      nanosecond       +/- 292 years           [  1678 AD,   2262 AD]
   ps      picosecond       +/- 106 days            [  1969 AD,   1970 AD]
   fs      femtosecond      +/- 2.6 hours           [  1969 AD,   1970 AD]
   as      attosecond       +/- 9.2 seconds         [  1969 AD,   1970 AD]
======== ================ ======================= ==========================

Business Day Functionality
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

    >>> np.busday_offset('2011-06-23', 1)
    numpy.datetime64('2011-06-24')

    >>> np.busday_offset('2011-06-23', 2)
    numpy.datetime64('2011-06-27')

When an input date falls on the weekend or a holiday,
:func:`busday_offset` first applies a rule to roll the
date to a valid business day, then applies the offset. The
default rule is 'raise', which simply raises an exception.
The rules most typically used are 'forward' and 'backward'.

.. admonition:: Example

    >>> np.busday_offset('2011-06-25', 2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: Non-business day date in busday_offset

    >>> np.busday_offset('2011-06-25', 0, roll='forward')
    numpy.datetime64('2011-06-27')

    >>> np.busday_offset('2011-06-25', 2, roll='forward')
    numpy.datetime64('2011-06-29')

    >>> np.busday_offset('2011-06-25', 0, roll='backward')
    numpy.datetime64('2011-06-24')

    >>> np.busday_offset('2011-06-25', 2, roll='backward')
    numpy.datetime64('2011-06-28')

In some cases, an appropriate use of the roll and the offset
is necessary to get a desired answer.

.. admonition:: Example

    The first business day on or after a date:

    >>> np.busday_offset('2011-03-20', 0, roll='forward')
    numpy.datetime64('2011-03-21','D')
    >>> np.busday_offset('2011-03-22', 0, roll='forward')
    numpy.datetime64('2011-03-22','D')

    The first business day strictly after a date:

    >>> np.busday_offset('2011-03-20', 1, roll='backward')
    numpy.datetime64('2011-03-21','D')
    >>> np.busday_offset('2011-03-22', 1, roll='backward')
    numpy.datetime64('2011-03-23','D')

The function is also useful for computing some kinds of days
like holidays. In Canada and the U.S., Mother's day is on
the second Sunday in May, which can be computed with a custom
weekmask.

.. admonition:: Example

    >>> np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
    numpy.datetime64('2012-05-13','D')

When performance is important for manipulating many business dates
with one particular choice of weekmask and holidays, there is
an object :class:`busdaycalendar` which stores the data necessary
in an optimized form.

np.is_busday():
```````````````
To test a datetime64 value to see if it is a valid day, use :func:`is_busday`.

.. admonition:: Example

    >>> np.is_busday(np.datetime64('2011-07-15'))  # a Friday
    True
    >>> np.is_busday(np.datetime64('2011-07-16')) # a Saturday
    False
    >>> np.is_busday(np.datetime64('2011-07-16'), weekmask="Sat Sun")
    True
    >>> a = np.arange(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
    >>> np.is_busday(a)
    array([ True,  True,  True,  True,  True, False, False], dtype='bool')

np.busday_count():
``````````````````
To find how many valid days there are in a specified range of datetime64
dates, use :func:`busday_count`:

.. admonition:: Example

    >>> np.busday_count(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
    5
    >>> np.busday_count(np.datetime64('2011-07-18'), np.datetime64('2011-07-11'))
    -5

If you have an array of datetime64 day values, and you want a count of
how many of them are valid dates, you can do this:

.. admonition:: Example

    >>> a = np.arange(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
    >>> np.count_nonzero(np.is_busday(a))
    5



Custom Weekmasks
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

Changes with NumPy 1.11
=======================

In prior versions of NumPy, the datetime64 type always stored
times in UTC. By default, creating a datetime64 object from a string or
printing it would convert from or to local time::

    # old behavior
    >>>> np.datetime64('2000-01-01T00:00:00')
    numpy.datetime64('2000-01-01T00:00:00-0800')  # note the timezone offset -08:00

A consensus of datetime64 users agreed that this behavior is undesirable
and at odds with how datetime64 is usually used (e.g., by pandas_). For
most use cases, a timezone naive datetime type is preferred, similar to the
``datetime.datetime`` type in the Python standard library. Accordingly,
datetime64 no longer assumes that input is in local time, nor does it print
local times::

    >>>> np.datetime64('2000-01-01T00:00:00')
    numpy.datetime64('2000-01-01T00:00:00')

For backwards compatibility, datetime64 still parses timezone offsets, which
it handles by converting to UTC. However, the resulting datetime is timezone
naive::

    >>> np.datetime64('2000-01-01T00:00:00-08')
    DeprecationWarning: parsing timezone aware datetimes is deprecated; this will raise an error in the future
    numpy.datetime64('2000-01-01T08:00:00')

As a corollary to this change, we no longer prohibit casting between datetimes
with date units and datetimes with timeunits. With timezone naive datetimes,
the rule for casting from dates to times is no longer ambiguous.

.. _pandas: http://pandas.pydata.org


Differences Between 1.6 and 1.7 Datetimes
=========================================

The NumPy 1.6 release includes a more primitive datetime data type
than 1.7. This section documents many of the changes that have taken
place.

String Parsing
``````````````

The datetime string parser in NumPy 1.6 is very liberal in what it accepts,
and silently allows invalid input without raising errors. The parser in
NumPy 1.7 is quite strict about only accepting ISO 8601 dates, with a few
convenience extensions. 1.6 always creates microsecond (us) units by
default, whereas 1.7 detects a unit based on the format of the string.
Here is a comparison.::

    # NumPy 1.6.1
    >>> np.datetime64('1979-03-22')
    1979-03-22 00:00:00
    # NumPy 1.7.0
    >>> np.datetime64('1979-03-22')
    numpy.datetime64('1979-03-22')

    # NumPy 1.6.1, unit default microseconds
    >>> np.datetime64('1979-03-22').dtype
    dtype('datetime64[us]')
    # NumPy 1.7.0, unit of days detected from string
    >>> np.datetime64('1979-03-22').dtype
    dtype('<M8[D]')

    # NumPy 1.6.1, ignores invalid part of string
    >>> np.datetime64('1979-03-2corruptedstring')
    1979-03-02 00:00:00
    # NumPy 1.7.0, raises error for invalid input
    >>> np.datetime64('1979-03-2corruptedstring')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: Error parsing datetime string "1979-03-2corruptedstring" at position 8

    # NumPy 1.6.1, 'nat' produces today's date
    >>> np.datetime64('nat')
    2012-04-30 00:00:00
    # NumPy 1.7.0, 'nat' produces not-a-time
    >>> np.datetime64('nat')
    numpy.datetime64('NaT')

    # NumPy 1.6.1, 'garbage' produces today's date
    >>> np.datetime64('garbage')
    2012-04-30 00:00:00
    # NumPy 1.7.0, 'garbage' raises an exception
    >>> np.datetime64('garbage')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: Error parsing datetime string "garbage" at position 0

    # NumPy 1.6.1, can't specify unit in scalar constructor
    >>> np.datetime64('1979-03-22T19:00', 'h')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: function takes at most 1 argument (2 given)
    # NumPy 1.7.0, unit in scalar constructor
    >>> np.datetime64('1979-03-22T19:00', 'h')
    numpy.datetime64('1979-03-22T19:00-0500','h')

    # NumPy 1.6.1, reads ISO 8601 strings w/o TZ as UTC
    >>> np.array(['1979-03-22T19:00'], dtype='M8[h]')
    array([1979-03-22 19:00:00], dtype=datetime64[h])
    # NumPy 1.7.0, reads ISO 8601 strings w/o TZ as local (ISO specifies this)
    >>> np.array(['1979-03-22T19:00'], dtype='M8[h]')
    array(['1979-03-22T19-0500'], dtype='datetime64[h]')

    # NumPy 1.6.1, doesn't parse all ISO 8601 strings correctly
    >>> np.array(['1979-03-22T12'], dtype='M8[h]')
    array([1979-03-22 00:00:00], dtype=datetime64[h])
    >>> np.array(['1979-03-22T12:00'], dtype='M8[h]')
    array([1979-03-22 12:00:00], dtype=datetime64[h])
    # NumPy 1.7.0, handles this case correctly
    >>> np.array(['1979-03-22T12'], dtype='M8[h]')
    array(['1979-03-22T12-0500'], dtype='datetime64[h]')
    >>> np.array(['1979-03-22T12:00'], dtype='M8[h]')
    array(['1979-03-22T12-0500'], dtype='datetime64[h]')

Unit Conversion
```````````````

The 1.6 implementation of datetime does not convert between units correctly.::

    # NumPy 1.6.1, the representation value is untouched
    >>> np.array(['1979-03-22'], dtype='M8[D]')
    array([1979-03-22 00:00:00], dtype=datetime64[D])
    >>> np.array(['1979-03-22'], dtype='M8[D]').astype('M8[M]')
    array([2250-08-01 00:00:00], dtype=datetime64[M])
    # NumPy 1.7.0, the representation is scaled accordingly
    >>> np.array(['1979-03-22'], dtype='M8[D]')
    array(['1979-03-22'], dtype='datetime64[D]')
    >>> np.array(['1979-03-22'], dtype='M8[D]').astype('M8[M]')
    array(['1979-03'], dtype='datetime64[M]')

Datetime Arithmetic
```````````````````

The 1.6 implementation of datetime only works correctly for a small subset of
arithmetic operations. Here we show some simple cases.::

    # NumPy 1.6.1, produces invalid results if units are incompatible
    >>> a = np.array(['1979-03-22T12'], dtype='M8[h]')
    >>> b = np.array([3*60], dtype='m8[m]')
    >>> a + b
    array([1970-01-01 00:00:00.080988], dtype=datetime64[us])
    # NumPy 1.7.0, promotes to higher-resolution unit
    >>> a = np.array(['1979-03-22T12'], dtype='M8[h]')
    >>> b = np.array([3*60], dtype='m8[m]')
    >>> a + b
    array(['1979-03-22T15:00-0500'], dtype='datetime64[m]')

    # NumPy 1.6.1, arithmetic works if everything is microseconds
    >>> a = np.array(['1979-03-22T12:00'], dtype='M8[us]')
    >>> b = np.array([3*60*60*1000000], dtype='m8[us]')
    >>> a + b
    array([1979-03-22 15:00:00], dtype=datetime64[us])
    # NumPy 1.7.0
    >>> a = np.array(['1979-03-22T12:00'], dtype='M8[us]')
    >>> b = np.array([3*60*60*1000000], dtype='m8[us]')
    >>> a + b
    array(['1979-03-22T15:00:00.000000-0500'], dtype='datetime64[us]')
