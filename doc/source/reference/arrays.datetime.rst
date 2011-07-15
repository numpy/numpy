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

Basic Datetimes
===============

The most basic way to create datetimes is from strings in
ISO 8601 date or datetime format. The unit for internal storage
is automatically selected from the form of the string, and can
be either a :ref:`date unit <arrays.dtypes.dateunits>` or a
:ref:`time unit <arrays.dtypes.timeunits>`. The date units are years ('Y'),
months ('M'), weeks ('W'), and days ('D'), while the time units are
hours ('h'), minutes ('m'), seconds ('s'), milliseconds ('ms'), and
more SI-prefix seconds-based units.

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

    Using UTC "Zulu" time:

    >>> np.datetime64('2005-02-25T03:30Z')
    numpy.datetime64('2005-02-24T21:30-0600')

    ISO 8601 specifies to use the local time zone
    if none is explicitly given:

    >>> np.datetime64('2005-02-25T03:30')
    numpy.datetime64('2005-02-25T03:30-0600')

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

An important exception to this rule is between datetimes with
:ref:`date units <arrays.dtypes.dateunits>` and datetimes with
:ref:`time units <arrays.dtypes.timeunits>`. This is because this kind
of conversion generally requires a choice of timezone and
particular time of day on the given date.

.. admonition:: Example

    >>> np.datetime64('2003-12-25', 's')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Cannot parse "2003-12-25" as unit 's' using casting rule 'same_kind'

    >>> np.datetime64('2003-12-25') == np.datetime64('2003-12-25T00Z')
    False


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

There are two Timedelta units, years and months, which are treated
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
a epoch of 1970-01-01T00:00Z. This means the supported dates are
always a symmetric interval around 1970.

Here are the date units:

.. _arrays.dtypes.dateunits:

======== ================ ======================= ==========================
  Code       Meaning       Time span (relative)    Time span (absolute)
======== ================ ======================= ==========================
   Y       year             +- 9.2e18 years         [9.2e18 BC, 9.2e18 AD]
   M       month            +- 7.6e17 years         [7.6e17 BC, 7.6e17 AD]
   W       week             +- 1.7e17 years         [1.7e17 BC, 1.7e17 AD]
   D       day              +- 2.5e16 years         [2.5e16 BC, 2.5e16 AD]
======== ================ ======================= ==========================

And here are the time units:

.. _arrays.dtypes.timeunits:

======== ================ ======================= ==========================
  Code       Meaning       Time span (relative)    Time span (absolute)
======== ================ ======================= ==========================
   h       hour             +- 1.0e15 years         [1.0e15 BC, 1.0e15 AD]
   m       minute           +- 1.7e13 years         [1.7e13 BC, 1.7e13 AD]
   s       second           +- 2.9e12 years         [ 2.9e9 BC,  2.9e9 AD]
   ms      millisecond      +- 2.9e9 years          [ 2.9e6 BC,  2.9e6 AD]
   us      microsecond      +- 2.9e6 years          [290301 BC, 294241 AD]
   ns      nanosecond       +- 292 years            [  1678 AD,   2262 AD]
   ps      picosecond       +- 106 days             [  1969 AD,   1970 AD]
   fs      femtosecond      +- 2.6 hours            [  1969 AD,   1970 AD]
   as      attosecond       +- 9.2 seconds          [  1969 AD,   1970 AD]
======== ================ ======================= ==========================

Business Day Functionality
==========================

To allow the datetime to be used in contexts where accounting for weekends
and holidays is important, NumPy includes a set of functions for
working with business days.

The function :func:`busday_offset` allows you to apply offsets
specified in business days to datetimes with a unit of 'day'. By default,
a business date is defined to be any date which falls on Monday through
Friday, but this can be customized with a weekmask and a list of holidays.

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
the second Sunday in May, which can be computed with a special
weekmask.

.. admonition:: Example

    >>> np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
    numpy.datetime64('2012-05-13','D')

When performance is important for manipulating many business date
with one particular choice of weekmask and holidays, there is
an object :class:`busdaycalendar` which stores the data necessary
in an optimized form.

The other two functions for business days are :func:`is_busday`
and :func:`busday_count`, which are more straightforward and
not explained here.
