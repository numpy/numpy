=========================================================================
NEP 4 — A (third) proposal for implementing some date/time types in NumPy
=========================================================================

:Author: Francesc Alted i Abad
:Contact: faltet@pytables.com
:Author: Ivan Vilata i Balaguer
:Contact: ivan@selidor.net
:Date: 2008-07-30
:Status: Deferred

Executive summary
=================

A date/time mark is something very handy to have in many fields where
one has to deal with data sets.  While Python has several modules that
define a date/time type (like the integrated ``datetime`` [1]_ or
``mx.DateTime`` [2]_), NumPy has a lack of them.

In this document, we are proposing the addition of a series of date/time
types to fill this gap.  The requirements for the proposed types are
two-folded: 1) they have to be fast to operate with and 2) they have to
be as compatible as possible with the existing ``datetime`` module that
comes with Python.


Types proposed
==============

To start with, it is virtually impossible to come up with a single
date/time type that fills the needs of every case of use.  So, after
pondering about different possibilities, we have stuck with *two*
different types, namely ``datetime64`` and ``timedelta64`` (these names
are preliminary and can be changed), that can have different time units
so as to cover different needs.

.. Important:: the time unit is conceived here as metadata that
  *complements* a date/time dtype, *without changing the base type*.  It
  provides information about the *meaning* of the stored numbers, not
  about their *structure*.

Now follows a detailed description of the proposed types.


``datetime64``
--------------

It represents a time that is absolute (i.e. not relative).  It is
implemented internally as an ``int64`` type.  The internal epoch is the
POSIX epoch (see [3]_).  Like POSIX, the representation of a date
doesn't take leap seconds into account.

In time unit *conversions* and time *representations* (but not in other
time computations), the value -2**63 (0x8000000000000000) is interpreted
as an invalid or unknown date, *Not a Time* or *NaT*.  See the section
on time unit conversions for more information.

Time units
~~~~~~~~~~

It accepts different time units, each of them implying a different time
span.  The table below describes the time units supported with their
corresponding time spans.

======== ================ ==========================
      Time unit               Time span (years)
------------------------- --------------------------
  Code       Meaning
======== ================ ==========================
   Y       year             [9.2e18 BC, 9.2e18 AD]
   M       month            [7.6e17 BC, 7.6e17 AD]
   W       week             [1.7e17 BC, 1.7e17 AD]
   B       business day     [3.5e16 BC, 3.5e16 AD]
   D       day              [2.5e16 BC, 2.5e16 AD]
   h       hour             [1.0e15 BC, 1.0e15 AD]
   m       minute           [1.7e13 BC, 1.7e13 AD]
   s       second           [ 2.9e9 BC,  2.9e9 AD]
   ms      millisecond      [ 2.9e6 BC,  2.9e6 AD]
   us      microsecond      [290301 BC, 294241 AD]
   c#      ticks (100ns)    [  2757 BC,  31197 AD]
   ns      nanosecond       [  1678 AD,   2262 AD]
======== ================ ==========================

The value of an absolute date is thus *an integer number of units of the
chosen time unit* passed since the internal epoch.  When working with
business days, Saturdays and Sundays are simply ignored from the count
(i.e. day 3 in business days is not Saturday 1970-01-03, but Monday
1970-01-05).

Building a ``datetime64`` dtype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed ways to specify the time unit in the dtype constructor are:

Using the long string notation::

  dtype('datetime64[us]')

Using the short string notation::

  dtype('M8[us]')

The default is microseconds if no time unit is specified.  Thus, 'M8' is equivalent to 'M8[us]'


Setting and getting values
~~~~~~~~~~~~~~~~~~~~~~~~~~

The objects with this dtype can be set in a series of ways::

  t = numpy.ones(3, dtype='M8[s]')
  t[0] = 1199164176    # assign to July 30th, 2008 at 17:31:00
  t[1] = datetime.datetime(2008, 7, 30, 17, 31, 01) # with datetime module
  t[2] = '2008-07-30T17:31:02'    # with ISO 8601

And can be get in different ways too::

  str(t[0])  -->  2008-07-30T17:31:00
  repr(t[1]) -->  datetime64(1199164177, 's')
  str(t[0].item()) --> 2008-07-30 17:31:00  # datetime module object
  repr(t[0].item()) --> datetime.datetime(2008, 7, 30, 17, 31)  # idem
  str(t)  -->  [2008-07-30T17:31:00  2008-07-30T17:31:01  2008-07-30T17:31:02]
  repr(t)  -->  array([1199164176, 1199164177, 1199164178],
                      dtype='datetime64[s]')

Comparisons
~~~~~~~~~~~

The comparisons will be supported too::

  numpy.array(['1980'], 'M8[Y]') == numpy.array(['1979'], 'M8[Y]')
  --> [False]

or by applying broadcasting::

  numpy.array(['1979', '1980'], 'M8[Y]') == numpy.datetime64('1980', 'Y')
  --> [False, True]

The next should work too::

  numpy.array(['1979', '1980'], 'M8[Y]') == '1980-01-01'
  --> [False, True]

because the right hand expression can be broadcasted into an array of 2
elements of dtype 'M8[Y]'.

Compatibility issues
~~~~~~~~~~~~~~~~~~~~

This will be fully compatible with the ``datetime`` class of the
``datetime`` module of Python only when using a time unit of
microseconds.  For other time units, the conversion process will lose
precision or will overflow as needed.  The conversion from/to a
``datetime`` object doesn't take leap seconds into account.


``timedelta64``
---------------

It represents a time that is relative (i.e. not absolute).  It is
implemented internally as an ``int64`` type.

In time unit *conversions* and time *representations* (but not in other
time computations), the value -2**63 (0x8000000000000000) is interpreted
as an invalid or unknown time, *Not a Time* or *NaT*.  See the section
on time unit conversions for more information.

Time units
~~~~~~~~~~

It accepts different time units, each of them implying a different time
span.  The table below describes the time units supported with their
corresponding time spans.

======== ================ ==========================
      Time unit               Time span
------------------------- --------------------------
  Code       Meaning
======== ================ ==========================
   Y       year             +- 9.2e18 years
   M       month            +- 7.6e17 years
   W       week             +- 1.7e17 years
   B       business day     +- 3.5e16 years
   D       day              +- 2.5e16 years
   h       hour             +- 1.0e15 years
   m       minute           +- 1.7e13 years
   s       second           +- 2.9e12 years
   ms      millisecond      +- 2.9e9 years
   us      microsecond      +- 2.9e6 years
   c#      ticks (100ns)    +- 2.9e4 years
   ns      nanosecond       +- 292 years
   ps      picosecond       +- 106 days
   fs      femtosecond      +- 2.6 hours
   as      attosecond       +- 9.2 seconds
======== ================ ==========================

The value of a time delta is thus *an integer number of units of the
chosen time unit*.

Building a ``timedelta64`` dtype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed ways to specify the time unit in the dtype constructor are:

Using the long string notation::

  dtype('timedelta64[us]')

Using the short string notation::

  dtype('m8[us]')

The default is micro-seconds if no default is specified:  'm8' is equivalent to 'm8[us]'


Setting and getting values
~~~~~~~~~~~~~~~~~~~~~~~~~~

The objects with this dtype can be set in a series of ways::

  t = numpy.ones(3, dtype='m8[ms]')
  t[0] = 12    # assign to 12 ms
  t[1] = datetime.timedelta(0, 0, 13000)   # 13 ms
  t[2] = '0:00:00.014'    # 14 ms

And can be get in different ways too::

  str(t[0])  -->  0:00:00.012
  repr(t[1]) -->  timedelta64(13, 'ms')
  str(t[0].item()) --> 0:00:00.012000   # datetime module object
  repr(t[0].item()) --> datetime.timedelta(0, 0, 12000)  # idem
  str(t)     -->  [0:00:00.012  0:00:00.014  0:00:00.014]
  repr(t)    -->  array([12, 13, 14], dtype="timedelta64[ms]")

Comparisons
~~~~~~~~~~~

The comparisons will be supported too::

  numpy.array([12, 13, 14], 'm8[ms]') == numpy.array([12, 13, 13], 'm8[ms]')
  --> [True, True, False]

or by applying broadcasting::

  numpy.array([12, 13, 14], 'm8[ms]') == numpy.timedelta64(13, 'ms')
  --> [False, True, False]

The next should work too::

  numpy.array([12, 13, 14], 'm8[ms]') == '0:00:00.012'
  --> [True, False, False]

because the right hand expression can be broadcasted into an array of 3
elements of dtype 'm8[ms]'.

Compatibility issues
~~~~~~~~~~~~~~~~~~~~

This will be fully compatible with the ``timedelta`` class of the
``datetime`` module of Python only when using a time unit of
microseconds.  For other units, the conversion process will lose
precision or will overflow as needed.


Examples of use
===============

Here it is an example of use for the ``datetime64``::

  In [5]: numpy.datetime64(42, 'us')
  Out[5]: datetime64(42, 'us')

  In [6]: print numpy.datetime64(42, 'us')
  1970-01-01T00:00:00.000042  # representation in ISO 8601 format

  In [7]: print numpy.datetime64(367.7, 'D')  # decimal part is lost
  1971-01-02  # still ISO 8601 format

  In [8]: numpy.datetime('2008-07-18T12:23:18', 'm')  # from ISO 8601
  Out[8]: datetime64(20273063, 'm')

  In [9]: print numpy.datetime('2008-07-18T12:23:18', 'm')
  Out[9]: 2008-07-18T12:23

  In [10]: t = numpy.zeros(5, dtype="datetime64[ms]")

  In [11]: t[0] = datetime.datetime.now()  # setter in action

  In [12]: print t
  [2008-07-16T13:39:25.315  1970-01-01T00:00:00.000
   1970-01-01T00:00:00.000  1970-01-01T00:00:00.000
   1970-01-01T00:00:00.000]

  In [13]: repr(t)
  Out[13]: array([267859210457, 0, 0, 0, 0], dtype="datetime64[ms]")

  In [14]: t[0].item()     # getter in action
  Out[14]: datetime.datetime(2008, 7, 16, 13, 39, 25, 315000)

  In [15]: print t.dtype
  dtype('datetime64[ms]')

And here it goes an example of use for the ``timedelta64``::

  In [5]: numpy.timedelta64(10, 'us')
  Out[5]: timedelta64(10, 'us')

  In [6]: print numpy.timedelta64(10, 'us')
  0:00:00.000010

  In [7]: print numpy.timedelta64(3600.2, 'm')  # decimal part is lost
  2 days, 12:00

  In [8]: t1 = numpy.zeros(5, dtype="datetime64[ms]")

  In [9]: t2 = numpy.ones(5, dtype="datetime64[ms]")

  In [10]: t = t2 - t1

  In [11]: t[0] = datetime.timedelta(0, 24)  # setter in action

  In [12]: print t
  [0:00:24.000  0:00:01.000  0:00:01.000  0:00:01.000  0:00:01.000]

  In [13]: print repr(t)
  Out[13]: array([24000, 1, 1, 1, 1], dtype="timedelta64[ms]")

  In [14]: t[0].item()     # getter in action
  Out[14]: datetime.timedelta(0, 24)

  In [15]: print t.dtype
  dtype('timedelta64[s]')


Operating with date/time arrays
===============================

``datetime64`` vs ``datetime64``
--------------------------------

The only arithmetic operation allowed between absolute dates is the
subtraction::

  In [10]: numpy.ones(3, "M8[s]") - numpy.zeros(3, "M8[s]")
  Out[10]: array([1, 1, 1], dtype=timedelta64[s])

But not other operations::

  In [11]: numpy.ones(3, "M8[s]") + numpy.zeros(3, "M8[s]")
  TypeError: unsupported operand type(s) for +: 'numpy.ndarray' and 'numpy.ndarray'

Comparisons between absolute dates are allowed.

Casting rules
~~~~~~~~~~~~~

When operating (basically, only the subtraction will be allowed) two
absolute times with different unit times, the outcome would be to raise
an exception.  This is because the ranges and time-spans of the different
time units can be very different, and it is not clear at all what time
unit will be preferred for the user.  For example, this should be
allowed::

  >>> numpy.ones(3, dtype="M8[Y]") - numpy.zeros(3, dtype="M8[Y]")
  array([1, 1, 1], dtype="timedelta64[Y]")

But the next should not::

  >>> numpy.ones(3, dtype="M8[Y]") - numpy.zeros(3, dtype="M8[ns]")
  raise numpy.IncompatibleUnitError  # what unit to choose?


``datetime64`` vs ``timedelta64``
---------------------------------

It will be possible to add and subtract relative times from absolute
dates::

  In [10]: numpy.zeros(5, "M8[Y]") + numpy.ones(5, "m8[Y]")
  Out[10]: array([1971, 1971, 1971, 1971, 1971], dtype=datetime64[Y])

  In [11]: numpy.ones(5, "M8[Y]") - 2 * numpy.ones(5, "m8[Y]")
  Out[11]: array([1969, 1969, 1969, 1969, 1969], dtype=datetime64[Y])

But not other operations::

  In [12]: numpy.ones(5, "M8[Y]") * numpy.ones(5, "m8[Y]")
  TypeError: unsupported operand type(s) for *: 'numpy.ndarray' and 'numpy.ndarray'

Casting rules
~~~~~~~~~~~~~

In this case the absolute time should have priority for determining the
time unit of the outcome.  That would represent what the people wants to
do most of the times.  For example, this would allow to do::

  >>> series = numpy.array(['1970-01-01', '1970-02-01', '1970-09-01'],
  dtype='datetime64[D]')
  >>> series2 = series + numpy.timedelta(1, 'Y')  # Add 2 relative years
  >>> series2
  array(['1972-01-01', '1972-02-01', '1972-09-01'],
  dtype='datetime64[D]')  # the 'D'ay time unit has been chosen


``timedelta64`` vs ``timedelta64``
----------------------------------

Finally, it will be possible to operate with relative times as if they
were regular int64 dtypes *as long as* the result can be converted back
into a ``timedelta64``::

  In [10]: numpy.ones(3, 'm8[us]')
  Out[10]: array([1, 1, 1], dtype="timedelta64[us]")

  In [11]: (numpy.ones(3, 'm8[M]') + 2) ** 3
  Out[11]: array([27, 27, 27], dtype="timedelta64[M]")

But::

  In [12]: numpy.ones(5, 'm8') + 1j
  TypeError: the result cannot be converted into a ``timedelta64``

Casting rules
~~~~~~~~~~~~~

When combining two ``timedelta64`` dtypes with different time units the
outcome will be the shorter of both ("keep the precision" rule).  For
example::

  In [10]: numpy.ones(3, 'm8[s]') + numpy.ones(3, 'm8[m]')
  Out[10]: array([61, 61, 61],  dtype="timedelta64[s]")

However, due to the impossibility to know the exact duration of a
relative year or a relative month, when these time units appear in one
of the operands, the operation will not be allowed::

  In [11]: numpy.ones(3, 'm8[Y]') + numpy.ones(3, 'm8[D]')
  raise numpy.IncompatibleUnitError  # how to convert relative years to days?

In order to being able to perform the above operation a new NumPy
function, called ``change_timeunit`` is proposed.  Its signature will
be::

  change_timeunit(time_object, new_unit, reference)

where 'time_object' is the time object whose unit is to be changed,
'new_unit' is the desired new time unit, and 'reference' is an absolute
date (NumPy datetime64 scalar) that will be used to allow the conversion
of relative times in case of using time units with an uncertain number
of smaller time units (relative years or months cannot be expressed in
days).

With this, the above operation can be done as follows::

  In [10]: t_years = numpy.ones(3, 'm8[Y]')

  In [11]: t_days = numpy.change_timeunit(t_years, 'D', '2001-01-01')

  In [12]: t_days + numpy.ones(3, 'm8[D]')
  Out[12]: array([366, 366, 366],  dtype="timedelta64[D]")


dtype vs time units conversions
===============================

For changing the date/time dtype of an existing array, we propose to use
the ``.astype()`` method.  This will be mainly useful for changing time
units.

For example, for absolute dates::

  In[10]: t1 = numpy.zeros(5, dtype="datetime64[s]")

  In[11]: print t1
  [1970-01-01T00:00:00  1970-01-01T00:00:00  1970-01-01T00:00:00
   1970-01-01T00:00:00  1970-01-01T00:00:00]

  In[12]: print t1.astype('datetime64[D]')
  [1970-01-01  1970-01-01  1970-01-01  1970-01-01  1970-01-01]

For relative times::

  In[10]: t1 = numpy.ones(5, dtype="timedelta64[s]")

  In[11]: print t1
  [1 1 1 1 1]

  In[12]: print t1.astype('timedelta64[ms]')
  [1000 1000 1000 1000 1000]

Changing directly from/to relative to/from absolute dtypes will not be
supported::

  In[13]: numpy.zeros(5, dtype="datetime64[s]").astype('timedelta64')
  TypeError: data type cannot be converted to the desired type

Business days have the peculiarity that they do not cover a continuous
line of time (they have gaps at weekends).  Thus, when converting from
any ordinary time to business days, it can happen that the original time
is not representable.  In that case, the result of the conversion is
*Not a Time* (*NaT*)::

  In[10]: t1 = numpy.arange(5, dtype="datetime64[D]")

  In[11]: print t1
  [1970-01-01  1970-01-02  1970-01-03  1970-01-04  1970-01-05]

  In[12]: t2 = t1.astype("datetime64[B]")

  In[13]: print t2  # 1970 begins in a Thursday
  [1970-01-01  1970-01-02  NaT  NaT  1970-01-05]

When converting back to ordinary days, NaT values are left untouched
(this happens in all time unit conversions)::

  In[14]: t3 = t2.astype("datetime64[D]")

  In[13]: print t3
  [1970-01-01  1970-01-02  NaT  NaT  1970-01-05]


Final considerations
====================

Why the ``origin`` metadata disappeared
---------------------------------------

During the discussion of the date/time dtypes in the NumPy list, the
idea of having an ``origin`` metadata that complemented the definition
of the absolute ``datetime64`` was initially found to be useful.

However, after thinking more about this, we found that the combination
of an absolute ``datetime64`` with a relative ``timedelta64`` does offer
the same functionality while removing the need for the additional
``origin`` metadata.  This is why we have removed it from this proposal.

Operations with mixed time units
--------------------------------

Whenever an operation between two time values of the same dtype with the
same unit is accepted, the same operation with time values of different
units should be possible (e.g. adding a time delta in seconds and one in
microseconds), resulting in an adequate time unit.  The exact semantics
of this kind of operations is defined int the "Casting rules"
subsections of the "Operating with date/time arrays" section.

Due to the peculiarities of business days, it is most probable that
operations mixing business days with other time units will not be
allowed.

Why there is not a ``quarter`` time unit?
-----------------------------------------

This proposal tries to focus on the most common used set of time units
to operate with, and the ``quarter`` can be considered more of a derived
unit.  Besides, the use of a ``quarter`` normally requires that it can
start at whatever month of the year, and as we are not including support
for a time ``origin`` metadata, this is not a viable venue here.
Finally, if we were to add the ``quarter`` then people should expect to
find a ``biweekly``, ``semester`` or ``biyearly`` just to put some
examples of other derived units, and we find this a bit too overwhelming
for this proposal purposes.


.. [1] https://docs.python.org/library/datetime.html
.. [2] https://www.egenix.com/products/python/mxBase/mxDateTime
.. [3] https://en.wikipedia.org/wiki/Unix_time


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
