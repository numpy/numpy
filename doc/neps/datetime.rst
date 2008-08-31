====================================================================
 A (second) proposal for implementing some date/time types in NumPy
====================================================================

:Author: Francesc Alted i Abad
:Contact: faltet@pytables.com
:Author: Ivan Vilata i Balaguer
:Contact: ivan@selidor.net
:Date: 2008-07-16


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
pondering about different possibilities, we have stick with *two*
different types, namely ``datetime64`` and ``timedelta64`` (these names
are preliminary and can be changed), that can have different resolutions
so as to cover different needs.

**Important note:** the resolution is conceived here as a metadata that
  *complements* a date/time dtype, *without changing the base type*.

Now it goes a detailed description of the proposed types.


``datetime64``
--------------

It represents a time that is absolute (i.e. not relative).  It is
implemented internally as an ``int64`` type.  The internal epoch is
POSIX epoch (see [3]_).

Resolution
~~~~~~~~~~

It accepts different resolutions and for each of these resolutions, it
will support different time spans.  The table below describes the
resolutions supported with its corresponding time spans.

+----------------------+----------------------------------+
|     Resolution       |         Time span (years)        |
+----------------------+----------------------------------+
|  Code |   Meaning    |                                  |
+======================+==================================+
|   Y   |  year        |      [9.2e18 BC, 9.2e18 AC]      |
|   Q   |  quarter     |      [3.0e18 BC, 3.0e18 AC]      |
|   M   |  month       |      [7.6e17 BC, 7.6e17 AC]      |
|   W   |  week        |      [1.7e17 BC, 1.7e17 AC]      |
|   d   |  day         |      [2.5e16 BC, 2.5e16 AC]      |
|   h   |  hour        |      [1.0e15 BC, 1.0e15 AC]      |
|   m   |  minute      |      [1.7e13 BC, 1.7e13 AC]      |
|   s   |  second      |      [ 2.9e9 BC,  2.9e9 AC]      |
|   ms  |  millisecond |      [ 2.9e6 BC,  2.9e6 AC]      |
|   us  |  microsecond |      [290301 BC, 294241 AC]      |
|   ns  |  nanosecond  |      [  1678 AC,   2262 AC]      |
+----------------------+----------------------------------+

Building a ``datetime64`` dtype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed way to specify the resolution in the dtype constructor
is:

Using parameters in the constructor::

  dtype('datetime64', res="us")  # the default res. is microseconds

Using the long string notation::

  dtype('datetime64[us]')   # equivalent to dtype('datetime64')

Using the short string notation::

  dtype('T8[us]')   # equivalent to dtype('T8')

Compatibility issues
~~~~~~~~~~~~~~~~~~~~

This will be fully compatible with the ``datetime`` class of the
``datetime`` module of Python only when using a resolution of
microseconds.  For other resolutions, the conversion process will
loose precision or will overflow as needed.


``timedelta64``
---------------

It represents a time that is relative (i.e. not absolute).  It is
implemented internally as an ``int64`` type.

Resolution
~~~~~~~~~~

It accepts different resolutions and for each of these resolutions, it
will support different time spans.  The table below describes the
resolutions supported with its corresponding time spans.

+----------------------+--------------------------+
|     Resolution       |         Time span        |
+----------------------+--------------------------+
|  Code |   Meaning    |                          |
+======================+==========================+
|   W   |  week        |      +- 1.7e17 years     |
|   D   |  day         |      +- 2.5e16 years     |
|   h   |  hour        |      +- 1.0e15 years     |
|   m   |  minute      |      +- 1.7e13 years     |
|   s   |  second      |      +- 2.9e12 years     |
|   ms  |  millisecond |      +- 2.9e9 years      |
|   us  |  microsecond |      +- 2.9e6 years      |
|   ns  |  nanosecond  |      +- 292 years        |
|   ps  |  picosecond  |      +- 106 days         |
|   fs  |  femtosecond |      +- 2.6 hours        |
|   as  |  attosecond  |      +- 9.2 seconds      |
+----------------------+--------------------------+

Building a ``timedelta64`` dtype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed way to specify the resolution in the dtype constructor
is:

Using parameters in the constructor::

  dtype('timedelta64', res="us")  # the default res. is microseconds

Using the long string notation::

  dtype('timedelta64[us]')   # equivalent to dtype('datetime64')

Using the short string notation::

  dtype('t8[us]')   # equivalent to dtype('t8')

Compatibility issues
~~~~~~~~~~~~~~~~~~~~

This will be fully compatible with the ``timedelta`` class of the
``datetime`` module of Python only when using a resolution of
microseconds.  For other resolutions, the conversion process will
loose precision or will overflow as needed.


Example of use
==============

Here it is an example of use for the ``datetime64``::

  In [10]: t = numpy.zeros(5, dtype="datetime64[ms]")

  In [11]: t[0] = datetime.datetime.now()  # setter in action

  In [12]: t[0]
  Out[12]: '2008-07-16T13:39:25.315'   # representation in ISO 8601 format

  In [13]: print t
  [2008-07-16T13:39:25.315  1970-01-01T00:00:00.0
  1970-01-01T00:00:00.0  1970-01-01T00:00:00.0  1970-01-01T00:00:00.0]

  In [14]: t[0].item()     # getter in action
  Out[14]: datetime.datetime(2008, 7, 16, 13, 39, 25, 315000)

  In [15]: print t.dtype
  datetime64[ms]

And here it goes an example of use for the ``timedelta64``::

  In [8]: t1 = numpy.zeros(5, dtype="datetime64[s]")

  In [9]: t2 = numpy.ones(5, dtype="datetime64[s]")

  In [10]: t = t2 - t1

  In [11]: t[0] = 24  # setter in action (setting to 24 seconds)

  In [12]: t[0]
  Out[12]: 24       # representation as an int64

  In [13]: print t
  [24  1  1  1  1]

  In [14]: t[0].item()     # getter in action
  Out[14]: datetime.timedelta(0, 24)

  In [15]: print t.dtype
  timedelta64[s]


Operating with date/time arrays
===============================

``datetime64`` vs ``datetime64``
--------------------------------

The only operation allowed between absolute dates is the subtraction::

  In [10]: numpy.ones(5, "T8") - numpy.zeros(5, "T8")
  Out[10]: array([1, 1, 1, 1, 1], dtype=timedelta64[us])

But not other operations::

  In [11]: numpy.ones(5, "T8") + numpy.zeros(5, "T8")
  TypeError: unsupported operand type(s) for +: 'numpy.ndarray' and 'numpy.ndarray'

``datetime64`` vs ``timedelta64``
---------------------------------

It will be possible to add and subtract relative times from absolute
dates::

  In [10]: numpy.zeros(5, "T8[Y]") + numpy.ones(5, "t8[Y]")
  Out[10]: array([1971, 1971, 1971, 1971, 1971], dtype=datetime64[Y])

  In [11]: numpy.ones(5, "T8[Y]") - 2 * numpy.ones(5, "t8[Y]")
  Out[11]: array([1969, 1969, 1969, 1969, 1969], dtype=datetime64[Y])

But not other operations::

  In [12]: numpy.ones(5, "T8[Y]") * numpy.ones(5, "t8[Y]")
  TypeError: unsupported operand type(s) for *: 'numpy.ndarray' and 'numpy.ndarray'

``timedelta64`` vs anything
---------------------------

Finally, it will be possible to operate with relative times as if they
were regular int64 dtypes *as long as* the result can be converted back
into a ``timedelta64``::

  In [10]: numpy.ones(5, 't8')
  Out[10]: array([1, 1, 1, 1, 1], dtype=timedelta64[us])

  In [11]: (numpy.ones(5, 't8[M]') + 2) ** 3
  Out[11]: array([27, 27, 27, 27, 27], dtype=timedelta64[M])

But::

  In [12]: numpy.ones(5, 't8') + 1j
  TypeError: The result cannot be converted into a ``timedelta64``


dtype/resolution conversions
============================

For changing the date/time dtype of an existing array, we propose to use
the ``.astype()`` method.  This will be mainly useful for changing
resolutions.

For example, for absolute dates::

  In[10]: t1 = numpy.zeros(5, dtype="datetime64[s]")

  In[11]: print t1
  [1970-01-01T00:00:00  1970-01-01T00:00:00  1970-01-01T00:00:00
   1970-01-01T00:00:00  1970-01-01T00:00:00]

  In[12]: print t1.astype('datetime64[d]')
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


Final considerations
====================

Why the ``origin`` metadata disappeared
---------------------------------------

During the discussion of the date/time dtypes in the NumPy list, the
idea of having an ``origin`` metadata that complemented the definition
of the absolute ``datetime64`` was initially found to be useful.

However, after thinking more about this, Ivan and me find that the
combination of an absolute ``datetime64`` with a relative
``timedelta64`` does offer the same functionality while removing the
need for the additional ``origin`` metadata.  This is why we have
removed it from this proposal.


Resolution and dtype issues
---------------------------

The date/time dtype's resolution metadata cannot be used in general as
part of typical dtype usage.  For example, in::

  numpy.zeros(5, dtype=numpy.datetime64)

we have to found yet a sensible way to pass the resolution.  Perhaps the
next would work::

  numpy.zeros(5, dtype=numpy.datetime64(res='Y'))

but we are not sure if this would collide with the spirit of the NumPy
dtypes.

At any rate, one can always do::

  numpy.zeros(5, dtype=numpy.dtype('datetime64', res='Y'))

BTW, prior to all of this, one should also elucidate whether::

  numpy.dtype('datetime64', res='Y')

or::

   numpy.dtype('datetime64[Y]')
   numpy.dtype('T8[Y]')

would be a consistent way to instantiate a dtype in NumPy.  We do really
think that could be a good way, but we would need to hear the opinion of
the expert.  Travis?



.. [1] http://docs.python.org/lib/module-datetime.html
.. [2] http://www.egenix.com/products/python/mxBase/mxDateTime
.. [3] http://en.wikipedia.org/wiki/Unix_time


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:

