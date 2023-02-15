.. _NEP51:

=====================================================
NEP 51 — Changing the representation of NumPy scalars
=====================================================
:Author: Sebastian Berg
:Status: Draft
:Type: Standards Track
:Created: 2022-09-13


Abstract
========

NumPy has scalar objects ("NumPy scalar") representing a single value
corresponding to a NumPy DType.  The representation of these currently
matches that of the Python builtins, giving::

    >>> np.float32(3.0)
    3.0

In this NEP we propose to change the representation to include the
NumPy scalar type information.  Changing the above example to::

    >>> np.float32(3.0)
    np.float32(3.0)

We expect that this change will help users distinguish the NumPy scalars
from the Python builtin types and clarify their behavior.

The distinction between NumPy scalars and Python builtins will further become
more important for users once :ref:`NEP 50 <NEP50>` is adopted.

These changes do lead to smaller incompatible and infrastructure changes
related to array printing.

Motivation and Scope
====================

This NEP proposes to change the representation of the following
NumPy scalars types to distinguish them from the Python scalars:

* ``np.bool_``
* ``np.uint8``, ``np.int8``, and all other integer scalars
* ``np.float16``, ``np.float32``, ``np.float64``, ``np.longdouble``
* ``np.complex64``, ``np.complex128``, ``np.clongdouble``
* ``np.str_``, ``np.bytes_``
* ``np.void``  (structured dtypes)

Additionally, the representation of the remaining NumPy scalars will be
adapted to print as ``np.`` rather than ``numpy.``:

* ``np.datetime64`` and ``np.timedelta64``
* ``np.void``  (unstructured version)

The NEP does not propose to change how these scalars print – only
their representation (``__repr__``) will be changed.
Further, array representation will not be affected since it already
includes the ``dtype=`` when necessary.

The main motivation behind the change is that the Python numerical types
behave differently from the NumPy scalars.
For example numbers with lower precision (e.g. ``uint8`` or ``float16``)
should be used with care and users should be aware when they are working
with them.  All NumPy integers can experience overflow which Python integers
do not.
These differences will be exacerbated when adopting :ref:`NEP 50 <NEP50>`
because the lower precision NumPy scalar will be preserved more often.
Even ``np.float64``, which is very similar to Python's ``float`` and inherits
from it, does behave differently for example when dividing by zero.

Another common source of confusion are the NumPy booleans.  Python programmers
sometimes write ``obj is True`` and will surprised when an object that shows
as ``True`` fails to pass the test.
It is much easier to understand this behavior when the value is
shown as ``np.True_``.

Not only do we expect the change to help users better understand and be
reminded of the differences between NumPy and Python scalars, but we also
believe that the awareness will greatly help debugging.

Usage and Impact
================

Most user code should not be impacted by the change, but users will now
often see NumPy values shown as::

    np.True_
    np.float64(3.0)
    np.int64(34)

and so on.  This will also mean that documentation and output in
Jupyter notebook cells will often show the type information intact.

``np.longdouble`` and ``np.clongdouble`` will print with single quotes::

    np.longdouble('3.0')

to allow round-tripping.  Additionally to this change, ``float128`` will
now always be printed as ``longdouble`` since the old name gives a false
impression of precision.

Backward compatibility
======================

We expect that most workflows will not be affected as only printing
changes.  In general we believe that informing users about the type
they are working with outweighs the need for adapting printing in
some instances.

The NumPy test suite includes code such as ``decimal.Decimal(repr(scalar))``.
This code needs to be modified to use the ``str()``.

An exception to this are downstream libraries with documentation and
especially documentation testing.
Since the representation of many values will change, in many cases
the documentation will have to be updated.
This is expected to require larger documentation fixups in the mid-term.

It may be necessary to adopt tools for doctest testing to
allow approximate value checking for the new representation.

Changes to ``arr.tofile()``
---------------------------
``arr.tofile()`` currently stores values as ``repr(arr.item())`` when in text
mode.  This is not always ideal since that may include a conversion to
Python.
One issue is that this would start saving longdouble as
``np.longdouble('3.1')`` which is clearly not desired.  We expect that this
method is rarely used for object arrays.  For string arrays, using the ``repr``
also leads to storing ``"string"`` or ``b"string"`` which seems rarely desired.

The proposal is to change the default (back) to use ``str`` rather than
``repr``.  If ``repr`` is desired, users will have to pass ``fmt=%r``.


Detailed description
====================

This NEP proposes to change the representation for NumPy scalars to:

* ``np.True_`` and ``np.False_`` for booleans (their singleton instances)
* ``np.scalar(<value>)``, i.e. ``np.float64(3.0)`` for all numerical dtypes.
* The value for ``np.longdouble`` and ``np.clongdouble`` will be given in quotes:
  ``np.longdouble('3.0')``.  This ensures that it can always roundtrip correctly
  and matches the way that ``decimal.Decimal`` behaves.
  For these two the size-based name such as ``float128`` will not be used
  as the actual size is platform-dependent and therefore misleading.
* ``np.str_("string")`` and ``np.bytes_(b"byte_string")`` for string dtypes.
* ``np.void((3, 5), dtype=[('a', '<i8'), ('b', 'u1')])`` (similar to arrays)
  for structured types. This will be valid syntax to recreate the scalar.

Unlike arrays, the scalar representation should round-trip correctly, so
longdouble values will be quoted and other values never be truncated.

In some places (i.e. masked arrays, void and record scalars) we will want to
print the representation without the type.  For example::

    np.void(('3.0',), dtype=[('a', 'f16')])  # longdouble

should print the 3.0 with quotes (to ensure round-tripping), but not repeat
the full ``np.longdouble('3.0')`` as the dtype includes the longdouble
information.
To allow this, a new semi-public ``np.core.array_print.get_formatter()`` will
be introduced to expand the current functionality (see Implementation).

Effects on Masked Arrays and Records
------------------------------------
Some other parts of NumPy will indirectly be changed.  Masked arrays
``fill_value`` will be adapted to only include the full scalar information
such as ``fill_value=np.float64(1e20)`` when the dtype of the array
mismatches.
For longdouble (with matching dtype), it will be printed as
``fill_value='3.1'`` including the quotes which (in principle but likely not
in practice) ensure round-tripping.
It should be noted that for strings it is typical for the dtypes to mismatch
in the string length.  So that strings will usually be printed as
``np.str_("N/A")``.

The ``np.record`` scalar will be aligned with ``np.void`` and print identically
to it (except the name itself).  For example as:
``np.record((3, 5), dtype=[('a', '<i8'), ('b', 'u1')])``

Details about ``longdouble`` and ``clongdouble``
------------------------------------------------

For ``longdouble`` and ``clongdouble`` values such as::

    np.sqrt(np.longdouble(2.))

may not roundtrip unless quoted as strings (as the conversion to a Python float
would lose precision).  This NEP proposes to use a single quote similar to
Python's decimal which prints as ``Decimal('3.0')``

``longdouble`` can have different precision and storage sizes varying from
8 to 16 bytes.  However, even if ``float128`` is correct because the number
is stored as 128 bits, it normally does not have 128 bit precision.
(``clongdouble`` is the same, but with twice the storage size.)

This NEP thus includes the proposal of changing the name of ``longdouble``
to always print as ``longdouble`` and never ``float128`` or ``float96``.
It does not include deprecating the ``np.float128`` alias.
However, such a deprecation may occur independently of the NEP.

Integer scalar type name and instance representation
----------------------------------------------------

One detail is that due to NumPy scalar types being based on the C types,
NumPy sometimes distinguishes them, for example on most 64 bit systems
(not windows)::

     >>> np.longlong
     numpy.longlong
     >>> np.longlong(3)
     np.int64(3)

The proposal will lead to the ``longlong`` name for the type while
using the ``int64`` form for the scalar.
This choice is made since ``int64`` is generally the more useful
information for users, but the type name itself must be precise.


Related Work
============

A PR to only change the representation of booleans was previously
made `here <https://github.com/numpy/numpy/pull/17592>`_.

The implementation is (at the time of writing) largely finished and can be
found `here <https://github.com/numpy/numpy/pull/22449>`_

Implementation
==============

The new representations can be mostly implemented on the scalar types with
the largest changes needed in the test suite.

The proposed changes for void scalars and masked ``fill_value`` makes it
necessary to expose the scalar representation without the type.

We propose introducing the semi-public API::

    np.core.arrayprint.get_formatter(*,
            data=None, dtype=None, fmt=None, options=None)

to replace the current internal ``_get_formatting_func``.  This will allow
two things compared to the old function:

* ``data`` may be ``None`` (if ``dtype`` is passed) allowing to not pass
  multiple values that will be printed/formatted later.
* ``fmt=`` will allow passing on format strings to a DType-specific element
  formatter in the future.  For now, ``get_formatter()`` will accept
  ``repr`` or ``str`` (the singletons not strings) to format the elements
  without type information (``'3.1'`` rather than ``np.longdouble('3.1')``).
  The implementation ensures that formatting matches except for the type
  information.

  The empty format string will print identically to ``str()`` (with possibly
  extra padding when data is passed).

``get_formatter()`` is expected to query a user DType's method in the future
allowing customized formatting for all DTypes.

Making ``get_formatter`` public allows it to be used for ``np.record`` and
masked arrays.
Currently, the formatters themselves seem semi-public; using a single
entry-point will hopefully provide a clear API for formatting NumPy values.

The large part for the scalar representation changes had previously been done
by Ganesh Kathiresan in [2]_.

Alternatives
============

Different representations can be considered: alternatives include spelling
``np.`` as ``numpy.`` or dropping the ``np.`` part from the numerical scalars.
We believe that using ``np.`` is sufficiently clear, concise, and does allow
copy pasting the representation.
Using only ``float64(3.0)`` without the ``np.`` prefix is more concise but
contexts may exists where the NumPy dependency is not fully clear and the name
could clash with other libraries.

For booleans an alternative would be to use ``np.bool_(True)`` or ``bool_(True)``.
However, NumPy boolean scalars are singletons and the proposed formatting is more
concise.  Alternatives for booleans were also discussed previously in [1]_.

For the string scalars, the confusion is generally less pronounced.  It may be
reasonable to defer changing these.

Non-finite values
-----------------
The proposal does not allow copy pasting ``nan`` and ``inf`` values.
They could be represented by ``np.float64('nan')`` or ``np.float64(np.nan)``
instead.
This is more concise and Python also uses ``nan`` and ``inf`` rather than
allowing copy-pasting by showing it as ``float('nan')``.  Arguably, it would be
a smaller addition in NumPy, where the will already be always printed.

Alternatives for the new ``get_formatter()``
--------------------------------------------
When ``fmt=`` is passed, and specifically for the main use (in this NEP) to
format to a ``repr`` or ``str``.
It would also be possible to use a ufunc or a direct formatting function
rather than wrapping it into a ```get_formatter()`` which relies on
instantiating a formatter class for the DType.

This NEP does not preclude creating a ufunc or making a special path.
However, NumPy array formatting commonly looks at all values to be formatted
in order to add padding for alignment or give uniform exponential output.
In this case ``data=`` is passed and used in preparation.  This form of
formatting (unlike the scalar case where ``data=None`` would be desired) is
unfortunately fundamentally incompatible with UFuncs.

The use of the singleton ``str`` and ``repr`` ensures that future formatting
strings like ``f"{arr:r}"`` are not in any way limited by using ``"r"`` or
``"s"`` instead.

Discussion
==========

* An initial discussion on this changed happened in the mailing list:
  https://mail.python.org/archives/list/numpy-discussion@python.org/thread/7GLGFHTZHJ6KQPOLMVY64OM6IC6KVMYI/
* There was a previous issue [1]_ and PR [2]_ to change only the
  representation of the NumPy booleans.  The PR was later updated to change
  the representation of all (or at least most) NumPy scalars.


References and Footnotes
========================

.. [1] https://github.com/numpy/numpy/issues/12950
.. [2] https://github.com/numpy/numpy/pull/17592

Copyright
=========

This document has been placed in the public domain.
