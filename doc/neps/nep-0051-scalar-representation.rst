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

Motivation and Scope
====================

This NEP proposes to change the representation of the following
NumPy scalars types to distinguish them from the Python scalars:

* ``np.bool_``
* ``np.uint8``, ``np.int8``, and all other integer scalars
* ``np.float16``, ``np.float32``, ``np.float64``, ``np.longdouble``
* ``np.complex64``, ``np.complex128``, ``np.clongdouble``
* ``np.str_``, ``np.bytes_``

Additionally, these representation will be slightly modified to align
with the new scheme to use the ``np`` prefix consistently:

* ``np.datetime64`` and ``np.timedelta64``
* ``np.void``

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
This differences will be exacerbated when adopting :ref:`NEP 50 <NEP50>`
because the lower precision NumPy scalar will be preserved more often.
Even ``float64``, which is very similar to Python's ``float`` and inherits
from it, does behave differently for example when dividing by zero.

A common confusion are also the NumPy booleans.  Python programmers
somtimes write ``obj is True`` and will surprised when an object that shows
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

to allow round-tripping.  Addtionally to this change, ``float128`` will
now always be printed as ``longdouble`` since the old name gives a wrong
impression of precision.

Backward compatibility
======================

We expect that most workflows will not be affected as only printing
changes.  In general we believe that informing users about the type
they are working with outweighs the need for adapting printing in
some instances.

The NumPy test suite includes code such as ``decimal.Decimal(repr(scalar))``.
This code needs to be modified to use the ```str()``.

.. admonition:: TODO

    If there is a lot of similar code, we may need a better solution than
    ``str()``, since it is plausible that ``str()`` should honor

An exception to this are downstream libraries with documentation and
especially documentation testing.
Since the representation of many values will change, in many cases
the documentation will have to be updated to use the new representation.
This is expected to require larger documentation fixups.

Further, it may be necessary to adept tools for doctest testing to
allow approximate value checking for the new representation.

.. admonition:: TODO

    While astropy's `pytest-doctestplus <https://github.com/astropy/pytest-doctestplus>`_
    itself should not be affected by this change, if other checkers are
    affected, these should be given a chance to release a new version
    before NumPy goes ahead with the change.

    Testing should happen before the final release to see how certain
    large downstream libraries (SciPy, astropy, etc.) are affected.
    These are expected to require larger changes to their documentation.


Detailed description
====================

This NEP proposes to change the represenatation for NumPy scalars to:

* ``np.True_`` and ``np.False_`` for booleans
* ``np.scalar(<value>)``, i.e. ``np.float64(3.0)`` for all numerical dtypes.
* The value for ``np.longdouble`` and ``np.clongdouble`` will be given in quotes:
  ``np.longdouble('3.0')``.  This ensures that it can always roundtrip correctly
  and matches ``decimal.Decimal``.
  Further, for these two the size based name such as ``float128`` will not
  be adopted, as it is platform dependend and imprecise.
* ``np.str_("string")`` and ``np.bytes_(b"byte_string")`` for string dtypes.

Where booleans are printed as their singletons since this is more concise.
For strings we include the ``np.`` as ``str_`` and ``bytes_`` on their
own may not be sufficient to indicate NumPy involvement.

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

Integer scalar type name and instance represenatation
-----------------------------------------------------

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

Implementation
==============

The new representations can be implemented on the scalar types.
Additional work may be necessary to ensure that the changes do not affect
array representation as well.

A large part of the implementation work has already been done by Ganesh
Kathiresan in the open draft PR [2]_.

Alternatives
============

Different representation could be discussed, main alternatives are spelling
``np.`` as ``numpy.`` or dropping the ``np.`` part from the numerical scalars.
We believe that using ``np.`` is sufficiently clear, concise, and does allow
copy pasting the representation.
Using only ``float64(3.0)`` without the ``np.`` prefix is more concise but
contexts may exists where the NumPy dependency is not fully clear and the name
could clash with other libraries.
of ``numpy`` or ``np`` for the numerical types to give for example
``np.float64(3.0)``.

For booleans an alternative would be to use ``np.bool_(True)`` or ``bool_(True)``.
However, NumPy boolean scalars are singletons and the proposed formatting is more
concise.  Alternatives for booleans were also discussed previously in [1]_.

For the string scalars, the confusion is generally less pronounced.  It may be
reasonable to defer changing these.


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
