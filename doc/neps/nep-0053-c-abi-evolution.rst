.. _NEP53:

===============================================
NEP 53 — Evolving the NumPy C-API for NumPy 2.0
===============================================

:Author: Sebastian Berg <sebastianb@nvidia.com>
:Status: Draft
:Type: Standard
:Created: 2022-04-10

Abstract
========

The NumPy C-API is used in downstream projects (often through Cython)
to extend NumPy functionality.  Supporting these packages generally means
that it is slow to evolve our C-API and some changes are not possible in a
normal NumPy release because NumPy must guarantee backwards compatibility:
A downstream package compiled against an old NumPy version (e.g. 1.17)
will generally work with a new NumPy version (e.g. 1.25).

A NumPy 2.0 release allows to *partially* break this promise:
We can accept that a SciPy version compiled with NumPy 1.17 (e.g. SciPy 1.10)
will *not* work with NumPy 2.0.
However, it must still be easy to create a single SciPy binary that is
compatible with both NumPy 1.x and NumPy 2.0.

Given these constraints this NEP outlines a path forward to allow large changes
to our C-API.  Similar to Python API changes proposed for NumPy 2.0 the NEP
aims to allow changes to an extend that *most* downstream packages are expected
to need no or only minor code changes.

The implementation of this NEP consists would consist of two steps:
1. As part of a general improvement, starting with NumPy 1.25 building with
   NumPy will by default export an older API version to allow backwards
   compatible builds with the newest available NumPy version.
   (New API is not available unless opted-in.)
2. The NumPy 2.0 will:

   * require recompilation of downstream packages against NumPy 2.0 to be
     compatible with NumPy 2.0.
   * need a ``numpy2_compat`` as a dependency when running on NumPy 1.x.
   * require some downstream code changes to adapt to changed API.


Motivation and Scope
====================

The NumPy API conists of more than 300 functions and numerous macros.
Many of these are outdated: some were only ever used within NumPy,
exist only for compatibility with NumPy's predecessors, or have no or only
a single known downstream user (i.e. SciPy).

Further, many structs used by NumPy have always been public making it
impossible to change them outside of a major release.
Some changes have been planned for years and were the reason for
``NPY_NO_DEPRECATED_API`` and further deprecations as explained in
:ref:`c_api_deprecations`.

While we probably have little reason to change the layout of the array struct
(``PyArrayObject_fields``) for example the development and improvement of
dtypes would be made easier by changing the `PyArray_Descr` struct.

This NEP proposes a few concrete changes to our C-API mainly as examples.
However, more changes will be handled on a case-by-case basis, and we do not
aim to provide a full list of changes in this NEP.

Adding state is out of scope
----------------------------
New developements such as CPython's support for subinterpreters and the
HPy API may require the NumPy C-API to evolve in a way that may require
(or at least prefer) state to be passed in.

As of now, we do not aim to include changes for this here.  We cannot expect
users to do large code updates to pass in for example an ``HPy`` context
to many NumPy functions.

While we could introduce a second API for this purpose in NumPy 2.0,
we expect that this is unnecessary and that the provisions introduced here:

* the ability to compile with the newest NumPy version but be compatible with
  older versions,
* and the possibility of updating a ``numpy2_compat`` package.

should allow to add such an API also in a minor release.


Usage and Impact
================

Backwards compatible builds
---------------------------

Backwards compatible builds will be described in more details in the
documentation.
Briefly, we will allow users to use a definition like::

    #define NPY_TARGET_VERSION NPY_1_22_API_VERSION

to select the version they wish to compile for (lowest version to be
compatible with).
By default the backwards compatibility will be such that the resulting binary
is compatible with the oldest NumPy version which supports the same
version of Python: NumPy 1.19.x was the first to support Python 3.9 and
NumPy 1.25 supports Python 3.9 or greater, so NumPy 1.25 defaults to 1.19
compatibility.
Thus, users of *new* API may be required to add the define,
but users of who want to be compatible with older versions need not do
anything unless they wish to have exceptionally long compatibility.

The API additions in the past years were so limited that such a change
should be necessary at most for a hand-full of users worldwide.

This mechanism is much the same as the `Python limited API`_ since NumPy's
C-API has a similar need for ABI stability.

Breaking the C-API and changing the ABI
---------------------------------------

NumPy has too many functions, many of which are aliases.  The following
lists *examples* of things we plan to remove and users will have to adapt
to be compatible with NumPy 2.0:

* ``PyArray_Mean`` and ``PyArray_Std`` are untested implementation similar to
  ``arr.mean()`` and  ``arr.std()``.  We are planning on removing these as they
  can be replaced with method calls relatively easily.
* The ``MapIter`` set of API functions (and struct) allows to implement
  advanced indexing like semantics downstream.  There was a single *historic*
  known user of this (theano) and the use-case would be faster and easier to
  implement in a different way.  The API is complicated, requires reaching
  deep into NumPy to be useful and its exposure makes the implementation
  more difficult.  Unless new important use cases are found, we propose to
  remove it.

An example for an ABI change is to change the layout of ``PyArray_Descr``
(the struct of ``np.dtype`` instances) to allow a larger maximum itemsize and
new flags (useful for future custom user DTypes).
For this specific change, users who access the structs fields directly
will have to change their code.  A downstream search shows that this should
not be very common, the main impacts are:

* Access of the ``descr->elsize`` field (and others) would have to be replaced
  with a macro's like ``PyDataType_ITEMSIZE(descr)`` (NumPy may include a
  version check when needed).
* Implementers of user defined dtypes, will have to change a few lines of code
  and luckily, there are very few of such user defined dtypes.
  (The details are that we rename the struct to ``PyArray_DescrProto`` for the
  static definition and fetch the actual instance from NumPy explicitly.)

A last example is increasing ``NPY_MAXDIMS`` to ``64``.
``NPY_MAXDIMS`` is mainly used to statically allocate scratch space::

    func(PyArrayObject *arr) { 
        npy_intp shape[NPY_MAXDIMS];
        /* Work with a shape or strides from the array */
    }

If NumPy changed it to 64 in a minor release, this would lead to undefined
behavior if the code was compiled with ``NPY_MAXDIMS=32`` but a 40 dimensional
array is passed in.
But the larger value is also a correct maximum on previous versions of NumPy
making it generally safe for NumPy 2.0 change.
(One can imagine code that wants to know the actual runtime value.
We have not seen such code in practice, but it would need to be adjusted.)

Impact on Cython users
----------------------

Cython users may use the NumPy C-API via ``cimport numpy as cnp``.
Due to the uncertainty of Cython development, there are two scenarios for
impact on Cython users.

If Cython 3 can be relied on, Cython users would be impacted *less* then C-API
users, because Cython 3 allows us to hide struct layout changes (i.e. changes
to ``PyArray_Descr``).
If this is not the case and we must support Cython 2.x, then Cython users
will also have to use a function/macro like ``PyDataType_ITEMSIZE()`` (or
use the Python object).  This is unfortunately less typical in Cython code,
but also unlikely to be a common pattern for dtype struct fields/attributes.

A further impact is that some future API additions such as new classes may
need to placed in a distinct ``.pyd`` file to avoid Cython generating code
that would fail on older NumPy versions.

End-user and packaging impact
-----------------------------

Packaging in a way that is compatible with NumPy 2.0 will require a
recompilation of downstream libraries that rely on the NumPy C-API.
This may take some time, although hopefully the process will start before
NumPy 2.0 is itself released.

Further, to allow bigger changes more easily in NumPy 2.0, we expect to
create a ``numpy2_compat`` package.
When a library is build with NumPy 2.0 but wants to support NumPy 1.x it will
have to depend on ``numpy2_compat``.  End-users should not need to be aware
of this dependency and an informative error can be raised when the module
is missing.

Some new API can be backported
-------------------------------
One large advantage of allowing users to compile with the newst version of
NumPy is that in some cases we will be able to backport new API.
Some new API functions can be written in terms of old ones or included
directly.

.. note::

    It may be possible to make functions public that were present but
    private in NumPy 1.x public via the compatible ``numpy2_compat`` package. 

This means that at some new API additions could be made available to
downstreams users faster.  They would require a new NumPy version for
*compilation* but their wheels can be backwards compatible with earlier
versions.


Implementation
==============

The first part of implementation (allowing building for an earlier API version)
is very straight forward since the NumPy C-API evolved slowly for
many years.
Some struct fields will be hidden by default and functions introduced in a
more recent version will be marked and hidden unless the
user opted in to a newer API version.
An implementation can be found in the `PR 23528`_.

The second part is mainly about identifying and implementing the desired
changes in a way that backwards compatibility will not be broken and API
breaks remain managable for downstream libraries.
Everyone change we do must have a brief note on how to adapt to the
API change (i.e. alternative functions).

NumPy 2 compatibility and API table changes
-------------------------------------------
To allow changing the API table, NumPy 2.0 would ship a different table than
NumPy 1.x (a table is a list of functions and symbols).

For compatibility we would need to translate the 1.x table to the 2.0 table.
This could be done in headers only in theory, but this seems unwieldy.
We thus propose to add a ``numpy2_compat`` package.  This packages main
purpose would be to provide a translation of the 1.x table to the 2.x one
in a single place (filling in any necessary blanks).

Introducing this package solves the "transition" issue because it allows
a user to:
* Install a SciPy version that is compatible with 2.0 and 1.x
* and keep using NumPy 1.x because of other packages they are using are not
  yet compatible.

The import of ``numpy2_compat`` (and an error when it is missing) will be
inserted by the NumPy eaders as part of the ``import_array()`` call.

Alternatives
============

There are always possibilities to decide not to do certain changes (e.g. due
to downstream users noting their continued need for it).  For example, the
function ``PyArray_Mean`` could be replaced by one to call ``array.mean()``
if necessary.

The NEP proposes to allow larger changes to our API table by introducing a
compatibility package ``numpy2_compat``.
We could do many changes without introducing such a package.

The default API version could be chosen to be older or as the current one.
An older version would be aimed at libraries who want a larger compatibility
than NEP 29 suggests.
Choosing the current would default to removing unnecessary compatibility shims
for users who do not distribute wheels.
The suggested default chooses to favors libraries that distribute wheels and
wish a compatibility range similar to NEP 29.  This is because compatibility
shims should be light-weight and we expect few libraries require a longer
compatibility.

Backward compatibility
======================

As mentioned above backwards compatibility is achieved by:
1. Forcing downstream to recompile with NumPy 2.0
2. Providing a ``numpy2_compat`` library.

But relies on users to adapt to changed C-API as described in the Usage and
Impact section.


Discussion
==========

* https://github.com/numpy/numpy/issues/5888 brought up previously that it
  would be helpful to allow exporting of an older API version in our headers.
  This was never implemented, instead we relied on `oldest-support-numpy`_.
* A first draft of this proposal was presented at the NumPy 2.0 planning
  meeting 2023-04-03.



References and Footnotes
========================

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. _oldest-support-numpy: https://github.com/scipy/oldest-supported-numpy

.. _Python limited API: https://docs.python.org/3/c-api/stable.html

.. _PR 23528: https://github.com/numpy/numpy/pull/23528


Copyright
=========

This document has been placed in the public domain. [1]_
