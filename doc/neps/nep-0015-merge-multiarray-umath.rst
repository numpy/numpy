=====================================
NEP 15 — Merging multiarray and umath
=====================================

:Author: Nathaniel J. Smith <njs@pobox.com>
:Status: Final
:Type: Standards Track
:Created: 2018-02-22
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-June/078345.html

Abstract
--------

Let's merge ``numpy.core.multiarray`` and ``numpy.core.umath`` into a
single extension module, and deprecate ``np.set_numeric_ops``.


Background
----------

Currently, numpy's core C code is split between two separate extension
modules.

``numpy.core.multiarray`` is built from
``numpy/core/src/multiarray/*.c``, and contains the core array
functionality (in particular, the ``ndarray`` object).

``numpy.core.umath`` is built from ``numpy/core/src/umath/*.c``, and
contains the ufunc machinery.

These two modules each expose their own separate C API, accessed via
``import_multiarray()`` and ``import_umath()`` respectively. The idea
is that they're supposed to be independent modules, with
``multiarray`` as a lower-level layer with ``umath`` built on top. In
practice this has turned out to be problematic.

First, the layering isn't perfect: when you write ``ndarray +
ndarray``, this invokes ``ndarray.__add__``, which then calls the
ufunc ``np.add``. This means that ``ndarray`` needs to know about
ufuncs – so instead of a clean layering, we have a circular
dependency. To solve this, ``multiarray`` exports a somewhat
terrifying function called ``set_numeric_ops``. The bootstrap
procedure each time you ``import numpy`` is:

1. ``multiarray`` and its ``ndarray`` object are loaded, but
   arithmetic operations on ndarrays are broken.

2. ``umath`` is loaded.

3. ``set_numeric_ops`` is used to monkeypatch all the methods like
   ``ndarray.__add__`` with objects from ``umath``.

In addition, ``set_numeric_ops`` is exposed as a public API,
``np.set_numeric_ops``.

Furthermore, even when this layering does work, it ends up distorting
the shape of our public ABI. In recent years, the most common reason
for adding new functions to ``multiarray``\'s "public" ABI is not that
they really need to be public or that we expect other projects to use
them, but rather just that we need to call them from ``umath``. This
is extremely unfortunate, because it makes our public ABI
unnecessarily large, and since we can never remove things from it then
this creates an ongoing maintenance burden. The way C works, you can
have internal API that's visible to everything inside the same
extension module, or you can have a public API that everyone can use;
you can't (easily) have an API that's visible to multiple extension
modules inside numpy, but not to external users.

We've also increasingly been putting utility code into
``numpy/core/src/private/``, which now contains a bunch of files which
are ``#include``\d twice, once into ``multiarray`` and once into
``umath``. This is pretty gross, and is purely a workaround for these
being separate C extensions. The ``npymath`` library is also
included in both extension modules.


Proposed changes
----------------

This NEP proposes three changes:

1. We should start building ``numpy/core/src/multiarray/*.c`` and
   ``numpy/core/src/umath/*.c`` together into a single extension
   module.

2. Instead of ``set_numeric_ops``, we should use some new, private API
   to set up ``ndarray.__add__`` and friends.

3. We should deprecate, and eventually remove, ``np.set_numeric_ops``.


Non-proposed changes
--------------------

We don't necessarily propose to throw away the distinction between
multiarray/ and umath/ in terms of our source code organization:
internal organization is useful! We just want to build them together
into a single extension module. Of course, this does open the door for
potential future refactorings, which we can then evaluate based on
their merits as they come up.

It also doesn't propose that we break the public C ABI. We should
continue to provide ``import_multiarray()`` and ``import_umath()``
functions – it's just that now both ABIs will ultimately be loaded
from the same C library. Due to how ``import_multiarray()`` and
``import_umath()`` are written, we'll also still need to have modules
called ``numpy.core.multiarray`` and ``numpy.core.umath``, and they'll
need to continue to export ``_ARRAY_API`` and ``_UFUNC_API`` objects –
but we can make one or both of these modules be tiny shims that simply
re-export the magic API object from where-ever it's actually defined.
(See ``numpy/core/code_generators/generate_{numpy,ufunc}_api.py`` for
details of how these imports work.)


Backward compatibility
----------------------

The only compatibility break is the deprecation of ``np.set_numeric_ops``.


Rejected alternatives
---------------------

Preserve ``set_numeric_ops`` for monkeypatching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In discussing this NEP, one additional use case was raised for
``set_numeric_ops``: if you have an optimized vector math library
(e.g. Intel's MKL VML, Sleef, or Yeppp), then ``set_numeric_ops`` can
be used to monkeypatch numpy to use these operations instead of
numpy's built-in vector operations. But, even if we grant that this is
a great idea, using ``set_numeric_ops`` isn't actually the best way to
do it. All ``set_numeric_ops`` allows you to do is take over Python's
syntactic operators (``+``, ``*``, etc.) on ndarrays; it doesn't let
you affect operations called via other APIs (e.g., ``np.add``), or
operations that don't have built-in syntax (e.g., ``np.exp``). Also,
you have to reimplement the whole ufunc machinery, instead of just the
core loop. On the other hand, the `PyUFunc_ReplaceLoopBySignature
<https://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html#c.PyUFunc_ReplaceLoopBySignature>`__
API – which was added in 2006 – allows replacement of the inner loops
of arbitrary ufuncs. This is both simpler and more powerful – e.g.
replacing the inner loop of ``np.add`` means your code will
automatically be used for both ``ndarray + ndarray`` as well as direct
calls to ``np.add``. So this doesn't seem like a good reason to not
deprecate ``set_numeric_ops``.


Discussion
----------

* https://mail.python.org/pipermail/numpy-discussion/2018-March/077764.html
* https://mail.python.org/pipermail/numpy-discussion/2018-June/078345.html

Copyright
---------

This document has been placed in the public domain.
