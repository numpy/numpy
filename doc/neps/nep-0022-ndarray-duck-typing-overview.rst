===========================================================
NEP 22 — Duck typing for NumPy arrays – high level overview
===========================================================

:Author: Stephan Hoyer <shoyer@google.com>, Nathaniel J. Smith <njs@pobox.com>
:Status: Final
:Type: Informational
:Created: 2018-03-22
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-September/078752.html

Abstract
--------

We outline a high-level vision for how NumPy will approach handling
“duck arrays”. This is an Informational-class NEP; it doesn’t
prescribe full details for any particular implementation. In brief, we
propose developing a number of new protocols for defining
implementations of multi-dimensional arrays with high-level APIs
matching NumPy.


Detailed description
--------------------

Traditionally, NumPy’s ``ndarray`` objects have provided two things: a
high level API for expression operations on homogenously-typed,
arbitrary-dimensional, array-structured data, and a concrete
implementation of the API based on strided in-RAM storage. The API is
powerful, fairly general, and used ubiquitously across the scientific
Python stack. The concrete implementation, on the other hand, is
suitable for a wide range of uses, but has limitations: as data sets
grow and NumPy becomes used in a variety of new environments, there
are increasingly cases where the strided in-RAM storage strategy is
inappropriate, and users find they need sparse arrays, lazily
evaluated arrays (as in dask), compressed arrays (as in blosc), arrays
stored in GPU memory, arrays stored in alternative formats such as
Arrow, and so forth – yet users still want to work with these arrays
using the familiar NumPy APIs, and re-use existing code with minimal
(ideally zero) porting overhead. As a working shorthand, we call these
“duck arrays”, by analogy with Python’s “duck typing”: a “duck array”
is a Python object which “quacks like” a numpy array in the sense that
it has the same or similar Python API, but doesn’t share the C-level
implementation.

This NEP doesn’t propose any specific changes to NumPy or other
projects; instead, it gives an overview of how we hope to extend NumPy
to support a robust ecosystem of projects implementing and relying
upon its high level API.

Terminology
~~~~~~~~~~~

“Duck array” works fine as a placeholder for now, but it’s pretty
jargony and may confuse new users, so we may want to pick something
else for the actual API functions. Unfortunately, “array-like” is
already taken for the concept of “anything that can be coerced into an
array” (including e.g. list objects), and “anyarray” is already taken
for the concept of “something that shares ndarray’s implementation,
but has different semantics”, which is the opposite of a duck array
(e.g., np.matrix is an “anyarray”, but is not a “duck array”). This is
a classic bike-shed so for now we’re just using “duck array”. Some
possible options though include: arrayish, pseudoarray, nominalarray,
ersatzarray, arraymimic, ...


General approach
~~~~~~~~~~~~~~~~

At a high level, duck array support requires working through each of
the API functions provided by NumPy, and figuring out how it can be
extended to work with duck array objects. In some cases this is easy
(e.g., methods/attributes on ndarray itself); in other cases it’s more
difficult. Here are some principles we’ve found useful so far:


Principle 1: Focus on “full” duck arrays, but don’t rule out “partial” duck arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can distinguish between two classes:

* “full” duck arrays, which aspire to fully implement np.ndarray’s
  Python-level APIs and work essentially anywhere that np.ndarray
  works

* “partial” duck arrays, which intentionally implement only a subset
  of np.ndarray’s API.

Full duck arrays are, well, kind of boring. They have exactly the same
semantics as ndarray, with differences being restricted to
under-the-hood decisions about how the data is actually stored. The
kind of people that are excited about making numpy more extensible are
also, unsurprisingly, excited about changing or extending numpy’s
semantics. So there’s been a lot of discussion of how to best support
partial duck arrays. We've been guilty of this ourself.

At this point though, we think the best general strategy is to focus
our efforts primarily on supporting full duck arrays, and only worry
about partial duck arrays as much as we need to to make sure we don't
accidentally rule them out for no reason.

Why focus on full duck arrays? Several reasons:

First, there are lots of very clear use cases. Potential consumers of
the full duck array interface include almost every package that uses
numpy (scipy, sklearn, astropy, ...), and in particular packages that
provide array-wrapping-classes that handle multiple types of arrays,
such as xarray and dask.array. Potential implementers of the full duck
array interface include: distributed arrays, sparse arrays, masked
arrays, arrays with units (unless they switch to using dtypes),
labeled arrays, and so forth. Clear use cases lead to good and
relevant APIs.

Second, the Anna Karenina principle applies here: full duck arrays are
all alike, but every partial duck array is partial in its own way:

* ``xarray.DataArray`` is mostly a duck array, but has incompatible
  broadcasting semantics.
* ``xarray.Dataset`` wraps multiple arrays in one object; it still
  implements some array interfaces like ``__array_ufunc__``, but
  certainly not all of them.
* ``pandas.Series`` has methods with similar behavior to numpy, but
  unique null-skipping behavior.
* scipy’s ``LinearOperator``\s support matrix multiplication and nothing else
* h5py and similar libraries for accessing array storage have objects
  that support numpy-like slicing and conversion into a full array,
  but not computation.
* Some classes may be similar to ndarray, but without supporting the
  full indexing semantics.

And so forth.

Despite our best attempts, we haven't found any clear, unique way of
slicing up the ndarray API into a hierarchy of related types that
captures these distinctions; in fact, it’s unlikely that any single
person even understands all the distinctions. And this is important,
because we have a *lot* of APIs that we need to add duck array support
to (both in numpy and in all the projects that depend on numpy!). By
definition, these already work for ``ndarray``, so hopefully getting
them to work for full duck arrays shouldn’t be so hard, since by
definition full duck arrays act like ``ndarray``. It’d be very
cumbersome to have to go through each function and identify the exact
subset of the ndarray API that it needs, then figure out which partial
array types can/should support it. Once we have things working for
full duck arrays, we can go back later and refine the APIs needed
further as needed. Focusing on full duck arrays allows us to start
making progress immediately.

In the future, it might be useful to identify specific use cases for
duck arrays and standardize narrower interfaces targeted just at those
use cases. For example, it might make sense to have a standard “array
loader” interface that file access libraries like h5py, netcdf, pydap,
zarr, ... all implement, to make it easy to switch between these
libraries. But that’s something that we can do as we go, and it
doesn’t necessarily have to involve the NumPy devs at all. For an
example of what this might look like, see the documentation for
`dask.array.from_array
<http://dask.pydata.org/en/latest/array-api.html#dask.array.from_array>`__.


Principle 2: Take advantage of duck typing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ndarray`` has a very large API surface area::

    In [1]: len(set(dir(np.ndarray)) - set(dir(object)))
    Out[1]: 138

And this is a huge **under**\estimate, because there are also many
free-standing functions in NumPy and other libraries which currently
use the NumPy C API and thus only work on ``ndarray`` objects. In type
theory, a type is defined by the operations you can perform on an
object; thus, the actual type of ``ndarray`` includes not just its
methods and attributes, but *all* of these functions. For duck arrays
to be successful, they’ll need to implement a large proportion of the
``ndarray`` API – but not all of it. (For example,
``dask.array.Array`` does not provide an equivalent to the
``ndarray.ptp`` method, presumably because no-one has ever noticed or
cared about its absence. But this doesn’t seem to have stopped people
from using dask.)

This means that realistically, we can’t hope to define the whole duck
array API up front, or that anyone will be able to implement it all in
one go; this will be an incremental process. It also means that even
the so-called “full” duck array interface is somewhat fuzzily defined
at the borders; there are parts of the ``np.ndarray`` API that duck
arrays won’t have to implement, but we aren’t entirely sure what those
are.

And ultimately, it isn’t really up to the NumPy developers to define
what does or doesn’t qualify as a duck array. If we want scikit-learn
functions to work on dask arrays (for example), then that’s going to
require negotiation between those two projects to discover
incompatibilities, and when an incompatibility is discovered it will
be up to them to negotiate who should change and how. The NumPy
project can provide technical tools and general advice to help resolve
these disagreements, but we can’t force one group or another to take
responsibility for any given bug.

Therefore, even though we’re focusing on “full” duck arrays, we
*don’t* attempt to define a normative “array ABC” – maybe this will be
useful someday, but right now, it’s not. And as a convenient
side-effect, the lack of a normative definition leaves partial duck
arrays room to experiment.

But, we do provide some more detailed advice for duck array
implementers and consumers below.

Principle 3: Focus on protocols
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Historically, numpy has had lots of success at interoperating with
third-party objects by defining *protocols*, like ``__array__`` (asks
an arbitrary object to convert itself into an array),
``__array_interface__`` (a precursor to Python’s buffer protocol), and
``__array_ufunc__`` (allows third-party objects to support ufuncs like
``np.exp``).

`NEP 16 <https://github.com/numpy/numpy/pull/10706>`_ took a
different approach: we need a duck-array equivalent of
``asarray``, and it proposed to do this by defining a version of
``asarray`` that would let through objects which implemented a new
AbstractArray ABC. As noted above, we now think that trying to define
an ABC is a bad idea for other reasons. But when this NEP was
discussed on the mailing list, we realized that even on its own
merits, this idea is not so great. A better approach is to define a
*method* that can be called on an arbitrary object to ask it to
convert itself into a duck array, and then define a version of
``asarray`` that calls this method.

This is strictly more powerful: if an object is already a duck array,
it can simply ``return self``. It allows more correct semantics: NEP
16 assumed that ``asarray(obj, dtype=X)`` is the same as
``asarray(obj).astype(X)``, but this isn’t true. And it supports more
use cases: if h5py supported sparse arrays, it might want to provide
an object which is not itself a sparse array, but which can be
automatically converted into a sparse array. See NEP <XX, to be
written> for full details.

The protocol approach is also more consistent with core Python
conventions: for example, see the ``__iter__`` method for coercing
objects to iterators, or the ``__index__`` protocol for safe integer
coercion. And finally, focusing on protocols leaves the door open for
partial duck arrays, which can pick and choose which subset of the
protocols they want to participate in, each of which have well-defined
semantics.

Conclusion: protocols are one honking great idea – let’s do more of
those.

Principle 4: Reuse existing methods when possible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It’s tempting to try to define cleaned up versions of ndarray methods
with a more minimal interface to allow for easier implementation. For
example, ``__array_reshape__`` could drop some of the strange
arguments accepted by ``reshape`` and ``__array_basic_getitem__``
could drop all the `strange edge cases
<http://www.numpy.org/neps/nep-0021-advanced-indexing.html>`__ of
NumPy’s advanced indexing.

But as discussed above, we don’t really know what APIs we need for
duck-typing ndarray. We would inevitably end up with a very long list
of new special methods. In contrast, existing methods like ``reshape``
and ``__getitem__`` have the advantage of already being widely
used/exercised by libraries that use duck arrays, and in practice, any
serious duck array type is going to have to implement them anyway.

Principle 5: Make it easy to do the right thing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Making duck arrays work well is going to be a community effort.
Documentation helps, but only goes so far. We want to make it easy to
implement duck arrays that do the right thing.

One way NumPy can help is by providing mixin classes for implementing
large groups of related functionality at once.
``NDArrayOperatorsMixin`` is a good example: it allows for
implementing arithmetic operators implicitly via the
``__array_ufunc__`` method. It’s not complete, and we’ll want more
helpers like that (e.g. for reductions).

(We initially thought that the importance of these mixins might be an
argument for providing an array ABC, since that’s the standard way to
do mixins in modern Python. But in discussion around NEP 16 we
realized that partial duck arrays also wanted to take advantage of
these mixins in some cases, so even if we did have an array ABC then
the mixins would still need some sort of separate existence. So never
mind that argument.)

Tentative duck array guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a general rule, libraries using duck arrays should insist upon the
minimum possible requirements, and libraries implementing duck arrays
should provide as complete of an API as possible. This will ensure
maximum compatibility. For example, users should prefer to rely on
``.transpose()`` rather than ``.swapaxes()`` (which can be implemented
in terms of transpose), but duck array authors should ideally
implement both.

If you are trying to implement a duck array, then you should strive to
implement everything. You certainly need ``.shape``, ``.ndim`` and
``.dtype``, but also your dtype attribute should actually be a
``numpy.dtype`` object, weird fancy indexing edge cases should ideally
work, etc. Only details related to NumPy’s specific ``np.ndarray``
implementation (e.g., ``strides``, ``data``, ``view``) are explicitly
out of scope.

A (very) rough sketch of future plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposals discussed so far – ``__array_ufunc__`` and some kind of
``asarray`` protocol – are clearly necessary but not sufficient for
full duck typing support. We expect the need for additional protocols
to support (at least) these features:

* **Concatenating** duck arrays, which would be used internally by other
  array combining methods like stack/vstack/hstack. The implementation
  of concatenate will need to be negotiated among the list of array
  arguments. We expect to use an ``__array_concatenate__`` protocol
  like ``__array_ufunc__`` instead of multiple dispatch.
* **Ufunc-like functions** that currently aren’t ufuncs. Many NumPy
  functions like median, percentile, sort, where and clip could be
  written as generalized ufuncs but currently aren’t. Either these
  functions should be written as ufuncs, or we should consider adding
  another generic wrapper mechanism that works similarly to ufuncs but
  makes fewer guarantees about how the implementation is done.
* **Random number generation** with duck arrays, e.g.,
  ``np.random.randn()``. For example, we might want to add new APIs
  like ``random_like()`` for generating new arrays with a matching
  shape *and* type – though we'll need to look at some real examples
  of how these functions are used to figure out what would be helpful.
* **Miscellaneous other functions** such as ``np.einsum``,
  ``np.zeros_like``, and ``np.broadcast_to`` that don’t fall into any
  of the above categories.
* **Checking mutability** on duck arrays, which would imply that they
  support assignment with ``__setitem__`` and the out argument to
  ufuncs. Many otherwise fine duck arrays are not easily mutable (for
  example, because they use some kinds of sparse or compressed
  storage, or are in read-only shared memory), and it turns out that
  frequently-used code like the default implementation of ``np.mean``
  needs to check this (to decide whether it can re-use temporary
  arrays).

We intentionally do not describe exactly how to add support for these
types of duck arrays here. These will be the subject of future NEPs.


Copyright
---------

This document has been placed in the public domain.
