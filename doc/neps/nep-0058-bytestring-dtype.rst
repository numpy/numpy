.. _NEP58:

=====================================
NEP 58 — A variable-width bytes DType
=====================================

:Author: Nathan Goldbaum
:Status: Draft
:Type: Standards Track
:Created: 2026-08-25

Abstract
--------

I propose ``ByteStringDType``, a variable-width bytes data type: the bytes
sibling of ``StringDType`` (:ref:`NEP 55 <NEP55>`). It reuses StringDType's
arena-backed storage, allocator, and missing-data machinery while

* storing Python :class:`bytes` and returning them as a NumPy scalar that
  subclasses ``bytes``,
* supporting embedded and trailing NUL bytes by construction, and
* exposing only operations meaningful on raw bytes.

Text and bytes never promote or cast implicitly. The only conversion between
``StringDType`` and ``ByteStringDType`` is via the
``np.strings.encode``/``np.strings.decode`` pair. These gain a C UFunc
implementation, along with the ability to add codec-aware variable-width loops
mirroring :meth:`str.encode` and :meth:`bytes.decode`.

A working prototype accompanies this NEP as its reference implementation.

Motivation and scope
--------------------

NumPy's only bytes data type is the fixed-width ``S`` dtype
(``np.dtypes.BytesDType``, scalar ``np.bytes_``). Because ``S`` is
null-padded, it cannot represent trailing NUL bytes::

    >>> np.array([b"x\x00"])[0]
    b'x'

This makes ``S`` unsuitable for generic byte streams such as binary record
formats, encoded blobs, and network data. The truncation cannot be fixed in
place: existing code relies on it, so reports have been closed as not planned
since 2011 (NumPy `issue #2414 <https://github.com/numpy/numpy/issues/2414>`__;
more recently `issue #25268 <https://github.com/numpy/numpy/issues/25268>`__,
which loses the final byte of a SHA-256 digest). :ref:`NEP 55 <NEP55>` called
this out, explicitly ruled a bytes/arbitrary-encoding dtype out of its scope, and
listed an improved binary dtype as complementary future work.  NumPy `issue
#27701 <https://github.com/numpy/numpy/issues/27701>`__ is the open request for a
StringDType equivalent for ``bytes``. This NEP proposes that dtype.

Many downstream libraries fall back to object arrays of ``bytes``
wherever they carry variable-width binary data, giving up NumPy's flat
memory layout and loop machinery. PyArrow materializes every Arrow binary
column as an `object array of boxed bytes objects
<https://github.com/apache/arrow/blob/118892700b95fdfa9a6b3e482a6e5399563f5d75/python/pyarrow/src/arrow/python/arrow_to_pandas.cc#L154-L180>`_,
and the reverse conversion from ``S`` has to guess each element's length
with ``strnlen``, `truncating at the first NUL
<https://github.com/apache/arrow/blob/118892700b95fdfa9a6b3e482a6e5399563f5d75/python/pyarrow/src/arrow/python/numpy_to_arrow.cc#L567-L577>`_. The
h5py library reads HDF5 variable-length byte strings `as object arrays
<https://github.com/h5py/h5py/blob/821e503405b5e26a1333b28f2b6418d1a2f8c88a/h5py/h5t.templ.pyx#L1900-L1903>`_.
Zarr defines a first-class ``variable_length_bytes`` data type and
`stores it in object arrays
<https://github.com/zarr-developers/zarr-python/blob/20ba31e3e1142fae83b178d6e0a29538c2b18725/src/zarr/core/dtype/npy/bytes.py#L938-L963>`_
for the same reason.  Astropy holds FITS variable-length binary columns
in object arrays and cannot round-trip fixed-width character columns
without rewriting their padding (`astropy#11341
<https://github.com/astropy/astropy/issues/11341>`__). pandas inherits
the ``S`` truncation for ``bytes`` columns (`pandas#58205
<https://github.com/pandas-dev/pandas/issues/58205>`__).

In scope:

* A variable-width, NUL-transparent bytes DType with the same operation
  *set* as fixed-width ``S`` (ASCII case folding and predicates,
  byte-indexed search/slice), the same missing-data support as
  StringDType, and casts to/from ``S``, void, and bool.
* A scalar type for the new DType that subclasses ``bytes`` and
  ``np.generic``.
* Explicit, codec-aware ``encode`` and ``decode`` ufuncs as the only
  text-to-bytes path for the variable-width pair.
* As a structural side effect, StringDType's UTF-8 assumptions are
  identified and confined to a small encoding-specific surface of the
  implementation.

Out of scope:

* Changing how Python ``bytes`` values are inferred (``np.array([b"x"])``
  stays fixed-width ``S``, as NEP 55 kept ``str`` inference at ``U``).
* Exposing other encodings (latin-1, utf-16) as array dtypes. The
  encoding-specific surface identified here is a starting point for an
  encoding-parameterized StringDType, but that is a new user-facing
  semantic that would need its own proposal.

Usage and impact
----------------

Because it uses the same representation and arena-backed storage as
``StringDType``, the new ``ByteStringDType`` supports embedded and trailing
NUL bytes automatically:

.. code-block:: python

    >>> import numpy as np
    >>> from numpy.dtypes import ByteStringDType

    >>> a = np.array([b"x\x00", b"a\x00b", b"\xff\xfe"], dtype=ByteStringDType())
    >>> a.tolist()                # trailing and embedded NULs survive
    [b'x\x00', b'a\x00b', b'\xff\xfe']
    >>> np.strings.str_len(a)     # lengths are in bytes, length-explicit
    array([2, 3, 2])
    >>> np.strings.find(a, b"\x00")
    array([ 1,  1, -1])

This dtype does not support the ``coerce`` argument that ``StringDType``
supports, so data that is not bytes will be rejected by ``np.array()``:

.. code-block:: python

    >>> np.array(["text"], dtype=ByteStringDType())
    Traceback (most recent call last):
        ...
    TypeError: ByteStringDType only allows bytes data, got an instance of
    'str'; convert text to bytes explicitly with str.encode(encoding)

Converting between ``StringDType`` and ``ByteStringDType`` happens through
``np.strings.encode`` and ``np.strings.decode``. While the ``encode``
default transitions (see :ref:`backward_compatibility`), the ByteStringDType
result is requested explicitly::

    >>> s = np.array(["héllo"], dtype=np.dtypes.StringDType())
    >>> b = np.strings.encode(s, "utf-8", dtype=ByteStringDType())
    >>> np.strings.decode(b, "utf-8")
    array(['héllo'], dtype=StringDType())

.. _backward_compatibility:

Backward compatibility
----------------------

There is only one major backward compatibility concern: dealing with
``np.strings.encode``. The function already exists and supports
``StringDType``, but sub-optimally in a manner that cannot preserve trailing NUL
bytes. This presents some awkward backward compatibility concerns for
this proposal. In the prototype branch, ``np.strings.encode`` emits a
``DeprecationWarning`` for StringDType input when the new ``dtype=`` argument is
unspecified. Until the default flips to ByteStringDType in a later release, its
behavior is otherwise unchanged: fixed-width ``S`` result, full codec and
error-mode support, and 0-d arrays for 0-d input. See :ref:`encoding_decoding`
for more detail.

Detailed Description
--------------------

The new dtype is exposed as ``ByteStringDType`` in ``np.dtypes``. The more
natural name ``BytesDType`` is already taken by the fixed-width ``S`` dtype.
Like ``StringDType``, it inherits directly from ``np.dtype``.

Like ``StringDType``, it stores variable-length data and records each entry's
length explicitly, so embedded and trailing NUL bytes survive where the
fixed-width bytes dtype would strip them. Unlike the ``S`` to ``StringDType``
cast, which rejects bytes that are not valid UTF-8, the ``S`` to
``ByteStringDType`` cast accepts any bytes.

Any ``bytes`` instance may be stored, including subclasses like
``np.bytes_``, which are stored as their raw bytes. Elements are returned
as instances of the scalar type described under :ref:`nep58-scalar`.
Support for buffer-protocol objects such as ``bytearray`` and ``memoryview``
is deferred to a later iteration.

The ``ByteStringDType`` constructor does not support coercing data to bytes.
Anything that is not bytes — including ``str`` — raises ``TypeError``.  The
dtype never assumes a text encoding. The only bridge between text and bytes is
the codec-aware ``encode``, which takes StringDType to ByteStringDType, and
``decode``, which takes ByteStringDType to StringDType. Conversions from
other dtypes are explicit casts, listed under :ref:`nep58-casts`:
``np.array([True]).astype(ByteStringDType())`` gives ``b"True"``, while
``np.array([True], dtype=ByteStringDType())`` raises because array
construction goes through setitem.

The supported operations are the same set the fixed-width ``S`` dtype and the
Python ``bytes`` type support. Search, slicing, lengths, and widths are all
measured in bytes.

The proposed type character is ``'R'`` (for *raw* bytes), with type number
``NPY_VBYTES = 2057``. No built-in dtype uses ``'R'``, but
the character is used for the ``rational2`` test dtype, which is not exposed
publicly. The ``'Y'`` code is also unused as a dtype character, but collides
with the datetime YEAR unit character.

.. _nep58-scalar:

Scalar type
===========

ByteStringDType gets its own scalar type, ``np.vbytes``: a type that subclasses
both ``bytes`` and ``np.generic``, holds its own copy of the bytes, and follows
the implementation of ``np.bytes_``. Scalar access returns an ``np.vbytes``
instance for a non-null entry and ``na_object`` for a null one; ``.item()`` and
``.tolist()`` return plain ``bytes``. ``np.bytes_`` cannot serve as the scalar
because NumPy needs a distinct scalar type to distinguish from ``np.bytes_``.

NEP 55 chose ``str`` as StringDType's scalar to avoid maintaining a subclass.
On reflection, this was probably a mistake since it is an exception from the
rest of NumPy and is difficult to capture in NumPy's type stubs.  NumPy `PR
#28196 <https://github.com/numpy/numpy/pull/28196>`__ proposes a
StringDType scalar (``np.vstr``) and remains open. Review there converged on
two points this NEP adopts: the scalar subclasses ``str``, here ``bytes``,
as well as ``np.generic``, and it owns a copy of its data instead of retaining
references to arena storage. StringDType continues to return Python ``str``
scalars on NumPy ``main``.

.. _nep58-missing-data:

Missing data
============

``na_object`` works as it does for ``StringDType``: the sentinel may be a
``bytes`` object, a NaN-like object, or any other object, with the same
null-propagation rules. The reasoning in NEP 55 applies unchanged, and the
object arrays this dtype replaces show why the sentinel stays a free
choice. pyarrow and polars export nulls in binary and string columns as
``None``. pandas exports ``float("nan")`` from its default string dtype
and ``pd.NA`` from nullable and Arrow-backed string and bytes columns, and
``to_numpy(na_value=...)`` lets the caller pick any other value.
scikit-learn's encoders treat ``None`` and ``nan`` in object arrays as two
distinct missing categories. StringDType users make the same range of
choices: `anndata
<https://github.com/scverse/anndata/blob/ca5234e62a5b757751a2f635803b42f2b5d61c5c/src/anndata/experimental/backed/_lazy_arrays.py#L186-L189>`__
uses ``pd.NA``, `h5col
<https://github.com/HDFGroup/h5col/blob/13d248afa1b4a9f7f83e8c8441e38a7e10ea45f4/src/h5col/strings.py#L40-L43>`__
uses ``None``, and the pyarrow pull request for StringDType conversion
(`apache/arrow#50951 <https://github.com/apache/arrow/pull/50951>`__)
tests ``None``, a placeholder string, and ``float("nan")``. ``pd.NA`` is a
pandas object NumPy cannot enumerate, so a closed set of sentinels would
exclude the arrays pandas produces (see :ref:`nep58-rejected-ideas`).

Two reasons are specific to this NEP. ``encode`` and ``decode`` propagate
nulls between the two dtypes (:ref:`nep58-null-round-trip`), so the bytes
side needs somewhere to put them. The missing-data machinery is shared, so
leaving it out of ByteStringDType would remove capability without
removing code.

The shared implementation follows StringDType's corrected missing-value
semantics: comparisons involving a NaN-like missing value return ``False``
except for inequality, which returns ``True``. Non-NaN object sentinels
support equality and inequality, with nulls equal to other nulls and distinct
from empty strings; ordered comparisons raise. These StringDType fixes
landed in NumPy `PR #32564 <https://github.com/numpy/numpy/pull/32564>`__.

The stubs parametrize ``StringDType`` by the type of ``na_object``, and
``ByteStringDType`` is typed the same way. That is exact for ``None``,
``pd.NA``, and other singleton sentinels. A NaN sentinel types as
``float``, the same value-level imprecision NumPy typing accepts for
integer ranges and fixed string widths.

.. _operation_surface:

Operation surface
=================

The dtype supports the complete fixed-``S`` operation set with
length-explicit byte semantics, targeting feature parity with the Python
``bytes`` builtin:

* predicates: ``isalpha``, ``isalnum``, ``isdigit``, ``isspace``,
  ``islower``, ``isupper``, ``istitle``
* search: ``find``, ``rfind``, ``index``, ``rindex``, ``count``,
  ``startswith``, ``endswith``
* transforms: ``replace``, ``strip``/``lstrip``/``rstrip``,
  ``expandtabs``, ``center``/``ljust``/``rjust``, ``zfill``,
  ``upper``/``lower``/``swapcase``/``capitalize``/``title``,
  ``translate``, ``mod``
* manipulation: ``add``, ``multiply``, ``minimum``/``maximum``,
  ``partition``/``rpartition``, ``slice``
* excluded permanently: ``isdecimal``, ``isnumeric`` (Unicode-only)

The prototype registers ``str_len``, ``isnan``, the six comparisons, and
one operation per loop-registration mechanism from the surface above:
``isalpha``; ``find`` and ``count``; the whitespace
``strip``/``lstrip``/``rstrip``; ``replace``; ``add``, ``multiply``, and
``minimum``/``maximum``; ``partition``/``rpartition``; ``slice``; and the
encode/decode bridge. The remainder of the surface above is deferred to
follow-up work.

.. _nep58-bytes-scalar-operands:

Python ``bytes`` scalars in mixed operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python ``bytes`` scalars still infer as fixed ``S``, so a mixed operation
promotes exactly as it would against an ``S`` array. Promotion only
chooses the loop, though; once it resolves to ByteStringDType, an exact
``bytes`` operand is converted again from the original object. This is
the bytes analogue of the ``NPY_ARRAY_WAS_PYTHON_STR`` mechanism
StringDType gained in NumPy `PR #32040
<https://github.com/numpy/numpy/pull/32040>`__, and it mirrors its
exact-type rule: subclasses, ``np.bytes_`` included, keep fixed-width
semantics, as ``np.str_`` does for StringDType. Trailing NULs therefore
survive mixed operations with exact ``bytes``: for a ByteStringDType
array ``arr``, ``np.strings.find(arr, b"\0")`` searches for ``b"\0"`` and
``arr + b"x\0"`` appends both bytes. Stores into an explicitly-typed
array (``arr[i] = b"q\0"``, including ``np.bytes_`` and other ``bytes``
subclass values) preserve trailing NULs; setitem accepts any ``bytes``
instance while the operand mechanism replaces only exact ``bytes``.

The rule these mechanisms implement is: when an exact ``bytes`` scalar
reaches an operation whose explicit or resolved target is ByteStringDType,
the value is packed from the original object or the operation fails.
It is never routed through fixed-width ``S`` first. The prototype also
preserves trailing NULs in exact ``bytes`` scalar operands to ``np.full``,
``np.copyto``, ``np.where``, flattened ``np.concatenate``, and ``np.choose``
when the target is ByteStringDType. It extends the shared scalar-conversion
machinery added for StringDType in NumPy `PR #32356
<https://github.com/numpy/numpy/pull/32356>`__. NumPy `PR #32497
<https://github.com/numpy/numpy/pull/32497>`__ subsequently extended scalar
handling in ``concatenate`` and ``choose``; the prototype reuses those
paths for bytes.

Python-level wrappers that convert untyped arguments with ``np.asarray``
before any target is known (``np.append``, ``np.isin`` and
``np.setdiff1d``, ``np.select``, ``np.pad``) lose trailing NULs for
StringDType and ByteStringDType alike; a typed array argument preserves
them. NumPy `issue #32431 <https://github.com/numpy/numpy/issues/32431>`_ tracks
this problem.

.. _nep58-null-round-trip:

Missing values round-trip between StringDType and ByteStringDType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Null elements propagate to null in both directions. A ``str`` or ``bytes``
``na_object`` is converted along with the data, using the same codec and
error mode, so the result's sentinel is still a string sentinel. Every
other sentinel (NaN-like or arbitrary objects) is reused unchanged.

If the sentinel itself does not convert — ``decode`` on an array whose
``na_object`` is ``b"\xff"``, say — the conversion raises up front, even
when every array element converts. Keeping the unconverted ``bytes``
sentinel on the result instead would demote it from a string sentinel to
an opaque object sentinel. String-sentinel nulls sort and compare like
ordinary strings; non-NaN object-sentinel nulls support equality and
inequality, but sorting and ordered comparisons raise.

The StringDType ``coerce`` parameter does not survive a round trip:
ByteStringDType descriptors cannot carry it, so ``decode`` always produces
a default ``coerce=True`` StringDType.

.. _encoding_decoding:

Text encoding and decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^

``_encode``/``_decode`` are *private* ufuncs. The prototype registers a single
utf-8/strict loop pair; the wrappers reject other encodings and error modes
on the variable-width paths. To enable future support for other encodings,
the public wrappers expose ``encoding`` and ``errors`` arguments. Invalid
bytes under ``errors='strict'`` raise ``UnicodeDecodeError`` with the same
position and reason as :meth:`bytes.decode`, whether the bytes are an
array element or the ``na_object``.

``np.strings.encode`` keeps its current behavior for StringDType input
when the new ``dtype=`` argument is unspecified: the result is fixed-width
``S`` via the object round trip, with the full codec and error-mode table.
A ``DeprecationWarning`` announces that the default will change to
ByteStringDType in a future release.
``dtype=np.dtypes.ByteStringDType()`` opts in early; ``dtype=np.bytes_``
keeps the fixed-width result and silences the warning. The intent is to
land the full encoding/errors matrix before the default flips, so that the
flip changes the result dtype and one class of values: encoded output that
ends in NUL bytes, which ``S`` strips and ByteStringDType keeps. That
covers text with a trailing U+0000 under any codec, and ASCII text under
UTF-16 or UTF-32. ``np.strings.decode`` needs no transition:
ByteStringDType input is new, and fixed-width input keeps returning ``U``.

Text file I/O
^^^^^^^^^^^^^

``np.loadtxt`` and ``np.genfromtxt`` read *text*, and ByteStringDType never
assumes an encoding. Both readers therefore refuse ``dtype=ByteStringDType()``
unless explicit converters are given (``converters=str.encode``).

Storage and interchange
^^^^^^^^^^^^^^^^^^^^^^^

The limits of StringDType's arena storage carry over unchanged.
ByteStringDType cannot be a structured field or a subarray element, does
not support the buffer protocol or ``np.memmap``, and ``np.save`` stores it
through the pickle path for custom DTypes, so loading needs
``allow_pickle=True``. ``nbytes`` and ``tobytes`` describe the packed
16-byte-per-element representation, not a contiguous copy of the logical
byte payload.

.. _nep58-casts:

Casts
^^^^^

The prototype registers the casts below. Casting levels follow
StringDType's, including the *safe* fixed-width ``S`` to ByteStringDType
cast: ``S`` cannot hold trailing NULs, so the cast loses nothing.
NumPy `PR #32095 <https://github.com/numpy/numpy/pull/32095>`__ made the
fixed-width to StringDType casts safe on the same grounds. Casts to and
from numeric, datetime, ``U``, and StringDType are not registered, so
``astype`` raises ``TypeError`` for them.

Conversions that inspect array values, such as ``arr.astype("S")`` and
``np.array(arr, dtype="V")``, infer an unspecified destination width from
the longest byte payload, including embedded and trailing NULs and the
representation used for missing values. This extends the StringDType
size inference added in NumPy `PR #32097
<https://github.com/numpy/numpy/pull/32097>`__. An explicit width still
truncates longer values. Descriptor-only resolution, such as
``np.concatenate([arr, arr], dtype="S")``, cannot infer a width and raises
``TypeError``. Fixed-width ``S`` still strips trailing NULs on scalar access,
regardless of the inferred width.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Cast
     - Level
     - Nulls and errors
   * - ByteStringDType to ByteStringDType, different ``na_object``
     - unsafe
     - null to null when both sides have a sentinel; when the target has
       none, a ``bytes`` sentinel is written as itself and any other
       sentinel as its ``repr``
   * - ``S`` to ByteStringDType
     - safe
     - ``S`` has no nulls; trailing NULs are already absent from ``S``
   * - ByteStringDType to ``S``
     - same kind
     - a null is written as the sentinel bytes or its ``repr``; values
       longer than an explicit target width truncate; array-value
       conversions infer an unspecified width
   * - ``V`` to ByteStringDType and back
     - same kind
     - as for ``S``, preserving trailing NULs within the target width;
       structured ``V`` raises ``TypeError``
   * - bool to ByteStringDType and back
     - same kind
     - a NaN-like null is ``True`` and ``None`` is ``False``; a ``bytes``
       sentinel follows its own truthiness, including a nonempty sentinel
       made entirely of NUL bytes

Numeric casts
^^^^^^^^^^^^^

The prototype registers no numeric casts. The semantics below are the
proposal for the follow-up that adds them.

Casting ``ByteStringDType`` to integer or floating point dtypes parses the
byte buffer directly, with the same semantics as calling Python ``int`` or
``float`` on a ``bytes`` value. Leading and trailing ASCII whitespace are
tolerated; embedded NULs and trailing garbage are rejected. There is no
cast in the other direction. ``bytes(int)`` is the surprising
fill-with-zeros constructor (see :pep:`467`) and ``bytes(float)`` raises, so
there is no Python behavior to mirror.

Public C API
^^^^^^^^^^^^

``PyArray_ByteStringDType`` will be exposed in the public C API through the
next free slot (40) of the DType API table, which takes the routine feature
version bump when the feature ships. The prototype does not include the
export. No other public C API is necessary: the existing ``NpyString`` C API
is sufficient to access array data safely.

Implementation
==============

Both DTypes use the existing ``PyArray_StringDTypeObject`` descriptor
struct with an unchanged layout. The struct is public: it is defined in
``ndarraytypes.h``, and ``NpyString_acquire_allocator`` takes a pointer to
it, so C code written against the ``NpyString`` API works on
ByteStringDType descriptors without changes. What makes the DTypes
distinct is their ``PyArray_DTypeMeta``, not the struct.

The encoding-specific parts of the StringDType implementation (scalar
construction, coercion to stored bytes, and ``na_object`` classification)
dispatch at runtime on the descriptor's type number; everything else is
shared. Because ``common_dtype`` never promotes the two DTypes with each
other, NumPy's rule for non-promotable dtypes applies: ``==`` and ``!=``
return all-``False`` and all-``True`` arrays, while ``np.equal`` itself and
the ordering operators raise ``TypeError``.

Reference implementation
========================

The ``bytestringdtype`` `development branch
<https://github.com/numpy/numpy/compare/main...ngoldbaum:numpy:bytestringdtype>`_
on my GitHub fork of NumPy implements the prototype functionality subset
listed above (see :ref:`operation_surface`). The StringDType suite passes
unchanged. A shared suite parametrizes the encoding-agnostic storage,
allocator, and dtype machinery over both DTypes. The test suite covers
the scalar type, ``np.bytes_`` parity, the encode/decode bridge, and the
cross-dtype comparison semantics.

Related Issues
==============

NumPy `issue #32431 <https://github.com/numpy/numpy/issues/32431>`__ remains
open and tracks trailing-NUL loss from Python scalar or sequence operands
in wrappers including ``np.append``, ``np.isin``, ``np.select``, and
``np.pad``.

Acceptance and release plan
===========================

Accepting this NEP ratifies the design. The prototype can be merged at that
point. The NEP becomes Final when the scalar type, the deferred operations,
the numeric casts, and the prototype's ``bytes`` scalar promotion for
value-based conversions have landed.

.. _nep58-alternatives:

Alternatives
------------

**A distinct descriptor struct.** Cleaner in isolation, but it would
require either duplicating the allocator API surface or adding a
DType-dispatching accessor layer: new public functions for downstream code
to learn, with no user-facing benefit. The shared struct costs one
vestigial byte per descriptor.

**C++ encoding-policy templates.** The encoding-specific dispatch could
be factored into compile-time policy structs, with the dtype slots as
function templates over the policy. Nothing in this proposal blocks
that; a future refactor could adopt it independently. I elected not to do
this to simplify the prototype implementation.

.. _nep58-rejected-ideas:

Rejected ideas
--------------

**Omitting missing-data support for static typing.** ByteStringDType
preserves StringDType's support for arbitrary ``na_object`` values rather
than removing that support to fit current limitations in static typing.

`PEP 586's discussion of illegal Literal parameters
<https://peps.python.org/pep-0586/#illegal-parameters-for-literal-at-type-check-time>`__
explicitly deferred float literal support for simplicity, citing the
difficulty of representing infinity and NaN and the expectation that APIs
would rarely depend on a float parameter's value. The `current typing
specification
<https://typing.python.org/en/latest/spec/literal.html#illegal-parameters-for-literal-at-type-check-time>`__
still excludes float literals. StringDType and ByteStringDType now give
the typing community and upstream Python a concrete use case for
revisiting that deferral: a NaN missing-data sentinel needs to be
distinguishable from ordinary float values in the element type.
Describing the sentinel as ``float`` loses that distinction.
The preferred resolution is to extend Python's typing specification to
express this existing runtime behavior precisely.

**A closed set of missing-data sentinels.** Restricting ``na_object`` to
``None`` and NaN, or to an enum, was suggested so that an array element
has an exact static type. The dtype's type parameter already records the
sentinel's type and only NaN types imprecisely, while every sentinel
outside the set, ``pd.NA`` included, would become unrepresentable. An
enum stored as the element would also defeat the checks users choose NaN
for, such as ``np.isnan`` and ``x != x``.

**Representing missing values with the bytes scalar.** Missing entries
return the configured ``na_object`` directly, as they do for StringDType.
The ``np.vbytes`` scalar represents nonmissing bytes values; it does not
wrap missing-value sentinels to give all array elements the same scalar
type.

Discussion
----------

* NumPy `issue #27701 <https://github.com/numpy/numpy/issues/27701>`__.
* `Mailing list thread <https://mail.python.org/archives/list/numpy-discussion@python.org/thread/O4BIZE4YIYAS2SYOMXIENVS7BP2NNNWI/>`_

Copyright
---------

This document has been placed in the public domain.
