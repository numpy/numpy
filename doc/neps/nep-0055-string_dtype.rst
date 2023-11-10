.. _NEP55:

=========================================================
NEP 55 — Add a UTF-8 Variable-Width String DType to NumPy
=========================================================

:Author: Nathan Goldbaum <ngoldbaum@quansight.com>
:Status: Draft
:Type: Standards Track
:Created: 2023-06-29


Abstract
--------

We propose adding a new string data type to NumPy where each item in the array
is an arbitrary length UTF-8 encoded string. This will enable performance,
memory usage, and usability improvements for NumPy users, including:

* Memory savings for workflows that currently use fixed-width strings and store
  primarily ASCII data or a mix of short and long strings in a single NumPy
  array.

* Downstream libraries and users will be able to move away from object arrays
  currently used as a substitute for variable-length string arrays, unlocking
  performance improvements by avoiding passes over the data outside of NumPy and
  allowing use of fast GIL-releasing C casts and string ufuncs for string
  operations.

* A more intuitive user-facing API for working with arrays of Python strings,
  without a need to think about the in-memory array representation.

Motivation and Scope
--------------------

First, we will describe how the current state of support for string or
string-like data in NumPy arose. Next, we will summarize the last major previous
discussion about this topic. Finally, we will describe the scope of the proposed
changes to NumPy as well as changes that are explicitly out of scope of this
proposal.

History of String Support in Numpy
**********************************

Support in NumPy for textual data evolved organically in response to early user
needs and then changes in the Python ecosystem.

Support for strings was added to NumPy to support users of the NumArray
``chararray`` type. Remnants of this are still visible in the NumPy API:
string-related functionality lives in ``np.char``, to support the
``np.char.chararray`` class. This class is not formally deprecated, but has a
had comment in its implementation suggesting string dtypes instead since NumPy
1.4.

NumPy's ``bytes_`` DType was originally used to represent the Python 2 ``str``
type before Python 3 support was added to NumPy. The bytes DType makes the most
sense when it is used to represent Python 2 strings or other null-terminated
byte sequences. However, ignoring data after the first null character means the
``bytes_`` DType is only suitable for bytestreams that do not contain nulls, so
it is a poor match for generic bytestreams.

The ``unicode`` DType was added to support the Python 2 ``unicode`` type. It
stores data in 32-bit UCS-4 codepoints (e.g. a UTF-32 encoding), which makes for
a straightforward implementation, but is inefficient for storing text that can
be represented well using a one-byte ASCII or Latin-1 encoding. This was not a
problem in Python 2, where ASCII or mostly-ASCII text could use the ``str``
DType.

With the arrival of Python 3 support in NumPy, the string DTypes were largely
left alone due to backward compatibility concerns, although the unicode DType
became the default DType for ``str`` data and the old ``string`` DType was
renamed the ``bytes_`` DType. This change left NumPy with the sub-optimal
situation of shipping a data type originally intended for null-terminated
bytestrings as the data type for *all* python ``bytes`` data, and a default
string type with an in-memory representation that consumes four times as much
memory than what is needed for data that can be represented well by a one-byte
ASCII or Latin-1 encoding.

Problems with Fixed-Width Strings
*********************************

Both existing string DTypes represent fixed-width sequences, allowing storage of
the string data in the array buffer. This avoids adding out-of-band storage to
NumPy, however, it makes for an awkward user interface. In particular, the
maximum string size must be inferred by NumPy or estimated by the user before
loading the data into a NumPy array or selecting an output DType for string
operations. In the worst case, this requires an expensive pass over the full
dataset to calculate the maximum length of an array element. It also wastes
memory when array elements have varying lengths. Pathological cases where an
array stores many short strings and a few very long strings are particularly bad
for wasting memory.

Downstream usage of string data in NumPy arrays has proven out the need for a
variable-width string data type. In practice, many downstream libraries avoid
using fixed-width strings due to usability issues and instead employ ``object``
arrays for storing strings. In particular, Pandas has explicitly deprecated
support for NumPy fixed-width strings, coerces NumPy fixed-width string arrays
to ``object`` arrays, and in the future may switch to only supporting string
data via PyArrow, which has native support for UTF-8 encoded variable-width
string arrays [1]_. This is unfortunate, since ``object`` arrays have no type
guarantees, necessitating expensive sanitization passes and operations using
object arrays cannot release the GIL.

Previous Discussions
--------------------

The project last publicly discussed this topic in depth in 2017, when Julian
Taylor proposed a fixed-width text data type parameterized by an encoding
[2]_. This started a wide-ranging discussion about pain points for working with
string data in NumPy and possible ways forward.

In the end, the discussion identified [3]_ [4]_ [5]_ two use-cases that the
current support for strings does a poor job of handling:

* Loading or memory-mapping scientific datasets with unknown encoding,
* Working with "a NumPy array of python strings" in a manner that allows
  transparent conversion between NumPy arrays and Python strings, including
  support for missing strings. The ``object`` DType partially satisfies this
  need, albeit with a cost of slow performance and no type checking.

As a result of this discussion, improving support for string data was added to
the NumPy project roadmap [6]_, with an explicit call-out to add a DType better
suited to memory-mapping bytes with any or no encoding, and a variable-width
string DType that supports missing data to replace usages of object string
arrays.

Proposed work
-------------

This NEP proposes adding ``StringDType``, a DType that stores variable-width
heap-allocated strings in Numpy arrays, to replace downstream usages of the
``object`` DType for string data. This work will heavily leverage recent
improvements in NumPy to improve support for user-defined DTypes, so we will
also necessarily be working on the data type internals in NumPy. In particular,
we propose to:

* Add a new variable-length string DType to NumPy, targeting NumPy 2.0.

* Work out issues related to adding a DType implemented using the experimental
  DType API to NumPy itself.

* Support for a user-provided missing data sentinel.

* A cleanup of ``np.char``, with new string ufuncs available in a new
  namespace for functions and types related to string support.

* An update to the ``npy`` and ``npz`` file formats to allow storage of
  arbitrary-length sidecar data.

The following is out of scope for this work:

* Changing DType inference for string data.

* Adding a DType for memory-mapping text in unknown encodings or a DType that
  attempts to fix issues with the ``bytes_`` DType.

* Fully agreeing on the semantics of a missing data sentinels or adding a
  missing data sentinel to NumPy itself.

* Implement SIMD optimizations for string operations.

While we're explicitly ruling out implementing these items as part of this work,
adding a new string DType helps set up future work that does implement some of
these items.

If implemented this NEP will make it easier to add a new fixed-width text DType
in the future by moving string operations into a long-term supported namespace
and improving the internal infrastructure in NumPy for handling strings. We are
also proposing a memory layout that should be amenable to SIMD optimization in
some cases, increasing the payoff for writing string operations as
SIMD-optimized ufuncs in the future.

While we are not proposing adding a missing data sentinel to NumPy, we are
proposing adding support for an optional, user-provided missing data sentinel,
so this does move NumPy a little closer to officially supporting missing
data. We are attempting to avoid resolving the disagreement described in
:ref:`NEP 26<NEP26>` and this proposal does not require or preclude adding a
missing data sentinel or bitflag-based missing data support to ``ndarray`` in
the future.

Usage and Impact
----------------

The DType is intended as a drop-in replacement for object string arrays. This
means that we intend to support as many downstream usages of object string
arrays as possible, including all supported NumPy functionality. Pandas is the
obvious first user, and substantial work has already occurred to add support in
a fork of Pandas. ``scikit-learn`` also uses object string arrays and will be
able to migrate to a DType with guarantees that the arrays contains only
strings. Both h5py [7]_ and PyTables [8]_ will be able to add first-class
support for variable-width UTF-8 encoded string datasets in HDF5. String data
are heavily used in machine-learning workflows and downstream machine learning
libraries will be able to leverage this new DType.

Users who wish to load string data into NumPy and leverage NumPy features like
fancy advanced indexing will have a natural choice that offers substantial
memory savings over fixed-width unicode strings and better validation guarantees
and overall integration with NumPy than object string arrays. Moving to a
first-class string DType also removes the need to acquire the GIL during string
operations, unlocking future optimizations that are impossible with object
string arrays.

Performance
***********

TODO TODO TODO update these numbers

Here we briefly describe preliminary performance measurements of the prototype
version of ``StringDType`` we have implemented outside of NumPy using the
experimental DType API. All benchmarks in this section were performed on a Dell
XPS 13 9380 running Ubuntu 22.04 and Python 3.11.3 compiled using pyenv. NumPy,
Pandas, and the ``StringDType`` prototype were all compiled with meson release
builds.

Currently, the ``StringDType`` prototype has comparable performance with object
arrays and fixed-width string arrays. One exception is array creation from
python strings, performance is somewhat slower than object arrays and comparable
to fixed-width unicode arrays::

  In [1]: from stringdtype import StringDType

  In [2]: import numpy as np

  In [3]: data = [str(i) * 10 for i in range(100_000)]

  In [4]: %timeit arr_object = np.array(data, dtype=object)
  3.55 ms ± 51.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [5]: %timeit arr_stringdtype = np.array(data, dtype=StringDType())
  12.9 ms ± 277 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [6]: %timeit arr_strdtype = np.array(data, dtype=str)
  11.7 ms ± 150 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In this example, object DTypes are substantially faster because the objects in
the ``data`` list can be directly interned in the array, while ``StrDType`` and
``StringDType`` need to copy the string data and ``StringDType`` needs to
convert the data to UTF-8 and perform additional heap allocations outside the
array buffer. In the future, if Python moves to a UTF-8 internal representation
for strings, the string loading performance of ``StringDType`` should improve.

String operations have similar performance::

  In [7]: %timeit np.array([s.capitalize() for s in data], dtype=object)
  30.2 ms ± 109 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  In [8]: %timeit np.char.capitalize(arr_stringdtype)
  38.5 ms ± 3.01 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

  In [9]: %timeit np.char.capitalize(arr_strdtype)
  46.4 ms ± 1.32 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

The poor performance here is a reflection of the slow iterator-based
implementation of operations in ``np.char``. If we were to rewrite these
operations as ufuncs, we could unlock substantial performance
improvements. Using the example of the ``add`` ufunc, which we have implemented
for the ``StringDType`` prototype::

  In [10]: %timeit arr_object + arr_object
  10 ms ± 308 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [11]: %timeit arr_stringdtype + arr_stringdtype
  5.91 ms ± 18.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [12]: %timeit np.char.add(arr_strdtype, arr_strdtype)
  65.9 ms ± 1.3 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

As described below, we have already updated a fork of Pandas to use a prototype
version of ``StringDType``. This demonstrates the performance improvements
available when data are already loaded into a NumPy array and are passed to a
third-party library. Currently Pandas attempts to coerce all ``str`` data to
``object`` DType by default, and has to check and sanitize existing ``object``
arrays that are passed in. This requires a copy or pass over the data made
unnecessary by first-class support for variable-width strings in both NumPy and
Pandas::

  In [13]: import pandas as pd

  In [14]: %timeit pd.Series(arr_stringdtype)
  20.9 µs ± 341 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

  In [15]: %timeit pd.Series(arr_object)
  1.08 ms ± 23.4 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

We have also implemented a Pandas extension DType that uses ``StringDType``
under the hood, which is also substantially faster for creating Pandas data
structures than the existing Pandas string DType that uses ``object`` arrays::

  In [16]: %timeit pd.Series(arr_stringdtype, dtype='string[numpy]')
  54.7 µs ± 1.38 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

  In [17]: %timeit pd.Series(arr_object, dtype='string[python]')
  1.39 ms ± 1.16 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

Backward compatibility
----------------------

We are not proposing a change to DType inference for python strings and do not
expect to see any impacts on existing usages of NumPy, besides warnings or
errors related to new deprecations in ``np.char``.

Detailed description
--------------------

Here we provide a detailed description of the version of ``StringDType`` we
would like to include in NumPy. This is mostly identical to the prototype, but
has a few differences that are impossible to implement in a DType that lives
outside of NumPy.

First, we describe the Python API for instantiating ``StringDType`` instances.
Second, we describe the in-memory representation, heap allocation strategy, and
thread safety concerns. This is followed by a description of the missing data
handling support and support for strict string type checking for array
elements. We next discuss the cast and ufunc implementations we will define and
discuss our plan for the string manipulation functions in ``np.char``. Finally
we describe out plan to update the ``npy`` and ``npz`` file formats to support
writing sidecar data.

Python API for ``StringDType``
******************************

The new DType will be accessible via the ``np.dtypes`` namespace:

  >>> from numpy.dtypes import StringDType
  >>> dt = StringDType()
  >>> dt
  numpy.dtypes.StringDType()

In addition, we propose reserving the character ``"T"`` (short for text) for
usage with ``np.dtype``, so the above would be identical to:

  >>> np.dtype("T")
  numpy.dtypes.StringDType()

In principle we do not need to reserve a character code and there is a desire to
move away from character codes. However, a substantial amount of downstream code
relies on checking DType character codes to discriminate between builtin NumPy
DTypes, and we think it would harm adoption to require users to refactor their
DType-handling code if they want to use ``StringDType``.

``StringDType`` can be used out of the box to represent strings of arbitrary
length in a NumPy array:

  >>> data = ["this is a very long string", "short string"]
  >>> arr = np.array(data, dtype=StringDType())
  >>> arr
  array(['this is a very long string', 'short string'], dtype=StringDType())

Note that unlike fixed-width strings, ``StringDType`` is not parameterized by
the maximum length of an array element, arbitrarily long or short strings can
live in the same array without needing to reserve storage for padding bytes in
the short strings.

The ``StringDType`` class will be a synonym for the default ``StringDType``
instance when the class is passed as a ``dtype`` argument in the NumPy Python
API. We have already converted most of the API surface to work like this, but
there are still a few spots that have not yet been converted and it's likely
third-party code has not been converted, so we will not emphasize this in the
docs. Emphasizing that ``StringDType`` is a class and ``StringDType()`` is an
instance is a more forward-looking API that the rest of the NumPy DType API can
move towards now that DType classes are importable from the ``np.dtypes``
namespace, so we will include an explicit instantiation of a ``StringDType``
object in the documentation even if it is not strictly necessary.

We propose associating the python ``str`` builtin as the DType's scalar type:

  >>> StringDType.type
  <class 'str'>

While this does create an API wart in that the mapping from builtin DType
classes to scalars in NumPy will no longer be one-to-one (the ``unicode``
DType's scalar type is ``str``), this avoids needing to define, optimize, or
maintain a ``str`` subclass for this purpose or other hacks to maintain this
one-to-one mapping. To maintain backward compatibility, the DType detected for a
list of python strings will remain a fixed-width unicode string.

As described below, ``StringDType`` supports two parameters that can adjust the
runtime behavior of the DType. We will not attempt to support parameters for the
dtype via a character code. If users need an instance of the DType that does not
use the default parameters, they will need to instantiate an instance of the
DType using the DType class.

We will also extend the ``NPY_TYPES`` enum in the C API with an ``NPY_VSTRING``
entry (there is already an ``NPY_STRING`` entry). This should not interfere with
legacy user-defined DTypes since the integer type numbers for these data types
begin at 256. In principle there is still room for hundreds more builtin
DTypes in the integer range available in the ``NPY_TYPES`` enum.

.. _memory:

Managing Heap Allocations
*************************

Since NumPy has no first-class support for ragged arrays, there is no way for a
variable-length string data type to store data in the array storage
buffer. Moreover, the assumption that each element of a NumPy array is a
constant number of bytes wide in the array buffer is deeply ingrained in NumPy
and libraries in the wider PyData ecosystem. It would be a substantial amount
of work to add support for ragged arrays in NumPy and downstream libraries, far
beyond the scope of adding support for variable-length strings.

Instead, we propose relaxing the requirement that all array data are stored in
the array buffer or inside of python objects. This DType would extend the
existing concept of an array of references in NumPy beyond the ``object`` DType
to include arrays that store data in sidecar heap-allocated buffers and use the
array to store metadata for the heap allocation.

Memory Layout and Small String Optimization
+++++++++++++++++++++++++++++++++++++++++++

TODO: Add a figure illustrating the memory layout of the various views, flags

Each array element would contain the contents of an opaque but sized C struct
with the following layout:

.. code-block:: C

   struct npy_packed_static_string {
       char packed_buffer[2*sizeof(size_t)];
   };

So it is only guaranteed to be the same size in memory as two `size_t` fields,
with no guarantees about data layout or precisely how the data are
stored. The raw string data can only be accessed once the packed string data are
loaded into an unpacked string with the following layout:

.. code-block:: C

   struct npy_unpacked_static_string {
       size_t size;
       const char *buf;
   };
   
Where ``size`` is the size, in bytes, of the string and ``buf`` is a const
pointer to the beginning of a UTF-8 encoded bytestream containing string
data. This is a *read-only* view onto the string, we will not expose a public
interface for modifying these strings. We do not append a trailing null
character to the byte stream, so users attempting to pass the ``buf`` field to
an API expecting a C string must create a copy with a trailing null. This choice
also means that unlike the fixed-width strings in NumPy, ``StringDType`` array
entries can contain arbitrary embedded or trailing null characters.

Internally, we represent the string as a union, with the following definition on
little-endian architectures:

.. code-block:: c

   typedef struct _npy_static_vstring_t {
      size_t offset;
      size_t size_and_flags;
   } _npy_static_string_t;

   typedef struct _short_string_buffer {
      char buf[sizeof(_npy_static_string_t) - 1];
      unsigned char size_and_flags;
   } _short_string_buffer;

   typedef union _npy_static_string_u {
    _npy_static_string_t vstring;
    _short_string_buffer direct_buffer;
   } _npy_static_string_u;

The `_npy_static_vstring_t` representation is most useful for representing
strings living on the heap directly or in an arena allocation, with the
``offset`` field either containing a ``size_t`` representation of the address
directly, or an integer offset into an arena allocation. The
``_short_string_buffer`` representation is most useful for the small string
optimization, with the string data stored in the ``buf`` field and the size in
the ``size_and_flags`` field. In both cases the ``size_and_flags`` field stores
both the ``size`` of the string as well as bitflags. Small strings store the
size in the final four bits of the buffer, reserving the first four bits of
``size_and_flags`` for flags. Heap strings or strings in arena allocations use
the most significant byte for flags, reserving the leading bytes for the string
size. This limits the maximum size string allowed to be stored in the array,
particularly on 32 bit systems where the limit is only 16 megabytes per string.

On big-endian systems, the layout is reversed, with the ``size_and_flags`` field
appearing first in the structs. This allows the implementation to always use the
most significant bits of the ``size_and_flags`` field for flags. The
endian-dependent layouts of these structs is an implementation detail and is not
publicly exposed in the API.

Whether or not a string is stored directly on the arena buffer or in the heap is
signaled by setting the ``NPY_STRING_SHORT`` flag on the string data. Because
the maximum size of a heap-allocated string is limited, this flag can never be
set for a valid heap string.

Arena Allocator
+++++++++++++++

Strings longer than the small string limit (15 bytes on 64 bit systems and 7
bytes on 32 bit systems) are stored on the heap. The bookkeeping for the
allocations is managed by an arena allocator attached to the ``StringDType``
instance associated with an array. The allocator will be exposed publicly as an
opaque ``npy_string_arena`` struct.

Internally, the ``npy_string_arena`` struct has the following layout:

.. code-block:: c

    struct npy_string_allocator {
        npy_string_malloc_func malloc;
        npy_string_free_func free;
        npy_string_realloc_func realloc;
        npy_string_arena arena;
    };

This allows us to group memory-allocation functions together and choose
different allocation functions at runtime if we desire.  The
``npy_string_arena`` struct has the following layout:

.. code-block:: c

    struct npy_string_arena {
        size_t cursor;
        size_t size;
        char *buffer;
    };

Where ``buffer`` is a pointer to the beginning of a heap-allocated arena,
``size`` is the size of that allocation, and ``cursor`` is the location in the
arena where the last arena allocation ended. The arena is filled using an
exponentially expanding buffer, allowing amortized O(1) insertion.

We do not expose the ``npy_string_arena`` struct or the underlying heap memory
allocations publicly.

Using a per-array arena allocator ensures that the string buffers for nearby
array elements are usually nearby on the heap. We do not guarantee that
neighboring array elements are contiguous on the heap to support the small
string optimization, missing data, and allow mutation of array entries. See
below for more discussion on how these topics affect the memory layout. Use of
the allocator is guarded by a mutex, see below for more discussion about thread
safety.

We plan to add an interface for packing, unpacking, allocating, freeing,
comparing, and copying strings with this layout via the arena allocator to the
public NumPy C API to ease downstream integration. All items in the API we
propose to add are prepended with ``npy_string``, ``NpyString`` or
``NPY_STRING`` for public typedefs, functions, or macros, respectively. This C
API is used to implement the DType using the NumPY DType API. The header and
implementation for the API only uses C standard library headers and does not
depend on the CPython or NumPy C API. See the ``static_string.h`` header in the
prototype for a full API listing and documentation.

Empty Strings and Missing Data
++++++++++++++++++++++++++++++

The empty string will be defined to be a zero-filled
``npy_packed_static_string``. This has the benefit that newly created array
buffer returned by ``calloc`` will be an array filled with empty strings by
construction. Missing strings will have an identical representation, except they
will have a flag, ``NPY_STRING_MISSING`` set in the flags field. Users will need
to check if a string is null before accessing an unpacked string buffer and we
have set up the C API in such a way as to force null-checking whenever a string
is unpacked.

Both missing and empty strings are stored directly in the array buffer and do
not need to reserve any space in the arena buffer.

Mutation and Thread Safety
++++++++++++++++++++++++++

Mutation introduces the possibility of data races and use-after-free errors when
an array is accessed and mutated by multiple threads. Additionally, if we
allocate mutated strings in the arena buffer and mandate contiguous storage
where the old string is replaced by the new one, mutating a single string may
trigger reallocating the arena buffer for the entire array. This is a
pathological performance degradation compared with object string arrays.

One solution would be to disable mutation, but inevitably there will be
downstream uses of object string arrays that mutate array elements that we would
like to support.

Instead, we have opted to pair the ``npy_string_allocator`` instance attached to
``StringDType`` instances with a ``PyThread_type_lock`` mutex. Any function in
the static string C API that allows manipulating heap-allocated data accepts an
``allocator`` argument. To use the C API correctly, a thread must acquire the
allocator mutex before any usage of the ``allocator``. This prevents parallel
access to the heap memory used by string arrays.

The ``PyThread_type_lock`` mutex is relatively heavyweight and does not provide
more sophisticated locking primitives that allow multiple simultaneous
readers. As part of the GIL-removal project, CPython is adding new
synchronization primitives to the C API for projects like NumPy to make use
of. When this happens, we can update the locking strategy to allow multiple
simultaneous reading threads, along with other fixes for threading bugs in NumPy
that will be needed once the GIL is removed.

Cython Support and the Buffer Protocol
++++++++++++++++++++++++++++++++++++++

It's impossible for ``StringDType`` to support the Python buffer protocol, so
Cython will not support idiomatic typed memoryview syntax for ``StringDType``
arrays unless special support is added in Cython in the future. We have some
preliminary ideas for ways to either update the buffer protocol [10]_ or make
use of the Arrow C data interface [11]_ to expose NumPy arrays for DTypes that
don't make sense in the buffer protocol, but those efforts will likely not come
to fruition in time for NumPy 2.0. This means adapting legacy Cython code that
uses arrays of fixed-width strings to work with ``StringDType`` will be
non-trivial. Adapting code that worked with object string arrays should be
straightforward since object arrays aren't supported by the buffer protocol
either and will likely have no types or have ``object`` type in Cython.

We will add cython ``nogil`` wrappers for the C API functions added to work with
packed strings to ease integration with downstream cython code.

Missing data support
********************

Missing data can be represented using a sentinel:

  >>> dt = StringDType(na_object=np.nan)
  >>> arr = np.array(["hello", nan, "world"], dtype=dt)
  >>> arr
  array(['hello', nan, 'world'], dtype=StringDType(na_object=nan))
  >>> arr[1]
  nan
  >>> np.isnan(arr[1])
  True
  >>> np.isnan(arr)
  array([False,  True, False])
  >>> np.empty(3, dtype=dt)
  array([nan, nan, nan])

We only propose supporting user-provided sentinels. By default, empty arrays
will be populated with empty strings:

  >>> np.empty(3, dtype=StringDType())
  array(['', '', ''], dtype=StringDType())

By only supporting user-provided missing data sentinels, we avoid resolving
exactly how NumPy itself should support missing data and the correct semantics
of the missing data object, leaving that up to users to decide. However, we *do*
detect whether the user is providing a NaN-like missing data value, a string
missing data value, or neither. We explain how we handle these cases below.

A cautious reader may be worried about the complexity of needing to handle three
different categories of missing data sentinel. The complexity here is reflective
of the flexibility of object arrays and the downstream usage patterns we've
found. Some users want comparisons with the sentinel to error, so they use
``None``. Others want comparisons to succeed and have some kind of meaningful
ordering, so they use some arbitrary, hopefully unique string. Other users want
to use something that acts like NaN in comparisons and arithmetic or is
literally NaN so that NumPy operations that specifically look for exactly NaN
work and there isn't a need to rewrite missing data handling outside of
NumPy. We believe it is possible to support all this, but it requires a bit of
hopefully manageable complexity.

NaN-like Sentinels
++++++++++++++++++

A NaN-like sentinel returns itself as the result of arithmetic operations. This
includes the python ``nan`` float and the Pandas missing data sentinel
``pd.NA``. We choose to make NaN-like sentinels inherit these behaviors in
operations, so the result of addition is the sentinel:

  >>> dt = StringDType(na_object=np.nan)
  >>> arr = np.array(["hello", np.nan, "world"], dtype=dt)
  >>> arr + arr
  array(['hellohello', nan, 'worldworld'], dtype=StringDType(na_object=nan))

We also chose to make a NaN-like sentinel sort to the end of the array,
following the behavior of sorting an array containing ``nan``.

  >>> np.sort(arr)
  array(['hello', 'world', nan], dtype=StringDType(na_object=nan))

String Sentinels
++++++++++++++++

A string missing data value is an instance of ``str`` or subtype of ``str`` and
will be used as the default value for empty arrays:

  >>> arr = np.empty(3, dtype=StringDType(na_object='missing'))
  >>> arr
  array(['missing', 'missing', 'missing'])

If such an array is passed to a string operation or a cast, "missing" entries
will be treated as if they have a value given by the string sentinel:

  >>> np.char.upper(arr)
  array(['MISSING', 'MISSING', 'MISSING'])

Comparison operations will similarly use the sentinel value directly for missing
entries. This is the primary usage of this pattern we've found in downstream
code, where a missing data sentinel like ``"__nan__"`` is passed to a low-level
sorting or partitioning algorithm.

Other Sentinels
+++++++++++++++

Any other python object will raise errors in operations or comparisons, just as
``None`` does as a missing data sentinel for object arrays currently:

  >>> dt = StringDType(na_object=None)
  >>> np.sort(np.array(["hello", None, "world"], dtype=dt))
  ValueError: Cannot compare null that is not a string or NaN-like value

Since comparisons need to raise an error, and the NumPy comparison API has no
way to signal value-based errors during a sort without holding the GIL, sorting
arrays that use arbitrary missing data sentinels will hold the GIL. We may also
attempt to relax this restriction by refactoring NumPy's comparison and sorting
implementation to allow value-based error propagation during a sort operation.

Implications for DType Inference
++++++++++++++++++++++++++++++++

If, in the future, we decide to break backward compatibility to make
``StringDType`` the default DType for ``str`` data, the support for arbitrary
objects as missing data sentinels may seem to pose a problem for implementing
DType inference. However, given that initial support for this DType will require
using the DType directly and will not be able to rely on NumPy to infer the
DType, we do not think this will be a major problem for downstream users of the
missing data feature. To use ``StringDType``, they will need to update
their code to explicitly specify a DType when an array is created, so if NumPy
changes DType inference in the future, their code will not change behavior and
there will never be a need for missing data sentinels to participate in DType
inference.

Coercing non-strings
********************

By default, non-string data are coerced to strings:

  >>> np.array([1, object(), 3.4], dtype=StringDType())
  array(['1', '<object object at 0x7faa2497dde0>', '3.4'], dtype=StringDType())

If this behavior is not desired, an instance of the DType can be created that
disables string coercion:

  >>> np.array([1, object(), 3.4], dtype=StringDType(coerce=False))
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ValueError: StringDType only allows string data when string coercion
  is disabled

This allows strict data validation in the same pass over the data NumPy uses to
create the array without a need for downstream libraries to implement their own
string validation in a separate, expensive, pass over the input array-like. We
have chosen not to make this the default behavior to follow NumPy fixed-width
strings, which coerce non-strings.

Casts, ufunc support, and string manipulation functions
*******************************************************

A full set of round-trip casts to the builtin NumPy DTypes will be available. In
addition, we will add implementations for the comparison operators as well as
an ``add`` loop that accepts two string arrays, ``multiply`` loops that
accept string and integer arrays, and an ``isnan`` loop. The ``isnan`` ufunc
will return ``True`` for entries that are NaN-like sentinels and ``False``
otherwise. Comparisons will sort data in order of unicode code point, as is
currently implemented for the fixed-width unicode DType. In the future NumPy or
a downstream library may add locale-aware sorting, case folding, and
normalization for NumPy unicode strings arrays, but we are not proposing adding
these features at this time.

Two ``StringDType`` instances are considered identical if they are created with
the same ``na_object`` and ``coerce`` parameter. We propose checking for unequal
``StringDType`` instances in the ``resolve_descriptors`` function of binary
ufuncs that take two string arrays and raising an error if an operation is
performed with unequal ``StringDType`` instances.

``np.strings`` namespace
************************

String operations will be available in a ``np.strings`` namespace that will
initially be populated with the ufunc-like functions in ``np.char``:

  >>> np.strings.upper((np.array(["hello", "world"], dtype=StringDType())
  array(['HELLO', 'WORLD'], dtype=StringDType())

In addition to outright removing the more obscure or deprecated functionality in
``np.char`` as part of the NumPy 2.0 API cleanup, we will deprecate and
eventually remove the existing string manipulation functions in ``np.char`` in
favor of ``np.strings``.

We feel ``np.strings`` is a more intuitive name and that NumPy 2.0 is as good a
time as any to fully eject the legacy baggage of ``chararray``.

This proposal is less for the sake of adding functionality and more
window-dressing than the other proposals in this NEP. If there isn't an appetite
for renaming the namespace, we could clean up ``np.char`` as planned for NumPy
2.0 and leave the ufunc-like functions in-place.

Serialization
*************

Since string data are stored outside the array buffer, serialization requires an
update to the ``npy`` file format, which currently can only include fixed-width
data in the array buffer. We propose defining format version 4.0, which adds an
additional optional ``sidecar_size`` header key that corresponds to the size, in
bytes, of an optional sidecar field that is written to disk following the array
data. If no sidecar storage is required, the writer will default to the current,
more widely compatible, file format and will not write a ``sidecar_size`` field
to the header. This also enables storage of arbitrary user-defined data types
once API hooks are added that allow a DType to serialize to the sidecar data and
deserialize from the loaded sidecar data.

This is an improvement over the current situation with object string arrays,
which can only be saved to an ``npy`` file using the ``allow_pickle=True``
option. Serializing arbitrary python objects requires the use of ``pickle``, so
there is no safe way to share untrusted ``npy`` files containing object string
arrays. This means users of object string arrays adopting ``StringDType`` will
also gain an officially supported way to safely share string data or load
variable-width string data from an untrusted source.

For cases where pickle support is required, support for pickling and unpickling
string arrays will also be implemented.

Related Work
------------

The main comparable prior art in the Python ecosystem is PyArrow arrays, which
support variable length strings via Apache Arrow's variable sized binary layout
[12]_. In this approach, the array buffer contains integer offsets that index
into a sidecar storage buffer. This allows a string array to be created using
only two heap allocations, leaves adjacent strings in the array contiguous in
memory, provides good cache locality, and enables straightforward SIMD
optimization. Mutation of string array elements isn't allowed and PyArrow only
supports 1D arrays, so the design space is somewhat different from NumPy.

Julia stores strings as UTF-8 encoded byte buffers. There is no special
optimization for string arrays in Julia, and string arrays are represented as
arrays of pointers in memory in the same way as any other array of sequences or
containers in Julia.

The tensorflow library supports variable-width UTF-8 encoded strings,
implemented with ``RaggedTensor``. This makes use of first-class support for
ragged arrays in tensorflow.


Implementation
--------------

A prototype version of ``StringDType`` using the experimental DType API is
available in the ``numpy-user-dtypes`` repository [13]_. Currently, most of the
functionality proposed above for the version of the DType we would like to add
to NumPy is already functioning. The major missing piece is the :ref:`arena
allocator <memory>` described above; currently memory is allocated using
``malloc`` and ``free``. We have an implementation plan for adding the arena
allocator [14]_.

We are focusing on implementation so there is no documentation yet, but the
tests illustrate what has been implemented [15]_. Note that if you are
interested in testing out the prototype, you will need to set the
``NUMPY_EXPERIMENTAL_DTYPE_API`` environment variable at runtime to enable the
experimental DType API in NumPy.

We have created a development branch of Pandas that supports creating Pandas
data structures using ``StringDType`` [16]_. This illustrates the refactoring
necessary to support ``StringDType`` in downstream libraries that make
substantial use of object string arrays.

While the NEP is being discussed, we plan on finishing the arena allocator
implementation and refactoring the prototype to minimize heap allocations and
keep string data contiguous in memory as much as possible.

If accepted, the bulk of the remaining work of this NEP is in preparing NumPy
for the DType, the work of adding the DType to NumPy itself, writing
documentation for the new DType, and updating the existing NumPy documentation
where appropriate. The steps will be as follows:

* Create an ``np.strings`` namespace, move the ufunc-like functions in
  ``np.char`` there, and deprecate ``np.char``.

* Formalize the update to the ``npy`` and ``npz`` serialization formats. We
  will add a hook to the DType API so that user DTypes can make use of the new
  sidecar storage capabilities and add support in ``StringDType`` while it is
  still outside the NumPy source tree.

* Move the ``StringDType`` implementation from an external extension module
  into NumPy, refactoring NumPy where appropriate. This new DType will be
  added in one large pull request including documentation updates. Where
  possible, we will extract fixes and refactorings unrelated to
  ``StringDType`` into smaller pull requests before issuing the main pull
  request.

* Deal with remaining issues in NumPy related to new DTypes. In particular,
  we are already aware that remaining usages of ``copyswap`` in ``NumPy``
  should be migrated to use a cast or an as-yet-to-be-added single-element
  copy DType API slot. We also need to ensure that DType classes can be used
  interchangeably with DType instances in the Python API everywhere it makes
  sense to do so and add useful errors in all other places DType instances
  can be passed in but DType classes don't make sense to use.

The third step depends on the first two steps, but the first two steps can be
done in parallel. The fourth step can also be done in parallel to the other
steps, but the process of adding the DType to NumPy will likely shake out more
issues.

We are hopeful that this work can be completed in time for NumPy 2.0 and we will
certainly complete the ``np.char`` migration before then. However, if need be,
the new DType and the addition to the ``npy`` file format can slip to NumPy 2.1
and is not required for the API changes slated for NumPy 2.0. That said, it
would be nice to have some big new features too!

Alternatives
------------

The main alternative is to maintain the status quo and offer object arrays as
the solution for arrays of variable-length strings. While this will work, it
means immediate memory usage and performance improvements, as well as future
performance improvements, will not be implemented anytime soon and NumPy will
lose relevance to other ecosystems with better support for arrays of textual
data.

We do not see the proposed DType as mutually exclusive to an improved
fixed-width binary DType that can represent arbitrary binary data or text in any
encoding and adding such a DType in the future will be easier once overall
support for string data in NumPy has improved after adding ``StringDType``.

Discussion
----------

- https://github.com/numpy/numpy/pull/24483
- https://mail.python.org/archives/list/numpy-discussion@python.org/thread/IHSVBZ7DWGMTOD6IEMURN23XM2BYM3RG/


References and Footnotes
------------------------

.. [1] https://github.com/pandas-dev/pandas/pull/52711
.. [2] https://mail.python.org/pipermail/numpy-discussion/2017-April/thread.html#76668
.. [3] https://mail.python.org/archives/list/numpy-discussion@python.org/message/WXWS4STFDSWFY6D7GP5UK2QB2NFPO3WE/
.. [4] https://mail.python.org/archives/list/numpy-discussion@python.org/message/DDYXJXRAAHVUGJGW47KNHZSESVBD5LKU/
.. [5] https://mail.python.org/archives/list/numpy-discussion@python.org/message/6TNJWGNHZF5DMJ7WUCIWOGYVZD27GQ7L/
.. [6] https://numpy.org/neps/roadmap.html#extensibility
.. [7] https://github.com/h5py/h5py/issues/624#issuecomment-676633529
.. [8] https://github.com/PyTables/PyTables/issues/499
.. [9] https://github.com/elliotgoodrich/SSO-23#sso-23
.. [10] https://discuss.python.org/t/buffer-protocol-and-arbitrary-data-types/26256
.. [11] https://arrow.apache.org/docs/format/CDataInterface.html
.. [12] https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
.. [13] https://github.com/numpy/numpy-user-dtypes/tree/main/stringdtype
.. [14] https://github.com/numpy/numpy-user-dtypes/issues/76#issuecomment-1650367922
.. [15] https://github.com/numpy/numpy-user-dtypes/tree/main/stringdtype/tests
.. [16] https://github.com/ngoldbaum/pandas/tree/stringdtype

Copyright
---------

This document has been placed in the public domain.
