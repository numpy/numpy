================================================
NEP 40 — Legacy Datatype Implementation in NumPy
================================================

:title: Legacy Datatype Implementation in NumPy
:Author: Sebastian Berg
:Author: Add...
:Status: Draft
:Type: Informational
:Created: 2019-07-17


Abstract
--------

As a preparation to further NumPy enhancement proposals 41, 42, and 43. This
NEP details the current status of NumPy datatypes as of NumPy 1.18.
It describes some of the technical aspects and concepts necessary to
motivate the following proposals.


Detailed Description
--------------------

Parametric Datatypes
^^^^^^^^^^^^^^^^^^^^

Some datatypes are inherently *parametric*, which is similar to the current
notion of *flexible* data types.
Currently, flexible is defined as a class ``np.flexible`` for scalars, which
is a superclass for the data types of variable length (string, bytes,
and void), this distinction is similarly exposed by the C-Macros
``PyDataType_ISFLEXIBLE`` and ``PyTypeNum_ISFLEXIBLE``.
For strings, ``"S8"`` can represent more strings than ``"S4"``.

The basic numerical datatypes are naturally represented as not flexible and
not parametric: Float64, Float32, etc. do have a byte order, but the described
values are unaffected by it, and it is always possible to cast them to the
native, canonical representation without loss of any information.

The concept of flexibility can be generalized to parametric datatypes.
For example the private ``AdaptFlexibleDType`` function also accepts the
naive datetime dtype as input to find the correct time unit.
The datetime dtype is thus parametric not in the storage, but instead in
what the stored value represents.
Currently ``np.can_cast("datetime64[s]", "datetime64[ms]", casting="safe")``
returns true, although it is unclear that this is desired or generalizes
to possible future data types such as physical units.

Thus we have data types (mainly strings) with the properties that:

1. Casting is not always safe (``np.can_cast("S8", "S4")``)
2. Array coercion should be able to discover the exact dtype, such as for
   ``np.array(["str1", 123.], dtype="S")`` where NumPy discovers the
   resulting dtype as ``"S5"``.
   (Without ``dtype="S"`` such behaviour is currently ill defined [gh-15327].)
   A form similar to ``dtype="S"`` is ``dtype="datetime64"`` which can
   discover the unit: ``np.array(["2017-02"], dtype="datetime64")``.

This notion highlights that some datatypes are more complex than the basic
numerical ones, which currently creates issues mainly in the implementation
of universal functions.


Dispatching of Universal Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently the dispatching of universal function (ufuncs) is limited for
user defined dtypes.
Typically, dispatching is done by finding the first loop for which all inputs can
be cast to safely (see also the current implementation section).

However, in some cases this is problematic and thus explicitly not allowed.
For example the ``np.isnat`` function is currently only defined for
datetime and timedelta.
Even though integers are defined to be safely castable to timedelta.
If this was not the case, calling
``np.isnat(np.array("NaT", "timedelta64").astype("int64"))`` would currently
return true, although the integer input array has no notion of "not a time".
If a universal function, such as most function in ``scipy.special``, is only
defined for ``float32`` and ``float64`` it will currently automatically
cast a ``float16`` silently to ``float32`` (similarly for any integer input).
This ensures successful execution, but allows a change in the output dtype
when support for new data types is added to a ufunc.

With respect to to user defined dtypes, dispatching works largely similar,
however, it enforces an exact match of the datatypes (type numbers).
Because the current method is separate and fairly slow, it will only match
loops defined for datatypes already existing in the inputs.
This can be a limitation: a function such as
``rational_divide(int, int) -> rational`` can only work easily if the user
calls it using ``rational_divide(int, int, dtype=rational)``.

For NumPy datatypes the order in which loops are registered is currently important.
However, this is only reliable if all loops are added when the ufunc is first defined.
Additional loops added when a new user datatypes is imported
must not be sensitive to the order in which imports occur.

There are two main approaches to better define the type resolution for user
defined types:

1. Allow for user dtypes to directly influence the loop selection.
   For example they may provide a function which return/select a loop
   when there is no exact matching loop available.
2. Define a total ordering of all implementations/loops, probably based on
   "safe casting" semantics, or semantics similar to that.

While option 2 may be less complex to reason about it remains to be seen
whether it is sufficient for all (or most) use cases.


Inner Loop and Error Handling in UFuncs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the correct implementation/loop is found, UFuncs currently mainly call
a single *inner-loop function*, which may be called multiple times to do
the full calculation.

A main issue is that especially parametric datatypes require passing
additional information to the inner-loop function to decide how to interpret
the data.
This is the reason why currently no universal functions for strings dtypes
exist (although technically possible within NumPy itself).
Note that it is currently possible to pass in the input array objects
(which in turn hold the datatypes when no casting is necessary).
However, the full array information should not be required and currently the
arrays are passed in before any casting occurs.
The feature is unused within NumPy and no known user exists.

Another issue is the error reporting from within the inner-loop function.
There exist currently two ways to do this:

1. by setting a Python exception
2. using the CPU floating point error flags.

Both of these are checked before returning to the user.
However, many integer functions currently can set neither of these errors,
so that checking the floating point error flags is unnecessary overhead.
On the other hand, there is no way to stop the iteration or pass out error
information which does not use the floating point flags or requires to hold
the Python global interpreter lock (GIL).

It seems necessary to provide more control to authors of inner loop functions.
This means allowing users to pass in and out information from the inner-loop
function more easily, while *not* providing the input array objects.
Most likely this will involve:

* Allowing the execution of additional code before the first and after
  the last inner-loop call.
* Returning an integer value from the inner-loop to allow stopping the
  iteration early and possibly propagate error information.
* Possibly, to allow specialized inner-loop selections. For example currently
  ``matmul`` and many reductions will execute optimized code for certain inputs.
  It may make sense to allow selecting such optimized loops beforehand.
  Allowing this may also help to bring casting (which uses this heavily) and
  ufunc implementations closer.

The issues surrounding the inner-loop functions have been discussed in some
detail in the github issue 12518 [gh-12518]_.


Value Based Casting
^^^^^^^^^^^^^^^^^^^

Casting is typically defined between two types:
A type is considered to cast safely to a second type when the second type
can represent all values of the first faithfully.
However, NumPy currently NumPy may inspect the actual value to decide
whether casting is safe or not [value_based]_.

This is useful for example in expressions such as::

    arr = np.array([1, 2, 3], dtype="int8")
    result = arr + 5
    assert result.dtype == np.dtype("int8")
    # If the value is larger, the result will change however:
    result = arr + 500
    assert result.dtype == np.dtype("int16")

In this expression, the python value (which originally has no datatype) is
represented as an ``int8`` or ``int16`` (the smallest possible data type).

NumPy currently does this even for NumPy scalars and zero dimensional arrays,
so that replacing ``5`` with ``np.int64(5)`` or ``np.array(5, dtype="int64")``
will lead to the same results, and thus ignores the existing datatype.
The same logic also applies to floating point scalars, which are allowed to
lose precision.
The behavior is not used when both inputs are scalars.

Although, the above behavior is defined in terms of casting the a given
scalar value as exposed also through ``np.result_type``, the main importance
is in the ufunc dispatching which currently relies on safe casting semantics.


Issues
""""""

There appears to be some agreement that the current method is
not desirable for values that have a datatype,
but may useful for pure python integers or floats as in the first
example.
However, any change of the datatype system and universal function dispatching
must initially fully support the current behavior.
A main difficulty is that for example the value ``156`` can be represented
by ``np.uint8`` and ``np.int16``.
It depends on the context which is considered the "minimal" representation
(for ufuncs the context may be given by the loop order).


The Object Datatype
^^^^^^^^^^^^^^^^^^^

The object datatype currently serves as a generic fallback for any value
which is not representable otherwise.
However, due to not having a well defined type, it has some issues,
for example when an array is filled with Python sequences::

    >>> l = [1, [2]]
    >>> np.array(l, dtype=np.object_)
    array([1, list([2])], dtype=object)  # a 1d array

    >>> a = np.empty((), dtype=np.object_)
    >>> a[...] = l
    ValueError: assignment to 0-d array  # ???
    >>> a[()] = l
    >>> a
    array(list([1, [2]]), dtype=object)

Further, without a well defined type, functions such as ``isnan()`` or ``conjugate()``
do not necessarily work for example for an array holding decimal values since they cannot
be specialized for :class:`decimal.Decimal`.
To improve this situation it seems desirable to make it easy to create
object dtypes that represent a specific python datatype and stores its object
inside the array in form of pointers.
Unlike most datatypes, Python objects require reference counting.
This means that additional methods to increment/decrement references and
visit all objects must be defined.
In practice, for most use cases it is sufficient to limit the creation of such
datatypes so that all functionality related to Python references is private
to NumPy.

Creating NumPy datatypes that match builtin Python objects also creates a few problems
that require more thoughts and discussion.
These issues do not need to solved right away:

* NumPy currently returns *scalars* even for array input in some cases, in most
  cases this works seamlessly. However, this is only true because the NumPy
  scalars behave much like NumPy arrays, a feature that general Python objects
  do not have.
* Seamless integration probably requires that ``np.array(scalar)`` finds the
  correct DType automatically since some operations (such as indexing) are
  always desired to return the scalar.
  This is problematic if multiple users independently decide to implement
  for example a DType for ``decimal.Decimal``.


Current Implementation
----------------------

These sections give a very brief overview of the current implementation, it is
not meant to be a comprehensive explanation, but a basic reference for further
technical NEPs.

Current ``dtype`` Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently ``np.dtype`` is a Python class with its instances being the
``np.dtype(">float64")``, etc. instances.
To set the actual behaviour of these instances, a prototype instance is stored
globally and looked up based on the ``dtype.typenum``.
This prototype instance is then copied (if necessary) and modified for
endianess.
For parametric datatypes (strings, void, datetime, and timedelta) additionally
the string lengths, fields, or datetime unit needs to be set.
All current datatypes within NumPy further support setting a metadata field
during creation which can be set to an arbitrary dictionary value, but seems
rarely used in practice (one recent and prominent user is h5py).

Many datatype specific functions are defined within a C structure called
:c:type:`PyArray_ArrFuncs`, which is part of each ``dtype`` instance and
has a similarity to Pythons ``PyNumberMethods``.
For user defined datatypes this structure is defined by the user, making
ABI compatible changes changes impossible.
This structure holds important information such as how to copy, cast,
and provides functions, such as comparing elements, converting to bool, or sorting.
Since some of these functions are vectorized operations, operating on more than
one element, they fit the model of ufuncs and do not need to be defined on the
datatype in the future.
For example the ``np.clip`` function was previously implemented using
``PyArray_ArrFuncs`` and is now implemented as ufuncs.

Discussion and Issues
"""""""""""""""""""""

A further issue with the current implementation is that, unlike methods,
they are not passed an instance of the dtype when called.
Instead, in many cases, the array which is being operated on is passed in
and typically only used to extract the datatype again.
A future API should likely stop passing in the full array object.
Since it will be necessary to fall back to the old definitions for
backward compatibility, the array object may not be available.
However, passing a "fake" array in which mainly the datatype is defined
is probably be a sufficient workaround
(see backward compatibility; alignment information may sometimes also be desired).

Although not extensively used outside of NumPy itself, the currently
``PyArray_Descr`` is a public structure.
This is especially also true for the ``ArrFunctions`` structure stored in
the ``f`` field.
Due to compatibility they may need to remain supported for a very long time,
with the possibility of replacing them by functions that dispatch to a newer API.

However, in the long run access to these structures will probably have to
be deprecated.


NumPy Scalars and Type Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a side note to the above datatype implementation, unlike the datatypes,
the NumPy scalars currently do provide a type hierarchy, including abstract
types such as ``np.inexact`` (see figure below).
In fact, some control flow within NumPy currently uses
``issubclass(a.dtype.type, np.inexact)``.

.. figure:: _static/nep-0040_dtype-hierarchy.png

   **Figure:** Hierarchy of type objects reproduced from the reference
   documentation. Some aliases such as ``np.intp`` are excluded. Datetime
   and timedelta are not shown.

NumPy scalars try to mimic zero dimensional arrays with a fixed datatype.
For the numerical (and unicode) datatypes, they are further limited to
native byte order.


Current Implementation of Casting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the main features which datatypes need to support is casting between one
another using ``arr.astype(new_dtype, casting="unsafe")``, or during execution
of ufuncs with different types (such as adding integer and floating point numbers).

Casting tables determine whether casting is possible or not.
However, these cannot handle the parametric dtypes such as strings.
The logic for parametric datatypes is defined mainly in ``PyArray_CanCastTo``
and currently cannot be customized for user defined datatypes.

The actual casting has two distinct parts:

1. ``copyswap``/``copyswapn`` are defined for each dtype and can handle
   byte-swapping for non-native byte orders as well as unaligned memory.
2. The generic casting code is provided by C functions which know how to
   cast aligned and contiguous memory from one dtype to another
   (both in native byte order).
2. C-level functions can be registered to cast aligned and contiguous memory
   from one dtype to another.
   The function may be provided with both arrays (although the parameter
   is sometimes ``NULL`` for scalars).
   NumPy will ensure that these functions receive native byte order input.
   The current implementation stores the functions either in a C-array
   on the datatype which is cast, or in a dictionary when casting to a user
   defined datatype.

When casting (small) buffers will be used when necessary to ensure
contiguity, alignment or native byte order.
In this case first ``copyswapn`` is called to and ensures that the cast function
can handle the input.
Generally NumPy will thus perform casting as chain of the three functions
``in_copyswapn -> castfunc -> out_copyswapn``.

The above multiple functions are wrapped into a single function (with metadata)
that handles the cast and is used for example during the buffered iteration used
by ufuncs.
This is the mechanism that is always used for user defined datatypes.
For most dtypes defined within NumPy itself, more specialized code is used to
find define a function to do the actual cast
(defined by the private ``PyArray_GetDTypeTransferFunction``).
This mechanism replaces most of the above mechanism and provides much faster
casts for for example when the inputs are not contiguous in memory.
However, it cannot be extended by user defined datatypes.

Related to casting, we currently have a ``PyArray_EquivTypes`` function which
indicate that a *view* is sufficient (and thus no cast is necessary).


DType handling in Universal functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Universal functions are implemented as ufunc objects with an ordered
list of datatype specific (based on the type number, not datatype instances)
implementations:
This list of implementations can be seen with ``ufunc.types`` where
all implementations are listed with their C-style typecodes.
For example::

    >>> np.add.types
    [...,
     'll->l',
     ...,
     'dd->d',
     ...]

Each of these types is associated with a single inner-loop function defined
in C, which does the actual calculation, and may be called multiple times.

The main step in finding the correct inner-loop function is to call a
:c:type:`PyUFunc_TypeResolutionFunc` which recieves the input dtypes
(in the form of the actual input arrays)
and will find the full type signature to be executed.

By default the ``TypeResolver`` is implemented by searching all of the implementations
listed in ``ufunc.types`` in order and stopping if all inputs can be safely cast to fit to the
current inner-loop function.
This means that if long (``l``) and double (``d``) arrays are added,
numpy will find that the ``'dd->d'`` definition works
(long can safely cast to double) and uses that.

In some cases this is not desirable. For example the ``np.isnat`` universal
function has a ``TypeResolver`` which rejects integer inputs instead of
allowing them to be cast to float.
In principle, downstream projects can currently use their own non-default
``TypeResolver``, since the corresponding C-structure necessary to do this
is public.
The only project known to do this is Astropy, which is willing to switch to
a new API if NumPy were to remove the possibility to replace the TypeResolver.

A second step necessary for parametric dtypes is currently performed within
the ``TypeResolver``:
i.e. the datetime and timedelta datatypes have to decide on the correct unit for
the operation and output array.
While this is part of the type resolution as of now,
it can be seen as separate step, which finds the correct dtype instances.
This separate step occurs only after deciding on the DType class
(i.e. the type number in current NumPy).

For user defined datatypes, the logic is similar, although separately
implemented.
It is currently only possible for user defined functions to be found/resolved
if any of the inputs (or the outputs) has the user datatype.
For example ``fraction_divide(int, int) -> Fraction`` can be implemented
but the call ``fraction_divide(4, 5)`` will fail because the loop that
includes the user datatype ``Fraction`` (as output) can only be found if any of
the inputs is already a ``Fraction``.
``fraction_divide(4, 5, dtype=Fraction)`` can be made to work, but is inconvenient.


Datatype Discovery during Array Coercion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When calling ``np.array(...)``, coercing general python object to a NumPy array,
all objects need to be inspected to find the correct dtype.
The input to ``np.array()`` are potentially nested Python sequences which hold
the final elements as generic Python objects.
NumPy has to unpack all nested sequences and then inspect the elements.
The final datatype is found by iterating all elements which will end up
in the array and:

1. discovering the dtype of the single element:
   * from array (or array like) or NumPy scalar using ``element.dtype``
   * using ``isinstance(..., float)`` for known Python types
     (note that these rules mean that subclasses are *currently* valid).
   * special rule for void datatypes to coerce tuples.
2. Promoting the current dtype with the next elements dtype using
   ``np.promote_types``.
3. If strings are found, the whole process is restarted (see also [gh15327]_),
   in a similar manner as if ``dtype="S"`` was given (see below).

If ``dtype=...`` is given, this dtype is used unmodified, unless
it is an unspecific *parametric dtype instance* which means "S0", "V0", "U0",
"datetime64", and "timdelta64".
These are thus flexible datatypes without length 0 – considered to be unsized –
and datetimes or timedelta without a unit attached ("generic unit").

In future DType class hierarchy, these may be represented by the class rather
than a special instance, since these special instances should not normally be
attached to an array.

If such a *parametric dtype instance* is provided for example using ``dtype="S"``
``PyArray_AdaptFlexibleDType`` is called and effectively inspects all values
using DType specific logic.
That is:

* Strings will use ``str(element)`` to find the length of most elements
* Datetime64 is capable of coercing from strings and guessing the correct unit.


Discussion and Issues
"""""""""""""""""""""

It seems probable that during normal discovery, the ``isinstance`` should rather
be strict ``type(element) is desired_type`` checks.
Further, the current ``AdaptFlexibleDType`` logic should be made available to
user DTypes and not be a secondary step, but instead replace, or be part of,
the normal discovery.



Related Work
------------

* Julia has similar split of abstract and concrete types [julia-types]_. 

* In Julia promotion can occur based on abstract types. If a promoter is
  defined, it will cast the inputs and then Julia can then retry to find
  an implementation with the new values [julia-promotion]_.

* ``xnd-project`` (https://github.com/xnd-project) with ndtypes and gumath

  * The ``xnd-project`` is similar to NumPy and defines data types as well
    as the possibility to extend them. A major difference is that it does
    not use promotion/casting within the ufuncs, but instead requires explicit
    definition of ``int32 + float64 -> float64`` loops.



Related Issues
--------------

``np.save`` currently translates all extension dtypes to void dtypes.
This means they cannot be stored using the ``npy`` format.
This is not an issue for the python pickle protocol, although it may require
some thought if we wish to ensure that such files can be loaded securely
without the possibility of executing malicious code
(i.e. without the ``allow_pickle=True`` keyword argument).

The additional existence of masked arrays and especially masked datatypes
within Pandas has the interesting implications of interoperability.
Since mask information is often stored separately, its handling requires
support by the container (array) object.
NumPy itself does not provide such support, and is not expected to add it
in the foreseeable future.
However, if in the interest of interoperability additions to the datatypes
within NumPy are helpful, doing such additions could be an option even if
they are not used by NumPy itself.


Discussion
----------

The above document is based on various ideas, suggestions, and issues many
of which have come up more than once.
As such it is difficult to make a complete list of discussions, the following
lists a subset of more recent ones:

* Draft on NEP by Stephan Hoyer after a developer meeting (was updated on the next developer meeting) https://hackmd.io/6YmDt_PgSVORRNRxHyPaNQ

* List of related documents gathered previously here https://hackmd.io/UVOtgj1wRZSsoNQCjkhq1g (TODO: Reduce to the most important ones):

  * https://github.com/numpy/numpy/pull/12630

    * Matti Picus NEP, discusses the technical side of subclassing  more from the side of ``ArrFunctions``

  * https://hackmd.io/ok21UoAQQmOtSVk6keaJhw and https://hackmd.io/s/ryTFaOPHE

    * (2019-04-30) Proposals for subclassing implementation approach.
  
  * Discussion about the calling convention of ufuncs and need for more powerful UFuncs: https://github.com/numpy/numpy/issues/12518

  * 2018-11-30 developer meeting notes: https://github.com/BIDS-numpy/docs/blob/master/meetings/2018-11-30-dev-meeting.md and subsequent draft for an NEP: https://hackmd.io/6YmDt_PgSVORRNRxHyPaNQ

    * BIDS Meeting on November 30, 2018 and document by Stephan Hoyer about what numpy should provide and thoughts of how to get there. Meeting with Eric Wieser, Matti Pincus, Charles Harris, Tyler Reddy, Stéfan van der Walt, and Travis Oliphant.
    * Important summaries of use cases.

  * SciPy 2018 brainstorming session: https://github.com/numpy/numpy/wiki/Dtype-Brainstorming

    * Good list of user stories/use cases.
    * Lists some requirements and some ideas on implementations



References and Footnotes
------------------------

.. _pandas_extension_arrays: https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extension-types

.. _xarray_dtype_issue: https://github.com/pydata/xarray/issues/1262

.. _pygeos: https://github.com/caspervdw/pygeos

.. _new_sort: https://github.com/numpy/numpy/pull/12945

.. _gh-12518: https://github.com/numpy/numpy/issues/12518

.. _value_based: Value based promotion denotes the behaviour that NumPy will inspect the value of scalars (and 0 dimensional arrays) to decide what the output dtype should be. ``np.array(1)`` typically gives an "int64" array, but ``np.array([1], dtype="int8") + 1`` will retain the "int8" of the first array.

.. _safe_casting: Safe casting denotes the concept that the value held by one dtype can be represented by another one without loss/change of information. Within current NumPy there are two slightly different usages. First, casting to string is considered safe, although it is not safe from a type perspective (it is safe in the sense that it cannot fail); this behaviour should be considered legacy. Second, int64 is considered to cast safely to float64 even though float64 cannot represent all int64 values correctly.

.. _flexible_dtype: A parametric dtype is a dtype for which conversion is not always safely possible. This is for example the case for current string dtypes, which can have different lengths. It is also true for datetime64 due to its attached unit. A non-parametric dtype should always have a canonical representation (i.e. a float64 may be in non-native byteorder, but the default is native byte order and it is always a valid representation).

.. _julia-types: https://docs.julialang.org/en/v1/manual/types/index.html#Abstract-Types-1

.. _julia-promotion: https://docs.julialang.org/en/v1/manual/conversion-and-promotion/

.. _PEP-384: https://www.python.org/dev/peps/pep-0384/

.. _gh-12518: https://github.com/numpy/numpy/issues/12518

Copyright
---------

This document has been placed in the public domain.
