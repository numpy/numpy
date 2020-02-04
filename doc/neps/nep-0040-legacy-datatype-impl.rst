================================================
NEP 40 — Legacy Datatype Implementation in NumPy
================================================

:title: Extensible Datatypes for NumPy
:Author: Sebastian Berg
:Author: Add...
:Status: Draft
:Type: Informational
:Created: 2019-07-17


Abstract
--------

As a preparation to further NumPy enhancement proposals 41, 42, and 43. This
NEP details the current status of NumPy datatypes as of NumPy 1.19.
It describes some of the technical aspects and concepts necessary to
motivate the following proposals.


Detailed Description
--------------------


Flexible Datatypes
^^^^^^^^^^^^^^^^^^

Some datatypes are inherently *flexible*.
Flexible here means the values representable by the datatype depend on the
specific instance.
As an example, ``datetime64[ns]`` can represent partial seconds which
``datetime64[s]`` cannot and ``"S8"`` can represent more strings than ``"S4"``.
Further, a categorical datatype only accepts certain values, which differ for
different instances.

The basic numerical datatypes are naturally represented as non-flexible:
Float64, Float32, etc. can all be specific dtypes [flexible_floating]_.

Flexible datatypes are interesting for two reasons:

1. Although both datetimes above naturally have the same dtype class with
   the instance only differing in their *unit*, a safe conversion is not
   necessarily possible.
2. When writing code such as ``np.array(["string1", 123.], dtype="S")``, NumPy
   will attempt to find the correct string length by using ``str(123.0)``.
   Without ``dtype="S"`` such behaviour is currently ill defined [gh-15327].
   However, only the first form seems necessary to be supported, the form
   without the DType class being passed should probably result in an error.
   (Currently, ``"S"`` typically behaves like a new ``np.String`` class although it is
   implemented as an ``"S0"`` instance.)


Dispatching of Universal Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently the dispatching of universal function (ufuncs) is limited for
user defined dtypes.
Typically, this is done by finding the first loop for which all inputs can
be cast to safely (see also the current implementation section).

However, in some cases this is problematic.
For example the ``np.isnant`` function is currently only defined for
datetime and timedelta, even though integers are defined to be safely
castable to timedelta.

With respect to to user defined dtypes, dispatching works largely similar,
however, it enforces an exact match of the datetypes (type numbers).
Because the current method is separate and fairly slow, it will also only check
for loops for datatypes already existing in the inputs.
This can be a limitation, a function such as ``rational_divide`` cannot
offer the user that two integer inputs will return rational output conveniently.

For NumPy datatypes currently the order in which loops are registered is important,
this will not be accaptable for user dtypes.
There are two approaches to better define the type resolution for user
defined types:

1. Allow more general mechanisms for loop selection into which user dtypes
   need a hook. I.e. functions which can select/return a loop that does not
   match exactly.
2. Define a total ordering of all implementations/loops, probably based on
   "safe casting" semantics, or semantics similar to that.

While option 2 may be less complex to reason about it remains to be seen
whether it is sufficient for all (or most) use cases.


Inner Loop and Error Handling in UFuncs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the correct implementation/loop is found, UFuncs are currently limited
in what information they provide and how error and setup works.
The main execution is defined as the *inner-loop function*, which may be called
multiple times to do the full calculation.

A user defined loop can pass static data into the function doing the actual
work.
However, this inner loop function may be called multiple times and it is
reasonable to expect use cases where for example scratch space should be
allocated beforehand.

A main issue is that at least flexible datatypes require passing in
additional information to decide how to interpret the data.
It should be noted that there currently is the possibility to get access to
the input array objects (which in turn hold the datatypes when no casting
is necessary).
However, this situation is not ideal.

Another further issue is the error reporting from within the inner-loop function.
There exist currently two ways to do this:

1. by setting a python error
2. by a cpu floating point error flag being set.

Both of these are checked before returning to the user.
However, for example many integer functions currently can set neither of the
errors, adding unnecessary overhead.
In contrast to that, they may pass out error information.
This is currently only possible by setting a python error which creates some
issues with releasing the global interpreter lock during execution on large
arrays.

It seems necessary to provide more control to users.
This means allowing users to pass in and out information from the inner-loop
function more easily, while *not* providing the input array objects.
Most likely this will involve:

* Allowing to execute additional code before and after the inner-loop call.
* Returning an integer value from the inner-loop to allow stopping the
  iteration and possibly pass out error information.
* Possibly, to allow specialized inner-loop selections. For example currently
  ``matmul`` and many reductions will execute optimized code for certain inputs.
  It may make sense to allow selecting such optimized loops beforehand.
  Allowing this may also help to bring casting (which uses this heavily) and
  ufunc implementations closer.

The issues surrounding the inner-loop functions have been discussed in some
detail in the github issue 12518 [gh-12518]_.




Value Based Casting
^^^^^^^^^^^^^^^^^^^

Currently, NumPy uses the value when deciding whether casting is safe or not [value_based]_.
This is done only when the object in question is a scalar or a zero dimensional
array.
The intention of changing this is not part of this NEP.
Even though it is not part of the NEP the intention is to take steps which
will allow limiting the current value based casting to only encompass Python
scalars in the future
(e.g. a python float ``3.`` will use it, but a NumPy ``np.float64(3.)`` will not).
Currently, such behaviour is partially supported even for user defined datatype.
*This support for user defined datatypes is considered unused legacy behaviour and may be removed without a transition period*.


The Object Datatype
^^^^^^^^^^^^^^^^^^^

The object datatype currently serves as a generic fallback for any value
which is not representable otherwise.
However, due to not having a well defined type, it has some issues,
for example when an array is filled with Python sequences.
Further, without a well defined type, functions such as ``isnan()`` or ``conjugate()``
do not necessarily work for example for an array holding decimal values since they cannot
use ``decimal.Decimal`` specific API.
To improve this situation it seems desirable to make it easy to create
object dtypes with a specific python datatype.
This is listed here specifically, for two reasons:

1. Object datatypes require reference counting, which adds complexity to
   which we may not want to (initially) hide from general user defined datatypes.
2. In many use-cases it seems likely that a generic type specific solution
   can be sufficient to improve type safety in user-code (e.g. by specifying
   that an array will only hold such ``decimal.Decimal`` objects.)

Creating python datatypes for builtin Python objects comes with certain issues
that require more thoughts and discussion, which do not need to be part of
the first steps of implementing new datatypes:

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

These secitons give a very brief overview of the current implementation, it is
not meant to be a comprehensive explanation, but a basic reference for further
technical NEPs.

Current ``dtype`` Implementation
""""""""""""""""""""""""""""""""

Currently ``np.dtype`` is a Python class with its instances being the
``np.dtype(">float64")``, etc. instances.
To set the actual behaviour of these instances, a prototype instance is stored
globally and looked up based on the ``dtype.typenum``.
This prototype instance is then copied (if necessary) and modified for
endianess or to support the string datatype which needs to additionally store
the maximum string length.

Many datatype specific functions are defined within a C structure called
``ArrayFuncs``, which is part of each ``dtype`` instance.
For user defined datatypes this structure is defined by the user, making
ABI compatible changes changes impossible.
This structure holds important information such as how to copy, cast,
or implement certain functionality, such as sorting.
We believe that some of the functionality currently defined here should not
be defined as part of the datatype but rather in a seperate (generalized)
ufunc.

A further issue with the current implementation is that, unlike methods,
they are not passed an instance of the dtype when called.
Instead in many cases the array which is being operated on is passed but
typically only used to extract the datatype again.
This should, however, not hinder new developement and replacement considerably,
since we believe that it is sufficient for backward compatibility if a "fake"
array is passed instead, with the assumption that only `PyArray_DESCR(arr)`
will be called on it.
In some cases alignment information of the array may be important and set as well.


NumPy Scalars and Type Hierarchy
""""""""""""""""""""""""""""""""

As a side note to the above datatype implementation, unlike the datatypes,
the numpy scalars currently do provide a type hierarchy, including abstract
types such as ``np.inexact``.
In fact, some control flow within NumPy currently uses
``issubclass(a.dtype.type, np.inexact)``.

NumPy scalars try to mimic zero dimensional arrays with a fixed datatype.
For the numerical datatypes, they are further limited to native byte order.


Current Implementation of Casting
"""""""""""""""""""""""""""""""""

One of the main features which datatypes need to support is casting between one
another using ``arr.astype(new_dtype, casting="unsafe")``, or during execution
of ufuncs with different types (such as adding integer and floating point numbers).

The decision of whether or not casting is possible is mainly done in monolithic
logic using casting tables where applicable; in particular casting tables
cannot handle the flexible datatypes: strings, void/structured, datetimes.

The actual casting has two distinct parts:

1. ``copyswap``/``copyswapn`` are defined for each dtype and can handle byte-swapping for non-native byte orders as well as unaligned memory.
2. ``castfuncs`` is filled on the ``from`` dtype and casts aligned and contiguous memory from one dtype to another (both in native byte order).
   Casting to builtin dtypes is normally defined in a C-vector.
   Casting to a user defined type is stored in an additional dictionary.

When casting (small) buffers will be used when necessary,
using the first ``copyswapn`` to ensure that the second ``castfunc`` can handle the data.
Generally NumPy will thus perform the casting as chain of the three functions
``in_copyswapn -> castfunc -> out_copyswapn``.

However, while user types use only these definitions,
almost all actual casting uses a second implementation which may or may not
combine the above functions.
This second implementation uses specialized implementations for various memory
layouts and optimizes most NumPy datatypes.
This works similar but not identical to specialized universal function inner loops.

Related to casting, we currently have a ``PyArray_EquivTypes`` function which
indicate that a *view* is sufficient (and thus no cast is necessary).


DType code for Universal functions
""""""""""""""""""""""""""""""""""

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
``TypeResolver`` which recieves the input dtypes (in the form of the actual input
arrays) and will find the full type signature to be executed.

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

A second step necessary for flexible dtypes is currently performed within
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
(For example ``fraction_divide(int, int) -> Fraction`` could be implemented
currently, because ``fraction_divide(4, 5)`` will fail, requiring the user
to provide more information.)



Dtatype Discovery during Array Coercion
---------------------------------------

When calling ``np.array(...)`` all objects within the array need to be
inspected to find the correct dtype.
This typically works by using:

1. dtype discovery for a single element:
   * from array (or array like) or NumPy scalar using ``element.dtype``
   * using ``isinstance(..., float)`` for known Python types
     (note that these rules mean that subclasses are *currently* valid).
   * special rule for void datatypes to coerce tuples.
2. Promoting the current dtype with the next elements dtype using
   ``np.promote_types``.
3. If strings are found, they are allowed to try again (see also [gh15327]_).

If ``dtype=...`` is given, this DType is used unless it is a
*flexible dtype instance* such as ``"S0"``
(in the DType class hierarchy, this would be represented by the string DType class).
If it is such a *flexible dtype instance* ``PyArray_AdaptFlexibleDType`` is called
which effectively inspects all values using DType specific logic.
That is:

* Strings will use ``str(element)`` to find the length of most elements
* Datetime64 is capable of coercing from strings and guessing the correct unit.

It seems probable that during normal discovery, the ``isinstance`` should rather
be strict ``type(element) is desired_type`` checks.
Further, the current ``AdaptFlexibleDType`` logic should be made available to
user DTypes and not made a secondary step, but instead be called instead of,
or be part of, the normal discovery.



Related Work
------------

* Julia has similar split of abstract and concrete types [julia-types]_. 

* In Julia promotion can occur based on abstract types. If a promoter is
  defined, it will cast the inputs and then Julia can then retry to find
  an implementation with the new values [julia-promotion]_.

* ``xnd-project`` (https://github.com/xnd-project) with ndtypes and gumath

  * The ``xnd-project`` solves a similar project, however, it does not use
    casting during function execution, so that it does not require a promotion
    step.


Existing ``PyArray_Descr`` fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although not extensively used outside of NumPy itself, the currently
``PyArray_Descr`` is a public structure.
This is especially also true for the ``ArrFunctions`` structure stored in
the ``f`` field. 
Due to compatibility they may need remain supported for a very long time,
with the possibility of replacing them by functions that dispatch to a newer API.

The access to these structures will have to be deprecated.


Open Issues
-----------

``np.load`` (and others) currently translate all extension dtypes to void dtypes.
This means they cannot be stored using the ``npy`` format.
If we wish to add support for ``npy`` format in user datatypes, a new protocol
will have to be designed to achieve this.

The additional existence of masked arrays and especially masked datatypes
within Pandas has the interesting implications of interoperability.
Even though NumPy itself may never have true support for masked values,
the datatype system could grow it.
This means support for datatypes where the actual mask information is not
stored as part of the datatype, but as part of a bit array in the array object.


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

    * BIDS Meeting on November 30, 2018 and document by Stephan Hoyer about what numpy should provide and thoughts of how to get there. Meeting with Eric Wieser, Matti Pincus, Charles Harris, Tyler Reddy, Stéfan van der Walt, and Travis Oliphant. (TODO: Full list of people?)
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

.. _flexible_dtype: A flexible dtype is a dtype for which conversion is not always safely possible. This is for example the case for current string dtypes, which can have different lengths. It is also true for datetime64 due to its attached unit. A non-flexible dtype should typically have a canonical representation (i.e. a float64 may be in non-native byteorder, but the default is native byte order).

.. _julia-types: https://docs.julialang.org/en/v1/manual/types/index.html#Abstract-Types-1

.. _julia-promotion: https://docs.julialang.org/en/v1/manual/conversion-and-promotion/

.. _PEP-384: https://www.python.org/dev/peps/pep-0384/


Copyright
---------

This document has been placed in the public domain.
