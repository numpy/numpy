=======================================
NEP 33 — Extensible Datatypes for NumPy
=======================================

:title: Extensible Datatypes for NumPy
:Author: Sebastian Berg
:Status: Draft
:Type: Informational
:Created: 2019-07-17


Abstract
--------

Datatypes in NumPy describe how to interpret each element in the array.
For the most part NumPy provides the usual numerical types, as well as additionally string and some datetime capabilities. 
The growing Python community, however, has need for more more diverse datatypes.
Example are datatypes with unit information attached (such as meters) or categorical datatypes.
However, the current NumPy datatype API is too limited to allow the creation
of these.
This NEP is the first step to enable such growth, and simplify their development
by allowing new datatypes to be defined in Python instead of C.
By refactoring our datatype API and improving its maintainability future development will become possible not only for external user datatypes, but also within NumPy.


The need for a large refactor arise for multiple reasons.
One of the main issue is the definition of typical functions (such as addition, multiplication, …) for "flexible" datatypes.
Such datatypes – for example the current fixed width strings – require additional steps to define that adding two strings of length four gives a string of length 8.
Similarly, a datatype such as a physical unit, must calculate the new unit information.
A second major issue is that the current casting,
the conversion between different datatypes,
is limited and behaves differently for user defined datatypes compared to NumPy
datatypes.

Internally, datetimes, which have a unit, require monolithic code paths in many places hardcoding their correct support.
While this works well enough for the limited usecase of datetimes within NumPy,
it is not accessible for external users and adds a maintenance burden.
This burden is exacerbated by the exposure of internal structures,
limiting even the development within NumPy,
such as the addition of new sorting methods. 

There are many aspects which limitat the creation of new user defined
datatypes:

* The definition of casting for flexible user types is either impossible or so complex that it has never been attempted.
* Type promotion, the operation deciding that adding float and integer values should return a float value, is very valuable for datatypes defined within NumPy but is limited in scope for user defined datatypes.
* There is a general issue that most operation where multiple datatypes may interact, are written in a monolithic manner. This works well for the simple numerical types, but does not extend well even for the current strings and datetimes.
* The current design means that a unit datatype is not able to define a ``.to_si()`` method to easily find the datatype which would represent the same values in SI units.
* Datatypes wrapping existing python types would enable better and simple support for example for variable length strings, or arrays containing arbitrary precision numbers.

The need to solve these issues is apparent in that there are multiple projects implementing physical units as an array-like class instead of a datatype, which would be the more natural solution.

To address these issues in NumPy multiple development stages are required:

* Phase I: Restructure and extend the datatype infrastructure 

  * Organize Datatypes like normal Python classes
  * Exposing a new and easily extensible API to extension authors

* Phase II: Restructure the way universal functions work:

  * Make it possible to allow functions such as ``np.add`` to be extended by user defined datatypes such as Units.
  * Allow efficient lookup for the correct implementation for user defined datatypes.
  * Enable simple definition of "promotion" rules. A Unit datatype should not need to list that it can be multiplied by: int8, int16, …, float16,… separately.

* Phase III: Growth of NumPy capabilities and Ecosystem

  * Cleanup of legacy behaviour in those parts where it is considered buggy
    or undesired.
  * Easy definition of new datatypes in Python
  * Assist the community in creating types such as Units or Categoricals
  * Allow strings to be used in functions such as ``np.equal`` or ``np.add``.
  * Removal of legacy code paths within NumPy to improve long term maintainability

This document serves as an outline some general aspects mainly for phases I and II.
It lists some general design considerations and some details on the current
implementation designed to be the foundation for future NEPs.

It should be noted that some of the benefits of a large refacor may only
take effect after the full deprecation of the current, legacy, implementation.
This will take years, however, this should not limit new developments and
is rather a reason to push forward with a more extensible API.


Explicite Decisions
-------------------

While largely an informational NEP serving as the basis for more technical proposals,
accepting this NEP
*represents a commitment from NumPy to move forward in a timely manner*.
This includes an agreement that finding good technical solutions
may be more important then finding the best solution if it delays developement
much.

No large incompatibilities are expected due to these changes,
but some changes are acceptable.
If these become more extensive then expected, a major release will be acceptable.

It also defines some specific design goals (details in sections below):
  1. Each basic datatype should be a class with most logic being implemented
     as special methods on the class. In the C-API, these correspond to specific
     slots.
  2. The current datatypes will be instances of these classes. Anything defined
     on the instance, will instead move to the class.
  3. The UFunc machinery will be changed to replace the current dispatching
     and type resolution system. The old system should be (mostly) supported
     as a legacy version for some time. This should thus not affect most users.
  4. Any new API provided to the user will hide implementation as much as
     possible. It should be identical, although may be more limited, to the
     API used to define the internal NumPy datatypes.
  5. The current numpy scalars will *not* be instances of datatypes.
  6. It is desirable to create type specific object datatypes.


Detailed Description
--------------------

Motivation and the Need for New User-Defined Datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current ecosystem has very few user defined datatypes using NumPy, the
two most promient being: ``rational`` and ``quaternion``.
These represent fairly simple datatypes which are not as strongly impacted
but the current limitations.
However, the current available for usertypes is in strong contrast to the need
for datatypes such as:

* categorical types (and variations thereof)
* bfloat16, used in deep learning
* physical units (such as meters)
* extending integer dtypes to have a sentinel NA value
* Geometrical objects [pygeos]_

to name just a few examples.
Some of these are partially solved, for example unit capability is provided
in ``astropy.units``, ``unyt``, or ``pint``. However, these have to subclass
or wrap ``ndarray`` when special datatypes would provide a much more natural
representation, and would immediately allow use within tools such as
``xarray`` [xarray_dtype_issue]_ or Dask.
The need for these datatypes has also already led to the implementation of
ExtensionArrays inside pandas [pandas_extension_arrays]_.


Datatypes as Python Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current NumPy datatypes are not full scale python classes.
They are instead (prototype) instances of a single ``np.dtype`` class.
This means that any special handling, e.g. for ``datetime`` datatypes will
be defined within the specific class instead of in monolithic general code.

The main user side effect of this will be that ``type(np.dtype(np.float64))``
will not be ``np.dtype`` anymore. However, ``isinstance`` will return the
correct value.
This will also add the possibility to use ``isinstance(dtype, np.Float64)``
thus removing the need to use ``dtype.kind``, ``dtype.char``, or ``dtype.type``
to do this check.

One side of this could include the creation of _abstract_ datatypes.
These would be datatypes such as ``np.Floating``, representing any floating
point number.
These would serve a similar purpose as pythons ABCs to establish a class
hierarchy.
There would be no instance of these, however (thus the name _abstract_).


C-API for creating new Datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An important first step is to revise the current C-API with which users
can create new datatypes.
The current API is limited in scope, and uses user allocated structures, which
makes it not extensible.
This has already limited the inclusion of new sorting methods into
NumPy [new_sort]_.

The new version shall thus replace the current ``ArrFuncs`` structure used
to define new datatypes.
Datatypes that currently exist and are defined using these slots will be
supported for the time being by falling back to the old definitions, but
will be deprecated.

A _possible_ solution to hide the implementation from the user and thus make
it extensible in the future is to model the API after pythons stable
API [PEP-384]:

.. code-block:: C

    static struct PyArrayDtypeMethodDef slots[] = {
        {NPY_dt_method, method_implementation},
        ...,
        {0, NULL}
    }

    typedef struct{
      PyTypeObject *typeobj;  /* type of python scalar */
      ...;
      PyType_Slot *slots; /* terminated by slot==0. */
    } PyArrayDTypeMeta_Spec;

    PyObject* PyArray_InitDTypeMetaFromSpec(
            PyArray_DTypeMeta *user_dtype, PyArrayDTypeMeta_Spec *dtype_spec);

The C-side slots should be designed to mirror Python side methods
such as ``dtype.__dtype_method__``, although the exposure to python may be
a later step in the implementation.


Python level interface
^^^^^^^^^^^^^^^^^^^^^^

While a python interface may be a second step, it is a main goal to implement
one.
For example, it is a specific design goal that for example the definition
of a Unit datatype should be possible from within Python.
Note that a Unit datatype can reuse much existing functionality, but needs
to add additional logic to it.

A second goal may be to allow defining new dtypes using type annotations:

.. code-block:: python

    @np.dtype
    class Coordinate:
       x: np.float64
       y: np.float64
       z: np.float64

to largely replace current structured datatypes.


Issues and Design Considerations
--------------------------------

The following section shall list a few issues, and design considerations
as an information and basis for future NEPs trying to tackle these.


Flexible Datatypes
^^^^^^^^^^^^^^^^^^

Some datatypes are inherently _flexible_.
Flexible here means the values representable by the datatype depend on the
specific instance.
As an example, ``datetime64[ns]`` can represent partial seconds which
``datetime64[s]`` cannot.
Another example are fixed length strings, or a categorical datatype which
needs to define the accepted values.

The basic numerical datatypes are naturally represented as non-flexible:
Float64, Float32, etc. can all be specific dtypes [flexible_floating]_.

Flexible datatypes are interesting for two reasons:

1. Although both datetimes above naturally have the same dtype class with
   the instance only differing in their _unit_, a safe conversion is not
   necessarily possible.
2. When writing code such as ``np.array(["string1", "string2"])``, NumPy
   will currenlty find the correct string length.
   This requires additional logic and complexity. It may or may not be desirable
   to support this for user defined datatypes.


Dispatching of Universal Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently the dispatching of universal functionsi (ufuncs) is limited for
user defined dtypes.
Typically, this is done by finding the first loop for which all inputs can
be cast to safely.
(see also the current implementation section).

However, in some cases this is problematic.
For example the ``np.isnant`` function is currently only defined for
datetime and timedelta even though integers are defined to be safely
castable to timedelta.

With respect to to user defined dtypes, this works largely similarly,
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


Scalars should not be instances of the datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simple datatypes such as ``float64`` (see also below), it seems
tempting that the instance of a ``np.dtype("float64")`` can be the scalar.
This idea may be even more so prominent due to the fact that scalars,
rather than datatypes, currently define the correct type hierarchy which
also applies to the datatypes (see also below).

However, we have specifically decided against this.
There are several reason for this.
First, the above described new datatypes should would be instances of DType
classes, making these also classes, while possible, adds an additional
complexity for users to understand.
Second, while the simple NumPy scalars such as ``float64`` may be such instances,
it should be possible to create a ``DecimalDType``.
Its instances would, however, would than be ``decimal.decimal`` instances,
which knows nothing about NumPy, adding complexity rather than reducing it.
Third, methods which are useful for instances (such as ``to_float()``)
cannot be defined for a datatype which is attached to a NumPy array.
Forth, in the reverse direction, scalars are currently only defined for
native byte order and do not need many of the methods and information that
generic datatypes need.

Overall, it seem rather than reducing the complexity, e.g. by merging
the two distinct type hierarchies, this would add complexity for the user.

A possible future path may be to rather simplify the current scalars to
be much simpler objects falling back to behaviour defined on the datatypes
instead.


Value Based Casting
^^^^^^^^^^^^^^^^^^^

Currently, numpy uses the value when deciding whether casting is safe or not [value_based]_.
This is done only when the object in question is a scalar or a zero dimensional
array.
The intention of changing this, is not explicitely part of this NEP, however,
partial support of this behaviour for user defined datatypes is considered
*unused legacy behaviour and may be dropped*.


The Object Datatype
^^^^^^^^^^^^^^^^^^^

The object datatype currently serves as a generic fallback for any value
which is not representable otherwise.
However, due to not having a well defined type, it has some issues for
example when sequences are included into the array.
It also means that certain functions such as ``isnan()`` or ``conjugate()``
do not work for an array holding decimal values.
To improve this situation it seems desirable to make it easy to create
object dtypes with a specific python datatype.
This is listed here specifically, for two reasons:

1. Object datatypes require reference counting, which adds complexity to
   which we may not want to (initially) hide from general user defined datatypes.
2. In many use-cases it seems likely that a generic type specific solution
   can be sufficient to improve type safety in user-code (e.g. by specfying
   that an array will only hold such ``decimal.Decimal`` objects.)

However, the actual implementation of this may be part of the future development
in Phase III.


Current Implementation
----------------------

These section give a very brief overview of the current implementation, it is
not meant to be a comprehensive explenatation.

Current ``dtype`` Implementation
""""""""""""""""""""""""""""""""

Currently ``np.dtype`` is a Python class with its instances being the
``np.dtype(">float64")``, etc. instances.
To set the actual behaviour of these instances, a protopye instance is stored
globally and looke dup based on the ``dtype.typenum``.
This prototype instance is then copied (if necessary) and adjust for byte
swapping, or for example to support the string datatype which needs to store
the length.

Many datatype specific functions are defined within a C structure called
``ArrayFuncs``, which is part of each ``dtype`` instance.
For user defined datatypes this structure is defined by the user, making
ABI compatible changes changes impossible.
This structure holds important information such as how to copy, cast,
or implement certain functionality, such as sorting.
We believe that some of the functionality currently defined here should not
be defined as part of the datatype but rather in a seperate (generalized)
ufunc.

A further issue with the current implementation is that instead of working
like Python methods, most of these functions are not provided with an instance
of the dtype when called, but instead of the array.
However, generally the array passed in is solely used to extract the datatype
again.
**We believe that it is sufficient for backward compatibility if the original
array is not passed in, but instead the passed in object only supports the
extraction of the datatype.**


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

One of the main features which datatypes need to support is casting between one another using ``arr.astype(new_dtype, casting="unsafe")``, or while executing ufuncs with different types (such as adding integer and floating point numbers).

The definition of whether or not casting is possible is mainly done in monolithic
logic and by the use of casting tables.

The actual casting has two distinct parts:

1. ``copyswap``/``copyswapn`` are defined for each dtype and can handle byte-swapping for non-native byte orders as well as unaligned memory.
2. ``castfuncs`` is filled on the ``from`` dtype and casts aligned and contiguous memory from one dtype to another (both in native byte order).
   Casting to builtin dtypes is normally defined in a C-vector.
   Casting to a user defined type is stored in an additional dictionary.

When casting (small) buffers will be used when necessary,
using the first ``copyswapn`` to ensure that the second ``castfunc`` can handle the data.
A general may thus implement the casting chain of the three functions
``in_copyswapn -> castfunc -> out_copyswapn``.

However, while user types use only these definitions,
almost all actual casting uses a monolithic code which may or may not combine the above functions.
This specific code uses specialized implementations for various memory layout
similar, but not identical, to specialized universal function inner loops.


DType code for Universal functions
""""""""""""""""""""""""""""""""""

Universal functions are implemented as UFunc objects with an ordered
list of datatype specific (based on the type number, not datatype instances)
implementations:
These are listed as ``ufunc.types``, using C-style typecodes.
For example:
```
>>> np.add.types
[...,
 'll->l',
 ...,
 'dd->d',
 ...]
```
Each of these types is associated with a single inner-loop function defined
in C, which does the actual calculation, and may be called multiple times.

The main step in finding the correct inner-loop function is to call a
``TypeResolver`` which is passed the input dtypes (in form of the actual input
arrays) and will find the full type signature to be executed.

By default the ``TypeResolver`` is implemented by searching all in
order and stopping as soon as all inputs can be safely cast to fit to the
current inner-loop function.
This means that if long (``l``) and double (``d``) arrays are added,
numpy will find that the ``'dd->d'`` defintion works
(long can safely cast to double) and uses that.

In some cases this is not desirable, for example the ``np.isnat`` universal
function has a ``TypeResolver`` which rejects integer inputs instead of
allowing them to be cast to float.
In principle downstream projects can currently use their own non-default
``TypeResolver``, since the corresponding C-structure necessary to do this
is public.
The only project known to do this is Astropy, and it is willing to adapt if
necessary.

A second step necessary for flexible dtypes is currently performed within
the ``TypeResolver``:
the datetime and timedelta datatypes have to decide on the correct unit for
the operation and output array.
While this is part of the type resolution as of now,
it can be seen as separate step which find the correct dtype instances,
after deciding on the dtype class (i.e. the type number in current NumPy).
This step can actually occur after deciding which loop to run and may be part
of the generally necessary check that all inputs can be cast if necessary.

For user defined datatypes, the logic is similar, although separately
implemented.
It is currently only possible for user defined functions to be found/resolved
if any of the inputs (or the output) has the user datatype.


Notes on Casting and DType Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The design presented here means that DType classes are first class objects and finding the correct DType class always happens first both for coercion from python and when finding the correct ``UFuncImpl`` to call.

For non-flexible DTypes, the second step is trivial, since they have a canonical implementation (if there is only a single instance, that one should be typically used for backward compatibility though). For flexible DTypes a second pass is needed, this is either an ``adjust_dtypes`` step within UFuncs, or ``__discover_descr_from_pyobject__`` when coercing within ``np.array``. For the latter, this generally means a second pass is necessary for flexible dtypes (although it may be possible to optimize that for common cases). In this case the ``__common_instance__`` method has to be used as well.

There is currently an open question whether ``adjust_dtypes`` may require the values in some cases. This is currently *not* strictly necessary (with the exception that ``objarr.astype("S")`` will use coercion rather than casting logic, a special case that needs to remain). It could be allowed by giving ``adjust_dtypes`` the input array in certain cases. For the moment it seems preferable to avoid this, if such a discovery step is required, it will require a helper function:

.. code-block:: python

    arr = np.random.randint(100, size=1000)
    categorical = find_categorical_dtype(arr)
    cat_array = arr.astype(categorical)  # may error if arr was mutated


Related Changes
---------------

This additional section details some related changes,
which are only partially tied to the general refactor.

**Stricter array rules for dtype discovery**

When coercing arrays with ``np.array`` and related functions, numpy currently uses ``isinstance(pyobj, float)`` logic (user types do not have this ability, they can only automatically be discovered from numpy scalars). In general, user dtypes should be capable of ensuring that specific input is coerced correctly.
However, in general these should be exact types and not ``isinstance`` checks. A python float subclass, could have a completely different meaning and should generally viewed as a ``"float64"`` dtype. Instead, the current ``isinstance`` checks should become a fallback discovery mechanisms and *be deprecated*.



Related Work
------------

* Julia has similar split of abstract and concrete types [julia-types]_. 

* In Julia promotion can occur based on abstract types. If a promoter is
  defined, it will be called and then retry the resolution [julia-promotion]_.

* ``xnd-project`` https://github.com/xnd-project) with ndtypes and gumath

  *  Different in that it does not use promotion at all.


Backward compatibility
----------------------

While the actual backward compatibility impact is not yet fully clear,
we anticipate, and accept the following changes:

* ``PyArray_DescrCheck`` currently tests explicitly for being an instance of PyArray_Descr. The Macro is thus not backward compatible (it cannot work in new NumPy versions). This Macro is not used often, for example not even SciPy uses it. This will require an ABI breakage, to mitigate this new versions of legacy numpy (e.g. 1.14.x, etc.) will be released to include a macro that is compatible with newer NumPy versions. Thus, downstream will may be forced to recompile, but can do so with a single (old) NumPy version.

* The array that is currently provided to some functions (such as cast functions), may not be provided anymore generally (unless easily available). For compatibility, a dummy array with the dtype information will be given instead. At least in some code paths, this is already the case.

* The ``scalarkind`` slot and registration of scalar casting will be removed/ignored without replacement (it currently allows partial value based. The ``PyArray_ScalarKind`` function will continue to work for builtin types, but will not be used internally and be deprecated.

* The type of any dtype instance will not be ``dtype`` anymore, instead, it it will be a subclass of DType.

* Current user dtypes are specifically defined as instances of ``np.dtype``, the instance used when registered is typically not held on to, but at the very least its type and base would have to be exchanged/modified. This may mean that the user created Descriptor struct/object is only partially usable (it does not need to be used though, and is not for either ``rational`` or ``quaternion``)


Existing ``PyArray_Descr`` fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although not extensively used outside of NumPy itself, the currentl
``PyArray_Descr`` is a public structure.
This is especially also true for the ``ArrFunctions`` strcuture stored in
the ``f`` field. 
Due to compatibility they may need remain supported for a very long time,
with the possibility of replacing them by functions that dispatch to a newer API.

The access to these structures will have to be deprecated.




Open Issues
-----------

``np.load`` (and others) currently translate all extension dtypes to void dtypes.
This means they cannot be stored using the ``npy`` format.
Similar issues probably exist with the buffer interface.

In some cases, the only option would be to raise an error instead of silently
converting the data.
For saving arrays we may have to force pickling right now,
although we could store known dtypes and force users to simply import that library first?



Discussion
----------

The above document is based on various ideas, suggestions, and issues many
of which have come up more than once.
As such it is difficult 

* Draft on NEP by Stephan Hoyer after a developer meeting (was updated on the next developer meeting) https://hackmd.io/6YmDt_PgSVORRNRxHyPaNQ

* List of related documents gathered previousl here https://hackmd.io/UVOtgj1wRZSsoNQCjkhq1g (TODO: Reduce to the most important ones):

  * https://github.com/numpy/numpy/pull/12630

    * Matti's NEP, discusses the technical side of subclassing  more from the side of ``ArrFunctions``

  * https://hackmd.io/ok21UoAQQmOtSVk6keaJhw and https://hackmd.io/s/ryTFaOPHE

    * (2019-04-30) Proposals for subclassing implementation approach.
  
  * Discussion about the calling convention of ufuncs and need for more powerful UFuncs: https://github.com/numpy/numpy/issues/12518

  * 2018-11-30 developer meeting notes: https://github.com/BIDS-numpy/docs/blob/master/meetings/2018-11-30-dev-meeting.md and subsequent draft for an NEP: https://hackmd.io/6YmDt_PgSVORRNRxHyPaNQ

    * BIDS Meeting on November 30, 2018 and document by Stephan Hoyer about what numpy should provide and thoughts of how to get there. Meeting with Eric Wieser, Matti Pincus, Charles Harris, and Travis Oliphant.
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
