==================================
NEP 41 — Improved Datatype Support
==================================

:title: Improved Datatype Support
:Author: Sebastian Berg
:Author: Add...
:Status: Draft
:Type: Standard Track
:Created: 2020-02-03



Abstract
--------

Datatypes in NumPy describe how to interpret each element in arrays.
For the most part, NumPy provides the usual numerical types, as well as additional string and some datetime capabilities. 
The growing Python community, however, has need for more more diverse datatypes.
Examples are datatypes with unit information attached (such as meters) or categorical datatypes (fixed set of possible values).
However, the current NumPy datatype API is too limited to allow the creation
of these.
This NEP is the first step to enable such growth, it will lead to 
a simpler developement of new datatypes with the aim that in the future
datatypes can be defined in Python instead of C.
By refactoring our datatype API and improving its maintainability,
future development will become possible, not only for external user datatypes,
but also within NumPy.


The need for a large refactor arises for multiple reasons.
One of the main issues is the definition of typical functions (such as addition, multiplication, …) for "flexible" datatypes.
To operate on flexible datatypes these functions need additional steps.
For example when adding two strings of length four, the result is a string
of length 8, which is different from the input.
Similarly, a datatype which has as a physical unit, must calculate the new unit information,
dividing a distance by a time, results in a speed.
A related major issue is that the current casting rules – the conversion between different datatypes –
is too limited for such flexible datatypes implemented outside of NumPy.

NumPy currently has strings and datetimes as flexible datatypes.
These, more than the other datatypes, require special code paths in otherwise
generic code.
For user defined datatypes, this is not accessible, but also within NumPy it
means an increased complexity since the concerns of different datatypes
are not well separated.
This burden is exacerbated by the exposure of internal structures,
limiting even the development within NumPy,
such as the addition of new sorting methods.

Thus, there are many factors which limit the creation of new user defined
datatypes:

* Creating casting rules for flexible user types is either impossible or so complex that it has never been attempted.
* Type promotion, e.g. the operation deciding that adding float and integer values should return a float value, is very valuable for datatypes defined within NumPy, but is limited in scope for user defined datatypes.
* There is a general issue that most operations where multiple datatypes may interact, are written in a monolithic manner. This works well for the simple numerical types, but does not extend well, even for current strings and datetimes.
* In the current design, a unit datatype cannot have a ``.to_si()`` method to easily find the datatype which would represent the same values in SI units.


The large need to solve these issues is apparent for example in the fact that
there are multiple projects implementing physical units as an array-like
class instead of a datatype, which would be the more natural solution.
Similarly, Pandas has made a push into the same direction and undoubtedly
the community would be best served if such new features could be common
between NumPy, Pandas, and other projects.

To address these issues in NumPy, multiple development stages are required:

* Phase I: Restructure and extend the datatype infrastructure 

  * Organize Datatypes like normal Python classes
  * Exposing a new and easily extensible API to extension authors

* Phase II: Restructure the way universal functions work:

  * Make it possible to allow functions such as ``np.add`` to be extended by user defined datatypes such as Units.
  * Allow efficient lookup for the correct implementation for user defined datatypes.
  * Enable simple definition of "promotion" rules. This should include the ability
    to reuse or fall back to existing rules for e.g. units, which, after resolving
    and unit information, should behave the same as the numerical datatype used
    to store the value.

* Phase III: Growth of NumPy and Scientific Python Ecosystem capabilities

  * Cleanup of legacy behaviour where it is considered buggy or undesired.
  * Easy definition of new datatypes in Python
  * Assist the community in creating types such as Units or Categoricals
  * Allow strings to be used in functions such as ``np.equal`` or ``np.add``.
  * Removal of legacy code paths within NumPy to improve long term maintainability

This document serves as a basis mainly for phases I and II.
It lists general design considerations and some details on the current
implementation designed to be the foundation for future NEPs.

It should be noted that some of the benefits of a large refactor may only
take effect after the full deprecation of the current, legacy implementation.
This will take years. However, this shall not limit new developments and
is rather a reason to push forward with a more extensible API.


Decisions
---------

Specifically, this NEP proposes to preliminarily move ahead with the first
point in the first Phase (see Implementation section).
It also lays out the final design goals and establish some user facing design
decisions and considerations.
No large incompatibilities are expected due to implementing the proposed full
changes (including all Phases),
ideally only requiring minor changes in downstream libraries on a similar scale
to other updates in the recent past.
However, if incompatibilities become more extensive then expected,
a major NumPy release is an acceptable solution, although even then
the vast majority of users shall not be affected.
A transition requiring large code adaptation, similar to the Python 2 to 3
transition is not anticipated and not covered by this NEP.

Detailed below, we accept the following design considerations:

1. Each basic datatype should be a class with most logic being implemented
   as special methods on the class. In the C-API, these correspond to specific
   slots.
2. The current datatypes will be instances of these classes. Anything defined
   on the instance, should instead be defined on the class (except things
   instance information such as itemsize, byteorder, ...).
4. All new API provided to the user will hide implementation as much as
   possible. The public API should be identical but may be more limited,
   to the API used to define the internal NumPy datatypes.
5. The current NumPy scalars will *not* be instances of datatypes.
3. The UFunc machinery will be changed to replace the current dispatching
   and type resolution system. The old system should be (mostly) supported
   as a legacy version for some time. This should thus not affect most users.
   
Additionally, as a general design principle, the addition of new user defined
datatypes shall *not* change the behaviour of programs.
For example ``common_dtype(a, b)`` must not be ``c`` unless ``a`` or ``b`` know
that ``c`` exists.

Examples
^^^^^^^^


Simple Numerical Types
""""""""""""""""""""""

Of course a major point is to allow new, simple, numerical types to integrate
better, especially into the ufunc machinery.
For these types the definitions of things such as ``np.common_type`` and
``np.can_cast`` are some of the most important things.
Having ``np.common_type``, it is (for the most part) possible to find
the correct ufunc to execute, since most ufuncs – such as add – effectively
only require ``np.result_type``::

    >>> np.add(arr1, arr2).dtype == np.result_type(arr1, arr2)

and ``np.result_type`` is largely identical to ``np.common_type``.


Fixed, high precision math
""""""""""""""""""""""""""

A prime example, encompassing all the main concepts that can be expected
(in a general sense) of a datatype, or fixed, but arbitrary precision
numbers, such as defined by ``mpmath``::

    >>> import mpmath as mp
    >>> print(mp.dps)  # the current (default) precision
    15

NumPy should be able to construct a native, memory efficient array from
a list of ``mpmath.mpf`` floating point objects::

    >>> arr1 = np.array(mp.arange(3))  # (mp.arange returns a list)
    >>> print(arr)  # Must find the correct precision from the objects:
    array(['0.0', '1.0', '2.0'], dtype=mpf[dps=15])

Although, we should also be able to specify the desired precision when
creating the datatype for the array. Here, I use ``np.dtype[mp.mpf]``
to find the DType class (the notation is not part of this NEP),
which is then instantiated with the desired parameter.
This could also be written as ``MpfDType`` class::

    >>> arr2 = np.array([1, 2, 3], dtype=np.dtype[mp.mpf](dps=100))
    >>> print(arr1 + arr2)
    array(['0.0', '2.0', '4.0'], dtype=mpf[dps=100])

The ``mpf`` datatype finds that the result of the operation should be the higher
precision one of the two, so uses a precision of 100.
Furthermore, we should be able to define casting, for example as in::

    >>> np.can_cast(arr1.dtype, arr2.dtype, casting="safe")
    True
    >>> np.can_cast(arr2.dtype, arr1.dtype, casting="safe")
    False
    >>> np.can_cast(arr2.dtype, arr2.dtype, casting="same_kind")
    True

Casting from float is a probably always at least a ``same_kind`` cast, but
in general, it is not safe::

    >>> np.can_cast(np.float64, np.dtype[mp.mpf](dps=4), casting="safe")
    False

Since the a float64 has a higer precision than the ``mpf`` datatype with
``dps=4``.
Alternatively, we can say that::

    >>> np.common_type(np.dtype[mp.mpf](dps=5), np.dtype[mp.mpf](dps=10))
    np.dtype[mp.mpf](dps=10)

And possibly even::

    >>> np.common_type(np.dtype[mp.mpf](dps=5), np.float64)
    np.dtype[mp.mpf](dps=16)  # equivalent precision to float64 (I believe)

since ``np.float64`` can be cast to a ``np.dtype[mp.mpf](dps=16)`` safely.


Categoricals
""""""""""""

Categoricals are interesting in that the objects that can be put in may or
may not be arbitrary.
Also, we may want a fixed set of categories, or a datatype which can add new
categories when necessary.
The fixed categories (defined ahead of time) is the most straight forward
categorical definition.
(Categoricals are *hard*, since there are many approaches to them!)

    >>> cat = Categorical(["eggs", "spam", "toast"])
    >>> breakfast = array(["eggs", "spam", "eggs", "toast"], dtype=cat)

Could store the array very efficiently, since it knows that there are only 3
categories.
Since a categorical in this sense knows almost nothing about the data stored
in it, few operations makes, sense, although equality probably does:

    >>> breakfast2 = array(["eggs", "eggs", "eggs", "eggs"], dtype=cat)
    >>> breakfast == breakfast2
    array[True, False, True, False])

The categorical datatype would probably work much like a dictionary, no two
items can be equal (checked on dtype creation), so that the equality operation
above can be performed very efficiently.
If the values define an order, the category labels (internally integers) could
be ordered the same way to allow efficient sorting and comparisons.

Whether or not casting is defined from one categorical with less to one with
strictly more values defined, is something that the Categorical datatype would
have to decide.

Python Enums DType
""""""""""""""""""

An example for a more complex datatype, encompassing an additional concept
would be a DType that could wrap ``enum.Enum``::

    >>> class Breakfast(enum.Enum):
    ...     spam, eggs, toast = 1, 2, '3'
    >>> table = array([Breakfast.spam, Breakfast.eggs, Breakfast.toast], dtype=EnumDType(Breakfast))
    >>> to_values(table)  # to_values is a ufunc here
    array([1, 2, '3'], dtype=object)
    >>> table[0]
    <Breakfast.spam: 1>

The following operations may be desirable but are *not* simply possible::

    >>> table2 = np.array([Breakfast.spam, Breakfast.eggs])  # discover same dtype
    >>> table == Breakfast.spam
    array([True, False, False])

To define these NumPy would need to find the correct Enum DType, which is
unclear how it should be done, because the type associated with ``EnumDType``
could be any ``Enum`` subclass.

An alternative approach, which solves those issues, is to create the DTypes as::

    >>> BreakfastDType = EnumDType(Breakfast)  # class factory
    >>> issubclass(BreakfastDType, EnumDType)
    True

Where ``BreakfastDType`` is a subclass and not just an instance of ``EnumDType``.
Which makes sense since also ``Enum`` is a class factory.

However, it is yet to be decided whether even the second logic should be allowed.
It would also be plausible, that the user has to use a new NumPy aware ``Enum``
class to write code similar to the above.



Unit on the Datatype
""""""""""""""""""""

There are different ways to define Units, depending on how the internal
machinery would be organized, one way is to have a single Unit datatype
for every existing numerical type.
This will be written as ``Unit[float64]``, the unit itself is part of the
DType instance ``Unit[float64]("m")`` us a ``float64`` with meters attached::

    >>> meters = np.array([1, 2, 3], dtype=np.float64) * unit.m  # meters
    >>> print(meters)
    array([1.0, 2.0, 3.0], dtype=Unit[float64]("m"))

Note that units are a bit tricky, since it is debatable, whether::

    >>> np.array([1.0, 2.0, 3.0], dtype=Unit[float64]("m"))

should be valid syntax (coercing the float scalars without a unit to meters).
Once the array is created, math will work without any issue::

    >>> meters * 2 * unit.seconds
    array([2.0, 4.0, 6.0], dtype=Unit[float64]("ms"))

Casting is not valid from one unit to the other, but can be between different
scales of the same dimensionality (although this may usually be "unsafe")::

    >>> meters.astype(Unit[float64]("s"))
    TypeError: Cannot cast meters to seconds.
    >>> meters.astype(Unit[float64]("km"))
    >>> meters.astype(meters.dtype.to_cgs())

The above notation may be somewhat clumsy, in some cases, and functions
could be used to convert things otherwise.
There may be ways to make these more convenient, but those must be left
for future discussions::

    >>> units.convert(meters, "km")
    >>> units.to_cgs(meters)

There are some open questions, for example whether additional methods
on the array object could exist to simplify some of the notions, and how these
would be implemented.

The interaction with other scalars would likely be defined through::

    >>> np.common_type(np.float64, Unit)
    Unit[np.float64](unitless)

*The main interesting point about units is that it is obvious that ufuncs
can be more involved than for simple numerical dtypes*.
For every universal function, the statement above that::

    >>> np.multiple(meters, seconds).dtype != np.result_type(meters, seconds)

In fact ``np.result_type(meters, seconds)`` must error without context
of the operation being done.
This example highlights how the specific ufunc loop
(loop with known, specific DTypes as inputs), has to be able to to make
certain decisions, before the actual calculation can start.


Detailed Description
--------------------

The following sections details some of the design decisions above and gives
more details on potential user datatype use cases motivating the need for
these changes.
Since datatype changes touch a large part of code and behaviours, NEP 40
reviews some of the concepts, issues, and the current implementation.


Motivation and the Need for New User-Defined Datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current ecosystem has very few user defined datatypes using NumPy, the
two most promient being: ``rational`` and ``quaternion``.
These represent fairly simple datatypes which are not as strongly impacted
but the current limitations.
However, the number of currently available user defined datatypes
is in strong contrast with the need for datatypes such as:

* categorical types
* bfloat16, used in deep learning
* physical units (such as meters)
* extending e.g. integer dtypes to have a sentinel NA value
* geometrical objects [pygeos]_
* datatypes for tracing/automatic differentiation
* high or arbitrary precision math
* ….

Some of these are partially solved; for example unit capability is provided
in ``astropy.units``, ``unyt``, or ``pint``. However, these have to be implemented
as array objects, which means that they have to deal with container operations
such as reshaping and how to work together with other array objects such as
``xarray`` [xarray_dtype_issue]_ or ``Dask``.
Implemented as a datatype, the concerns of the array object (container) and
the data handling can be clearly separated.
Thus a pure container, such as ``xarray`` or ``Dask`` should work with new
datatypes without even knowing about their existence.

The need for such datatypes has also already led to the implementation of
ExtensionArrays inside pandas [pandas_extension_arrays]_.


Datatypes as Python Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current NumPy datatypes are not full scale python classes.
They are instead (prototype) instances of a single ``np.dtype`` class.
Changing this, means that any special handling, e.g. for ``datetime``
can be moved to the Datetime class instead of in monolithic general code
(e.g. current ``PyArray_AdjustFlexibleDType``).

The main API side effect of this, is that special method are not anymore
housed on the dtype instances, but instead as methods on the class.
This is the typical design pattern used in Python.
Adding a new, natural point to store these methods and information, will
further allow to refine the API to ensure that it can grow in the future
(the current API cannot be extended due to how it is made public).

The main user side effect of this will be that ``type(np.dtype(np.float64))``
will not be ``np.dtype`` anymore. However, ``isinstance`` will return the
correct value.
This will also add the possibility to use ``isinstance(dtype, np.Float64)``
thus removing the need to use ``dtype.kind``, ``dtype.char``, or ``dtype.type``
to do this check.

If DTypes were full scale Python classes, the question of subclassing arises.
Inheritance, however, appears problematic and a complexity best avoided
(at least initially) for container datatypes.
To still define a class hierarchy and subclass order, a possible approach is to allow
the creation of *abstract* datatypes.
An example for an abstract datatype would be ``np.Floating``,
representing any floating point number.
These can serve the same purpose as Python's abstract base classes.


Scalars should not be instances of the datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simple datatypes such as ``float64`` (see also below), it seems
tempting that the instance of a ``np.dtype("float64")`` can be the scalar.
This idea may be even more appealing due to the fact that scalars,
rather than datatypes, currently define a useful type hierarchy.

However, we have specifically decided against this.
There are several reason for this.
First, the above described new datatypes would be instances of DType
classes.
Making these instances themselves classes, while possible, adds an additional
complexity that users need to understand.
Second, while the simple NumPy scalars such as ``float64`` may be such instances,
it should be possible to create data types for Python objects without enforcing
NumPy as a dependency.
Python objects that do not depend on NumPy cannot be instances of a NumPy DType
however.
Third, methods which are useful for instances (such as ``to_float()``)
cannot be defined for a datatype which is attached to a NumPy array.
While at the same time scalars are currently only defined for
native byte order and do not need many of the methods and information that
generic datatypes require.

Overall, it seem rather than reducing the complexity, i.e. by merging
the two distinct type hierarchies, making scalars instances of DTypes would
add complexity for the user.

A possible future path may be to instead simplify the current NumPy scalars to
be much simpler objects which largely derived their behaviour from the datatypes.



C-API for creating new Datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An important first step is to revise the current C-API with which users
can create new datatypes.
The current API is limited in scope, and accesses user allocated structures,
which means it not extensible since no new members can be added to the structure
without losing binary compatibility.
This has already limited the inclusion of new sorting methods into
NumPy [new_sort]_.

The new version shall thus replace the current ``ArrFuncs`` structure used
to define new datatypes.
Datatypes that currently exist and are defined using these slots will be
supported for the time being by falling back to the old definitions, but
will be deprecated.

A *possible* solution is to hide the implementation from the user and thus make
it extensible in the future is to model the API after Python's stable
API [PEP-384]:

.. code-block:: C

    static struct PyArrayMethodDef slots[] = {
        {NPY_dt_method, method_implementation},
        ...,
        {0, NULL}
    }

    typedef struct{
      PyTypeObject *typeobj;  /* type of python scalar */
      ...;
      PyType_Slot *slots;
    } PyArrayDTypeMeta_Spec;

    PyObject* PyArray_InitDTypeMetaFromSpec(
            PyArray_DTypeMeta *user_dtype, PyArrayDTypeMeta_Spec *dtype_spec);

The C-side slots should be designed to mirror Python side methods
such as ``dtype.__dtype_method__``, although the exposure to Python may be
a later step in the implementation to reduce the complexity of the initial
implementation.


Python level interface
^^^^^^^^^^^^^^^^^^^^^^

While a Python interface is a second step, it is a main feature of this NEP
to enable a Python interface and work towards it.
For example, it is a specific long term design goal that the definition
of a Unit datatype should be possible from within Python.
Note that a Unit datatype can reuse much existing functionality, but needs
to add additional logic to it.

One approach, or additional API may be to allow defining new dtypes using type annotations:

.. code-block:: python

    @np.dtype
    class Coordinate:
       x: np.float64
       y: np.float64
       z: np.float64

to largely replace current structured datatypes.


C-API Changes to the UFunc Machinery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Proposed changes to the UFunc machinery will be part of NEP 43.
However, the following changes will be necessary (see NEP 40 for a detailed
description of the current implementation and its issues):

* The current UFunc type resolution must be adapted to allow better control
  for user dtypes as well as resolve current inconsistencies.
* The inner-loop used in UFuncs must be expanded to include a return value.
  Further, error reporting must be improved and passing in dtype specific
  enabled.


Implementation
--------------

The required changes necessary to NumPy are massive and touch a large part
of the code base.
We thus propose the implementation of the above steps listed in Phases I and II.

Although it is possible that new DTypes will only be useful after Phase II
is finished, this NEP proposes to start implementation in an incremental way.
This means that in a first step the ``DType`` classes will be added, with
all, or most, new exposed API points giving a ``PreliminaryDTypeAPIWarning``.

This allows for smaller patches and further future changes. In these first
steps no, or only very limited, new C-API shall be exposed.
The addition of new ``DTypes`` will then allow to address other changes
more incremental:

1. A new machinery for array coercion, with the goal of enabling user DTypes
   to behave in a full featured manner.
2. The replacement or wrapping of the current casting machinery.
3. Incremental redifinition of the current ``ArrFunctions`` slots into
   DType method slots.

Parallel to these, after step 1. is finished, the Phase II of revising the
UFunc machinery can be addressed.

In particular the step of creating a C defined ``DTypeMeta`` class with its
instances being ``DTypeClasses`` as mentioned above is a necessary first step
with useful semantics.
This ``DTypeMeta`` must thus be implemented before being widely used to
restructure or enhance current code, thus we propose to proceed with mainly
private additions to the DType classes.


Backward compatibility
----------------------

While the actual backward compatibility impact is not yet fully clear,
we anticipate, and accept the following changes:

* **Python API**:
  * ``type(np.dtype("f8"))`` will be a subclass of ``np.dtype``, while right
    now ``type(np.dtype("f8")) is np.dtype``.
    Code should use ``isinstance`` checks, and in very rare cases may have to
    be adapted to use it.

* **C-API**:
    * In old versions of NumPy ``PyArray_DescrCheck`` is a macro which uses
      ``type(dtype) is np.dtype``. When compiling against an old NumPy version,
      the macro may have to be replaced with the corresponding
      ``PyObject_IsInstance`` call. (If this is a problem, we could backport
      fixing the macro)

   * The UFunc machinery changes will break *limited* parts of the current
     implementation. Replacing e.g. the default ``TypeResolver`` is expected
     to remain supported for a time, although optimized masked inner loop iteration
     (which is not even used *within* numpy) is expected to not remain supported
     and lead to errors instead.

* **dtype implementors (C-API)**:
  * The array that is currently provided to some functions (such as cast functions),
    may not be provided anymore generally.
    For example ``PyArray_Descr->f->nonzero`` or ``PyArray_Descr->f->copyswapn``,
    may instead receive a dummy array object with only some fields (mainly the
    dtype), being valid.
    At least in some code paths, a similar mechanism is already used.

  * The ``scalarkind`` slot and registration of scalar casting will be
     removed/ignored without replacement.
     It currently allows partial value based casting.
     The ``PyArray_ScalarKind`` function will continue to work for builtin types,
     but will not be used internally and be deprecated.


   * Current user dtypes are specifically defined as instances of ``np.dtype``.
     The creation works by the user providing a prototype instance.
     NumPy will need to modify at least the type during registration.
     This has no effect for either ``rational`` or ``quaternion`` and mutation
     of the structure seems unlikely after registration.

Since there is a fairly large API surface concerning datatypes, further limitations
or the limitation of a certain function to currently existing datatypes is
likely to occur.
For example current functions which currently use the type number as input
should probably be replaced with functions taking DType classes instead
in the long term.
Although public, large parts of this C-API seems very rarely and possibly
completely used by downstream projects.


Discussion
----------

See NEP 40 for a list of previous meetings, and discussions.


References
----------

.. _pandas_extension_arrays: https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extension-types

.. _xarray_dtype_issue: https://github.com/pydata/xarray/issues/1262

.. _pygeos: https://github.com/caspervdw/pygeos

.. _new_sort: https://github.com/numpy/numpy/pull/12945

.. _PEP-384: https://www.python.org/dev/peps/pep-0384/


Copyright
---------

This document has been placed in the public domain.
