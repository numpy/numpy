.. _NEP42:

==============================================================================
NEP 42 — New and extensible DTypes
==============================================================================

:title: New and extensible DTypes
:Author: Sebastian Berg
:Author: Ben Nathanson
:Author: Marten van Kerkwijk
:Status: Accepted
:Type: Standard
:Created: 2019-07-17
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2020-October/081038.html

.. note::

    This NEP is third in a series:

    - :ref:`NEP 40 <NEP40>` explains the shortcomings of NumPy's dtype implementation.

    - :ref:`NEP 41 <NEP41>` gives an overview of our proposed replacement.

    - NEP 42 (this document) describes the new design's datatype-related APIs.

    - :ref:`NEP 43 <NEP43>` describes the new design's API for universal functions.


******************************************************************************
Abstract
******************************************************************************

NumPy's dtype architecture is monolithic -- each dtype is an instance of  a
single class. There's no principled way to expand it for new dtypes, and the
code is difficult to read and maintain.

As :ref:`NEP 41 <NEP41>` explains, we are proposing a new architecture that is
modular and open to user additions. dtypes will derive from a new ``DType``
class serving as the extension point for new types. ``np.dtype("float64")``
will return an instance of a ``Float64`` class, a subclass of root class
``np.dtype``.

This NEP is one of two that lay out the design and API of this new
architecture. This NEP addresses dtype implementation; :ref:`NEP 43 <NEP43>` addresses
universal functions.

.. note::

    Details of the private and external APIs may change to reflect user
    comments and implementation constraints. The underlying principles and
    choices should not change significantly.


******************************************************************************
Motivation and scope
******************************************************************************

Our goal is to allow user code to create fully featured dtypes for a broad
variety of uses, from physical units (such as meters) to domain-specific
representations of geometric objects. :ref:`NEP 41 <NEP41>` describes a number
of these new dtypes and their benefits.

Any design supporting dtypes must consider:

- How shape and dtype are determined when an array is created
- How array elements are stored and accessed
- The rules for casting dtypes to other dtypes

In addition:

- We want dtypes to comprise a class hierarchy open to new types and to
  subhierarchies, as motivated in :ref:`NEP 41 <NEP41>`.

And to provide this,

- We need to define a user API.

All these are the subjects of this NEP.

- The class hierarchy, its relation to the Python scalar types, and its
  important attributes are described in `nep42_DType class`_.

- The functionality that will support dtype casting is described in `Casting`_.

- The implementation of item access and storage, and the way shape and dtype
  are determined when creating an array, are described in :ref:`nep42_array_coercion`.

- The functionality for users to define their own DTypes is described in
  `Public C-API`_.

The API here and in :ref:`NEP 43 <NEP43>` is entirely on the C side. A Python-side version
will be proposed in a future NEP. A future Python API is expected to be
similar, but provide a more convenient API to reuse the functionality of
existing DTypes. It could also provide shorthands to create structured DTypes
similar to Python's
`dataclasses <https://docs.python.org/3.8/library/dataclasses.html>`_.


******************************************************************************
Backward compatibility
******************************************************************************

The disruption is expected to be no greater than that of a typical NumPy
release.

- The main issues are noted in :ref:`NEP 41 <NEP41>` and will mostly affect
  heavy users of the NumPy C-API.

- Eventually we will want to deprecate the API currently used for creating
  user-defined dtypes.

- Small, rarely noticed inconsistencies are likely to change. Examples:

  - ``np.array(np.nan, dtype=np.int64)`` behaves differently from
    ``np.array([np.nan], dtype=np.int64)`` with the latter raising an error.
    This may require identical results (either both error or both succeed).
  - ``np.array([array_like])`` sometimes behaves differently from
    ``np.array([np.array(array_like)])``
  - array operations may or may not preserve dtype metadata

- Documentation that describes the internal structure of dtypes will need
  to be updated.

The new code must pass NumPy's regular test suite, giving some assurance that
the changes are compatible with existing code.

******************************************************************************
Usage and impact
******************************************************************************

We believe the few structures in this section are sufficient to consolidate
NumPy's present functionality and also to support complex user-defined DTypes.

The rest of the NEP fills in details and provides support for the claim.

Again, though Python is used for illustration, the implementation is a C API only; a
future NEP will tackle the Python API.

After implementing this NEP, creating a DType will be possible by implementing
the following outlined DType base class,
that is further described in `nep42_DType class`_:

.. code-block:: python
    :dedent: 0

    class DType(np.dtype):
        type : type        # Python scalar type
        parametric : bool  # (may be indicated by superclass)

        @property
        def canonical(self) -> bool:
            raise NotImplementedError

        def ensure_canonical(self : DType) -> DType:
            raise NotImplementedError

For casting, a large part of the functionality is provided by the "methods" stored
in ``_castingimpl``

.. code-block:: python
    :dedent: 0

        @classmethod
        def common_dtype(cls : DTypeMeta, other : DTypeMeta) -> DTypeMeta:
            raise NotImplementedError

        def common_instance(self : DType, other : DType) -> DType:
            raise NotImplementedError

        # A mapping of "methods" each detailing how to cast to another DType
        # (further specified at the end of the section)
        _castingimpl = {}

For array-coercion, also part of casting:

.. code-block:: python
    :dedent: 0

        def __dtype_setitem__(self, item_pointer, value):
            raise NotImplementedError

        def __dtype_getitem__(self, item_pointer, base_obj) -> object:
            raise NotImplementedError

        @classmethod
        def __discover_descr_from_pyobject__(cls, obj : object) -> DType:
            raise NotImplementedError

        # initially private:
        @classmethod
        def _known_scalar_type(cls, obj : object) -> bool:
            raise NotImplementedError


Other elements of the casting implementation is the ``CastingImpl``:

.. code-block:: python
    :dedent: 0

    casting = Union["safe", "same_kind", "unsafe"]

    class CastingImpl:
        # Object describing and performing the cast
        casting : casting

        def resolve_descriptors(self, Tuple[DTypeMeta], Tuple[DType|None] : input) -> (casting, Tuple[DType]):
            raise NotImplementedError

        # initially private:
        def _get_loop(...) -> lowlevel_C_loop:
            raise NotImplementedError

which describes the casting from one DType to another. In
:ref:`NEP 43 <NEP43>` this ``CastingImpl`` object is used unchanged to
support universal functions.
Note that the name ``CastingImpl`` here will be generically called
``ArrayMethod`` to accommodate both casting and universal functions.


******************************************************************************
Definitions
******************************************************************************
.. glossary::

   dtype
      The dtype *instance*; this is the object attached to a numpy array.

   DType
      Any subclass of the base type ``np.dtype``.

   coercion
      Conversion of Python types to NumPy arrays and values stored in a NumPy
      array.

   cast
      Conversion of an array to a different dtype.

   parametric type
       A dtype whose representation can change based on a parameter value,
       like a string dtype with a length parameter. All members of the current
       ``flexible`` dtype class are parametric. See
       :ref:`NEP 40 <parametric-datatype-discussion>`.

   promotion
      Finding a dtype that can perform an operation on a mix of dtypes without
      loss of information.

   safe cast
      A cast is safe if no information is lost when changing type.

On the C level we use ``descriptor`` or ``descr`` to mean
*dtype instance*. In the proposed C-API, these terms will distinguish
dtype instances from DType classes.

.. note::
   NumPy has an existing class hierarchy for scalar types, as
   seen :ref:`in the figure <nep-0040_dtype-hierarchy>` of
   :ref:`NEP 40 <NEP40>`, and the new DType hierarchy will resemble it. The
   types are used as an attribute of the single dtype class in the current
   NumPy; they're not dtype classes. They neither harm nor help this work.

.. _nep42_DType class:

******************************************************************************
The DType class
******************************************************************************

This section reviews the structure underlying the proposed DType class,
including the type hierarchy and the use of abstract DTypes.

Class getter
==============================================================================

To create a DType instance from a scalar type users now call
``np.dtype`` (for instance, ``np.dtype(np.int64)``). Sometimes it is
also necessary to access the underlying DType class; this comes up in
particular with type hinting because the "type" of a DType instance is
the DType class. Taking inspiration from type hinting, we propose the
following getter syntax::

    np.dtype[np.int64]

to get the DType class corresponding to a scalar type. The notation
works equally well with built-in and user-defined DTypes.

This getter eliminates the need to create an explicit name for every
DType, crowding the ``np`` namespace; the getter itself signifies the
type. It also opens the possibility of making ``np.ndarray`` generic
over DType class using annotations like::

    np.ndarray[np.dtype[np.float64]]

The above is fairly verbose, so it is possible that we will include
aliases like::

    Float64 = np.dtype[np.float64]

in ``numpy.typing``, thus keeping annotations concise but still
avoiding crowding the ``np`` namespace as discussed above. For a
user-defined DType::

    class UserDtype(dtype): ...

one can do ``np.ndarray[UserDtype]``, keeping annotations concise in
that case without introducing boilerplate in NumPy itself. For a
user-defined scalar type::

    class UserScalar(generic): ...

we would need to add a typing overload to ``dtype``::

    @overload
    __new__(cls, dtype: Type[UserScalar], ...) -> UserDtype

to allow ``np.dtype[UserScalar]``.

The initial implementation probably will return only concrete (not abstract)
DTypes.

*This item is still under review.*


Hierarchy and abstract classes
==============================================================================

We will use abstract classes as building blocks of our extensible DType class
hierarchy.

1. Abstract classes are inherited cleanly, in principle allowing checks like
   ``isinstance(np.dtype("float64"), np.inexact)``.

2. Abstract classes allow a single piece of code to handle a multiplicity of
   input types. Code written to accept Complex objects can work with numbers
   of any precision; the precision of the results is determined by the
   precision of the arguments.

3. There's room for user-created families of DTypes. We can envision an
   abstract ``Unit`` class for physical units, with a concrete subclass like
   ``Float64Unit``. Calling ``Unit(np.float64, "m")`` (``m`` for meters) would
   be equivalent to ``Float64Unit("m")``.

4. The implementation of universal functions in :ref:`NEP 43 <NEP43>` may require
   a class hierarchy.

**Example:** A NumPy ``Categorical`` class would be a match for pandas
``Categorical`` objects, which can contain integers or general Python objects.
NumPy needs a DType that it can assign a Categorical to, but it also needs
DTypes like ``CategoricalInt64`` and ``CategoricalObject`` such that
``common_dtype(CategoricalInt64, String)`` raises an error, but
``common_dtype(CategoricalObject, String)`` returns an ``object`` DType. In
our scheme, ``Categorical`` is an abstract type with ``CategoricalInt64`` and
``CategoricalObject`` subclasses.


Rules for the class structure, illustrated :ref:`below <nep42_hierarchy_figure>`:

1. Abstract DTypes cannot be instantiated. Instantiating an abstract DType
   raises an error, or perhaps returns an instance of a concrete subclass.
   Raising an error will be the default behavior and may be required initially.

2. While abstract DTypes may be superclasses, they may also act like Python's
   abstract base classes (ABC) allowing registration instead of subclassing.
   It may be possible to simply use or inherit from Python ABCs.

3. Concrete DTypes may not be subclassed. In the future this might be relaxed
   to allow specialized implementations such as a GPU float64 subclassing a
   NumPy float64.

The
`Julia language <https://docs.julialang.org/en/v1/manual/types/#man-abstract-types-1>`_
has a similar prohibition against subclassing concrete types.
For example methods such as the later ``__common_instance__`` or
``__common_dtype__`` cannot work for a subclass unless they were designed
very carefully.
It helps avoid unintended vulnerabilities to implementation changes that
result from subclassing types that were not written to be subclassed.
We believe that the DType API should rather be extended to simplify wrapping
of existing functionality.

The DType class requires C-side storage of methods and additional information,
to be implemented by a ``DTypeMeta`` class. Each ``DType`` class is an
instance of ``DTypeMeta`` with a well-defined and extensible interface;
end users ignore it.

.. _nep42_hierarchy_figure:
.. figure:: _static/dtype_hierarchy.svg
    :figclass: align-center


Miscellaneous methods and attributes
==============================================================================

This section collects definitions in the DType class that are not used in
casting and array coercion, which are described in detail below.

* Existing dtype methods (:class:`numpy.dtype`) and C-side fields are preserved.

* ``DType.type`` replaces ``dtype.type``. Unless a use case arises,
  ``dtype.type`` will be deprecated.
  This indicates a Python scalar type which represents the same values as
  the DType. This is the same type as used in the proposed `Class getter`_
  and for `DType discovery during array coercion`_.
  (This can may also be set for abstract DTypes, this is necessary
  for array coercion.)

* A new ``self.canonical`` property generalizes the notion of byte order to
  indicate whether data has been stored in a default/canonical way. For
  existing code, "canonical" will just signify native byte order, but it can
  take on new meanings in new DTypes -- for instance, to distinguish a
  complex-conjugated instance of Complex which stores ``real - imag`` instead
  of ``real + imag``. The ISNBO ("is
  native byte order") flag might be repurposed as the canonical flag.

* Support is included for parametric DTypes. A DType will be deemed parametric
  if it inherits from ParametricDType.

* DType methods may resemble or even reuse existing Python slots. Thus Python
  special slots are off-limits for user-defined DTypes (for instance, defining
  ``Unit("m") > Unit("cm")``), since we may want to develop a meaning for these
  operators that is common to all DTypes.

* Sorting functions are moved to the DType class. They may be implemented by
  defining a method ``dtype_get_sort_function(self, sortkind="stable") ->
  sortfunction`` that must return ``NotImplemented`` if the given ``sortkind``
  is not known.

* Functions that cannot be removed are implemented as special methods.
  Many of these were previously defined part of the :c:type:`PyArray_ArrFuncs`
  slot of the dtype instance (``PyArray_Descr *``) and include functions
  such as ``nonzero``, ``fill`` (used for ``np.arange``), and
  ``fromstr`` (used to parse text files).
  These old methods will be deprecated and replacements
  following the new design principles added.
  The API is not defined here. Since these methods can be deprecated and renamed
  replacements added, it is acceptable if these new methods have to be modified.

* Use of ``kind`` for non-built-in types is discouraged in favor of
  ``isinstance`` checks.  ``kind`` will return the ``__qualname__`` of the
  object to ensure uniqueness for all DTypes. On the C side, ``kind`` and
  ``char`` are set to ``\0`` (NULL character).
  While ``kind`` will be discouraged, the current ``np.issubdtype``
  may remain the preferred method for this type of check.

* A method ``ensure_canonical(self) -> dtype`` returns a new dtype (or
  ``self``) with the ``canonical`` flag set.

* Since NumPy's approach is to provide functionality through unfuncs,
  functions like sorting that will be implemented in DTypes might eventually be
  reimplemented as generalized ufuncs.

.. _nep_42_casting:

******************************************************************************
Casting
******************************************************************************

We review here the operations related to casting arrays:

- Finding the "common dtype," returned by :func:`numpy.promote_types` and
  :func:`numpy.result_type`

- The result of calling :func:`numpy.can_cast`

We show how casting arrays with ``astype(new_dtype)`` will be implemented.

`Common DType` operations
==============================================================================

When input types are mixed, a first step is to find a DType that can hold
the result without loss of information -- a "common DType."

Array coercion and concatenation both return a common dtype instance. Most
universal functions use the common DType for dispatching, though they might
not use it for a result (for instance, the result of a comparison is always
bool).

We propose the following implementation:

-  For two DType classes::

       __common_dtype__(cls, other : DTypeMeta) -> DTypeMeta

   Returns a new DType, often one of the inputs, which can represent values
   of both input DTypes.  This should usually be minimal:
   the common DType of ``Int16`` and ``Uint16`` is ``Int32`` and not ``Int64``.
   ``__common_dtype__``  may return NotImplemented to defer to other and,
   like Python operators, subclasses take precedence (their
   ``__common_dtype__`` method is tried first).

-  For two instances of the same DType::

    __common_instance__(self: SelfT, other : SelfT) -> SelfT

   For nonparametric built-in dtypes, this returns a canonicalized copy of
   ``self``, preserving metadata. For nonparametric user types, this provides
   a default implementation.

-  For instances of different DTypes, for example ``>float64`` and ``S8``,
   the operation is done in three steps:

   1. ``Float64.__common_dtype__(type(>float64), type(S8))``
      returns ``String`` (or defers to ``String.__common_dtype__``).

   2. The casting machinery (explained in detail below) provides the
      information that ``">float64"`` casts to ``"S32"``

   3. ``String.__common_instance__("S8", "S32")`` returns the final ``"S32"``.

The benefit of this handoff is to reduce duplicated code and keep concerns
separate. DType implementations don't need to know how to cast, and the
results of casting can be extended to new types, such as a new string encoding.

This means the implementation will work like this::

    def common_dtype(DType1, DType2):
        common_dtype = type(dtype1).__common_dtype__(type(dtype2))
        if common_dtype is NotImplemented:
            common_dtype = type(dtype2).__common_dtype__(type(dtype1))
            if common_dtype is NotImplemented:
                raise TypeError("no common dtype")
        return common_dtype

    def promote_types(dtype1, dtype2):
        common = common_dtype(type(dtype1), type(dtype2))

        if type(dtype1) is not common:
            # Find what dtype1 is cast to when cast to the common DType
            # by using the CastingImpl as described below:
            castingimpl = get_castingimpl(type(dtype1), common)
            safety, (_, dtype1) = castingimpl.resolve_descriptors(
                    (common, common), (dtype1, None))
            assert safety == "safe"  # promotion should normally be a safe cast

        if type(dtype2) is not common:
            # Same as above branch for dtype1.

        if dtype1 is not dtype2:
            return common.__common_instance__(dtype1, dtype2)

Some of these steps may be optimized for nonparametric DTypes.

Since the type returned by ``__common_dtype__`` is not necessarily one of the
two arguments, it's not equivalent to NumPy's "safe" casting.
Safe casting works for ``np.promote_types(int16, int64)``, which returns
``int64``, but fails for::

    np.promote_types("int64", "float32") -> np.dtype("float64")

It is the responsibility of the DType author to ensure that the inputs
can be safely cast to the ``__common_dtype__``.

Exceptions may apply. For example, casting ``int32`` to
a (long enough) string is  at least at this time  considered "safe".
However ``np.promote_types(int32, String)`` will *not* be defined.

**Example:**

``object`` always chooses ``object`` as the common DType.  For
``datetime64`` type promotion is defined with no other datatype, but if
someone were to implement a new higher precision datetime, then::

   HighPrecisionDatetime.__common_dtype__(np.dtype[np.datetime64])

would return ``HighPrecisionDatetime``, and the casting implementation,
as described below, may need to decide how to handle the datetime unit.


**Alternatives:**

-  We're pushing the decision on common DTypes to the DType classes. Suppose
   instead we could turn to a universal algorithm based on safe casting,
   imposing a total order on DTypes and returning the first type that both
   arguments could cast to safely.

   It would be difficult to devise a reasonable total order, and it would have
   to accept new entries. Beyond that, the approach is flawed because
   importing a type can change the behavior of a program. For example, a
   program requiring the common DType of ``int16`` and ``uint16`` would
   ordinarily get the built-in type ``int32`` as the first match; if the
   program adds ``import int24``, the first match becomes ``int24`` and the
   smaller type might make the program overflow for the first time. [1]_

-  A more flexible common DType could be implemented in the future where
   ``__common_dtype__`` relies on information from the casting logic.
   Since ``__commond_dtype__`` is a method a such a default implementation
   could be added at a later time.

-  The three-step handling of differing dtypes could, of course, be coalesced.
   It would lose the value of splitting in return for a possibly faster
   execution. But few cases would benefit. Most cases, such as array coercion,
   involve a single Python type (and thus dtype).


The cast operation
==============================================================================

Casting is perhaps the most complex and interesting DType operation. It
is much like a typical universal function on arrays, converting one input to a
new output, with two distinctions:

- Casting always requires an explicit output datatype.
- The NumPy iterator API requires access to functions that are lower-level
  than what universal functions currently need.

Casting can be complex and may not implement all details of each input
datatype (such as non-native byte order or unaligned access). So a complex
type conversion might entail 3 steps:

1. The input datatype is normalized and prepared for the cast.
2. The cast is performed.
3. The result, which is in a normalized form, is cast to the requested
   form (non-native byte order).

Further, NumPy provides different casting kinds or safety specifiers:

* ``equivalent``, allowing only byte-order changes
* ``safe``, requiring a type large enough to preserve value
* ``same_kind``, requiring a safe cast or one within a kind, like float64 to float32
* ``unsafe``, allowing any data conversion

and in some cases a cast may be just a view.

We need to support the two current signatures of ``arr.astype``:

- For DTypes: ``arr.astype(np.String)``

  - current spelling ``arr.astype("S")``
  - ``np.String`` can be an abstract DType

- For dtypes: ``arr.astype(np.dtype("S8"))``


We also have two signatures of ``np.can_cast``:

- Instance to class: ``np.can_cast(dtype, DType, "safe")``
- Instance to instance: ``np.can_cast(dtype, other_dtype, "safe")``

On the Python level ``dtype`` is overloaded to mean class or instance.

A third ``can_cast`` signature, ``np.can_cast(DType, OtherDType, "safe")``,may be used
internally but need not be exposed to Python.

During DType creation, DTypes will be able to pass a list of ``CastingImpl``
objects, which can define casting to and from the DType.

One of them should define the cast between instances of that DType. It can be
omitted if the DType has only a single implementation and is nonparametric.

Each ``CastingImpl`` has a distinct DType signature:

  ``CastingImpl[InputDtype, RequestedDtype]``

and implements the following methods and attributes:


* To report safeness,

  ``resolve_descriptors(self, Tuple[DTypeMeta], Tuple[DType|None] : input) -> casting, Tuple[DType]``.

  The ``casting`` output reports safeness (safe, unsafe, or same-kind), and
  the tuple is used for more multistep casting, as in the example below.

* To get a casting function,

  ``get_loop(...) -> function_to_handle_cast (signature to be decided)``

  returns a low-level implementation of a strided casting function
  ("transfer function") capable of performing the
  cast.

  Initially the implementation will be *private*, and users will only be
  able to provide strided loops with the signature.

* For performance, a ``casting`` attribute taking a value of  ``equivalent``, ``safe``,
  ``unsafe``, or ``same-kind``.


**Performing a cast**

.. _nep42_cast_figure:

.. figure:: _static/casting_flow.svg
    :figclass: align-center

The above figure illustrates a multistep
cast of an ``int24`` with a value of ``42`` to a string of length 20
(``"S20"``).

We've picked an example where the implementer has only provided limited
functionality: a function to cast an ``int24`` to an ``S8`` string (which can
hold all 24-bit integers). This means multiple conversions are needed.

The full process is:

1. Call

   ``CastingImpl[Int24, String].resolve_descriptors((Int24, String), (int24, "S20"))``.

   This provides the information that ``CastingImpl[Int24, String]`` only
   implements the cast of ``int24`` to ``"S8"``.

2. Since ``"S8"`` does not match ``"S20"``, use

   ``CastingImpl[String, String].get_loop()``

   to find the transfer (casting) function to convert an ``"S8"`` into an ``"S20"``

3. Fetch the transfer function to convert an ``int24`` to an ``"S8"`` using

   ``CastingImpl[Int24, String].get_loop()``

4. Perform the actual cast using the two transfer functions:

   ``int24(42) -> S8("42") -> S20("42")``.

   ``resolve_descriptors`` allows the implementation for

   ``np.array(42, dtype=int24).astype(String)``

   to call

   ``CastingImpl[Int24, String].resolve_descriptors((Int24, String), (int24, None))``.

   In this case the result of ``(int24, "S8")`` defines the correct cast:

   ``np.array(42, dtype=int24).astype(String) == np.array("42", dtype="S8")``.

**Casting safety**

To compute ``np.can_cast(int24, "S20", casting="safe")``, only the
``resolve_descriptors`` function is required and
is called in the same way as in :ref:`the figure describing a cast <nep42_cast_figure>`.

In this case, the calls to ``resolve_descriptors``, will also provide the
information that ``int24 -> "S8"`` as well as ``"S8" -> "S20"`` are safe
casts, and thus also the ``int24 -> "S20"`` is a safe cast.

In some cases, no cast is necessary. For example, on most Linux systems
``np.dtype("long")`` and ``np.dtype("longlong")`` are different dtypes but are
both 64-bit integers. In this case, the cast can be performed using
``long_arr.view("longlong")``. The information that a cast is a view will be
handled by an additional flag.  Thus the ``casting`` can have the 8 values in
total: the original 4 of ``equivalent``, ``safe``, ``unsafe``, and ``same-kind``,
plus ``equivalent+view``, ``safe+view``, ``unsafe+view``, and
``same-kind+view``. NumPy currently defines ``dtype1 == dtype2`` to be True
only if byte order matches. This functionality can be replaced with the
combination of "equivalent" casting and the "view" flag.

(For more information on the ``resolve_descriptors`` signature see the
:ref:`nep42_C-API` section below and :ref:`NEP 43 <NEP43>`.)


**Casting between instances of the same DType**

To keep down the number of casting
steps, CastingImpl must be capable of any conversion between all instances
of this DType.

In general the DType implementer must include ``CastingImpl[DType, DType]``
unless there is only a singleton instance.

**General multistep casting**

We could implement certain casts, such as ``int8`` to ``int24``,
even if the user provides only an ``int16 -> int24`` cast. This proposal does
not provide that, but future work might find such casts dynamically, or at least
allow ``resolve_descriptors`` to return arbitrary ``dtypes``.

If ``CastingImpl[Int8, Int24].resolve_descriptors((Int8, Int24), (int8, int24))``
returns ``(int16, int24)``, the actual casting process could be extended to include
the ``int8 -> int16`` cast. This adds a step.


**Example:**

The implementation for casting integers to datetime would generally
say that this cast is unsafe (because it is always an unsafe cast).
Its ``resolve_descriptors`` function may look like::

     def resolve_descriptors(self, DTypes, given_dtypes):
        from_dtype, to_dtype = given_dtypes
        from_dtype = from_dtype.ensure_canonical()  # ensure not byte-swapped
        if to_dtype is None:
            raise TypeError("Cannot convert to a NumPy datetime without a unit")
        to_dtype = to_dtype.ensure_canonical()  # ensure not byte-swapped

        # This is always an "unsafe" cast, but for int64, we can represent
        # it by a simple view (if the dtypes are both canonical).
        # (represented as C-side flags here).
        safety_and_view = NPY_UNSAFE_CASTING | _NPY_CAST_IS_VIEW
        return safety_and_view, (from_dtype, to_dtype)

.. note::

    While NumPy currently defines integer-to-datetime casts, with the possible
    exception of the unit-less ``timedelta64`` it may be better to not define
    these casts at all.  In general we expect that user defined DTypes will be
    using custom methods such as ``unit.drop_unit(arr)`` or ``arr *
    unit.seconds``.


**Alternatives:**

- Our design objectives are:
  -  Minimize the number of DType methods and avoid code duplication.
  -  Mirror the implementation of universal functions.

- The decision to use only the DType classes in the first step of finding the
  correct ``CastingImpl`` in addition to defining ``CastingImpl.casting``,
  allows to retain the current default implementation of
  ``__common_dtype__`` for existing user defined dtypes, which could be
  expanded in the future.

- The split into multiple steps may seem to add complexity rather than reduce
  it, but it consolidates the signatures of ``np.can_cast(dtype, DTypeClass)``
  and ``np.can_cast(dtype, other_dtype)``.

  Further, the API guarantees separation of concerns for user DTypes. The user
  ``Int24`` dtype does not have to handle all string lengths if it does not
  wish to do so.  Further, an encoding added to the ``String`` DType would
  not affect the overall cast. The ``resolve_descriptors`` function
  can keep returning the default encoding and the ``CastingImpl[String,
  String]`` can take care of any necessary encoding changes.

- The main alternative is moving most of the information that is here pushed
  into the ``CastingImpl`` directly into methods on the DTypes. But this
  obscures the similarity between casting and universal functions. It does
  reduce indirection, as noted below.

- An earlier proposal defined two methods ``__can_cast_to__(self, other)`` to
  dynamically return ``CastingImpl``. This
  removes the requirement to define all possible casts at DType creation
  (of one of the involved DTypes).

  Such an API could be added later. It resembles Python's ``__getattr__`` in
  providing additional control over attribute lookup.


**Notes:**

``CastingImpl`` is used as a name in this NEP to clarify that it implements
all functionality related to a cast. It is meant to be identical to the
``ArrayMethod`` proposed in NEP 43 as part of restructuring ufuncs to handle
new DTypes. All type definitions are expected to be named ``ArrayMethod``.

The way dispatching works for ``CastingImpl`` is planned to be limited
initially and fully opaque. In the future, it may or may not be moved into a
special UFunc, or behave more like a universal function.


.. _nep42_array_coercion:


Coercion to and from Python objects
==============================================================================

When storing a single value in an array or taking it out, it is necessary to
coerce it -- that is, convert it -- to and from the low-level representation
inside the array.

Coercion is slightly more complex than typical casts. One reason is that a
Python object could itself be a 0-dimensional array or scalar with an
associated DType.

Coercing to and from Python scalars requires two to three
methods that largely correspond to the current definitions:

1. ``__dtype_setitem__(self, item_pointer, value)``

2. ``__dtype_getitem__(self, item_pointer, base_obj) -> object``;
   ``base_obj`` is for memory management and usually ignored; it points to
   an object owning the data. Its only role is to support structured datatypes
   with subarrays within NumPy, which currently return views into the array.
   The function returns an equivalent Python scalar (i.e. typically a NumPy
   scalar).

3. ``__dtype_get_pyitem__(self, item_pointer, base_obj) -> object`` (initially
   hidden for new-style user-defined datatypes, may be exposed on user
   request). This corresponds to the ``arr.item()`` method also used by
   ``arr.tolist()`` and returns Python floats, for example, instead of NumPy
   floats.

(The above is meant for C-API. A Python-side API would have to use byte
buffers or similar to implement this, which may be useful for prototyping.)

When a certain scalar
has a known (different) dtype, NumPy may in the future use casting instead of
``__dtype_setitem__``.

A user datatype is (initially) expected to implement
``__dtype_setitem__`` for its own ``DType.type`` and all basic Python scalars
it wishes to support (e.g. ``int`` and ``float``). In the future a
function ``known_scalar_type`` may be made public to allow a user dtype to signal
which Python scalars it can store directly.


**Implementation:** The pseudocode implementation for setting a single item in
an array from an arbitrary Python object ``value`` is (some
functions here are defined later)::

    def PyArray_Pack(dtype, item_pointer, value):
        DType = type(dtype)
        if DType.type is type(value) or DType.known_scalartype(type(value)):
            return dtype.__dtype_setitem__(item_pointer, value)

        # The dtype cannot handle the value, so try casting:
        arr = np.array(value)
        if arr.dtype is object or arr.ndim != 0:
            # not a numpy or user scalar; try using the dtype after all:
            return dtype.__dtype_setitem__(item_pointer, value)

         arr.astype(dtype)
         item_pointer.write(arr[()])

where the call to ``np.array()`` represents the dtype discovery and is
not actually performed.

**Example:** Current ``datetime64`` returns ``np.datetime64`` scalars and can
be assigned from ``np.datetime64``. However, the datetime
``__dtype_setitem__`` also allows assignment from date strings ("2016-05-01")
or Python integers. Additionally the datetime ``__dtype_get_pyitem__``
function actually returns a Python ``datetime.datetime`` object (most of the
time).


**Alternatives:** This functionality could also be implemented as a cast to and
from the ``object`` dtype.
However, coercion is slightly more complex than typical casts.
One reason is that in general a Python object could itself be a
zero-dimensional array or scalar with an associated DType.
Such an object has a DType, and the correct cast to another DType is already
defined::

    np.array(np.float32(4), dtype=object).astype(np.float64)

is identical to::

    np.array(4, dtype=np.float32).astype(np.float64)

Implementing the first ``object`` to ``np.float64`` cast explicitly,
would require the user to take to duplicate or fall back to existing
casting functionality.

It is certainly possible to describe the coercion to and from Python objects
using the general casting machinery, but the ``object`` dtype is special and
important enough to be handled by NumPy using the presented methods.

**Further issues and discussion:**

- The ``__dtype_setitem__`` function duplicates some code, such as coercion
  from a string.

  ``datetime64`` allows assignment from string, but the same conversion also
  occurs for casting from the string dtype to ``datetime64``.

  We may in the future expose the ``known_scalartype`` function to allow the
  user to implement such duplication.

  For example, NumPy would normally use

  ``np.array(np.string_("2019")).astype(datetime64)``

  but ``datetime64`` could choose to use its ``__dtype_setitem__`` instead
  for performance reasons.

- There is an issue about how subclasses of scalars should be handled. We
  anticipate to stop automatically detecting the dtype for
  ``np.array(float64_subclass)`` to be float64. The user can still provide
  ``dtype=np.float64``. However, the above automatic casting using
  ``np.array(scalar_subclass).astype(requested_dtype)`` will fail. In many
  cases, this is not an issue, since the Python ``__float__`` protocol can be
  used instead.  But in some cases, this will mean that subclasses of Python
  scalars will behave differently.

.. note::

    *Example:* ``np.complex256`` should not use ``__float__`` in its
    ``__dtype_setitem__`` method in the future unless it is a known floating
    point type.  If the scalar is a subclass of a different high precision
    floating point type (e.g. ``np.float128``) then this currently loses
    precision without notifying the user.
    In that case ``np.array(float128_subclass(3), dtype=np.complex256)``
    may fail unless the ``float128_subclass`` is first converted to the
    ``np.float128`` base class.


DType discovery during array coercion
==============================================================================

An important step in the use of NumPy arrays is creation of the array from
collections of generic Python objects.

**Motivation:** Although the distinction is not clear currently, there are two main needs::

    np.array([1, 2, 3, 4.])

needs to guess the correct dtype based on the Python objects inside.
Such an array may include a mix of datatypes, as long as they can be
promoted.
A second use case is when users provide the output DType class, but not the
specific DType instance::

    np.array([object(), None], dtype=np.dtype[np.string_])  # (or `dtype="S"`)

In this case the user indicates that ``object()`` and ``None`` should be
interpreted as strings.
The need to consider the user provided DType also arises for a future
``Categorical``::

    np.array([1, 2, 1, 1, 2], dtype=Categorical)

which must interpret the numbers as unique categorical values rather than
integers.

There are three further issues to consider:

1. It may be desirable to create datatypes associated
   with normal Python scalars (such as ``datetime.datetime``) that do not
   have a ``dtype`` attribute already.

2. In general, a datatype could represent a sequence, however, NumPy currently
   assumes that sequences are always collections of elements
   (the sequence cannot be an element itself).
   An example would be a ``vector`` DType.

3. An array may itself contain arrays with a specific dtype (even
   general Python objects).  For example:
   ``np.array([np.array(None, dtype=object)], dtype=np.String)``
   poses the issue of how to handle the included array.

Some of these difficulties arise because finding the correct shape
of the output array and finding the correct datatype are closely related.

**Implementation:** There are two distinct cases above:

1. The user has provided no dtype information.

2. The user provided a DType class  -- as represented, for example, by ``"S"``
   representing a string of any length.

In the first case, it is necessary to establish a mapping from the Python type(s)
of the constituent elements to the DType class.
Once the DType class is known, the correct dtype instance needs to be found.
In the case of strings, this requires to find the string length.

These two cases shall be implemented by leveraging two pieces of information:

1. ``DType.type``: The current type attribute to indicate which Python scalar
   type is associated with the DType class (this is a *class* attribute that always
   exists for any datatype and is not limited to array coercion).

2. ``__discover_descr_from_pyobject__(cls, obj) -> dtype``: A classmethod that
   returns the correct descriptor given the input object.
   Note that only parametric DTypes have to implement this.
   For nonparametric DTypes using the default instance will always be acceptable.

The Python scalar type which is already associated with a DType through the
``DType.type`` attribute maps from the DType to the Python scalar type.
At registration time, a DType may choose to allow automatically discover for
this Python scalar type.
This requires a lookup in the opposite direction, which will be implemented
using global a mapping (dictionary-like) of::

   known_python_types[type] = DType

Correct garbage collection requires additional care.
If both the Python scalar type (``pytype``) and ``DType`` are created dynamically,
they will potentially be deleted again.
To allow this, it must be possible to make the above mapping weak.
This requires that the ``pytype`` holds a reference of ``DType`` explicitly.
Thus, in addition to building the global mapping, NumPy will store the ``DType`` as
``pytype.__associated_array_dtype__`` in the Python type.
This does *not* define the mapping and should *not* be accessed directly.
In particular potential inheritance of the attribute does not mean that NumPy will use the
superclasses ``DType`` automatically. A new ``DType`` must be created for the
subclass.

.. note::

    Python integers do not have a clear/concrete NumPy type associated right
    now. This is because during array coercion NumPy currently finds the first
    type capable of representing their value in the list of `long`, `unsigned
    long`, `int64`, `unsigned int64`, and `object` (on many machines `long` is
    64 bit).

    Instead they will need to be implemented using an ``AbstractPyInt``. This
    DType class can then provide ``__discover_descr_from_pyobject__`` and
    return the actual dtype which is e.g. ``np.dtype("int64")``. For
    dispatching/promotion in ufuncs, it will also be necessary to dynamically
    create ``AbstractPyInt[value]`` classes (creation can be cached), so that
    they can provide the current value based promotion functionality provided
    by ``np.result_type(python_integer, array)`` [2]_ .

To allow for a DType to accept inputs as scalars that are not basic Python
types or instances of ``DType.type``, we use ``known_scalar_type`` method.
This can allow discovery of a ``vector`` as a scalar (element) instead of a sequence
(for the command ``np.array(vector, dtype=VectorDType)``) even when ``vector`` is itself a
sequence or even an array subclass. This will *not* be public API initially,
but may be made public at a later time.

**Example:** The current datetime DType requires a
``__discover_descr_from_pyobject__`` which returns the correct unit for string
inputs.  This allows it to support::

    np.array(["2020-01-02", "2020-01-02 11:24"], dtype="M8")

By inspecting the date strings. Together with the common dtype
operation, this allows it to automatically find that the datetime64 unit
should be "minutes".


**NumPy internal implementation:** The implementation to find the correct dtype
will work similar to the following pseudocode::

    def find_dtype(array_like):
        common_dtype = None
        for element in array_like:
            # default to object dtype, if unknown
            DType = known_python_types.get(type(element), np.dtype[object])
            dtype = DType.__discover_descr_from_pyobject__(element)

            if common_dtype is None:
                common_dtype = dtype
            else:
                common_dtype = np.promote_types(common_dtype, dtype)

In practice, the input to ``np.array()`` is a mix of sequences and array-like
objects, so that deciding what is an element requires to check whether it
is a sequence.
The full algorithm (without user provided dtypes) thus looks more like::

    def find_dtype_recursive(array_like, dtype=None):
        """
        Recursively find the dtype for a nested sequences (arrays are not
        supported here).
        """
        DType = known_python_types.get(type(element), None)

        if DType is None and is_array_like(array_like):
            # Code for a sequence, an array_like may have a DType we
            # can use directly:
            for element in array_like:
                dtype = find_dtype_recursive(element, dtype=dtype)
            return dtype

        elif DType is None:
            DType = np.dtype[object]

        # dtype discovery and promotion as in `find_dtype` above

If the user provides ``DType``, then this DType will be tried first, and the
``dtype`` may need to be cast before the promotion is performed.

**Limitations:** The motivational point 3. of a nested array
``np.array([np.array(None, dtype=object)], dtype=np.String)`` is currently
(sometimes) supported by inspecting all elements of the nested array.
User DTypes will implicitly handle these correctly if the nested array
is of ``object`` dtype.
In some other cases NumPy will retain backward compatibility for existing
functionality only.
NumPy uses such functionality to allow code such as::

    >>> np.array([np.array(["2020-05-05"], dtype="S")], dtype=np.datetime64)
    array([['2020-05-05']], dtype='datetime64[D]')

which discovers the datetime unit ``D`` (days).
This possibility will not be accessible to user DTypes without an
intermediate cast to ``object`` or a custom function.

The use of a global type map means that an error or warning has to be given if
two DTypes wish to map to the same Python type. In most cases user DTypes
should only be implemented for types defined within the same library to avoid
the potential for conflicts. It will be the DType implementor's responsibility
to be careful about this and use avoid registration when in doubt.

**Alternatives:**

- Instead of a global mapping, we could rely on the scalar attribute
  ``scalar.__associated_array_dtype__``. This only creates a difference in
  behavior for subclasses, and the exact implementation can be undefined
  initially. Scalars will be expected to derive from a NumPy scalar. In
  principle NumPy could, for a time, still choose to rely on the attribute.

- An earlier proposal for the ``dtype`` discovery algorithm used a two-pass
  approach, first finding the correct ``DType`` class and only then
  discovering the parametric ``dtype`` instance. It was rejected as
  needlessly complex. But it would have enabled value-based promotion
  in universal functions, allowing::

    np.add(np.array([8], dtype="uint8"), [4])

  to return a ``uint8`` result (instead of ``int16``), which currently happens for::

    np.add(np.array([8], dtype="uint8"), 4)

  (note the list ``[4]`` instead of scalar ``4``).
  This is not a feature NumPy currently has or desires to support.

**Further issues and discussion:** It is possible to create a DType
such as Categorical, array, or vector which can only be used if ``dtype=DType``
is provided. Such DTypes cannot roundtrip correctly. For example::

    np.array(np.array(1, dtype=Categorical)[()])

will result in an integer array. To get the original ``Categorical`` array
``dtype=Categorical`` will need to be passed explicitly.
This is a general limitation, but round-tripping is always possible if
``dtype=original_arr.dtype`` is passed.


.. _nep42_c-api:

******************************************************************************
Public C-API
******************************************************************************

DType creation
==============================================================================

To create a new DType the user will need to define the methods and attributes
outlined in the `Usage and impact`_ section and detailed throughout this
proposal.

In addition, some methods similar to those in :c:type:`PyArray_ArrFuncs` will
be needed for the slots struct below.

As mentioned in :ref:`NEP 41 <NEP41>`, the interface to define this DType
class in C is modeled after :PEP:`384`: Slots and some additional information
will be passed in a slots struct and identified by ``ssize_t`` integers::

    static struct PyArrayMethodDef slots[] = {
        {NPY_dt_method, method_implementation},
        ...,
        {0, NULL}
    }

    typedef struct{
      PyTypeObject *typeobj;    /* type of python scalar or NULL */
      int flags                 /* flags, including parametric and abstract */
      /* NULL terminated CastingImpl; is copied and references are stolen */
      CastingImpl *castingimpls[];
      PyType_Slot *slots;
      PyTypeObject *baseclass;  /* Baseclass or NULL */
    } PyArrayDTypeMeta_Spec;

    PyObject* PyArray_InitDTypeMetaFromSpec(PyArrayDTypeMeta_Spec *dtype_spec);

All of this is passed by copying.

**TODO:** The DType author should be able to define new methods for the
DType, up to defining a full object, and, in the future, possibly even
extending the ``PyArrayDTypeMeta_Type`` struct. We have to decide what to make
available initially. A solution may be to allow inheriting only from an
existing class: ``class MyDType(np.dtype, MyBaseclass)``. If ``np.dtype`` is
first in the method resolution order, this also prevents an undesirable
override of slots like ``==``.

The ``slots`` will be identified by names which are prefixed with ``NPY_dt_``
and are:

* ``is_canonical(self) -> {0, 1}``
* ``ensure_canonical(self) -> dtype``
* ``default_descr(self) -> dtype`` (return must be native and should normally be a singleton)
* ``setitem(self, char *item_ptr, PyObject *value) -> {-1, 0}``
* ``getitem(self, char *item_ptr, PyObject (base_obj) -> object or NULL``
* ``discover_descr_from_pyobject(cls, PyObject) -> dtype or NULL``
* ``common_dtype(cls, other) -> DType, NotImplemented, or NULL``
* ``common_instance(self, other) -> dtype or NULL``

Where possible, a default implementation will be provided if the slot is
omitted or set to ``NULL``. Nonparametric dtypes do not have to implement:

* ``discover_descr_from_pyobject`` (uses ``default_descr`` instead)
* ``common_instance`` (uses ``default_descr`` instead)
* ``ensure_canonical`` (uses ``default_descr`` instead).

Sorting is expected to be implemented using:

* ``get_sort_function(self, NPY_SORTKIND sort_kind) -> {out_sortfunction, NotImplemented, NULL}``.

For convenience, it will be sufficient if the user implements only:

* ``compare(self, char *item_ptr1, char *item_ptr2, int *res) -> {-1, 0, 1}``


**Limitations:** The ``PyArrayDTypeMeta_Spec`` struct is clumsy to extend (for
instance, by adding a version tag to the ``slots`` to indicate a new, longer
version). We could use a function to provide the struct; it would require
memory management but would allow ABI-compatible extension (the struct is
freed again when the DType is created).


CastingImpl
==============================================================================

The external API for ``CastingImpl`` will be limited initially to defining:

* ``casting`` attribute, which can be one of the supported casting kinds.
  This is the safest cast possible. For example, casting between two NumPy
  strings is of course "safe" in general, but may be "same kind" in a specific
  instance if the second string is shorter. If neither type is parametric the
  ``resolve_descriptors`` must use it.

* ``resolve_descriptors(PyArrayMethodObject *self, PyArray_DTypeMeta *DTypes[2],
  PyArray_Descr *dtypes_in[2], PyArray_Descr *dtypes_out[2], NPY_CASTING *casting_out)
  -> int {0, -1}`` The out
  dtypes must be set correctly to dtypes which the strided loop
  (transfer function) can handle.  Initially the result must have instances
  of the same DType class as the ``CastingImpl`` is defined for. The
  ``casting`` will be set to ``NPY_EQUIV_CASTING``, ``NPY_SAFE_CASTING``,
  ``NPY_UNSAFE_CASTING``, or ``NPY_SAME_KIND_CASTING``.
  A new, additional flag,
  ``_NPY_CAST_IS_VIEW``, can be set to indicate that no cast is necessary and a
  view is sufficient to perform the cast. The cast should return
  ``-1`` when an error occurred. If a cast is not possible (but no error
  occurred), a ``-1`` result should be returned *without* an error set.
  *This point is under consideration, we may use ``-1`` to indicate
  a general error, and use a different return value for an impossible cast.*
  This means that it is *not* possible to inform the user about why a cast is
  impossible.

* ``strided_loop(char **args, npy_intp *dimensions, npy_intp *strides,
  ...) -> int {0, -1}`` (signature will be fully defined in :ref:`NEP 43 <NEP43>`)

This is identical to the proposed API for ufuncs. The additional ``...``
part of the signature will include information such as the two ``dtype``\s.
More optimized loops are in use internally, and
will be made available to users in the future (see notes).

Although verbose, the API will mimic the one for creating a new DType:

.. code-block:: C

    typedef struct{
      int flags;                  /* e.g. whether the cast requires the API */
      int nin, nout;              /* Number of Input and outputs (always 1) */
      NPY_CASTING casting;        /* The "minimal casting level" */
      PyArray_DTypeMeta *dtypes;  /* input and output DType class */
      /* NULL terminated slots defining the methods */
      PyType_Slot *slots;
    } PyArrayMethod_Spec;

The focus differs between casting and general ufuncs.  For example, for casts
``nin == nout == 1`` is always correct, while for ufuncs ``casting`` is
expected to be usually `"no"`.

**Notes:** We may initially allow users to define only a single loop.
Internally NumPy optimizes far more, and this should be made public
incrementally in one of two ways:

* Allow multiple versions, such as:

  * contiguous inner loop
  * strided inner loop
  * scalar inner loop

* Or, more likely, expose the ``get_loop`` function which is passed additional
  information, such as the fixed strides (similar to our internal API).

* The casting level denotes the minimal guaranteed casting level and can be
  ``-1`` if the cast may be impossible.  For most non-parametric casts, this
  value will be the casting level.  NumPy may skip the ``resolve_descriptors``
  call for ``np.can_cast()`` when the result is ``True`` based on this level.

The example does not yet include setup and error handling. Since these are
similar to the UFunc machinery, they  will be defined in :ref:`NEP 43 <NEP43>` and then
incorporated identically into casting.

The slots/methods used will be prefixed with ``NPY_meth_``.


**Alternatives:**

- Aside from name changes and signature tweaks, there seem to be few
  alternatives to the above structure. The proposed API using ``*_FromSpec``
  function is a good way to achieve a stable and extensible API. The slots
  design is extensible and can be changed without breaking binary
  compatibility. Convenience functions can still be provided to allow creation
  with less code.

- One downside is that compilers cannot warn about function-pointer
  incompatibilities.


******************************************************************************
Implementation
******************************************************************************

Steps for implementation are outlined in the Implementation section of
:ref:`NEP 41 <NEP41>`. In brief, we first will rewrite the internals of
casting and array coercion. After that, the new public API will be added
incrementally. We plan to expose it in a preliminary state initially to gain
experience. All functionality currently implemented on the dtypes will be
replaced systematically as new features are added.


******************************************************************************
Alternatives
******************************************************************************

The space of possible implementations is large, so there have been many
discussions, conceptions, and design documents. These are listed in
:ref:`NEP 40 <NEP40>`. Alternatives were also been discussed in the
relevant sections above.


******************************************************************************
References
******************************************************************************

.. [1] To be clear, the program is broken: It should not have stored a value
  in the common DType that was below the lowest int16 or above the highest
  uint16. It avoided overflow earlier by an accident of implementation.
  Nonetheless,  we insist that program behavior not be altered just by
  importing a type.

.. [2] NumPy currently inspects the value to allow the operations::

     np.array([1], dtype=np.uint8) + 1
     np.array([1.2], dtype=np.float32) + 1.

   to return a ``uint8`` or ``float32`` array respectively.  This is
   further described in the documentation for :func:`numpy.result_type`.


******************************************************************************
Copyright
******************************************************************************

This document has been placed in the public domain.
