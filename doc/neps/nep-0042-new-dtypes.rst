========================================
NEP 42 — Implementation of New DataTypes
========================================

:title: Extensible Datatypes for NumPy
:Author: Sebastian Berg
:Status: Draft
:Type: Standard
:Created: 2019-07-17


Abstract
--------

NEP 40 and 41 detailed the need for the creation of a new datatype system within
NumPy to serve downstream use cases better and improve the maintainability
and the ability to extend NumPy.
A main issue with the current dtype API is that datatypes are written as
a single Python class with special instances for each of the actual datatypes.
While this certainly has been a practical approach in implementing numerical
datatypes, it does not allow to naturally split up logic:
Functions, such as ``can_cast`` have explicit logic for each datatype.
This monolithic code structure means that user defined datatypes do not have
the same possibilities as NumPy datatypes have.
It also makes reasoning/modifying datatypes harder, since a modification
in one datatype touches code involving others.
As detailed in NEP 41, the desired general design is to create classes for
each of the NumPy provided datatypes, meaning that ``np.dtype("float64")``
is an instance of a ``Float64`` type and a subclass of ``np.dtype``.
This will allow moving all logic into special methods on the ``np.dtype``
subclasses.

This document proposes this new API for the datatypes itself.
A second proposal NEP YY details the proposed changes to the universal
functions.
Note that only the implementation of both NEPs will provide the desired full
functionality.
The implementation may be possible in steps, however, there is little or no
gain expected if only this proposal is implemented.
On the other hand, this proposal is a prerequisite for the universal function
improvements.




Detailed Description
--------------------

NEP 41 makes the decision to create a full class hierarchy for datatypes
using special methods on the datatypes to implement any necessary logic.
This NEP suggests a specific choice of API, and set of these methods.
Here these are suggested as C-API slots, however, this translates identically
to python methods.

Additionally, the NEP proposes to implement the notion of *abstract* dtypes
below.
Further, we detail – in part – how the proposed methods (C-API slots)
enable all necessary use cases.

Each section below will begin with a short description of the reason for the
specific choice.


Nomenclature
^^^^^^^^^^^^

As a brief note on nomenclature, it should be noted that ``dtype`` normally
denotes the dtype *instance*, which is the object attached to a numpy array.
On the other hand the ``DType`` class or type is the subclass of ``np.dtype``.
On the C-level we currently use the name ``descriptor`` or ``descr`` in many
cases interchangeable with the dtype instance, this will be used in proposed
C-API names for clarification.
Note that the notion of dtype class is currently represented only as
the ``dtype.num`` and ``dtype.char``.
Please see the dtype hierarchy figure as an illustration of this important
distinction.

Currently classes do exist as ``np.float64``, however,
these are not dtypes but the corresponding scalar classes
(see also NEP 40 and 41 on a discussion why these are largely unrelated to the proposed changes).


Hierarchy of DataTypes and Abstract DTypes
""""""""""""""""""""""""""""""""""""""""""

**Motivation:**
The creation of a DType classes has already been decided in NEP 41.
This proposes the notion of _abstract_ DTypes.
There are multiple reasons for this:
1. It allows the definition of a class hierarchy, allowing in principle to write
   ``isinstance(np.dtype("float64"), np.Inexact)``.
   **DEPENDING ON WHAT WE DECIDE FOR UFUNCS, THAT MAY BE IMPORTANT THERE**
2. Allowing to write such an abstract datatype can enable code such as
   ``arr.astype(np.Complex)`` as a way to spell the desire to cast to a
   complex data type.
3. It anticipates the creation of families of DTypes by users.
   For example allowing the creation of a ``Unit`` with a concrete
   ``Float64Unit``.


**Description:**

The figure below shows the proposed datatype hierarchy.
It should be noted that abstract DTypes are distinct mainly in two ways:

1. They do not have instances. Instantiating an abstract DType has to return
   a concrete subclass or raise an error (default, and possibly enforced
   initially).
2. Unlike concrete DTypes, abstract DTypes can be superclasses, they may also
   serve like pythons abstract base classes (ABC). Likely this would make them
   instances of pythons ABCMeta (which would probably require an ``DTypeMetaABC``
   class though).

These two rules are identical to the type choices made for example in the Julia
language.
It allows for the creation of a datatype hierarchy, but avoids issues with
subclassing concrete DTypes directly:
Logic such as ``can_cast`` does not cleanly inherit e.g. from a
``Int64`` to a ``Datetime64`` even though the ``Datetime64`` could be seen
as an integer with only a unit attached (and thus as a subclass).
From the user perspective the introduction of abstract DTypes is
thus rather the refusal of allowing subclassing
and inheritance for concrete DTypes.
While avoiding possible problems with subclassing and inheritance it still
allows the transparent definition of a class/type hierarchy.
Note that subclassing may be a possible mechanism to extend the datatypes
in the future, however, most likely not to implement new datatypes, but rather
specialized implementations such as a GPU float64 subclassing a NumPy float64.

As largely an implementation detail the DType class will require the homogeneous
storage of additional, fixed, methods and information.
This requires the creation of a ``DTypeMeta`` class.
Each ``DType`` class is an instance of ``DTypeMeta`` with a well defined
and extensible interface.

.. image:: _static/dtype_hierarchy.svg


Methods/Slots defined for each DType
""""""""""""""""""""""""""""""""""""

NEP 33 detailed that all logic should be defined through special methods
on the DTypes.
This section will list a specific set of methods (in the form of python methods).
The C-side equivalent slot signature will be summarized below after proposing
the general C-API for defining new Datatypes.
Note that while the slots are defined as special python methods here, this is
for convenience to the reader and *not* meant to imply the identical exposure
as a Python API.
This should be proposed as a separate NEP.

Some of the methods may be similar or even reuse existing Python slots.
User defined DType classes are discouraged from defining or useing Pythons
special slots without feedback, in order to allow defining them later.
For example ``dtype1 & dtype2`` (or maybe ``|``) could be a shorthand for
``np.common_dtype``, and comparisons should be defined mainly through casting
logic.



Additional Information
^^^^^^^^^^^^^^^^^^^^^^

In addition to the below more detailed methods, the following general
information is currently provided and will be defined on the class:

* ``parametric``:
  * Many dtypes are not parametric, i.e. they have a canonical
    representation and casting from/to it is always safe.
  * DTypes which are not parametric, must provide a canonical dtype instance
    which should be a singleton.
  * Parametric dtypes must define some additional methods.
  * We can additionally create a ``ParametricDType`` class for this purpose,
    although for the C-side API a flag seems simpler.

* ``is_canonical(self) -> Boolean`` method (Alternative: an attribute?)
  * Instead of byteorder, we may want an ``is_canonical`` flag (we could just
    reuse the ISNBO flag – "is native byte order"),
    this flag signals that the data is stored in the default/canonical way.
    In practice this is always an NBO check, but generalization should be possible.
    A use case is a complex-conjugated instance of Complex which is a
    non-canonical representation, but may be native byte order.

* ``ensure_canonical(self) -> dtype`` return a new dtype, or new reference to ``self``,
  the returned dtype must fullfill ``dtype.is_native() is True`` and be otherwise
  identical.

* ``DType.type`` is the associated scalar type, also ``dtype.type`` is defined
  in the same way and must be identical.

Additionally, existing methods (and C-side fields) will be provided, although
fields such as the "kind" and and "char" may be set to invalid values on
the C-side, and access may error on the Python side.
(This may be adapted later if it turns out to be an issue for newly implemented
user dtypes).

Another example of methods that should be moved to the DType class are the
various sorting functions, which shall be implemented by defining a slot:

* ``__dtype_get_sort_function__(self, sortkind="stable") -> sortfunction``

Which returns must return ``NotImplemented`` if the given ``sortkind`` is not
known.
Similarly, any function implemented previously which cannot be removed will
be implemented as a special method.
Since these methods can be deprecated and new (renamed) replacements added,
the API is not fixed and it is acceptable if it changes.

Another, possibly preferable (but long term) goal may be to have a gufunc
for each sorting kind and register an implementation with that.


Coercion to and from Python Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coercing to and from python scalars requires two functions:

1. ``__dtype_setitem__(self, item_pointer, value)``
2. ``__dtype_getitem__(self, item_pionter, base_obj) -> object``
   The ``base_obj`` should be ignored normally, it is provided *only* for
   memory management purposes, pointing to an object owning the data.
   It exists only to allow support of structured datatypes with subarrays
   within NumPy, which (currently) return views into the array.
   The function returns an equivalent Python scalar (i.e. typically a NumPy
   scalar).
3. ``__dtype_get_pyitem(self, item_pointer, base_obj) -> object``
   (initially hidden for new style user defined datatypes, may be exposed
   on user request).

(The above is meant for C-API. A Python side API would have to use byte
buffers or similar to implement this, which may be useful e.g. for prototyping.)

This largely corresponds to the current definitions. Note the the last
item may be hidden from user defined datatypes.
It currently needs to exist to support the difference between indexing a
numpy array and ``arr.item()``/``arr.tolist()`` which return python floats
and integers instead of NumPy ones.
Current known user defined datatypes do not use the distinction and return
``getitem`` for both of these (user dtypes can choose this using a flag).

**Alternative:**

Setitem and getitem are really just casting from and two object datatype,
and if defined like this, the object datatype would use them in its implementation.

The use of these function may initially be the more practical variant, since it
is close to the current implementation and optimization into strided loops
seems unnecessary for objects.
However, in the future it may be desirable to slowly move towards a purely
casting based API.


DType Discovery during Array Coercion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An important step in the usage of NumPy arrays is the creation of the array
itself from generic python objects.

**Motivation:**
Although the distinction is not clear currently, there are two main needs::

    np.array([1, 2, 3, 4.])

needs to guess the correct dtype based on the python objects inside.
Such an array may include a mix of datatypes, as long as they can be clearly
promoted.
Currently not clearly distinct (but partially existing for strings) is the
use case of::

    # currently `np.String` would be spelled `"S"`. 
    np.array([object(), None], dtype=np.String)

which forces each object to be interpreted as string. This is anticipated
to be useful for example for categorical datatypes::

    np.array([1, 2, 1, 1, 2], dtype=Categorical)

can be implemented.
(For NumPy ``datetime64`` this is also currently used to allow string input.)

There are three further issues to consider:

1. It is desirable that datatypes can be created which are associated to normal
   python scalars (such as ``datetime.Datetime``), which do not have a ``dtype``
   attribute already.
2. A general datatype could represent a sequence, however, NumPy currently
   assumes that sequences always include elements (the sequence cannot be an
   element itself). An example for this is a ``vector`` DType.
3. An array may contain arrays with specific a dtype (even general python objects).
   In the above example of ``np.array([np.array(None, dtype=object)], dtype=np.String)``
   this creates the issue of how to handle the included array.

Some of these difficulties arise due to the fact that finding the correct shape
of the output array and finding the correct datatype are closely related.

**Implementation:**

There are two distinct cases given above: First, when the user has provided no
dtype information, and second when the user provided a DType class – 
a notion that is currently represented e.g. by ``"S"``, but not cleanly supported.

In the first case, it is necessary to establish a mapping from the python type
that was put in to the DType class.
Further, the correct dtype instance needs to be found.
This shall be implemented by the use of two informations:

1. ``type``: The current type attribute to indicate which python type is
   associated with the DType class (this is a *class* attribute that always
   exist for any datatype and is not limited to array coercion).
2. The reverse lookup remain hardcoded for the basic Python types initially.
   Otherwise rely on creating scalar classes deriving from ``np.generic``
   (these classes could be automatically generated) or a new superclass of
   ``np.generic``.
   This method may be expanded later (see alternatives).
3. ``__discover_descr_from_pyobject__(cls, obj) -> dtype``: A classmethod that
   returns the correct descriptor given the input object.
   *Note that only parametric DTypes have to implement this*, most datatypes
   can simply use a default (singleton) dtype instance which is found only
   based on the ``type(obj)`` of the Python object.

The ``type`` which is already associated with any dtype through the ``dtype.type``
attribute maps the dtype to the Python type.
However, initially we anticipate automatic conversion only for scalars which
subclass from a base scalar object, such as ``np.generic`` (although possibly
a class below ``np.generic`` to avoid arry-scalar behaviour). 
This will be cached globally to create a mapping (dictionary)
``knwon_python_types[type] = DType``.
NumPy currently uses a small hard-coded a mapping and conversion of numpy scalars
(inheriting from ``np.generic``) to achieve this, however, this forces a certain
structure on the associated ``type``.

.. note::

    Python integers do not have a clear/specific NumPy type associated to them
    right now.  Instead they will need to be be implemented using an
    ``AbstractPyInt``, this DType class, can than provide
    ``__discover_descr_from_pyobject__`` and return the actual dtype which
    is e.g. ``np.dtype("int64")``.
    For dispatching/promotion in ufuncs, it will also probably be necessary
    to dynamically create ``AbstractPyInt[value]`` classes (creation can be
    cached), so that they can provide the current functionality provided by
    ``np.result_type(python_integer, array)``.

Once the datatype is found (a step that is skipped if the user provides it),
the actual descriptor has to be discovered.
For most datatypes, which are not parametric, this is always a canonical default
instance and ``__discover_descr_from_pyobject__`` does not need to be defined.
For parametric datatypes, however, such as strings, it is necessary to find the
correct string length and thus inspect the given python object, this is
also necessary for example for a Categorical datatype.
In this case ``__discover_descr_from_pyobject__`` is called and must return
a dtype instance.
*Implementors should aim to avoid complex logic in this function.*

After the dtype instance is found, the common dtype with the current previous
found common dtype instance to find a single dtype that represents all inputs
and thus support for example mixed integer and floating point input.

Any object which cannot be associated with a dtype will be handled as a
sequence and discovered recursively.

If one of the objects is an array, or array-like object, its datatype is
used directly.
In the case where the user provided a DType, it will be force-cast to the
user given datatype (class). Force-casting is the current behaviour, and it
may be made more restrictive in the future.

**Limitations:**

The above issue 3. is currently (sometimes) supported by NumPy so that
the values of an included array are inspected.
Support in those cases may be kept for compatibility, however,
it will not be exposed to user datatypes.
This means that if e.g. an array with a parametric string dtype is coerced above
(or cast) to an array of a fixed length string dtype (with unknown length),
this will result in an error.
Such a conversion will require passing the correct DType (fixed length of the
string) or providing a utility function to the user.

The use of a global type map means that an error or warning has to be given
if two DTypes wish to map to the same python type, in most cases DTypes user
DTypes should only be implemented for types defined within the same library to
avoid the potential for conflicts and .
It is the DType implementors responsibility to be careful about this and use
the flag to disable registration when in doubt.

**Alternatives:**

Instead of only supporting a specific set of Python types and additionally
scalars subclassing a NumPy scalar class, we could also create a global
mapping from Python type to DType.
In general, such a mapping should be easy to add at a later point,
so that it seems safer to require subclassing initially.
A global mapping has the disadvantage that user created datatype could
"register" themselves for the same Python type.

An initial alternative suggestion was to use a two-pass approach instead.
The first pass would only find the correct DType class, and the second pass
would then find correct dtype instance (the second pass is often not necessary).
The advantage of this is that the DType class information is vital for universal
functions to decide which loop to execute.
The first pass would provide the full information necessary for value based
casting currently implemented for scalars, giving even the possibility of
expanding it to e.g. list inputs ``np.add(np.array([8], dtype="uint8"), [4])``
giving a ``uint8`` result.
This is mainly related to the question to how the common dtype is found above.
It seems unlikely that this is useful, and similar to a global, could be
added later if deemed necessary.


Common DType Operations
^^^^^^^^^^^^^^^^^^^^^^^

Numpy currently provides functionality of ``np.result_type`` and
``np.promote_types`` (while more common, ``np.result_type`` has some more
complex logic due to implementing value based promotion [value_based]_).
(Note that the name ``np.common_type`` is associated with the scalars and not
directly with dtypes and is of limited usefulness)

To distinguish between the promotion occuring during universal function application,
here common type operation is used instead of promotion.

**Motivation:**
This common type operations is vital for the above array coercion when different
input types are mixed.
It also provides the logic currently used to decide the output dtype of
``np.concatenate()`` and on its own is a useful operation.

Furthermore, it may be used to find the correct dtype to use for functions with
different inputs (including universal functions).
This includes an interesting distinction:

1. Universal functions use the DType classes for dispatching, they thus
   require the common DType class (as a first step).
   While this can help with finding the correct loop to execute, the loop
   may not need the actual common dtype instance.
   (Hypothetical example:
   ``float_arr + string_arr -> string``, but the output string length is
   not the same as ``np.concatenate(float_arr, string_arr)).dtype``.)
2. Array coercion, and concatenate require the common dtype *instance*.   

**Implementation:**
The implementation of the common dtype (instance) has some overlap with
casting.
Casting from a specific dtype (Float64) to a String needs to find
the correct string length (a step that is mainly necessary for parametric dtypes).

We propose the following implementation:

1. ``__common_dtype__(cls, other : DTypeMeta) -> DTypeMeta`` answers what the common
   DType class is. It may return ``NotImplemented`` to defer to ``other``.
   (For abstract DTypes, subclasses get precedence, concrete types are always
   leaves, so always get preference or are tried from left to right). 
2. ``__common_instance__(self, other : cls) -> cls`` is used when two instances
   of the same DType are given. For builtin dtypes (that are not parametric), this
   currently always returns ``self`` (but ensures native byte order).
   This is to preserve metadata. We can thus provide a default implementation
   for non-parametric user dtypes.

These two cases do *not* cover the case where two different dtype instances
need to be promoted. For example `">float64"` and `"S8"`.
The solution is partially "outsourced" to the casting machinery by
splitting the operation up into three steps:

1. ``__common_dtype__(type(>float64), type(S8))`` returns `String`.
2. The casting machinery provides the information that `">float64"` casts
   to `"S32"` (see below for how casting will be defined).
3. ``__common_instance__("S8", "S32")`` returns the final `"S32"`. 

The main reason for this is to avoid identical functionality in different
which may lead to inconsistent implementations.
The design (together with casting) naturally separates the concerns of
different Datatypes.
Even if tempted, the above Float64 cannot assume it knows how to step 3 correctly.


**Note:**

The common type operation cannot be simplified to using only safe casting
logic.
As a fall-back testing whether one of the inputs can be safely cast to the
other could be used when no specific ``__common_dtype__`` is implemented.
However, this does not allow for the case of::

    np.promote_types("int64", "float32") -> np.dtype("float64")

However, *if one DType can be safely cast to the other this should also be
the common DType*. The operation is mainly required because the common dtype
will often be neither of the inputs.

If a "safe casting" fallback is desired (the default implementation),
this has to be implemented by the overriding implementation.

**Alternatives:**

The use of casting for common dtype (instance) neatly separates the concerns
and allows for a minimal set of duplicate functionality being implemented.
In cases of mixed DType (classes), it also adds an additional indirection
into finding the common dtype.
The common dtype (of two instances) could thus be implemented explicitly to avoid
this indirection, potentially only as a fast-path.
The above suggestion assumes that this is, however, not a speed relevant path,
since in most cases, e.g. in array coercion, only a single python type (and thus
dtype) is involved.
The proposed design hinges in the implementation of casting to be
separated into its own ufunc-like object as described below.

In principle common DType could be defined only based on "safe casting" rules,
if we order all DTypes and find the first one both can cast to safely.
However, the issue with this approach is that a newly added DType can change
the behaviour of an existing program. For example, a new ``int24`` would be
the first valid common type for ``int16`` and ``uint16``, demoting the currently
defined behaviour of ``int32``.
This API extension could be allowed in the future, while adding it may be
more involved, the current proposal for defining casts is fully opaque in
this regard and thus extensible.


Casting
^^^^^^^

Arguably the most complex and interesting operation which is provided
by DTypes is the ability to cast from one dtype to another.
The casting operation is much like a typical function (universal function) on
arrays converting one input to a new output.
There mainly two distinctions:

1. Casting always requires an explicit output datatype to be given.
2. The NumPy iterator API requires access to lower level functions than
   is necessary for current universal functions. 

Casting from one dtype to another can be complex, and generally a casting
function may not implement all details of each input datatype (such as
non-native byte order or unaligned access).
Thus casting naturally is performed in up to three steps:

1. The input datatype is normalized and prepared for the actual cast.
2. The cast is performed.
3. The cast result, which is in a normalized form, is cast to the requested
   form (non-native byte order).

although often only step 2. is required.


**Motivation:**

Similar to the common dtype/DType operation above, we again have to use cases:

1. ``arr.astype(np.String)`` (current spelling ``arr.astype("S")``)
2. ``arr.astype(np.dtype("S8"))``.

Where the first case is also noted in NEP 40 and 41 as a design goal, since
``np.String`` could also be an abstract DType as mentioned above.

The implementation of casting should also come with as little duplicate
implementation as necessary, i.e. to avoid unnecessary methods on the
DTypes.
Furthermore, it is desirable that casting is implemented similar to universal
functions.

Analogous to the above, the following also need to be defined:

1. ``np.can_cast(dtype, DType, "safe")`` (instance to class)
2. ``np.can_cast(dtype, other_dtype, "safe")`` (casting an instance to another instance)

overloading the meaning of ``dtype`` to mean either class or instance
(on the Python level).
The question of ``np.can_cast(DType, OtherDType, "safe")`` is also possibly
and may be used internally.
However, it is initially not necessary to expose to Python.


**Implementation:**

During DType creation, DTypes will have the ability to pass a list of
``CastingImpl`` objects, which can define casting to and from the DType.
One of these ``CastingImpl`` objects is special because it should define
the cast within the same DType (from one instance to another).
A DType which does not define this, must have only a single implementation
and not be parametric.

DTypes such as a Unit datatype, wrapping an existing Numerical datatype *must*
be able to access these ``CastingImpl``.
The will be able to wrap and modify them as necessary. This means that
a DType which wraps another can automatically define casts for any DType it
knows at construction time (see also Alternatives).

These return a ``CastingImpl`` defined in some more detail in the next section.
It also answers the last question: ``np.can_cast(DType, OtherDType, "safe")``
since ``CastingImpl`` defines a ``CastingImpl.cast_kind = "safe"``.

Each ``CastingImpl`` has a specific DType signature:
``CastingImpl[InputDtype, RequestedDtype]``.
Additionally, it will have one more method::

    adjust_descriptors(self, Tuple[DType] : input, casting="safe") -> Tuple[DType]

(this method is common with the ufunc machineray, see NEP YY).
Here, valid values for ``input`` are:

* ``(input_dtype, None)``
* ``(input_dtype, requested_dtype)``

Where input and requested dtypes must be instances of ``InputDType`` and ``RequestedDtype``.
In the first case, when ``None`` is given, no dtype instance was requested.
The returned values will be a new tuple of two datatypes, filling in ``None``
if necessary.
Note that the output *can* differ from the input.
At the beginning casting was described as a, possibly, three step process.
The ``CastingImpl`` only defines the single step 2.
If the returned datatypes differ from the provided ones, this means that additional
casting steps are required for either input or output.

This problem is solved by introducing one additional slot:

* ``ADType.__within_dtype_castingimpl__ = CastingImpl[ADType, ADType]``

which must be capable of handle any remaining steps, typically byte swapping.
Unlike the first ``CastingImpl``, it is an error if its ``adjust_dtype``
function does not return the input unchanged (except filling in a ``None``).

To provide the actual casting functionality, an additional method:

* ``get_transferfunction(...)``

is necessary to provide a low-level C-implementation.
However, this method shall *not* be part of the public API, instead
users will initially be limited in what casting functions they can provide
(e.g. only contiguous loops of multiple items as is used right now).


**Alternatives:**

The choice of using only the DType classes in the first step of finding the
correct ``CastingImpl`` means that the default implementation of
``__common_dtype__`` has a reasonable definition of "safe casting" between
DTypes classes (although e.g. the concatenate operation using it may still
fail when attempting to find the actual common instance or cast).

The split into multiple steps may seem to add complexity
rather than reduce it, however, it consolidates that we have the two distinct
signatures of ``np.can_cast(dtype, DTypeClass)`` and ``np.can_cast(dtype, other_dtype)``.
Further, the above API guarantees the separation of concerns for user DTypes.
A user ``Int24`` dtype may know how to cast to a ``ArbitraryWidthInteger``,
but does not require any specific knowledge about ``ArbitraryWidthInteger``.
``Int24`` is allowed to cast to *any* ``ArbitraryWidthInteger`` which is knows
to be safe and NumPy will then ask ``ArbitraryWidthInteger`` to cast to the
specifically requested instance.
If ``ArbitraryWidthInteger`` has smaller representations, even ones that ``Int24``
does not know exist,
this provides the information that the cast is unsafe in the followup step.

The main alternative to the proposed design is to move most of the information
which is here pushed into the ``CastingImpl`` directly into methods
on the DTypes. This, however, will not allow the close similarity between casting
and universal functions. On the up side, it reduces the necessary indirection
as noted below.

An initial proposal defined two methods ``__can_cast_to__(self, other)``
to dynamically return ``CastingImpl``.
The advantage of this addition is that it removes the requirement to know all
possible casts at DType creation time (of one of the involved DTypes).
Such API could be added at a later time. It should be noted, however,
that it would be mainly useful for inheritance like logic, which can be
problematic. As an example two different ``Float64WithUnit`` implementations
both could infer that they can unsafely cast between one another when in fact
some conbinations should cast safely or preserve the Unit (both of which the
"base" ``Float64`` would discard).
In the proposed implementation this is not possible, since the two implementations
are not aware of each other.


**Notes:**

The proposed ``CastingImpl`` this designed to be compatible with the
``UFuncImpl`` proposed in NEP YY.
While initially it will be a distinct object, the aim is that ``CastingImpl``
can be a subclass of ``UFuncImpl`` (at least conceptionally).
Once this happens, this will naturally allow the use of a ``CastingImpl`` to
pass around a specialized casting function directly if so wished.

In the future, we may considering adding a way to spell out that specific
casts are known to be *not* possible.

In the above ``CastingImpl`` is described as a python object, in practice the
current plan is to implement it as a C-side structure stored on the ``from``
datatype.
Python side API to get an equivalent ``CastingImpl`` object will be created,
but storing it (similar to the current implementation) on the ``from`` datatype
avoids the creation of cyclic reference counts.

The way dispatching works for ``CastingImpl`` is planned to be limited initially
and fully opaque.
In the future, it may or may not be moved into a special UFunc, or behave
more like a universal function.


C-Side API
^^^^^^^^^^

.. note:: At the time of writing, this API is a general design goal.
          Due to the size of the proposed changes, details and names will be
          in flux and updated as necessary.

DType creation
""""""""""""""

As already mentioned in NEP 41, the interface to define new DataTypes in C
is modelled after the limited API in Python, the above mentioned slots,
and some additional necessary information will thus be passed within a slots
struct and identified by ``ssize_t`` integers::

    static struct PyArrayMethodDef slots[] = {
        {NPY_dt_method, method_implementation},
        ...,
        {0, NULL}
    }

    typedef struct{
      PyTypeObject *typeobj;    /* type of python scalar */
      int is_parametric;        /* Is the dtype parametric? */
      int is_abstract;          /* Is the dtype abstract? */
      int flags                 /* flags (to be discussed) */
      /* NULL terminated CastingImpl; is copied and references are stolen */
      CastingImpl *castingimpls[];
      PyType_Slot *slots;
    } PyArrayDTypeMeta_Spec;

    PyObject* PyArray_InitDTypeMetaFromSpec(
            PyArray_DTypeMeta *user_dtype, PyArrayDTypeMeta_Spec *dtype_spec);

all of this information will be copied during instantiation.

The proposed method slots are (prepended with ``NPY_dt_``), these are
detailed above and given here for summary:

* ``is_native(self) -> {0, 1}``
* ``ensure_native(self) -> dtype``
* ``default_descr(self) -> dtype`` (return must be native and should normally be a singleton)
* ``get_sort_function(self, NPY_SORTKIND sort_kind) -> {out_sortfunction, NotImplemented(?), NULL}``.
  If the sortkind is not understand it may be good to return a special symbol.
* ``setitem(self, char *item_ptr, PyObject *value) -> {-1, 0}``
* ``getitem(self, char *item_ptr, PyObject (base_obj) -> object or NULL``
* ``discover_descr_from_pyobject(cls, PyObject) -> dtype or NULL``
* ``common_dtype(cls, other) -> DType, NotImplemented, or NULL``
* ``common_instance(self, other) -> dtype or NULL``

If not set, most slots are filled with slots which either error or defer automatically.
Non-parametric dtypes do not have to implement:

* ``discover_descr_from_pyobject`` (uses ``default_descr`` instead)
* ``common_instance`` (uses ``default_descr`` instead)
* ``ensure_native`` (uses ``default_descr`` instead)

Which will be correct for most dtypes *which do not store metadata*.

Other slots may be replaced by convenience versions, e.g. sorting methods
can be defined by providing:

* ``compare(self, char *item_ptr1, char *item_ptr2, int *res) -> {-1, 0}``
  *TODO: We would like an error return, is this reasonable? (similar to old
  python compare)*

which uses generic sorting functionality.


CastingImpl
"""""""""""

The external API for ``CastingImpl`` will be limited initially to defining:

* ``cast_kind`` attribute, which can be one of the supported casting kinds.
  This is the safest cast possible. For example casting between two NumPy
  strings is of course "safe" in general, but may be "same kind" in a specific
  instance if the second string is shorter. If neither type is parametric this
  ``adjust_descriptors`` must use it. 
* ``adjust_descriptors(dtypes_in[2], dtypes_out[2], casting) -> int {0, -1}``
* ``strided_loop(char **args, npy_intp *dimensions, npy_intp *strides, dtypes[2]) -> int {0, nonzero}`` (must currently succeed)

This is identical to the proposed API for ufuncs. By default the two dtypes
are passed in as the last argument. On error return (if no error is set) a
generic error will be given.
More optimized loops are in use internally, and made available to users
in the future (see notes)
The iterator API currently does not currently support casting errors, this is
a bug that needs to be fixed. While it is not fixed the loop should always
succeed (return 0).

Although verbose, the API shall mimic the one for creating a new DType.
The ``PyArrayCastingImpl_Spec`` will include a field for ``dtypes`` and
identical to a ``PyArrayUFuncImpl_Spec``::

    typedef struct{
      int needs_api;                 /* whether the cast requires the API */
      PyArray_DTypeMeta *in_dtype;   /* input DType class */
      PyArray_DTypeMeta *out_dtype;  /* output DType class */
      /* NULL terminated slots defining the methods */
      PyType_Slot *slots;
    } PyArrayUFuncImpl_Spec;

The actual creation function ``PyArrayCastingImpl_FromSpec()`` will additionally
require a ``casting`` parameter to define the default (maximum) casting safety.
The internal representation of ufuncs and casting implementations may differ
initially if it makes implementation simpler, but should be kept opaque to
allow future merging.

**Notes:**

We may initially allow users to define only a single loop.
However, internally NumPy optimizes far more, and this should be made
public incrementally, by either allowing to provide multiple versions, such
as:

* contiguous inner loop
* strided inner loop
* scalar inner loop

or more likely through an additional ``get_inner_loop`` function which has
additional information, such as the fixed strides (similar to our internal API).

The above example does not yet include the definition of setup/teardown
functionality, which may overlap with ``get_inner_loop``.
Since these are similar to the UFunc machinery, this should be defined in
detail in NEP 43 and then incorporated identically into casting.

Also the ``needs_api`` decision may actually be moved into a setup function,
and removed or mainly provided as a convenience flag.

The slots/methods used will be prefixed ``NPY_uf_`` for similarity to the ufunc
machinery.



Alternatives
""""""""""""

Aside from name changes, and possible signature tweaks, there seems few
alternatives to the above structure.
Keeping the creation process close the Python limited API has some advantage.
Convenience functions could still be provided to allow creation with less
code.
The central point in the above design is that the enumerated slots design
is extensible and can be changed without breaking binary compatibility.
A downside is the possible need to pass in e.g. integer flags using a void
pointer inside this structure.

A downside of this is that compilers cannot warn about function
pointer incompatibilities, there is currently no proposed solution to this.


Issues
^^^^^^

Any possible design decision will have issues, two of which should be mentioned
here.
The above split into Python objects has the disadvantage that reference cycles
naturally occur, unless ``CastingImpl`` is bound every time it is returned.
Although normally Numpy DTypes are not expected to have a limited lifetime,
this may require some thought.

A second downside is that by splitting up the code into more natural and
logical parts, some exceptions will be less specific.
This should be alleviated almost entirely by exception chaining, although it
is likely that the quality of some error messages will be impacted at least
temporarily.


Implementation
--------------

Internally a few implementation details have to be decided. These will be
fully opaque to the user and can be changed at a later time.

This includes:

* How ``CastingImpl`` lookup, and thus the decision whether a cast is possible,
  is defined. (This is speed relevant, although mainly during a transition
  phase where UFuncs where NEP YY is not yet implemented).
  Thus, it is not very relevant to the NEP. It is only necessary to ensure fast
  lookup during the transition phase for the current builtin Numerical types.


Discussion
----------

There is a large space of possible implementations with many discussions
in various places, as well as initial thoughts and design documents.
These are listed in the discussion of NEP 40 and not repeated here for
brievaty.


Copyright
---------

This document has been placed in the public domain.
