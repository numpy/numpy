=======================================
NEP XX — Extensible Datatypes for NumPy
=======================================

:title: Extensible Datatypes for NumPy
:Author: Sebastian Berg
:Status: Draft
:Type: Standards Track
:Created: 2019-07-17


Abstract
--------


Datatypes in NumPy describe how to interpret each element in the array.
For the most part NumPy provides the usual numerical types, as well as additionally string and some datetime capabilities. 
The growing python community, however, has need for more a more diverse datatypes.
Some example of these are datatypes with unit information attached (such as meters) or categorical datatypes.
However, the current NumPy datatype API is too limited to grow.
This NEP wishes to enable such growth, as well as allow it to be more naturally by providing the ability to define new datatypes from within Python code.
By refactoring our datatype API and improving its maintainability future development will become possible not only for external extensions, but also within NumPy.


The need for a large refactor arise for multiple reasons.
One of the main issue is the definition of typical functions (such as addition, multiplication, …) for so called "flexible" datatypes. Such datatypes – for example the builtin fixed width strings – require additional steps to define for example that adding two strings of length four gives a string of length 8. Similarly, a datatype such as a physical unit, must calculate the new unit information.
A second major issue is that the current casting is limited and behaves differently for user defined datatypes.

Internally, for example the addition of datetimes, which also have a unit, means monolithic code paths were added in many places hardcoding their correct support. While this works well enough within NumPy, it is not possible for external libraries and adds a maintenance burden.
This burden is exacerbated by the exposure of internal structures to outside users, which makes even the addition of new sorting methods difficult or not possible. 

To name more technical aspects of limitations with respect to user defined datatypes or flexible datatypes:

* The definition of casting for flexible user types is either impossible or so complex that it has never been attempted.
* Type promotion, e.g. the operation deciding that adding float and integer values should return a float value is very valuable for datatypes defined within NumPy but is limited in scope for user defined datatypes.
* There is a general issue that most operation where multiple datatypes may interact, are written in a monolithic manner. This works well for the simple numerical types, but does not extend well even for the current strings and datetimes.
* The current design means that for example unit datatype is not able to define a ``.to_si()`` method to easily find the datatype which would cast to SI units.
* Datatypes wrapping existing python types would enable better and simple support for example for variable length strings, or arrays containing arbitrary precision numbers.

The need to solve these issues is apparent in that there are for example multiple projects implementing physical units as an array-like class instead of a datatype, which would be the more natural solution.

To solve these issues we need multiple stages:

* Phase I: Restructure and extend the datatype infrastructure 

  * Creating a new Datatype structure removing the need for monolithic code
  * Exposing a new and easily extensible API to extension authors

* Phase II: Restructure the way universal functions work:

  * Make it possible to allow functions such as ``np.add`` to be extended by user defined datatypes such as Units.
  * Allow efficient lookup for the correct implementation for user defined datatypes.
  * Enable simple definition of "promotion" rules. A Unit datatype should not need to list that it can be multiplied by: int8, int16, …, float16,… separately.

* Phase III: Growth of NumPy capabilities and Ecosystem

  * Cleanup of legacy behaviour in those parts where it is considered buggy.
  * Easy definitions of datatypes in Python
  * Allow community in creating types such as Units or Categoricals
  * Allow strings to be used in functions such as ``np.equal`` or ``np.add``.
  * Removal of legacy code paths within NumPy to improve long term maintainability

This document will focus primarily on Phase I and partially Phase II, since the concerns cannot fully be separated.
While Phase I and II, should in themselves improve the maintainability of NumPy, they are the necessary first steps for any long term improvement and to allow Phase III, which will see the biggest user-facing improvements.
Finally, some of the benefits of a large refactor can only take effect after the full deprecation of legacy implementation. This will take years, however, it should not limit new developments and is mainly a reason to push forward sooner with an easily extensible API.


Motivation and Scope
--------------------

*Probably largely covered by Abstract now, reword...*

NumPy uses its dtype system to describe each element inside its (homogeneous) arrays. While this works well for the numerical types defined by NumPy, it currently requires large monolithic logic in many places, for example to define how to cast a numeric dtype to a string one.
This logic is defined by hard coded paths that are not extensible by user defined dtypes, limiting new dtypes in scope. Furthermore, new dtypes cannot be defined from python, which makes the learning
curve to defining new dtypes harder.

Both things limit the number of user defined dtypes and make the addition of new dtypes within numpy unnecessarily hard. This is clear when examining the current ecosystem, as there are only two publicly available and used dtypes defined outside of numpy: ``rational`` and ``quaternion``.

This is in contrast to a large need for user defined dtypes to represent things such as:

* categorical types (and variations thereof)
* bfloat16, used in deep learning
* physical units (such as meters)
* extending integer dtypes to warn/error on overflow
* extending integer dtypes to have a sentinel NA value

to name just a few examples, some theoretical, some not. While for example units are partially solved in ``astropy.units`` or ``unyt``, these subclass ``ndarray`` which is not ideal for information that is directly associated with the dtype.
The large need of these has already led to the implementation of ExtensionArrays inside pandas [pandas_extension_arrays]_.

This NEP suggests to implement a new infrastructure for the definition of new dtypes. Importantly, that same infrastructure will be used by all builtin dtypes defined within NumPy itself.
It does not suggest the implementation or inclusion of new dtypes inside NumPy itself, nor does it wish to fix inconsistencies of the current (value based) promotion [value_based].

Additionally, the implementation of AbstractDTypes will allow definition of a full type hierarchy. This will allow specifying certain casts withing the hierarchy, such as ensuring or testing that an array is ``Floating`` or ``Numerical``.
The hierarchy can be used for defining promotion rules for universal functions (such as adding a floating point to an integer resulting in a floating point number),
without the need to define the rule for each type combination individually. (This will be discussed in more detailed in a separate [UFunc-NEP]_)

A central part of the NEP is to hide the implementation details to allow changes in the future without breaking API and ABI compatibility.


Overview Graphic
----------------

.. image:: _static/dtype_hierarchy.svg


Detailed description
--------------------

While in some other parts the distinction may not always be necessary, to clarify the discussion of the implementation, we will use the following nomenclature:
 
* DType **class** is the class object, it cannot be attached to an array. Note that there is currently no notion for such objects in NumPy, instead this translates to the ``np.dtype("float64").num``.

* dtype (instance) or dtype (all lower case) is the object which can be attached to an array. (This is consistent with ``array.dtype`` which must be a dtype instance/descriptor.) The ``type(dtype)`` is a ``DType`` which is a subclass of ``np.dtype``. Within the numpy C-API, this is often also called a descriptor, and ``descriptor`` is occasionally used synonymous with dtype instance to clarify variable or method names below (where capitalization may not be clear).

* DTypeMeta is ``type(DType)``. This is a subclass of ``type``, for implementation purposes. This is the ``type(np.dtype)``. Users should not notice this, instead all DTypes subclass ``np.dtype``.

This means that the ``arr.dtype`` attribute is an instance of a specific DType, such as ``Int64``, giving a method resolution order:: 

    type(np.dtype("int64")).mro() == [<type 'numpy.Int64'>, <type 'numpy.dtype'>, <type 'object'>]

This design also means that methods such as ``.to_si()``, to convert a unit to its SI representation, have a clear home on their respective DType class.

To implement full featured DTypes, the following methods need to be implemented within NumPy and by externally written extension DTypes. The methods can then be used in the current API similar to Python's operators (for example by ``np.common_type``).


DType class methods/slots
^^^^^^^^^^^^^^^^^^^^^^^^^

Methods noted in *italic script* are only required by flexible DTypes, methods in **bold script** are the main methods new DTypes should typically implement.

General functions related to coercion:

- ``associated_python_types`` Set of python scalars associated resolved to/by this DType, we can allow adding new types dynamically. (default to just the single ``.type``)
  
  - *Not necessary, at least not initially*
  - Downstream implementers should *not* associate sequence types (at least unless they are defined by the same module). Tricky differences (and changes) for ``np.array(registered_sequence)`` could occur. First, due to change upon module import. Second, due to behaviour change when a specific DType (class) is passed in. Users will have to always pass in such a type.

*comments*::

   > [name=mattip] needs a justification: when would you want to use more than one?
   > [name=seberg] Actually, you are right. I wanted this for value based casting, but I do not need to map np.int64 -> integer, that is legacy behaviour we want to deprecate and need to add in a second step.
   > [name=TomAugspurger] Is the question about why you would need more than one python type? https://github.com/ContinuumIO/cyberpandas/blob/0db23fcdf91eede631ed603cb321f39d834bd12c/cyberpandas/ip_array.py#L22-L50 has an example. That's a pandas ExtensionDtype for IP addresses. But the workaround there (a metaclass that registers the Python types) isn't too bad.

* **``__dtype_getitem__(self, item_pointer)``**: Convert to python object.
  
  * NumPy currently has two modes to convert to scalar, ``.item()/.tolist()`` and ``[]`` indexing. The first coerces to python scalars, the second creates a 0d ndarray. This probably means the addition of ``__dtype_getitem_numpy_scalar__`` which defaults to ``__dtype_getitem__``, typical DTypes should not use the distinction.

  > [name=shoyer] to clarify, this converts a Python object into an array element? It would also be nice include arguments in this list of methods...

* **``__dtype_setitem__(self, item_pointer) -> PyObject``**: Coerce from python object.

* *``__discover_descr_from_pyobject__(cls, obj) -> descr``*: Find e.g. the string size (or unit), unnecessary if there is a clear default descriptor/instance.

* ``__discover_dtype_from_pytype__(cls, typeobj) -> DType``: Given a python type, which DType is used to represent it in a numpy array?
  
  - **TODO: Probably just remove for now** (we do need something internally to discover from python _object_, but from python _type_ has probably no use case)
  - *Currently we use ``isinstance(obj, float)`` to map to "float64", I believe this should be considered legacy behaviour and deprecated. Normally types should match exactly, the float could be a float with a unit attached...*
  - Required for value based casting (although this could be limited to internal NumPy usage and special cases)
  - May be necessary for ``np.array([1, 2, 3, 4], dtype=AbstractUserDType)``, allowing the AbstractUserDType to take more control of how to interpret the data.
  - For value based casting, this function actually requires the object (not just its type) as input, and the result would not be cacheable.
  - (May be necessary to pick up a DType based on ABCs or other rules).
  - Note, that ``np.asarray([object1, object2])`` does not necessarily need this, since this is already solved by direct type association.

  *comments*::
  
    > [name=shoyer] how would ``__discover_dtype_from_pytype__`` work if it's only defined on DType classes? It seems like by the time you can find this method, you already know the required DType.
    > [name=seberg] Should probably hide it away or even get rid off for now. I think I thought we may need the extra power, but I cannot think of any use case (although initially, I thought we may want to match subtypes, etc.). One reason would be ``np.array(..., dtype=OddDType)``, but that only seems necessary if OddDType takes full control and also indicates whether the object is a sequence. Or that just dispatches to ``OddDType.__coerce_to_array__``

Related to casting and promotion:

* **``__can_cast_from_other__(cls, other, casting="safe") -> CastingImpl``**: Defines casting, the function answers whether a cast from another DType (class) or descriptor instance to this one and returns a ``CastingImpl`` describing that cast.
  
  - This is a classmethod, it only answers whether casting between the two is possible in principle.
  - If the question is whether specific descriptors/instances are castable (strings of specific lengths or specified units) whether or not the casting is possible needs more careful value-based checking. It can be answered directly only for non-flexible dtypes.
  - Returns NotImplemented if the DType does not know how to perform the cast, (may signal an error if it is known to be impossible).

* **``__can_cast_to_other__(cls, other, casting="safe") -> CastingImpl``**: Reversed casting lookup.

* **``__common_dtype__(cls, other) -> DType``**: Operator returning a (new) DType class capable of describing both inputs.
  
  - A default implementation will be provided based on "safe" casting for backward compatibility, however, the fallback may be deprecated.
  - Within numpy this usually aligns to "safe" casting rules.

* *``__common_instance__(descr1, descr2) -> descr``*: For a flexible DType, returns the common instance. For example the maximum string length for two strings.

* **``default_descr(cls) -> descr``**: The default DType representation (see ``__ensure_native__`` below). Should return an immutable singleton.

* *``__ensure_native__(self) -> descr``*: A bound method of flexible dtype instances to ensure native byte order (or any other non-canonical representation). Within NumPy this is maps to ">U8" (unicode strings with non-native byte order). Non-flexible dtypes should always return ``default_descr()``, in which case they do not need to define it.



AbstractDType class methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users should typically not need to define AbstractDTypes, unless purely for type hierarchy purposes. However, when they are defined, they can have specific slots:

* ``default_dtype(cls) -> DType``: Used to answer the question that a python integer should normally be translated to a ``long`` (or in the future hopefully `intp`). (Maybe normal DTypes should have it, but would just return the class itself).

* ``minimal_dtype(cls) -> DType``: In UFunc type resolution, we use the minimal type for value based casting.

These are necessary when, after a ``common_dtype`` operation results in an AbstractDType which needs to be converted to a concrete one. Strictly speaking, usually, this should only happen for value based casting (implementing these slots may thus be limited to within NumPy initially).

The following methods are available to ``AbstractDTypes``: ``__can_cast_from_other__``, ``__discover_descr_from_pyobject__``, ``__discover_dtype_from_pytype__``, ``_associated_python_types``. This allows, for example, casting to `Floating`. Which can use``float64`` for most/all non-float input but leave the type unchanged for input that is already ``Floating`` (such as ``float128`` or ``float32``).


Internally (not exposed to public API) we also require a ``_value_based_casting`` flag, since ``__discover_dtype_from_pytype__`` requires the value for python integer and floats (and right now also our own scalars and 0D arrays). Since this is private, the implementation could be changed later.
The fact that 0D arrays and numpy scalars also use value based casting will be hardcoded, since it is considered legacy behaviour.


Metadata describing the DataType *instance*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We should allow setting these on the class, for the common case where they are fixed. Old style user dtype instances actually support byteswapping:

* itemsize: (Dtype class may define it, but does not need to be)

* type: The scalar type associated with the dtype. 

* flexible:

  * Many dtypes are not flexible, i.e. they have a canonical
    representation and casting from/to it is always safe.
  * DTypes which are not flexible, must have a ``default_descr`` for
    that canonical implementation which should be a singleton.
  * Flexible dtypes must define some additional attributes and
    additional slots on ``CastingImpl``.

* byteorder (not strictly necessary)

* is_native

  * Instead of byteorder, we may want an ``is_native`` flag (we could just reuse the ISNBO flag – "is native byte order"), this flag signals that the data is stored in the default/canonical way. In practice this is always an NBO check, but generalization should be possible. A use case would be to have a complex-conjugated instance of Complex which is not the normal representation and typically has to be cast).

* alignment information for structured dtype construction. (May be possible to guess if the itemsize is fixed, at least for some dtypes, but maybe better to force to be provided).

* ...

**Methods currently defined by ``ArrFuncs``** (due to the visibility of this struct, read access to it will remain supported at least for old style dtypes for some time; It may be useful to detect and error on changes of the struct).

 * sorting, argsorting
 * take
 * byteswap (``copyswapn``)
 * etc.

Where some of the last slots should be redefined as (generalized) ufuncs
and deprecated.
Many of these functions get an array argument, which already is sometimes just a dummy object with the dtype attached. This practice has to remain. The hope is to "move" these by allowing the registration e.g. with ``NPY_dt_legacy_take`` in the new API. Deprecation and changes can
then happen in a second step.

C level implementation
^^^^^^^^^^^^^^^^^^^^^^

The C-level implementation of dtypes should largely follow similar designs
as CPython. Unlike CPython, however, the layout shall be completely hidden
from the downstream user, with most methods not accessible (initially).

Thus, to create a new data type from the C-level, it will be necessary to
use a code similar to this (modeled after [PEP-384]):

.. code-block:: C

    static struct PyArrayDtypeMethodDef slots[] = {
        {NPY_dt_cast_from_to, cast_from_to},
        {NPY_dt_cast_to_from, cast_to_from},
        {NPY_dt_, }
        ...,
        {NPY_dt_type, &ScalarTypeObject}
        ...,
        {NPY_dt_take, take},
        ...,
        {0, NULL}
    }

    typedef struct{
      int flexible;  /* Could be part of flags */
      int abstract;  /* Could be part of flags */
      npy_intp itemsize;  /* May be -1 if size not fixed */
      int flags;  /* Similar to current flags */
      PyTypeObject *typeobj;  /* type of python scalar */
      PyType_Slot *slots; /* terminated by slot==0. */
    } PyArrayDTypeMeta_Spec;

     
     /*
      * Note that PyArray_DTypeMeta is a python type object and previously
      * defined as a subclass of PyArrayDescr_Type.
      * Users may extend it. The function below initializes the DTypeMeta class.
      * The legacy registration will create a new DTypeMeta intance
      * (dtype subclass) and initalize it based on the existing information.
      */
    PyObject* PyArray_InitDTypeMetaFromSpec(
            PyArray_DTypeMeta *user_dtype, PyArrayDTypeMeta_Spec *dtype_spec);



Casting Implementation
^^^^^^^^^^^^^^^^^^^^^^


Current Implementation of Casting
"""""""""""""""""""""""""""""""""

One of the main features which datatypes need to support is casting between one another using ``arr.astype(new_dtype, casting="unsafe")``, or while executing ufuncs with different types (such as adding integer and floating point numbers). Currently casting is defined in multiple ways:

1. ``copyswap``/``copyswapn`` are defined for each dtype and can handle byte-swapping for non-native byte orders as well as unaligned memory.
2. ``castfuncs`` is filled on the ``from`` dtype and casts aligned and contiguous memory from one dtype to another (both in native byte order). Casting to builtin dtypes is normally in a C-vector. Casting to a user defined type is stored in an additional dictionary.

When casting (small) buffers will be used when necessary, using the first ``copyswapn`` to ensure that the second ``castfunc`` can handle the data. A general call will thus have ``input -> in_copyswapn -> castfunc -> out_copyswapn ->output``.

However, while user types use only these definitions, almost all actual casting uses a monolithic code which may or may not combine the above functions, supports strided memory layout and specialized implementations.


Proposed Casting API
""""""""""""""""""""

The above section lists two steps of casting that are currently used (at least for user defined dtypes). That is, casting within the same DType using ``copyswapn`` for unaligned or byte swapped data, as well as casting between two dtypes using defined ``castfuncs``.

Within NumPy, there are many specialized functions. There is thus a need to find a way to make many implementations available. The proposal here is to create a new ``CastingImpl`` object as a home for the specific ``castfunc`` or ``copyswapn`` functions. That is a ``CastingImpl`` is defined for two (or one) destinct DType ``CastingImpl[FromDType->ToDType]`` (where from and to DType can be identical). In practice the ``CastingImpl`` will be attached to one of the two DTypes, and returned by its corresponding slots (it may be automatically generated for the special case where ``FromDType`` and ``ToDType`` are identical).

This design is chosen for multiple reasons:

1. By using a ``CastingImpl`` Python object with largely hidden API, it is possible to start with providing only a few additional capabilities to extension dtype authors to begin with, while having full flexibility to allow performance relevant additions later.
2. ``CastingImpl`` should be a special subclass of ``UFuncImpl``. While it needs to expose (initially internally to NumPy) additional functionality to enable casting, this will allow to do ``cast_int_to_string = np.get_casting_implementation()`` and use ``cast_int_to_string`` as a (specialized) UFunc. (Note that it is not required that ``CastingImpl`` is a ``UFuncImpl`` subclass to begin with. In a first implementation ``CastingImpl`` may be a very limited object only useful internally as a home for user provided functionality.)
3. Casting between flexible DTypes (such as string lengths) requires to adjust the flexible dtype instances (casting a ``float32`` to string results in a different string length than a ``float64``). This is similar to the same need in UFuncs and ``CastingImpl`` can provide this functionality.


``CastingImpl`` Definition and Usage
""""""""""""""""""""""""""""""""""""

With the exception of casting to and from objects, which can be defined using the above slots, and casting within the same dtype, which can (for non-flexible dtypes) be defined by ``copyswap(n)``, casting functions have to be defined specifically.
The above slots only define the resolution step to find the correct ``CastingImpl``.

The CastingImpl (similar to a UfuncImpl) will have to define:

* *``adjust_descriptors``* or a similar method, which takes the input dtype instance (and possibly given output one) and define the output descr.

  * For a ``CastingImpl`` defining casting within a dtype, these *must* match with the actual input ones.
  * A CastingImpl which does not handle any flexible dtypes does not need to define ``adjust_descriptors``
  * The signature will be ``adjust_descritpors((input_dtype, out_dtype or None), casting="safe")`` with the function returning a new, modified set of ``dtypes`` (all instances) or raising a ``TypeError``.

* An (initially) internal slot: ``get_casting_function()`` with a signature identical (or similar) to

  .. code-block:: C

     PyArray_GetDTypeTransferFunction(
                    int aligned,
                    npy_intp src_stride, npy_intp dst_stride,
                    PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                    int move_references,
                    PyArray_StridedUnaryOp **out_stransfer,
                    NpyAuxData **out_transferdata,
                    int *out_needs_api)

  (although the handling of NpyAuxData needs to be discussed, i.e. it could be replaced with setup/teardown funtions on the ``UfuncImpl``/``CastingImpl``).

  * The strides are ``MAX_INTP`` if they are not fixed, so an optimal function can be chosen.
  * It might make sense to pass a buffersize?
  * Currently source or destination can be NULL (relevant for objects), object handling could be tricky, and we may think about limiting them? Also related to the ``move_references`` flag.
  * (This has overlap or identical to enabling some ufunc chaining) 

Users will initially only be able to define ``CastingImpl`` in a very limited manner, the minimal API for allowing all past features is:

* Automatically wrapping of ``copyswap``/``copyswapn`` for simplicity and existing dtypes.

* From a contiguous casting function and additional strided input, passing in the dtype instances. 

However, this should be expanded, and we may simply allow downstream users to override the ``get_casting_function`` slot in the future.


Python level interface
^^^^^^^^^^^^^^^^^^^^^^

To expose these slots to python, automatic wrappers shall be created for
the slots if they are defined as python functions (under specific names).
Some slots maybe exposed as capsule objects to:

* Allow reusing existing slots from other dtypes while avoiding slowdowns
* Replacing slots with faster C versions (including by just in time compilers)

The Python interface may be a second step, and may be limited in many cases.
However, it is a specific design goal to allow the definition e.g. of Unit dtypes
in Python. This will also require similar python wrapping for ``CastingImpl`` and
``UFuncImpl`` (and resolvers) in general.

The exact API should be an additional NEP. A likely design goal will be to allow defining new dtypes using:

.. code-block:: python

    @np.dtype
    class Coordinate:
       x: np.float64
       y: np.float64
       z: np.float64


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

This additional section details some related changes, which are only partially tied to the general refactor.

**Stricter array rules for dtype discovery**

When coercing arrays with ``np.array`` and related functions, numpy currently uses ``isinstance(pyobj, float)`` logic (user types do not have this ability, they can only automatically be discovered from numpy scalars). In general, user dtypes should be capable of ensuring that specific input is coerced correctly.
However, in general these should be exact types and not ``isinstance`` checks. A python float subclass, could have a completely different meaning and should generally viewed as a ``"float64"`` dtype. Instead, the current ``isinstance`` checks should become a fallback discovery mechanisms and *be deprecated*.



Related Work
------------

**TODO:** This section should list relevant and/or similar technologies, possibly in other
libraries. It does not need to be comprehensive, just list the major examples of
prior and relevant art.

* Julia has similar split of abstract and concrete types [julia-types]_. 

* In Julia promotion can occur based on abstract types. If a promoter is
  defined, it will be called and then retry the resolution [julia-promotion]_.

* ``xnd-project`` https://github.com/xnd-project) with ndtypes and gumath

  *  Different in that it does not use promotion at all.


Implementation
--------------

First the definition and instantiation of new style data types need to be defined and the current ones replaced/wrapped. (Hopefully this is a bit tricky but fairly straight forward with wrappers)

The main difficult is the casting logic has to be fully reimplemented, since the old code will often not be directly usable (although in some cases it might be a valid). Wrapping all the dtype transfer functions into ``CastingImpl`` is hopefully not too difficult, since it should be possible to simply fall back to the current code. It would be good to incrementally replace it later.

While strong performance issues are not anticipated, in some cases performance regressions may have to be addressed. Finding out whether a cast is possible may become significantly slower in some cases, however, with updates of the UFunc machinery, it should also be used much less. Value based casting is likely to become somewhat slower, which may impact scalar performance in some cases. In most such cases fast paths can be added for builtin dtypes.


**TODO:** This section lists the major steps required to implement the NEP.  Where
possible, it should be noted where one step is dependent on another, and which
steps may be optionally omitted.  Where it makes sense, each step should
include a link to related pull requests as the implementation progresses.

Any pull requests or development branches containing work on this NEP should
be linked to from here.  (A NEP does not need to be implemented in a single
pull request if it makes sense to implement it in discrete phases).


Backward compatibility
----------------------

The following changes will be backward incompatible:

* ``PyArray_DescrCheck`` currently tests explicitly for being an instance of PyArray_Descr. The Macro is thus not backward compatible (it cannot work in new NumPy versions). This Macro is not used often, for example not even SciPy uses it. This will require an ABI breakage, to mitigate this new versions of legacy numpy (e.g. 1.14.x, etc.) will be released to include a macro that is compatible with newer NumPy versions. Thus, downstream will may be forced to recompile, but can do so with a single (old) NumPy version.

* The array that is currently provided to some functions (such as cast functions), may not be provided anymore generally (unless easily available). For compatibility, a dummy array with the dtype information will be given instead. At least in some code paths, this is already the case.

* The ``scalarkind`` slot and registration of scalar casting will be removed/ignored without replacement (it currently allows partial value based. The ``PyArray_ScalarKind`` function will continue to work for builtin types, but will not be used internally and be deprecated.

* The type of any dtype instance will not be ``dtype`` anymore, instead, it it will be a subclass of DType.

* Current user dtypes are specifically defined as instances of ``np.dtype``, the instance used when registered is typically not held on to, but at the very least its type and base would have to be exchanged/modified. This may mean that the user created Descriptor struct/object is only partially usable (it does not need to be used though, and is not for either ``rational`` or ``quaternion``)


Existing ``PyArray_Descr`` slots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although not extensively used outside of NumPy itself, the currently defined slots of ``PyArray_Descr`` are public. This is especially true for the ``ArrFuncs`` stored in the ``f`` field, which are also public. 
Due to compatibility need remain supported for a very long time, with the possibility of replacing them by functions that dispatch to a newer API.

NumPy should get a macro ``NPY_ALLOW_DESCRIPTOR_ACCESS`` which will have to be defined to access the structs directly. In a first version, the macro may just print a compile time warning.
The macro will allow access to the struct, however, this shall only be guaranteed to work for a transition period of four to five years. After this time, downstream projects should not be expected to compile on an old version still requiring the old API. At this point the next major release of NumPy can drop support.

A similar, timeline shall also apply to the use of old style dtype registration functions.


Alternatives
------------

* Instead of the casting as a slot approach, casting could be handled more like a UFunc with resolver registration. The slot approach means that two function calls are sometimes necessary, however, is in general the simpler lookup for this very special function. It has the small advantage that third parties cannot easily change the casting logic of other types.

* While this NEP does not forbid future expansion to allow subclassing of DTypes (other than ``np.dtype`` itself), this seems not desirable. Subclassing can be very confusing with respect to casting and UFunc dispatching being inherited. Instead subclassing of ``AbstractDTtypes`` is specifically allowed thus separating many of the issue of inheritance from the use case of defining a type hierarchy.
  
  * TODO: ``AbstractDTypes`` can define certain slots, we have to decide how/if they are inherited! Most likely, it should be required to override/unset them (that also allows us to have flexibility down the road).

* It would be possible to limit the subclassing capabilities (at least within C) to allow even more flexibility. This seems unnecessary, since we can always allocate a storage area within our private ``DTypeMeta`` instance struct. This way users can use the full Python API to create their own (d)types in C and add methods such as ``.to_si()`` for units.

* TODO: Decide whether we need DTypeMeta(Type) and DTypeMeta(HeapType)? My guess is that there is just not much of a point in making things even more confusing.



Rejected Alternatives/Different Approaches
------------------------------------------


Instances of dtypes should be scalars
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When considering the numerical NumPy types such as ``float32``, ``float64``, etc. It is tempting to suggest that the scalars should be instances of the dtype. In this way a ``Float64`` class would both have all information of a DType, while ``Float64(1.)`` is its instances. (As opposed to currently ``np.dtype("float64").type(1.)`` being an instance of a scalar with a ``.dtype`` attribute).

This has been rejected for the following reasons:

1. Most importantly, for existing Python types, it is not viable to store additional information (as required by NumPy as use with an array) on the type itself. For example, it should be possible to write a DType for ``decimal.Decimal``. However, for such a ``DecimalDType`` it is impossible that its scalars are also instances (they already are instances of ``decimal.Decimal``).

2. Scalars currently do not have, and likely do not require, information such as non-native byte order, making both the types and the instances more complex than necessary.

3. While a beautiful concept, unless practically all types have such additional DType capabilities (i.e. this was a language feature of Python), the practical advantage seems to be small.

In short, this idea seems impractical while the actual enhancements offered seem unclear.


Keep the current layout of all NumPy dtypes being direct instances of ``np.dtype``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This would remove the ``DType`` class object introduced above. Such a class would still be used for user defined DTypes. The current layout still largely supports this as a (very) flexible ``DType`` class, but is not meant to be used this way. In this sense the ``DType`` class forms more a wide category of different ``dtypes``. 
The main advantage of that approach is the attempt to leave the current NumPy machinery as much intact as possible.

This has been rejected for the following reasons:

* While more invasive, the suggestion in this NEP should not be harder to implement (although possibly more work).

* A big category spanning floating point numbers and integers, means that casting still would have to rely on additional type numbers to see that casting to an ``>int64`` is done by first casting to ``<int64``.

* It seems not straight forward to define an AbstractDType hierarchy as above, with its additional features.



Open Issues
-----------

``np.load`` (and others) currently translate all extension dtypes to void dtypes. This means they cannot be stored using the ``npy`` format. Similar issues exist with the buffer interface.

In some cases, the only option would be to raise an error instead of silently converting the data (which probably makes sense). For saving arrays we may have to force pickling right now, although we could store known dtypes and force users to simply import that library first?



Discussion
----------

**TODO:** This section may just be a bullet list including links to any discussions regarding the NEP. HackMD links may not be the best...

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

.. _UFunc-NEP: https://hackmd.io/y7ghitUtRQaMyaHFGe-ueQ (Update link or refer to NEP number)

.. _value_based: Value based promotion denotes the behaviour that NumPy will inspect the value of scalars (and 0 dimensional arrays) to decide what the output dtype should be. ``np.array(1)`` typically gives an "int64" array, but ``np.array([1], dtype="int8") + 1`` will retain the "int8" of the first array.

.. _safe_casting: Safe casting denotes the concept that the value held by one dtype can be represented by another one without loss/change of information. Within current NumPy there are two slightly different usages. First, casting to string is considered safe, although it is not safe from a type perspective (it is safe in the sense that it cannot fail); this behaviour should be considered legacy. Second, int64 is considered to cast safely to float64 even though float64 cannot represent all int64 values correctly.

.. _flexible_dtype: A flexible dtype is a dtype for which conversion is not always safely possible. This is for example the case for current string dtypes, which can have different lengths. It is also true for datetime64 due to its attached unit. A non-flexible dtype should typically have a canonical representation (i.e. a float64 may be in non-native byteorder, but the default is native byte order).

.. _julia-types: https://docs.julialang.org/en/v1/manual/types/index.html#Abstract-Types-1

.. _julia-promotion: https://docs.julialang.org/en/v1/manual/conversion-and-promotion/

.. _PEP-384: https://www.python.org/dev/peps/pep-0384/


Copyright
---------

This document has been placed in the public domain.
