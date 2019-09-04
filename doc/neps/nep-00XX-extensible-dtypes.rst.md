---
title: Extensible Datatypes for NumPy
---

[todo/meeting notes](https://hackmd.io/5aV5K49pT8GtxBcMgC7PYQ)


NEP XX — Extensible Datatypes for NumPy
=======================================

:Author: Sebastian Berg <sebastian@sipsolutions.net>
:Author: ...
:Status: Draft
:Type: Standards Track
:Created: 2019-07-17



Abstract
--------

The interpretation of each element inside a NumPy array is described by a datatype (dtype/DType) object which is attached to the numpy array.It is currently possible to define new dtypes via the C-API but user-defined dtypes lack full featured casting/conversion between data types.
This NEP suggests a full redesign of the dtypes.
This redesign will promote user-defined dtypes to behave just as built in dtypes. The goal is that user-defined dtypes can be created via the Python API, can be based off existing dtypes, and all the logic like casting will Just Work with the new dtypes. In the first iteration, this will most likely be implemeted by basing a new DType on existing functions. In the future low level control is desired. While the Python API is an important goal, it is also a second step and will be defined in detail in another NEP.

*To achieve the design of having new DType classes, it is necessary to at least break ABI compatibility of ``PyArray_DescrCheck``. See below for mitigations/notes.*

TODO: Add that this can support bitpattern masks, but not bitarrays stored/attached to the array.

Motivation and Scope
--------------------

NumPy uses its dtype system to describe each element inside its (homogeneous) arrays. While this works well for the numerical types defined by NumPy, it currently requires large monolithic logic in many places, for example to define how to cast a numeric dtype to a string one.
This logic is defined by hard coded paths that are not extensible by user defined dtypes, limiting new dtypes in scope. Furthermore, new dtypes cannot be defined from python, which makes the learning
curve to defining new dtypes harder.

Both things limit the number of user defined dtypes and make the addition of new dtypes within numpy unnecessarily hard. This is clear when examining the current ecosystem, as there are only two publicly available and occasionally used dtypes defined outside of numpy: ``rational`` and ``quaternion``.

This is in contrast to a large need for user defined dtypes to represent things such as:
  * categorical types (and variations thereof)
  * bfloat16, used in deep learning
  * physical units (such as meters)
  * extending integer dtypes to warn/error on overflow
  * extending integer dtypes to have a sentinal NAN value

to name just a few examples, some theoretical, some not. While for example units are partially solved in ``astropy.units`` or ``unyt``, these subclass `ndarray` which seems not ideal for information which is directly associated with the dtype.
The large need of these has already led to the implementation of ExtensionArrays inside pandas [1].

This NEP suggests to implement a new infrastructure for the definition of new dtypes. Importantly, that same infrastructure will be used by all builtin dtypes defined within NumPy itself.
It does not suggest the implementation or inclusion of new dtypes inside NumPy itself, nor does it wish to fix inconsistencies of the current (value based) promotion.

Additionally, the implementation of ``AbstractDTypes`` will allow definition of a full type hierarchy. This will allow specifying certain casts withing the heirarchy, such as ensuring or testing that an array is ``Floating`` or ``Numerical``.
The hierarchy can be used for defining promotion rules for universal functions (such as adding a floating point to an integer resulting in a floating point number),
<!---  > [name=mattip] huh? the result is boolean? --->
without the need to define the rule for each type combination individually. (This is discussed in more detailed in a separate [NEP](https://hackmd.io/y7ghitUtRQaMyaHFGe-ueQ)) 
> [name=mattip] so the two NEPS should be one PR?
> [name=seberg] Thought so first, but as long as we limit the ``CastingImpl`` enough, that is not necessary. The type hierarchy allows for multiple dispatching inside UFuncs (how much of that we actually use or support is another point)

An important part of the NEP is to hide the implementation details to allow changes in the future without breaking API and ABI compatibility.


Overview Graphic
----------------

![overview_hierarchy_and_type_definitions](https://i.imgur.com/cu8xMBA.png)


Detailed description
--------------------
> [name=Matti Picus] Perhaps DType should refer to the class, and dtype to the instance?
> [name=seberg] Yeah, will look into changing that, descriptor is nice, but really just is the name in the C-API right now, and also just confusing as well probably.


While in some other parts the distinction may not always be clear, to clarify the discussion of the implementation, we will use the following nomenclature:
 
* DType **class** is the class object, it cannot be attached to an array. Note that there is currently no notion for such objects in NumPy, instead this translates to the ``dtype.num``.

* descriptor may be used synomymous as dtype (instance) or dtype (all lower case). Only a descriptor can be attached to an array object. (This is consistent with ``array.dtype`` which must be a dtype instance/descriptor.) The `type(dtype)` is a ``DType`` subclass.

* DTypeMeta is ``type(DType)``. This is a subclass of `type`, for implementation purposes.
> [name=mattip] Confusing. Do you mean `type(descriptor)` which should be a DType or `type(Dtype)` which IMO should be `class` 
> [name=seberg] ``type(DType)``, this is needed if we do not want to  put all our information into a python side classslot (ABC) does this. I think it might be needed in any case, just to define the MetaClass correctly (e.g. to have control on python side subclassing attempts).

This means that the `arr.dtype` attribute is an instance of a specific DType, such as ``Int64``, giving a method resolution order: ``type(np.dtype("int64")).mro() == [<type 'numpy.Int64'>, <type 'numpy.dtype'>, <type 'object'>]``.

To implement full featured DTypes, the following methods need to be implemented within numpy and by externally written extension DTypes. The methods can then be used in the current API similar to Python's operators (for example by `np.common_type`).


### DType class methods

Methods noted in *italic script* are only required by flexible DTypes, methods in **bold script** are the main methods new DTypes should typically implement.

General functions related to coercion:
  * **``_associated_python_types``**: Set of python scalars associated resolved to/by this DType, we can allow adding new types dynamically. (default to just the single `.type`)
      - More than one (or none) is occasionally necessary. Most of the numpy scalars need to map to an AbstractDType for handling value based casting (i.e. all numpy integer scalars and python integers). Also numpy and python strings (for the moment) map to the String dtype.
  > [name=mattip] needs a justification: when would you want to use more than one?
  > [name=seberg] Hmmm, could be done with a single one possibly, have to think about it (added an argument above)
  * **``__dtype_getitem__``** Convert to python object.
      * NumPy currently has two modes to convert to scalar, `.item()/.tolist()` and `[]` indexing. The first coerces to python scalars, the second creates a 0d ndarray. This probably means the addition of ``__dtype_getitem_numpy_scalar__`` which defaults to ``__dtype_getitem__``, typical DTypes should not use the distinction.
  * **``__dtype_setitem__``** Coerce from python object.
* *``__discover_descr_from_pyobject__``*: Find e.g. the string size (or unit), unnecessary if there is a clear default descriptor/instance.
  * ``__discover_dtype_from_pytype__``: Given a python type, which DType is used to represent it in a numpy array?
    - Required for value based casting (although this could be limited to internal NumPy usage and special cases)
    - May be necessary for ``np.array([1, 2, 3, 4], dtype=AbstractUserDType)``
    - For value based casting, this function actually requires the object (not just its type) as input, and the result would not be cacheable.
    - (May be necessary to pick up a DType based on ABCs or other rules).
    - Note, that ``np.asarray([object1, object2])`` does not necessarily need this, since this is already solved by direct type association.


Related to casting and promotion:
  * **``__can_cast_from_other__``**: Defines casting, the function answers whether a cast from another DType (class) or descriptor instance to this one and returns a `CastingImpl` describing that cast.
      - This is a classmethod, it only answers whether casting between the two is possible in principle.
      - If the question is whether specific descriptors/instances are castable (strings of specific lengths or specified units) whether or not the casting is possible needs more careful value-based checking. It can be answered directly only for non-flexible dtypes .
      - Returns NotImplemented if the DType does not know how to perform the cast, (may signal an error if it is known to be impossible).
  * **``__can_cast_to_other__``**: Reversed casting lookup.
  * **``__common_dtype__``**: Operator returning a (new) DType class capable of describing both inputs.
    - A default implementation will be provided based on "safe" casting for backward compatibility, however, the fallback may be deprecated.
    - Within numpy this usually aligns to "safe" casting rules.
  * *``__common_instance__``*: For a flexible DType, returns the common instance. For example the maximum string length for two strings.
  * **``default_descr()``**: The default DType representation (see ``__ensure_native__`` below). Should return an immutable singleton.
  * *``__ensure_native__``*: A bound method of flexible dtype instances to ensure native byte order (or any other non-canonical representation). Within NumPy this is maps to ">U8" (unicode strings with non-native byte order). Non-flexible dtypes should always return ``default_descr()``, in which case they do not need to define it.



### AbstractDType class methods

Users should typically not need to define AbstractDTypes, unless purely for type hierarchy purposes. However, when they are defined, they can have specific slots:
  * `default_dtype`: Used to answer the question that a python integer should normally be translated to a `long` (or in the future hopefully `intp`). (Maybe normal DTypes should have it, but would just return the class itself).
  * `minimal_dtype`: In UFunc type resolution, we use the minimal type for value based casting.

These are necessary when, after a ``common_dtype`` operation results in an AbstractDType which needs to be converted to a concrete one. Strictly speaking, usually, this should only happen for value based casting (implementing these slots may thus be limited to within NumPy initially).

The following methods are available to `AbstractDType`s: `__can_cast_from_other__`, `__discover_descr_from_pyobject__`, `__discover_dtype_from_pytype__`, `_associated_python_types`. This allows, for example, casting to `Floating`, returning ``float64`` but using ``float128`` if necessary (or ``float32`` if already ``float32``).

> [name=mattip] Not clear to me. Maybe "casting a non-float will prefer `float64` but casting a float to Floating will return `type(self)`"

Internally (not exposed to public API) we also require a `_value_based_casting` flag, since `__discover_dtype_from_pytype__` requires the value for python integer and floats (and right now also our own scalars and 0D arrays). Since this is private, the implementation could be changed later. The fact that 0D arrays and numpy scalars also use value based casting will be hardcoded, since it is considered legacy behaviour.

> [name=mattip] Stopped reviewing here

### Metadata describing the DataType *instance*

We should allow setting these on the class, for the common case where they are fixed. Old style user descriptors actually support byteswapping:
   * itemsize: (Dtype class may define it, but does not need to be)
   * type: The scalar type associated with the dtype. 
   * flexible:
       * Many dtypes are not flexible, i.e. they have a canonical
         representation and casting from/to it is always safe.
       * DTypes which are not flexible, must have a `default_descr` for
         that canonical implementation which should be a singleton.
       * Flexible dtypes must define some additional attributes and
         additional slots on `CastingImpl`.
   * byteorder (not strictly necessary)
   * is_native
       * Instead of byteorder, we may want an `is_native` flag (we could just reuse the ISNBO flag – "is native byte order"), this flag signals that the data is stored in the default/canonical way. In practice this is always an NBO check, but generalization should be possible. A use case would be to have a complex-conjugated instance of Complex which is not the normal representation and typically has to be cast).
   * alignment information for structured dtype construction. (May be possible to guess if the itemsize is fixed, at least for some dtypes, but maybe better to force to be provided).
   * ...

**Methods currently defined by ``ArrFuncs``** (due to the opaque nature of this struct, read access to it will remain supported at least for builtin dtypes for some time; It may be useful to detect and error on changes of the struct).
   * sorting, argsorting
   * take
   * byteswap (``copyswapn``)
   * etc.

Where some of the last slots should be redefined as (generalized) ufuncs
and deprecated.
Note that the metadata described in point 2 describes a specfic instance,
while methods/slots live on the dtype class.

### C level implementation

The C-level implementation of dtypes should largely follow similar designs
as CPython. Unlike CPython, however, the layout shall be completely hidden
from the downstream user, with most methods not accessable (initially).

Thus, to create a new data type from the C-level, it will be necessary to
use a code similar to this (modelled after [PEP-384]):

```C
static struct PyArrayDtypeMethodDef abstract_methods[] = {
    {npy_adt_cast_from_to, cast_from_to},
    {npy_adt_cast_to_from, cast_to_from},
    {0, NULL}
}


static struct PyArrayDtypeMethodDef concrete_methods[] = {
    {npy_cdt_take, take},
    // TODO: Decide how to handle the multiple sorting methods:
    {npy_cdt_sort, sort},
    {0, NULL}
}

// ALTERNATIVE: Could use strings, instead of npy_*_... integers/enum.

typedef struct {
    /* Possibly PyTypeSpec (to simplify creation) */
    /* Other necessary information */
    int flags,  // (some flags, such as flexible yes/no.)
    *PyArrayDtypeMethodDef abstract_methods,
    *PyArrayDTypeMethodDef concrete_methods,
} PyArrayDTypeSpec

 
 /*
  * Note that PyArrayDescr_Type will be typed as a DtypeMeta struct.
  * Users may extend it. The function below initializes the DTypeMeta class.
  * The legacy registration will create a new DTypeMeta intance
  * (dtype subclass) and initalize it based on the existing information.
  */
PyObject* PyArrayDescrType_InitFromSpec(PyArrayDescr_Type *user_dtype, PyArrayDTypeSpec *);
```


### Casting Implementation


#### Current Implementation of Casting

One of the main features which datatypes need to support is casting between one another using ``arr.astype(new_dtype, casting="unsafe")``, or while executing ufuncs with different types (such as adding integer and floating point numbers). Currently casting is defined in multiple ways:

1. ``copyswap``/``copyswapn`` are defined for each dtype and can handle byte-swapping for non-native byte orders as well as unaligned memory.
2. ``castfuncs`` is filled on the ``from`` dtype and casts aligned and contiguous memory from one dtype to another (both in native byte order). Casting to builtin dtypes is normally in a C-vector. Casting to a user defined type is stored in an additional dictionary.

When casting (small) buffers will be used when necessary, using the first ``copyswapn`` to ensure that the second ``castfunc`` can handle the data. A general call will thus have ``input -> in_copyswapn -> castfunc -> out_copyswapn ->output``.

However, while user types use only these definitions, almost all actual casting uses a monolithic code which may or may not combine the above functions, supports strided memory layout and specialized implementations.


#### Proposed Casting API

The above section lists two steps of casting that are currently used (at least for user defined dtypes). That is, casting within the same DType using ``copyswapn`` for unaligned or byte swapped data, as well as casting between two dtypes using defined ``castfuncs``.

Within numpy, there are many specialized functions. There is thus a need to find a way to make many implementations available. The proposal here is to create a new ``CastingImpl`` object as a home for the specific ``castfunc`` or ``copyswapn`` functions. That is a ``CastingImpl`` is defined for two (or one) destinct DType ``CastingImpl[FromDType->ToDType]`` (where from and to DType can be identical). In practice the ``CastingImpl`` will be attached to one of the two DTypes, and returned by its corresponding slots (it may be automatically generated for the special case where ``FromDType`` and ``ToDType`` are identical).

This design is chosen for multiple reasons:
1. By using a ``CastingImpl`` Python object with largely hidden API, it is possible to start with providing only a few additional capabilities to extension dtype authors to begin with, while having full flexibility to allow performance relevant additions later.
2. ``CastingImpl`` should be a special subclass of ``UFuncImpl``. While it needs to expose (initially internally to NumPy) additional functionality to enable casting, this will allow to do ``cast_int_to_string = np.get_casting_implementation()`` and use ``cast_int_to_string`` as a (specialized) UFunc. (Note that it is not required that ``CastingImpl`` is a ``UFuncImpl`` subclass to begin with. In a first implementation ``CastingImpl`` may be a very limited object only useful internally as a home for user provided functionality.)
3. Casting between flexible DTypes (such as string lengths) requires to adjust the flexible dtype instances (casting a ``float32`` to string results in a different string length than a ``float64``). This is similar to the same need in UFuncs and ``CastingImpl`` can provide this functionality.


##### ``CastingImpl`` Definition and Usage

With the exception of casting to and from objects, which can be defined using the above slots, and casting within the same dtype, which can (for non-flexible dtypes) be defined by ``copyswap(n)``, casting functions have to be defined specifically.
The above slots only define the resolution step to find the correct ``CastingImpl``.

The CastingImpl (similar to a UfuncImpl) will have to define:
  * *``adjust_descriptors``* or a similar method, which takes the input descriptor instance (and possibly given output one) and define the output descr.
      * For a ``CastingImpl`` defining casting within a dtype, these *must* match with the actual input ones.
      * A CastingImpl which does not handle any flexible dtypes does not need to define ``adjust_descriptors``
      * The signature will be ``adjust_descritpors((input_dtype, out_dtype or None), casting="safe")`` with the function returning a new, modified set of ``dtypes`` (all instances) or raising a ``TypeError``.
  * An (initially) internal slot: ``get_casting_function()`` with a signature identical (or similar) to:
    ```C
    PyArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
    ```
    (although the handling of NpyAuxData needs to be discussed, i.e. it could be replaced with setup/teardown funtions on the ``UfuncImpl``/``CastingImpl``).
        * The strides are ``MAX_INTP`` if they are not fixed, so an optimal function can be chosen.
        * It might make sense to pass a buffersize?
        * Currently source or destination can be NULL (relevant for objects), object handleling could be tricky, and we may think about limiting them? Also related to the ``move_references`` flag.
        * (This has overlap or identical to enabling some ufunc chaining) 

Users will initially only be able to define ``CastingImpl`` in a very limited manner, the minimal API for allowing all past features is:
  * Automatically wrapping of ``copyswap``/``copyswapn`` for simplicity and existing dtypes.
  * From a contiguous casting function and additional strided input, passing in the dtype instances. 

However, this should be expanded, and we may simply allow downstream users to override the ``get_casting_function`` slot in the future.


### Python level interface

To expose these slots to python, automatic wrappers shall be created for
the slots if they are defined as python functions (under specific names).
Some slots maybe exposed as capsule objects to:
  * Allow reusing existing slots from other dtypes while avoiding slowdowns
  * Replacing slots with faster C versions (including by just in time compilers)

The Python interface may be a second step, and may be limited in many cases.
However, it is a specific design goal to allow the defintion e.g. of Unit dtypes
in Python. This will also require similar python wrapping for ``CastingImpl`` and
``UFuncImpl`` (and resolvers) in general.

The exact API should be an additional NEP. A likely design goal will be to allow defining new dtypes using:

```
@np.dtype
class Coordinate:
   x: np.float64
   y: np.float64
   z: np.float64
```


### Notes on Casting and DType Discovery

The design presented here means that DType classes are first class objects and finding the correct DType class always happens first both for coercion from python and when finding the correct ``UFuncImpl`` to call.

For non-flexible DTypes, the second step is trivial, since they have a canonical implementation (if there is only a single instance, that one should be typically used for backward compatibility though). For flexible DTypes a second pass is needed, this is either an ``adjust_dtypes`` step within UFuncs, or ``__discover_descr_from_pyobject__`` when coercing within ``np.array``. For the latter, this generally means a second pass is necessary for flexible dtypes (although it may be possible to optimize that for common cases). In this case the ``__common_instance__`` method has to be used as well.

There is currently an open question whether ``adjust_dtypes`` may require the values in some cases. This is currently *not* strictly necessary (with the exception that ``objarr.astype("S")`` will use coercion rather than casting logic, a special case that needs to remain). It could be allowed by giving ``adjust_dtypes`` the input array in certain cases. For the moment it seems preferable to avoid this, if such a disovery step is required, it will require a helper function::

```python
arr = np.random.randint(100, size=1000)
categorical = find_categorical_dtype(arr)
cat_array = arr.astype(categorical)  # may error if arr was mutated
```



Related Work
------------

**TODO:** This section should list relevant and/or similar technologies, possibly in other
libraries. It does not need to be comprehensive, just list the major examples of
prior and relevant art.


Implementation
--------------

First the definition and instanciation of new style data types need to be defined and the current ones replaced/wrapped. (Hopefully this is a bit tricky but fairly straight forward with wrappers)

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

  * ``PyArray_DescrCheck`` currently tests explicitly for being an instance of PyArray_Descr. The Macro is thus not backward compatible (it cannot work in new numpy versions). This Macro is not used often, for example not even SciPy uses it. This will require an ABI breakage, to migitate this new versions of legacy numpy (e.g. 1.14.x, etc.) will be released to include a macro that is compatible with newer NumPy versions. Thus, downstream will may be forced to recompile, but can do so with a single (old) NumPy version.
  * The array that is currently provided to some functions (such as cast functions), may not be provided anymore generally (unless easily available). For compatibility, a dummy array with the dtype information will be given instead.
  * The ``scalarkind`` slot and registration of scalar casting will be removed/ignored without replacement (it currently allows partial value based. The ``PyArray_ScalarKind`` function will continue to work for builtin types, but will not be used internally and be deprecated.
  * The type of any dtype instance will not be ``dtype`` anymore, instead, it it will be a subclass of DType.
  * Current user dtypes are specifically defined as instances of ``np.dtype``, the instance used when registered is typically not held on to, but at the very least its type and base would have to be exchanged/modified. This may mean that the user created Descriptor struct/object is only partially usable (it does not need to be used though, and is not for either `rational` or `quaternion`)


Alternatives
------------

* Instead of the casting as a slot approach, casting could be handled more like a UFunc with resolver registration. The slot approach means that two function calls are sometimes necessary, however, is in general the simpler lookup for this very special function. It has the small advantage that third parties cannot easily change the casting logic of other types.

* While this NEP does not forbid future expansion to allow subclassing of DTypes (other than ``np.dtype`` itself), this seems not desirable. Subclassing can be very confusing with respect to casting and UFunc dispatching being inherited. Instead subclassing of ``AbstractDTtypes`` is specifically allowed thus separating many of the issue of inheritance from the use case of defining a type hierarchy.
    * TODO: ``AbstractDTypes`` can define certain slots, we have to decide how/if they are inherited! Most likely, it should be required to override/unset them (that also allows us to have flexibility down the road).


Rejected Alternatives/Different Approaches
------------------------------------------


#### Instances of dtypes should be scalars

When considering the numerical NumPy types such as ``float32``, ``float64``, etc. It is tempting to suggest that the scalars should be instances of the dtype. In this way a ``Float64`` class would both have all information of a DType, while ``Float64(1.)`` is its instances. (As opposed to currently ``np.dtype("float64").type(1.)`` being an instance of a scalar with a ``.dtype`` attribute).

This has been rejected for the following reasons:

1. Most importantly, for existing Python types, it is not viable to store additional information (as required by NumPy as use with an array) on the type itself. For example, it should be possible to write a DType for ``decimal.Decimal``. However, for such a ``DecimalDType`` it is impossible that its scalars are also instances (they already are instances of ``decimal.Decimal``).
2. Scalars currently do not have, and likely do not require, information such as non-native byte order, making both the types and the instances more complex than necessary.
3. While a beautiful concept, unless practically all types have such additional DType capabilities (i.e. this was a language feature of Python), the practical advantage seems to be small.

In short, this idea seems impractical while the actual enhancements offered seem unclear.


#### Keep the current layout of all NumPy dtypes being direct instances of ``np.dtype``

This would remove the ``DType`` class object introduced above. Such a class would still be used for user defined DTypes. The current layout still largely supports this as a (very) flexible ``DType`` class, but is not meant to be used this way. In this sense the ``DType`` class forms more a wide category of different ``dtypes``. 
The main advantage of that approach is the attempt to leave the current NumPy machinery as much intact as possible.

This has been rejected for the following reasons:

* While more invasive, the suggestion in this NEP should not be harder to implement (although possibly more work).
* A big category spanning floating point numbers and integers, means that casting still would have to rely on additional type numbers to see that casting to an ``>int64`` is done by first casting to ``<int64``.
* It seems not straight forward to define an AbstractDType hierarchy as above, with its additional features.


Discussion
----------

**TODO:** This section may just be a bullet list including links to any discussions
regarding the NEP:

- This includes links to mailing list threads or relevant GitHub issues.


References and Footnotes
------------------------


.. _PEP-384: https://www.python.org/dev/peps/pep-0384/
.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document is publish under the Open Publication License [_Open].
(**TODO:** maybe public domain is OK/works?)