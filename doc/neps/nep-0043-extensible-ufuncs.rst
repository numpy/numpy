.. _NEP43:

==============================================================================
NEP 43 — Enhancing the Extensibility of UFuncs
==============================================================================

:title: Enhancing the Extensibility of UFuncs
:Author: Sebastian Berg
:Status: Draft
:Type: Standard
:Created: 2020-06-20


.. note::

    This NEP is third in a series:

    - :ref:`NEP 40 <NEP40>` explains the shortcomings of NumPy's dtype implementation.

    - :ref:`NEP 41 <NEP41>` gives an overview of our proposed replacement.

    - :ref:`NEP 42 <NEP42>`  describes the new design's datatype-related APIs.

    - NEP 43 (this document) describes the new design's API for universal functions.


******************************************************************************
Abstract
******************************************************************************

The previous NEP 42 document proposes the creation of new DTypes which can
be defined by users outside of NumPy itself.
While implementing NEP 42 will users to create arrays with a custom dtype
and stored values, this NEP outlines how NumPy will operate on these arrays
in the future.
The main aspect is that functions operating on NumPy arrays are called
"universal functions" (ufunc) which include all math functions, such as
``np.add``, ``np.multiply``, and even ``np.matmul``.
Universal functions may operate on multiple arrays with potentially
different datatypes.
Their nature means that they also must efficiently operate on many elements.

This NEP proposes to expand the design of universal functions.
It defines a clear distinction between the ufunc which can operate
on many different dtypes such as floats or integers,
and the new ``ArrayMethod`` defining the functionality for a fixed dtypes.

.. note::

    Details of the private and external APIs may change to reflect user
    comments and implementation constraints. The underlying principles and
    choices should not change significantly.

******************************************************************************
Motivation and scope
******************************************************************************

As a continuation of NEP 42, the goal of this NEP is to extend universal
functions to DTypes defined outside of NumPy.
While the main motivation is enabling new user defined DTypes, this will
significantly simplify defining universal functions for NumPy string or
structured dtypes.
Until now, these dtypes are not supported by any of NumPy's functions
(such as ``np.add`` or ``np.equal``), due to difficulties arising from
their parametric nature (compare NEP 41 and 42).

Functions on arrays must handle a number of distinct steps which are
described in more detail in section `Steps involved in a UFunc call`_.
The most important ones are:

- Organizing all functionality which a new DType requires to define a
  ufunc call, currently sometimes called "inner-loop".
- Deal with input for which no exact matching function is found.
  For example when ``int32`` and ``float64`` are added, the ``int32``
  is cast to ``float64``.  This requires a distinct "promotion" step.

After organizing and defining these, we need to:

- Define the user API necessary to customize both of the above point
- Allow convenient reuse of existing functionality.
  For example a DType representing physical units, such as meters,
  should be able to fall back to NumPy's existing math implementations.

This NEP details how these requirements will be achieved in NumPy.

- All DType specific functionality that is currently part of the ufunc
  definition will be defined by a new `ArrayMethod`_ object.
  This ``ArrayMethod`` object will be the new way to describe any function
  operating on arrays.

- Ufuncs must dispatch to the ``ArrayMethod`` and potentially use promotion
  to find the correct ``ArrayMethod`` to use.
  This will be described in the `Promotion and dispatching`_ section.

A new C-API will be outlined in each section. A future Python API is
expected to be very similar and the C-API is presented in terms of Python
code.

The NEP proposes a large, but necessary, refactor of the NumPy ufunc internals.
This modernization will not affect users directly and is not only a necessary
step for new DTypes, but also future improvements to the ufunc machinery.

While the most important restructure proposed is the new ``ArrayMethod``
object, the largest long term consideration is the design choice for
promotion and dispatching.


***********************
Backwards Compatibility
***********************

The general backwards compatibility issues have also been listed
previously in NEP 41.

The vast majority of users should not see any changes beyond those typical
for NumPy released.
There are two main users (or use-cases) impacted by the proposed changes:

1. The Numba package uses direct access to the NumPy C-loops and modifies
   the NumPy ufunc struct directly for its own purposes.
2. E.g. Astropy uses its own type resolver, meaning that a default switch over
   from the existing type resolution to a new default Promoter may not
   be fully smooth.
3. It is currently possible to register loops for dtype *instances*.
   This is theoretically useful for structured dtypes and would be a resolution
   step happening *after* the DType resolution step proposed here.


This NEP will try hard to maintain backward compatibility as much as
possible, even though both of these projects have signaled willingness to
breaking changes.

The main reason why NumPy will be able to provide backward compatibility
is that:

* Legacy inner-loops can be wrapped adding an indirection to the call but
  maintaining full backwards compatibility.
  The ``get_loop`` function can in this case search the existing
  inner-loop functions (which are stored on the ufunc directly) in order
  to maintain full compatibility even with potential direct structure access.
* Legacy type resolvers can be called as a fallback (potentially caching
  the result). The resolver may need to be called twice (once for the DType
  resolution and once for the ``resolve_descriptor`` implementation).
* The fallback to the legacy type resolver should in most cases handle loops
  defined for such structured dtype instances.  This is because if there is no
  other ``np.Void`` implementation, the legacy fallback will retain the old
  behaviour.

The masked type resolvers specifically will *not* remain supported, but
have no known users (this even includes NumPy, which only uses the default
itself).

While the above changes potentially break some workflows,
we believe that the long term improvements vastly outweigh this.
Further, packages such as astropy and Numba are capable of adapting so that
end-users may need to update their libraries but not their code.


******************************************************************************
Usage and impact
******************************************************************************

This NEP restructures how operations on NumPy arrays are defined both
within NumPy and for external users.
The NEP mainly concerns those who either extend ufuncs for custom DTypes
or create custom ufuncs themselves.  It does not aim to finalize all
potential use-cases, but rather restructure NumPy in an extensible way
where solving these issues will be possible incrementally.


Overview and end user API 
=========================

to give an overview of how this NEP proposes the structure of ufuncs,
the following describe the possible exposure of the proposed restructure
to the end user.

Universal functions are much like a Python method defined on the DType of
the array when considering a ufunc with only a single input::

    res = np.positive(arr)

could be implemented (conceptionally) as:

    positive_impl = arr.dtype.positive
    res = positive_impl(arr)

However, unlike methods, ``positive_impl`` is not stored on the dtype itself.
It is rather the implementation of ``np.positive`` for a specific DType.
Current NumPy partially exposes this "choice of implementation" using
the ``dtype`` (or more exact ``signature``) attribute in universal functions,
although these are rarely used:

    np.positive(arr, dtype=np.float64)

forces NumPy to use the ``positive_impl`` written specifically for the Float64
DType.

This NEP makes this distinction more explicit, by creating a new object to
represent ``positive_impl``::

    positive_impl = np.positive.resolve_impl(type(arr.dtype))

While the creation of a ``positive_impl`` object and the ``resolve_impl``
method is part of this NEP, the following code::

    res = positive_impl(arr)

may not be implemented initially and is not central to the redesign.

In general NumPy universal functions can take many inputs.
This requires looking up the implementation by considering all of them
and makes ufuncs "multi-methods" with respect to the input DTypes::

    add_impl = np.add.resolve_impl(type(arr1.dtype), type(arr2.dtype))

This NEP defines how ``positive_impl`` and ``add_impl`` will be represented
as a new ``ArrayMethod`` and can be defined outside of NumPy.
Further, it defines how ``resolve_impl`` will be implemented, covering the
dispatching and promotion.

The reasons for this split may be made more clear in the section
`Steps involved in a UFunc call`_.


Defining a new ufunc implementation
===================================

An example of how to add a new loop, will look the following way;
initially using a C-API:

.. code-block:: python

    class StringEquality(BoundArrayMethod):
        nin = 1
        nout = 1
        DTypes = (String, String, Bool)

        def resolve_descriptors(context, given_descrs):
            """The strided loop supports all input string dtype instances
            and always returns a boolean. (String is always native byte order.)

            Defining this function is not necessary, since NumPy can provide
            it by default.
            """
            assert isinstance(given_descrs[0], context.DTypes[0])
            assert isinstance(given_descrs[1], context.DTypes[1])
            
            # The operation is always "safe" casting (most ufuncs are)
            return (given_descrs[0], given_descrs[1], context.DTypes[2]()), "safe"

        def strided_loop(context, n, data, strides):
            # n: Number of 
            # data: Pointers to the array data.
            # strides: strides to iterate all elements
            num_chars1 = context.descriptors[0].itemsize
            num_chars2 = context.descriptors[0].itemsize

            # C code using the above information to compare the strings in
            # both arrays.  In particular, this loop requires the `num_chars1`
            # and `num_chars2`.  Information which is currently not easily
            # available.

    np.equal.register_impl(StringEquality)
    del StringEquality  # may be deleted.


This definition will be sufficient to create a new loop, although the
structure will allow for expansion in the future; something that is already
required to implement casting within NumPy itself.
We use ``BoundArrayMethod`` and a ``context`` object here.  These
are described and motivated in details later. Briefly:

* ``context`` is a generalization of the ``self`` that Python passes to its
  methods.
* ``BoundArrayMethod`` is roughly equivalent to the Python distinction that
  ``class.method`` is a method, while ``class().method`` returns a "bound" method.


Customizing Dispatching and Promotion
=====================================

Finding the correct implementation when ``np.positive.resolve_impl()`` is
called is largely an implementation detail.
But, in some cases it may be necessary to influence this process when no
implementation matches the requested DTypes exactly:

.. code-block:: python

    np.multiple.resolve_impl((Timedelta64, Int8, None))

will not find a loop, because NumPy only defines a loop for multiplying
``Timedelta64`` with ``Int64``.
In simple cases, NumPy will use a default promotion step to attempt to find
the correct implementation, but to implement the above step, we will allow
the following:

.. code-block:: python

    def promote_timedelta_integer(ufunc, dtypes):
        new_dtypes = (Timdelta64, Int64, dtypes[-1])
        # Resolve again, using Int64:
        return ufunc.resolve_impl(new_dtypes)

    np.multiple.register_promoter(
        (Timdelta64, SignedInteger, None), promote_timedelta_integer)

Where ``SignedInteger`` is an abstract DType (compare NEP 42).


.. _steps_of_a_ufunc_call:

****************************************************************************
Steps involved in a UFunc call
****************************************************************************

Before going into more detailed API choices, it is necessary to review the
typical steps involved in a call to a universal function in NumPy.

A UFunc call consists of into multiple steps:

1. Resolution of ``__array_ufunc__`` for container types, such as a Dask
   array handling the full process, rather than NumPy.
   This step is performed first, and unaffected by this NEP.

2. *Promotion and dispatching*

   * Given the DTypes of all inputs we need to find the correct implementation
     for the ufuncs functionality. E.g. an implementation for ``float64``
     or ``int64``, but also a user-defined DType.

   * When no exact implementation exists, *promotion* has to be performed.
     For example, adding ``float32`` and ``float64`` is implemented by
     first casting the ``float32`` to ``float64``.

3. *DType Adaptation:*

   * The step has to perform no special work for non-parametric dtypes.
   * For example, if a loop adds two strings, it is necessary to define the
     correct output (and possibly input) dtypes.  ``S5 + S4 -> S9``, while
     an ``upper`` function has the signature ``S5 -> S5``.

4. Preparing the actual iteration. This step is largely handled by ``NpyIter`` (the iterator).

   * Allocate all outputs and temporary buffers which are necessary perform
     casts.
   * Finds the best iteration order, which includes information such as
     a broadcasted stride always being 0.

5. Setup may include finding an optimal function to do the operation and
   include:

   * Clearing of floating point exception flags (if necessary),
   * Possibly allocating temporary working space,
   * Setting (and potentially finding) the inner-loop function.  Finding
     the inner-loop function could allow specialized implementations in the
     future.
     For example casting currently use one function for contiguous casts
     and another function for generic strided casts to optimize speed.
     Reductions do similar optimizations, however these currently handled
     inside the inner-loop function itself.
   * Signal whether the inner-loop requires the Python API, or whether
     the GIL may be released.

6. Run the DType specific *inner-loop*

   * The loop may require access to additional data, such as dtypes or
     additional data set in the previous step.

7. Teardown may be necessary to undo any setup done in step 5
   such as checking for floating point errors.

The ``ArrayMethod`` provides a concept to group steps 3 to 7.
However, from a user perspective, it is necessary provide all information
for step 3, 5, and 7. At this time, steps 4 and 6 are functionality provided
by NumPy and cannot be customized.

The second step is promotion and dispatching which will also be restructured
with new API to influence the process.

Step 1 is listed for completeness and together with 4 and 6 are not directly
affected by this NEP.

The following sections first give an overview of the Array method and then
the new dispatching and promotion design.

The following picture gives an overview of these steps and how they will be
structured:

.. figure:: _static/nep43-sketch.svg
    :figclass: align-center


*****************************************************************************
ArrayMethod
*****************************************************************************

The central proposal is the creation of the ``ArrayMethod``, as an object
describing each function.
We use the ``class`` syntax to describe the information required to create
a new ``ArrayMethod`` object:

.. code-block:: python
    :dedent: 0

    class ArrayMethod:
        str : name  # Name, mainly useful for debugging

        # Casting safety information (almost always "safe", necessary to
        # unify casting and universal functions)
        Casting : casting = "safe"

        # More general flags:
        int : flags 

        @staticmethod
        def resolve_descriptors(
                Context: context, Tuple[DType]: given_descrs)-> Casting, Tuple[DType]:
            """Returns the safety of the operation (casting safety) and the
            """
            # A default implementation can be provided for non-parametric
            # output dtypes.
            raise NotImplementedError

        @staticmethod
        def get_loop(Context : context, strides, ...) -> strided_loop_function, flags:
            """Returns the low-level C (strided inner-loop) function which
            performs the actual operation.
            
            This method may initially private, users will be able to provide
            a set of optimized inner-loop functions instead:
            
            * `strided_inner_loop`
            * `contiguous_inner_loop`
            * `unaligned_strided_loop`
            * ...
            """
            raise NotImplementedError

        @staticmethod
        def strided_inner_loop(Context : context, data, strides,...):
            """The inner-loop (equivalent to the current ufunc loop)
            which is returned by the default `get_loop()` implementation."""
            raise NotImplementedError

With ``Context`` providing mostly static information about the function call:

.. code-block:: python
    :dedent: 0

    class Context:
        # The ArrayMethod object itself:
        ArrayMethod : method

        # Information about the caller, e.g. the ufunc, such as `np.add`:
        callable : caller = None
        # The number of input arguments:
        int : nin = 1
        # The number of output arguments:
        int : nout = 1
        # The DTypes this Method operates on/is defined for:
        Tuple[DTypeMeta] : dtypes
        # The actual dtypes instances the inner-loop operates on:
        Tuple[DType] : descriptors

        # Any additional information required. In the future, this will
        # generalize or duplicate things currently stored on the ufunc:
        #  - The ufunc signature of generalized ufuncs
        #  - The identity used for reductions

And ``flags`` stored properties, for whether:

* the ``ArrayMethod`` supports unaligned input and output arrays
* the inner-loop function requires the Python API (GIL)
* NumPy has to check the floating point error CPU flags.

More details will be added, since this NEP is concerned primarily with the big
picture design choice.


The call ``Context``
====================

The call "context" may seem surprising.  This object represents a similar
concept as Python passing ``self`` to all methods.
The following details the reasons for the above ``Context`` as it is.

To understand its existence, and the structure, it is helpful to remember
that a Python method can be written in the following way
(see also the `documentation of ``__get__``
<https://docs.python.org/3.8/reference/datamodel.html#object.__get__>`_):

.. code-block:: python

    class BoundMethod:
        def __init__(self, instance, method):
            self.instance = instance
            self.method = method

        def __call__(self, *args, **kwargs):
            return self.method.function(self.instance, *args, **kwargs)


    class Method:
        def __init__(self, function):
            self.function = function

        def __get__(self, instance, owner=None):
            assert instance is not None  # unsupported here
            return BoundMethod(instance, self)            


With which the following two methods behave identical:

.. code-block:: python

    def function(self):
        print(self)

    class MyClass:
        def method1(self):
            print(self)

        method2 = Method(function)

And both will print the same result:

.. code-block:: python

    >>> myinstance = MyClass()
    >>> myinstance.method1()
    <__main__.MyClass object at 0x7eff65436d00>
    >>> myinstance.method2()
    <__main__.MyClass object at 0x7eff65436d00>

Here the ``self.instance`` would be all that the above ``Context`` consists of.
There are two reasons for the more general ``Context``:

1. Unlike a method which operates on a single class instance, the ``ArrayMethod``
   operates on many input arrays and thus many dtypes.
2. The ``__call__`` of the ``BoundMethod`` above contains only a single call
   to the function. A ufunc will require multiple function calls.
   For example the inner-loop function is often called more than once.

Just as Python requires the distinction of a method and a bound method,
NumPy will have a ``BoundArrayMethod``, which stores all of the constant
information that is part of the ``Context``, such as:

* The ``DTypes``
* The number of input and ouput arguments
* The ufunc signature

Fortunately, most users and even ufunc implementers will not have to worry
much about these internal details; just like few Python users need to know
about the ``__get__`` dunder method.
A ``context`` object or C-structure provides all necessary data to the
fast C-functions and a convenient API creates the new ``ArrayMethod`` or
``BoundArrayMethod`` as required.


.. _ArrayMethod_specs:

ArrayMethod Specifications
==========================

These specifications provide a minimal initial C-API, which shall be expanded
in the future, for example to allow specialized inner-loops.

Briefly, NumPy currently relies fully on strided inner-loops and, this
will be the only allowed method of defining a ufunc initially.
We expect the addition of a ``setup`` function or exposure of ``get_loop``
in the future.

UFuncs require the same information as casting, giving the following
definitions (see also :ref:`NEP 42 <NEP42>` ``CastingImpl``):

* A new structure to be passed to the resolve function and inner-loop::
  
        typedef struct {
            PyObject *caller;  /* The ufunc object */
            PyArrayMethodObject *method;

            int nin, nout;

            PyArray_DTypeMeta **dtypes;
            /* Operand descriptors, filled in by resolve_desciptors */
            PyArray_Descr **descriptors;

            void *reserved;  // For Potential in threading (Interpreter state)
        } PyArrayMethod_Context
  
  This structure may be appended to include additional information in future
  versions of NumPy and includes all constant loop metadata.

  We could version this structure, although it may be simpler to version
  the ``ArrayMethod`` itself.

* Similar to casting, ufuncs may need to find the correct loop dtype
  or indicate that a loop is only capable of handling certain instances of
  the involved DTypes (e.g. only native byte order).  This is handled by
  an ``resolve_descriptors`` function (identical to the ``resolve_descriptors``
  of ``CastingImpl``)::

      NPY_CASTING
      resolve_descriptors(
              PyArrayMethod_Context *context,
              PyArray_Descr *given_dtypes[nin+nout],
              PyArray_Descr *loop_dtypes[nin+nout]);

  The function writes ``loop_dtypes`` based on the given ``given_dtypes``.
  This typically means filling in the descriptor of the output(s).
  Although often also the input descriptor(s) have to be found, e.g.
  to ensure native byte order when needed by the inner-loop.
  
  In most cases an ``ArrayMethod`` will have non-parametric output DTypes
  so that a default implementation can be provided.

* An additional ``void *user_data`` will usually be typed to extend
  the existing ``NpyAuxData *`` struct::
  
        struct {
            NpyAuxData_FreeFunc *free;
            NpyAuxData_CloneFunc *clone;
            /* To allow for a bit of expansion without breaking the ABI */
           void *reserved[2];
        } NpyAuxData;

  This struct is currently mainly used for the NumPy internal casting
  machinery and as of now both ``free`` and ``clone`` must be provided,
  although this could be relaxed.

  Unlike NumPy casts, the vast majority of ufuncs currently does not require
  this additional scratch-space, but may need simple flagging capability
  for example for implementing warnings (see Error and Warning Handling below).
  To simplify this NumPy will pass a single zero initialized ``npy_intp *``
  when ``user_data`` is not set. 

* The optional ``get_loop`` function will not be public initially, to avoid
  small design choices due to differences in the ufunc and casting APIs::

        innerloop *
        get_loop(
            PyArrayMethod_Context *context,
            /* (move_references is currently used internally for casting) */
            int aligned, int move_references,
            npy_intp *strides,
            PyArray_StridedUnaryOp **out_loop,
            NpyAuxData **userdata,
            NPY_ARRAYMETHOD_FLAGS *flags);
  
  The ``NPY_ARRAYMETHOD_FLAGS`` can indicate whether the Python API is required
  and floating point errors must be checked.

* The inner-loop function::

    int inner_loop(PyArrayMethod_Context *context, ..., void *userdata);

  Will have the identical signature to current inner-loops with the following
  changes:

  * A return value to indicate an error when returning ``-1`` instead of ``0``.
    When returning ``-1`` a Python error must be set.
  * The new, first argument ``PyArrayMethod_Context *`` to pass in potentially
    required information about the ufunc or descriptors in a convenient way.
  * The ``void *userdata`` will be the ``NpyAuxData **userdata`` as set by
    ``get_loop``.  If ``get_loop`` does not set ``userdata`` a ``npy_intp *``
    is passed instead (see `Error Handling`_ below for the motivation).

  *Note:* Since ``get_loop`` is expected to be private in the exact implementation
  of the ``userdata`` can be modified until final exposure.

Creation of a new ``BoundArrayMethod`` will use a ``PyArrayMethod_FromSpec()``
function.  A shorthand will allow direct registration to a ufunc using
``PyUFunc_AddImplementationFromSpec()``.  The specification is expected
to contain the following (this may extend in the future)::

    typedef struct {
        const char *name;  /* Generic name, mainly for debugging */
        int nin, nout;
        NPY_CASTING casting;
        NPY_ARRAYMETHOD_FLAGS flags;
        PyArray_DTypeMeta **dtypes;
        PyType_Slot *slots;
    } PyArrayMethod_Spec;


Discussion and Alternatives
===========================

The above split into an ``ArrayMethod`` and ``Context`` and the additional
requirement of a ``BoundArrayMethod`` seems a necessary split mirroring the
implementation of methods and bound methods in Python.

One reason for this requirement is that it allows storing the ``ArrayMethod``
object in many cases without holding references to the ``DTypes`` which may
be important if DTypes are created (and deleted) dynamically.
(This is a complex topic, which may not have a complete solution, but the
approach solves the issue for casting.)

There seem no alternatives to this structure.  Separating the DType
specific steps from the general ufunc dispatching and promotion is
absolutely necessary to allow future extension and flexibility.
Furthermore, it allows unifying casting and ufuncs.

Since the structure of ``ArrayMethod`` and ``BoundArrayMethod`` will be
opaque, there are no larger design implications aside from the choice of
making them Python objects.

This NEP does not lay out any alternatives. A new structure to house the
loop function and separate 


``resolve_descriptors``
------------------------

The ``resolve_descriptors`` method is possibly the main innovation of this
NEP and it is central also in the implementation of casting in NEP 42.

By ensuring that every ``ArrayMethod`` provides ``resolve_descriptors`` we
define a unified, clear API for step 3 in `Steps involved in a UFunc call`_.
This step is required to allocate output arrays and has to happen before
things like casting can be handled.

While the returned casting-safety (``NPY_CASTING``) will almost always be
"safe" for universal functions, including it has two big advantages:

1. Returning the casting safety is very important in NEP 42 for casting and
   allows the use of ``ArrayMethod`` also there.
2. There may be future desire to implement fast but unsafe implementations,
   for example for ``int64 + int64 -> int32`` which is unsafe from a casting
   perspective. Currently, this would use ``int64 + int64 -> int64`` and then
   cast to ``int32``, an implementation that includes the cast directly would
   have to signal that it effectively includes a "same-kind" cast.


``get_loop`` method
-------------------

Currently, NumPy ufuncs typically only provide a single strided loop, so that
the ``get_loop`` method may seem unnecessary, at least initially.
For this reason we plan for ``get_loop`` to be a private function initially.

However, ``get_loop`` is required for casting where specialized loops are
used even beyond strided and contiguous loops.  The ``get_loop`` function
must thus be a full replacement for the internal ``PyArray_GetDTypeTransferFunction``.

In the future, ``get_loop`` may be made public or a new ``setup`` function
be exposed to allow more control over setting up.
Further, we could expand ``get_loop`` and allow the ``ArrayMethod`` implementer
to take full control, including the outer iteration.


Extending the inner-loop signature
----------------------------------

Extending the inner-loop signature is another central and necessary part of
the NEP.

**Passing in the "Context":**

Passing in the ``Context`` allows for potentially easier extending of
the signature in the future and access to the dtype instances which
the inner-loop operates on.
This is useful information for parametric dtypes. For example comparing
two strings must know the length of both strings. And this information is
stored on the dtype instances.

In principle passing in Context is not necessary, as it could be set up
as part of ``userdata`` in the ``get_loop`` function.
In this NEP we propose passing this struct to simplify creation of loops for
parametric DTypes.  Further, it may proof useful for passing information
such as the ``PyInterpreterState`` (for threading) or the ``caller`` to
allow printing a better error message including the name of the original
ufunc called.

**Passing in user data:**

The current casting implementation uses the existing ``NpyAuxData *`` to pass
in additional data as defined by ``get_loop``.
There may be good alternatives to the use of this structure, although it
seems like a simple solution, which is already used in NumPy and public API.

``NpyAyxData *`` is a light weight, allocated structure, since it already
exists in NumPy and has a ``free`` slot, it seems a natural choice.
To simplify some use-cases (see "Error Handling" below), we will pass a
``npy_intp *userdata = 0`` instead when ``userdata`` is not provided.

*Note: Since ``get_loop`` is expected to be private initially, this may not
be available publically, and thus may change with experience gained with
this new structure.*

The return value to indicate an error is an important, but currently missing,
feature in NumPy. The error handling is further complicated by the way
CPUs signal floating point errors.
Both are discussed in the next section.

Error Handling
""""""""""""""

In general inner-loops should set errors right away. However, they may also run
without the GIL. This requires locking the GIL, setting a Python error
and returning ``-1`` to indicate an error occurred::

    int
    inner_loop(PyArrayMethod_Context *context, ..., void *userdata)
    {
        NPY_ALLOW_C_API_DEF

        for (npy_intp i = 0; i < N; i++) {
            /* calculation */

            if (error_occurred) {
                NPY_ALLOW_C_API;
                PyErr_SetString(PyExc_ValueError,
                    "Error occurred inside inner_loop.");
                NPY_DISABLE_C_API
                return -1;
            }
        }
        return 0;
    }

Floating point errors are special, since they requires checking the hardware
state.
This is costly and inconvenient if done on every single call.
Thus, NumPy will handle these if flagged by the ``ArrayMethod``.
An ``ArrayMethod`` should never cause floating point error flags to be set
if it flags that these should not be checked. This could interfere when
chaining multiple methods, in particular when casting is necessary.

An alternative solution would be to allow setting the error only at a later
teardown stage at the same time when NumPy will also check the floating
point error flags.

We decided against this pattern at this time, it seems more complex and
generally unnecessary.
While safely grabbing the GIL in the loop may require passing in an additional
``PyThreadState`` or ``PyInterpreterState`` in the future (for subinterpreter
support), this is acceptable and planned.
While it may be useful in some cases, setting the error at a later point would
also add some complexity.
For instance, if operation is paused (which can happen for casting in particular),
this error check would have to be run explicitly.

We expect that setting errors immediately is the easiest and most convenient
solution and more complex solution will require more experience to design.

However, handling warnings is slightly more complex: A warning should be
given exactly once for each function call (i.e. for the whole array) even
if naively it would be given many times.
To simplify such a use case, we will pass in ``npy_intp *userdata = 0``
by default which can be used to store flags (or other simple persistent data).
For instance, we could imagine an integer multiplication loop which warns
when an overflow occurred::

    int
    integer_multiply(PyArrayMethod_Context *context, ..., npy_intp *userdata)
    {
        int overflow;
        NPY_ALLOW_C_API_DEF

        for (npy_intp i = 0; i < N; i++) {
            *out = multiply_integers(*in1, *in2, &overflow);

            if (overflow && !*userdata) {
                NPY_ALLOW_C_API;
                if (PyErr_Warn(PyExc_UserWarning,
                        "Integer overflow detected.") < 0) {
                    NPY_DISABLE_C_API
                    return -1;
                }
                *userdata = 1;
                NPY_DISABLE_C_API
        }
        return 0;
    }

*TODO:* The idea of passing an ``npy_intp`` scratch space when ``userdata``
is not set seems very convenient, but I am uncertain about it, since I am not
aware of any similar prior art.  This "scratch space" could also be part of
the ``context`` in principle.



Reusing existing Loops/Implementations
======================================

For many DTypes adding additional C-level (or python level) loops will be
sufficient and require no more than a single strided loop implementation.
Everything else can be provided by NumPy.  If the loop works with
parametric DTypes, the ``resolve_descriptors`` function *must* additionally
be provided.

However, in some use-cases it is desired to call back to an existing loop.
In Python, this can be achieved by simply calling into the original ufunc
(when parametric types are involved potentially twice, due to calling one
more time from ``resolve_descriptors``).

For better performance in C, and for large arrays, it is desirable to reuse
an existing ``ArrayMethod`` as much as possible, so that its inner-loop function
can be used directly without any overhead.
We will thus allow to create ``ArrayMethod`` by passing in an existing
``ArrayMethod``.

This wrapped loop will have two additional methods:

* ``view_inputs(Tuple[DType]: input_descr) -> Tuple[DType]`` replacing the
  user input descriptors with descriptors matching the wrapped loop.
  It must be possible to *view* the inputs as the output.
  For example for ``Unit[Float64]("m") + Unit[Float32]("km")`` this will
  return ``float64 + int32``. The original ``resolve_descriptors`` will
  convert this to ``float64 + float64``.

* ``wrap_outputs(Tuple[DType]: input_descr) -> Tuple[DType]`` replacing the
  resolved descriptors with with the desired actual loop descriptors.
  The original ``resolve_descriptors`` function will be called between these
  two calls, so that the output descriptors may not be set in the first call.
  In the above example it will use the ``float64`` as returned (which might
  have changed the byte-order), and further resolve the physical unit making
  the final signature::
  
      ``Unit[Float64]("m") + Unit[Float64]("m") -> Unit[Float64]("m")``

  the UFunc machinery will take care of casting the "km" input to "m".


The ``view_inputs`` method allows passing the correct inputs into the
original ``resolve_descriptors`` function, while ``wrap_outputs`` ensures
the correct descriptors are used for output allocation and input buffering casts.

An important use case for this is that of an abstract Unit DType
with subclasses for each numeric dtype (which could be dynamically created)::

    Unit[Float64]("m")
    # with Unit[Float64] being the concrete DType:
    isinstance(Unit[Float64], Unit)  # is True

Such a ``Unit[Float64]("m")`` instance has a well defined signature with
respect to type promotion.
The author of the ``Unit`` DType can implement most necessary logic by
wrapping the existing math functions and using the two additional methods
above.
Using the *promotion* step, this will allow to create a register a single
promoter for the abstract ``Unit`` DType with the ``ufunc``.
The promoter can then add the wrapped concrete ``ArrayMethod`` dynamically
at promotion time, and NumPy can cache (or store it) after the first call.

**Alternative use-case:**

A different use-case is that of a ``Unit(float64, "m")`` DType, where
the numerical type is part of the DType parameter.
This approach is possible, but will require a custom ``ArrayMethod``
which wraps existing loops.
It must also always require require two steps of dispatching
(one to the ``Unit`` DType and a second one for the numerical type).

Further, the efficient implementation will require the ability to
fetch and reuse the inner-loop function from another ``ArrayMethod``.
(Which is probably necessary for users like Numba, but it is uncertain
whether it should be a common pattern and it cannot be accessible from
Python itself.)


.. _promotion_and_dispatching:

*************************
Promotion and dispatching
*************************

NumPy ufuncs are multi-methods in the sense that they operate on (or with)
multiple DTypes at once.
While the input (and outpyt) dtypes are attached to NumPy arrays,
the ``ndarray`` type itself does not carry the information of which
function to apply to the data.

For example, given the input::

    arr1 = np.array([1, 2, 3], dtype=np.int64)
    arr2 = np.array([1, 2, 3], dtype=np.float64)
    np.add(arr1, arr2)

has to find the correct ``ArrayMethod`` to perform the operation.
Ideally, there is an exact match defined, e.g. if the above was written
as ``np.add(arr1, arr1)``, the ``ArrayMethod[Int64, Int64, out=Int64]`` matches
exactly and can be used.
However, in the above example there is no direct match, requiring a
promotion step.

**Description of the Promotion and dispatching Process:**

1. By default any UFunc has a promotion which uses the common DType of all
   inputs and tries again.  This is well defined for most mathematical
   functions, but can be disabled or customized if necessary.

2. Users can *register* new Promoters just as they can register a
   new ``ArrayMethod``.  These will use abstract DTypes to allow matching
   a large variation of signatures.
   The return value of a promotion function shall be a new ``ArrayMethod``
   or ``NotImplemented``.  It must be consistent over multiple calls with
   the same input to allow allows caching of the result.

The signature of a promotion function consists is defined by::

    promoter(np.ufunc: ufunc, Tuple[DTypeMeta]: DTypes): -> Union[ArrayMethod, NotImplemented]

Note that DTypes may contain the outputs DType, however, normally the
output DType should *not* affect which ``ArrayMethod`` is chosen.

In most cases, it should not be necessary to add a custom promotion function,
however, an example which requires this is multiplication with a
unit.
In NumPy ``timedelta64`` can be multiplied with most integers.
However, NumPy only defines a loop (``ArrayMethod``) for ``timedelta64 * int64``
so that multiplying with ``int32`` would fail.

To allow this, the following promoter can be registered for
``[Timedelta64, Integral, None]``::

    def promote(ufunc, DTypes):
        res = list(DTypes)
        try:
            res[1] = np.common_dtype(DTypes[1], Int64)
        except TypeError:
            return NotImplemented

        # Could check that res[1] is actually Int64
        return ufunc.resolve_impl(tuple(res))

In this case, just as a ``Timedelta64 * int64`` and ``int64 * timedelta64``
``ArrayMethod`` is necessary, a second promoter will have to be registered to
handle the case where the integer is passed first.

Promoter and ``ArrayMethod`` are discovered by finding the best matching one.
Initially, it will be an error if ``NotImplemented`` is returned or if two
promoters match the input equally well *unless* the mismatch occurs due to
unspecified output arguments:
When two signatures are identical for all inputs, but differ in the output
the first one registered is used.
In all other cases, the use of a more precise ``AbstractDType`` will allow to
resolve any disambiguities.

This above rules enable loop specialization if an output is supplied
or the full loop is specified.  It should not typically be necessary,
but allows resolving ``np.logic_or``, etc. which have both
``Object, Object->Bool`` and ``Object, Object->Object`` loops (using the
first by default).  In principle it can be used to add loops by-passing
casting, such as ``float32 + float32 -> float64`` *without* casting both
inputs to ``float64``.


Discussion and alternatives
===========================

Instead of resolving and returning a new implementation, we could also
return a new set of DTypes to use for dispatching.  This works, however,
it has the disadvantage that it cannot be possible to dispatch to a loop
defined on a different ufunc.


**Rejected Alternatives:**

In the above the promoters use a multiple dispatching style type resolution
while the current UFunc machinery rather uses the first
"safe" loop (see also :ref:`NEP 40 <NEP40>`) in an ordered hierarchy.

While the "safe" casting rule seems not restrictive enough, we could imagine
using a new "promote" casting rule, or the common-DType logic to find the
best matching loop by upcasting the inputs as necessary.

One downside to this approach is that upcasting alone allows upcasting the
result beyond what is expected by users:
Currently (which will remain supported as a fallback) any ufunc which defines
only a float64 loop will also work for float16 and float32 by *upcasting*::

    >>> from scipy.special import erf
    >>> erf(np.array([4.], dtype=np.float16))  # float16
    array([1.], dtype=float32)

with a float32 result.  It is impossible to change the ``erf`` function to
return a float16 result without possibly changing the result of following code.
In general, we argue that automatic upcasting should not occur in cases
where a less precise loop can be reasonably defined, *unless* the ufunc
author defines this behaviour intentionally.

This considerations means that upcasting has to be limited by some additional
method.

*Alternative 1:*

Assuming general upcasting is not intended, a rule must be defined to
limit upcasting the input from ``float16 -> float32`` either using generic
logic on the DTypes or the UFunc itself (or a combination of both).
The UFunc cannot do this easily on its own, since it cannot know all possible
DTypes which register loops.
Consider the two examples:

First (should be rejected):

* Input: ``float16 * float16``
* Existing loop: ``float32 * float32``

Second (should be accepted):

* Input: ``timedelta64 * int32``
* Existing loop: ``timedelta64 * int16``


This requires either:

1. The ``timedelta64`` to somehow signal that the ``int64`` upcast is
   always supported if it is involved in the operation.
2. The ``float32 * float32`` loop to reject upcasting.

Implementing the first approach requires signaling that upcasts are
acceptable in the specific context.  This would require additional hooks
and may not be simple for complex DTypes.

For the second approach in most cases a simple ``np.common_dtype`` rule will
work for initial dispatching, however, even this is only clearly the case
for homogeneous loops.
This option will require adding a function to check whether the input
is a valid upcast to each loop individually, which seems problematic.
In many cases a default could be provided (homogeneous signature).

*Alternative 2:*

An alternative "promotion" step is to ensure that the *output* DType matches
with the loop after first finding the correct output DType.
If the output DTypes are known, finding a safe loop becomes easy.
In the majority of cases this works, the correct output dtype is just::

    np.common_dtype(*input_DTypes)

or some fixed DType (e.g. Bool for logical functions).

However, it fails for example in the ``timedelta64 * int32`` case above since
there is a-priory no way to know that the "expected" result type of this
output is indeed ``timedelta64`` (``np.common_dtype(Datetime64, Int32)`` fails).
This requires some additional knowledge of the timedelta64 precision being
int64. Since a ufunc can have an arbitrary number of (relevant) inputs
it would thus at least require an additional ``__promoted_dtypes__`` method
on ``Datetime64`` (and all DTypes).

A further limitation is shown by masked DTypes.  Logical functions do not
have a boolean result when masked are involved, which would thus require the
original ufunc author to anticipate masked DTypes in this scheme.
Similarly, some functions defined for complex values will return real numbers
while others return complex numbers.  If the original author did not anticipate
complex numbers, the promotion may be incorrect for a later added complex loop.


We believe that promoters, while allowing for an huge theoretical complexity,
are the best solution:

1. Promotion allows for dynamically adding new loops. E.g. it is possible
   to define an abstract Unit DType, which dynamically creates classes to
   wrap existing other DTypes.  Using a single promoter, this DType can
   dynamically wrap existing ``ArrayMethod`` enabling it to find the correct
   Loop in a single lookup instead of otherwise two.
2. The promotion logic will usually err on the safe side: A newly added
   loop cannot be misused unless a promoter is added as well.
3. They put the burden of carefully thinking of whether the logic is correct
   on the programmer adding new loops to a UFunc.  (Compared to Alternative 2)
4. In case of incorrect existing promotion, writing a promoter to restrict
   or refine a generic rule is possible.  In general a promotion rule should
   never return an *incorrect* promotion, but if it the existing promotion
   logic fails or is incorrect for a newly added loop, the loop can add a
   new promoter to refine the logic.

The option of having each loop verify that no upcast occured is probably
the best alternative, but does not include the ability to dynamically
adding new loops.

The main downsides of general promoters is that they allow a possible
very large complexity.
A third-party library *could* add incorrect promotions to NumPy, however,
this is already possible by adding new incorrect loops.
In general we believe we can rely on downstream projects to use this
power and complexity carefully and responsibly.

*******************************************************
Notes and User Guidelines for Promoters and ArrayMethod
*******************************************************

In general adding a promoter to a UFunc must be done very carefully.
A promoter should never affect loops which can be reasonably defined
by other datatypes.  Defining a hypothetical ``erf(UnitFloat16)`` loop
must not lead to ``erf(float16)``.
In general a promoter should fulfill the requirements that:

* Be conservative when defining a new promotion rule. An incorrect result
  is a much more dangerous error than an unexpected error.
* One of the (abstract) DTypes added should typically match specifically with a
  DType (or family of DTypes) defined by your project.
  Never add promotion rules which go beyond normal common DType rules!
  It is *not* reasonable to add a loop for ``int16 + uint16 -> int24`` if
  you write an ``int24`` dtype. The result of this operation was already
  defined previously as ``int32`` and will be used with this assumption.
* A promoter (or loop) should never affect existing other loop results.
  Additionally, to changes in the resulting dtype, do not add for example
  faster but less precise loops/promoter.
* Try to stay within a clear, linear hierarchy for all promotion (and casting)
  related logic. NumPy itself breaks this logic for integers and floats
  (they are not strictly linear, since int64 cannot promote to float32).
* Loops and promoters can be added by any project, which could be:

  * The project defining the ufunc
  * The project defining the DType
  * A third-party project

  Try to find out which is the best project to add the loop.  If neither
  the project defining the ufunc or the project defining the DType add the
  loop, issues with multiple definitions (which are rejected) may arise
  and care should be taken that the loop behaviour is always more desirable
  than an error.

In some cases exceptions to these rules may make sense, however, in general
we ask you to use extreme caution and when in doubt create a new UFunc
instead.  This clearly notifies the users of differing rules.
When in doubt, ask on the NumPy mailing list or issue tracker!


**************
Implementation
**************

Implementation of this NEP will entail a large refactor and restructuring
of the current ufunc machinery (as well casting.

The implementation unfortunately will require large maintenance of the
UFunc machinery, since both the actual UFunc loop calls, as well as the
the initial dispatching steps have to be modified.

In general, the correct ``ArrayMethod``, also those returned by a promoter,
will be cached (or stored) inside a hashtable for efficient lookup.


**********
Discussion
**********

There is a large space of possible implementations with many discussions
in various places, as well as initial thoughts and design documents.
These are listed in the discussion of :ref:`NEP 40 <NEP40>` and not repeated here for
brevity.

A long discussion which touches many of these points and points towards
similar solutions can be found in
`the github issue 12518 "What should be the calling convention for ufunc inner loop signatures?" <https://github.com/numpy/numpy/issues/12518>`_


**********
References
**********

Please see NEP 40 and 41 for more discussion and references.


*********
Copyright
*********

This document has been placed in the public domain.
