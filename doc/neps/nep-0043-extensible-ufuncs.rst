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

    - NEP 42 (this document) describes the new design's datatype-related APIs.

    - NEP 43 describes the new design's API for universal functions.


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
- All convenient reuse of existing functionality.
  For example a DType representing physical units, such as meters,
  should be able to conveniently use NumPy's existing math functionality.



This requires two distinct things:

1. Splitting out all DType specific functionality that is currently part
   of the ufunc definition.  We wish to achieve this by defining a new
   ``ArrayMethod`` object.
2. 

Additionally, we need to consider how 


******************************************************************************
Backward compatibility
******************************************************************************




******************************************************************************
Usage and impact
******************************************************************************




Detailed Description
--------------------

NEP 41 and NEP 42 laid out the general proposed structure into DType
classes (similar to the current type numbers) and dtype instances.
Universal functions have a fairly fixed signature of multiple input arrays
and certain additional features. But depending on the input DTypes, they
have to perform a different operation.  In general these operations are
registered by users, e.g. implementers of a user-defined DTypes.


Introduction to General Concepts
""""""""""""""""""""""""""""""""

Universal functions as described above are much like a Python method
defined on the DType of the array when considering a ufunc with only
a single input::

    res = np.positive(arr)

could be implemented (conceptionally) as:

    positive_impl = arr.dtype.positive
    res = positive_impl(arr)

However, unlike methods, ``positive_impl`` is not stored on the dtype itself
a choice that this NEP does not wish to modify.
It is rather the implementation for ``np.positive`` for a specific DType.
Current NumPy partially exposes this "choice of implementation" using
the ``dtype`` (or more exact ``signature``) attribute in universal functions,
although these are rarely used:

    np.positive(arr, dtype=np.float64)

forces NumPy to use the ``positive_impl`` written specifically for the Float64
DType.
This NEP requires to partially represent this implementation explicitly::

    positive_impl = np.positive.resolve_impl(type(arr.dtype))

Although, this ``positive_impl`` and ``resolve_impl`` is required for certain
functionality, the following code::

    res = positive_impl(arr)

is not part of this NEP, as it is not central to the proposal.

As mentioned above, in general NumPy universal functions can take many
inputs, requiring looking up the implementation by considering all of them
making NumPy ufunc "multi-methods" with respect to the input *DTypes*::

    add_impl = np.add.resolve_impl(type(arr1.dtype), type(arr2.dtype))

This NEP strives to define the minimal API necessary to achieve
the above lists of steps and make it fully extensible for authors of both
custom ufuncs and DTypes.
There are two distinct API design decisions as part of this NEP:

1. For the above to work, we must also define::

       np.positive.resolve_impl(MyDType)

   In the case of multiple inputs, often *promotion* must occur:
   ``float32 + float64`` is actually handled by the implementation for
   ``float64 + float64``.  NumPy automatically converts (casts) the first
   argument.

   This is described in the :ref:`Promotion and Dispatching" <promotion_and_dispatching>`
   section and step 2 of the
   :ref:`The Steps involved in a UFunc call <steps_of_a_ufunc_call>` section.

2. The definition of the ``ArrayMethod`` specification, to allow users
   to add new "implementations" to a UFunc. This defines how the user
   can implement a new ``positive_impl`` for their custom DType and
   includes all necessary information to adapt steps 3-7 of the
   :ref:`The Steps involved in a UFunc call <steps_of_a_ufunc_call>` section.
   
       positive_impl = np.positive.resolve_impl(MyDateTime)

   exists and::

       np.positive(my_datetime_array)

   can succeed.
   
   This is described in the :ref:`"ArrayMethod Specifications" <ArrayMethod_specs>`
   and following sections.


.. _steps_of_a_ufunc_call:

Steps involved in a UFunc call
""""""""""""""""""""""""""""""

A UFunc call consists of into multiple steps:

1. Resolution of ``__array_ufunc__`` for container types, such as a Dask
   array handling the full process, rather than NumPy.
   This step is performed first, and unaffected by this NEP.

2. *Promotion and Dispatching*

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

5. *Loop-specific setup* may include for example:

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

7. *Loop-specific Teardown* may be necessary to undo any setup done in step 5
   such as checking for floating point errors.

This NEP proposes a new registration approach for step 2 by creating a private
``ArrayMethod`` structure which will be filled using an extensible API,
and which may be exposed as a Python object later.
This shall allow users to define custom behaviour for steps 3, 5, and 7,
while extending the inner-loop function (step 6) accordingly.

The following sections go into more details, and are seperated into the
two main topics of *promotion and dispatching* and the further C-API
provided to the user for the ufunc execution.


ArrayMethod Registration
""""""""""""""""""""""

*TODO:* we need to briefly mention registration, even if the details of
how to register it are in the specs or even later!

.. _ArrayMethod_specs::

ArrayMethod Specifications
""""""""""""""""""""""""

These specifications provide a minimal initial API, which shall be expanded
in the future, for example to allow specialized inner-loops.

Briefly, NumPy currently relies fully on strided inner-loops and, this
will be the only allowed method of defining a ufunc initially.
With additional setup and teardown functionality, as well as the
``adapt_descriptors`` function mirroring the same functionality as required
for casting (see also :ref:`NEP 42 <NEP42>` ``CastingImpl``).
This gives the following function definitions:

* Similar to casting, also ufuncs may need to find the correct output DType
  or indicate that a loop is only capable of handling certain instances of
  the involved DTypes (e.g. only native byte order).  This is handled by
  an ``resolve_descriptors`` function (similar to ``adjust_descriptors``
  of ``CastingImpl``)::

      int resolve_descriptors(
              PyUFuncObject *ufunc, PyArray_DTypeMeta *DTypes[nargs],
              PyArray_Descr *in_dtypes[nin+nout],
              PyArray_Descr *out_dtypes[nin+nout]);

  The function writes ``out_dtypes`` based on the given ``in_dtypes``.
  This typically means filling in the descriptor of the output(s).
  Although often also the input descriptor(s) have to be found,
  e.g. to ensure native byte order when needed by the inner-loop.

* Define a new structure to be passed in the inner-loop, which can be
  partially modified by the setup/teardown as well::
  
      typedef struct {
          PyUFuncObject *ufunc,
          /* could add information about __call__, reduce, etc. here */
          // method
          /* The exact operand dtypes of the inner-loop */
          PyArray_Descr const *dtypes;
          /* 
           * Reserved always NULL field, for potentially passing in the
           * PyThreadState or PyInterpreterState in the future.
           */
          void *reserved;  
          /* Per-loop (global) user data, equivalent to the current void* */
          void const *user_loop_data;
      } PyArray_UFuncData
  
  This structure may be appended to include additional information in future
  versions of NumPy and includes all constant loop metadata.

  **TODO:** We could version the PyArray_UFuncData struct.

* An additional ``void *user_data`` which will usually be typed to extend
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

* The *optional* setup function::

      innerloop *
      setup(PyArray_UFuncData *data, int *needs_api, npy_intp *fixed_strides,
            void *user_data);
  
  Is passed the above struct and may modify (only) the ``user_data`` slot
  and potentially further slots in the future.  The function returns
  the inner-loop or ``NULL`` to indicate an error.

  **TODO:** did I note whether this is initially public? I do not think it
  has to be...

* The inner-loop function::

    int inner_loop(..., PyArray_UFuncData *data);

  Will have the identical signature to current inner-loops with the difference
  of the additional return value and passing ``PyArray_UfuncData *`` instead
  of the previous ``void *`` representing ``user_loop_data``.
  The inner-loop shall return 0 to indicate continued (successful) execution.
  A non-zero return value will terminate the iteration.
  The inner-loop shall return a *negative* value (e.g. -1) with a Python
  exception set when an error occurred.

* Teardown of ``user_data`` is handled by the ``user_data->free`` field.
  The ``user_data->clone``
  ``NpyAuxData *`` is existing public API in NumPy, however, it is currently
  de-facto only used for internal casting.

* A flag to ask NumPy to perform floating point error checking (after custom
  setup and before user teardown).

To simplify setup and not require the implementation of setup/teardown for
the majority of ufuncs, NumPy provides floating error setup and teardown
if flagged during registration.


**Notes**

Alternatives and details to the ``resolve_descriptors`` function are described
below.

The current signature must be extended to allow the return value, as well
as error reporting.  The choice of passing in a pointer to a struct means
minimal adjustments to current functions which do not require it (they only
need a ``0`` return value).  It may also simplify the addition of future
parameters if necessary.

The main alternative would be extending the signature either by only a
return value giving a much higher burden to implement a user setup.


**Error and Warning Handling**

In general inner-loops should set errors right away. However, they may also run
without the GIL. This requires locking the GIL, setting a Python error
and returning ``-1`` to indicate an error occurred::

    int
    inner_loop(..., PyArray_UFuncData *data)
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

This may be problematic in the future for Python subinterpreter support,
in which case the interpreter state or threadstate shall be passed in
(i.e. the reserved, currently NULL field).

Floating point error is special, since it requires checking the hardware
state, which may be costly to do on every call (and inconvenient), NumPy
will handle these, if flagged by the ``ArrayMethod``.

In an initial *alternative* draft, error setting was allowed to be done
at teardown time similar to how floating point errors require checking.
We decide against allowing this pattern because it requires additional
checks if the computation is paused.  While this does not happen for
ufuncs currently, it does happen for casting within ``np.nditer``.

In general, we expect that errors can always be set immediately.
Warnings, should typically be given once *per call*, and not repeated
if the warning applies to multiple elements.
To make warning setting from inside the inner-loop function simpler,
or possibly do other things.  A single `npy_intp *user_data` is passed
instead of ``user_data`` if ``user_data`` is otherwise unused.
This allows to store a flag and avoids giving the warning more than once.
For any more complex use-cases, ``NpyAuxData *user_data`` has to be allocated.

**TODO:** I am not sure about this approach to scratch-space, but it would be
nice if we can have a simple default.  The alternative is to make a simple
extended ``NpyAuxData *``, to not require the user to implement it.
Or...?


Reusing existing Loops/Implementations
""""""""""""""""""""""""""""""""""""""

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


Details for ``resolve_descriptors``
"""""""""""""""""""""""""""""""""""

*TODO: Should this function also get the full set of information which
I want to pass in to the setup/teardown/inner-loop?  On the one hand, much
of the information is not yet available/defined (or is set here).  On the
other hand, some of the info is useful, and it may be nice to just have
a homogeneous calling convention.*

The UFunc machinery must know the correct dtypes to use before arrays can
be allocated. The arrays creation itself is desirable to happen before any
setup functionality to allow potential choosing of an optimized loops.

**Notes:**

There are a few possible extension to this function.  Currently, it also
takes care of casting in general.  This is not necessary, however, it
would be possible to extend the signature with casting indication for
*outputs*.
This would allow registering e.g. ``float64 + float64 -> float32`` as an explicit
(faster) loop while indicating that it is an unsafe cast on the result array,
which requires the user to specify ``casting="unsafe"``.

The current design allows such a specialized loop (with access to the
initially private ``setup``), from within the ``float64+float64->float64``
implementation only.


``ufunc.resolve_impl``
""""""""""""""""""""""

In the Introduction we describe use the following pattern::

    positive_impl = np.positive.resolve_impl(type(arr.dtype))

where ``positive_impl`` is defined by the ``ArrayMethod`` specifications above.

The ``ArrayMethod`` as defined above does not encompass all information included
in the UFunc and is explicitly passed the ``DTypes`` it is registered for.
This is to ensure that ``ArrayMethod`` is both lightweight and could be deleted
more easily in the event that a ``DType`` itself is deleted (making the
``ArrayMethod`` inaccessible.

For the reader wishing more details/thoughts, the pattern is rather more
similar to::

    class BoundArrayMethod:
        def __init__(self, ufunc, DTypes):
            self.ufunc = ufunc
            self.DTypes = DTypes

        @staticmethod
        def resolve_descriptors(ufunc, DTypes, input_dtypes):
            raise NotImplementedError

        @staticmethod
        def inner_loop(ufunc, DTypes, input_dtypes):
            raise NotImplementedError

Note the use of ``staticmethod`` in the example.  This bears some
similarity to methods: A method is passed the ``self`` argument, but
a method is otherwise a function, without any state of its own.
In this regard, ``ArrayMethod`` defines the "unbound method"::

    integer = 8
    unbound_conjugate = type(integer).conjugate

while::

    conjugate_impl = np.conjugate.resolve_impl(type(arr.dtype))

corresponds to the "bound method"::

    integer.conjugate

which is passed the relevant metadata (ufunc and DTypes), in a similar way
that a method is passed ``self``.
The current NEP does not allow the representation of the "unbound method"
as a Python object as of now.


.. _promotion_and_dispatching::

Promotion and Dispatching
"""""""""""""""""""""""""

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

**Description of the Promotion and Dispatching Process:**

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


**Alternative Details:**

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


Further Notes and User Guidelines for Promoters and ArrayMethod
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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


Implementation
--------------

The implementation unfortunately will require large maintenance of the
UFunc machinery, since both the actual UFunc loop calls, as well as the
the initial dispatching steps have to be modified.

In general, the correct ``ArrayMethod``, also those returned by a promoter,
will be cached (or stored) inside a hashtable for efficient lookup.



Backwards Compatibility
-----------------------

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

This NEP will try hard to maintain backward compatibility as much as
possible.  Most existing NumPy loops currently do not require any of the
additional parameter defined here, and currently user-defined registered
loops must remain supported.
Numba will require these to be stored and looked up the same way as
previously.

In general, NumPy this effort will try to remain compatible with Numba,
but full compatibility is hard to guarantee.
Numba is expected to adapt to the new NumPy release.

**TODO:** We should discuss this part, basically, I think we can go quite
far, but in practice we have to see what is practical...

Legacy type resolvers as used by Astropy should keep working
in most cases, although Astropy has signaled willingness in updating their
code.

The masked type resolvers specifically are unlikely to remain supported
but have no known users (this even includes NumPy, which only uses the default
itself).




Discussion
----------

There is a large space of possible implementations with many discussions
in various places, as well as initial thoughts and design documents.
These are listed in the discussion of :ref:`NEP 40 <NEP40>` and not repeated here for
brevity.

A long discussion which touches many of these points and points towards
similar solutions can be found in
`the github issue 12518 "What should be the calling convention for ufunc inner loop signatures?" <https://github.com/numpy/numpy/issues/12518>`_


References
----------

.. [1] NumPy currently inspects the value to allow the operations::

     np.array([1], dtype=np.uint8) + 1
     np.array([1.2], dtype=np.float32) + 1.

   to return a ``uint8`` or ``float32`` array respectively.  This is
   further described in the documentation of `numpy.result_type`.


Copyright
---------

This document has been placed in the public domain.
