==================================
NEP 11 — Deferred UFunc Evaluation
==================================

:Author: Mark Wiebe <mwwiebe@gmail.com>
:Content-Type: text/x-rst
:Created: 30-Nov-2010
:Status: Deferred

********
Abstract
********

This NEP describes a proposal to add deferred evaluation to NumPy's
UFuncs.  This will allow Python expressions like
"a[:] = b + c + d + e" to be evaluated in a single pass through all
the variables at once, with no temporary arrays.  The resulting
performance will likely be comparable to the *numexpr* library,
but with a more natural syntax.

This idea has some interaction with UFunc error handling and
the UPDATEIFCOPY flag, affecting the design and implementation,
but the result allows for the usage of deferred evaluation
with minimal effort from the Python user's perspective.

**********
Motivation
**********

NumPy's style of UFunc execution causes suboptimal performance for
large expressions, because multiple temporaries are allocated and
the inputs are swept through in multiple passes.  The *numexpr* library
can outperform NumPy for such large expressions, by doing the execution
in small cache-friendly blocks, and evaluating the whole expression
per element.  This results in one sweep through each input, which
is significantly better for the cache.

For an idea of how to get this kind of behavior in NumPy without
changing the Python code, consider the C++ technique of
expression templates. These can be used to quite arbitrarily
rearrange expressions using
vectors or other data structures, example,::

    A = B + C + D;

can be transformed into something equivalent to::

    for(i = 0; i < A.size; ++i) {
        A[i] = B[i] + C[i] + D[i];
    }

This is done by returning a proxy object that knows how to calculate
the result instead of returning the actual object.  With modern C++
optimizing compilers, the resulting machine code is often the same
as hand-written loops.  For an example of this, see the
`Blitz++ Library <http://www.oonumerics.org/blitz/docs/blitz_3.html>`_.
A more recently created library for helping write expression templates
is `Boost Proto <http://beta.boost.org/doc/libs/1_44_0/doc/html/proto.html>`_.

By using the same idea of returning a proxy object in Python, we
can accomplish the same thing dynamically.  The return object is
an ndarray without its buffer allocated, and with enough knowledge
to calculate itself when needed.  When a "deferred array" is
finally evaluated, we can use the expression tree made up of
all the operand deferred arrays, effectively creating a single new
UFunc to evaluate on the fly.


*******************
Example Python Code
*******************

Here's how it might be used in NumPy.::

    # a, b, c are large ndarrays

    with np.deferredstate(True):

        d = a + b + c
        # Now d is a 'deferred array,' a, b, and c are marked READONLY
        # similar to the existing UPDATEIFCOPY mechanism.

        print d
        # Since the value of d was required, it is evaluated so d becomes
        # a regular ndarray and gets printed.

        d[:] = a*b*c
        # Here, the automatically combined "ufunc" that computes
        # a*b*c effectively gets an out= parameter, so no temporary
        # arrays are needed whatsoever.

        e = a+b+c*d
        # Now e is a 'deferred array,' a, b, c, and d are marked READONLY

        d[:] = a
        # d was marked readonly, but the assignment could see that
        # this was due to it being a deferred expression operand.
        # This triggered the deferred evaluation so it could assign
        # the value of a to d.

There may be some surprising behavior, though.::

    with np.deferredstate(True):

        d = a + b + c
        # d is deferred

        e[:] = d
        f[:] = d
        g[:] = d
        # d is still deferred, and its deferred expression
        # was evaluated three times, once for each assignment.
        # This could be detected, with d being converted to
        # a regular ndarray the second time it is evaluated.

I believe the usage that should be recommended in the documentation
is to leave the deferred state at its default, except when
evaluating a large expression that can benefit from it.::

    # calculations

    with np.deferredstate(True):
        x = <big expression>

    # more calculations

This will avoid surprises which would be cause by always keeping
deferred usage True, like floating point warnings or exceptions
at surprising times when deferred expression are used later.
User questions like "Why does my print statement throw a
divide by zero error?" can hopefully be avoided by recommending
this approach.

********************************
Proposed Deferred Evaluation API
********************************

For deferred evaluation to work, the C API needs to be aware of its
existence, and be able to trigger evaluation when necessary.  The
ndarray would gain two new flag.

    ``NPY_ISDEFERRED``

        Indicates the expression evaluation for this ndarray instance
        has been deferred.

    ``NPY_DEFERRED_WASWRITEABLE``

        Can only be set when ``PyArray_GetDeferredUsageCount(arr) > 0``.
        It indicates that when ``arr`` was first used in a deferred
        expression, it was a writeable array.  If this flag is set,
        calling ``PyArray_CalculateAllDeferred()`` will make ``arr``
        writeable again.

.. note:: QUESTION

    Should NPY_DEFERRED and NPY_DEFERRED_WASWRITEABLE be visible
    to Python, or should accessing the flags from python trigger
    PyArray_CalculateAllDeferred if necessary?

The API would be expanded with a number of functions.

``int PyArray_CalculateAllDeferred()``

    This function forces all currently deferred calculations to occur.

    For example, if the error state is set to ignore all, and
    np.seterr({all='raise'}), this would change what happens
    to already deferred expressions.  Thus, all the existing
    deferred arrays should be evaluated before changing the
    error state.

``int PyArray_CalculateDeferred(PyArrayObject* arr)``

    If 'arr' is a deferred array, allocates memory for it and
    evaluates the deferred expression.  If 'arr' is not a deferred
    array, simply returns success.  Returns NPY_SUCCESS or NPY_FAILURE.

``int PyArray_CalculateDeferredAssignment(PyArrayObject* arr, PyArrayObject* out)``

    If 'arr' is a deferred array, evaluates the deferred expression
    into 'out', and 'arr' remains a deferred array.  If 'arr' is not
    a deferred array, copies its value into out.  Returns NPY_SUCCESS
    or NPY_FAILURE.

``int PyArray_GetDeferredUsageCount(PyArrayObject* arr)``

    Returns a count of how many deferred expressions use this array
    as an operand.

The Python API would be expanded as follows.

 ``numpy.setdeferred(state)``

    Enables or disables deferred evaluation. True means to always
    use deferred evaluation.  False means to never use deferred
    evaluation.  None means to use deferred evaluation if the error
    handling state is set to ignore everything.  At NumPy initialization,
    the deferred state is None.

    Returns the previous deferred state.

``numpy.getdeferred()``

    Returns the current deferred state.

``numpy.deferredstate(state)``

    A context manager for deferred state handling, similar to
    ``numpy.errstate``.


Error Handling
==============

Error handling is a thorny issue for deferred evaluation.  If the
NumPy error state is {all='ignore'}, it might be reasonable to
introduce deferred evaluation as the default, however if a UFunc
can raise an error, it would be very strange for the later 'print'
statement to throw the exception instead of the actual operation which
caused the error.

What may be a good approach is to by default enable deferred evaluation
only when the error state is set to ignore all, but allow user control with
'setdeferred' and 'getdeferred' functions.  True would mean always
use deferred evaluation, False would mean never use it, and None would
mean use it only when safe (i.e. the error state is set to ignore all).

Interaction With UPDATEIFCOPY
=============================

The ``NPY_UPDATEIFCOPY`` documentation states:

    The data area represents a (well-behaved) copy whose information
    should be transferred back to the original when this array is deleted.

    This is a special flag that is set if this array represents a copy
    made because a user required certain flags in PyArray_FromAny and a
    copy had to be made of some other array (and the user asked for this
    flag to be set in such a situation). The base attribute then points
    to the “misbehaved” array (which is set read_only). When the array
    with this flag set is deallocated, it will copy its contents back to
    the “misbehaved” array (casting if necessary) and will reset the
    “misbehaved” array to NPY_WRITEABLE. If the “misbehaved” array was
    not NPY_WRITEABLE to begin with then PyArray_FromAny would have
    returned an error because NPY_UPDATEIFCOPY would not have been possible.

The current implementation of UPDATEIFCOPY assumes that it is the only
mechanism mucking with the writeable flag in this manner.  These mechanisms
must be aware of each other to work correctly.  Here's an example of how
they might go wrong:

1. Make a temporary copy of 'arr' with UPDATEIFCOPY ('arr' becomes read only)
2. Use 'arr' in a deferred expression (deferred usage count becomes one,
   NPY_DEFERRED_WASWRITEABLE is **not** set, since 'arr' is read only)
3. Destroy the temporary copy, causing 'arr' to become writeable
4. Writing to 'arr' destroys the value of the deferred expression

To deal with this issue, we make these two states mutually exclusive.

* Usage of UPDATEIFCOPY checks the ``NPY_DEFERRED_WASWRITEABLE`` flag,
  and if it's set, calls ``PyArray_CalculateAllDeferred`` to flush
  all deferred calculation before proceeding.
* The ndarray gets a new flag ``NPY_UPDATEIFCOPY_TARGET`` indicating
  the array will be updated and made writeable at some point in the
  future.  If the deferred evaluation mechanism sees this flag in
  any operand, it triggers immediate evaluation.

Other Implementation Details
============================

When a deferred array is created, it gets references to all the
operands of the UFunc, along with the UFunc itself.  The
'DeferredUsageCount' is incremented for each operand, and later
gets decremented when the deferred expression is calculated or
the deferred array is destroyed.

A global list of weak references to all the deferred arrays
is tracked, in order of creation.  When ``PyArray_CalculateAllDeferred``
gets called, the newest deferred array is calculated first.
This may release references to other deferred arrays contained
in the deferred expression tree, which then
never have to be calculated.

Further Optimization
====================

Instead of conservatively disabling deferred evaluation when any
errors are not set to 'ignore', each UFunc could give a set
of possible errors it generates.  Then, if all those errors
are set to 'ignore', deferred evaluation could be used even
if other errors are not set to ignore.

Once the expression tree is explicitly stored, it is possible to
do transformations on it.  For example add(add(a,b),c) could
be transformed into add3(a,b,c), or add(multiply(a,b),c) could
become fma(a,b,c) using the CPU fused multiply-add instruction
where available.

While I've framed deferred evaluation as just for UFuncs, it could
be extended to other functions, such as dot().  For example, chained
matrix multiplications could be reordered to minimize the size
of intermediates, or peep-hole style optimizer passes could search
for patterns that match optimized BLAS/other high performance
library calls.

For operations on really large arrays, integrating a JIT like LLVM into
this system might be a big benefit.  The UFuncs and other operations
would provide bitcode, which could be inlined together and optimized
by the LLVM optimizers, then executed.  In fact, the iterator itself
could also be represented in bitcode, allowing LLVM to consider
the entire iteration while doing its optimization.
