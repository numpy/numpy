==============================================
NEP 10 — Optimizing Iterator/UFunc Performance
==============================================

:Author: Mark Wiebe <mwwiebe@gmail.com>
:Content-Type: text/x-rst
:Created: 25-Nov-2010
:Status: Final

*****************
Table of Contents
*****************

.. contents::

********
Abstract
********

This NEP proposes to replace the NumPy iterator and multi-iterator
with a single new iterator, designed to be more flexible and allow for
more cache-friendly data access.  The new iterator also subsumes much
of the core ufunc functionality, making it easy to get the current
ufunc benefits in contexts which don't precisely fit the ufunc mold.
Key benefits include:

* automatic reordering to find a cache-friendly access pattern
* standard and customizable broadcasting
* automatic type/byte-order/alignment conversions
* optional buffering to minimize conversion memory usage
* optional output arrays, with automatic allocation when unsupplied
* automatic output or common type selection

A large fraction of this iterator design has already been implemented with
promising results.  Construction overhead is slightly greater (a.flat:
0.5 us, nditer(a): 1.4 us and broadcast(a,b): 1.4 us, nditer([a,b]):
2.2 us), but, as shown in an example, it is already possible to improve
on the performance of the built-in NumPy mechanisms in pure Python code
together with the iterator.  One example rewrites np.add, getting a
four times improvement with some Fortran-contiguous arrays, and
another improves image compositing code from 1.4s to 180ms.

The implementation attempts to take into account
the design decisions made in the NumPy 2.0 refactor, to make its future
integration into libndarray relatively simple.

**********
Motivation
**********

NumPy defaults to returning C-contiguous arrays from UFuncs.  This can
result in extremely poor memory access patterns when dealing with data
that is structured differently.  A simple timing example illustrates
this with a more than eight times performance hit from adding
Fortran-contiguous arrays together.  All timings are done using NumPy
2.0dev (Nov 22, 2010) on an Athlon 64 X2 4200+, with a 64-bit OS.::

    In [1]: import numpy as np
    In [2]: a = np.arange(1000000,dtype=np.float32).reshape(10,10,10,10,10,10)
    In [3]: b, c, d = a.copy(), a.copy(), a.copy()

    In [4]: timeit a+b+c+d
    10 loops, best of 3: 28.5 ms per loop

    In [5]: timeit a.T+b.T+c.T+d.T
    1 loops, best of 3: 237 ms per loop

    In [6]: timeit a.T.ravel('A')+b.T.ravel('A')+c.T.ravel('A')+d.T.ravel('A')
    10 loops, best of 3: 29.6 ms per loop

In this case, it is simple to recover the performance by switching to
a view of the memory, adding, then reshaping back.  To further examine
the problem and see how it isn’t always as trivial to work around,
let’s consider simple code for working with image buffers in NumPy.

Image Compositing Example
=========================

For a more realistic example, consider an image buffer.  Images are
generally stored in a Fortran-contiguous order, and the colour
channel can be treated as either a structured 'RGB' type or an extra
dimension of length three.  The resulting memory layout is neither C-
nor Fortran-contiguous, but is easy to work with directly in NumPy,
because of the flexibility of the ndarray.  This appears ideal, because
it makes the memory layout compatible with typical C or C++ image code,
while simultaneously giving natural access in Python. Getting the color
of pixel (x,y) is just ‘image[x,y]’.

The performance of this layout in NumPy turns out to be very poor.
Here is code which creates two black images, and does an ‘over’
compositing operation on them.::

    In [9]: image1 = np.zeros((1080,1920,3), dtype=np.float32).swapaxes(0,1)
    In [10]: alpha1 = np.zeros((1080,1920,1), dtype=np.float32).swapaxes(0,1)
    In [11]: image2 = np.zeros((1080,1920,3), dtype=np.float32).swapaxes(0,1)
    In [12]: alpha2 = np.zeros((1080,1920,1), dtype=np.float32).swapaxes(0,1)
    In [13]: def composite_over(im1, al1, im2, al2):
       ....:     return (im1 + (1-al1)*im2, al1 + (1-al1)*al2)

    In [14]: timeit composite_over(image1,alpha1,image2,alpha2)
    1 loops, best of 3: 3.51 s per loop

If we give up the convenient layout, and use the C-contiguous default,
the performance is about seven times better.::

    In [16]: image1 = np.zeros((1080,1920,3), dtype=np.float32)
    In [17]: alpha1 = np.zeros((1080,1920,1), dtype=np.float32)
    In [18]: image2 = np.zeros((1080,1920,3), dtype=np.float32)
    In [19]: alpha2 = np.zeros((1080,1920,1), dtype=np.float32)

    In [20]: timeit composite_over(image1,alpha1,image2,alpha2)
    1 loops, best of 3: 581 ms per loop

But this is not all, since it turns out that broadcasting the alpha
channel is exacting a performance price as well.  If we use an alpha
channel with 3 values instead of one, we get::

    In [21]: image1 = np.zeros((1080,1920,3), dtype=np.float32)
    In [22]: alpha1 = np.zeros((1080,1920,3), dtype=np.float32)
    In [23]: image2 = np.zeros((1080,1920,3), dtype=np.float32)
    In [24]: alpha2 = np.zeros((1080,1920,3), dtype=np.float32)

    In [25]: timeit composite_over(image1,alpha1,image2,alpha2)
    1 loops, best of 3: 313 ms per loop

For a final comparison, let’s see how it performs when we use
one-dimensional arrays to ensure just a single loop does the
calculation.::

    In [26]: image1 = np.zeros((1080*1920*3), dtype=np.float32)
    In [27]: alpha1 = np.zeros((1080*1920*3), dtype=np.float32)
    In [28]: image2 = np.zeros((1080*1920*3), dtype=np.float32)
    In [29]: alpha2 = np.zeros((1080*1920*3), dtype=np.float32)

    In [30]: timeit composite_over(image1,alpha1,image2,alpha2)
    1 loops, best of 3: 312 ms per loop

To get a reference performance number, I implemented this simple operation
straightforwardly in C (careful to use the same compile options as NumPy).
If I emulated the memory allocation and layout of the Python code, the
performance was roughly 0.3 seconds, very much in line with NumPy’s
performance.  Combining the operations into one pass reduced the time
to roughly 0.15 seconds.

A slight variation of this example is to use a single memory block
with four channels (1920,1080,4) instead of separate image and alpha.
This is more typical in image processing applications, and here’s how
that looks with a C-contiguous layout.::

    In [31]: image1 = np.zeros((1080,1920,4), dtype=np.float32)
    In [32]: image2 = np.zeros((1080,1920,4), dtype=np.float32)
    In [33]: def composite_over(im1, im2):
       ....:     ret = (1-im1[:,:,-1])[:,:,np.newaxis]*im2
       ....:     ret += im1
       ....:     return ret

    In [34]: timeit composite_over(image1,image2)
    1 loops, best of 3: 481 ms per loop

To see the improvements that implementation of the new iterator as
proposed can produce, go to the example continued after the
proposed API, near the bottom of the document.

*************************
Improving Cache-Coherency
*************************

In order to get the best performance from UFunc calls, the pattern of
memory reads should be as regular as possible. Modern CPUs attempt to
predict the memory read/write pattern and fill the cache ahead of time.
The most predictable pattern is for all the inputs and outputs to be
sequentially processed in the same order.

I propose that by default, the memory layout of the UFunc outputs be as
close to that of the inputs as possible.  Whenever there is an ambiguity
or a mismatch, it defaults to a C-contiguous layout.

To understand how to accomplish this, we first consider the strides of
all the inputs after the shapes have been normalized for broadcasting.
By determining whether a set of strides are compatible and/or ambiguous,
we can determine an output memory layout which maximizes coherency.

In broadcasting, the input shapes are first transformed to broadcast
shapes by prepending singular dimensions, then the broadcast strides
are created, where any singular dimension’s stride is set to zero.

Strides may be negative as well, and in certain cases this can be
normalized to fit the following discussion.  If all the strides for a
particular axis are negative or zero, the strides for that dimension
can be negated after adjusting the base data pointers appropriately.

Here's an example of how three inputs with C-contiguous layouts result in
broadcast strides.  To simplify things, the examples use an itemsize of 1.

==================  ========  =======  =======
Input shapes:       (5,3,7)   (5,3,1)  (1,7)
Broadcast shapes:   (5,3,7)   (5,3,1)  (1,1,7)
Broadcast strides:  (21,7,1)  (3,1,0)  (0,0,1)
==================  ========  =======  =======

*Compatible Strides* - A set of strides are compatible if there exists
a permutation of the axes such that the strides are decreasing for every
stride in the set, excluding entries that are zero.

The example above satisfies the definition with the identity permutation.
In the motivation image example, the strides are slightly different if
we separate the colour and alpha information or not.  The permutation
which demonstrates compatibility here is the transposition (0,1).

=============================  =====================  =====================
Input/Broadcast shapes:        Image (1920, 1080, 3)  Alpha (1920, 1080, 1)
Broadcast strides (separate):  (3,5760,1)             (1,1920,0)
Broadcast strides (together):  (4,7680,1)             (4,7680,0)
=============================  =====================  =====================

*Ambiguous Strides* - A set of compatible strides are ambiguous if
more than one permutation of the axes exists such that the strides are
decreasing for every stride in the set, excluding entries that are zero.

This typically occurs when every axis has a 0-stride somewhere in the
set of strides.  The simplest example is in two dimensions, as follows.

==================  =====  =====
Broadcast shapes:   (1,3)  (5,1)
Broadcast strides:  (0,1)  (1,0)
==================  =====  =====

There may, however, be unambiguous compatible strides without a single
input forcing the entire layout, as in this example:

==================  =======  =======
Broadcast shapes:   (1,3,4)  (5,3,1)
Broadcast strides:  (0,4,1)  (3,1,0)
==================  =======  =======

In the face of ambiguity, we have a choice to either completely throw away
the fact that the strides are compatible, or try to resolve the ambiguity
by adding an additional constraint.  I think the appropriate choice
is to resolve it by picking the memory layout closest to C-contiguous,
but still compatible with the input strides.

Output Layout Selection Algorithm
=================================

The output ndarray memory layout we would like to produce is as follows:

===============================  =============================================
Consistent/Unambiguous strides:  The single consistent layout
Consistent/Ambiguous strides:    The consistent layout closest to C-contiguous
Inconsistent strides:            C-contiguous
===============================  =============================================

Here is pseudo-code for an algorithm to compute the permutation for the
output layout.::

    perm = range(ndim) # Identity, i.e. C-contiguous
    # Insertion sort, ignoring 0-strides
    # Note that the sort must be stable, and 0-strides may
    # be reordered if necessary, but should be moved as little
    # as possible.
    for i0 = 1 to ndim-1:
        # ipos is where perm[i0] will get inserted
        ipos = i0
        j0 = perm[i0]
        for i1 = i0-1 to 0:
            j1 = perm[i1]
            ambig, shouldswap = True, False
            # Check whether any strides are ordered wrong
            for strides in broadcast_strides:
                if strides[j0] != 0 and strides[j1] != 0:
                    if strides[j0] > strides[j1]:
                        # Only set swap if it's still ambiguous.
                        if ambig:
                            shouldswap = True
                    else:
                        # Set swap even if it's not ambiguous,
                        # because not swapping is the choice
                        # for conflicts as well.
                        shouldswap = False
                    ambig = False
            # If there was an unambiguous comparison, either shift ipos
            # to i1 or stop looking for the comparison
            if not ambig:
                if shouldswap:
                    ipos = i1
                else:
                    break
        # Insert perm[i0] into the right place
        if ipos != i0:
           for i1 = i0-1 to ipos:
             perm[i1+1] = perm[i1]
           perm[ipos] = j0
    # perm is now the closest consistent ordering to C-contiguous
    return perm

*********************
Coalescing Dimensions
*********************

In many cases, the memory layout allows for the use of a one-dimensional
loop instead of tracking multiple coordinates within the iterator.
The existing code already exploits this when the data is C-contiguous,
but since we're reordering the axes, we can apply this optimization
more generally.

Once the iteration strides have been sorted to be monotonically
decreasing, any dimensions which could be coalesced are side by side.
If for all the operands, incrementing by strides[i+1] shape[i+1] times
is the same as incrementing by strides[i], or strides[i+1]*shape[i+1] ==
strides[i], dimensions i and i+1 can be coalesced into a single dimension.

Here is pseudo-code for coalescing.::

    # Figure out which pairs of dimensions can be coalesced
    can_coalesce = [False]*ndim
    for strides, shape in zip(broadcast_strides, broadcast_shape):
        for i = 0 to ndim-2:
            if strides[i+1]*shape[i+1] == strides[i]:
                can_coalesce[i] = True
    # Coalesce the types
    new_ndim = ndim - count_nonzero(can_coalesce)
    for strides, shape in zip(broadcast_strides, broadcast_shape):
        j = 0
        for i = 0 to ndim-1:
            # Note that can_coalesce[ndim-1] is always False, so
            # there is no out-of-bounds access here.
            if can_coalesce[i]:
                shape[i+1] = shape[i]*shape[i+1]
            else:
                strides[j] = strides[i]
                shape[j] = shape[i]
                j += 1

*************************
Inner Loop Specialization
*************************

Specialization is handled purely by the inner loop function, so this
optimization is independent of the others.  Some specialization is
already done, like for the reduce operation.  The idea is mentioned in
http://projects.scipy.org/numpy/wiki/ProjectIdeas, “use intrinsics
(SSE-instructions) to speed up low-level loops in NumPy.”

Here are some possibilities for two-argument functions,
covering the important cases of add/subtract/multiply/divide.

* The first or second argument is a single value (i.e. a 0 stride
  value) and does not alias the output.  arr = arr + 1; arr = 1 + arr

  * Can load the constant once instead of reloading it from memory every time

* The strides match the size of the data type. C- or
  Fortran-contiguous data, for example

  * Can do a simple loop without using strides

* The strides match the size of the data type, and they are
  both 16-byte aligned (or differ from 16-byte aligned by the same offset)

  * Can use SSE to process multiple values at once

* The first input and the output are the same single value
  (i.e. a reduction operation).

  * This is already specialized for many UFuncs in the existing code

The above cases are not generally mutually exclusive, for example a
constant argument may be combined with SSE when the strides match the
data type size, and reductions can be optimized with SSE as well.

**********************
Implementation Details
**********************

Except for inner loop specialization, the discussed
optimizations significantly affect ufunc_object.c and the
PyArrayIterObject/PyArrayMultiIterObject used to do the broadcasting.
In general, it should be possible to emulate the current behavior where it
is desired, but I believe the default should be to produce and manipulate
memory layouts which will give the best performance.

To support the new cache-friendly behavior, we introduce a new
option ‘K’ (for “keep”) for any ``order=`` parameter.

The proposed ‘order=’ flags become as follows:

===  =====================================================================================
‘C’  C-contiguous layout
‘F’  Fortran-contiguous layout
‘A’  ‘F’ if the input(s) have a Fortran-contiguous layout, ‘C’ otherwise (“Any Contiguous”)
‘K’  a layout equivalent to ‘C’ followed by some permutation of the axes, as close to the layout of the input(s) as possible (“Keep Layout”)
===  =====================================================================================

Or as an enum::

    /* For specifying array memory layout or iteration order */
    typedef enum {
            /* Fortran order if inputs are all Fortran, C otherwise */
            NPY_ANYORDER=-1,
            /* C order */
            NPY_CORDER=0,
            /* Fortran order */
            NPY_FORTRANORDER=1,
            /* An order as close to the inputs as possible */
            NPY_KEEPORDER=2
    } NPY_ORDER;


Perhaps a good strategy is to first implement the capabilities discussed
here without changing the defaults.  Once they are implemented and
well-tested, the defaults can change from ``order='C'`` to ``order='K'``
everywhere appropriate.  UFuncs additionally should gain an ``order=``
parameter to control the layout of their output(s).

The iterator can do automatic casting, and I have created a sequence
of progressively more permissive casting rules.  Perhaps for 2.0, NumPy
could adopt this enum as its preferred way of dealing with casting.::

    /* For specifying allowed casting in operations which support it */
    typedef enum {
            /* Only allow identical types */
            NPY_NO_CASTING=0,
            /* Allow identical and byte swapped types */
            NPY_EQUIV_CASTING=1,
            /* Only allow safe casts */
            NPY_SAFE_CASTING=2,
            /* Allow safe casts and casts within the same kind */
            NPY_SAME_KIND_CASTING=3,
            /* Allow any casts */
            NPY_UNSAFE_CASTING=4
    } NPY_CASTING;

Iterator Rewrite
================

Based on an analysis of the code, it appears that refactoring the existing
iteration objects to implement these optimizations is prohibitively
difficult.  Additionally, some usage of the iterator requires modifying
internal values or flags, so code using the iterator would have to
change anyway.  Thus we propose creating a new iterator object which
subsumes the existing iterator functionality and expands it to account
for the optimizations.

High level goals for the replacement iterator include:

* Small memory usage and a low number of memory allocations.
* Simple cases (like flat arrays) should have very little overhead.
* Combine single and multiple iteration into one object.

Capabilities that should be provided to user code:

* Iterate in C, Fortran, or “Fastest” (default) order.
* Track a C-style or Fortran-style flat index if requested
  (existing iterator always tracks a C-style index).  This can be done
  independently of the iteration order.
* Track the coordinates if requested (the existing iterator requires
  manually changing an internal iterator flag to guarantee this).
* Skip iteration of the last internal dimension so that it can be
  processed with an inner loop.
* Jump to a specific coordinate in the array.
* Iterate an arbitrary subset of axes (to support, for example, reduce
  with multiple axes at once).
* Ability to automatically allocate output parameters if a NULL input
  is provided,  These outputs should have a memory layout matching
  the iteration order, and are the mechanism for the ``order='K'``
  support.
* Automatic copying and/or buffering of inputs which do not satisfy
  type/byte-order/alignment requirements.  The caller's iteration inner
  loop should be the same no matter what buffering or copying is done.

Notes for implementation:

* User code must never touch the inside of the iterator. This allows
  for drastic changes of the internal memory layout in the future, if
  higher-performance implementation strategies are found.
* Use a function pointer instead of a macro for iteration.
  This way, specializations can be created for the common cases,
  like when ndim is small, for different flag settings, and when the
  number of arrays iterated is small.  Also, an iteration pattern
  can be prescribed that makes a copy of the function pointer first
  to allow the compiler to keep the function pointer
  in a register.
* Dynamically create the memory layout, to minimize the number of
  cache lines taken up by the iterator (for LP64,
  sizeof(PyArrayIterObject) is about 2.5KB, and a binary operation
  like plus needs three of these for the Multi-Iterator).
* Isolate the C-API object from Python reference counting, so that
  it can be used naturally from C.  The Python object then becomes
  a wrapper around the C iterator.  This is analogous to the
  PEP 3118 design separation of Py_buffer and memoryview.

Proposed Iterator Memory Layout
===============================

The following struct describes the iterator memory.  All items
are packed together, which means that different values of the flags,
ndim, and niter will produce slightly different layouts.  ::

    struct {
        /* Flags indicate what optimizations have been applied, and
         * affect the layout of this struct. */
        uint32 itflags;
        /* Number of iteration dimensions.  If FLAGS_HASCOORDS is set,
         * it matches the creation ndim, otherwise it may be smaller.  */
        uint16 ndim;
        /* Number of objects being iterated.  This is fixed at creation time. */
        uint16 niter;

        /* The number of times the iterator will iterate */
        intp itersize;

        /* The permutation is only used when FLAGS_HASCOORDS is set,
         * and is placed here so its position depends on neither ndim
         * nor niter. */
        intp perm[ndim];

        /* The data types of all the operands */
        PyArray_Descr *dtypes[niter];
        /* Backups of the starting axisdata 'ptr' values, to support Reset */
        char *resetdataptr[niter];
        /* Backup of the starting index value, to support Reset */
        npy_intp resetindex;

        /* When the iterator is destroyed, Py_XDECREF is called on all
           these objects */
        PyObject *objects[niter];

        /* Flags indicating read/write status and buffering
         * for each operand. */
        uint8 opitflags[niter];
        /* Padding to make things intp-aligned again */
        uint8 padding[];

        /* If some or all of the inputs are being buffered */
        #if (flags&FLAGS_BUFFERED)
        struct buffer_data {
            /* The size of the buffer, and which buffer we're on.
             * the i-th iteration has i = buffersize*bufferindex+pos
             */
            intp buffersize;
            /* For tracking position inside the buffer */
            intp size, pos;
            /* The strides for the pointers */
            intp stride[niter];
            /* Pointers to the data for the current iterator position.
             * The buffer_data.value ptr[i] equals either
             * axis_data[0].ptr[i] or buffer_data.buffers[i] depending
             * on whether copying to the buffer was necessary.
             */
            char* ptr[niter];
            /* Functions to do the copyswap and casting necessary */
            transferfn_t readtransferfn[niter];
            void *readtransferdata[niter];
            transferfn_t writetransferfn[niter];
            void *writetransferdata[niter];
            /* Pointers to the allocated buffers for operands
             * which the iterator determined needed buffering
             */
            char *buffers[niter];
        };
        #endif /* FLAGS_BUFFERED */

        /* Data per axis, starting with the most-frequently
         * updated, and in decreasing order after that. */
        struct axis_data {
            /* The shape of this axis */
            intp shape;
            /* The current coordinate along this axis */
            intp coord;
            /* The operand and index strides for this axis
            intp stride[niter];
            {intp indexstride;} #if (flags&FLAGS_HASINDEX);
            /* The operand pointers and index values for this axis */
            char* ptr[niter];
            {intp index;} #if (flags&FLAGS_HASINDEX);
        }[ndim];
    };

The array of axis_data structs is ordered to be in increasing rapidity
of increment updates.  If the ``perm`` is the identity, this means it’s
reversed from the C-order.  This is done so data items touched
most often are closest to the beginning of the struct, where the
common properties are, resulting in increased cache coherency.
It also simplifies the iternext call, while making getcoord and
related functions slightly more complicated.

Proposed Iterator API
=====================

The existing iterator API includes functions like PyArrayIter_Check,
PyArray_Iter* and PyArray_ITER_*.  The multi-iterator array includes
PyArray_MultiIter*, PyArray_Broadcast, and PyArray_RemoveSmallest.  The
new iterator design replaces all of this functionality with a single object
and associated API.  One goal of the new API is that all uses of the
existing iterator should be replaceable with the new iterator without
significant effort.

The C-API naming convention chosen is based on the one in the numpy-refactor
branch, where libndarray has the array named ``NpyArray`` and functions
named ``NpyArray_*``.  The iterator is named ``NpyIter`` and functions are
named ``NpyIter_*``.

The Python exposure has the iterator named ``np.nditer``.  One possible
release strategy for this iterator would be to release a 1.X (1.6?) version
with the iterator added, but not used by the NumPy code.  Then, 2.0 can
be release with it fully integrated.  If this strategy is chosen, the
naming convention and API should be finalized as much as possible before
the 1.X release.  The name ``np.iter`` can't be used because it conflicts
with the Python built-in ``iter``.  I would suggest the name ``np.nditer``
within Python, as it is currently unused.

In addition to the performance goals set out for the new iterator,
it appears the API can be refactored to better support some common
NumPy programming idioms.

By moving some functionality currently in the UFunc code into the
iterator, it should make it easier for extension code which wants
to emulate UFunc behavior in cases which don't quite fit the
UFunc paradigm.  In particular, emulating the UFunc buffering behavior
is not a trivial enterprise.

Old -> New Iterator API Conversion
----------------------------------

For the regular iterator:

===============================  =============================================
``PyArray_IterNew``              ``NpyIter_New``
``PyArray_IterAllButAxis``       ``NpyIter_New`` + ``axes`` parameter **or**
                                 Iterator flag ``NPY_ITER_NO_INNER_ITERATION``
``PyArray_BroadcastToShape``     **NOT SUPPORTED** (but could be, if needed)
``PyArrayIter_Check``            Will need to add this in Python exposure
``PyArray_ITER_RESET``           ``NpyIter_Reset``
``PyArray_ITER_NEXT``            Function pointer from ``NpyIter_GetIterNext``
``PyArray_ITER_DATA``            ``NpyIter_GetDataPtrArray``
``PyArray_ITER_GOTO``            ``NpyIter_GotoCoords``
``PyArray_ITER_GOTO1D``          ``NpyIter_GotoIndex``
``PyArray_ITER_NOTDONE``         Return value of ``iternext`` function pointer
===============================  =============================================

For the multi-iterator:

===============================  =============================================
``PyArray_MultiIterNew``         ``NpyIter_MultiNew``
``PyArray_MultiIter_RESET``      ``NpyIter_Reset``
``PyArray_MultiIter_NEXT``       Function pointer from ``NpyIter_GetIterNext``
``PyArray_MultiIter_DATA``       ``NpyIter_GetDataPtrArray``
``PyArray_MultiIter_NEXTi``      **NOT SUPPORTED** (always lock-step iteration)
``PyArray_MultiIter_GOTO``       ``NpyIter_GotoCoords``
``PyArray_MultiIter_GOTO1D``     ``NpyIter_GotoIndex``
``PyArray_MultiIter_NOTDONE``    Return value of ``iternext`` function pointer
``PyArray_Broadcast``            Handled by ``NpyIter_MultiNew``
``PyArray_RemoveSmallest``       Iterator flag ``NPY_ITER_NO_INNER_ITERATION``
===============================  =============================================

For other API calls:

===============================  =============================================
``PyArray_ConvertToCommonType``  Iterator flag ``NPY_ITER_COMMON_DTYPE``
===============================  =============================================


Iterator Pointer Type
---------------------

The iterator structure is internally generated, but a type is still needed
to provide warnings and/or errors when the wrong type is passed to
the API.  We do this with a typedef of an incomplete struct

``typedef struct NpyIter_InternalOnly NpyIter;``


Construction and Destruction
----------------------------

``NpyIter* NpyIter_New(PyArrayObject* op, npy_uint32 flags, NPY_ORDER order, NPY_CASTING casting, PyArray_Descr* dtype, npy_intp a_ndim, npy_intp *axes, npy_intp buffersize)``

    Creates an iterator for the given numpy array object ``op``.

    Flags that may be passed in ``flags`` are any combination
    of the global and per-operand flags documented in
    ``NpyIter_MultiNew``, except for ``NPY_ITER_ALLOCATE``.

    Any of the ``NPY_ORDER`` enum values may be passed to ``order``.  For
    efficient iteration, ``NPY_KEEPORDER`` is the best option, and the other
    orders enforce the particular iteration pattern.

    Any of the ``NPY_CASTING`` enum values may be passed to ``casting``.
    The values include ``NPY_NO_CASTING``, ``NPY_EQUIV_CASTING``,
    ``NPY_SAFE_CASTING``, ``NPY_SAME_KIND_CASTING``, and
    ``NPY_UNSAFE_CASTING``.  To allow the casts to occur, copying or
    buffering must also be enabled.

    If ``dtype`` isn't ``NULL``, then it requires that data type.
    If copying is allowed, it will make a temporary copy if the data
    is castable.  If ``UPDATEIFCOPY`` is enabled, it will also copy
    the data back with another cast upon iterator destruction.

    If ``a_ndim`` is greater than zero, ``axes`` must also be provided.
    In this case, ``axes`` is an ``a_ndim``-sized array of ``op``'s axes.
    A value of -1 in ``axes`` means ``newaxis``. Within the ``axes``
    array, axes may not be repeated.

    If ``buffersize`` is zero, a default buffer size is used,
    otherwise it specifies how big of a buffer to use.  Buffers
    which are powers of 2 such as 512 or 1024 are recommended.

    Returns NULL if there is an error, otherwise returns the allocated
    iterator.

    To make an iterator similar to the old iterator, this should work.::

        iter = NpyIter_New(op, NPY_ITER_READWRITE,
                            NPY_CORDER, NPY_NO_CASTING, NULL, 0, NULL);

    If you want to edit an array with aligned ``double`` code,
    but the order doesn't matter, you would use this.::

        dtype = PyArray_DescrFromType(NPY_DOUBLE);
        iter = NpyIter_New(op, NPY_ITER_READWRITE |
                            NPY_ITER_BUFFERED |
                            NPY_ITER_NBO,
                            NPY_ITER_ALIGNED,
                            NPY_KEEPORDER,
                            NPY_SAME_KIND_CASTING,
                            dtype, 0, NULL);
        Py_DECREF(dtype);

``NpyIter* NpyIter_MultiNew(npy_intp niter, PyArrayObject** op, npy_uint32 flags, NPY_ORDER order, NPY_CASTING casting, npy_uint32 *op_flags, PyArray_Descr** op_dtypes, npy_intp oa_ndim, npy_intp **op_axes, npy_intp buffersize)``

    Creates an iterator for broadcasting the ``niter`` array objects provided
    in ``op``.

    For normal usage, use 0 for ``oa_ndim`` and NULL for ``op_axes``.
    See below for a description of these parameters, which allow for
    custom manual broadcasting as well as reordering and leaving out axes.

    Any of the ``NPY_ORDER`` enum values may be passed to ``order``.  For
    efficient iteration, ``NPY_KEEPORDER`` is the best option, and the other
    orders enforce the particular iteration pattern.  When using
    ``NPY_KEEPORDER``, if you also want to ensure that the iteration is
    not reversed along an axis, you should pass the flag
    ``NPY_ITER_DONT_NEGATE_STRIDES``.

    Any of the ``NPY_CASTING`` enum values may be passed to ``casting``.
    The values include ``NPY_NO_CASTING``, ``NPY_EQUIV_CASTING``,
    ``NPY_SAFE_CASTING``, ``NPY_SAME_KIND_CASTING``, and
    ``NPY_UNSAFE_CASTING``.  To allow the casts to occur, copying or
    buffering must also be enabled.

    If ``op_dtypes`` isn't ``NULL``, it specifies a data type or ``NULL``
    for each ``op[i]``.

    The parameter ``oa_ndim``, when non-zero, specifies the number of
    dimensions that will be iterated with customized broadcasting.
    If it is provided, ``op_axes`` must also be provided.
    These two parameters let you control in detail how the
    axes of the operand arrays get matched together and iterated.
    In ``op_axes``, you must provide an array of ``niter`` pointers
    to ``oa_ndim``-sized arrays of type ``npy_intp``.  If an entry
    in ``op_axes`` is NULL, normal broadcasting rules will apply.
    In ``op_axes[j][i]`` is stored either a valid axis of ``op[j]``, or
    -1 which means ``newaxis``.  Within each ``op_axes[j]`` array, axes
    may not be repeated.  The following example is how normal broadcasting
    applies to a 3-D array, a 2-D array, a 1-D array and a scalar.::

        npy_intp oa_ndim = 3;               /* # iteration axes */
        npy_intp op0_axes[] = {0, 1, 2};    /* 3-D operand */
        npy_intp op1_axes[] = {-1, 0, 1};   /* 2-D operand */
        npy_intp op2_axes[] = {-1, -1, 0};  /* 1-D operand */
        npy_intp op3_axes[] = {-1, -1, -1}  /* 0-D (scalar) operand */
        npy_intp *op_axes[] = {op0_axes, op1_axes, op2_axes, op3_axes};

    If ``buffersize`` is zero, a default buffer size is used,
    otherwise it specifies how big of a buffer to use.  Buffers
    which are powers of 2 such as 512 or 1024 are recommended.

    Returns NULL if there is an error, otherwise returns the allocated
    iterator.

    Flags that may be passed in ``flags``, applying to the whole
    iterator, are:

        ``NPY_ITER_C_INDEX``, ``NPY_ITER_F_INDEX``

            Causes the iterator to track an index matching C or
            Fortran order. These options are mutually exclusive.

        ``NPY_ITER_COORDS``

            Causes the iterator to track array coordinates.
            This prevents the iterator from coalescing axes to
            produce bigger inner loops.

        ``NPY_ITER_NO_INNER_ITERATION``

            Causes the iterator to skip iteration of the innermost
            loop, allowing the user of the iterator to handle it.

            This flag is incompatible with ``NPY_ITER_C_INDEX``,
            ``NPY_ITER_F_INDEX``, and ``NPY_ITER_COORDS``.

        ``NPY_ITER_DONT_NEGATE_STRIDES``

            This only affects the iterator when NPY_KEEPORDER is specified
            for the order parameter.  By default with NPY_KEEPORDER, the
            iterator reverses axes which have negative strides, so that
            memory is traversed in a forward direction.  This disables
            this step.  Use this flag if you want to use the underlying
            memory-ordering of the axes, but don't want an axis reversed.
            This is the behavior of ``numpy.ravel(a, order='K')``, for
            instance.

        ``NPY_ITER_COMMON_DTYPE``

            Causes the iterator to convert all the operands to a common
            data type, calculated based on the ufunc type promotion rules.
            The flags for each operand must be set so that the appropriate
            casting is permitted, and copying or buffering must be enabled.

            If the common data type is known ahead of time, don't use this
            flag.  Instead, set the requested dtype for all the operands.

        ``NPY_ITER_REFS_OK``

            Indicates that arrays with reference types (object
            arrays or structured arrays containing an object type)
            may be accepted and used in the iterator.  If this flag
            is enabled, the caller must be sure to check whether
            ``NpyIter_IterationNeedsAPI(iter)`` is true, in which case
            it may not release the GIL during iteration.

        ``NPY_ITER_ZEROSIZE_OK``

            Indicates that arrays with a size of zero should be permitted.
            Since the typical iteration loop does not naturally work with
            zero-sized arrays, you must check that the IterSize is non-zero
            before entering the iteration loop.

        ``NPY_ITER_REDUCE_OK``

            Permits writeable operands with a dimension with zero
            stride and size greater than one.  Note that such operands
            must be read/write.

            When buffering is enabled, this also switches to a special
            buffering mode which reduces the loop length as necessary to
            not trample on values being reduced.

            Note that if you want to do a reduction on an automatically
            allocated output, you must use ``NpyIter_GetOperandArray``
            to get its reference, then set every value to the reduction
            unit before doing the iteration loop.  In the case of a
            buffered reduction, this means you must also specify the
            flag ``NPY_ITER_DELAY_BUFALLOC``, then reset the iterator
            after initializing the allocated operand to prepare the
            buffers.

        ``NPY_ITER_RANGED``

            Enables support for iteration of sub-ranges of the full
            ``iterindex`` range ``[0, NpyIter_IterSize(iter))``.  Use
            the function ``NpyIter_ResetToIterIndexRange`` to specify
            a range for iteration.

            This flag can only be used with ``NPY_ITER_NO_INNER_ITERATION``
            when ``NPY_ITER_BUFFERED`` is enabled.  This is because
            without buffering, the inner loop is always the size of the
            innermost iteration dimension, and allowing it to get cut up
            would require special handling, effectively making it more
            like the buffered version.

        ``NPY_ITER_BUFFERED``

            Causes the iterator to store buffering data, and use buffering
            to satisfy data type, alignment, and byte-order requirements.
            To buffer an operand, do not specify the ``NPY_ITER_COPY``
            or ``NPY_ITER_UPDATEIFCOPY`` flags, because they will
            override buffering.  Buffering is especially useful for Python
            code using the iterator, allowing for larger chunks
            of data at once to amortize the Python interpreter overhead.

            If used with ``NPY_ITER_NO_INNER_ITERATION``, the inner loop
            for the caller may get larger chunks than would be possible
            without buffering, because of how the strides are laid out.

            Note that if an operand is given the flag ``NPY_ITER_COPY``
            or ``NPY_ITER_UPDATEIFCOPY``, a copy will be made in preference
            to buffering.  Buffering will still occur when the array was
            broadcast so elements need to be duplicated to get a constant
            stride.

            In normal buffering, the size of each inner loop is equal
            to the buffer size, or possibly larger if ``NPY_ITER_GROWINNER``
            is specified.  If ``NPY_ITER_REDUCE_OK`` is enabled and
            a reduction occurs, the inner loops may become smaller depending
            on the structure of the reduction.

        ``NPY_ITER_GROWINNER``

            When buffering is enabled, this allows the size of the inner
            loop to grow when buffering isn't necessary.  This option
            is best used if you're doing a straight pass through all the
            data, rather than anything with small cache-friendly arrays
            of temporary values for each inner loop.

        ``NPY_ITER_DELAY_BUFALLOC``

            When buffering is enabled, this delays allocation of the
            buffers until one of the ``NpyIter_Reset*`` functions is
            called.  This flag exists to avoid wasteful copying of
            buffer data when making multiple copies of a buffered
            iterator for multi-threaded iteration.

            Another use of this flag is for setting up reduction operations.
            After the iterator is created, and a reduction output
            is allocated automatically by the iterator (be sure to use
            READWRITE access), its value may be initialized to the reduction
            unit.  Use ``NpyIter_GetOperandArray`` to get the object.
            Then, call ``NpyIter_Reset`` to allocate and fill the buffers
            with their initial values.

    Flags that may be passed in ``op_flags[i]``, where ``0 <= i < niter``:

        ``NPY_ITER_READWRITE``, ``NPY_ITER_READONLY``, ``NPY_ITER_WRITEONLY``

            Indicate how the user of the iterator will read or write
            to ``op[i]``.  Exactly one of these flags must be specified
            per operand.

        ``NPY_ITER_COPY``

            Allow a copy of ``op[i]`` to be made if it does not
            meet the data type or alignment requirements as specified
            by the constructor flags and parameters.

        ``NPY_ITER_UPDATEIFCOPY``

            Triggers ``NPY_ITER_COPY``, and when an array operand
            is flagged for writing and is copied, causes the data
            in a copy to be copied back to ``op[i]`` when the iterator
            is destroyed.

            If the operand is flagged as write-only and a copy is needed,
            an uninitialized temporary array will be created and then copied
            to back to ``op[i]`` on destruction, instead of doing
            the unnecessary copy operation.

        ``NPY_ITER_NBO``, ``NPY_ITER_ALIGNED``, ``NPY_ITER_CONTIG``

            Causes the iterator to provide data for ``op[i]``
            that is in native byte order, aligned according to
            the dtype requirements, contiguous, or any combination.

            By default, the iterator produces pointers into the
            arrays provided, which may be aligned or unaligned, and
            with any byte order.  If copying or buffering is not
            enabled and the operand data doesn't satisfy the constraints,
            an error will be raised.

            The contiguous constraint applies only to the inner loop,
            successive inner loops may have arbitrary pointer changes.

            If the requested data type is in non-native byte order,
            the NBO flag overrides it and the requested data type is
            converted to be in native byte order.

        ``NPY_ITER_ALLOCATE``

            This is for output arrays, and requires that the flag
            ``NPY_ITER_WRITEONLY`` be set.  If ``op[i]`` is NULL,
            creates a new array with the final broadcast dimensions,
            and a layout matching the iteration order of the iterator.

            When ``op[i]`` is NULL, the requested data type
            ``op_dtypes[i]`` may be NULL as well, in which case it is
            automatically generated from the dtypes of the arrays which
            are flagged as readable.  The rules for generating the dtype
            are the same is for UFuncs.  Of special note is handling
            of byte order in the selected dtype.  If there is exactly
            one input, the input's dtype is used as is.  Otherwise,
            if more than one input dtypes are combined together, the
            output will be in native byte order.

            After being allocated with this flag, the caller may retrieve
            the new array by calling ``NpyIter_GetOperandArray`` and
            getting the i-th object in the returned C array.  The caller
            must call Py_INCREF on it to claim a reference to the array.

        ``NPY_ITER_NO_SUBTYPE``

            For use with ``NPY_ITER_ALLOCATE``, this flag disables
            allocating an array subtype for the output, forcing
            it to be a straight ndarray.

            TODO: Maybe it would be better to introduce a function
            ``NpyIter_GetWrappedOutput`` and remove this flag?

        ``NPY_ITER_NO_BROADCAST``

            Ensures that the input or output matches the iteration
            dimensions exactly.

        ``NPY_ITER_WRITEABLE_REFERENCES``

            By default, the iterator fails on creation if the iterator
            has a writeable operand where the data type involves Python
            references.  Adding this flag indicates that the code using
            the iterator is aware of this possibility and handles it
            correctly.

``NpyIter *NpyIter_Copy(NpyIter *iter)``

    Makes a copy of the given iterator.  This function is provided
    primarily to enable multi-threaded iteration of the data.

    *TODO*: Move this to a section about multithreaded iteration.

    The recommended approach to multithreaded iteration is to
    first create an iterator with the flags
    ``NPY_ITER_NO_INNER_ITERATION``, ``NPY_ITER_RANGED``,
    ``NPY_ITER_BUFFERED``, ``NPY_ITER_DELAY_BUFALLOC``, and
    possibly ``NPY_ITER_GROWINNER``.  Create a copy of this iterator
    for each thread (minus one for the first iterator).  Then, take
    the iteration index range ``[0, NpyIter_GetIterSize(iter))`` and
    split it up into tasks, for example using a TBB parallel_for loop.
    When a thread gets a task to execute, it then uses its copy of
    the iterator by calling ``NpyIter_ResetToIterIndexRange`` and
    iterating over the full range.

    When using the iterator in multi-threaded code or in code not
    holding the Python GIL, care must be taken to only call functions
    which are safe in that context.  ``NpyIter_Copy`` cannot be safely
    called without the Python GIL, because it increments Python
    references.  The ``Reset*`` and some other functions may be safely
    called by passing in the ``errmsg`` parameter as non-NULL, so that
    the functions will pass back errors through it instead of setting
    a Python exception.

``int NpyIter_UpdateIter(NpyIter *iter, npy_intp i, npy_uint32 op_flags, NPY_CASTING casting, PyArray_Descr *dtype)`` **UNIMPLEMENTED**

    Updates the i-th operand within the iterator to possibly have a new
    data type or more restrictive flag attributes.  A use-case for
    this is to allow the automatic allocation to determine an
    output data type based on the standard NumPy type promotion rules,
    then use this function to convert the inputs and possibly the
    automatic output to a different data type during processing.

    This operation can only be done if ``NPY_ITER_COORDS`` was passed
    as a flag to the iterator.  If coordinates are not needed,
    call the function ``NpyIter_RemoveCoords()`` once no more calls to
    ``NpyIter_UpdateIter`` are needed.

    If the i-th operand has already been copied, an error is thrown.  To
    avoid this, leave all the flags out except the read/write indicators
    for any operand that later has ``NpyIter_UpdateIter`` called on it.

    The flags that may be passed in ``op_flags`` are
    ``NPY_ITER_COPY``, ``NPY_ITER_UPDATEIFCOPY``,
    ``NPY_ITER_NBO``, ``NPY_ITER_ALIGNED``, ``NPY_ITER_CONTIG``.

``int NpyIter_RemoveAxis(NpyIter *iter, npy_intp axis)``

    Removes an axis from iteration.  This requires that
    ``NPY_ITER_COORDS`` was set for iterator creation, and does not work
    if buffering is enabled or an index is being tracked. This function
    also resets the iterator to its initial state.

    This is useful for setting up an accumulation loop, for example.
    The iterator can first be created with all the dimensions, including
    the accumulation axis, so that the output gets created correctly.
    Then, the accumulation axis can be removed, and the calculation
    done in a nested fashion.

    **WARNING**: This function may change the internal memory layout of
    the iterator.  Any cached functions or pointers from the iterator
    must be retrieved again!

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.


``int NpyIter_RemoveCoords(NpyIter *iter)``

    If the iterator has coordinates, this strips support for them, and
    does further iterator optimizations that are possible if coordinates
    are not needed.  This function also resets the iterator to its initial
    state.

    **WARNING**: This function may change the internal memory layout of
    the iterator.  Any cached functions or pointers from the iterator
    must be retrieved again!

    After calling this function, ``NpyIter_HasCoords(iter)`` will
    return false.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.

``int NpyIter_RemoveInnerLoop(NpyIter *iter)``

    If UpdateIter/RemoveCoords was used, you may want to specify the
    flag ``NPY_ITER_NO_INNER_ITERATION``.  This flag is not permitted
    together with ``NPY_ITER_COORDS``, so this function is provided
    to enable the feature after ``NpyIter_RemoveCoords`` is called.
    This function also resets the iterator to its initial state.

    **WARNING**: This function changes the internal logic of the iterator.
    Any cached functions or pointers from the iterator must be retrieved
    again!

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.

``int NpyIter_Deallocate(NpyIter *iter)``

    Deallocates the iterator object.  This additionally frees any
    copies made, triggering UPDATEIFCOPY behavior where necessary.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.

``int NpyIter_Reset(NpyIter *iter, char **errmsg)``

    Resets the iterator back to its initial state, at the beginning
    of the iteration range.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.  If errmsg is non-NULL,
    no Python exception is set when ``NPY_FAIL`` is returned.
    Instead, \*errmsg is set to an error message.  When errmsg is
    non-NULL, the function may be safely called without holding
    the Python GIL.

``int NpyIter_ResetToIterIndexRange(NpyIter *iter, npy_intp istart, npy_intp iend, char **errmsg)``

    Resets the iterator and restricts it to the ``iterindex`` range
    ``[istart, iend)``.  See ``NpyIter_Copy`` for an explanation of
    how to use this for multi-threaded iteration.  This requires that
    the flag ``NPY_ITER_RANGED`` was passed to the iterator constructor.

    If you want to reset both the ``iterindex`` range and the base
    pointers at the same time, you can do the following to avoid
    extra buffer copying (be sure to add the return code error checks
    when you copy this code).::

        /* Set to a trivial empty range */
        NpyIter_ResetToIterIndexRange(iter, 0, 0);
        /* Set the base pointers */
        NpyIter_ResetBasePointers(iter, baseptrs);
        /* Set to the desired range */
        NpyIter_ResetToIterIndexRange(iter, istart, iend);

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.  If errmsg is non-NULL,
    no Python exception is set when ``NPY_FAIL`` is returned.
    Instead, \*errmsg is set to an error message.  When errmsg is
    non-NULL, the function may be safely called without holding
    the Python GIL.

``int NpyIter_ResetBasePointers(NpyIter *iter, char **baseptrs, char **errmsg)``

    Resets the iterator back to its initial state, but using the values
    in ``baseptrs`` for the data instead of the pointers from the arrays
    being iterated.  This functions is intended to be used, together with
    the ``op_axes`` parameter, by nested iteration code with two or more
    iterators.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.  If errmsg is non-NULL,
    no Python exception is set when ``NPY_FAIL`` is returned.
    Instead, \*errmsg is set to an error message.  When errmsg is
    non-NULL, the function may be safely called without holding
    the Python GIL.

    *TODO*: Move the following into a special section on nested iterators.

    Creating iterators for nested iteration requires some care.  All
    the iterator operands must match exactly, or the calls to
    ``NpyIter_ResetBasePointers`` will be invalid.  This means that
    automatic copies and output allocation should not be used haphazardly.
    It is possible to still use the automatic data conversion and casting
    features of the iterator by creating one of the iterators with
    all the conversion parameters enabled, then grabbing the allocated
    operands with the ``NpyIter_GetOperandArray`` function and passing
    them into the constructors for the rest of the iterators.

    **WARNING**: When creating iterators for nested iteration,
    the code must not use a dimension more than once in the different
    iterators.  If this is done, nested iteration will produce
    out-of-bounds pointers during iteration.

    **WARNING**: When creating iterators for nested iteration, buffering
    can only be applied to the innermost iterator.  If a buffered iterator
    is used as the source for ``baseptrs``, it will point into a small buffer
    instead of the array and the inner iteration will be invalid.

    The pattern for using nested iterators is as follows.::

        NpyIter *iter1, *iter1;
        NpyIter_IterNext_Fn iternext1, iternext2;
        char **dataptrs1;

        /*
         * With the exact same operands, no copies allowed, and
         * no axis in op_axes used both in iter1 and iter2.
         * Buffering may be enabled for iter2, but not for iter1.
         */
        iter1 = ...; iter2 = ...;

        iternext1 = NpyIter_GetIterNext(iter1);
        iternext2 = NpyIter_GetIterNext(iter2);
        dataptrs1 = NpyIter_GetDataPtrArray(iter1);

        do {
            NpyIter_ResetBasePointers(iter2, dataptrs1);
            do {
                /* Use the iter2 values */
            } while (iternext2(iter2));
        } while (iternext1(iter1));

``int NpyIter_GotoCoords(NpyIter *iter, npy_intp *coords)``

    Adjusts the iterator to point to the ``ndim`` coordinates
    pointed to by ``coords``.  Returns an error if coordinates
    are not being tracked, the coordinates are out of bounds,
    or inner loop iteration is disabled.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.

``int NpyIter_GotoIndex(NpyIter *iter, npy_intp index)``

    Adjusts the iterator to point to the ``index`` specified.
    If the iterator was constructed with the flag
    ``NPY_ITER_C_INDEX``, ``index`` is the C-order index,
    and if the iterator was constructed with the flag
    ``NPY_ITER_F_INDEX``, ``index`` is the Fortran-order
    index.  Returns an error if there is no index being tracked,
    the index is out of bounds, or inner loop iteration is disabled.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.

``npy_intp NpyIter_GetIterSize(NpyIter *iter)``

    Returns the number of elements being iterated.  This is the product
    of all the dimensions in the shape.

``npy_intp NpyIter_GetReduceBlockSizeFactor(NpyIter *iter)`` **UNIMPLEMENTED**

    This provides a factor that must divide into the blocksize used
    for ranged iteration to safely multithread a reduction.  If
    the iterator has no reduction, it returns 1.

    When using ranged iteration to multithread a reduction, there are
    two possible ways to do the reduction:

    If there is a big reduction to a small output, make a temporary
    array initialized to the reduction unit for each thread, then have
    each thread reduce into its temporary.  When that is complete,
    combine the temporaries together.  You can detect this case by
    observing that ``NpyIter_GetReduceBlockSizeFactor`` returns a
    large value, for instance half or a third of ``NpyIter_GetIterSize``.
    You should also check that the output is small just to be sure.

    If there are many small reductions to a big output, and the reduction
    dimensions are inner dimensions, ``NpyIter_GetReduceBlockSizeFactor``
    will return a small number, and as long as the block size you choose
    for multithreading is ``NpyIter_GetReduceBlockSizeFactor(iter)*n``
    for some ``n``, the operation will be safe.

    The bad case is when the a reduction dimension is the outermost
    loop in the iterator.  For example, if you have a C-order
    array with shape (3,1000,1000), and you reduce on dimension 0,
    ``NpyIter_GetReduceBlockSizeFactor`` will return a size equal to
    ``NpyIter_GetIterSize`` for ``NPY_KEEPORDER`` or ``NPY_CORDER``
    iteration orders.  While it is bad for the CPU cache, perhaps
    in the future another order possibility could be provided, maybe
    ``NPY_REDUCEORDER``, which pushes the reduction axes to the inner
    loop, but otherwise is the same as ``NPY_KEEPORDER``.

``npy_intp NpyIter_GetIterIndex(NpyIter *iter)``

    Gets the ``iterindex`` of the iterator, which is an index matching
    the iteration order of the iterator.

``void NpyIter_GetIterIndexRange(NpyIter *iter, npy_intp *istart, npy_intp *iend)``

    Gets the ``iterindex`` sub-range that is being iterated.  If
    ``NPY_ITER_RANGED`` was not specified, this always returns the
    range ``[0, NpyIter_IterSize(iter))``.

``int NpyIter_GotoIterIndex(NpyIter *iter, npy_intp iterindex)``

    Adjusts the iterator to point to the ``iterindex`` specified.
    The IterIndex is an index matching the iteration order of the iterator.
    Returns an error if the ``iterindex`` is out of bounds,
    buffering is enabled, or inner loop iteration is disabled.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.

``int NpyIter_HasInnerLoop(NpyIter *iter``

    Returns 1 if the iterator handles the inner loop,
    or 0 if the caller needs to handle it.  This is controlled
    by the constructor flag ``NPY_ITER_NO_INNER_ITERATION``.

``int NpyIter_HasCoords(NpyIter *iter)``

    Returns 1 if the iterator was created with the
    ``NPY_ITER_COORDS`` flag, 0 otherwise.

``int NpyIter_HasIndex(NpyIter *iter)``

    Returns 1 if the iterator was created with the
    ``NPY_ITER_C_INDEX`` or ``NPY_ITER_F_INDEX``
    flag, 0 otherwise.

``int NpyIter_IsBuffered(NpyIter *iter)``

    Returns 1 if the iterator was created with the
    ``NPY_ITER_BUFFERED`` flag, 0 otherwise.

``int NpyIter_IsGrowInner(NpyIter *iter)``

    Returns 1 if the iterator was created with the
    ``NPY_ITER_GROWINNER`` flag, 0 otherwise.

``npy_intp NpyIter_GetBufferSize(NpyIter *iter)``

    If the iterator is buffered, returns the size of the buffer
    being used, otherwise returns 0.

``npy_intp NpyIter_GetNDim(NpyIter *iter)``

    Returns the number of dimensions being iterated.  If coordinates
    were not requested in the iterator constructor, this value
    may be smaller than the number of dimensions in the original
    objects.

``npy_intp NpyIter_GetNIter(NpyIter *iter)``

    Returns the number of objects being iterated.

``npy_intp *NpyIter_GetAxisStrideArray(NpyIter *iter, npy_intp axis)``

    Gets the array of strides for the specified axis. Requires that
    the iterator be tracking coordinates, and that buffering not
    be enabled.

    This may be used when you want to match up operand axes in
    some fashion, then remove them with ``NpyIter_RemoveAxis`` to
    handle their processing manually.  By calling this function
    before removing the axes, you can get the strides for the
    manual processing.

    Returns ``NULL`` on error.

``int NpyIter_GetShape(NpyIter *iter, npy_intp *outshape)``

    Returns the broadcast shape of the iterator in ``outshape``.
    This can only be called on an iterator which supports coordinates.

    Returns ``NPY_SUCCEED`` or ``NPY_FAIL``.

``PyArray_Descr **NpyIter_GetDescrArray(NpyIter *iter)``

    This gives back a pointer to the ``niter`` data type Descrs for
    the objects being iterated.  The result points into ``iter``,
    so the caller does not gain any references to the Descrs.

    This pointer may be cached before the iteration loop, calling
    ``iternext`` will not change it.

``PyObject **NpyIter_GetOperandArray(NpyIter *iter)``

    This gives back a pointer to the ``niter`` operand PyObjects
    that are being iterated.  The result points into ``iter``,
    so the caller does not gain any references to the PyObjects.

``PyObject *NpyIter_GetIterView(NpyIter *iter, npy_intp i)``

    This gives back a reference to a new ndarray view, which is a view
    into the i-th object in the array ``NpyIter_GetOperandArray()``,
    whose dimensions and strides match the internal optimized
    iteration pattern.  A C-order iteration of this view is equivalent
    to the iterator's iteration order.

    For example, if an iterator was created with a single array as its
    input, and it was possible to rearrange all its axes and then
    collapse it into a single strided iteration, this would return
    a view that is a one-dimensional array.

``void NpyIter_GetReadFlags(NpyIter *iter, char *outreadflags)``

    Fills ``niter`` flags. Sets ``outreadflags[i]`` to 1 if
    ``op[i]`` can be read from, and to 0 if not.

``void NpyIter_GetWriteFlags(NpyIter *iter, char *outwriteflags)``

    Fills ``niter`` flags. Sets ``outwriteflags[i]`` to 1 if
    ``op[i]`` can be written to, and to 0 if not.

Functions For Iteration
-----------------------

``NpyIter_IterNext_Fn NpyIter_GetIterNext(NpyIter *iter, char **errmsg)``

    Returns a function pointer for iteration.  A specialized version
    of the function pointer may be calculated by this function
    instead of being stored in the iterator structure. Thus, to
    get good performance, it is required that the function pointer
    be saved in a variable rather than retrieved for each loop iteration.

    Returns NULL if there is an error.  If errmsg is non-NULL,
    no Python exception is set when ``NPY_FAIL`` is returned.
    Instead, \*errmsg is set to an error message.  When errmsg is
    non-NULL, the function may be safely called without holding
    the Python GIL.

    The typical looping construct is as follows.::

        NpyIter_IterNext_Fn iternext = NpyIter_GetIterNext(iter, NULL);
        char **dataptr = NpyIter_GetDataPtrArray(iter);

        do {
            /* use the addresses dataptr[0], ... dataptr[niter-1] */
        } while(iternext(iter));

    When ``NPY_ITER_NO_INNER_ITERATION`` is specified, the typical
    inner loop construct is as follows.::

        NpyIter_IterNext_Fn iternext = NpyIter_GetIterNext(iter, NULL);
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *stride = NpyIter_GetInnerStrideArray(iter);
        npy_intp *size_ptr = NpyIter_GetInnerLoopSizePtr(iter), size;
        npy_intp iiter, niter = NpyIter_GetNIter(iter);

        do {
            size = *size_ptr;
            while (size--) {
                /* use the addresses dataptr[0], ... dataptr[niter-1] */
                for (iiter = 0; iiter < niter; ++iiter) {
                    dataptr[iiter] += stride[iiter];
                }
            }
        } while (iternext());

    Observe that we are using the dataptr array inside the iterator, not
    copying the values to a local temporary.  This is possible because
    when ``iternext()`` is called, these pointers will be overwritten
    with fresh values, not incrementally updated.

    If a compile-time fixed buffer is being used (both flags
    ``NPY_ITER_BUFFERED`` and ``NPY_ITER_NO_INNER_ITERATION``), the
    inner size may be used as a signal as well.  The size is guaranteed
    to become zero when ``iternext()`` returns false, enabling the
    following loop construct.  Note that if you use this construct,
    you should not pass ``NPY_ITER_GROWINNER`` as a flag, because it
    will cause larger sizes under some circumstances.::

        /* The constructor should have buffersize passed as this value */
        #define FIXED_BUFFER_SIZE 1024

        NpyIter_IterNext_Fn iternext = NpyIter_GetIterNext(iter, NULL);
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *stride = NpyIter_GetInnerStrideArray(iter);
        npy_intp *size_ptr = NpyIter_GetInnerLoopSizePtr(iter), size;
        npy_intp i, iiter, niter = NpyIter_GetNIter(iter);

        /* One loop with a fixed inner size */
        size = *size_ptr;
        while (size == FIXED_BUFFER_SIZE) {
            /*
             * This loop could be manually unrolled by a factor
             * which divides into FIXED_BUFFER_SIZE
             */
            for (i = 0; i < FIXED_BUFFER_SIZE; ++i) {
                /* use the addresses dataptr[0], ... dataptr[niter-1] */
                for (iiter = 0; iiter < niter; ++iiter) {
                    dataptr[iiter] += stride[iiter];
                }
            }
            iternext();
            size = *size_ptr;
        }

        /* Finish-up loop with variable inner size */
        if (size > 0) do {
            size = *size_ptr;
            while (size--) {
                /* use the addresses dataptr[0], ... dataptr[niter-1] */
                for (iiter = 0; iiter < niter; ++iiter) {
                    dataptr[iiter] += stride[iiter];
                }
            }
        } while (iternext());

``NpyIter_GetCoords_Fn NpyIter_GetGetCoords(NpyIter *iter, char **errmsg)``

    Returns a function pointer for getting the coordinates
    of the iterator.  Returns NULL if the iterator does not
    support coordinates.  It is recommended that this function
    pointer be cached in a local variable before the iteration
    loop.

    Returns NULL if there is an error.  If errmsg is non-NULL,
    no Python exception is set when ``NPY_FAIL`` is returned.
    Instead, \*errmsg is set to an error message.  When errmsg is
    non-NULL, the function may be safely called without holding
    the Python GIL.

``char **NpyIter_GetDataPtrArray(NpyIter *iter)``

    This gives back a pointer to the ``niter`` data pointers.  If
    ``NPY_ITER_NO_INNER_ITERATION`` was not specified, each data
    pointer points to the current data item of the iterator.  If
    no inner iteration was specified, it points to the first data
    item of the inner loop.

    This pointer may be cached before the iteration loop, calling
    ``iternext`` will not change it.  This function may be safely
    called without holding the Python GIL.

``npy_intp *NpyIter_GetIndexPtr(NpyIter *iter)``

    This gives back a pointer to the index being tracked, or NULL
    if no index is being tracked.  It is only useable if one of
    the flags ``NPY_ITER_C_INDEX`` or ``NPY_ITER_F_INDEX``
    were specified during construction.

When the flag ``NPY_ITER_NO_INNER_ITERATION`` is used, the code
needs to know the parameters for doing the inner loop.  These
functions provide that information.

``npy_intp *NpyIter_GetInnerStrideArray(NpyIter *iter)``

    Returns a pointer to an array of the ``niter`` strides,
    one for each iterated object, to be used by the inner loop.

    This pointer may be cached before the iteration loop, calling
    ``iternext`` will not change it. This function may be safely
    called without holding the Python GIL.

``npy_intp* NpyIter_GetInnerLoopSizePtr(NpyIter *iter)``

    Returns a pointer to the number of iterations the
    inner loop should execute.

    This address may be cached before the iteration loop, calling
    ``iternext`` will not change it.  The value itself may change during
    iteration, in particular if buffering is enabled.  This function
    may be safely called without holding the Python GIL.

``void NpyIter_GetInnerFixedStrideArray(NpyIter *iter, npy_intp *out_strides)``

    Gets an array of strides which are fixed, or will not change during
    the entire iteration.  For strides that may change, the value
    NPY_MAX_INTP is placed in the stride.

    Once the iterator is prepared for iteration (after a reset if
    ``NPY_DELAY_BUFALLOC`` was used), call this to get the strides
    which may be used to select a fast inner loop function.  For example,
    if the stride is 0, that means the inner loop can always load its
    value into a variable once, then use the variable throughout the loop,
    or if the stride equals the itemsize, a contiguous version for that
    operand may be used.

    This function may be safely called without holding the Python GIL.

Examples
--------

A copy function using the iterator.  The ``order`` parameter
is used to control the memory layout of the allocated
result.

If the input is a reference type, this function will fail.
To fix this, the code must be changed to specially handle writeable
references, and add ``NPY_ITER_WRITEABLE_REFERENCES`` to the flags.::

    /* NOTE: This code has not been compiled/tested */
    PyObject *CopyArray(PyObject *arr, NPY_ORDER order)
    {
        NpyIter *iter;
        NpyIter_IterNext_Fn iternext;
        PyObject *op[2], *ret;
        npy_uint32 flags;
        npy_uint32 op_flags[2];
        npy_intp itemsize, *innersizeptr, innerstride;
        char **dataptrarray;

        /*
         * No inner iteration - inner loop is handled by CopyArray code
         */
        flags = NPY_ITER_NO_INNER_ITERATION;
        /*
         * Tell the constructor to automatically allocate the output.
         * The data type of the output will match that of the input.
         */
        op[0] = arr;
        op[1] = NULL;
        op_flags[0] = NPY_ITER_READONLY;
        op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

        /* Construct the iterator */
        iter = NpyIter_MultiNew(2, op, flags, order, NPY_NO_CASTING,
                                op_flags, NULL, 0, NULL);
        if (iter == NULL) {
            return NULL;
        }

        /*
         * Make a copy of the iternext function pointer and
         * a few other variables the inner loop needs.
         */
        iternext = NpyIter_GetIterNext(iter);
        innerstride = NpyIter_GetInnerStrideArray(iter)[0];
        itemsize = NpyIter_GetDescrArray(iter)[0]->elsize;
        /*
         * The inner loop size and data pointers may change during the
         * loop, so just cache the addresses.
         */
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        dataptrarray = NpyIter_GetDataPtrArray(iter);

        /*
         * Note that because the iterator allocated the output,
         * it matches the iteration order and is packed tightly,
         * so we don't need to check it like the input.
         */
        if (innerstride == itemsize) {
            do {
                memcpy(dataptrarray[1], dataptrarray[0],
                                        itemsize * (*innersizeptr));
            } while (iternext(iter));
        } else {
            /* Should specialize this further based on item size... */
            npy_intp i;
            do {
                npy_intp size = *innersizeptr;
                char *src = dataaddr[0], *dst = dataaddr[1];
                for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {
                    memcpy(dst, src, itemsize);
                }
            } while (iternext(iter));
        }

        /* Get the result from the iterator object array */
        ret = NpyIter_GetOperandArray(iter)[1];
        Py_INCREF(ret);

        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            return NULL;
        }

        return ret;
    }

Python Lambda UFunc Example
---------------------------

To show how the new iterator allows the definition of efficient UFunc-like
functions in pure Python, we demonstrate the function ``luf``, which
makes a lambda-expression act like a UFunc.  This is very similar to the
``numexpr`` library, but only takes a few lines of code.

First, here is the definition of the ``luf`` function.::

    def luf(lamdaexpr, *args, **kwargs):
        """Lambda UFunc

            e.g.
            c = luf(lambda i,j:i+j, a, b, order='K',
                                casting='safe', buffersize=8192)

            c = np.empty(...)
            luf(lambda i,j:i+j, a, b, out=c, order='K',
                                casting='safe', buffersize=8192)
        """

        nargs = len(args)
        op = args + (kwargs.get('out',None),)
        it = np.nditer(op, ['buffered','no_inner_iteration'],
                [['readonly','nbo_aligned']]*nargs +
                                [['writeonly','allocate','no_broadcast']],
                order=kwargs.get('order','K'),
                casting=kwargs.get('casting','safe'),
                buffersize=kwargs.get('buffersize',0))
        while not it.finished:
            it[-1] = lamdaexpr(*it[:-1])
            it.iternext()

        return it.operands[-1]

Then, by using ``luf`` instead of straight Python expressions, we
can gain some performance from better cache behavior.::

    In [2]: a = np.random.random((50,50,50,10))
    In [3]: b = np.random.random((50,50,1,10))
    In [4]: c = np.random.random((50,50,50,1))

    In [5]: timeit 3*a+b-(a/c)
    1 loops, best of 3: 138 ms per loop

    In [6]: timeit luf(lambda a,b,c:3*a+b-(a/c), a, b, c)
    10 loops, best of 3: 60.9 ms per loop

    In [7]: np.all(3*a+b-(a/c) == luf(lambda a,b,c:3*a+b-(a/c), a, b, c))
    Out[7]: True


Python Addition Example
-----------------------

The iterator has been mostly written and exposed to Python.  To
see how it behaves, let's see what we can do with the np.add ufunc.
Even without changing the core of NumPy, we will be able to use
the iterator to make a faster add function.

The Python exposure supplies two iteration interfaces, one which
follows the Python iterator protocol, and another which mirrors the
C-style do-while pattern.  The native Python approach is better
in most cases, but if you need the iterator's coordinates or
index, use the C-style pattern.

Here is how we might write an ``iter_add`` function, using the
Python iterator protocol.::

    def iter_add_py(x, y, out=None):
        addop = np.add

        it = np.nditer([x,y,out], [],
                    [['readonly'],['readonly'],['writeonly','allocate']])

        for (a, b, c) in it:
            addop(a, b, c)

        return it.operands[2]

Here is the same function, but following the C-style pattern.::

    def iter_add(x, y, out=None):
        addop = np.add

        it = np.nditer([x,y,out], [],
                    [['readonly'],['readonly'],['writeonly','allocate']])

        while not it.finished:
            addop(it[0], it[1], it[2])
            it.iternext()

        return it.operands[2]

Some noteworthy points about this function:

* Cache np.add as a local variable to reduce namespace lookups
* Inputs are readonly, output is writeonly, and will be allocated
  automatically if it is None.
* Uses np.add's out parameter to avoid an extra copy.

Let's create some test variables, and time this function as well as the
built-in np.add.::

    In [1]: a = np.arange(1000000,dtype='f4').reshape(100,100,100)
    In [2]: b = np.arange(10000,dtype='f4').reshape(1,100,100)
    In [3]: c = np.arange(10000,dtype='f4').reshape(100,100,1)

    In [4]: timeit iter_add(a, b)
    1 loops, best of 3: 7.03 s per loop

    In [5]: timeit np.add(a, b)
    100 loops, best of 3: 6.73 ms per loop

At a thousand times slower, this is clearly not very good.  One feature
of the iterator, designed to help speed up the inner loops, is the flag
``no_inner_iteration``.  This is the same idea as the old iterator's
``PyArray_IterAllButAxis``, but slightly smarter.  Let's modify
``iter_add`` to use this feature.::

    def iter_add_noinner(x, y, out=None):
        addop = np.add

        it = np.nditer([x,y,out], ['no_inner_iteration'],
                    [['readonly'],['readonly'],['writeonly','allocate']])

        for (a, b, c) in it:
            addop(a, b, c)

        return it.operands[2]

The performance improves dramatically.::

    In[6]: timeit iter_add_noinner(a, b)
    100 loops, best of 3: 7.1 ms per loop

The performance is basically as good as the built-in function!  It
turns out this is because the iterator was able to coalesce the last two
dimensions, resulting in 100 adds of 10000 elements each.  If the
inner loop doesn't become as large, the performance doesn't improve
as dramatically.  Let's use ``c`` instead of ``b`` to see how this works.::

    In[7]: timeit iter_add_noinner(a, c)
    10 loops, best of 3: 76.4 ms per loop

It's still a lot better than seven seconds, but still over ten times worse
than the built-in function.  Here, the inner loop has 100 elements,
and it's iterating 10000 times.  If we were coding in C, our performance
would already be as good as the built-in performance, but in Python
there is too much overhead.

This leads us to another feature of the iterator, its ability to give
us views of the iterated memory.  The views it gives us are structured
so that processing them in C-order, like the built-in NumPy code does,
gives the same access order as the iterator itself.  Effectively, we
are using the iterator to solve for a good memory access pattern, then
using other NumPy machinery to efficiently execute it.  Let's
modify ``iter_add`` once again.::

    def iter_add_itview(x, y, out=None):
        it = np.nditer([x,y,out], [],
                    [['readonly'],['readonly'],['writeonly','allocate']])

        (a, b, c) = it.itviews
        np.add(a, b, c)

        return it.operands[2]

Now the performance pretty closely matches the built-in function's.::

    In [8]: timeit iter_add_itview(a, b)
    100 loops, best of 3: 6.18 ms per loop

    In [9]: timeit iter_add_itview(a, c)
    100 loops, best of 3: 6.69 ms per loop

Let us now step back to a case similar to the original motivation for the
new iterator.  Here are the same calculations in Fortran memory order instead
Of C memory order.::

    In [10]: a = np.arange(1000000,dtype='f4').reshape(100,100,100).T
    In [12]: b = np.arange(10000,dtype='f4').reshape(100,100,1).T
    In [11]: c = np.arange(10000,dtype='f4').reshape(1,100,100).T

    In [39]: timeit np.add(a, b)
    10 loops, best of 3: 34.3 ms per loop

    In [41]: timeit np.add(a, c)
    10 loops, best of 3: 31.6 ms per loop

    In [44]: timeit iter_add_itview(a, b)
    100 loops, best of 3: 6.58 ms per loop

    In [43]: timeit iter_add_itview(a, c)
    100 loops, best of 3: 6.33 ms per loop

As you can see, the performance of the built-in function dropped
significantly, but our newly-written add function maintained essentially
the same performance.  As one final test, let's try several adds chained
together.::

    In [4]: timeit np.add(np.add(np.add(a,b), c), a)
    1 loops, best of 3: 99.5 ms per loop

    In [9]: timeit iter_add_itview(iter_add_itview(iter_add_itview(a,b), c), a)
    10 loops, best of 3: 29.3 ms per loop

Also, just to check that it's doing the same thing,::

    In [22]: np.all(
       ....: iter_add_itview(iter_add_itview(iter_add_itview(a,b), c), a) ==
       ....: np.add(np.add(np.add(a,b), c), a)
       ....: )

    Out[22]: True

Image Compositing Example Revisited
-----------------------------------

For motivation, we had an example that did an 'over' composite operation
on two images.  Now let's see how we can write the function with
the new iterator.

Here is one of the original functions, for reference, and some
random image data.::

    In [5]: rand1 = np.random.random_sample(1080*1920*4).astype(np.float32)
    In [6]: rand2 = np.random.random_sample(1080*1920*4).astype(np.float32)
    In [7]: image1 = rand1.reshape(1080,1920,4).swapaxes(0,1)
    In [8]: image2 = rand2.reshape(1080,1920,4).swapaxes(0,1)

    In [3]: def composite_over(im1, im2):
      ....:     ret = (1-im1[:,:,-1])[:,:,np.newaxis]*im2
      ....:     ret += im1
      ....:     return ret

    In [4]: timeit composite_over(image1,image2)
    1 loops, best of 3: 1.39 s per loop

Here's the same function, rewritten to use a new iterator.  Note how
easy it was to add an optional output parameter.::

    In [5]: def composite_over_it(im1, im2, out=None, buffersize=4096):
      ....:     it = np.nditer([im1, im1[:,:,-1], im2, out],
      ....:                     ['buffered','no_inner_iteration'],
      ....:                     [['readonly']]*3+[['writeonly','allocate']],
      ....:                     op_axes=[None,[0,1,np.newaxis],None,None],
      ....:                     buffersize=buffersize)
      ....:     while not it.finished:
      ....:         np.multiply(1-it[1], it[2], it[3])
      ....:         it[3] += it[0]
      ....:         it.iternext()
      ....:     return it.operands[3]

    In [6]: timeit composite_over_it(image1, image2)
    1 loops, best of 3: 197 ms per loop

A big speed improvement, over even the best previous attempt using
straight NumPy and a C-order array!  By playing with the buffer size, we can
see how the speed improves until we hit the limits of the CPU cache
in the inner loop.::

    In [7]: timeit composite_over_it(image1, image2, buffersize=2**7)
    1 loops, best of 3: 1.23 s per loop

    In [8]: timeit composite_over_it(image1, image2, buffersize=2**8)
    1 loops, best of 3: 699 ms per loop

    In [9]: timeit composite_over_it(image1, image2, buffersize=2**9)
    1 loops, best of 3: 418 ms per loop

    In [10]: timeit composite_over_it(image1, image2, buffersize=2**10)
    1 loops, best of 3: 287 ms per loop

    In [11]: timeit composite_over_it(image1, image2, buffersize=2**11)
    1 loops, best of 3: 225 ms per loop

    In [12]: timeit composite_over_it(image1, image2, buffersize=2**12)
    1 loops, best of 3: 194 ms per loop

    In [13]: timeit composite_over_it(image1, image2, buffersize=2**13)
    1 loops, best of 3: 180 ms per loop

    In [14]: timeit composite_over_it(image1, image2, buffersize=2**14)
    1 loops, best of 3: 192 ms per loop

    In [15]: timeit composite_over_it(image1, image2, buffersize=2**15)
    1 loops, best of 3: 280 ms per loop

    In [16]: timeit composite_over_it(image1, image2, buffersize=2**16)
    1 loops, best of 3: 328 ms per loop

    In [17]: timeit composite_over_it(image1, image2, buffersize=2**17)
    1 loops, best of 3: 345 ms per loop

And finally, to double check that it's working, we can compare the two
functions.::

    In [18]: np.all(composite_over(image1, image2) ==
        ...:        composite_over_it(image1, image2))
    Out[18]: True

Image Compositing With NumExpr
------------------------------

As a test of the iterator, numexpr has been enhanced to allow use of
the iterator instead of its internal broadcasting code.  First, let's
implement the composite operation with numexpr.::

    In [22]: def composite_over_ne(im1, im2, out=None):
       ....:     ima = im1[:,:,-1][:,:,np.newaxis]
       ....:     return ne.evaluate("im1+(1-ima)*im2")

    In [23]: timeit composite_over_ne(image1,image2)
    1 loops, best of 3: 1.25 s per loop

This beats the straight NumPy operation, but isn't very good.  Switching
to the iterator version of numexpr, we get a big improvement over the
straight Python function using the iterator.  Note that this is on
a dual core machine.::

    In [29]: def composite_over_ne_it(im1, im2, out=None):
       ....:     ima = im1[:,:,-1][:,:,np.newaxis]
       ....:     return ne.evaluate_iter("im1+(1-ima)*im2")

    In [30]: timeit composite_over_ne_it(image1,image2)
    10 loops, best of 3: 67.2 ms per loop

    In [31]: ne.set_num_threads(1)
    In [32]: timeit composite_over_ne_it(image1,image2)
    10 loops, best of 3: 91.1 ms per loop
