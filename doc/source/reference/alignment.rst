.. _alignment:

Memory Alignment
================

Numpy Alignment Goals
---------------------

There are three use-cases related to memory alignment in numpy (as of 1.14):

 1. Creating structured datatypes with fields aligned like in a C-struct.
 2. Speeding up copy operations by using uint assignment in instead of memcpy
 3. Guaranteeing safe aligned access for ufuncs/setitem/casting code

Numpy uses two different forms of alignment to achieve these goals:
"True alignment" and "Uint alignment".

"True" alignment refers to the architecture-dependent alignment of an
equivalent C-type in C. For example, in x64 systems ``numpy.float64`` is
equivalent to ``double`` in C. On most systems this has either an alignment of
4 or 8 bytes (and this can be controlled in gcc by the option
``malign-double``).  A variable is aligned in memory if its memory offset is a
multiple of its alignment. On some systems (eg sparc) memory alignment is
required, on others it gives a speedup.

"Uint" alignment depends on the size of a datatype. It is defined to be the
"True alignment" of the uint used by numpy's copy-code to copy the datatype, or
undefined/unaligned if there is no equivalent uint. Currently numpy uses uint8,
uint16, uint32, uint64 and uint64 to copy data of size 1,2,4,8,16 bytes
respectively, and all other sized datatypes cannot be uint-aligned.

For example, on a (typical linux x64 gcc) system, the numpy ``complex64``
datatype is implemented as ``struct { float real, imag; }``. This has "true"
alignment of 4 and "uint" alignment of 8 (equal to the true alignment of
``uint64``).

Variables in Numpy which control and describe alignment
-------------------------------------------------------

There are 4 relevant uses of the word ``align`` used in numpy:

 * The ``dtype.alignment`` attribute (``descr->alignment`` in C). This is meant
   to reflect the "true alignment" of the type. It has arch-dependent default
   values for all datatypes, with the exception of structured types created
   with ``align=True`` as described below.
 * The ``ALIGNED`` flag of an ndarray, computed in ``IsAligned`` and checked
   by ``PyArray_ISALIGNED``. This is computed from ``dtype.alignment``.
   It is set to ``True`` if every item in the array is at a memory location
   consistent with ``dtype.alignment``, which is the case if the data ptr and
   all strides of the array are multiples of that alignment.
 * The ``align`` keyword of the dtype constructor, which only affects structured
   arrays. If the structure's field offsets are not manually provided numpy
   determines offsets automatically. In that case, ``align=True`` pads the
   structure so that each field is "true" aligned in memory and sets
   ``dtype.alignment`` to be the largest of the field "true" alignments. This
   is like what C-structs usually do. Otherwise if offsets or itemsize were
   manually provided ``align=True`` simply checks that all the fields are
   "true" aligned and that the total itemsize is a multiple of the largest
   field alignment. In either case ``dtype.isalignedstruct`` is also set to
   True.
 * ``IsUintAligned`` is used to determine if an ndarray is "uint aligned" in
   an analagous way to how ``IsAligned`` checks for true-alignment.

Consequences of alignment
-------------------------

Here is how the variables above are used:

 1. Creating aligned structs: In order to know how to offset a field when
    ``align=True``, numpy looks up ``field.dtype.alignment``. This includes
    fields which are nested structured arrays.
 2. Ufuncs: If the ``ALIGNED`` flag of an array is False, ufuncs will
    buffer/cast the array before evaluation. This is needed since ufunc inner
    loops access raw elements directly, which might fail on some archs if the
    elements are not true-aligned.
 3. Getitem/setitem/copyswap function: Similar to ufuncs, these functions
    generally have two code paths. If ``ALIGNED`` is False they will
    use a code path that buffers the arguments so they are true-aligned.
 4. Strided copy code: Here, "uint alignment" is used instead.  If the itemsize
    of an array is equal to 1, 2, 4, 8 or 16 bytes and the array is uint
    aligned then instead numpy will do ``*(uintN*)dst) = *(uintN*)src)`` for
    appropriate N. Otherwise numpy copies by doing ``memcpy(dst, src, N)``.
 5. Nditer code: Since this often calls the strided copy code, it must
    check for "uint alignment".
 6. Cast code: if the array is "uint aligned" this will essentially do
    ``*dst = CASTFUNC(*src)``. If not, it does
    ``memmove(srcval, src); dstval = CASTFUNC(srcval); memmove(dst, dstval)``
    where dstval/srcval are aligned.

Note that in principle, only "true alignment" is required for casting code.
However, because the casting code and copy code are deeply intertwined they
both use "uint" alignment. This should be safe assuming uint alignment is
always larger than true alignment, though it can cause unnecessary buffering if
an array is "true aligned" but not "uint aligned". If there is ever a big
rewrite of this code it would be good to allow them to use different
alignments.


