.. currentmodule:: numpy

.. _arrays.maskna:

****************
NA-Masked Arrays
****************

.. versionadded:: 1.7.0

NumPy 1.7 adds preliminary support for missing values using an interface
based on an NA (Not Available) placeholder, implemented as masks in the
core ndarray. This system is highly flexible, allowing NAs to be used
with any underlying dtype, and supports creating multiple views of the same
data with different choices of NAs.

.. note:: The NA API is *experimental*, and may undergo changes in future
   versions of NumPy.  The current implementation based on masks will likely be
   supplemented by a second one based on bit-patterns, and it is possible that
   a difference will be made between missing and ignored data.

Other Missing Data Approaches
=============================

The previous recommended approach for working with missing values was the
:mod:`numpy.ma` module, a subclass of ndarray written purely in Python.
By placing NA-masks directly in the NumPy core, it's possible to avoid
the need for calling "ma.<func>(arr)" instead of "np.<func>(arr)".

Another approach many people have taken is to use NaN as the
placeholder for missing values. There are a few functions
like :func:`numpy.nansum` which behave similarly to usage of the
ufunc.reduce *skipna* parameter.

As experienced in the R language, a programming interface based on an
NA placeholder is generally more intuitive to work with than direct
mask manipulation.

Missing Data Model
==================

The model adopted by NumPy for missing values is that NA is a
placeholder for a value which is there, but is unknown to computations.
The value may be temporarily hidden by the mask, or may be unknown
for any reason, but could be any value the dtype of the array is able
to hold.

This model affects computations in specific, well-defined ways. Any time
we have a computation, like *c = NA + 1*, we must reason about whether
*c* will be an NA or not. The NA is not available now, but maybe a
measurement will be made later to determine what its value is, so anything
we calculate must be consistent with it eventually being revealed. One way
to do this is with thought experiments imagining we have discovered
the value of this NA. If the NA is 0, then *c* is 1. If the NA is
100, then *c* is 101. Because the value of *c* is ambiguous, it
isn't available either, so must be NA as well.

A consequence of separating the NA model from the dtype is that, unlike
in R, NaNs are not considered to be NA. An NA is a value that is completely
unknown, whereas a NaN is usually the result of an invalid computation
as defined in the IEEE 754 floating point arithmetic specification.

Most computations whose input is NA will output NA as well, a property
known as propagation. Some operations, however, always produce the
same result no matter what the value of the NA is. The clearest
example of this is with the logical operations *and* and *or*.  Since both
np.logical_or(True, True) and np.logical_or(False, True) are True,
all possible boolean values on the left hand side produce the
same answer. This means that np.logical_or(np.NA, True) can produce
True instead of the more conservative np.NA. There is a similar case
for np.logical_and.

A similar, but slightly deceptive, example is wanting to treat (NA * 0.0)
as 0.0 instead of as NA. This is invalid because the NA might be Inf
or NaN, in which case the result is NaN instead of 0.0. This idea is
valid for integer dtypes, but NumPy still chooses to return NA because
checking this special case would adversely affect performance.

The NA Object
=============

In the root numpy namespace, there is a new object NA. This is not
the only possible instance of an NA as is the case for None, since an NA
may have a dtype associated with it and has been designed for future
expansion to carry a multi-NA payload. It can be used in computations
like any value::

    >>> np.NA
    NA
    >>> np.NA * 3
    NA(dtype='int64')
    >>> np.sin(np.NA)
    NA(dtype='float64')

To check whether a value is NA, use the :func:`numpy.isna` function::

    >>> np.isna(np.NA)
    True
    >>> np.isna(1.5)
    False
    >>> np.isna(np.nan)
    False
    >>> np.isna(np.NA * 3)
    True
    >>> (np.NA * 3) is np.NA
    False


Creating NA-Masked Arrays
=========================

Because having NA support adds some overhead to NumPy arrays, one
must explicitly request it when creating arrays. There are several ways
to get an NA-masked array. The easiest way is to include an NA
value in the list used to construct the array.::

    >>> a = np.array([1,3,5])
    >>> a
    array([1, 3, 5])
    >>> a.flags.maskna
    False

    >>> b = np.array([1,3,np.NA])
    >>> b
    array([1, 3, NA])
    >>> b.flags.maskna
    True

If one already has an array without an NA-mask, it can be added
by directly setting the *maskna* flag to True. Assigning an NA
to an array without NA support will raise an error rather than
automatically creating an NA-mask, with the idea that supporting
NA should be an explicit user choice.::

    >>> a = np.array([1,3,5])
    >>> a[1] = np.NA
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: Cannot assign NA to an array which does not support NAs
    >>> a.flags.maskna = True
    >>> a[1] = np.NA
    >>> a
    array([1, NA, 5])

Most array construction functions have a new parameter *maskna*, which
can be set to True to produce an array with an NA-mask.::

    >>> np.arange(5., maskna=True)
    array([ 0.,  1.,  2.,  3.,  4.], maskna=True)
    >>> np.eye(3, maskna=True)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]], maskna=True)
    >>> np.array([1,3,5], maskna=True)
    array([1, 3, 5], maskna=True)

Creating NA-Masked Views
========================

It will sometimes be desirable to view an array with an NA-mask, without
adding an NA-mask to that array. This is possible by taking an NA-masked
view of the array. There are two ways to do this, one which simply
guarantees that the view has an NA-mask, and another which guarantees that the
view has its own NA-mask, even if the array already had an NA-mask.

Starting with a non-masked array, we can use the :func:`ndarray.view` method
to get an NA-masked view.::

    >>> a = np.array([1,3,5])
    >>> b = a.view(maskna=True)

    >>> b[2] = np.NA
    >>> a
    array([1, 3, 5])
    >>> b
    array([1, 3, NA])

    >>> b[0] = 2
    >>> a
    array([2, 3, 5])
    >>> b
    array([2, 3, NA])


It is important to be cautious here, though, since if the array already
has a mask, this will also take a view of that mask. This means the original
array's mask will be affected by assigning NA to the view.::

    >>> a = np.array([1,np.NA,5])
    >>> b = a.view(maskna=True)

    >>> b[2] = np.NA
    >>> a
    array([1, NA, NA])
    >>> b
    array([1, NA, NA])

    >>> b[1] = 4
    >>> a
    array([1, 4, NA])
    >>> b
    array([1, 4, NA])


To guarantee that the view created has its own NA-mask, there is another
flag *ownmaskna*. Using this flag will cause a copy of the array's mask
to be created for the view when the array already has a mask.::

    >>> a = np.array([1,np.NA,5])
    >>> b = a.view(ownmaskna=True)

    >>> b[2] = np.NA
    >>> a
    array([1, NA, 5])
    >>> b
    array([1, NA, NA])

    >>> b[1] = 4
    >>> a
    array([1, NA, 5])
    >>> b
    array([1, 4, NA])


In general, when an NA-masked view of an array has been taken, any time
an NA is assigned to an element of the array the data for that element
will remain untouched. This mechanism allows for multiple temporary
views with NAs of the same original array.

NA-Masked Reductions
====================

Many of NumPy's reductions like :func:`numpy.sum` and :func:`numpy.std`
have been extended to work with NA-masked arrays. A consequence of the
missing value model is that any NA value in an array will cause the
output including that value to become NA.::

    >>> a = np.array([[1,2,np.NA,3], [0,np.NA,1,1]])
    >>> a.sum(axis=0)
    array([1, NA, NA, 4])
    >>> a.sum(axis=1)
    array([NA, NA], dtype=int64)

This is not always the desired result, so NumPy includes a parameter
*skipna* which causes the NA values to be skipped during computation.::

    >>> a = np.array([[1,2,np.NA,3], [0,np.NA,1,1]])
    >>> a.sum(axis=0, skipna=True)
    array([1, 2, 1, 4])
    >>> a.sum(axis=1, skipna=True)
    array([6, 2])

Iterating Over NA-Masked Arrays
===============================

The :class:`nditer` object can be used to iterate over arrays with
NA values just like over normal arrays.::

    >>> a = np.array([1,3,np.NA])
    >>> for x in np.nditer(a):
    ...     print x,
    ...
    1 3 NA
    >>> b = np.zeros(3, maskna=True)
    >>> for x, y in np.nditer([a,b], op_flags=[['readonly'],
    ...                                        ['writeonly']]):
    ...     y[...] = -x
    ...
    >>> b
    array([-1., -3.,  NA])

When using the C-API version of the nditer, one must explicitly
add the NPY_ITER_USE_MASKNA flag and take care to deal with the NA
mask appropriately. In the Python exposure, this flag is added
automatically.

Planned Future Additions
========================

The NA support in 1.7 is fairly preliminary, and is focused on getting
the basics solid. This particularly meant getting the API in C refined
to a level where adding NA support to all of NumPy and to third party
software using NumPy would be a reasonable task.

The biggest missing feature within the core is supporting NA values with
structured arrays. The design for this involves a mask slot for each
field in the structured array, motivated by the fact that many important
uses of structured arrays involve treating the structured fields like
another dimension.

Another feature that was discussed during the design process is the ability
to support more than one NA value. The design created supports this multi-NA
idea with the addition of a payload to the NA value and to the NA-mask.
The API has been designed in such a way that adding this feature in a future
release should be possible without changing existing API functions in any way.

To see a more complete list of what is supported and unsupported in the
1.7 release of NumPy, please refer to the release notes.

During the design phase of this feature, two implementation approaches
for NA values were discussed, called "mask" and "bitpattern". What
has been implemented is the "mask" approach, but the design document,
or "NEP", describes a way both approaches could co-operatively exist
in NumPy, since each has both pros and cons. This design document is
available in the file "doc/neps/missing-data.rst" of the NumPy source
code.
