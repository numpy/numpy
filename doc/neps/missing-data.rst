:Title: Missing Data Functionality in NumPy
:Author: Mark Wiebe <mwwiebe@gmail.com>
:Date: 2011-06-23

*****************
Table of Contents
*****************

.. contents::

********
Abstract
********

Users interested in dealing with missing data within NumPy are generally
pointed to the masked array subclass of the ndarray, generally known
as 'numpy.ma'. This class has a number of users who depend strongly
on its capabilities, but people who are accustomed to the deep integration
of the missing data placeholder "NA" in the R project and others who
find the programming interface challenging or inconsistent tend not
to use it.

This NEP proposes to integrate a mask-based missing data solution
into NumPy, with an additional NA bit pattern-based missing data solution
that can be implemented  concurrently or later which would integrate seamlessly
with the mask-based solution.

The mask-based solution and the NA bit pattern-based solutions in this
proposal offer the exact same missing value abstraction, with several
differences in performance, memory overhead, and flexibility.

The mask-based solution is more flexible, supporting all behaviors of the
NA bit pattern-based solution, but leaving the hidden values untouched
whenever an element is masked.

The NA bit pattern-based solution requires less memory, is bit-level
compatible with the 64-bit floating point representation used in R, but
does not preserve the hidden values and in fact requires stealing at
least one bit pattern from the underlying dtype to represent the missing
value NA.

Both solutions are generic in the sense that they can be used with
custom data types very easily, with no effort in the case of the masked
solution, and with the requirement that a bit pattern to sacrifice be
chosen in the case of the NA bit pattern solution.

**************************
Definition of Missing Data
**************************

Unknown Yet Existing Data
=========================

In order to be able to develop an intuition about what computation
will be done by various NumPy functions, a consistent conceptual
model of what a missing element means must be applied. The approach
taken in the R project is to define a missing element as something which
does have a valid value, but that value is unknown. This proposal
adopts this behavior as as the default for all operations involving
missing values.

In this interpretation, nearly any computation with a missing input produces
a missing output. For example, 'sum(a)' would produce a missing value
if 'a' contained just one missing element. When the output value does
not depend on one of the inputs, it is reasonable to output a value
that is not NA, such as logical_and(NA, False) == False.

Some more complex arithmetic operations, such as matrix products, are
well defined with this interpretation, and the result should be
the same as is the missing values were NaNs. Actually implementing
such things to the theoretical limit is probably not worth it,
and in many cases either raising an exception or returning all
missing values may be preferred to doing precise calculations.
Care must be taken here when dealing with the values and the masks,
to preserve the semantics that masking a value never touches
the element's backing memory.

Data That Doesn't Exist
=======================

Another useful interpretation is that the missing elements should be
treated as if they didn't exist in the array, and the operation should
do its best to interpret what that means according to the data
that's left. In this case, 'mean(a)' would compute the mean of just
the values that are unmasked, adjusting both the sum and count it
uses based on the mask.

This kind of data can arise when conforming sparsely sampled data
into a regular sampling pattern, and is a useful interpretation so 
use when attempting to get best-guess answers for many statistical queries.

In R, many functions take a parameter "na.rm=T" which means to treat
the data as if the NA values are not part of the data set. This proposal
defines a standard parameter "skipmissing=True" for this same purpose. 

Data That Is Being Temporarily Ignored
======================================

It can be useful to temporarily treat some array elements as if they
were NA, possibly in many different configurations. This is a common
use case for masks, and the mask-based implementation of missing values
supports this usage by having the strict requirement that the data
storage backing any missing array elements never be touched.

In general, this can be done by first creating a view, then either adding
a mask if there isn't one yet, or having the view create its own copy of
the mask instead of retaining a view of the original's mask.

********************************
Missing Values as Seen in Python
********************************

Working With Missing Values
===========================

NumPy will gain a global singleton called numpy.NA, similar to None,
but with semantics reflecting its status as a missing value. In particular,
trying to treat it as a boolean will raise an exception, and comparisons
with it will produce numpy.NA instead of True or False. These basics are
adopted from the behavior of the NA value in the R project.

For example,::

    >>> np.array([1.0, 2.0, np.NA, 7.0], masked=True)
    array([1., 2., NA, 7.], masked=True)
    >>> np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]')
    array([1., 2., NA, 7.], dtype='NA[<f8]')

produce arrays with values [1.0, 2.0, <inaccessible>, 7.0] /
mask [Unmasked, Unmasked, Masked, Unmasked], and
values [1.0, 2.0, <NA bit pattern>, 7.0] respectively.

It may be worth overloading the np.NA __call__ method to accept a dtype,
returning a zero-dimensional array with a missing value of that dtype.
Without doing this, NA printouts would look like::

    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], masked=True))
    array(NA, dtype='float64', masked=True)
    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]'))
    array(NA, dtype='NA[<f8]')

but with this, they could be printed as::
    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], masked=True))
    NA('float64')
    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]'))
    NA('NA[<f8]')

Assigning a value to an array always causes that element to not be NA,
transparently unmasking it if necessary.. Assigning numpy.NA to the array
masks that element or assigns the NA bit pattern for the particular dtype.
In the mask-based implementation, the storage behind a missing value may never
be accessed in any way, other than to unmask it by assigning its value.

While numpy.NA works to mask values, it does not itself have a dtype.
This means that returning the numpy.NA singleton from an operation
like 'arr[0]' would be throwing away the dtype, which is still
valuable to retain, so 'arr[0]' will return a zero-dimensional
array either with its value masked, or containing the NA bit pattern
for the array's dtype. To test if the value is missing, the function
"np.ismissing(arr[0])" will be provided. One of the key reasons for the
NumPy scalars is to allow their values into dictionaries. Having a
missing value as the key in a dictionary is a bad idea, so the NumPy
scalars will not support missing values in any form.

All operations which write to masked arrays will not affect the value
unless they also unmask that value. This allows the storage behind
masked elements to still be relied on if they are still accessible
from another view which doesn't have them masked. For example::

    >>> a = np.array([1,2])
    >>> b = a.view()
    >>> b.flags.hasmask = True
    >>> b
    array([1,2], masked=True)
    >>> b[0] = np.NA
    >>> b
    array([NA,2], masked=True)
    >>> a
    array([1,2])
    >>> # The underlying number 1 value in 'a[0]' was untouched

Copying values between the mask-based implementation and the
NA bit pattern implementation will transparently do the correct thing,
turning the NA bit pattern into a masked value, or a masked value
into the NA bit pattern where appropriate. The one exception is
if a valid value in a masked array happens to have the NA bit pattern,
copying this value to the NA form of the dtype will cause it to
become NA as well.

If np.NA or masked values are copied to an array without support for
missing values enabled, an exception will be raised. Adding a mask to
the target array would be problematic, because then having a mask
would be a "viral" property consuming extra memory and reducing
performance in unexpected ways.

By default, the string "NA" will be used to represent missing values
in str and repr outputs. A global configuration will allow
this to be changed. The array2string function will also gain a
'maskedstr=' parameter so this could be changed to "<missing>" or
other values people may desire.

For floating point numbers, Inf and NaN are separate concepts from
missing values. If a division by zero occurs in an array with default
missing value support, an unmasked Inf or NaN will be produced. To
mask those values, a further 'a[np.logical_not(a.isfinite(a)] = np.NA'
can achieve that. For the NA bit pattern approach, the parameterized
dtype('NA[f8,InfNan]') described in a later section can be used to get
these semantics without the extra manipulation.

A manual loop through a masked array like::

    for i in xrange(len(a)):
        a[i] = np.log(a[i])

works even with masked values, because 'a[i]' returns a zero-dimensional
array with a missing value instead of the singleton np.NA for the missing
elements. If np.NA was returned, np.log would have to raise an exception
because it doesn't know the log of which dtype it's meant to call, whether
it's a missing float or a missing string, for example.

Accessing a Boolean Mask
========================

The mask used to implement missing data in the masked approach is not
accessible from Python directly. This is partially due to differing
opinions on whether True in the mask should mean "missing" or "not missing"
Additionally, exposing the mask directly would preclude a potential
space optimization, where a bit-level instead of a byte-level mask
is used to get a factor of eight memory usage improvement.

To access the mask values, there are two functions provided,
'np.ismissing' and 'np.isavail', which test for NA or available values
respectively. These functions work equivalently for masked arrays
and NA bit pattern dtypes.

Creating Masked Arrays
======================

There are two flags which indicate and control the nature of the mask
used in masked arrays.

First is 'arr.flags.hasmask', which is True for all masked arrays and
may be set to True to add a mask to an array which does not have one.

Second is 'arr.flags.ownmask', which is True if the array owns the
memory to the mask, and False if the array has no mask, or has a view
into the mask of another array. If this is set to False in a masked
array, the array will create a copy of the mask so that further modifications
to the mask will not affect the array being viewed.

Mask Implementation Details
===========================

The memory ordering of the mask will always match the ordering of
the array it is associated with. A Fortran-style array will have a
Fortran-style mask, etc.

When a view of an array with a mask is taken, the view will have
a mask which is also a view of the mask in the original
array. This means unmasking values in views will also unmask them
in the original array, and if a mask is added to an array, it will
not be possible to ever remove that mask except to create a new array
copying the data but not the mask.

It is still possible to temporarily treat an array with a mask without
giving it one, by first creating a view of the array and then adding a
mask to that view. A data set can be viewed with multiple different
masks simultaneously, by creating multiple views, and giving each view
a mask.

New ndarray Methods
===================

New functions added to the numpy namespace are::

    np.ismissing(arr)
        Returns a boolean array with True whereever the array is masked
        or matches the NA bit pattern, and False elsewhere

    np.isavail(arr)
        Returns a boolean array with False whereever the array is masked
        or matches the NA bit pattern, and True elsewhere

New functions added to the ndarray are::

    arr.copy(..., replacena=None)
        Modification to the copy function which replaces NA values,
        either masked or with the NA bit pattern, with the 'replacena='
        parameter suppled. When 'replacena' isn't None, the copied
        array is unmasked and has the 'NA' part stripped from the
        parameterized type ('NA[f8]' becomes just 'f8').

    arr.view(masked=True)
        This is a shortcut for 'a = arr.view(); a.flags.hasmask=True'.

Element-wise UFuncs With Missing Values
=======================================

As part of the implementation, ufuncs and other operations will
have to be extended to support masked computation. Because this
is a useful feature in general, even outside the context of
a masked array, in addition to working with masked arrays ufuncs
will take an optional 'mask=' parameter which allows the use
of boolean arrays to choose where a computation should be done.
This functions similar to a "where" clause on the ufunc.::

    >>> np.add(a, b, out=b, mask=(a > threshold))

A benefit of having this 'mask=' parameter is that it provides a way
to temporarily treat an object with a mask without ever creating a
masked array object.

If the 'out' parameter isn't specified, use of the 'mask=' parameter
will produce a array with a mask as the result.

For boolean operations, the R project special cases logical_and and
logical_or so that logical_and(NA, False) is False, and
logical_or(NA, True) is True. On the other hand, 0 * NA isn't 0, but
here the NA could represent Inf or NaN, in which case 0 * the backing
value wouldn't be 0 anyway.

For NumPy element-wise ufuncs, the design won't support this ability
for the mask of the output to depend simultaneously on the mask and
the value of the inputs. The NumPy 1.6 nditer, however, makes it
fairly easy to write standalone functions which look and feel just
like ufuncs, but deviate from their behavior. The functions logical_and
and logical_or can be moved into standalone function objects which are
backwards compatible with the current ufuncs.

Reduction UFuncs With Missing Values
====================================

Reduction operations like 'sum', 'prod', 'min', and 'max' will operate
consistently with the idea that a masked value exists, but its value
is unknown.

An optional parameter 'skipna=' will be added to those functions
which can interpret it appropriately to do the operation as if just
the unmasked values existed.

With 'skipna=True', when all the input values are masked,
'sum' and 'prod' will produce the additive and multiplicative identities
respectively, while 'min' and 'max' will produce masked values.
Statistics operations which require a count, like 'mean' and 'std'
will also use the unmasked value counts for their calculations if
'skipna=True', and produce masked values when all the inputs are masked.

Some examples::

    >>> a = np.array([1., 3., np.NA, 7.], masked=True)
    >>> np.sum(a)
    array(NA, dtype='<f8', masked=True)
    >>> np.sum(a, skipna=True)
    11.0
    >>> np.mean(a)
    array(NA, dtype='<f8', masked=True)
    >>> np.mean(a)
    3.6666666666666665
    >>> a = np.array([np.NA, np.NA], dtype='f8', masked=True)
    >>> np.sum(a, skipna=True)
    0.0
    >>> np.max(a, skipna=True)
    array(NA, dtype='<f8', masked=True)

PEP 3118
========

PEP 3118 doesn't have any mask mechanism, so arrays with masks will
not be accessible through this interface. Similarly, it doesn't support
the specification of dtypes with NA bit patterns, so the parameterized NA
dtypes will also not be accessible through this interface.

If NumPy did allow access through PEP 3118, this would circumvent the
missing value abstraction in a very damaging way. Other libraries would
try to use masked arrays, and silently get access to the data without
also getting access to the mask or being aware of the missing value
abstraction the mask and data together are following.

Unresolved Design Questions
===========================

The existing masked array implementation has a "hardmask" feature,
which prevents values from ever being unmasked by assigning a value.
This would be an internal array flag, named something like
'arr.flags.hardmask'.

If the hardmask feature is implemented, boolean indexing could
return a hardmasked array instead of a flattened array with the
arbitrary choice of C-ordering as it currently does. While this
improves the abstraction of the array significantly, it is not
a compatible change.

**********************************
Alternative Designs Without a Mask
**********************************

Parameterized Data Type With NA Signal Values
=============================================

A masked array isn't the only way to deal with missing data, and
some systems deal with the problem by defining a special "NA" value,
for data which is missing. This is distinct from NaN floating point
values, which are the result of bad floating point calculation values,
but many people use NaNs for this purpose.

In the case of IEEE floating point values, it is possible to use a
particular NaN value, of which there are many, for "NA", distinct
from NaN. For signed integers, a reasonable approach would be to use
the minimum storable value, which doesn't have a corresponding positive
value. For unsigned integers, the maximum storage value seems most
reasonable.

With the goal of providing a general mechanism, a parameterized type
mechanism for this is much more attractive than creating separate
nafloat32, nafloat64, naint64, nauint64, etc dtypes. If this is viewed
as an alternative way of treating the mask except without value preservation,
this parameterized type can work together with the mask in a special
way to produce a value + mask combination on the fly, and use the
exact same computational infrastructure as the masked array system.
This allows one to avoid the need to write special case code for each
ufunc and for each na* dtype, something that is hard to avoid when
building a separate independent dtype implementation for each na* dtype.

Reliable conversions with the NA bit pattern preserved across primitive
types requires consideration as well. Even in the simple case of
double -> float, where this is supported by hardware, the NA value
will get lost because the NaN payload is typically not preserved.
The ability to have different bit masks specified for the same underlying
type also needs to convert properly. With a well-defined interface
converting to/from a (value,flag) pair, this becomes straightforward
to support generically.

This approach also provides some opportunities for some subtle variations
with IEEE floats. By default, one exact bit-pattern, a silent NaN with
a payload that won't be generated by hardware floating point operations,
would be used. The choice R has made could be this default.

Additionally, it might be nice to sometimes treat all NaNs as missing values.
This requires a slightly more complex mapping to convert the floating point
values into mask/value combinations, and converting back would always
produce the default NaN used by NumPy. Finally, treating both NaNs
and Infs as missing values would be just a slight variation of the NaN
version.

Strings require a slightly different handling, because they
may be any size. One approach is to use a one-character signal consisting
of one of the first 32 ASCII/unicode values. There are many possible values
to use here, like 0x15 'Negative Acknowledgement' or 0x10 'Data Link Escape'.

The Object dtype has an obvious signal, the np.NA singleton itself. Any
dtype with object semantics won't be able to have this customized, since
specifying bit patterns applies only to plain binary data, not data
with object semantics of construction and destructions.

Struct dtypes are more of a core primitive dtype, in the same fashion that
this parameterized NA-capable dtype is. It won't be possible to put
these as the parameter for the parameterized NA-dtype.

The dtype names would be parameterized similar to how the datetime64
is parameterized by the metadata unit. What name to use may require some
debate, but "NA" seems like a reasonable choice. With the default
missing value bit-pattern, these dtypes would look like
np.dtype('NA[float32]'), np.dtype('NA[f8]'), or np.dtype('NA[i64]').

To override the bit pattern that signals a missing value, a raw
value in the format of a hexadecimal unsigned integer can be given,
and in the above special cases for floating point, special strings
can be provided. The defaults for some cases, written explicitly in this
form, are then::

    np.dtype('NA[?,0x02]')
    np.dtype('NA[i4,0x80000000]')
    np.dtype('NA[u4,0xffffffff]')
    np.dtype('NA[f4,0x7f8007a2')
    np.dtype('NA[f8,0x7ff00000000007a2') (R-compatible bitpattern)
    np.dtype('NA[S16,0x15]') (using the NAK character as the signal).

    np.dtype('NA[f8,NaN]') (for any NaN)
    np.dtype('NA[f8,InfNaN]') (for any NaN or Inf)

When no parameter is specified a flexible NA dtype is created, which itself
cannot hold values, but will conform to the input types in funcions like
'np.astype'. The dtype 'f8' maps to 'NA[f8]', and [('a', 'f4'), ('b', 'i4')]
maps to [('a', 'NA[f4]'), ('b', 'NA[i4]')]. Thus, to view the memory
of an 'f8' array 'arr' with 'NA[f8]', you can say arr.view(dtype='NA').

Parameterized Data Type Which Adds Additional Memory for the NA Flag
====================================================================

Another alternative to having a separate mask added to the array is
to introduced a parameterized type, which takes a primitive dtype
as an argument. The dtype "i8" would turn into "maybe[i8]", and
a byte flag would be appended to the dtype to indicate whether the
value was NA or not.

This approach adds memory overhead greater or equal to keeping a separate
mask, but has better locality. To keep the dtype aligned, an 'i8' would
need to have 16 bytes to retain proper alignment, a 100% overhead compared
to 12.5% overhead for a separately kept mask.

***************
Acknowledgments
***************

In addition to feedback Travis Oliphant and others at Enthought,
this NEP has been revised based on a great deal of feedback from
the NumPy-Discussion mailing list. The people participating in
the discussion are::

    Nathaniel Smith
    Robert Kern
    Charles Harris
    Gael Varoquaux
    Eric Firing
    Keith Goodman
    Pierre GM
    Christopher Barker
    Josef Perktold
    Ben Root
    Laurent Gautier
    Neal Becker
    Bruce Southey
    Matthew Brett
    Wes McKinney
    Lluís
    Olivier Delalleau
    Alan G Isaac

I apologize if I missed anyone.
