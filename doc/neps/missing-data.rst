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
pointed to the masked array subclass of the ndarray, known
as 'numpy.ma'. This class has a number of users who depend strongly
on its capabilities, but people who are accustomed to the deep integration
of the missing data placeholder "NA" in the R project and others who
find the programming interface challenging or inconsistent tend not
to use it.

This NEP proposes to integrate a mask-based missing data solution
into NumPy, with an additional bitpattern-based missing data solution
that can be implemented  concurrently or later integrating seamlessly
with the mask-based solution.

The mask-based solution and the bitpattern-based solutions in this
proposal offer the exact same missing value abstraction, with several
differences in performance, memory overhead, and flexibility.

The mask-based solution is more flexible, supporting all behaviors of the
bitpattern-based solution, but leaving the hidden values untouched
whenever an element is masked.

The bitpattern-based solution requires less memory, is bit-level
compatible with the 64-bit floating point representation used in R, but
does not preserve the hidden values and in fact requires stealing at
least one bit pattern from the underlying dtype to represent the missing
value NA.

Both solutions are generic in the sense that they can be used with
custom data types very easily, with no effort in the case of the masked
solution, and with the requirement that a bit pattern to sacrifice be
chosen in the case of the bitpattern solution.

**************************
Definition of Missing Data
**************************

In order to be able to develop an intuition about what computation
will be done by various NumPy functions, a consistent conceptual
model of what a missing element means must be applied.
Ferreting out the behaviors people need or want when they are working
with "missing data" seems to be tricky, but I believe that it boils
down to two different ideas, each of which is internally self-consistent.

One of them, the "unknown yet existing data" interpretation, can be applied
rigorously to all computations, while the other makes sense for
some statistical operations like standard deviation but not for
linear algebra operations like matrix product.
Thus, making "unknown yet existing data" be the default interpretation
is superior, providing a consistent model across all computations,
and for those operations where the other interpretation makes sense,
an optional parameter "skipna=" can be added.

For people who want the other interpretation to be default, a mechanism
proposed elsewhere for customizing subclass ufunc behavior with a
_numpy_ufunc_ member function would allow a subclass with a different
default to be created.

Unknown Yet Existing Data (NA)
==============================

This is the approach taken in the R project, defining a missing element
as something which does have a valid value which isn't known, or is
NA (not available). This proposal adopts this behavior as as the
default for all operations involving missing values.

In this interpretation, nearly any computation with a missing input produces
a missing output. For example, 'sum(a)' would produce a missing value
if 'a' contained just one missing element. When the output value does
not depend on one of the inputs, it is reasonable to output a value
that is not NA, such as logical_and(NA, False) == False.

Some more complex arithmetic operations, such as matrix products, are
well defined with this interpretation, and the result should be
the same as if the missing values were NaNs. Actually implementing
such things to the theoretical limit is probably not worth it,
and in many cases either raising an exception or returning all
missing values may be preferred to doing precise calculations.

Data That Doesn't Exist Or Is Being Skipped (IGNORE)
====================================================

Another useful interpretation is that the missing elements should be
treated as if they didn't exist in the array, and the operation should
do its best to interpret what that means according to the data
that's left. In this case, 'mean(a)' would compute the mean of just
the values that are available, adjusting both the sum and count it
uses based on which values are missing. To be consistent, the mean of
an array of all missing values must produce the same result as the
mean of a zero-sized array without missing value support.

This kind of data can arise when conforming sparsely sampled data
into a regular sampling pattern, and is a useful interpretation to 
use when attempting to get best-guess answers for many statistical queries.

In R, many functions take a parameter "na.rm=T" which means to treat
the data as if the NA values are not part of the data set. This proposal
defines a standard parameter "skipna=True" for this same purpose. 

********************************************
Implementation Techniques For Missing Values
********************************************

In addition to there being two different interpretations of missing values,
there are two different commonly used implementation techniques for
missing values. While there are some differing default behaviors between
existing implementations of the techniques, I believe that the design
choices made in a new implementation must be made based on their merits,
not by rote copying of previous designs.

Both masks and bitpatterns have different strong and weak points,
depending on the application context. This NEP thus proposes to implement
both. To enable the writing of generic "missing value" code which does
not have to worry about whether the arrays it is using have taken one
or the other approach, the missing value semantics will be identical
for the two implementations.

Bit Patterns Signalling Missing Values (bitpattern)
===================================================

One or more patterns of bits, for example a NaN with
a particular payload, are chosen to represent the missing value
placeholder NA.

A consequence of this approach is that assigning NA changes the bits
holding the value, so that value is gone.

Additionally, for some types such as integers, a good and proper value
must be sacrificed to enable this functionality.

Boolean Masks Signalling Missing Values (mask)
==============================================

A mask is a parallel array of booleans, either one byte per element or
one bit per element, allocated alongside the existing array data. In this
NEP, the convention is chosen that True means the element is valid
(unmasked), and False means the element is NA.

By taking care when writing any C algorithm that works with values
and masks together, it is possible to have the memory for a value
that is masked never be written to. This feature allows multiple
simultaneous views of the same data with different choices of what
is missing, a feature requested by many people on the mailing list.

This approach places no limitations on the values of the underlying
data type, it may take on any binary pattern without affecting the
NA behavior.

*****************
Glossary of Terms
*****************

Because the above discussions of the different concepts and their
relationships are tricky to understand, here are more succinct
definitions of the terms used in this NEP.

NA (Not Available)
    A placeholder for a value which is unknown to computations. That
    value may be temporarily hidden with a mask, may have been lost
    due to hard drive corruption, or gone for any number of reasons.
    This is the same as NA in the R project.

IGNORE (Skip/Ignore)
    A placeholder which should be treated by computations as if no value does
    or could exist there. For sums, this means act as if the value
    were zero, and for products, this means act as if the value were one.
    It's as if the array were compressed in some fashion to not include
    that element.

bitpattern
    A technique for implementing either NA or IGNORE, where a particular
    set of bit patterns are chosen from all the possible bit patterns of the
    value's data type to signal that the element is NA or IGNORE.

mask
    A technique for implementing either NA or IGNORE, where a
    boolean or enum array parallel to the data array is used to signal
    which elements are NA or IGNORE.

numpy.ma
    The existing implementation of a particular form of masked arrays,
    which is part of the NumPy codebase.

********************************
Missing Values as Seen in Python
********************************

Working With Missing Values
===========================

NumPy will gain a global singleton called numpy.NA, similar to None,
but with semantics reflecting its status as a missing value. In particular,
trying to treat it as a boolean will raise an exception, and comparisons
with it will produce numpy.NA instead of True or False. These basics are
adopted from the behavior of the NA value in the R project. To dig
deeper into the ideas, http://en.wikipedia.org/wiki/Ternary_logic#Kleene_logic
provides a starting point.

For example,::

    >>> np.array([1.0, 2.0, np.NA, 7.0], namasked=True)
    array([1., 2., NA, 7.], namasked=True)
    >>> np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]')
    array([1., 2., NA, 7.], dtype='NA[<f8]')

produce arrays with values [1.0, 2.0, <inaccessible>, 7.0] /
mask [Unmasked, Unmasked, Masked, Unmasked], and
values [1.0, 2.0, <NA bitpattern>, 7.0] respectively.

It may be worth overloading the np.NA __call__ method to accept a dtype,
returning a zero-dimensional array with a missing value of that dtype.
Without doing this, NA printouts would look like::

    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], namasked=True))
    array(NA, dtype='float64', namasked=True)
    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]'))
    array(NA, dtype='NA[<f8]')

but with this, they could be printed as::

    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], namasked=True))
    NA('float64')
    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]'))
    NA('NA[<f8]')

Assigning a value to an array always causes that element to not be NA,
transparently unmasking it if necessary. Assigning numpy.NA to the array
masks that element or assigns the NA bitpattern for the particular dtype.
In the mask-based implementation, the storage behind a missing value may never
be accessed in any way, other than to unmask it by assigning its value.

While numpy.NA works to mask values, it does not itself have a dtype.
This means that returning the numpy.NA singleton from an operation
like 'arr[0]' would be throwing away the dtype, which is still
valuable to retain, so 'arr[0]' will return a zero-dimensional
array either with its value masked, or containing the NA bitpattern
for the array's dtype. To test if the value is missing, the function
"np.isna(arr[0])" will be provided. One of the key reasons for the
NumPy scalars is to allow their values into dictionaries. Having a
missing value as the key in a dictionary is a bad idea, so the NumPy
scalars will not support missing values in any form.

All operations which write to masked arrays will not affect the value
unless they also unmask that value. This allows the storage behind
masked elements to still be relied on if they are still accessible
from another view which doesn't have them masked. For example::

    >>> a = np.array([1,2])
    >>> b = a.view()
    >>> b.flags.hasnamask = True
    >>> b
    array([1,2], namasked=True)
    >>> b[0] = np.NA
    >>> b
    array([NA,2], namasked=True)
    >>> a
    array([1,2])
    >>> # The underlying number 1 value in 'a[0]' was untouched

Copying values between the mask-based implementation and the
bitpattern implementation will transparently do the correct thing,
turning the bitpattern into a masked value, or a masked value
into the bitpattern where appropriate. The one exception is
if a valid value in a masked array happens to have the NA bitpattern,
copying this value to the NA form of the dtype will cause it to
become NA as well.

When operations are done between arrays with NA dtypes and masked arrays,
the result will be masked arrays. This is because in some cases the
NA dtypes cannot represent all the values in the masked array, so
going to masked arrays is the only way to preserve all aspects of the data.

If np.NA or masked values are copied to an array without support for
missing values enabled, an exception will be raised. Adding a mask to
the target array would be problematic, because then having a mask
would be a "viral" property consuming extra memory and reducing
performance in unexpected ways.

By default, the string "NA" will be used to represent missing values
in str and repr outputs. A global configuration will allow
this to be changed. The array2string function will also gain a
'nastr=' parameter so this could be changed to "<missing>" or
other values people may desire.

For floating point numbers, Inf and NaN are separate concepts from
missing values. If a division by zero occurs in an array with default
missing value support, an unmasked Inf or NaN will be produced. To
mask those values, a further 'a[np.logical_not(a.isfinite(a)] = np.NA'
can achieve that. For the bitpattern approach, the parameterized
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

To access a mask directly, there are two functions provided. They
work equivalently for both arrays with masks and NA bit
patterns, so they are specified in terms of NA and available values
instead of masked and unmasked values. The functions are
'np.isna' and 'np.isavail', which test for NA or available values
respectively.

Creating Masked Arrays
======================

There are two flags which indicate and control the nature of the mask
used in masked arrays.

First is 'arr.flags.hasnamask', which is True for all masked arrays and
may be set to True to add a mask to an array which does not have one.

Second is 'arr.flags.ownnamask', which is True if the array owns the
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

    np.isna(arr)
        Returns a boolean array with True whereever the array is masked
        or matches the NA bitpattern, and False elsewhere

    np.isavail(arr)
        Returns a boolean array with False whereever the array is masked
        or matches the NA bitpattern, and True elsewhere

New functions added to the ndarray are::

    arr.copy(..., replacena=None)
        Modification to the copy function which replaces NA values,
        either masked or with the NA bitpattern, with the 'replacena='
        parameter suppled. When 'replacena' isn't None, the copied
        array is unmasked and has the 'NA' part stripped from the
        parameterized type ('NA[f8]' becomes just 'f8').

    arr.view(namasked=True)
        This is a shortcut for 'a = arr.view(); a.flags.hasnamask=True'.

Element-wise UFuncs With Missing Values
=======================================

As part of the implementation, ufuncs and other operations will
have to be extended to support masked computation. Because this
is a useful feature in general, even outside the context of
a masked array, in addition to working with masked arrays ufuncs
will take an optional 'where=' parameter which allows the use
of boolean arrays to choose where a computation should be done.::

    >>> np.add(a, b, out=b, where=(a > threshold))

A benefit of having this 'where=' parameter is that it provides a way
to temporarily treat an object with a mask without ever creating a
masked array object. In the example above, this would only do the
add for the array elements with True in the 'where' clause, and neither
'a' nor 'b' need to be masked arrays.

If the 'out' parameter isn't specified, use of the 'where=' parameter
will produce an array with a mask as the result, with missing values
for everywhere the 'where' clause had the value False.

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

    >>> a = np.array([1., 3., np.NA, 7.], namasked=True)
    >>> np.sum(a)
    array(NA, dtype='<f8', masked=True)
    >>> np.sum(a, skipna=True)
    11.0
    >>> np.mean(a)
    NA('<f8')
    >>> np.mean(a, skipna=True)
    3.6666666666666665

    >>> a = np.array([np.NA, np.NA], dtype='f8', namasked=True)
    >>> np.sum(a, skipna=True)
    0.0
    >>> np.max(a, skipna=True)
    array(NA, dtype='<f8', namasked=True)
    >>> np.mean(a)
    NA('<f8')
    >>> np.mean(a, skipna=True)
    /home/mwiebe/virtualenvs/dev/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2374: RuntimeWarning: invalid value encountered in double_scalars
      return mean(axis, dtype, out)
    nan

The functions 'np.any' and 'np.all' require some special consideration,
just as logical_and and logical_or do. Maybe the best way to describe
their behavior is through a series of examples::

    >>> np.any(np.array([False, False, False], namasked=True))
    False
    >>> np.any(np.array([False, NA, False], namasked=True))
    NA
    >>> np.any(np.array([False, NA, True], namasked=True))
    True

    >>> np.all(np.array([True, True, True], namasked=True))
    True
    >>> np.all(np.array([True, NA, True], namasked=True))
    NA
    >>> np.all(np.array([False, NA, True], namasked=True))
    False

Parameterized NA Data Types
===========================

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

Reliable conversions with the NA bitpattern preserved across primitive
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
cannot hold values, but will conform to the input types in functions like
'np.astype'. The dtype 'f8' maps to 'NA[f8]', and [('a', 'f4'), ('b', 'i4')]
maps to [('a', 'NA[f4]'), ('b', 'NA[i4]')]. Thus, to view the memory
of an 'f8' array 'arr' with 'NA[f8]', you can say arr.view(dtype='NA').

PEP 3118
========

PEP 3118 doesn't have any mask mechanism, so arrays with masks will
not be accessible through this interface. Similarly, it doesn't support
the specification of dtypes with NA or IGNORE bitpatterns, so the
parameterized NA dtypes will also not be accessible through this interface.

If NumPy did allow access through PEP 3118, this would circumvent the
missing value abstraction in a very damaging way. Other libraries would
try to use masked arrays, and silently get access to the data without
also getting access to the mask or being aware of the missing value
abstraction the mask and data together are following.

Cython
======

Cython uses PEP 3118 to work with NumPy arrays, so currently it will
simply refuse to work with them as described in the "PEP 3118" section.

In order to properly support NumPy missing values, Cython will need to
be modified in some fashion to add this support. Likely the best way
to do this will be to include it with supporting np.nditer, which
is most likely going to have an enhancement to make writing missing
value algorithms easier.

Hard Masks
==========

The numpy.ma implementation has a "hardmask" feature,
which prevents values from ever being unmasked by assigning a value.
This would be an internal array flag, named something like
'arr.flags.hardmask'.

If the hardmask feature is implemented, boolean indexing could
return a hardmasked array instead of a flattened array with the
arbitrary choice of C-ordering as it currently does. While this
improves the abstraction of the array significantly, it is not
a compatible change.

Shared Masks
============

One feature of numpy.ma is called 'shared masks'.

http://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.sharedmask

This feature cannot be supported by a masked implementation of
missing values without directly violating the missing value abstraction.
If the same mask memory is shared between two arrays 'a' and 'b', assigning
a value to a masked element in 'a' will simultaneously unmask the
element with matching index in 'b'. Because this isn't at the same time
assigning a valid value to that element in 'b', this has violated the
abstraction. For this reason, shared masks will not be supported
by the mask-based missing value implementation.

This is slightly different from what happens when taking a view
of an array with masked missing value support, where a view of
both the mask and the data are taken simultaneously. The result
is two views which share the same mask memory and the same data memory,
which still preserves the missing value abstraction.

************************
C Implementation Details
************************

The first version to implement is the array masks, because it is
the more general approach. The mask itself is an array, but since
it is intended to never be directly accessible from Python, it won't
be a full ndarray itself. The mask always has the same shape as
the array it's attached to, so it doesn't need its own shape. For
an array with a struct dtype, however, the mask will have a different
dtype than just a straight bool, so it does need its own dtype.
This gives us the following additions to the PyArrayObject::

    /*
     * Descriptor for the mask dtype.
     *   If no mask: NULL
     *   If mask   : bool/structured dtype of bools
     */
    PyArray_Descr *maskdescr;
    /*
     * Raw data buffer for mask. If the array has the flag
     * NPY_ARRAY_OWNNAMASK enabled, it owns this memory and
     * must call PyArray_free on it when destroyed.
     */
    char *maskdata;
    /*
     * Just like dimensions and strides point into the same memory
     * buffer, we now just make the buffer 3x the nd instead of 2x
     * and use the same buffer.
     */
    npy_intp *maskstrides;

There are 2 (or 3) flags which must be added to the array flags::

    NPY_ARRAY_HASNAMASK
    NPY_ARRAY_OWNNAMASK
    /* To possibly add in a later revision */
    NPY_ARRAY_HARDNAMASK

******************************
C API Access: Masked Iteration
******************************

TODO: Describe details about how the nditer will be extended to allow
functions to do masked iteration, transparently working with both
NA dtypes or masked arrays in one implementation.

********************
Rejected Alternative
********************

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
    E. Antero Tammi
    Jason Grout 
    Dag Sverre Seljebotn
    Joe Harrington
    Gary Strangman
    Chris Jordan-Squire

I apologize if I missed anyone.
