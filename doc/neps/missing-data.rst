:Title: Missing Data Functionality in NumPy
:Author: Mark Wiebe <mwwiebe@gmail.com>
:Copyright: Copyright 2011 by Enthought, Inc
:License: CC By-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0/)
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

NA (Not Available/Propagate)
    A placeholder for a value which is unknown to computations. That
    value may be temporarily hidden with a mask, may have been lost
    due to hard drive corruption, or gone for any number of reasons.
    For sums and products this means to produce NA if any of the inputs
    are NA. This is the same as NA in the R project.

IGNORE (Ignore/Skip)
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

Python API
    All the interface mechanisms that are exposed to Python code
    for using missing values in NumPy. This API is designed to be
    Pythonic and fit into the way NumPy works as much as possible.

C API
    All the implementation mechanisms exposed for CPython extensions
    written in C that want to support NumPy missing value support.
    This API is designed to be as natural as possible in C, and
    is usually prioritizes flexibility and high performance.

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

    >>> np.array([1.0, 2.0, np.NA, 7.0], maskna=True)
    array([1., 2., NA, 7.], maskna=True)
    >>> np.array([1.0, 2.0, np.NA, 7.0], dtype='NA')
    array([1., 2., NA, 7.], dtype='NA[<f8]')
    >>> np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f4]')
    array([1., 2., NA, 7.], dtype='NA[<f4]')

produce arrays with values [1.0, 2.0, <inaccessible>, 7.0] /
mask [Exposed, Exposed, Hidden, Exposed], and
values [1.0, 2.0, <NA bitpattern>, 7.0] for the masked and
NA dtype versions respectively.

The np.NA singleton may accept a dtype= keyword parameter, indicating
that it should be treated as an NA of a particular data type. This is also
a mechanism for preserving the dtype in a NumPy scalar-like fashion.
Here's what this looks like::

    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], maskna=True))
    NA(dtype='<f8')
    >>> np.sum(np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]'))
    NA(dtype='NA[<f8]')

Assigning a value to an array always causes that element to not be NA,
transparently unmasking it if necessary. Assigning numpy.NA to the array
masks that element or assigns the NA bitpattern for the particular dtype.
In the mask-based implementation, the storage behind a missing value may never
be accessed in any way, other than to unmask it by assigning its value.

To test if a value is missing, the function "np.isna(arr[0])" will
be provided. One of the key reasons for the NumPy scalars is to allow
their values into dictionaries.

All operations which write to masked arrays will not affect the value
unless they also unmask that value. This allows the storage behind
masked elements to still be relied on if they are still accessible
from another view which doesn't have them masked. For example, the
following was run on the missingdata work-in-progress branch::

    >>> a = np.array([1,2])
    >>> b = a.view(maskna=True)
    >>> b
    array([1, 2], maskna=True)
    >>> b[0] = np.NA
    >>> b
    array([NA, 2], maskna=True)
    >>> a
    array([1, 2])
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
this to be changed, exactly extending the way nan and inf are treated.
The following works in the current draft implementation::

    >>> a = np.arange(6, maskna=True)
    >>> a[3] = np.NA
    >>> a
    array([0, 1, 2, NA, 4, 5], maskna=True)
    >>> np.set_printoptions(nastr='blah')
    >>> a
    array([0, 1, 2, blah, 4, 5], maskna=True)

For floating point numbers, Inf and NaN are separate concepts from
missing values. If a division by zero occurs in an array with default
missing value support, an unmasked Inf or NaN will be produced. To
mask those values, a further 'a[np.logical_not(a.isfinite(a)] = np.NA'
can achieve that. For the bitpattern approach, the parameterized
dtype('NA[f8,InfNan]') described in a later section can be used to get
these semantics without the extra manipulation.

A manual loop through a masked array like::

    >>> a = np.arange(5., maskna=True)
    >>> a[3] = np.NA
    >>> a
    array([ 0.,  1.,  2., NA,  4.], maskna=True)
    >>> for i in range(len(a)):
    ...     a[i] = np.log(a[i])
    ...
    __main__:2: RuntimeWarning: divide by zero encountered in log
    >>> a
    array([       -inf,  0.        ,  0.69314718, NA,  1.38629436], maskna=True)

works even with masked values, because 'a[i]' returns an NA object
with a data type associated, that can be treated properly by the ufuncs.

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

Creating NA-Masked Arrays
=========================

The usual way to create an array with an NA mask is to pass the keyword
parameter maskna=True to one of the constructors. Most functions that
create a new array take this parameter, and produce an NA-masked
array with all its elements exposed when the parameter is set to True.

There are also two flags which indicate and control the nature of the mask
used in masked arrays. These flags can be used to add a mask, or ensure
the mask isn't a view into another array's mask.

First is 'arr.flags.maskna', which is True for all masked arrays and
may be set to True to add a mask to an array which does not have one.

Second is 'arr.flags.ownmaskna', which is True if the array owns the
memory to the mask, and False if the array has no mask, or has a view
into the mask of another array. If this is set to True in a masked
array, the array will create a copy of the mask so that further modifications
to the mask will not affect the original mask from which the view was taken.

NA-Masks When Constructing From Lists
=====================================

The initial design of NA-mask construction was to make all construction
fully explicit. This turns out to be unwieldy when working interactively
with NA-masked arrays, and having an object array be created instead of
an NA-masked array can be very surprising.

Because of this, the design has been changed to enable an NA-mask whenever
creating an array from lists which have an NA object in them. There could
be some debate of whether one should create NA-masks or NA-bitpatterns
by default, but due to the time constraints it was only feasible to tackle
NA-masks, and extending the NA-mask support more fully throughout NumPy seems
much more reasonable than starting another system and ending up with two
incomplete systems.

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

    np.isna(arr) [IMPLEMENTED]
        Returns a boolean array with True whereever the array is masked
        or matches the NA bitpattern, and False elsewhere

    np.isavail(arr)
        Returns a boolean array with False whereever the array is masked
        or matches the NA bitpattern, and True elsewhere

New functions added to the ndarray are::

    arr.copy(..., replacena=np.NA)
        Modification to the copy function which replaces NA values,
        either masked or with the NA bitpattern, with the 'replacena='
        parameter suppled. When 'replacena' isn't NA, the copied
        array is unmasked and has the 'NA' part stripped from the
        parameterized dtype ('NA[f8]' becomes just 'f8').

        The default for replacena is chosen to be np.NA instead of None,
        because it may be desirable to replace NA with None in an
        NA-masked object array.

        For future multi-NA support, 'replacena' could accept a dictionary
        mapping the NA payload to the value to substitute for that
        particular NA. NAs with payloads not appearing in the dictionary
        would remain as NA unless a 'default' key was also supplied.

        Both the parameter to replacena and the values in the dictionaries
        can be either scalars or arrays which get broadcast onto 'arr'.

    arr.view(maskna=True) [IMPLEMENTED]
        This is a shortcut for
        >>> a = arr.view()
        >>> a.flags.maskna = True

    arr.view(ownmaskna=True) [IMPLEMENTED]
        This is a shortcut for
        >>> a = arr.view()
        >>> a.flags.maskna = True
        >>> a.flags.ownmaskna = True

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

    >>> a = np.array([1., 3., np.NA, 7.], maskna=True)
    >>> np.sum(a)
    array(NA, dtype='<f8', maskna=True)
    >>> np.sum(a, skipna=True)
    11.0
    >>> np.mean(a)
    NA(dtype='<f8')
    >>> np.mean(a, skipna=True)
    3.6666666666666665

    >>> a = np.array([np.NA, np.NA], dtype='f8', maskna=True)
    >>> np.sum(a, skipna=True)
    0.0
    >>> np.max(a, skipna=True)
    array(NA, dtype='<f8', maskna=True)
    >>> np.mean(a)
    NA(dtype='<f8')
    >>> np.mean(a, skipna=True)
    /home/mwiebe/virtualenvs/dev/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2374: RuntimeWarning: invalid value encountered in double_scalars
      return mean(axis, dtype, out)
    nan

The functions 'np.any' and 'np.all' require some special consideration,
just as logical_and and logical_or do. Maybe the best way to describe
their behavior is through a series of examples::

    >>> np.any(np.array([False, False, False], maskna=True))
    False
    >>> np.any(np.array([False, np.NA, False], maskna=True))
    NA
    >>> np.any(np.array([False, np.NA, True], maskna=True))
    True

    >>> np.all(np.array([True, True, True], maskna=True))
    True
    >>> np.all(np.array([True, np.NA, True], maskna=True))
    NA
    >>> np.all(np.array([False, np.NA, True], maskna=True))
    False

Since 'np.any' is the reduction for 'np.logical_or', and 'np.all'
is the reduction for 'np.logical_and', it makes sense for them to
have a 'skipna=' parameter like the other similar reduction functions.

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

Future Expansion to multi-NA Payloads
=====================================

The packages SAS and Stata both support multiple different "NA" values.
This allows one to specify different reasons for why a value, for
example homework that wasn't done because the dog ate it or the student
was sick. In these packages, the different NA values have a linear ordering
which specifies how different NA values combine together.

In the sections on C implementation details, the mask has been designed
so that a mask with a payload is a strict superset of the NumPy boolean
type, and the boolean type has a payload of just zero. Different payloads
combine with the 'min' operation.

The important part of future-proofing the design is making sure
the C ABI-level choices and the Python API-level choices have a natural
transition to multi-NA support. Here is one way multi-NA support could look::

    >>> a = np.array([np.NA(1), 3, np.NA(2)], maskna='multi')
    >>> np.sum(a)
    NA(1, dtype='<i4')
    >>> np.sum(a[1:])
    NA(2, dtype='<i4')
    >>> b = np.array([np.NA, 2, 5], maskna=True)
    >>> a + b
    array([NA(0), 5, NA(2)], maskna='multi')

The design of this NEP does not distinguish between NAs that come
from an NA mask or NAs that come from an NA dtype. Both of these get
treated equivalently in computations, with masks dominating over NA
dtypes.::

    >>> a = np.array([np.NA, 2, 5], maskna=True)
    >>> b = np.array([1, np.NA, 7], dtype='NA')
    >>> a + b
    array([NA, NA, 12], maskna=True)

The multi-NA approach allows one to distinguish between these NAs,
through assigning different payloads to the different types. If we
extend the 'skipna=' parameter to accept a list of payloads in addition
to True/False, one could do this::

    >>> a = np.array([np.NA(1), 2, 5], maskna='multi')
    >>> b = np.array([1, np.NA(0), 7], dtype='NA[f4,multi]')
    >>> a + b
    array([NA(1), NA(0), 12], maskna='multi')
    >>> np.sum(a, skipna=0)
    NA(1, dtype='<i4')
    >>> np.sum(a, skipna=1)
    7
    >>> np.sum(b, skipna=0)
    8
    >>> np.sum(b, skipna=1)
    NA(0, dtype='<f4')
    >>> np.sum(a+b, skipna=(0,1))
    12

Differences with numpy.ma
=========================

The computational model that numpy.ma uses does not strictly adhere to
either the NA or the IGNORE model. This section exhibits some examples
of how these differences affect simple computations. This information
will be very important for helping users navigate between the systems,
so a summary probably should be put in a table in the documentation.::

    >>> a = np.random.random((3, 2))
    >>> mask = [[False, True], [True, True], [False, False]]
    >>> b1 = np.ma.masked_array(a, mask=mask)
    >>> b2 = a.view(maskna=True)
    >>> b2[mask] = np.NA

    >>> b1
    masked_array(data =
     [[0.110804969841 --]
     [-- --]
     [0.955128477746 0.440430735546]],
                 mask =
     [[False  True]
     [ True  True]
     [False False]],
           fill_value = 1e+20)
    >>> b2
    array([[0.110804969841, NA],
           [NA, NA],
           [0.955128477746, 0.440430735546]],
           maskna=True)

    >>> b1.mean(axis=0)
    masked_array(data = [0.532966723794 0.440430735546],
                 mask = [False False],
           fill_value = 1e+20)

    >>> b2.mean(axis=0)
    array([NA, NA], dtype='<f8', maskna=True)
    >>> b2.mean(axis=0, skipna=True)
    array([0.532966723794 0.440430735546], maskna=True)

For functions like np.mean, when 'skipna=True', the behavior
for all NAs is consistent with an empty array::

    >>> b1.mean(axis=1)
    masked_array(data = [0.110804969841 -- 0.697779606646],
                 mask = [False  True False],
           fill_value = 1e+20)

    >>> b2.mean(axis=1)
    array([NA, NA, 0.697779606646], maskna=True)
    >>> b2.mean(axis=1, skipna=True)
    RuntimeWarning: invalid value encountered in double_scalars
    array([0.110804969841, nan, 0.697779606646], maskna=True)

    >>> np.mean([])
    RuntimeWarning: invalid value encountered in double_scalars
    nan

In particular, note that numpy.ma generally skips masked values,
except returns masked when all the values are masked, while
the 'skipna=' parameter returns zero when all the values are NA,
to be consistent with the result of np.sum([])::

    >>> b1[1]
    masked_array(data = [-- --],
                 mask = [ True  True],
           fill_value = 1e+20)
    >>> b2[1]
    array([NA, NA], dtype='<f8', maskna=True)
    >>> b1[1].sum()
    masked
    >>> b2[1].sum()
    NA(dtype='<f8')
    >>> b2[1].sum(skipna=True)
    0.0

    >>> np.sum([])
    0.0

Boolean Indexing
================

Indexing using a boolean array containing NAs does not have a consistent
interpretation according to the NA abstraction. For example::

    >>> a = np.array([1, 2])
    >>> mask = np.array([np.NA, True], maskna=True)
    >>> a[mask]
    What should happen here?

Since the NA represents a valid but unknown value, and it is a boolean,
it has two possible underlying values::

    >>> a[np.array([True, True])]
    array([1, 2])
    >>> a[np.array([False, True])]
    array([2])

The thing which changes is the length of the output array, nothing which
itself can be substituted for NA. For this reason, at least initially,
NumPy will raise an exception for this case.

Another possibility is to add an inconsistency, and follow the approach
R uses. That is, to produce the following::

    >>> a[mask]
    array([NA, 2], maskna=True)

If, in user testing, this is found necessary for pragmatic reasons,
the feature should be added even though it is inconsistent.

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

Interaction With Pre-existing C API Usage
=========================================

Making sure existing code using the C API, whether it's written in C, C++,
or Cython, does something reasonable is an important goal of this implementation.
The general strategy is to make existing code which does not explicitly
tell numpy it supports NA masks fail with an exception saying so. There are
a few different access patterns people use to get ahold of the numpy array data,
here we examine a few of them to see what numpy can do. These examples are
found from doing google searches of numpy C API array access.

Numpy Documentation - How to extend NumPy
-----------------------------------------

http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#dealing-with-array-objects

This page has a section "Dealing with array objects" which has some advice for how
to access numpy arrays from C. When accepting arrays, the first step it suggests is
to use PyArray_FromAny or a macro built on that function, so code following this
advice will properly fail when given an NA-masked array it doesn't know how to handle.

The way this is handled is that PyArray_FromAny requires a special flag, NPY_ARRAY_ALLOWNA,
before it will allow NA-masked arrays to flow through.

http://docs.scipy.org/doc/numpy/reference/c-api.array.html#NPY_ARRAY_ALLOWNA

Code which does not follow this advice, and instead just calls PyArray_Check() to verify
its an ndarray and checks some flags, will silently produce incorrect results. This style
of code does not provide any opportunity for numpy to say "hey, this array is special",
so also is not compatible with future ideas of lazy evaluation, derived dtypes, etc.

Tutorial From Cython Website
----------------------------

http://docs.cython.org/src/tutorial/numpy.html

This tutorial gives a convolution example, and all the examples fail with
Python exceptions when given inputs that contain NA values.

Before any Cython type annotation is introduced, the code functions just
as equivalent Python would in the interpreter.

When the type information is introduced, it is done via numpy.pxd which
defines a mapping between an ndarray declaration and PyArrayObject \*.
Under the hood, this maps to __Pyx_ArgTypeTest, which does a direct
comparison of Py_TYPE(obj) against the PyTypeObject for the ndarray.

Then the code does some dtype comparisons, and uses regular python indexing
to access the array elements. This python indexing still goes through the
Python API, so the NA handling and error checking in numpy still can work
like normal and fail if the inputs have NAs which cannot fit in the output
array. In this case it fails when trying to convert the NA into an integer
to set in in the output.

The next version of the code introduces more efficient indexing. This
operates based on Python's buffer protocol. This causes Cython to call
__Pyx_GetBufferAndValidate, which calls __Pyx_GetBuffer, which calls
PyObject_GetBuffer. This call gives numpy the opportunity to raise an
exception if the inputs are arrays with NA-masks, something not supported
by the Python buffer protocol.

Numerical Python - JPL website
------------------------------

http://dsnra.jpl.nasa.gov/software/Python/numpydoc/numpy-13.html

This document is from 2001, so does not reflect recent numpy, but it is the
second hit when searching for "numpy c api example" on google.

There first example, heading "A simple example", is in fact already invalid for
recent numpy even without the NA support. In particular, if the data is misaligned
or in a different byteorder, it may crash or produce incorrect results.

The next thing the document does is introduce PyArray_ContiguousFromObject, which
gives numpy an opportunity to raise an exception when NA-masked arrays are used,
so the later code will raise exceptions as desired.

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
     *   If mask   : bool/uint8/structured dtype of mask dtypes
     */
    PyArray_Descr *maskna_dtype;
    /*
     * Raw data buffer for mask. If the array has the flag
     * NPY_ARRAY_OWNMASKNA enabled, it owns this memory and
     * must call PyArray_free on it when destroyed.
     */
    npy_mask *maskna_data;
    /*
     * Just like dimensions and strides point into the same memory
     * buffer, we now just make the buffer 3x the nd instead of 2x
     * and use the same buffer.
     */
    npy_intp *maskna_strides;

These fields can be accessed through the inline functions::

    PyArray_Descr *
    PyArray_MASKNA_DTYPE(PyArrayObject *arr);

    npy_mask *
    PyArray_MASKNA_DATA(PyArrayObject *arr);

    npy_intp *
    PyArray_MASKNA_STRIDES(PyArrayObject *arr);

    npy_bool
    PyArray_HASMASKNA(PyArrayObject *arr);

There are 2 or 3 flags which must be added to the array flags, both
for requesting NA masks and for testing for them::

    NPY_ARRAY_MASKNA
    NPY_ARRAY_OWNMASKNA
    /* To possibly add in a later revision */
    NPY_ARRAY_HARDMASKNA

To allow the easy detection of NA support, and whether an array
has any missing values, we add the following functions:

PyDataType_HasNASupport(PyArray_Descr* dtype)
    Returns true if this is an NA dtype, or a struct
    dtype where every field has NA support.

PyArray_HasNASupport(PyArrayObject* obj)
    Returns true if the array dtype has NA support, or
    the array has an NA mask.

PyArray_ContainsNA(PyArrayObject* obj)
    Returns false if the array has no NA support. Returns
    true if the array has NA support AND there is an
    NA anywhere in the array.

int PyArray_AllocateMaskNA(PyArrayObject* arr, npy_bool ownmaskna, npy_bool multina)
    Allocates an NA mask for the array, ensuring ownership if requested
    and using NPY_MASK instead of NPY_BOOL for the dtype if multina is True.

Mask Binary Format
==================

The format of the mask itself is designed to indicate whether an
element is masked or not, as well as contain a payload so that multiple
different NAs with different payloads can be used in the future.
Initially, we will simply use the payload 0.

The mask has type npy_uint8, and bit 0 is used to indicate whether
a value is masked. If ((m&0x01) == 0), the element is masked, otherwise
it is unmasked. The rest of the bits are the payload, which is (m>>1).
The convention for combining masks with payloads is that smaller
payloads propagate. This design gives 128 payload values to masked elements,
and 128 payload values to unmasked elements.

The big benefit of this approach is that npy_bool also
works as a mask, because it takes on the values 0 for False and 1
for True. Additionally, the payload for npy_bool, which is always
zero, dominates over all the other possible payloads.

Since the design involves giving the mask its own dtype, we can
distinguish between masking with a single NA value (npy_bool mask),
and masking with multi-NA (npy_uint8 mask). Initial implementations
will just support the npy_bool mask.

An idea that was discarded is to allow the combination of masks + payloads
to be a simple 'min' operation. This can be done by putting the payload
in bits 0 through 6, so that the payload is (m&0x7f), and using bit 7
for the masking flag, so ((m&0x80) == 0) means the element is masked.
The fact that this makes masks completely different from booleans, instead
of a strict superset, is the primary reason this choice was discarded.

********************************************
C Iterator API Changes: Iteration With Masks
********************************************

For iteration and computation with masks, both in the context of missing
values and when the mask is used like the 'where=' parameter in ufuncs,
extending the nditer is the most natural way to expose this functionality.

Masked operations need to work with casting, alignment, and anything else
which causes values to be copied into a temporary buffer, something which
is handled nicely by the nditer but difficult to do outside that context.

First we describe iteration designed for use of masks outside the
context of missing values, then the features which include missing
value support.

Iterator Mask Features
======================

We add several new per-operand flags:

NPY_ITER_WRITEMASKED
    Indicates that any copies done from a buffer to the array are
    masked. This is necessary because READWRITE mode could destroy
    data if a float array was being treated like an int array, so
    copying to the buffer and back would truncate to integers. No
    similar flag is provided for reading, because it may not be possible
    to know the mask ahead of time, and copying everything into
    the buffer will never destroy data.

    The code using the iterator should only write to values which
    are not masked by the mask specified, otherwise the result will
    be different depending on whether buffering is enabled or not.

NPY_ITER_ARRAYMASK
    Indicates that this array is a boolean mask to use when copying
    any WRITEMASKED argument from a buffer back to the array. There
    can be only one such mask, and there cannot also be a virtual
    mask.

    As a special case, if the flag NPY_ITER_USE_MASKNA is specified
    at the same time, the mask for the operand is used instead
    of the operand itself. If the operand has no mask but is
    based on an NA dtype, that mask exposed by the iterator converts
    into the NA bitpattern when copying from the buffer to the
    array.

NPY_ITER_VIRTUAL
    Indicates that this operand is not an array, but rather created on
    the fly for the inner iteration code. This allocates enough buffer
    space for the code to read/write data, but does not have
    an actual array backing the data. When combined with NPY_ITER_ARRAYMASK,
    allows for creating a "virtual mask", specifying which values
    are unmasked without ever creating a full mask array.

Iterator NA-array Features
==========================

We add several new per-operand flags:

NPY_ITER_USE_MASKNA
    If the operand has an NA dtype, an NA mask, or both, this adds a new
    virtual operand to the end of the operand list which iterates
    over the mask for the particular operand.

NPY_ITER_IGNORE_MASKNA
    If an operand has an NA mask, by default the iterator will raise
    an exception unless NPY_ITER_USE_MASKNA is specified. This flag
    disables that check, and is intended for cases where one has first
    checked that all the elements in the array are not NA using the
    PyArray_ContainsNA function.

    If the dtype is an NA dtype, this also strips the NA-ness from the
    dtype, showing a dtype that does not support NA.

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

In addition to feedback from Travis Oliphant and others at Enthought,
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
    Peter

I apologize if I missed anyone.
