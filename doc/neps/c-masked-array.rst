:Title: Masked Array Functionality in C
:Author: Mark Wiebe <mwwiebe@gmail.com>
:Date: 2011-06-23

*****************
Table of Contents
*****************

.. contents::

********
Abstract
********

The existing masked array functionality in NumPy is useful for many
people, however it has a number of issues that prevent it from being
the preferred solution in some important cases. By implementing mask
functionality into the core ndarray object, all the current issues
with the system can be resolved in a high performance and flexible manner.

The integration with ufuncs and other numpy core functions like sum is weak.
This could be dealt with either through a better function overloading
mechanism or moving the mask into the core ndarray.

In the current masked array, calculations are done for the whole array,
then masks are patched up afterwords. This means that invalid calculations
sitting in masked elements can raise warnings or exceptions even though they
shouldn't, so the ufunc error handling mechanism can't be relied on.

While no comprehensive benchmarks appear to exist, poor performance is
sometimes cited as a problem as well.

**************************
Definition of Missing Data
**************************

Unknown Yet Existing Data
=========================

In order to be able to develop an intuition about what computation
will be done by various NumPy functions, a consistent conceptual
model of what a masked element means must be applied. The approach
taken in the R project is to define a masked element as something which
does have a valid value, but that value is unknown.

In this interpretation, any computation with a masked input produces
a masked output. For example, 'sum(a)' would produce a masked value
if 'a' contained just one masked element.

Some more complex arithmetic operations, such as matrix products, are
well defined with this interpretation, and the result should be
the same as is the missing values were NaNs. Actually implementing
such things to the theoretical limit is probably not worth it,
and in many cases either raising an exception or returning all
missing values may be preferred to doing precise calculations.

This approach should likely be the default uniformly throughout NumPy,
because it will consistently flag problems by default, instead of
silently producing incorrect results because a missing value is
hidden deep within an array.

Data That Doesn't Exist
=======================

Another useful interpretation is that the masked elements should be
treated as if they didn't exist in the array, and the operation should
do its best interpretation of what that means according to the data
that's left. In this case, 'mean(a)' would compute the mean of just
the values that are unmasked, adjusting both the sum and count it
uses based on the mask.

This approach is useful when working with messy data and the analysis
being done is trying to produce the best result that's possible from
the data that is available.

In R, many functions take a parameter "na.rm=T" which means to treat
the data as if the NA values are not part of the data set.

Data That Is Being Temporarily Ignored
======================================

Iterpreting the meaning of temporarily ignored data requires
choosing between one of the missing data interpretations above.
This is a common use case for masks, which are an elegant mechanism
to implement this.

**************************
The Mask as Seen in Python
**************************

Working With Masked Values
==========================

NumPy will gain a global singleton called numpy.NA, similar to None,
but with semantics reflecting its status as a missing value. In particular,
trying to treat it as a boolean will raise an exception, and comparisons
with it will produce numpy.NA instead of True or False. These basics are
adopted from the behavior of the NA value in the R project.

Assigning a value to the array always unmasks that element. Assigning
numpy.NA to the array masks that element. The storage behind a masked
value may never be accessed in any way, other than to unmask it by
assigning a value.

Because numpy.NA is a global singleton, it will be possible to test
whether a value is masked by saying "arr[0] is np.NA".

All operations which write to masked arrays will not affect the value
unless they also unmask that value. This allows the storage behind
masked elements to still be relied on if they are still accessible
from another view which doesn't have them masked.

If np.NA or masked values are copied to an array without a mask, an
exception will be raised. Adding a validitymask to the target array
would be problematic, because then having a mask would be a "viral"
property consuming extra memory and reducing performance in unexpected
ways. To assign a value would require a default value, which is
something that should be explicitly stated, so a function like
"a.assign_from_masked(b, maskedvalue=3.0)" needs to be created.

By default, the string "NA" will be used to represent masked values
in str and repr outputs. A global default configuration will allow
this to be changed. The array2string function will also gain a
'maskedstr=' parameter so this could be changed to "NA" or
other values people may desire. For example,::

    >>>np.array([1.0, 2.0, np.NA, 7.0], masked=True)

will produce an array with values [1.0, 2.0, <inaccessible>, 7.0], and
validitymask [True, True, False, True].

For floating point numbers, Inf and NaN are separate concepts from
missing values. If a division by zero occurs, an unmasked Inf or NaN will
be produced. To mask those values, a further 'a.validitymask &= np.isfinite(a)'
can achieve that.

A manual loop through a masked array like::

    for i in xrange(len(a)):
        a[i] = np.log(a[i])

should work, something that is a little bit tricky because the global
singleton np.NA has no type, and doesn't follow the type promotion rules.
A good approach to deal with this needs to be found.

The 'validitymask' Property
===========================

The array object will get a new property 'validitymask', which behaves very
similar to a boolean array. When this property isn't None, it
has a shape exactly matching the array's shape, and for struct dtypes,
has a matching dtype with every type in the struct replaced with bool.

The reason for calling it 'validitymask' instead of just 'mask' or something
shorter is that this object is not intended to be the primary way to work
with masked values. It provides an interface for working with the mask,
but primarily the mask will be changed transparently based on manipulating
values and using the global singleton 'numpy.NA'.

The validitymask value is True for values that exist in the array, and False
for values that do not. This is the same convention used in most places
masks are used, for instance for image masks specifying which are valid
pixels and which are transparent. This is the reverse of the convention
in the current masked array subclass, but I think changing this is worth
the trouble for the long term benefit.

When an array has no mask, as indicated by the 'arr.flags.hasmask'
property being False, a mask may be added either by assigning True to
'arr.flags.hasmask', or assigning a boolean array to 'arr.validitymask'.
If the array already has a validitymask, this operation will raise an
exception unless the single value False is being assigned, which will
mask all the elements. The &= operator, however will be allowed, as
it can only cause unmasked values to become masked.

The memory ordering of the validitymask will always match the ordering of
the array it is associated with. A Fortran-style array will have a
Fortran-style validitymask, etc.

When a view of an array with a validitymask is taken, the view will have
a validitymask which is also a view of the validitymask in the original
array. This means unmasking values in views will also unmask them
in the original array, and if a mask is added to an array, it will
not be possible to ever remove that mask except to create a new array
copying the data but not the mask.

It is still possible to temporarily treat an array with a mask without
giving it one, by first creating a view of the array and then adding a
mask to that view. A data set can be viewed with multiple different
masks simultaneously, by creating multiple views, and giving each view
a mask.

When a validitymask gets added, the array to which it was added owns
the validitymask. This is indicated by the 'arr.flags.ownmask' flag.
When a view of an array with a validity mask is taken, the view does
not own its validitymask. In this case, it is possible to assign
'arr.flags.ownmask = True', which gives 'arr' its own copy of the
validitymask it is using, allowing it to be changed without affecting
the mask of the array being viewed.

New ndarray Methods
===================

In addition to the 'mask' property, the ndarray needs several new
methods to easily work with masked values. The proposed methods for
an np.array *a* are::

    a.assign_from_masked(b, fillvalue, casting='same_kind'):
        This is equivalent to a[...] = b, with the provided maskedvalue
        being substituted wherever there is missing data. This is
        intended for use when 'a' has no mask, but 'b' does.

    a.fill_masked(value)
        This is exactly like a.fill(value), but only modifies the
        masked elements of 'a'. All values of 'a' become unmasked.

    a.fill_unmasked(value)
        This is exactly like a.fill(value), but only modifies the
        unmasked elements of a. The mask remains unchanged.

    a.copy_filled(fillvalue, order='K', ...)
        Exactly like a.copy(), except always produces an array
        without a mask and uses 'fillvalue' for any masked values.

Masked Element-wise UFuncs
==========================

As part of the implementation, ufuncs and other operations will
have to be extended to support masked computation. Because this
is a useful feature in general, even outside the context of
a masked array, in addition to working with masked arrays ufuncs
will take an optional 'mask=' parameter which allows the use
of boolean arrays to choose where a computation should be done.
This functions similar to a "where" clause on the ufunc.::

    np.add(a, b, out=b, mask=(a > threshold))

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

Masked Reduction UFuncs
=======================

Reduction operations like 'sum', 'prod', 'min', and 'max' will operate
consistently with the idea that a masked value exists, but its value
is unknown.

An optional parameter 'skipna=False' will be added to those functions
which can interpret it appropriately to do the operation as if just
the unmasked values existed. When all the input values are masked,
'sum' and 'prod' will produce the additive and multiplicative identities
respectively, while 'min' and 'max' will produce masked values. With
this parameter enabled, statistics operations which require a count,
like 'mean' and 'std' will also use the unmasked value counts for
their calculations, and produce masked values when all the inputs are masked.

PEP 3118
========

PEP 3118 doesn't have any mask mechanism, so arrays with masks will
not be accessible through this interface.

Unresolved Design Questions
===========================

The existing masked array implementation has a "hardmask" feature,
which freezes the mask.  This would be an internal
array flag, with 'a.mask.harden()' and 'a.mask.soften()' performing the
functions of 'a.harden_mask()' and 'a.soften_mask()' in the current masked
array. There would also need to be an 'a.mask.ishard' property.

If the hardmask feature is implemented, boolean indexing could
return a hardmasked array instead of a flattened array with the
arbitrary choice of C-ordering as it currently does. While this
improves the abstraction of the array significantly, it is not
a compatible change.

****************************
Possible Alternative Designs
****************************

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
    Benjamin Root
    Laurent Gautier
    Neal Becker
    Bruce Southey
    Matthew Brett
    Wes McKinney
    Lluís
    Olivier Delalleau
    Alan G Isaac

I apologize if I missed anyone.
