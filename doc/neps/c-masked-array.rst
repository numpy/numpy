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

*************************
Definition of Masked Data
*************************

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

Data Types With NA Signal Values
================================

A masked array isn't the only way to deal with missing data, and
some systems deal with the problem by defining a special "NA" value,
for data which is missing. This is distinct from NaN floating point
values, which are the result of bad floating point calculation values.

In the case of IEEE floating point values, it is possible to use a
particular NaN value, of which there are many, for "NA", distinct
from NaN. For integers, a reasonable approach would be to use
the minimum storable value, which doesn't have a corresponding positive
value, so is perhaps reasonable to dispense with in most contexts.

The trouble with this approach is that it requires a large amount
of special case code in each data type, and writing a new data type
supporting missing data requires defining a mechanism for a special
signal value which may not be possible in general. This causes the
missing value logic to be replicated many times, something that can be
error-prone. This is also a lot more code for all the various ufuncs
than a general masked mechanism which can use the unmasked loop for
a default implementation.

The masked array approach, on the other hand, works with all data types
in a uniform fashion, adding the cost of one byte per value storage
for the mask. The attractiveness of being able to define a new custom
data type for NumPy and have it automatically work with missing values
is one of the reasons the masked approach has been chosen over special
signal values.

Implementing masks as described in this NEP does not preclude also
creating data types with special "NA" values.

Parameterized Type Which Adds an NA Flag
========================================

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
the discussion are:

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

I apologize if I missed anyone.
