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
the preferred solution in many cases. By implementing mask functionality
into the core ndarray object, all the current issues with the system
can be resolved in a high performance and flexible manner.

One key problem is a lack of orthogonality with other features, for
instance creating a masked array with physical quantities can't be
done because both are separate subclasses of ndarray. The only reasonable
way to deal with this is to move the mask into the core ndarray.

The integration with ufuncs and other numpy core functions like sum is weak.
This could be dealt with either through a better function overloading
mechanism or moving the mask into the core ndarray.

In the current masked array, calculations are done for the whole array,
then masks are patched up afterwords. This means that invalid calculations
sitting in masked elements can raise warnings or exceptions even though they
shouldn't, so the ufunc error handling mechanism can't be relied on.

While no comprehensive benchmarks appear to exist, poor performance is
sometimes cited as a problem as well.

***********************************************
Possible Alternative: Data Types With NA Values
***********************************************

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
signal value which may not be possible in general.

The masked array approach, on the other hand, works with all data types
in a uniform fashion, adding the cost of one byte per value storage
for the mask. The attractiveness of being able to define a new custom
data type for NumPy and have it automatically work with missing values
is one of the reasons the masked approach has been chosen over special
signal values.

Implementing masks as described in this NEP does not preclude also
creating data types with special "NA" values.

**************************
The Mask as Seen in Python
**************************

The 'mask' Property
===================

The array object will get a new property 'mask', which behaves very
similar to a boolean array. When this property isn't None, it
has a shape exactly matching the array's shape, and for struct dtypes,
has a matching dtype with every type in the struct replaced with bool.

The mask value is True for values that exist in the array, and False
for values that do not. This is the same convention used in most places
masks are used, for instance for image masks specifying which are valid
pixels and which are transparent. This is the reverse of the convention
in the current masked array subclass, but I think fixing this is worth
the trouble for the long term benefit.

When an array has no mask, as indicated by the 'mask' property being
None, a mask may be added by assigning a boolean array broadcastable
to the shape of the array. If the array already has a mask, this
operation will raise an exception unless the single value False is
being assigned, which will mask all the elements. The &= operator,
however will be allowed, as it can only cause unmasked values to become
masked.

The memory ordering of the mask will always match the ordering of
the array it is associated with. A Fortran-style array will have a
Fortran-style mask, etc.

When a view of an array with a mask is taken, the view will have a mask
which is also a view of the mask in the original array. This means unmasking
values in views will also unmask them in the original array, and if
a mask is added to an array, it will not be possible to ever remove that
mask except to create a new array copying the data but not the mask.

It is still possible to temporarily treat an array with a mask without
giving it one, by first creating a view of the array and then adding a
mask to that view.

Working With Masked Values
==========================

Assigning a value to the array always unmasks that element. There is
no interface to "unmask" elements except through assigning values.
The storage behind a masked value may never be accessed in any way,
other than to unmask it by assigning a value. If a masked view of
an array is taken, for instance, and another masked array is copied
over it, any values which stay masked will not have their underlying
value modified.

If masked values are copied to an array without a mask, an exception will
be raised. Adding a mask to the target array would be problematic, because
then having a mask would be a "viral" property consuming extra memory
and reducing performance in unexpected ways. To assign a value would require
a default value, which is something that should be explicitly stated,
so a function like "a.assign_from_masked(b, maskedvalue=3.0)" needs to
be created.

Except for object arrays, the None value will be used to represent
missing values in repr and str representations, except array2string
will gain a 'maskedstr=' parameter so this could be changed to "NA" or
other values people may desire. For example,::

    >>>np.array([1.0, 2.0, None, 7.0], masked=True)

will produce an array with values [1.0, 2.0, <inaccessible>, 7.0], and
mask [True, True, False, True].

For floating point numbers, Inf and NaN are separate concepts from
missing values. If a division by zero occurs, an unmasked Inf or NaN will
be produced. To mask those values, a further "a.mask &= np.isfinite(a)"
can achieve that.

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

Reduction operations like 'sum', 'prod', 'min', and 'max' will operate as
if the values weren't there, applying the operation to the unmasked
values. If all the input values are masked, 'sum' and 'prod' will produce
the additive and multiplicative identities respectively, while 'min'
and 'max' will produce masked values.

Statistics operations which require a count, like 'mean' and 'std' will
also use the unmasked value counts for their calculations, and produce
masked values when all the inputs are masked.

Unresolved Design Questions
===========================

Scalars will not be modified to have a mask, so this leaves two options
for what value should be returned when retrieving a single masked value.
Either 'None', or a zero-dimensional masked array. The former follows
the convention of returning an immutable value from such accesses,
while the later preserves type information, so the correct choice
will require some discussion to resolve.

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

There is some consternation about the conventional True/False
interpretation of the mask, centered around the name "mask". One
possibility to deal with this is to call it a "validity mask" in
all documentation, which more clearly indicates that True means
valid data. If this isn't sufficient, an alternate name for the
attribute could be found, like "a.validitymask", "a.validmask",
or "a.validity".
