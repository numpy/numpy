=============================================================
NEP 24 — Missing Data Functionality - Alternative 1 to NEP 12
=============================================================

:Author: Nathaniel J. Smith <njs@pobox.com>, Matthew Brett <matthew.brett@gmail.com>
:Status: Deferred
:Type: Standards Track
:Created: 2011-06-30


Abstract
--------

*Context: this NEP was written as an alternative to NEP 12, which at the time of writing
had an implementation that was merged into the NumPy master branch.*

The principle of this NEP is to separate the APIs for masking and for missing values, according to

* The current implementation of masked arrays (NEP 12)
* This proposal.

This discussion is only of the API, and not of the implementation.

Detailed description
--------------------


Rationale
^^^^^^^^^

The purpose of this NEP is to define two interfaces -- one for handling
'missing values', and one for handling 'masked arrays'.

An ordinary value is something like an integer or a floating point number. A
*missing* value is a placeholder for an ordinary value that is for some
reason unavailable. For example, in working with statistical data, we often
build tables in which each row represents one item, and each column
represents properties of that item. For instance, we might take a group of
people and for each one record height, age, education level, and income, and
then stick these values into a table. But then we discover that our research
assistant screwed up and forgot to record the age of one of our individuals.
We could throw out the rest of their data as well, but this would be
wasteful; even such an incomplete row is still perfectly usable for some
analyses (e.g., we can compute the correlation of height and income). The
traditional way to handle this would be to stick some particular meaningless
value in for the missing data, e.g., recording this person's age as 0. But
this is very error prone; we may later forget about these special values
while running other analyses, and discover to our surprise that babies have
higher incomes than teenagers. (In this case, the solution would be to just
leave out all the items where we have no age recorded, but this isn't a
general solution; many analyses require something more clever to handle
missing values.) So instead of using an ordinary value like 0, we define a
special "missing" value, written "NA" for "not available".

Therefore, missing values have the following properties: Like any other
value, they must be supported by your array's dtype -- you can't store a
floating point number in an array with dtype=int32, and you can't store an NA
in it either. You need an array with dtype=NAint32 or something (exact syntax
to be determined). Otherwise, they act exactly like any other values. In
particular, you can apply arithmetic functions and so forth to them. By
default, any function which takes an NA as an argument always returns an NA
as well, regardless of the values of the other arguments. This ensures that
if we try to compute the correlation of income with age, we will get "NA",
meaning "given that some of the entries could be anything, the answer could
be anything as well". This reminds us to spend a moment thinking about how we
should rephrase our question to be more meaningful. And as a convenience for
those times when you do decide that you just want the correlation between the
known ages and income, then you can enable this behavior by adding a single
argument to your function call.

For floating point computations, NAs and NaNs have (almost?) identical
behavior. But they represent different things -- NaN an invalid computation
like 0/0, NA a value that is not available -- and distinguishing between
these things is useful because in some situations they should be treated
differently. (For example, an imputation procedure should replace NAs with
imputed values, but probably should leave NaNs alone.) And anyway, we can't
use NaNs for integers, or strings, or booleans, so we need NA anyway, and
once we have NA support for all these types, we might as well support it for
floating point too for consistency.

A masked array is, conceptually, an ordinary rectangular numpy array, which
has had an arbitrarily-shaped mask placed over it. The result is,
essentially, a non-rectangular view of a rectangular array. In principle,
anything you can accomplish with a masked array could also be accomplished by
explicitly keeping a regular array and a boolean mask array and using numpy
indexing to combine them for each operation, but combining them into a single
structure is much more convenient when you need to perform complex operations
on the masked view of an array, while still being able to manipulate the mask
in the usual ways. Therefore, masks are preserved through indexing, and
functions generally treat masked-out values as if they were not even part of
the array in the first place. (Maybe this is a good heuristic: a length-4
array in which the last value has been masked out behaves just like an
ordinary length-3 array, so long as you don't change the mask.) Except, of
course, that you are free to manipulate the mask in arbitrary ways whenever
you like; it's just a standard numpy array.

There are some simple situations where one could use either of these tools to
get the job done -- or other tools entirely, like using designated surrogate
values (age=0), separate mask arrays, etc. But missing values are designed to
be particularly helpful in situations where the missingness is an intrinsic
feature of the data -- where there's a specific value that **should** exist,
if it did exist we'd it'd mean something specific, but it **doesn't**. Masked
arrays are designed to be particularly helpful in situations where we just
want to temporarily ignore some data that does exist, or generally when we
need to work with data that has a non-rectangular shape (e.g., if you make
some measurement at each point on a grid laid over a circular agar dish, then
the points that fall outside the dish aren't missing measurements, they're
just meaningless).

Initialization
^^^^^^^^^^^^^^

First, missing values can be set and be displayed as ``np.NA, NA``::

   >>> np.array([1.0, 2.0, np.NA, 7.0], dtype='NA[f8]')
   array([1., 2., NA, 7.], dtype='NA[<f8]')

As the initialization is not ambiguous, this can be written without the NA
dtype::

   >>> np.array([1.0, 2.0, np.NA, 7.0])
   array([1., 2., NA, 7.], dtype='NA[<f8]')

Masked values can be set and be displayed as ``np.IGNORE, IGNORE``::

   >>> np.array([1.0, 2.0, np.IGNORE, 7.0], masked=True)
   array([1., 2., IGNORE, 7.], masked=True)

As the initialization is not ambiguous, this can be written without
``masked=True``::

   >>> np.array([1.0, 2.0, np.IGNORE, 7.0])
   array([1., 2., IGNORE, 7.], masked=True)

Ufuncs
^^^^^^

By default, NA values propagate::

   >>> na_arr = np.array([1.0, 2.0, np.NA, 7.0])
   >>> np.sum(na_arr)
   NA('float64')

unless the ``skipna`` flag is set::

   >>> np.sum(na_arr, skipna=True)
   10.0

By default, masking does not propagate::

   >>> masked_arr = np.array([1.0, 2.0, np.IGNORE, 7.0])
   >>> np.sum(masked_arr)
   10.0

unless the ``propmask`` flag is set::

   >>> np.sum(masked_arr, propmask=True)
   IGNORE

An array can be masked, and contain NA values::

   >>> both_arr = np.array([1.0, 2.0, np.IGNORE, np.NA, 7.0])

In the default case, the behavior is obvious::

   >>> np.sum(both_arr)
   NA('float64')

It's also obvious what to do with ``skipna=True``::

   >>> np.sum(both_arr, skipna=True)
   10.0
   >>> np.sum(both_arr, skipna=True, propmask=True)
   IGNORE

To break the tie between NA and MSK, NAs propagate harder::

   >>> np.sum(both_arr, propmask=True)
   NA('float64')

Assignment
^^^^^^^^^^

is obvious in the NA case::

   >>> arr = np.array([1.0, 2.0, 7.0])
   >>> arr[2] = np.NA
   TypeError('dtype does not support NA')
   >>> na_arr = np.array([1.0, 2.0, 7.0], dtype='NA[f8]')
   >>> na_arr[2] = np.NA
   >>> na_arr
   array([1., 2., NA], dtype='NA[<f8]')

Direct assignnent in the masked case is magic and confusing, and so happens only
via the mask::

   >>> masked_array = np.array([1.0, 2.0, 7.0], masked=True)
   >>> masked_arr[2] = np.NA
   TypeError('dtype does not support NA')
   >>> masked_arr[2] = np.IGNORE
   TypeError('float() argument must be a string or a number')
   >>> masked_arr.visible[2] = False
   >>> masked_arr
   array([1., 2., IGNORE], masked=True)


Copyright
---------

This document has been placed in the public domain.
