======================================
NEP 25 — NA support via special dtypes
======================================

:Author: Nathaniel J. Smith <njs@pobox.com>
:Status: Deferred
:Type: Standards Track
:Created: 2011-07-08

Abstract
========

*Context: this NEP was written as an additional alternative to NEP 12 (NEP 24
is another alternative), which at the time of writing had an implementation
that was merged into the NumPy master branch.*

To try and make more progress on the whole missing values/masked arrays/...
debate, it seems useful to have a more technical discussion of the pieces
which we *can* agree on. This is the second, which attempts to nail down the
details of how NAs can be implemented using special dtype's.

Rationale
---------

An ordinary value is something like an integer or a floating point number. A
missing value is a placeholder for an ordinary value that is for some reason
unavailable. For example, in working with statistical data, we often build
tables in which each row represents one item, and each column represents
properties of that item. For instance, we might take a group of people and
for each one record height, age, education level, and income, and then stick
these values into a table. But then we discover that our research assistant
screwed up and forgot to record the age of one of our individuals. We could
throw out the rest of their data as well, but this would be wasteful; even
such an incomplete row is still perfectly usable for some analyses (e.g., we
can compute the correlation of height and income). The traditional way to
handle this would be to stick some particular meaningless value in for the
missing data,e.g., recording this person's age as 0. But this is very error
prone; we may later forget about these special values while running other
analyses, and discover to our surprise that babies have higher incomes than
teenagers. (In this case, the solution would be to just leave out all the
items where we have no age recorded, but this isn't a general solution; many
analyses require something more clever to handle missing values.) So instead
of using an ordinary value like 0, we define a special "missing" value,
written "NA" for "not available".

There are several possible ways to represent such a value in memory. For
instance, we could reserve a specific value (like 0, or a particular NaN, or
the smallest negative integer) and then ensure that this value is treated
specially by all arithmetic and other operations on our array. Another option
would be to add an additional mask array next to our main array, use this to
indicate which values should be treated as NA, and then extend our array
operations to check this mask array whenever performing computations. Each
implementation approach has various strengths and weaknesses, but here we focus
on the former (value-based) approach exclusively and leave the possible
addition of the latter to future discussion. The core advantages of this
approach are (1) it adds no additional memory overhead, (2) it is
straightforward to store and retrieve such arrays to disk using existing file
storage formats, (3) it allows binary compatibility with R arrays including NA
values, (4) it is compatible with the common practice of using NaN to indicate
missingness when working with floating point numbers, (5) the dtype is already
a place where "weird things can happen" -- there are a wide variety of dtypes
that don't act like ordinary numbers (including structs, Python objects,
fixed-length strings, ...), so code that accepts arbitrary numpy arrays already
has to be prepared to handle these (even if only by checking for them and
raising an error). Therefore adding yet more new dtypes has less impact on
extension authors than if we change the ndarray object itself.

The basic semantics of NA values are as follows. Like any other value, they
must be supported by your array's dtype -- you can't store a floating point
number in an array with dtype=int32, and you can't store an NA in it either.
You need an array with dtype=NAint32 or something (exact syntax to be
determined). Otherwise, NA values act exactly like any other values. In
particular, you can apply arithmetic functions and so forth to them. By
default, any function which takes an NA as an argument always returns an NA as
well, regardless of the values of the other arguments. This ensures that if we
try to compute the correlation of income with age, we will get "NA", meaning
"given that some of the entries could be anything, the answer could be anything
as well". This reminds us to spend a moment thinking about how we should
rephrase our question to be more meaningful. And as a convenience for those
times when you do decide that you just want the correlation between the known
ages and income, then you can enable this behavior by adding a single argument
to your function call.

For floating point computations, NAs and NaNs have (almost?) identical
behavior. But they represent different things -- NaN an invalid computation
like 0/0, NA a value that is not available -- and distinguishing between these
things is useful because in some situations they should be treated differently.
(For example, an imputation procedure should replace NAs with imputed values,
but probably should leave NaNs alone.) And anyway, we can't use NaNs for
integers, or strings, or booleans, so we need NA anyway, and once we have NA
support for all these types, we might as well support it for floating point too
for consistency.

General strategy
================

Numpy already has a general mechanism for defining new dtypes and slotting them
in so that they're supported by ndarrays, by the casting machinery, by ufuncs,
and so on. In principle, we could implement NA-dtypes just using these existing
interfaces. But we don't want to do that, because defining all those new ufunc
loops etc. from scratch would be a huge hassle, especially since the basic
functionality needed is the same in all cases. So we need some generic
functionality for NAs -- but it would be better not to bake this in as a single
set of special "NA types", since users may well want to define new custom
dtypes that have their own NA values, and have them integrate well the rest of
the NA machinery. Our strategy, therefore, is to avoid the `mid-layer mistake`_
by exposing some code for generic NA handling in different situations, which
dtypes can selectively use or not as they choose.

.. _mid-layer mistake: https://lwn.net/Articles/336262/

Some example use cases:
  1. We want to define a dtype that acts exactly like an int32, except that the
     most negative value is treated as NA.
  2. We want to define a parametrized dtype to represent `categorical data`_,
     and the bit-pattern to be used for NA depends on the number of categories
     defined, so our code needs to play an active role handling it rather than
     simply deferring to the standard machinery.
  3. We want to define a dtype that acts like an length-10 string and supports
     NAs. Since our string may hold arbitrary binary values, we want to actually
     allocate 11 bytes for it, with the first byte a flag indicating whether this
     string is NA and the rest containing the string content.
  4. We want to define a dtype that allows multiple different types of NA data,
     which print differently and can be distinguished by the new ufunc that we
     define called ``is_na_of_type(...)``, but otherwise takes advantage of the
     generic NA machinery for most operations.

.. _categorical data: http://mail.scipy.org/pipermail/numpy-discussion/2010-August/052401.html

dtype C-level API extensions
============================

The `PyArray_Descr`_ struct gains the following new fields::

  void * NA_value;
  PyArray_Descr * NA_extends;
  int NA_extends_offset;

.. _PyArray_Descr: http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#PyArray_Descr

The following new flag values are defined::

  NPY_NA_AUTO_ARRFUNCS
  NPY_NA_AUTO_CAST
  NPY_NA_AUTO_UFUNC
  NPY_NA_AUTO_UFUNC_CHECKED
  NPY_NA_AUTO_ALL /* the above flags OR'ed together */

The `PyArray_ArrFuncs`_ struct gains the following new fields::

  void (*isna)(void * src, void * dst, npy_intp n, void * arr);
  void (*clearna)(void * data, npy_intp n, void * arr);

.. _PyArray_ArrFuncs: http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#PyArray_ArrFuncs

We add at least one new convenience macro::

  #define NPY_NA_SUPPORTED(dtype) ((dtype)->f->isna != NULL)

The general idea is that anywhere where we used to call a dtype-specific
function pointer, the code will be modified to instead:

  1. Check for whether the relevant ``NPY_NA_AUTO_...`` bit is enabled, the
     NA_extends field is non-NULL, and the function pointer we wanted to call
     is NULL.
  2. If these conditions are met, then use ``isna`` to identify which entries
     in the array are NA, and handle them appropriately. Then look up whatever
     function we were *going* to call using this dtype on the ``NA_extends``
     dtype instead, and use that to handle the non-NA elements.

For more specifics, see following sections.

Note that if ``NA_extends`` points to a parametrized dtype, then the dtype
object it points to must be fully specified. For example, if it is a string
dtype, it must have a non-zero ``elsize`` field.

In order to handle the case where the NA information is stored in a field next
to the `real' data, the ``NA_extends_offset`` field is set to a non-zero value;
it must point to the location within each element of this dtype where some data
of the ``NA_extends`` dtype is found. For example, if we have are storing
10-byte strings with an NA indicator byte at the beginning, then we have::

  elsize == 11
  NA_extends_offset == 1
  NA_extends->elsize == 10

When delegating to the ``NA_extends`` dtype, we offset our data pointer by
``NA_extends_offset`` (while keeping our strides the same) so that it sees an
array of data of the expected type (plus some superfluous padding). This is
basically the same mechanism that record dtypes use, IIUC, so it should be
pretty well-tested.

When delegating to a function that cannot handle "misbehaved" source data (see
the ``PyArray_ArrFuncs`` documentation for details), then we need to check for
alignment issues before delegating (especially with a non-zero
``NA_extends_offset``). If there's a problem, when we need to "clean up" the
source data first, using the usual mechanisms for handling misaligned data. (Of
course, we should usually set up our dtypes so that there aren't any alignment
issues, but someone screws that up, or decides that reduced memory usage is
more important to them then fast inner loops, then we should still handle that
gracefully, as we do now.)

The ``NA_value`` and ``clearna`` fields are used for various sorts of casting.
``NA_value`` is a bit-pattern to be used when, for example, assigning from
np.NA. ``clearna`` can be a no-op if ``elsize`` and ``NA_extends->elsize`` are
the same, but if they aren't then it should clear whatever auxiliary NA storage
this dtype uses, so that none of the specified array elements are NA.

Core dtype functions
--------------------

The following functions are defined in ``PyArray_ArrFuncs``. The special
behavior described here is enabled by the NPY_NA_AUTO_ARRFUNCS bit in the dtype
flags, and only enabled if the given function field is *not* filled in.

``getitem``: Calls ``isna``. If ``isna`` returns true, returns np.NA.
Otherwise, delegates to the ``NA_extends`` dtype.

``setitem``: If the input object is ``np.NA``, then runs
``memcpy(self->NA_value, data, arr->dtype->elsize);``. Otherwise, calls
``clearna``, and then delegates to the ``NA_extends`` dtype.

``copyswapn``, ``copyswap``: FIXME: Not sure whether there's any special
handling to use for these?

``compare``: FIXME: how should this handle NAs? R's sort function *discards*
NAs, which doesn't seem like a good option.

``argmax``: FIXME: what is this used for? If it's the underlying implementation
for np.max, then it really needs some way to get a skipna argument. If not,
then the appropriate semantics depends on what it's supposed to accomplish...

``dotfunc``: QUESTION: is it actually guaranteed that everything has the same
dtype? FIXME: same issues as for ``argmax``.

``scanfunc``: This one's ugly. We may have to explicitly override it in all of
our special dtypes, because assuming that we want the option of, say, having
the token "NA" represent an NA value in a text file, we need some way to check
whether that's there before delegating. But ``ungetc`` is only guaranteed to
let us put back 1 character, and we need 2 (or maybe 3 if we actually check for
"NA "). The other option would be to read to the next delimiter, check whether
we have an NA, and if not then delegate to ``fromstr`` instead of ``scanfunc``,
but according to the current API, each dtype might in principle use a totally
different rule for defining "the next delimiter". So... any ideas? (FIXME)

``fromstr``: Easy -- check for "NA ", if present then assign ``NA_value``,
otherwise call ``clearna`` and delegate.

``nonzero``: FIXME: again, what is this used for? (It seems redundant with
using the casting machinery to cast to bool.) Probably it needs to be modified
so that it can return NA, though...

``fill``: Use ``isna`` to check if either of the first two values is NA. If so,
then fill the rest of the array with ``NA_value``. Otherwise, call ``clearna``
and then delegate.

``fillwithvalue``: Guess this can just delegate?

``sort``, ``argsort``: These should probably arrange to sort NAs to a
particular place in the array (either the front or the back -- any opinions?)

``scalarkind``: FIXME: I have no idea what this does.

``castdict``, ``cancastscalarkindto``, ``cancastto``: See section on casting
below.

Casting
-------

FIXME: this really needs attention from an expert on numpy's casting rules. But
I can't seem to find the docs that explain how casting loops are looked up and
decided between (e.g., if you're casting from dtype A to dtype B, which dtype's
loops are used?), so I can't go into details. But those details are tricky and
they matter...

But the general idea is, if you have a dtype with ``NPY_NA_AUTO_CAST`` set,
then the following conversions are automatically allowed:

  * Casting from the underlying type to the NA-type: this is performed by the
  * usual ``clearna`` + potentially-strided copy dance. Also, ``isna`` is
  * called to check that none of the regular values have been accidentally
  * converted into NA; if so, then an error is raised.
  * Casting from the NA-type to the underlying type: allowed in principle, but
    if ``isna`` returns true for any of the values that are to be converted,
    then again, an error is raised. (If you want to get around this, use
    ``np.view(array_with_NAs, dtype=float)``.)
  * Casting between the NA-type and other types that do not support NA: this is
    allowed if the underlying type is allowed to cast to the other type, and is
    performed by combining a cast to or from the underlying type (using the
    above rules) with a cast to or from the other type (using the underlying
    type's rules).
  * Casting between the NA-type and other types that do support NA: if the
    other type has NPY_NA_AUTO_CAST set, then we use the above rules plus the
    usual dance with ``isna`` on one array being converted to ``NA_value``
    elements in the other. If only one of the arrays has NPY_NA_AUTO_CAST set,
    then it's assumed that that dtype knows what it's doing, and we don't do
    any magic. (But this is one of the things that I'm not sure makes sense, as
    per my caveat above.)

Ufuncs
------

All ufuncs gain an additional optional keyword argument, ``skipNA=``, which
defaults to False.

If ``skipNA == True``, then the ufunc machinery *unconditionally* calls
``isna`` for any dtype where NPY_NA_SUPPORTED(dtype) is true, and then acts as
if any values for which isna returns True were masked out in the ``where=``
argument (see miniNEP 1 for the behavior of ``where=``). If a ``where=``
argument is also given, then it acts as if the ``isna`` values had be ANDed out
of the ``where=`` mask, though it does not actually modify the mask. Unlike the
other changes below, this is performed *unconditionally* for any dtype which
has an ``isna`` function defined; the NPY_NA_AUTO_UFUNC flag is *not* checked.

If NPY_NA_AUTO_UFUNC is set, then ufunc loop lookup is modified so that
whenever it checks for the existence of a loop on the current dtype, and does
not find one, then it also checks for a loop on the ``NA_extends`` dtype. If
that loop is found, then it uses it in the normal way, with the exceptions that
(1) it is only called for values which are not NA according to ``isna``, (2) if
the output array has NPY_NA_AUTO_UFUNC set, then ``clearna`` is called on it
before calling the ufunc loop, (3) pointer offsets are adjusted by
``NA_extends_offset`` before calling the ufunc loop. In addition, if
NPY_NA_AUTO_UFUNC_CHECK is set, then after evaluating the ufunc loop we call
``isna`` on the *output* array, and if there are any NAs in the output which
were not in the input, then we raise an error. (The intention of this is to
catch cases where, say, we represent NA using the most-negative integer, and
then someone's arithmetic overflows to create such a value by accident.)

FIXME: We should go into more detail here about how NPY_NA_AUTO_UFUNC works
when there are multiple input arrays, of which potentially some have the flag
set and some do not.

Printing
--------

FIXME: There should be some sort of mechanism by which values which are NA are
automatically repr'ed as NA, but I don't really understand how numpy printing
works, so I'll let someone else fill in this section.

Indexing
--------

Scalar indexing like ``a[12]`` goes via the ``getitem`` function, so according
to the proposal as described above, if a dtype delegates ``getitem``, then
scalar indexing on NAs will return the object ``np.NA``. (If it doesn't
delegate ``getitem``, of course, then it can return whatever it wants.)

This seems like the simplest approach, but an alternative would be to add a
special case to scalar indexing, where if an ``NPY_NA_AUTO_INDEX`` flag were
set, then it would call ``isna`` on the specified element. If this returned
false, it would call ``getitem`` as usual; otherwise, it would return a 0-d
array containing the specified element. The problem with this is that it breaks
expressions like ``if a[i] is np.NA: ...``. (Of course, there is nothing nearly
so convenient as that for NaN values now, but then, NaN values don't have their
own global singleton.) So for now we stick to scalar indexing just returning
``np.NA``, but this can be revisited if anyone objects.

Python API for generic NA support
=================================

NumPy will gain a global singleton called numpy.NA, similar to None, but with
semantics reflecting its status as a missing value. In particular, trying to
treat it as a boolean will raise an exception, and comparisons with it will
produce numpy.NA instead of True or False. These basics are adopted from the
behavior of the NA value in the R project. To dig deeper into the ideas,
http://en.wikipedia.org/wiki/Ternary_logic#Kleene_logic provides a starting
point.

Most operations on ``np.NA`` (e.g., ``__add__``, ``__mul__``) are overridden to
unconditionally return ``np.NA``.

The automagic dtype detection used for expressions like ``np.asarray([1, 2,
3])``, ``np.asarray([1.0, 2.0. 3.0])`` will be extended to recognize the
``np.NA`` value, and use it to automatically switch to a built-in NA-enabled
dtype (which one being determined by the other elements in the array). A simple
``np.asarray([np.NA])`` will use an NA-enabled float64 dtype (which is
analogous to what you get from ``np.asarray([])``). Note that this means that
expressions like ``np.log(np.NA)`` will work: first ``np.NA`` will be coerced
to a 0-d NA-float array, and then ``np.log`` will be called on that.

Python-level dtype objects gain the following new fields::

  NA_supported
  NA_value

``NA_supported`` is a boolean which simply exposes the value of the
``NPY_NA_SUPPORTED`` flag; it should be true if this dtype allows for NAs,
false otherwise. [FIXME: would it be better to just key this off the existence
of the ``isna`` function? Even if a dtype decides to implement all other NA
handling itself, it still has to define ``isna`` in order to make ``skipNA=``
work correctly.]

``NA_value`` is a 0-d array of the given dtype, and its sole element contains
the same bit-pattern as the dtype's underlying ``NA_value`` field. This makes
it possible to determine the default bit-pattern for NA values for this type
(e.g., with ``np.view(mydtype.NA_value, dtype=int8)``).

We *do not* expose the ``NA_extends`` and ``NA_extends_offset`` values at the
Python level, at least for now; they're considered an implementation detail
(and it's easier to expose them later if they're needed then unexpose them if
they aren't).

Two new ufuncs are defined: ``np.isNA`` returns a logical array, with true
values where-ever the dtype's ``isna`` function returned true. ``np.isnumber``
is only defined for numeric dtypes, and returns True for all elements which are
not NA, and for which ``np.isfinite`` would return True.

Builtin NA dtypes
=================

The above describes the generic machinery for NA support in dtypes. It's
flexible enough to handle all sorts of situations, but we also want to define a
few generally useful NA-supporting dtypes that are available by default.

For each built-in dtype, we define an associated NA-supporting dtype, as
follows:

* floats: the associated dtype uses a specific NaN bit-pattern to indicate NA
  (chosen for R compatibility)
* complex: we do whatever R does (FIXME: look this up -- two NA floats,
  probably?)
* signed integers: the most-negative signed value is used as NA (chosen for R
  compatibility)
* unsigned integers: the most-positive value is used as NA (no R compatibility
  possible).
* strings: the first byte (or, in the case of unicode strings, first 4 bytes)
  is used as a flag to indicate NA, and the rest of the data gives the actual
  string. (no R compatibility possible)
* objects: Two options (FIXME): either we don't include an NA-ful version, or
  we use np.NA as the NA bit pattern.
* boolean: we do whatever R does (FIXME: look this up -- 0 == FALSE, 1 == TRUE,
  2 == NA?)

Each of these dtypes is trivially defined using the above machinery, and are
what are automatically used by the automagic type inference machinery (for
``np.asarray([True, np.NA, False])``, etc.).

They can also be accessed via a new function ``np.withNA``, which takes a
regular dtype (or an object that can be coerced to a dtype, like 'float') and
returns one of the above dtypes. Ideally ``withNA`` should also take some
optional arguments that let you describe which values you want to count as NA,
etc., but I'll leave that for a future draft (FIXME).

FIXME: If ``d`` is one of the above dtypes, then should ``d.type`` return?

The NEP also contains a proposal for a somewhat elaborate
domain-specific-language for describing NA dtypes. I'm not sure how great an
idea that is. (I have a bias against using strings as data structures, and find
the already existing strings confusing enough as it is -- also, apparently the
NEP version of numpy uses strings like 'f8' when printing dtypes, while my
numpy uses object names like 'float64', so I'm not sure what's going on there.
``withNA(float64, arg1=value1)`` seems like a more pleasant way to print a
dtype than "NA[f8,value1]", at least to me.) But if people want it, then cool.

Type hierarchy 
--------------

FIXME: how should we do subtype checks, etc., for NA dtypes? What does
``issubdtype(withNA(float), float)`` return? How about
``issubdtype(withNA(float), np.floating)``?

Serialization
-------------


Copyright
---------

This document has been placed in the public domain.
