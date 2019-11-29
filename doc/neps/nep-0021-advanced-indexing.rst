==================================================
NEP 21 — Simplified and explicit advanced indexing
==================================================

:Author: Sebastian Berg
:Author: Stephan Hoyer <shoyer@google.com>
:Status: Draft
:Type: Standards Track
:Created: 2015-08-27


Abstract
--------

NumPy's "advanced" indexing support for indexing array with other arrays is
one of its most powerful and popular features. Unfortunately, the existing
rules for advanced indexing with multiple array indices are typically confusing
to both new, and in many cases even old, users of NumPy. Here we propose an
overhaul and simplification of advanced indexing, including two new "indexer"
attributes ``oindex`` and ``vindex`` to facilitate explicit indexing.

Background
----------

Existing indexing operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy arrays currently support a flexible range of indexing operations:

- "Basic" indexing involving only slices, integers, ``np.newaxis`` and ellipsis
  (``...``), e.g., ``x[0, :3, np.newaxis]`` for selecting the first element
  from the 0th axis, the first three elements from the 1st axis and inserting a
  new axis of size 1 at the end. Basic indexing always return a view of the
  indexed array's data.
- "Advanced" indexing, also called "fancy" indexing, includes all cases where
  arrays are indexed by other arrays. Advanced indexing always makes a copy:

  - "Boolean" indexing by boolean arrays, e.g., ``x[x > 0]`` for
    selecting positive elements.
  - "Vectorized" indexing by one or more integer arrays, e.g., ``x[[0, 1]]``
    for selecting the first two elements along the first axis. With multiple
    arrays, vectorized indexing uses broadcasting rules to combine indices along
    multiple dimensions. This allows for producing a result of arbitrary shape
    with arbitrary elements from the original arrays.
  - "Mixed" indexing involving any combinations of the other advancing types.
    This is no more powerful than vectorized indexing, but is sometimes more
    convenient.

For clarity, we will refer to these existing rules as "legacy indexing".
This is only a high-level summary; for more details, see NumPy's documentation
and and `Examples` below.

Outer indexing
~~~~~~~~~~~~~~

One broadly useful class of indexing operations is not supported:

- "Outer" or orthogonal indexing treats one-dimensional arrays equivalently to
  slices for determining output shapes. The rule for outer indexing is that the
  result should be equivalent to independently indexing along each dimension
  with integer or boolean arrays as if both the indexed and indexing arrays
  were one-dimensional. This form of indexing is familiar to many users of other
  programming languages such as MATLAB, Fortran and R.

The reason why NumPy omits support for outer indexing is that the rules for
outer and vectorized conflict. Consider indexing a 2D array by two 1D integer
arrays, e.g., ``x[[0, 1], [0, 1]]``:

- Outer indexing is equivalent to combining multiple integer indices with
  ``itertools.product()``. The result in this case is another 2D array with
  all combinations of indexed elements, e.g.,
  ``np.array([[x[0, 0], x[0, 1]], [x[1, 0], x[1, 1]]])``
- Vectorized indexing is equivalent to combining multiple integer indices with
  ``zip()``. The result in this case is a 1D array containing the diagonal
  elements, e.g., ``np.array([x[0, 0], x[1, 1]])``.

This difference is a frequent stumbling block for new NumPy users. The outer
indexing model is easier to understand, and is a natural generalization of
slicing rules. But NumPy instead chose to support vectorized indexing, because
it is strictly more powerful.

It is always possible to emulate outer indexing by vectorized indexing with
the right indices. To make this easier, NumPy includes utility objects and
functions such as ``np.ogrid`` and ``np.ix_``, e.g.,
``x[np.ix_([0, 1], [0, 1])]``. However, there are no utilities for emulating
fully general/mixed outer indexing, which could unambiguously allow for slices,
integers, and 1D boolean and integer arrays.

Mixed indexing
~~~~~~~~~~~~~~

NumPy's existing rules for combining multiple types of indexing in the same
operation are quite complex, involving a number of edge cases.

One reason why mixed indexing is particularly confusing is that at first glance
the result works deceptively like outer indexing. Returning to our example of a
2D array, both ``x[:2, [0, 1]]`` and ``x[[0, 1], :2]`` return 2D arrays with
axes in the same order as the original array.

However, as soon as two or more non-slice objects (including integers) are
introduced, vectorized indexing rules apply. The axes introduced by the array
indices are at the front, unless all array indices are consecutive, in which
case NumPy deduces where the user "expects" them to be. Consider indexing a 3D
array ``arr`` with shape ``(X, Y, Z)``:

1. ``arr[:, [0, 1], 0]`` has shape ``(X, 2)``.
2. ``arr[[0, 1], 0, :]`` has shape ``(2, Z)``.
3. ``arr[0, :, [0, 1]]`` has shape ``(2, Y)``, not ``(Y, 2)``!

These first two cases are intuitive and consistent with outer indexing, but
this last case is quite surprising, even to many higly experienced NumPy users.

Mixed cases involving multiple array indices are also surprising, and only
less problematic because the current behavior is so useless that it is rarely
encountered in practice. When a boolean array index is mixed with another boolean or
integer array, boolean array is converted to integer array indices (equivalent
to ``np.nonzero()``) and then broadcast. For example, indexing a 2D array of
size ``(2, 2)`` like ``x[[True, False], [True, False]]`` produces a 1D vector
with shape ``(1,)``, not a 2D sub-matrix with shape ``(1, 1)``.

Mixed indexing seems so tricky that it is tempting to say that it never should
be used. However, it is not easy to avoid, because NumPy implicitly adds full
slices if there are fewer indices than the full dimensionality of the indexed
array. This means that indexing a 2D array like `x[[0, 1]]`` is equivalent to
``x[[0, 1], :]``. These cases are not surprising, but they constrain the
behavior of mixed indexing.

Indexing in other Python array libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indexing is a useful and widely recognized mechanism for accessing
multi-dimensional array data, so it is no surprise that many other libraries in
the scientific Python ecosystem also support array indexing.

Unfortunately, the full complexity of NumPy's indexing rules mean that it is
both challenging and undesirable for other libraries to copy its behavior in all
of its nuance. The only full implementation of NumPy-style indexing is NumPy
itself. This includes projects like dask.array and h5py, which support *most*
types of array indexing in some form, and otherwise attempt to copy NumPy's API
exactly.

Vectorized indexing in particular can be challenging to implement with array
storage backends not based on NumPy. In contrast, indexing by 1D arrays along
at least one dimension in the style of outer indexing is much more acheivable.
This has led many libraries (including dask and h5py) to attempt to define a
safe subset of NumPy-style indexing that is equivalent to outer indexing, e.g.,
by only allowing indexing with an array along at most one dimension. However,
this is quite challenging to do correctly in a general enough way to be useful.
For example, the current versions of dask and h5py both handle mixed indexing
in case 3 above inconsistently with NumPy. This is quite likely to lead to
bugs.

These inconsistencies, in addition to the broader challenge of implementing
every type of indexing logic, make it challenging to write high-level array
libraries like xarray or dask.array that can interchangeably index many types of
array storage. In contrast, explicit APIs for outer and vectorized indexing in
NumPy would provide a model that external libraries could reliably emulate, even
if they don't support every type of indexing.

High level changes
------------------

Inspired by multiple "indexer" attributes for controlling different types
of indexing behavior in pandas, we propose to:

1. Introduce ``arr.oindex[indices]`` which allows array indices, but
   uses outer indexing logic.
2. Introduce ``arr.vindex[indices]`` which use the current
   "vectorized"/broadcasted logic but with two differences from
   legacy indexing:
       
   * Boolean indices are not supported. All indices must be integers,
     integer arrays or slices.
   * The integer index result dimensions are always the first axes
     of the result array. No transpose is done, even for a single
     integer array index.

3. Plain indexing on arrays will start to give warnings and eventually
   errors in cases where one of the explicit indexers should be preferred:

   * First, in all cases where legacy and outer indexing would give
     different results.
   * Later, potentially in all cases involving an integer array.

These constraints are sufficient for making indexing generally consistent
with expectations and providing a less surprising learning curve with
``oindex``.

Note that all things mentioned here apply both for assignment as well as
subscription.

Understanding these details is *not* easy. The `Examples` section in the
discussion gives code examples.
And the hopefully easier `Motivational Example` provides some
motivational use-cases for the general ideas and is likely a good start for
anyone not intimately familiar with advanced indexing.


Detailed Description
--------------------

Proposed rules
~~~~~~~~~~~~~~

From the three problems noted above some expectations for NumPy can
be deduced:

1. There should be a prominent outer/orthogonal indexing method such as
   ``arr.oindex[indices]``.

2. Considering how confusing vectorized/fancy indexing can be, it should
   be possible to be made more explicitly (e.g. ``arr.vindex[indices]``).

3. A new ``arr.vindex[indices]`` method, would not be tied to the
   confusing transpose rules of fancy indexing, which is for example
   needed for the simple case of a single advanced index. Thus,
   no transposing should be done. The axes created by the integer array
   indices are always inserted at the front, even for a single index.

4. Boolean indexing is conceptionally outer indexing. Broadcasting
   together with other advanced indices in the manner of legacy
   indexing is generally not helpful or well defined.
   A user who wishes the "``nonzero``" plus broadcast behaviour can thus
   be expected to do this manually. Thus, ``vindex`` does not need to
   support boolean index arrays.

5. An ``arr.legacy_index`` attribute should be implemented to support
   legacy indexing. This gives a simple way to update existing codebases
   using legacy indexing, which will make the deprecation of plain indexing
   behavior easier. The longer name ``legacy_index`` is intentionally chosen
   to be explicit and discourage its use in new code.

6. Plain indexing ``arr[...]`` should return an error for ambiguous cases.
   For the beginning, this probably means cases where ``arr[ind]`` and
   ``arr.oindex[ind]`` return different results give deprecation warnings.
   This includes every use of vectorized indexing with multiple integer arrays.
   Due to the transposing behaviour, this means that``arr[0, :, index_arr]``
   will be deprecated, but ``arr[:, 0, index_arr]`` will not for the time being.

7. To ensure that existing subclasses of `ndarray` that override indexing
   do not inadvertently revert to default behavior for indexing attributes,
   these attribute should have explicit checks that disable them if
   ``__getitem__`` or ``__setitem__`` has been overriden.

Unlike plain indexing, the new indexing attributes are explicitly aimed
at higher dimensional indexing, several additional changes should be implemented:

* The indexing attributes will enforce exact dimension and indexing match.
  This means that no implicit ellipsis (``...``) will be added. Unless
  an ellipsis is present the indexing expression will thus only work for
  an array with a specific number of dimensions.
  This makes the expression more explicit and safeguards against wrong
  dimensionality of arrays.
  There should be no implications for "duck typing" compatibility with
  builtin Python sequences, because Python sequences only support a limited
  form of "basic indexing" with integers and slices.

* The current plain indexing allows for the use of non-tuples for
  multi-dimensional indexing such as ``arr[[slice(None), 2]]``.
  This creates some inconsistencies and thus the indexing attributes
  should only allow plain python tuples for this purpose.
  (Whether or not this should be the case for plain indexing is a
  different issue.)

* The new attributes should not use getitem to implement setitem,
  since it is a cludge and not useful for vectorized
  indexing. (not implemented yet)


Open Questions
~~~~~~~~~~~~~~

* The names ``oindex``, ``vindex`` and ``legacy_index`` are just suggestions at
  the time of writing this, another name NumPy has used for something like
  ``oindex`` is ``np.ix_``. See also below.

* ``oindex`` and ``vindex`` could always return copies, even when no array
  operation occurs. One argument for allowing a view return is that this way
  ``oindex`` can be used as a general index replacement.
  However, there is one argument for returning copies. It is possible for
  ``arr.vindex[array_scalar, ...]``, where ``array_scalar`` should be
  a 0-D array but is not, since 0-D arrays tend to be converted.
  Copying always "fixes" this possible inconsistency.

* The final state to morph plain indexing in is not fixed in this PEP.
  It is for example possible that `arr[index]`` will be equivalent to
  ``arr.oindex`` at some point in the future.
  Since such a change will take years, it seems unnecessary to make
  specific decisions at this time.

* The proposed changes to plain indexing could be postponed indefinitely or
  not taken in order to not break or force major fixes to existing code bases.


Alternative Names
~~~~~~~~~~~~~~~~~

Possible names suggested (more suggestions will be added).

==============  ============ ========
**Orthogonal**  oindex       oix
**Vectorized**  vindex       vix
**Legacy**      legacy_index l/findex
==============  ============ ========


Subclasses
~~~~~~~~~~

Subclasses are a bit problematic in the light of these changes. There are
some possible solutions for this. For most subclasses (those which do not
provide ``__getitem__`` or ``__setitem__``) the special attributes should
just work. Subclasses that *do* provide it must be updated accordingly
and should preferably not subclass ``oindex`` and ``vindex``.

All subclasses will inherit the attributes, however, the implementation
of ``__getitem__`` on these attributes should test
``subclass.__getitem__ is ndarray.__getitem__``. If not, the
subclass has special handling for indexing and ``NotImplementedError``
should be raised, requiring that the indexing attributes is also explicitly
overwritten. Likewise, implementations of ``__setitem__`` should check to see
if ``__setitem__`` is overriden.

A further question is how to facilitate implementing the special attributes.
Also there is the weird functionality where ``__setitem__`` calls
``__getitem__`` for non-advanced indices. It might be good to avoid it for
the new attributes, but on the other hand, that may make it even more
confusing.

To facilitate implementations we could provide functions similar to
``operator.itemgetter`` and ``operator.setitem`` for the attributes.
Possibly a mixin could be provided to help implementation. These improvements
are not essential to the initial implementation, so they are saved for
future work.

Implementation
--------------

Implementation would start with writing special indexing objects available
through ``arr.oindex``, ``arr.vindex``, and ``arr.legacy_index`` to allow these
indexing operations. Also, we would need to start to deprecate those plain index
operations which are not ambiguous.
Furthermore, the NumPy code base will need to use the new attributes and
tests will have to be adapted.


Backward compatibility
----------------------

As a new feature, no backward compatibility issues with the new ``vindex``
and ``oindex`` attributes would arise.

To facilitate backwards compatibility as much as possible, we expect a long
deprecation cycle for legacy indexing behavior and propose the new
``legacy_index`` attribute.

Some forward compatibility issues with subclasses that do not specifically
implement the new methods may arise.


Alternatives
------------

NumPy may not choose to offer these different type of indexing methods, or
choose to only offer them through specific functions instead of the proposed
notation above.

We don't think that new functions are a good alternative, because indexing
notation ``[]`` offer some syntactic advantages in Python (i.e., direct
creation of slice objects) compared to functions.

A more reasonable alternative would be write new wrapper objects for alternative
indexing with functions rather than methods (e.g., ``np.oindex(arr)[indices]``
instead of ``arr.oindex[indices]``). Functionally, this would be equivalent,
but indexing is such a common operation that we think it is important to
minimize syntax and worth implementing it directly on `ndarray` objects
themselves. Indexing attributes also define a clear interface that is easier
for alternative array implementations to copy, nonwithstanding ongoing
efforts to make it easier to override NumPy functions [2]_.

Discussion
----------

The original discussion about vectorized vs outer/orthogonal indexing arose
on the NumPy mailing list:

 * https://mail.python.org/pipermail/numpy-discussion/2015-April/072550.html

Some discussion can be found on the original pull request for this NEP:

 * https://github.com/numpy/numpy/pull/6256

Python implementations of the indexing operations can be found at:

 * https://github.com/numpy/numpy/pull/5749
 * https://gist.github.com/shoyer/c700193625347eb68fee4d1f0dc8c0c8


Examples
~~~~~~~~

Since the various kinds of indexing is hard to grasp in many cases, these
examples hopefully give some more insights. Note that they are all in terms
of shape.
In the examples, all original dimensions have 5 or more elements,
advanced indexing inserts smaller dimensions.
These examples may be hard to grasp without working knowledge of advanced
indexing as of NumPy 1.9.

Example array::

    >>> arr = np.ones((5, 6, 7, 8))


Legacy fancy indexing
---------------------

Note that the same result can be achieved with ``arr.legacy_index``, but the
"future error" will still work in this case.

Single index is transposed (this is the same for all indexing types)::

    >>> arr[[0], ...].shape
    (1, 6, 7, 8)
    >>> arr[:, [0], ...].shape
    (5, 1, 7, 8)


Multiple indices are transposed *if* consecutive::

    >>> arr[:, [0], [0], :].shape  # future error
    (5, 1, 8)
    >>> arr[:, [0], :, [0]].shape  # future error
    (1, 5, 7)


It is important to note that a scalar *is* integer array index in this sense
(and gets broadcasted with the other advanced index)::

    >>> arr[:, [0], 0, :].shape
    (5, 1, 8)
    >>> arr[:, [0], :, 0].shape  # future error (scalar is "fancy")
    (1, 5, 7)


Single boolean index can act on multiple dimensions (especially the whole
array). It has to match (as of 1.10. a deprecation warning) the dimensions.
The boolean index is otherwise identical to (multiple consecutive) integer
array indices::

    >>> # Create boolean index with one True value for the last two dimensions:
    >>> bindx = np.zeros((7, 8), dtype=np.bool_)
    >>> bindx[0, 0] = True
    >>> arr[:, 0, bindx].shape
    (5, 1)
    >>> arr[0, :, bindx].shape
    (1, 6)


The combination with anything that is not a scalar is confusing, e.g.::

    >>> arr[[0], :, bindx].shape  # bindx result broadcasts with [0]
    (1, 6)
    >>> arr[:, [0, 1], bindx].shape  # IndexError


Outer indexing
--------------

Multiple indices are "orthogonal" and their result axes are inserted 
at the same place (they are not broadcasted)::

    >>> arr.oindex[:, [0], [0, 1], :].shape
    (5, 1, 2, 8)
    >>> arr.oindex[:, [0], :, [0, 1]].shape
    (5, 1, 7, 2)
    >>> arr.oindex[:, [0], 0, :].shape
    (5, 1, 8)
    >>> arr.oindex[:, [0], :, 0].shape
    (5, 1, 7)


Boolean indices results are always inserted where the index is::

    >>> # Create boolean index with one True value for the last two dimensions:
    >>> bindx = np.zeros((7, 8), dtype=np.bool_)
    >>> bindx[0, 0] = True
    >>> arr.oindex[:, 0, bindx].shape
    (5, 1)
    >>> arr.oindex[0, :, bindx].shape
    (6, 1)


Nothing changed in the presence of other advanced indices since::

    >>> arr.oindex[[0], :, bindx].shape
    (1, 6, 1)
    >>> arr.oindex[:, [0, 1], bindx].shape
    (5, 2, 1)


Vectorized/inner indexing
-------------------------

Multiple indices are broadcasted and iterated as one like fancy indexing,
but the new axes are always inserted at the front::

    >>> arr.vindex[:, [0], [0, 1], :].shape
    (2, 5, 8)
    >>> arr.vindex[:, [0], :, [0, 1]].shape
    (2, 5, 7)
    >>> arr.vindex[:, [0], 0, :].shape
    (1, 5, 8)
    >>> arr.vindex[:, [0], :, 0].shape
    (1, 5, 7)


Boolean indices results are always inserted where the index is, exactly
as in ``oindex`` given how specific they are to the axes they operate on::

    >>> # Create boolean index with one True value for the last two dimensions:
    >>> bindx = np.zeros((7, 8), dtype=np.bool_)
    >>> bindx[0, 0] = True
    >>> arr.vindex[:, 0, bindx].shape
    (5, 1)
    >>> arr.vindex[0, :, bindx].shape
    (6, 1)


But other advanced indices are again transposed to the front::

    >>> arr.vindex[[0], :, bindx].shape
    (1, 6, 1)
    >>> arr.vindex[:, [0, 1], bindx].shape
    (2, 5, 1)


Motivational Example
~~~~~~~~~~~~~~~~~~~~

Imagine having a data acquisition software storing ``D`` channels and
``N`` datapoints along the time. She stores this into an ``(N, D)`` shaped
array. During data analysis, we needs to fetch a pool of channels, for example
to calculate a mean over them.

This data can be faked using::

    >>> arr = np.random.random((100, 10))

Now one may remember indexing with an integer array and find the correct code::

    >>> group = arr[:, [2, 5]]
    >>> mean_value = arr.mean()

However, assume that there were some specific time points (first dimension
of the data) that need to be specially considered. These time points are
already known and given by::

    >>> interesting_times = np.array([1, 5, 8, 10], dtype=np.intp)

Now to fetch them, we may try to modify the previous code::

    >>> group_at_it = arr[interesting_times, [2, 5]]
    IndexError: Ambiguous index, use `.oindex` or `.vindex`

An error such as this will point to read up the indexing documentation.
This should make it clear, that ``oindex`` behaves more like slicing.
So, out of the different methods it is the obvious choice
(for now, this is a shape mismatch, but that could possibly also mention
``oindex``)::

    >>> group_at_it = arr.oindex[interesting_times, [2, 5]]

Now of course one could also have used ``vindex``, but it is much less
obvious how to achieve the right thing!::

    >>> reshaped_times = interesting_times[:, np.newaxis]
    >>> group_at_it = arr.vindex[reshaped_times, [2, 5]]


One may find, that for example our data is corrupt in some places.
So, we need to replace these values by zero (or anything else) for these
times. The first column may for example give the necessary information,
so that changing the values becomes easy remembering boolean indexing::

    >>> bad_data = arr[:, 0] > 0.5
    >>> arr[bad_data, :] = 0  # (corrupts further examples)

Again, however, the columns may need to be handled more individually (but in
groups), and the ``oindex`` attribute works well::

    >>> arr.oindex[bad_data, [2, 5]] = 0

Note that it would be very hard to do this using legacy fancy indexing.
The only way would be to create an integer array first::

    >>> bad_data_indx = np.nonzero(bad_data)[0]
    >>> bad_data_indx_reshaped = bad_data_indx[:, np.newaxis]
    >>> arr[bad_data_indx_reshaped, [2, 5]]

In any case we can use only ``oindex`` to do all of this without getting
into any trouble or confused by the whole complexity of advanced indexing.

But, some new features are added to the data acquisition. Different sensors
have to be used depending on the times. Let us assume we already have
created an array of indices::

    >>> correct_sensors = np.random.randint(10, size=(100, 2))

Which lists for each time the two correct sensors in an ``(N, 2)`` array.

A first try to achieve this may be ``arr[:, correct_sensors]`` and this does
not work. It should be clear quickly that slicing cannot achieve the desired
thing. But hopefully users will remember that there is ``vindex`` as a more
powerful and flexible approach to advanced indexing.
One may, if trying ``vindex`` randomly, be confused about::

    >>> new_arr = arr.vindex[:, correct_sensors]

which is neither the same, nor the correct result (see transposing rules)!
This is because slicing works still the same in ``vindex``. However, reading
the documentation and examples, one can hopefully quickly find the desired
solution::

    >>> rows = np.arange(len(arr))
    >>> rows = rows[:, np.newaxis]  # make shape fit with correct_sensors
    >>> new_arr = arr.vindex[rows, correct_sensors]
    
At this point we have left the straight forward world of ``oindex`` but can
do random picking of any element from the array. Note that in the last example
a method such as mentioned in the ``Related Questions`` section could be more
straight forward. But this approach is even more flexible, since ``rows``
does not have to be a simple ``arange``, but could be ``interesting_times``::

    >>> interesting_times = np.array([0, 4, 8, 9, 10])
    >>> correct_sensors_at_it = correct_sensors[interesting_times, :]
    >>> interesting_times_reshaped = interesting_times[:, np.newaxis]
    >>> new_arr_it = arr[interesting_times_reshaped, correct_sensors_at_it]

Truly complex situation would arise now if you would for example pool ``L``
experiments into an array shaped ``(L, N, D)``. But for ``oindex`` this should
not result into surprises. ``vindex``, being more powerful, will quite
certainly create some confusion in this case but also cover pretty much all
eventualities.


Copyright
---------

This document is placed under the CC0 1.0 Universell (CC0 1.0) Public Domain Dedication [1]_.


References and Footnotes
------------------------

.. [1] To the extent possible under law, the person who associated CC0 
   with this work has waived all copyright and related or neighboring
   rights to this work. The CC0 license may be found at
   https://creativecommons.org/publicdomain/zero/1.0/
.. [2] e.g., see NEP 18,
   http://www.numpy.org/neps/nep-0018-array-function-protocol.html
