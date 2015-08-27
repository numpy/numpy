==========================================================
Implementing intuitive and full featured advanced indexing
==========================================================

:Author: Sebastian Berg
:Status: Draft
:Type: Standards Track
:Created: 2015-08-27


Abstract
--------

Advanced indexing with multiple array indices is typically confusing to
both new, and in many cases even old, users of NumPy. To avoid this problem
and allow for more and clearer features, we propose to:

1. Introduce ``arr.oindex[indices]`` which allows advanced indices, but
   uses outer indexing logic.
2. Introduce ``arr.vindex[indices]`` which use the current
   "vectorized"/broadcasted logic but with two differences from
   fancy indexing:
       
   1. Boolean indices always use the outer indexing logic.
      (Multi dimensional booleans should be allowed).
   2. The integer index result dimensions are always the first axes
      of the result array. No transpose is done, even for a single
      integer array index.

3. Plain indexing on the array will only give warnings and eventually
   errors either:
     
   * when there is ambiguity between legacy fancy and outer indexing
     (note that ``arr[[1, 2], :, 0]`` is such a case, an integer
     can be the "second" integer index array),
   * when any integer index array is present (possibly additional for
     more then one boolean index array).

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

Old style advanced indexing with multiple array (boolean or integer) indices,
also called "fancy indexing", tends to be very confusing for new users.
While fancy (or legacy) indexing is useful in many cases one would naively
assume that the result of multiple 1-d ranges is analogous to multiple
slices along each dimension (also called "outer indexing").

However, legacy fancy indexing with multiple arrays broadcasts these arrays
into a single index over multiple dimensions. There are three main points
of confusion when multiple array indices are involved:

1. Most new users will usually expect outer indexing (consistent with
   slicing). This is also the most common way of handling this in other
   packages or languages.
2. The axes introduced by the array indices are at the front, unless
   all array indices are consecutive, in which case one can deduce where
   the user "expects" them to be:

   * `arr[:, [0, 1], :, [0, 1]]` will have the first dimension shaped 2.
   * `arr[:, [0, 1], [0, 1]]` will have the second dimension shaped 2.

3. When a boolean array index is mixed with another boolean or integer
   array, the result is very hard to understand (the boolean array is
   converted to integer array indices and then broadcast), and hardly
   useful.
   There is no well defined broadcast for booleans, so that boolean
   indices are logically always "outer" type indices.


Furthermore, since other packages use outer type logic, enabling its
use in NumPy, will allow simpler cross package code.


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

4. Boolean indexing is conceptionally outer indexing. A broadcasting
   together with other advanced indices in the manner of legacy
   "fancy indexing" is generally not helpful or well defined.
   A user who wishes the "``nonzero``" plus broadcast behaviour can thus
   be expected to do this manually. Thus, ``vindex`` would still use
   outer type indexing for boolean index arrays.
   Note that using this rule a single boolean index can still index into
   multiple dimensions at once.

5. An ``arr.lindex`` or ``arr.findex`` should likely be implemented to allow
   legacy fancy indexing indefinetly. This also gives a simple way to
   update fancy indexing code making deprecations to plain indexing
   easier.

6. Plain indexing ``arr[...]`` should return an error for ambiguous cases.
   For the beginning, this probably means cases where ``arr[ind]`` and
   ``arr.oindex[ind]`` return different results give deprecation warnings.
   Due to the transposing behaviour, this means that``arr[0, :, index_arr]``
   will be deprecated, but ``arr[:, 0, index_arr]`` will not for the time
   being.

Unlike plain indexing, the new indexing attributes are explicitly aimed
at higher dimensional indexing, two more changes should be implemented:

* The indexing attributes will enforce exact dimension and indexing match.
  This means that no implicit ellipsis (``...``) will be added. Unless
  an ellipsis is present the indexing expression will thus only work for
  an array with a specific number of dimensions.
  This makes the expression more explicit and safeguards against wrong
  dimensionality of arrays.

* The current plain indexing allows for the use of non-tuples for
  multi-dimensional indexing such as ``arr[[slice(None), 2]]``.
  This creates some inconsistencies and thus the indexing attributes
  should only allow plain python tuples for this purpose.
  (Whether or not this should be the case for plain indexing is a
  different issue.)

* The new attributes should not do the getitem to implement setitem
  branch, since it is a cludge and not useful for vectorized
  indexing. (not implemented yet)


Open Questions
~~~~~~~~~~~~~~

* The names ``oindex`` and ``vindex`` are just suggestions at the time of
  writing this, another name NumPy has used for something like ``oindex``
  is ``np.ix_``. See also below.

* It would be possible to limit the use of boolean indices in ``vindex``,
  assuming that they are rare and to some degree special. Alternatively,
  boolean indices could be broadcasted as in legacy fancy indexing in
  principle.

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

* Should there be/is it possible to have a mechanism to not allow the new
  indexing attributes for subclasses unless specifically implemented?

* Should we try to warn users about the special indexing attributes
  not being implemented? If a subclass has its own indexing, inheriting
  it from ndarray should be wrong.
 


Alternative Names
~~~~~~~~~~~~~~~~~

Possible names suggested (more suggestions will be added).

==============  ======== ======= ============
**Orthogonal**  oindex   oix
**Vectorized**  vindex   vix
**Legacy**      l/findex         legacy_index
==============  ======== ======= ============


Subclasses
~~~~~~~~~~

Subclasses are a bit problematic in the light of these changes. There are
some possible solutions for this. For most subclasses (those which do not
provide ``__getitem__`` or ``__setitem__``) the special attributes should
just work. Subclasses that *do* provide it must be updated accordingly
and should preferably not subclass working versions of these attributes.

All subclasses will inherit the attributes, however, it seems possible
to test ``subclass.__getitem__.__classobj__`` when getting i.e.
``subclass.vindex``. If this is not ``ndarray``, the subclass has special
handling and an ``AttributeError`` can be given.

A further question is how to facilitate implementing the special attributes.
Also there is the weird functionality where ``__setitem__`` calls
``__getitem__`` for non-advanced indices. It might be good to avoid it for
the new attributes, but on the other hand, that may make it even more
confusing.

To facilitate implementations we could provide functions similar to
``operator.itemgetter`` and ``operator.setitem`` for the attributes.
Possibly a MixIn could be provided to help implementation, but further
ideas on this are necessary.

Related Operation
~~~~~~~~~~~~~~~~~

There exist a further indexing or indexing like method. That is the
inverse of a command such as ``np.argmin(arr, axis=axis)``, to pick
the specific elements *along* an axis given an array of (at least
typically) the same size.

These function are added to NumPy in versoin 15 as `take_along_axis` and
`put_along_axis`.


Implementation
--------------

Implementation of a special indexing object available through
``arr.oindex``, ``arr.vindex``, and ``arr.lindex`` to allow these indexing
operations. Also starting to deprecate those plain index operations
which are not ambiguous.
Furthermore, the NumPy code base will need to use the new attributes and
tests will have to be adapted.


Backward compatibility
----------------------

As a new feature, no backward compatibility issues would arise.
Some forward compatibility issues with subclasses that do not specifically
implement the new methods may arise.


Alternatives
------------

NumPy may not choose to offer these different type of indexing methods, or
choose to only offer them through specific function instead of the proposed
notation above.
For variations see also the open questions section above.


Discussion
----------

Some discussion can be found on the pull request:

 * https://github.com/numpy/numpy/pull/6256

A python implementation of the indexing operations can be found at:

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

Note that the same result can be achieved with ``arr.lindex``, but the
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
but the new axes area always inserted at the front::

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
does not have to be a simple ``arange``, but could be ``intersting_times``::

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

