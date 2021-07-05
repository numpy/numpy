.. _basics.indexing:

***************
Indexing basics
***************

.. seealso::

   :ref:`Indexing <arrays.indexing>`

   :ref:`Indexing routines <routines.indexing>`

Array indexing refers to any use of the square brackets ([]) to index
array values. There are many options to indexing, which give NumPy
indexing great power, but with power comes some complexity and the
potential for confusion. This section is just an overview of the
various options and issues related to indexing. Further details can be
found in the relevant sections of :ref:`arrays.indexing`.


Basic indexing
==============

Basic indexing will be triggered through single element indexing or
slicing and striding.

Single element indexing works
exactly like that for other standard Python sequences but has been expanded
to support  multidimensional indexing
for multidimensional NumPy arrays. For more details and examples, see
:ref:`single-element-indexing`.

It is possible to slice and stride arrays to extract arrays of the
same number of dimensions, but of different sizes than the original.
The slicing and striding work exactly the same way it does for lists
and tuples except that they can be applied to multiple dimensions as
well. See :ref:`basic-slicing-and-indexing` for more information and examples.


Advanced indexing
=================

Arrays can be indexed with other arrays to
select lists of values out of arrays into new arrays. There are
two different ways of accomplishing this. One uses one or more arrays
of index values. The other involves giving a boolean array of the proper
shape to indicate the values to be selected. Index arrays are a very
powerful tool that allows one to avoid looping over individual elements in
arrays and thus greatly improve performance. See :ref:`advanced-indexing`
for more information and examples.


Other indexing options
======================

:ref:`arrays.indexing.fields`, :ref:`flat-iterator-indexing` are a couple
of other indexing options.


.. _assigning-values-to-indexed-arrays:

Assigning values to indexed arrays
==================================

As mentioned, one can select a subset of an array to assign to using
a single index, slices, and index and mask arrays. The value being
assigned to the indexed array must be shape consistent (the same shape
or broadcastable to the shape the index produces). For example, it is
permitted to assign a constant to a slice: ::

 >>> x = np.arange(10)
 >>> x[2:7] = 1

or an array of the right size: ::

 >>> x[2:7] = np.arange(5)

Note that assignments may result in changes if assigning
higher types to lower types (like floats to ints) or even
exceptions (assigning complex to floats or ints): ::

 >>> x[1] = 1.2
 >>> x[1]
 1
 >>> x[1] = 1.2j
 TypeError: can't convert complex to int


Unlike some of the references (such as array and mask indices)
assignments are always made to the original data in the array
(indeed, nothing else would make sense!). Note though, that some
actions may not work as one may naively expect. This particular
example is often surprising to people: ::

 >>> x = np.arange(0, 50, 10)
 >>> x
 array([ 0, 10, 20, 30, 40])
 >>> x[np.array([1, 1, 3, 1])] += 1
 >>> x
 array([ 0, 11, 20, 31, 40])

Where people expect that the 1st location will be incremented by 3.
In fact, it will only be incremented by 1. The reason is that
a new array is extracted from the original (as a temporary) containing
the values at 1, 1, 3, 1, then the value 1 is added to the temporary,
and then the temporary is assigned back to the original array. Thus
the value of the array at x[1]+1 is assigned to x[1] three times,
rather than being incremented 3 times.

.. _dealing-with-variable-indices:

Dealing with variable numbers of indices within programs
========================================================

The indexing syntax is very powerful but limiting when dealing with
a variable number of indices. For example, if you want to write
a function that can handle arguments with various numbers of
dimensions without having to write special case code for each
number of possible dimensions, how can that be done? If one
supplies to the index a tuple, the tuple will be interpreted
as a list of indices. For example (using the previous definition
for the array z): ::

 >>> indices = (1,1,1,1)
 >>> z[indices]
 40

So one can use code to construct tuples of any number of indices
and then use these within an index.

Slices can be specified within programs by using the slice() function
in Python. For example: ::

 >>> indices = (1,1,1,slice(0,2)) # same as [1,1,1,0:2]
 >>> z[indices]
 array([39, 40])

Likewise, ellipsis can be specified by code by using the Ellipsis
object: ::

 >>> indices = (1, Ellipsis, 1) # same as [1,...,1]
 >>> z[indices]
 array([[28, 31, 34],
        [37, 40, 43],
        [46, 49, 52]])

For this reason, it is possible to use the output from the 
:meth:`np.nonzero() <ndarray.nonzero>` function directly as an index since
it always returns a tuple of index arrays.

Because the special treatment of tuples, they are not automatically
converted to an array as a list would be. As an example: ::

 >>> z[[1,1,1,1]] # produces a large array
 array([[[[27, 28, 29],
          [30, 31, 32], ...
 >>> z[(1,1,1,1)] # returns a single value
 40


