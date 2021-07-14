.. for doctests
   >>> import numpy as np

.. _arrays.indexing:

********
Indexing
********

.. seealso::

   :ref:`basics.indexing`

   :ref:`Indexing routines <routines.indexing>`

Array indexing refers to any use of the square brackets ([]) to index
array values. There are many options to indexing, which give NumPy
indexing great power, but with power comes some complexity and the
potential for confusion. This section is just an overview of the
various options and issues related to indexing. Further details can be
found in the relevant sections of :ref:`basics.indexing`.


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