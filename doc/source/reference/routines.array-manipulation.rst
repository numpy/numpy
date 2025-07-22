Array manipulation routines
***************************

.. currentmodule:: numpy

Basic operations
================
.. autosummary::
   :toctree: generated/

    copyto
    ndim
    shape
    size

Changing array shape
====================
.. autosummary::
   :toctree: generated/

   reshape
   ravel
   ndarray.flat
   ndarray.flatten

Transpose-like operations
=========================
.. autosummary::
   :toctree: generated/

   moveaxis
   rollaxis
   swapaxes
   ndarray.T
   transpose
   permute_dims
   matrix_transpose (Array API compatible)

Changing number of dimensions
=============================
.. autosummary::
   :toctree: generated/

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast
   broadcast_to
   broadcast_arrays
   expand_dims
   squeeze

Changing kind of array
======================
.. autosummary::
   :toctree: generated/

   asarray
   asanyarray
   asmatrix
   asfortranarray
   ascontiguousarray
   asarray_chkfinite
   require

Joining arrays
==============
.. autosummary::
   :toctree: generated/

   concatenate
   concat
   stack
   block
   vstack
   hstack
   dstack
   column_stack

Splitting arrays
================
.. autosummary::
   :toctree: generated/

   split
   array_split
   dsplit
   hsplit
   vsplit
   unstack

Tiling arrays
=============
.. autosummary::
   :toctree: generated/

   tile
   repeat

Adding and removing elements
============================
.. autosummary::
   :toctree: generated/

   delete
   insert
   append
   resize
   trim_zeros
   unique
   pad

Rearranging elements
====================
.. autosummary::
   :toctree: generated/

   flip
   fliplr
   flipud
   roll
   rot90
   shifted
