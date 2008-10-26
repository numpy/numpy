.. _routines.ma:

Masked array operations
***********************

.. currentmodule:: numpy

Creation
--------

.. autosummary::
   :toctree: generated/

   ma.masked_array

Converting to ndarray
---------------------

.. autosummary::
   :toctree: generated/

   ma.filled
   ma.common_fill_value
   ma.default_fill_value
   ma.masked_array.get_fill_value
   ma.maximum_fill_value
   ma.minimum_fill_value

Inspecting the array
--------------------

.. autosummary::
   :toctree: generated/

   ma.getmask
   ma.getmaskarray
   ma.getdata
   ma.count_masked

Modifying the mask
------------------

.. autosummary::
   :toctree: generated/

   ma.make_mask
   ma.mask_cols
   ma.mask_or
   ma.mask_rowcols
   ma.mask_rows
   ma.harden_mask
   ma.ids
