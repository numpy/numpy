.. _routines.io:

Input and output
================

.. currentmodule:: numpy

NumPy binary files (npy, npz)
-----------------------------
.. autosummary::
   :toctree: generated/

   load
   save
   savez
   savez_compressed
   lib.npyio.NpzFile

The format of these binary file types is documented in
:py:mod:`numpy.lib.format`

Text files
----------
.. autosummary::
   :toctree: generated/

   loadtxt
   savetxt
   genfromtxt
   fromregex
   fromstring
   ndarray.tofile
   ndarray.tolist

Raw binary files
----------------

.. autosummary::

   fromfile
   ndarray.tofile

String formatting
-----------------
.. autosummary::
   :toctree: generated/

   array2string
   array_repr
   array_str
   format_float_positional
   format_float_scientific

Memory mapping files
--------------------
.. autosummary::
   :toctree: generated/

   memmap
   lib.format.open_memmap

Text formatting options
-----------------------
.. autosummary::
   :toctree: generated/

   set_printoptions
   get_printoptions
   printoptions

Base-n representations
----------------------
.. autosummary::
   :toctree: generated/

   binary_repr
   base_repr

Data sources
------------
.. autosummary::
   :toctree: generated/

   lib.npyio.DataSource

Binary format description
-------------------------
.. autosummary::
   :toctree: generated/

   lib.format
