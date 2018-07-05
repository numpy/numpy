Input and output
****************

.. currentmodule:: numpy

NumPy binary files (NPY, NPZ)
-----------------------------
.. autosummary::
   :toctree: generated/

   load
   save
   savez
   savez_compressed

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

Text formatting options
-----------------------
.. autosummary::
   :toctree: generated/

   set_printoptions
   get_printoptions
   set_string_function
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

   DataSource

Binary Format Description
-------------------------
.. autosummary::
   :template: autosummary/minimal_module.rst
   :toctree: generated/

    lib.format
