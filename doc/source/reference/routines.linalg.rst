.. _routines.linalg:

Linear algebra (:mod:`numpy.linalg`)
************************************

.. currentmodule:: numpy

Matrix and vector products
--------------------------
.. autosummary::
   :toctree: generated/

   dot
   vdot
   inner
   outer
   matmul
   tensordot
   einsum
   linalg.matrix_power
   kron

Decompositions
--------------
.. autosummary::
   :toctree: generated/

   linalg.cholesky
   linalg.qr
   linalg.svd

Matrix eigenvalues
------------------
.. autosummary::
   :toctree: generated/

   linalg.eig
   linalg.eigh
   linalg.eigvals
   linalg.eigvalsh

Norms and other numbers
-----------------------
.. autosummary::
   :toctree: generated/

   linalg.norm
   linalg.cond
   linalg.det
   linalg.matrix_rank
   linalg.slogdet
   trace

Solving equations and inverting matrices
----------------------------------------
.. autosummary::
   :toctree: generated/

   linalg.solve
   linalg.tensorsolve
   linalg.lstsq
   linalg.inv
   linalg.pinv
   linalg.tensorinv

Exceptions
----------
.. autosummary::
   :toctree: generated/

   linalg.LinAlgError

Linear algebra on several matrices at once
------------------------------------------

.. versionadded:: 1.8.0

Several of the linear algebra routines listed above are able to
compute results for several matrices at once, if they are stacked into
the same array.

This is indicated in the documentation via input parameter
specifications such as ``a : (..., M, M) array_like``. This means that
if for instance given an input array ``a.shape == (N, M, M)``, it is
interpreted as a "stack" of N matrices, each of size M-by-M. Similar
specification applies to return values, for instance the determinant
has ``det : (...)`` and will in this case return an array of shape
``det(a).shape == (N,)``. This generalizes to linear algebra
operations on higher-dimensional arrays: the last 1 or 2 dimensions of
a multidimensional array are interpreted as vectors or matrices, as
appropriate for each operation.
