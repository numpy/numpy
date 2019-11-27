.. _routines.linalg:

.. module:: numpy.linalg

Linear algebra (:mod:`numpy.linalg`)
************************************

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient
low level implementations of standard linear algebra algorithms. Those
libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that
take advantage of specialized processor functionality are preferred. Examples
of such libraries are OpenBLAS_, MKL (TM), and ATLAS. Because those libraries
are multithreaded and processor dependent, environmental variables and external
packages such as threadpoolctl_ may be needed to control the number of threads
or specify the processor architecture.

.. _OpenBLAS: https://www.openblas.net/
.. _threadpoolctl: https://github.com/joblib/threadpoolctl

Frequently asked question about `numpy.linalg` include "Why do both NumPy and
SciPy have `linalg` submodules, with duplicated functions?" and "Which should
I use?"  The following is from the SciPy FAQ:

* `scipy.linalg` is a more complete wrapping of Fortran LAPACK using f2py.
* One of the design goals of NumPy was to make it buildable without a Fortran
  compiler, and if you don’t have LAPACK available, NumPy will use its own
  implementation. SciPy requires a Fortran compiler to be built, and heavily
  depends on wrapped Fortran code.
* The ``linalg`` modules in NumPy and SciPy have some common functions but
  with different docstrings, and `scipy.linalg` contains functions not found
  in `numpy.linalg`, such as functions related to LU decomposition and the
  Schur decomposition, multiple ways of calculating the pseudoinverse, and
  matrix transcendentals, like the matrix logarithm. Some functions that exist
  in both have augmented functionality in `scipy.linalg`; for example,
  `scipy.linalg.eig` can take a second matrix argument for solving generalized
  eigenvalue problems.


.. currentmodule:: numpy

Matrix and vector products
--------------------------
.. autosummary::
   :toctree: generated/

   dot
   linalg.multi_dot
   vdot
   inner
   outer
   matmul
   tensordot
   einsum
   einsum_path
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

.. _routines.linalg-broadcasting:

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
