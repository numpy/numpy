.. _routines.linalg:

.. module:: numpy.linalg

Linear algebra (:mod:`numpy.linalg`)
====================================

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

The SciPy library also contains a `~scipy.linalg` submodule, and there is
overlap in the functionality provided by the SciPy and NumPy submodules.  SciPy
contains functions not found in `numpy.linalg`, such as functions related to
LU decomposition and the Schur decomposition, multiple ways of calculating the
pseudoinverse, and matrix transcendentals such as the matrix logarithm.  Some
functions that exist in both have augmented functionality in `scipy.linalg`.
For example, `scipy.linalg.eig` can take a second matrix argument for solving
generalized eigenvalue problems.  Some functions in NumPy, however, have more
flexible broadcasting options.  For example, `numpy.linalg.solve` can handle
"stacked" arrays, while `scipy.linalg.solve` accepts only a single square
array as its first argument.

.. note::

   The term *matrix* as it is used on this page indicates a 2d `numpy.array`
   object, and *not* a `numpy.matrix` object. The latter is no longer
   recommended, even for linear algebra. See
   :ref:`the matrix object documentation<matrix-objects>` for
   more information.

The ``@`` operator
------------------

Introduced in NumPy 1.10.0, the ``@`` operator is preferable to
other methods when computing the matrix product between 2d arrays. The
:func:`numpy.matmul` function implements the ``@`` operator.

.. currentmodule:: numpy

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/

   dot
   linalg.multi_dot
   vdot
   vecdot
   linalg.vecdot
   inner
   outer
   matmul
   linalg.matmul (Array API compatible location)
   matvec
   vecmat
   tensordot
   linalg.tensordot (Array API compatible location)
   einsum
   einsum_path
   linalg.matrix_power
   kron
   linalg.cross

Decompositions
--------------
.. autosummary::
   :toctree: generated/

   linalg.cholesky
   linalg.outer
   linalg.qr
   linalg.svd
   linalg.svdvals

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
   linalg.matrix_norm (Array API compatible)
   linalg.vector_norm (Array API compatible)
   linalg.cond
   linalg.det
   linalg.matrix_rank
   linalg.slogdet
   trace
   linalg.trace (Array API compatible)

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

Other matrix operations
-----------------------
.. autosummary::
   :toctree: generated/

   diagonal
   linalg.diagonal (Array API compatible)
   linalg.matrix_transpose (Array API compatible)

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
