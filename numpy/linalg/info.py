"""\
Core Linear Algebra Tools
===========

 Linear Algebra Basics:

   norm       --- Vector or matrix norm
   inv        --- Inverse of a square matrix
   solve      --- Solve a linear system of equations
   det        --- Determinant of a square matrix
   lstsq      --- Solve linear least-squares problem
   pinv       --- Pseudo-inverse (Moore-Penrose) using lstsq

 Eigenvalues and Decompositions:

   eig        --- Eigenvalues and vectors of a square matrix
   eigh       --- Eigenvalues and eigenvectors of a Hermitian matrix
   eigvals    --- Eigenvalues of a square matrix
   eigvalsh   --- Eigenvalues of a Hermitian matrix.
   svd        --- Singular value decomposition of a matrix
   cholesky   --- Cholesky decomposition of a matrix

"""

depends = ['core']
