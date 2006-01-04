"""\
Core Linear Algebra Tools
===========

 Linear Algebra Basics:

   inv        --- Find the inverse of a square matrix
   solve      --- Solve a linear system of equations
   det        --- Find the determinant of a square matrix
   lstsq      --- Solve linear least-squares problem
   pinv       --- Pseudo-inverse (Moore-Penrose) using lstsq

 Eigenvalues and Decompositions:

   eig        --- Find the eigenvalues and vectors of a square matrix
   eigh       --- Find the eigenvalues and eigenvectors of a Hermitian matrix
   eigvals    --- Find the eigenvalues of a square matrix
   eigvalsh   --- Find the eigenvalues of a Hermitian matrix. 
   svd        --- Singular value decomposition of a matrix
   cholesky   --- Cholesky decomposition of a matrix   

"""

depends = ['core']

