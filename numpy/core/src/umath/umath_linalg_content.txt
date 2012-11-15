		 gufuncs_linalg, umath_linalg modules
		 ------------------------------------

umath_linalg implements a series of gufuncs that implement several functions
already present in numpy.linalg as generalized universal functions. There are
some extra gufuncs thrown in. This allows broadcasting of calls over full 
vector/matrices of data.

gufuncs_linalg provides a very thin wrapper around umath_linalg so that the
interface resembles more the linalg module. The wrappers are mostly used to
provide keyword parameters that wouldn't be supported using a plain gufunc.
Note that this module is written in Python, importing the umath_linalg 
privately

Usage of gufuncs_linalg over direct use of umath_linalg is recommended. It uses
the kernels in umath_linalg and the speed difference should be minimal.

Testing will be done at the gufuncs_linalg level, and the thin layer may 
implement workarounds around known issues.


			     umath_linalg
			     ------------

			       Contents
The following functions are already implemented:

- inner1d: (n),(n) -> ()
  dot product of the vectors in the inner dimension. Uses BLAS.
- inner2d: (n),(n),(n) -> ()
  u,v,w being the inputs: sum(u[i]*v[i]*w[i]) for all i in n. Copied for 
  convenience from umath_tests
- matrix_multiply: (m,k),(k,n) -> (m,n)
  matrix multiplication. Uses BLAS.
- slogdet: (m,m)->(),()
  sign and logarithm of the determinant of the input. Uses LAPACK.
- det: (m,m)->()
  determinant of the input, computed using slogdet and thus, uses LAPACK.
- eigh_lo: (m,n)->(m)(m,n)
  eigenvalues and eigenvectors of symmetric/hermitian matrices, encoded in the
  lower diagonal. Uses LAPACK.
- eigh_up: (m,m)->(m)(m,m)
  eigenvalues and eigenvectors of symmetric/hermitian matrices, encoded in the
  upper diagonal. Uses LAPACK.
- eigvalsh_lo: (m,m)->(m)
  eigenvalues of symmetric/hermitian matrices, encoded in the lower triangular
  part. Uses LAPACK.
- eigvalsh_up: (m,m)->(m)
  eigenvalues of symmetric/hermitian matrices, encoded in the upper triangular
  part. Uses LAPACK.
- solve: (m,m),(m,n)->(m,n)
  Solve X for a system AX=B, X and B being matrices. Uses LAPACK.
- solve1: (m,m),(m)->(m)
  Solve x for a system Ax=b, x and b being vectors. Uses LAPACK.
- inv: (m,m)->(m,m)
  Compute the inverse of a matrix. Implemented using solve. Uses LAPACK.
- cholesky: (m,m)->(m,m)
  Perform cholesky decomposition of hermitian positive-definite matrices.
  The lower diagonal of the input matrix is used. Uses LAPACK.
- svd_m: (m,n)->(m)
  singular value decomposition of the input matrix. Use this function when
  m<=n and only the singular values are needed. Uses LAPACK.
- svd_n: (m,n)->(n)
  singular value decomposition of the input matrix. Use this function when
  m>=n and only the singular values are needed. Uses LAPACK.
- svd_m_s, svd_m_f, svd_n_s, svd_n_f *
  singular value decomposition resulting in U s V. See note on svd.
- eig: (m,m)->(m),(m,m)
  eigenvalues and eigenvectors of general matrices. Uses LAPACK.
- eigvals; (m,m)->(m)
  eigenvalues of general matrices. Uses LAPACK.
- quadratic_form: (m),(m,n),(n)->()
  computes the quadratic form uQv. Uses BLAS.
- add3: (),(),()->()
  3-way element-wise addition.
- multiply3: (),(),()->()
  3-way element-wise product.
- multiply3_add: (),(),(),()->()
  3-way element-wise product plus addition.
- multply_add: (),(),()->()
  element-wise multiply add.
- multiply_add2: (),(),(),()->()
  element-wise product with 2 additions.
- multiply4: (),(),(),()->()
  4-way element-wise product.
- multiply4_add: (),(),(),(),()->()
  4-way element-wise product plus addition.
- chosolve_lo
  solve a system AX=B where A is symmetric/hermitian (using lapack potrf, potrs on the
  lower triangle)
- chosolve_up
  solve a system AX=B where A is symmetric/hermitian (using lapack potrf, potrs on the
  upper triangle)
- chosolve1_lo
  solve a system Ax=b where A is symmetric/hermitian (using lapack potrf, potrs on the
  lower triangle)
- chosolve1_up
  solve a system Ax=b where A is symmetric/hermitian (using lapack potrf, potrs on the
  upper triangle)
- poinv_lo
  compute the inverse of a positive definite symmetric/hermitian matrix (using lapack 
  ptorf, potri on the lower triangle)
- poinv_up
  compute the inverse of a positive definite symmetric/hermitian matrix (using lapack 
  ptorf, potri on the upper triangle)

		    Note on uniform parameters
There are some configuration parameters in some of the functions in the linalg
module that configure how some computation is to be done. Examples are eigh and
eigvalsh, that take an optional keywork parameter that allows to specify 
whether to use the upper or the lower triangular part (UPLO). That parameter is
considered uniform for all broadcast operations. As in the gufunc interface
there is no way to specify uniform parameters, it has been encoded in the gufunc
name, so for eigh we have the functions eigh_lo and eigh_up (for 'L' and 'U').

Similar techniques have been applied in other places.

			     Note on svd
svd uses uniform parameters to specify the result types it will be producing
(full_matrices and compute_uv in numpy.linalg).
svd is also special in the sense that the signature of the function depends on
the size of the parameters, as for a matrix m by n, the result is a vector with
min(m,n) values. This is not supported by the gufunc harness, so specialized
versions of the gufunc are provided that encode whether m<n or m>n, so they
can specify the signature accordingly. This is also encoded in the gufunc name.

The function to use, in pseudocode, would be:

func = 'svd'
if (m < n):
   func += '_m'
else:
   func += '_n'

if compute_uv:
   if full_matrices:
      func += '_f'
   else:
      func += '_s'


			   *** Changes ***
2012-11-15
    Fixed the bug in matrix_multiply
    Fixed the bug in eig 
2012-11-13
    Created gufuncs_linalg module. This module is a wrapper for the gufuncs to
        make the interface easier to use.
    Added a test module. It test ate the level of gufuncs_linalg interface.
        Several tests added, but still work in progress
    Fixed problem in inner1d and innerwt when the parameter had only one
        dimension
    Fixed several small bugs found while tests were written. Mostly related to
        complex versions of the functions.
    There are some known bugs (working on them):
        matrix_multiply seems not to always work properly with dtype cdouble.
        eigvals not working properly with dtype csingle.
<before 2012-11-13>
    There was a bug in nympy gufunc harness when dealing with signatures with
        repeated names for dimensions in a given matrix (for example (n,n)). 
	This has been fixed.
    Created the umath_linalg module
			   ***************

