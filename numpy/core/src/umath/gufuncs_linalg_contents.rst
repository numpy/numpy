=======================
 gufuncs_linalg module
=======================

gufuncs_linalg implements a series of linear algebra functions as gufuncs.
Most of these functions are already present in numpy.linalg, but as they
are implemented using gufunc kernels they can be broadcasting. Some parts
that are python in numpy.linalg are implemented inside C functions, as well
as the iteration used when used on vectors. This can result in faster
execution as well.

In addition, there are some ufuncs thrown in that implement fused operations
over numpy vectors that can result in faster execution on large vector 
compared to non-fused versions (for example: multiply_add, multiply3).

In fact, gufuncs_linalg is a very thin wrapper of python code that wraps
the actual kernels (gufuncs). This wrapper was needed in order to provide
a sane interface for some functions. Mostly working around limitations on
what can be described in a gufunc signature. Things like having one dimension
of a result depending on the minimum of two dimensions of the sources (like
in svd) or passing an uniform keyword parameter to the whole operation
(like UPLO on functions over symmetric/hermitian matrices).

The gufunc kernels are in a c module named _umath_linalg, that is imported
privately in gufuncs_linalg.

==========
 Contents
==========
Here is an enumeration of the functions. These are the functions exported by
the module and should appear in its __all__ attribute. All the functions
contain a docstring explaining them in detail.

General
=======
- inner1d
- innerwt
- matrix_multiply
- quadratic_form

Lineal Algebra
==============
- det
- slogdet
- cholesky
- eig
- eigvals
- eigh
- eigvalsh
- solve
- svd
- chosolve
- inv
- poinv

Fused Operations
================
- add3
- multiply3
- multiply3_add
- multiply_add
- multiply_add2
- multiply4
- multiply4_add

================
 Error Handling
================
Unlike the numpy.linalg module, this module does not use exceptions to notify
errors in the execution of the kernels. As these functions are thougth to be 
used in a vector way it didn't seem appropriate to raise exceptions on failure
of an element. So instead, when an error computing an element occurs its 
associated result will be set to an invalid value (all NaNs).

Exceptions can occur if the arguments fail to map properly to the underlying
gufunc (due to signature mismatch, for example).

================================
 Notes about the implementation
================================
Where possible, the wrapper functions map directly into a gufunc implementing
the computation.

That's not always the case, as due to limitations of the gufunc interface some
functions cannot be mapped straight into a kernel.

Two cases come to mind:
- An uniform parameter is needed to configure the way the computation is 
performed (like UPLO in the functions working on symmetric/hermitian matrices)
- svd, where it was impossible to map the function to a gufunc signature.

In the case of uniform parameters like UPLO, there are two separate entry points
in the C module that imply either 'U' or 'L'. The wrapper just selects the
kernel to use by checking the appropriate keyword parameter. This way a
function interface similar to numpy.linalg can be kept.

In the case of SVD not only there were problems with the support of keyword
arguments. There was the added problem of the signature system not being able
to cope with the needs of this functions. Just for the singular values a
a signature like (m,n)->(min(m,n)) was needed. This has been worked around by
implementing different kernels for the cases where min(m,n) == m and where
min(m,n) == n. The wrapper code automatically selects the appropriate one.


