
"""
 x=CreateGenerator(seed) creates an random number generator stream
   seed < 0  ==>  Use the default initial seed value.
   seed = 0  ==>  Set a "random" value for the seed from the system clock.
   seed > 0  ==>  Set seed directly (32 bits only).
   x.ranf() samples from that stream.
   x.sample(n) returns a vector from that stream.

 ranf() returns a stream of random numbers
 random_sample(n) returns a vector of length n filled with random numbers
"""
import scipy.base as Numeric
from scipy.lib.mtrand import *
import scipy.linalg as LinearAlgebra

# some aliases
ranf = random_sample
random = random_sample

def rand(*args):
    """rand(d1,...,dn) returns a matrix of the given dimensions
    which is initialized to random numbers from a uniform distribution
    in the range [0,1).
    """
    return random_sample(size=args)
    
def randn(*args):
    """u = randn(d0,d1,...,dn) returns zero-mean, unit-variance Gaussian
    random numbers in an array of size (d0,d1,...,dn).
    """
    return standard_normal(args)

def multivariate_normal(mean, cov, shape=[]):
       """multivariate_normal(mean, cov) or multivariate_normal(mean, cov, [m, n, ...])
          returns an array containing multivariate normally distributed random numbers
          with specified mean and covariance.

          mean must be a 1 dimensional array. cov must be a square two dimensional
          array with the same number of rows and columns as mean has elements.

          The first form returns a single 1-D array containing a multivariate
          normal.

          The second form returns an array of shape (m, n, ..., cov.shape[0]).
          In this case, output[i,j,...,:] is a 1-D array containing a multivariate
          normal."""
       # Check preconditions on arguments
       mean = Numeric.array(mean)
       cov = Numeric.array(cov)
       if len(mean.shape) != 1:
              raise ArgumentError, "mean must be 1 dimensional."
       if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
              raise ArgumentError, "cov must be 2 dimensional and square."
       if mean.shape[0] != cov.shape[0]:
              raise ArgumentError, "mean and cov must have same length."
       # Compute shape of output
       if isinstance(shape, int):
           shape = [shape]
       final_shape = list(shape[:])
       final_shape.append(mean.shape[0])
       # Create a matrix of independent standard normally distributed random
       # numbers. The matrix has rows with the same length as mean and as
       # many rows are necessary to form a matrix of shape final_shape.
       x = standard_normal(Numeric.multiply.reduce(final_shape))
       x.shape = (Numeric.multiply.reduce(final_shape[0:len(final_shape)-1]),
                  mean.shape[0])
       # Transform matrix of standard normals into matrix where each row
       # contains multivariate normals with the desired covariance.
       # Compute A such that matrixmultiply(transpose(A),A) == cov.
       # Then the matrix products of the rows of x and A has the desired
       # covariance. Note that sqrt(s)*v where (u,s,v) is the singular value
       # decomposition of cov is such an A.
       (u,s,v) = LinearAlgebra.singular_value_decomposition(cov)
       x = Numeric.matrixmultiply(x*Numeric.sqrt(s),v)
       # The rows of x now have the correct covariance but mean 0. Add
       # mean to each row. Then each row will have mean mean.
       Numeric.add(mean,x,x)
       x.shape = tuple(final_shape)
       return x

# XXX: should we also bring over mean_var_test() from random_lite.py? It seems 
# out of place.