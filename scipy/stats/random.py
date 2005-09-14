import ranlib
import Numeric
import LinearAlgebra
import sys
import math
from types import *

# Extended RandomArray to provide more distributions:
# normal, beta, chi square, F, multivariate normal,
# exponential, binomial, multinomial
# Lee Barford, Dec. 1999.

class ArgumentError(Exception):
    pass

def seed(x=0,y=0):
    """seed(x, y), set the seed using the integers x, y;
    Set a random one from clock if  y == 0
    """
    if type (x) != IntType or type (y) != IntType :
        raise ArgumentError, "seed requires integer arguments."
    if y == 0:
        import time
        t = time.time()
        ndigits = int(math.log10(t))
        base = 10**(ndigits/2)
        x = int(t/base)
        y = 1 + int(t%base)
    ranlib.set_seeds(x,y)

seed()

def get_seed():
    "Return the current seed pair"
    return ranlib.get_seeds()

def _build_random_array(fun, args, shape=[]):
# Build an array by applying function fun to
# the arguments in args, creating an array with
# the specified shape.
# Allows an integer shape n as a shorthand for (n,).
    if isinstance(shape, IntType):
        shape = [shape]
    if len(shape) != 0:
        n = Numeric.multiply.reduce(shape)
        s = apply(fun, args + (n,))
        s.shape = shape
        return s
    else:
        n = 1
        s = apply(fun, args + (n,))
        return s[0]

def random(shape=[]):
    "random(n) or random([n, m, ...]) returns array of random numbers"
    return _build_random_array(ranlib.sample, (), shape)

def uniform(minimum, maximum, shape=[]):
    """uniform(minimum, maximum, shape=[]) returns array of given shape of random reals
    in given range"""
    return minimum + (maximum-minimum)*random(shape)

def randint(minimum, maximum=None, shape=[]):
    """randint(min, max, shape=[]) = random integers >=min, < max
    If max not given, random integers >= 0, <min"""
    if not isinstance(minimum, IntType):
        raise ArgumentError, "randint requires first argument integer"
    if maximum is None:
        maximum = minimum
        minimum = 0
    if not isinstance(maximum, IntType):
        raise ArgumentError, "randint requires second argument integer"
    a = ((maximum-minimum)* random(shape))
    if isinstance(a, Numeric.ArrayType):
        return minimum + a.astype(Numeric.Int)
    else:
        return minimum + int(a)

def random_integers(maximum, minimum=1, shape=[]):
    """random_integers(max, min=1, shape=[]) = random integers in range min-max inclusive"""
    return randint(minimum, maximum+1, shape)

def permutation(n):
    "permutation(n) = a permutation of indices range(n)"
    return Numeric.argsort(random(n))

def standard_normal(shape=[]):
    """standard_normal(n) or standard_normal([n, m, ...]) returns array of
           random numbers normally distributed with mean 0 and standard
           deviation 1"""
    return _build_random_array(ranlib.standard_normal, (), shape)

def normal(mean, std, shape=[]):
        """normal(mean, std, n) or normal(mean, std, [n, m, ...]) returns
           array of random numbers randomly distributed with specified mean and
           standard deviation"""
        s = standard_normal(shape)
        return s * std + mean

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
       if isinstance(shape, IntType): shape = [shape]
       final_shape = list(shape[:])
       final_shape.append(mean.shape[0])
       # Create a matrix of independent standard normally distributed random
       # numbers. The matrix has rows with the same length as mean and as
       # many rows are necessary to form a matrix of shape final_shape.
       x = ranlib.standard_normal(Numeric.multiply.reduce(final_shape))
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
       x.shape = final_shape
       return x

def exponential(mean, shape=[]):
    """exponential(mean, n) or exponential(mean, [n, m, ...]) returns array
      of random numbers exponentially distributed with specified mean"""
   # If U is a random number uniformly distributed on [0,1], then
   #      -ln(U) is exponentially distributed with mean 1, and so
   #      -ln(U)*M is exponentially distributed with mean M.
    x = random(shape)
    Numeric.log(x, x)
    Numeric.subtract(0.0, x, x)
    Numeric.multiply(mean, x, x)
    return x

def beta(a, b, shape=[]):
    """beta(a, b) or beta(a, b, [n, m, ...]) returns array of beta distributed random numbers."""
    return _build_random_array(ranlib.beta, (a, b), shape)

def gamma(a, r, shape=[]):
    """gamma(a, r) or gamma(a, r, [n, m, ...]) returns array of gamma distributed random numbers."""
    return _build_random_array(ranlib.gamma, (a, r), shape)

def F(dfn, dfd, shape=[]):
    """F(dfn, dfd) or F(dfn, dfd, [n, m, ...]) returns array of F distributed random numbers with dfn degrees of freedom in the numerator and dfd degrees of freedom in the denominator."""
    return ( chi_square(dfn, shape) / dfn) / ( chi_square(dfd, shape) / dfd)

def noncentral_F(dfn, dfd, nconc, shape=[]):
    """noncentral_F(dfn, dfd, nonc) or noncentral_F(dfn, dfd, nonc, [n, m, ...]) returns array of noncentral F distributed random numbers with dfn degrees of freedom in the numerator and dfd degrees of freedom in the denominator, and noncentrality parameter nconc."""
    return ( noncentral_chi_square(dfn, nconc, shape) / dfn ) / ( chi_square(dfd, shape) / dfd )

def chi_square(df, shape=[]):
    """chi_square(df) or chi_square(df, [n, m, ...]) returns array of chi squared distributed random numbers with df degrees of freedom."""
    return _build_random_array(ranlib.chisquare, (df,), shape)

def noncentral_chi_square(df, nconc, shape=[]):
    """noncentral_chi_square(df, nconc) or chi_square(df, nconc, [n, m, ...]) returns array of noncentral chi squared distributed random numbers with df degrees of freedom and noncentrality parameter."""
    return _build_random_array(ranlib.noncentral_chisquare, (df, nconc), shape)

def binomial(trials, p, shape=[]):
    """binomial(trials, p) or binomial(trials, p, [n, m, ...]) returns array of binomially distributed random integers.

           trials is the number of trials in the binomial distribution.
           p is the probability of an event in each trial of the binomial distribution."""
    return _build_random_array(ranlib.binomial, (trials, p), shape)

def negative_binomial(trials, p, shape=[]):
    """negative_binomial(trials, p) or negative_binomial(trials, p, [n, m, ...]) returns
           array of negative binomially distributed random integers.

           trials is the number of trials in the negative binomial distribution.
           p is the probability of an event in each trial of the negative binomial distribution."""
    return _build_random_array(ranlib.negative_binomial, (trials, p), shape)

def multinomial(trials, probs, shape=[]):
    """multinomial(trials, probs) or multinomial(trials, probs, [n, m, ...]) returns
           array of multinomial distributed integer vectors.

           trials is the number of trials in each multinomial distribution.
           probs is a one dimensional array. There are len(prob)+1 events.
           prob[i] is the probability of the i-th event, 0<=i<len(prob).
           The probability of event len(prob) is 1.-Numeric.sum(prob).

       The first form returns a single 1-D array containing one multinomially
           distributed vector.

           The second form returns an array of shape (m, n, ..., len(probs)).
           In this case, output[i,j,...,:] is a 1-D array containing a multinomially
           distributed integer 1-D array."""
        # Check preconditions on arguments
    probs = Numeric.array(probs)
    if len(probs.shape) != 1:
        raise ArgumentError, "probs must be 1 dimensional."
        # Compute shape of output
    if type(shape) == type(0): shape = [shape]
    final_shape = shape[:]
    final_shape.append(probs.shape[0]+1)
    x = ranlib.multinomial(trials, probs.astype(Numeric.Float32), Numeric.multiply.reduce(shape))
        # Change its shape to the desire one
    x.shape = final_shape
    return x

def poisson(mean, shape=[]):
    """poisson(mean) or poisson(mean, [n, m, ...]) returns array of poisson
           distributed random integers with specified mean."""
    return _build_random_array(ranlib.poisson, (mean,), shape)


def mean_var_test(x, type, mean, var, skew=[]):
    n = len(x) * 1.0
    x_mean = Numeric.sum(x)/n
    x_minus_mean = x - x_mean
    x_var = Numeric.sum(x_minus_mean*x_minus_mean)/(n-1.0)
    print "\nAverage of ", len(x), type
    print "(should be about ", mean, "):", x_mean
    print "Variance of those random numbers (should be about ", var, "):", x_var
    if skew != []:
       x_skew = (Numeric.sum(x_minus_mean*x_minus_mean*x_minus_mean)/9998.)/x_var**(3./2.)
       print "Skewness of those random numbers (should be about ", skew, "):", x_skew

def test():
    x, y = get_seed()
    print "Initial seed", x, y
    seed(x, y)
    x1, y1 = get_seed()
    if x1 != x or y1 != y:
        raise SystemExit, "Failed seed test."
    print "First random number is", random()
    print "Average of 10000 random numbers is", Numeric.sum(random(10000))/10000.
    x = random([10,1000])
    if len(x.shape) != 2 or x.shape[0] != 10 or x.shape[1] != 1000:
        raise SystemExit, "random returned wrong shape"
    x.shape = (10000,)
    print "Average of 100 by 100 random numbers is", Numeric.sum(x)/10000.
    y = uniform(0.5,0.6, (1000,10))
    if len(y.shape) !=2 or y.shape[0] != 1000 or y.shape[1] != 10:
        raise SystemExit, "uniform returned wrong shape"
    y.shape = (10000,)
    if Numeric.minimum.reduce(y) <= 0.5 or Numeric.maximum.reduce(y) >= 0.6:
        raise SystemExit, "uniform returned out of desired range"
    print "randint(1, 10, shape=[50])"
    print randint(1, 10, shape=[50])
    print "permutation(10)", permutation(10)
    print "randint(3,9)", randint(3,9)
    print "random_integers(10, shape=[20])"
    print random_integers(10, shape=[20])
    s = 3.0
    x = normal(2.0, s, [10, 1000])
    if len(x.shape) != 2 or x.shape[0] != 10 or x.shape[1] != 1000:
        raise SystemExit, "standard_normal returned wrong shape"
    x.shape = (10000,)
    mean_var_test(x, "normally distributed numbers with mean 2 and variance %f"%(s**2,), 2, s**2, 0)
    x = exponential(3, 10000)
    mean_var_test(x, "random numbers exponentially distributed with mean %f"%(s,), s, s**2, 2)
    x = multivariate_normal(Numeric.array([10,20]), Numeric.array(([1,2],[2,4])))
    print "\nA multivariate normal", x
    if x.shape != (2,): raise SystemExit, "multivariate_normal returned wrong shape"
    x = multivariate_normal(Numeric.array([10,20]), Numeric.array([[1,2],[2,4]]), [4,3])
    print "A 4x3x2 array containing multivariate normals"
    print x
    if x.shape != (4,3,2): raise SystemExit, "multivariate_normal returned wrong shape"
    x = multivariate_normal(Numeric.array([-100,0,100]), Numeric.array([[3,2,1],[2,2,1],[1,1,1]]), 10000)
    x_mean = Numeric.sum(x)/10000.
    print "Average of 10000 multivariate normals with mean [-100,0,100]"
    print x_mean
    x_minus_mean = x - x_mean
    print "Estimated covariance of 10000 multivariate normals with covariance [[3,2,1],[2,2,1],[1,1,1]]"
    print Numeric.matrixmultiply(Numeric.transpose(x_minus_mean),x_minus_mean)/9999.
    x = beta(5.0, 10.0, 10000)
    mean_var_test(x, "beta(5.,10.) random numbers", 0.333, 0.014)
    x = gamma(.01, 2., 10000)
    mean_var_test(x, "gamma(.01,2.) random numbers", 2*100, 2*100*100)
    x = chi_square(11., 10000)
    mean_var_test(x, "chi squared random numbers with 11 degrees of freedom", 11, 22, 2*Numeric.sqrt(2./11.))
    x = F(5., 10., 10000)
    mean_var_test(x, "F random numbers with 5 and 10 degrees of freedom", 1.25, 1.35)
    x = poisson(50., 10000)
    mean_var_test(x, "poisson random numbers with mean 50", 50, 50, 0.14)
    print "\nEach element is the result of 16 binomial trials with probability 0.5:"
    print binomial(16, 0.5, 16)
    print "\nEach element is the result of 16 negative binomial trials with probability 0.5:"
    print negative_binomial(16, 0.5, [16,])
    print "\nEach row is the result of 16 multinomial trials with probabilities [0.1, 0.5, 0.1 0.3]:"
    x = multinomial(16, [0.1, 0.5, 0.1], 8)
    print x
    print "Mean = ", Numeric.sum(x)/8.

if __name__ == '__main__':
    test()
