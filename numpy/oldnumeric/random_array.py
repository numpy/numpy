"""Backward compatible module for RandomArray

"""
from __future__ import division, absolute_import, print_function

__all__ = ['ArgumentError','F','beta','binomial','chi_square', 'exponential',
           'gamma', 'get_seed', 'mean_var_test', 'multinomial',
           'multivariate_normal', 'negative_binomial', 'noncentral_F',
           'noncentral_chi_square', 'normal', 'permutation', 'poisson',
           'randint', 'random', 'random_integers', 'seed', 'standard_normal',
           'uniform']

ArgumentError = ValueError

import numpy.random.mtrand as mt
import numpy as np

def seed(x=0, y=0):
    if (x == 0 or y == 0):
        mt.seed()
    else:
        mt.seed((x,y))

def get_seed():
    raise NotImplementedError(
          "If you want to save the state of the random number generator.\n"
          "Then you should use obj = numpy.random.get_state() followed by.\n"
          "numpy.random.set_state(obj).")

def random(shape=[]):
    "random(n) or random([n, m, ...]) returns array of random numbers"
    if shape == []:
        shape = None
    return mt.random_sample(shape)

def uniform(minimum, maximum, shape=[]):
    """uniform(minimum, maximum, shape=[]) returns array of given shape of random reals
    in given range"""
    if shape == []:
        shape = None
    return mt.uniform(minimum, maximum, shape)

def randint(minimum, maximum=None, shape=[]):
    """randint(min, max, shape=[]) = random integers >=min, < max
    If max not given, random integers >= 0, <min"""
    if not isinstance(minimum, int):
        raise ArgumentError("randint requires first argument integer")
    if maximum is None:
        maximum = minimum
        minimum = 0
    if not isinstance(maximum, int):
        raise ArgumentError("randint requires second argument integer")
    a = ((maximum-minimum)* random(shape))
    if isinstance(a, np.ndarray):
        return minimum + a.astype(np.int)
    else:
        return minimum + int(a)

def random_integers(maximum, minimum=1, shape=[]):
    """random_integers(max, min=1, shape=[]) = random integers in range min-max inclusive"""
    return randint(minimum, maximum+1, shape)

def permutation(n):
    "permutation(n) = a permutation of indices range(n)"
    return mt.permutation(n)

def standard_normal(shape=[]):
    """standard_normal(n) or standard_normal([n, m, ...]) returns array of
           random numbers normally distributed with mean 0 and standard
           deviation 1"""
    if shape == []:
        shape = None
    return mt.standard_normal(shape)

def normal(mean, std, shape=[]):
    """normal(mean, std, n) or normal(mean, std, [n, m, ...]) returns
    array of random numbers randomly distributed with specified mean and
    standard deviation"""
    if shape == []:
        shape = None
    return mt.normal(mean, std, shape)

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
    if shape == []:
        shape = None
    return mt.multivariate_normal(mean, cov, shape)

def exponential(mean, shape=[]):
    """exponential(mean, n) or exponential(mean, [n, m, ...]) returns array
      of random numbers exponentially distributed with specified mean"""
    if shape == []:
        shape = None
    return mt.exponential(mean, shape)

def beta(a, b, shape=[]):
    """beta(a, b) or beta(a, b, [n, m, ...]) returns array of beta distributed random numbers."""
    if shape == []:
        shape = None
    return mt.beta(a, b, shape)

def gamma(a, r, shape=[]):
    """gamma(a, r) or gamma(a, r, [n, m, ...]) returns array of gamma distributed random numbers."""
    if shape == []:
        shape = None
    return mt.gamma(a, r, shape)

def F(dfn, dfd, shape=[]):
    """F(dfn, dfd) or F(dfn, dfd, [n, m, ...]) returns array of F distributed random numbers with dfn degrees of freedom in the numerator and dfd degrees of freedom in the denominator."""
    if shape == []:
        shape = None
    return mt.f(dfn, dfd, shape)

def noncentral_F(dfn, dfd, nconc, shape=[]):
    """noncentral_F(dfn, dfd, nonc) or noncentral_F(dfn, dfd, nonc, [n, m, ...]) returns array of noncentral F distributed random numbers with dfn degrees of freedom in the numerator and dfd degrees of freedom in the denominator, and noncentrality parameter nconc."""
    if shape == []:
        shape = None
    return mt.noncentral_f(dfn, dfd, nconc, shape)

def chi_square(df, shape=[]):
    """chi_square(df) or chi_square(df, [n, m, ...]) returns array of chi squared distributed random numbers with df degrees of freedom."""
    if shape == []:
        shape = None
    return mt.chisquare(df, shape)

def noncentral_chi_square(df, nconc, shape=[]):
    """noncentral_chi_square(df, nconc) or chi_square(df, nconc, [n, m, ...]) returns array of noncentral chi squared distributed random numbers with df degrees of freedom and noncentrality parameter."""
    if shape == []:
        shape = None
    return mt.noncentral_chisquare(df, nconc, shape)

def binomial(trials, p, shape=[]):
    """binomial(trials, p) or binomial(trials, p, [n, m, ...]) returns array of binomially distributed random integers.

           trials is the number of trials in the binomial distribution.
           p is the probability of an event in each trial of the binomial distribution."""
    if shape == []:
        shape = None
    return mt.binomial(trials, p, shape)

def negative_binomial(trials, p, shape=[]):
    """negative_binomial(trials, p) or negative_binomial(trials, p, [n, m, ...]) returns
           array of negative binomially distributed random integers.

           trials is the number of trials in the negative binomial distribution.
           p is the probability of an event in each trial of the negative binomial distribution."""
    if shape == []:
        shape = None
    return mt.negative_binomial(trials, p, shape)

def multinomial(trials, probs, shape=[]):
    """multinomial(trials, probs) or multinomial(trials, probs, [n, m, ...]) returns
           array of multinomial distributed integer vectors.

           trials is the number of trials in each multinomial distribution.
           probs is a one dimensional array. There are len(prob)+1 events.
           prob[i] is the probability of the i-th event, 0<=i<len(prob).
           The probability of event len(prob) is 1.-np.sum(prob).

       The first form returns a single 1-D array containing one multinomially
           distributed vector.

           The second form returns an array of shape (m, n, ..., len(probs)).
           In this case, output[i,j,...,:] is a 1-D array containing a multinomially
           distributed integer 1-D array."""
    if shape == []:
        shape = None
    return mt.multinomial(trials, probs, shape)

def poisson(mean, shape=[]):
    """poisson(mean) or poisson(mean, [n, m, ...]) returns array of poisson
           distributed random integers with specified mean."""
    if shape == []:
        shape = None
    return mt.poisson(mean, shape)


def mean_var_test(x, type, mean, var, skew=[]):
    n = len(x) * 1.0
    x_mean = np.sum(x,axis=0)/n
    x_minus_mean = x - x_mean
    x_var = np.sum(x_minus_mean*x_minus_mean,axis=0)/(n-1.0)
    print("\nAverage of ", len(x), type)
    print("(should be about ", mean, "):", x_mean)
    print("Variance of those random numbers (should be about ", var, "):", x_var)
    if skew != []:
        x_skew = (np.sum(x_minus_mean*x_minus_mean*x_minus_mean,axis=0)/9998.)/x_var**(3./2.)
        print("Skewness of those random numbers (should be about ", skew, "):", x_skew)

def test():
    obj = mt.get_state()
    mt.set_state(obj)
    obj2 = mt.get_state()
    if (obj2[1] - obj[1]).any():
        raise SystemExit("Failed seed test.")
    print("First random number is", random())
    print("Average of 10000 random numbers is", np.sum(random(10000),axis=0)/10000.)
    x = random([10,1000])
    if len(x.shape) != 2 or x.shape[0] != 10 or x.shape[1] != 1000:
        raise SystemExit("random returned wrong shape")
    x.shape = (10000,)
    print("Average of 100 by 100 random numbers is", np.sum(x,axis=0)/10000.)
    y = uniform(0.5,0.6, (1000,10))
    if len(y.shape) !=2 or y.shape[0] != 1000 or y.shape[1] != 10:
        raise SystemExit("uniform returned wrong shape")
    y.shape = (10000,)
    if np.minimum.reduce(y) <= 0.5 or np.maximum.reduce(y) >= 0.6:
        raise SystemExit("uniform returned out of desired range")
    print("randint(1, 10, shape=[50])")
    print(randint(1, 10, shape=[50]))
    print("permutation(10)", permutation(10))
    print("randint(3,9)", randint(3,9))
    print("random_integers(10, shape=[20])")
    print(random_integers(10, shape=[20]))
    s = 3.0
    x = normal(2.0, s, [10, 1000])
    if len(x.shape) != 2 or x.shape[0] != 10 or x.shape[1] != 1000:
        raise SystemExit("standard_normal returned wrong shape")
    x.shape = (10000,)
    mean_var_test(x, "normally distributed numbers with mean 2 and variance %f"%(s**2,), 2, s**2, 0)
    x = exponential(3, 10000)
    mean_var_test(x, "random numbers exponentially distributed with mean %f"%(s,), s, s**2, 2)
    x = multivariate_normal(np.array([10,20]), np.array(([1,2],[2,4])))
    print("\nA multivariate normal", x)
    if x.shape != (2,): raise SystemExit("multivariate_normal returned wrong shape")
    x = multivariate_normal(np.array([10,20]), np.array([[1,2],[2,4]]), [4,3])
    print("A 4x3x2 array containing multivariate normals")
    print(x)
    if x.shape != (4,3,2): raise SystemExit("multivariate_normal returned wrong shape")
    x = multivariate_normal(np.array([-100,0,100]), np.array([[3,2,1],[2,2,1],[1,1,1]]), 10000)
    x_mean = np.sum(x,axis=0)/10000.
    print("Average of 10000 multivariate normals with mean [-100,0,100]")
    print(x_mean)
    x_minus_mean = x - x_mean
    print("Estimated covariance of 10000 multivariate normals with covariance [[3,2,1],[2,2,1],[1,1,1]]")
    print(np.dot(np.transpose(x_minus_mean),x_minus_mean)/9999.)
    x = beta(5.0, 10.0, 10000)
    mean_var_test(x, "beta(5.,10.) random numbers", 0.333, 0.014)
    x = gamma(.01, 2., 10000)
    mean_var_test(x, "gamma(.01,2.) random numbers", 2*100, 2*100*100)
    x = chi_square(11., 10000)
    mean_var_test(x, "chi squared random numbers with 11 degrees of freedom", 11, 22, 2*np.sqrt(2./11.))
    x = F(5., 10., 10000)
    mean_var_test(x, "F random numbers with 5 and 10 degrees of freedom", 1.25, 1.35)
    x = poisson(50., 10000)
    mean_var_test(x, "poisson random numbers with mean 50", 50, 50, 0.14)
    print("\nEach element is the result of 16 binomial trials with probability 0.5:")
    print(binomial(16, 0.5, 16))
    print("\nEach element is the result of 16 negative binomial trials with probability 0.5:")
    print(negative_binomial(16, 0.5, [16,]))
    print("\nEach row is the result of 16 multinomial trials with probabilities [0.1, 0.5, 0.1 0.3]:")
    x = multinomial(16, [0.1, 0.5, 0.1], 8)
    print(x)
    print("Mean = ", np.sum(x,axis=0)/8.)

if __name__ == '__main__':
    test()
