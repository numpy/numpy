
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
import Numeric
from RNG import *

standard_generator = CreateGenerator(-1)

def ranf():
        "ranf() = a random number from the standard generator."
        return standard_generator.ranf()

def random_sample(*n):
        """random_sample(n) = array of n random numbers;
        random_sample(n1, n2, ...)= random array of shape (n1, n2, ..)"""

        if not n:
                return standard_generator.sample(1)
        m = 1
        for i in n:
                m = m * i
        return Numeric.reshape (standard_generator.sample(m), n)

