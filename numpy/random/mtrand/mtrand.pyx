# mtrand.pyx -- A Pyrex wrapper of Jean-Sebastien Roy's RandomKit
#
# Copyright 2005 Robert Kern (robert.kern@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

include "Python.pxi"
include "numpy.pxi"

cdef extern from "math.h":
    double exp(double x)
    double log(double x)
    double floor(double x)
    double sin(double x)
    double cos(double x)

cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    char *rk_strerror[2]
    
    # 0xFFFFFFFFUL
    unsigned long RK_MAX

    void rk_seed(unsigned long seed, rk_state *state)
    rk_error rk_randomseed(rk_state *state)
    unsigned long rk_random(rk_state *state)
    long rk_long(rk_state *state)
    unsigned long rk_ulong(rk_state *state)
    unsigned long rk_interval(unsigned long max, rk_state *state)
    double rk_double(rk_state *state)
    void rk_fill(void *buffer, size_t size, rk_state *state)
    rk_error rk_devfill(void *buffer, size_t size, int strong)
    rk_error rk_altfill(void *buffer, size_t size, int strong,
            rk_state *state)
    double rk_gauss(rk_state *state)

cdef extern from "distributions.h":
    
    double rk_normal(rk_state *state, double loc, double scale)
    double rk_standard_exponential(rk_state *state)
    double rk_exponential(rk_state *state, double scale)
    double rk_uniform(rk_state *state, double loc, double scale)
    double rk_standard_gamma(rk_state *state, double shape)
    double rk_gamma(rk_state *state, double shape, double scale)
    double rk_beta(rk_state *state, double a, double b)
    double rk_chisquare(rk_state *state, double df)
    double rk_noncentral_chisquare(rk_state *state, double df, double nonc)
    double rk_f(rk_state *state, double dfnum, double dfden)
    double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc)
    double rk_standard_cauchy(rk_state *state)
    double rk_standard_t(rk_state *state, double df)
    double rk_vonmises(rk_state *state, double mu, double kappa)
    double rk_pareto(rk_state *state, double a)
    double rk_weibull(rk_state *state, double a)
    double rk_power(rk_state *state, double a)
    double rk_laplace(rk_state *state, double loc, double scale)
    double rk_gumbel(rk_state *state, double loc, double scale)
    double rk_logistic(rk_state *state, double loc, double scale)
    double rk_lognormal(rk_state *state, double mode, double sigma)
    double rk_rayleigh(rk_state *state, double mode)
    double rk_wald(rk_state *state, double mean, double scale)
    double rk_triangular(rk_state *state, double left, double mode, double right)
    
    long rk_binomial(rk_state *state, long n, double p)
    long rk_binomial_btpe(rk_state *state, long n, double p)
    long rk_binomial_inversion(rk_state *state, long n, double p)
    long rk_negative_binomial(rk_state *state, long n, double p)
    long rk_poisson(rk_state *state, double lam)
    long rk_poisson_mult(rk_state *state, double lam)
    long rk_poisson_ptrs(rk_state *state, double lam)
    long rk_zipf(rk_state *state, double a)
    long rk_geometric(rk_state *state, double p)
    long rk_hypergeometric(rk_state *state, long good, long bad, long sample)
    long rk_logseries(rk_state *state, double p)

ctypedef double (* rk_cont0)(rk_state *state)
ctypedef double (* rk_cont1)(rk_state *state, double a)
ctypedef double (* rk_cont2)(rk_state *state, double a, double b)
ctypedef double (* rk_cont3)(rk_state *state, double a, double b, double c)

ctypedef long (* rk_disc0)(rk_state *state)
ctypedef long (* rk_discnp)(rk_state *state, long n, double p)
ctypedef long (* rk_discnmN)(rk_state *state, long n, long m, long N)
ctypedef long (* rk_discd)(rk_state *state, double a)


cdef extern from "initarray.h":
   void init_by_array(rk_state *self, unsigned long *init_key, 
                      unsigned long key_length)

# Initialize numpy
import_array()

import numpy as _sp

cdef object cont0_array(rk_state *state, rk_cont0 func, object size):
    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state)
    else:
        array = <ndarray>_sp.empty(size, _sp.float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state)
        return array

cdef object cont1_array(rk_state *state, rk_cont1 func, object size, ndarray oa):
    cdef double *array_data
    cdef double *oa_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef flatiter itera
    cdef broadcast multi
    cdef int scalar

    scalar = 0
    if oa.nd == 0:
        oa_data = <double *>oa.data
        scalar = 1

    if size is None:
        if scalar:
            return func(state, oa_data[0])
        else:
            array = <ndarray>PyArray_SimpleNew(oa.nd, oa.dimensions, NPY_DOUBLE)
            length = PyArray_SIZE(array)
            array_data = <double *>array.data
            itera = <flatiter>PyArray_IterNew(<object>oa)
            for i from 0 <= i < length:
                array_data[i] = func(state, (<double *>(itera.dataptr))[0])
                PyArray_ITER_NEXT(itera)
    else:
        array = <ndarray>_sp.empty(size, _sp.float64)
        array_data = <double *>array.data
        if scalar:
            length = PyArray_SIZE(array)
            for i from 0 <= i < length:
                array_data[i] = func(state, oa_data[0])
        else:
            multi = <broadcast>PyArray_MultiIterNew(2, <void *>array,
                                                    <void *>oa)
            if (multi.size != PyArray_SIZE(array)): 
                raise ValueError("size is not compatible with inputs")
            for i from 0 <= i < multi.size:
                oa_data = <double *>PyArray_MultiIter_DATA(multi, 1)
                array_data[i] = func(state, oa_data[0])
                PyArray_MultiIter_NEXTi(multi, 1)
    return array

cdef object cont2_array(rk_state *state, rk_cont2 func, object size,
                        ndarray oa, ndarray ob):
    cdef double *array_data
    cdef double *oa_data
    cdef double *ob_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef broadcast multi
    cdef int scalar

    scalar = 0
    if oa.nd == 0 and ob.nd == 0:
        oa_data = <double *>oa.data
        ob_data = <double *>ob.data
        scalar = 1
        
    if size is None:
        if scalar:
            return func(state, oa_data[0], ob_data[0])
        else:
            multi = <broadcast> PyArray_MultiIterNew(2, <void *>oa, <void *>ob)
            array = <ndarray> PyArray_SimpleNew(multi.nd, multi.dimensions, NPY_DOUBLE)
            array_data = <double *>array.data
            for i from 0 <= i < multi.size:
                oa_data = <double *>PyArray_MultiIter_DATA(multi, 0)
                ob_data = <double *>PyArray_MultiIter_DATA(multi, 1)
                array_data[i] = func(state, oa_data[0], ob_data[0])
                PyArray_MultiIter_NEXT(multi)
    else:
        array = <ndarray>_sp.empty(size, _sp.float64)
        array_data = <double *>array.data
        if scalar:
            length = PyArray_SIZE(array)
            for i from 0 <= i < length:
                array_data[i] = func(state, oa_data[0], ob_data[0])
        else:
            multi = <broadcast>PyArray_MultiIterNew(3, <void*>array, <void *>oa, <void *>ob)
            if (multi.size != PyArray_SIZE(array)):
                raise ValueError("size is not compatible with inputs")                 
            for i from 0 <= i < multi.size:
                oa_data = <double *>PyArray_MultiIter_DATA(multi, 1)
                ob_data = <double *>PyArray_MultiIter_DATA(multi, 2)
                array_data[i] = func(state, oa_data[0], ob_data[0])
                PyArray_MultiIter_NEXTi(multi, 1)
                PyArray_MultiIter_NEXTi(multi, 2)
    return array

cdef object cont3_array(rk_state *state, rk_cont3 func, object size, ndarray oa, 
    ndarray ob, ndarray oc):

    cdef double *array_data
    cdef double *oa_data
    cdef double *ob_data
    cdef double *oc_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef broadcast multi
    cdef int scalar

    scalar = 0
    if (oa.nd ==0 and ob.nd==0 and oc.nd == 0):
        oa_data = <double *>oa.data
        ob_data = <double *>ob.data
        oc_data = <double *>oc.data
        scalar = 1
        
    if size is None:
        if scalar:
            return func(state, oa_data[0], ob_data[0], oc_data[0])
        else:
            multi = <broadcast> PyArray_MultiIterNew(3, <void *>oa, <void *>ob, <void *>oc)
            array = <ndarray> PyArray_SimpleNew(multi.nd, multi.dimensions, NPY_DOUBLE)
            array_data = <double *>array.data
            for i from 0 <= i < multi.size:
                oa_data = <double *>PyArray_MultiIter_DATA(multi, 0)
                ob_data = <double *>PyArray_MultiIter_DATA(multi, 1)
                oc_data = <double *>PyArray_MultiIter_DATA(multi, 2)
                array_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
                PyArray_MultiIter_NEXT(multi)
    else:
        array = <ndarray>_sp.empty(size, _sp.float64)
        array_data = <double *>array.data        
        if scalar:
            length = PyArray_SIZE(array)
            for i from 0 <= i < length:
                array_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
        else:
            multi = <broadcast>PyArray_MultiIterNew(4, <void*>array, <void *>oa,
                                                    <void *>ob, <void *>oc)
            if (multi.size != PyArray_SIZE(array)):
                raise ValueError("size is not compatible with inputs")                 
            for i from 0 <= i < multi.size:
                oa_data = <double *>PyArray_MultiIter_DATA(multi, 1)
                ob_data = <double *>PyArray_MultiIter_DATA(multi, 2)
                oc_data = <double *>PyArray_MultiIter_DATA(multi, 3)
                array_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
                PyArray_MultiIter_NEXT(multi)                
    return array

cdef object disc0_array(rk_state *state, rk_disc0 func, object size):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state)
    else:
        array = <ndarray>_sp.empty(size, int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state)
        return array

cdef object discnp_array(rk_state *state, rk_discnp func, object size, ndarray on, ndarray op):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef double *op_data
    cdef long *on_data
    cdef int scalar
    cdef broadcast multi

    scalar = 0
    if (on.nd == 0 and op.nd == 0):
        on_data = <long *>on.data
        op_data = <double *>op.data
        scalar = 1
        
    if size is None:
        if (scalar):
            return func(state, on_data[0], op_data[0])
        else:
            multi = <broadcast> PyArray_MultiIterNew(2, <void *>on, <void *>op)
            array = <ndarray> PyArray_SimpleNew(multi.nd, multi.dimensions, NPY_LONG)
            array_data = <long *>array.data
            for i from 0 <= i < multi.size:
                on_data = <long *>PyArray_MultiIter_DATA(multi, 0)
                op_data = <double *>PyArray_MultiIter_DATA(multi, 1)
                array_data[i] = func(state, on_data[0], op_data[0])
                PyArray_MultiIter_NEXT(multi)
    else:
        array = <ndarray>_sp.empty(size, int)        
        if (scalar):
            length = PyArray_SIZE(array)
            array_data = <long *>array.data
            for i from 0 <= i < length:
                array_data[i] = func(state, on_data[0], op_data[0])
        else:
            multi = <broadcast>PyArray_MultiIterNew(3, <void*>array, <void *>on, <void *>op)
            if (multi.size != PyArray_SIZE(array)):
                raise ValueError("size is not compatible with inputs")                 
            for i from 0 <= i < multi.size:
                on_data = <long *>PyArray_MultiIter_DATA(multi, 1)
                op_data = <double *>PyArray_MultiIter_DATA(multi, 2)
                array_data[i] = func(state, on_data[0], op_data[0])
                PyArray_MultiIter_NEXTi(multi, 1)
                PyArray_MultiIter_NEXTi(multi, 2)
            
    return array

cdef object discnmN_array(rk_state *state, rk_discnmN func, object size, 
    ndarray on, ndarray om, ndarray oN):
    cdef long *array_data
    cdef long *on_data
    cdef long *om_data
    cdef long *oN_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef broadcast multi
    cdef int scalar

    scalar = 0
    if (on.nd == 0 and om.nd == 0 and oN.nd==0):
        scalar = 1
        on_data = <long *>on.data
        om_data = <long *>om.data
        oN_data = <long *>oN.data        

    if size is None:
        if (scalar):
            return func(state, on_data[0], om_data[0], oN_data[0])
        else:
            multi = <broadcast> PyArray_MultiIterNew(3, <void *>on, <void *>om, <void *>oN)
            array = <ndarray> PyArray_SimpleNew(multi.nd, multi.dimensions, NPY_LONG)
            array_data = <long *>array.data
            for i from 0 <= i < multi.size:
                on_data = <long *>PyArray_MultiIter_DATA(multi, 0)
                om_data = <long *>PyArray_MultiIter_DATA(multi, 1)
                oN_data = <long *>PyArray_MultiIter_DATA(multi, 2)                
                array_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
                PyArray_MultiIter_NEXT(multi)
    else:
        array = <ndarray>_sp.empty(size, int)
        array_data = <long *>array.data
        if (scalar):
            length = PyArray_SIZE(array)
            for i from 0 <= i < length:
                array_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
        else:
            multi = <broadcast>PyArray_MultiIterNew(4, <void*>array, <void *>on, <void *>om,
                                                    <void *>oN)
            if (multi.size != PyArray_SIZE(array)):
                raise ValueError("size is not compatible with inputs") 
            for i from 0 <= i < multi.size:
                on_data = <long *>PyArray_MultiIter_DATA(multi, 1)
                om_data = <long *>PyArray_MultiIter_DATA(multi, 2)
                oN_data = <long *>PyArray_MultiIter_DATA(multi, 3)
                array_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
                PyArray_MultiIter_NEXT(multi)

    return array

cdef object discd_array(rk_state *state, rk_discd func, object size, ndarray oa):
    cdef long *array_data
    cdef double *oa_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef broadcast multi
    cdef flatiter itera
    cdef int scalar 

    scalar = 0
    if (oa.nd == 0):
        oa_data = <double *>oa.data
        scalar =1

    if size is None:
        if (scalar):
            return func(state, oa_data[0])
        else:
            array = <ndarray>PyArray_SimpleNew(oa.nd, oa.dimensions, NPY_LONG)
            length = PyArray_SIZE(array)
            array_data = <long *>array.data
            itera = <flatiter>PyArray_IterNew(<object>oa)
            for i from 0 <= i < length:
                array_data[i] = func(state, (<double *>(itera.dataptr))[0])
                PyArray_ITER_NEXT(itera)
    else:
        array = <ndarray>_sp.empty(size, int)
        array_data = <long *>array.data
        if (scalar):
            length = PyArray_SIZE(array)
            for i from 0 <= i < length:
                array_data[i] = func(state, oa_data[0])
        else:
            multi = <broadcast>PyArray_MultiIterNew(2, <void *>array, <void *>oa)
            if (multi.size != PyArray_SIZE(array)): 
                raise ValueError("size is not compatible with inputs")                
            for i from 0 <= i < multi.size:
                oa_data = <double *>PyArray_MultiIter_DATA(multi, 1)
                array_data[i] = func(state, oa_data[0])
                PyArray_MultiIter_NEXTi(multi, 1)
    return array

cdef double kahan_sum(double *darr, long n):
    cdef double c, y, t, sum
    cdef long i
    sum = darr[0]
    c = 0.0
    for i from 1 <= i < n:
        y = darr[i] - c
        t = sum + y
        c = (t-sum) - y
        sum = t
    return sum

cdef class RandomState:
    """Container for the Mersenne Twister PRNG.

    Constructor
    -----------
    RandomState(seed=None): initializes the PRNG with the given seed. See the
        seed() method for details.

    Distribution Methods
    -----------------
    RandomState exposes a number of methods for generating random numbers drawn
    from a variety of probability distributions. In addition to the
    distribution-specific arguments, each method takes a keyword argument
    size=None. If size is None, then a single value is generated and returned.
    If size is an integer, then a 1-D numpy array filled with generated values
    is returned. If size is a tuple, then a numpy array with that shape is
    filled and returned.
    """
    cdef rk_state *internal_state

    def __init__(self, seed=None):
        self.internal_state = <rk_state*>PyMem_Malloc(sizeof(rk_state))

        self.seed(seed)

    def __dealloc__(self):
        if self.internal_state != NULL:
            PyMem_Free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed=None):
        """Seed the generator.

        seed(seed=None)

        seed can be an integer, an array (or other sequence) of integers of any
        length, or None. If seed is None, then RandomState will try to read data
        from /dev/urandom (or the Windows analogue) if available or seed from
        the clock otherwise.
        """
        cdef rk_error errcode
        cdef ndarray obj "arrayObject_obj"
        if seed is None:
            errcode = rk_randomseed(self.internal_state)
        elif type(seed) is int:
            rk_seed(seed, self.internal_state)
        else:
            obj = <ndarray>PyArray_ContiguousFromObject(seed, NPY_LONG, 1, 1)
            init_by_array(self.internal_state, <unsigned long *>(obj.data),
                obj.dimensions[0])

    def get_state(self):
        """Return a tuple representing the internal state of the generator.

        get_state() -> ('MT19937', int key[624], int pos)
        """
        cdef ndarray state "arrayObject_state"
        state = <ndarray>_sp.empty(624, int)
        memcpy(<void*>(state.data), self.internal_state.key, 624*sizeof(long))
        return ('MT19937', state, self.internal_state.pos)
        
    def set_state(self, state):
        """Set the state from a tuple.
        
        state = ('MT19937', int key[624], int pos)
        
        set_state(state)
        """
        cdef ndarray obj "arrayObject_obj"
        cdef int pos
        algorithm_name = state[0]
        if algorithm_name != 'MT19937':
            raise ValueError("algorithm must be 'MT19937'")
        key, pos = state[1:]
        obj = <ndarray>PyArray_ContiguousFromObject(key, NPY_LONG, 1, 1)
        if obj.dimensions[0] != 624:
            raise ValueError("state must be 624 longs")
        memcpy(self.internal_state.key, <void*>(obj.data), 624*sizeof(long))
        self.internal_state.pos = pos
    
    # Pickling support:
    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)

    def __reduce__(self):
        return (_sp.random.__RandomState_ctor, (), self.get_state())

    # Basic distributions:
    def random_sample(self, size=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        random_sample(size=None) -> random values
        """
        return cont0_array(self.internal_state, rk_double, size)

    def tomaxint(self, size=None):
        """Returns random integers x such that 0 <= x <= sys.maxint.

        tomaxint(size=None) -> random values
        """
        return disc0_array(self.internal_state, rk_long, size)

    def randint(self, low, high=None, size=None):
        """Return random integers x such that low <= x < high.

        randint(low, high=None, size=None) -> random values

        If high is None, then 0 <= x < low.
        """
        cdef long lo, hi, diff
        cdef long *array_data
        cdef ndarray array "arrayObject"
        cdef long length
        cdef long i

        if high is None:
            lo = 0
            hi = low
        else:
            lo = low
            hi = high

        diff = hi - lo - 1
        if diff < 0:
            raise ValueError("low >= high")
    
        if size is None:
            return rk_interval(diff, self.internal_state) + lo
        else:
            array = <ndarray>_sp.empty(size, int)
            length = PyArray_SIZE(array)
            array_data = <long *>array.data
            for i from 0 <= i < length:
                array_data[i] = lo + <long>rk_interval(diff, self.internal_state)
            return array

    def bytes(self, unsigned int length):
        """Return random bytes.

        bytes(length) -> str
        """
        cdef void *bytes
        bytes = PyMem_Malloc(length)
        rk_fill(bytes, length, self.internal_state)
        bytestring = PyString_FromStringAndSize(<char*>bytes, length)
        PyMem_Free(bytes)
        return bytestring

    def uniform(self, low=0.0, high=1.0, size=None):
        """Uniform distribution over [low, high).

        uniform(low=0.0, high=1.0, size=None) -> random values
        """
        cdef ndarray olow
        cdef ndarray ohigh
        cdef ndarray odiff
        cdef object temp
        olow = <ndarray>PyArray_FROM_OTF(low, NPY_DOUBLE, NPY_ALIGNED)
        ohigh = <ndarray>PyArray_FROM_OTF(high, NPY_DOUBLE, NPY_ALIGNED)
        temp = _sp.subtract(ohigh, olow)
        Py_INCREF(temp) # needed to get around Pyrex's automatic reference-counting
                        #  rules because EnsureArray steals a reference
        odiff = <ndarray>PyArray_EnsureArray(temp)
        return cont2_array(self.internal_state, rk_uniform, size, olow, odiff)

    def rand(self, *args):
        """Return an array of the given dimensions which is initialized to 
        random numbers from a uniform distribution in the range [0,1).

        rand(d0, d1, ..., dn) -> random values

        Note:  This is a convenience function. If you want an
                    interface that takes a tuple as the first argument
                    use numpy.random.random_sample(shape_tuple).
        
        """
        if len(args) == 0:
            return self.random_sample()
        else:
            return self.random_sample(size=args)

    def randn(self, *args):
        """Returns zero-mean, unit-variance Gaussian random numbers in an 
        array of shape (d0, d1, ..., dn).

        randn(d0, d1, ..., dn) -> random values

        Note:  This is a convenience function. If you want an
                    interface that takes a tuple as the first argument
                    use numpy.random.standard_normal(shape_tuple).
        """
        if len(args) == 0:
            return self.standard_normal()
        else:
            return self.standard_normal(args)

    def random_integers(self, low, high=None, size=None):
        """Return random integers x such that low <= x <= high.

        random_integers(low, high=None, size=None) -> random values.

        If high is None, then 1 <= x <= low.
        """
        if high is None:
            high = low
            low = 1
        return self.randint(low, high+1, size)

    # Complicated, continuous distributions:
    def standard_normal(self, size=None):
        """Standard Normal distribution (mean=0, stdev=1).

        standard_normal(size=None) -> random values
        """
        return cont0_array(self.internal_state, rk_gauss, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Normal distribution (mean=loc, stdev=scale).

        normal(loc=0.0, scale=1.0, size=None) -> random values
        """
        cdef ndarray oloc
        cdef ndarray oscale
        oloc = <ndarray>PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = <ndarray>PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oscale, 0)):
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_normal, size, oloc, oscale)

    def beta(self, a, b, size=None):
        """Beta distribution over [0, 1].

        beta(a, b, size=None) -> random values
        """
        cdef ndarray oa
        cdef ndarray ob
        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        ob = <ndarray>PyArray_FROM_OTF(b, NPY_DOUBLE, NPY_ALIGNED)

        if _sp.any(_sp.less_equal(oa, 0)):
            raise ValueError("a <= 0")
        if _sp.any(_sp.less_equal(ob, 0)):        
            raise ValueError("b <= 0")
        return cont2_array(self.internal_state, rk_beta, size, oa, ob)

    def exponential(self, scale=1.0, size=None):
        """Exponential distribution.

        exponential(scale=1.0, size=None) -> random values
        """
        cdef ndarray oscale
        oscale = <ndarray> PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return cont1_array(self.internal_state, rk_exponential, size, oscale)

    def standard_exponential(self, size=None):
        """Standard exponential distribution (scale=1).

        standard_exponential(size=None) -> random values
        """
        return cont0_array(self.internal_state, rk_standard_exponential, size)

    def standard_gamma(self, shape, size=None):
        """Standard Gamma distribution.

        standard_gamma(shape, size=None) -> random values
        """
        cdef ndarray oshape
        oshape = <ndarray> PyArray_FROM_OTF(shape, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oshape, 0.0)):
            raise ValueError("shape <= 0")
        return cont1_array(self.internal_state, rk_standard_gamma, size, oshape)

    def gamma(self, shape, scale=1.0, size=None):
        """Gamma distribution.

        gamma(shape, scale=1.0, size=None) -> random values
        """
        cdef ndarray oshape
        cdef ndarray oscale
        oshape = <ndarray>PyArray_FROM_OTF(shape, NPY_DOUBLE, NPY_ALIGNED)
        oscale = <ndarray>PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oshape, 0.0)):
            raise ValueError("shape <= 0")
        if _sp.any(_sp.less_equal(oscale, 0.0)):        
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_gamma, size, oshape, oscale)

    def f(self, dfnum, dfden, size=None):
        """F distribution.

        f(dfnum, dfden, size=None) -> random values
        """
        cdef ndarray odfnum
        cdef ndarray odfden
        odfnum = <ndarray>PyArray_FROM_OTF(dfnum, NPY_DOUBLE, NPY_ALIGNED)
        odfden = <ndarray>PyArray_FROM_OTF(dfden, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(odfnum, 0.0)):
            raise ValueError("dfnum <= 0")            
        if _sp.any(_sp.less_equal(odfden, 0.0)):        
            raise ValueError("dfden <= 0")
        return cont2_array(self.internal_state, rk_f, size, odfnum, odfden)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """Noncentral F distribution.

        noncentral_f(dfnum, dfden, nonc, size=None) -> random values
        """
        cdef ndarray odfnum
        cdef ndarray odfden
        cdef ndarray ononc
        odfnum = <ndarray>PyArray_FROM_OTF(dfnum, NPY_DOUBLE, NPY_ALIGNED)
        odfden = <ndarray>PyArray_FROM_OTF(dfden, NPY_DOUBLE, NPY_ALIGNED)
        ononc = <ndarray>PyArray_FROM_OTF(nonc, NPY_DOUBLE, NPY_ALIGNED)        
        
        if _sp.any(_sp.less_equal(odfnum, 1.0)):
            raise ValueError("dfnum <= 1")            
        if _sp.any(_sp.less_equal(odfden, 0.0)):        
            raise ValueError("dfden <= 0")
        if _sp.any(_sp.less(ononc, 0.0)):
            raise ValueError("nonc < 0")
        return cont3_array(self.internal_state, rk_noncentral_f, size, odfnum,
            odfden, ononc)

    def chisquare(self, df, size=None):
        """Chi^2 distribution.

        chisquare(df, size=None) -> random values
        """
        cdef ndarray odf
        odf = <ndarray>PyArray_FROM_OTF(df, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(odf, 0.0)):        
            raise ValueError("df <= 0")
        return cont1_array(self.internal_state, rk_chisquare, size, odf)

    def noncentral_chisquare(self, df, nonc, size=None):
        """Noncentral Chi^2 distribution.

        noncentral_chisquare(df, nonc, size=None) -> random values
        """
        cdef ndarray odf
        cdef ndarray ononc
        odf = <ndarray>PyArray_FROM_OTF(df, NPY_DOUBLE, NPY_ALIGNED)
        ononc = <ndarray>PyArray_FROM_OTF(nonc, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(odf, 0.0)):
            raise ValueError("df <= 1")
        if _sp.any(_sp.less_equal(ononc, 0.0)):        
            raise ValueError("nonc < 0")
        return cont2_array(self.internal_state, rk_noncentral_chisquare, size,
            odf, ononc)
    
    def standard_cauchy(self, size=None):
        """Standard Cauchy with mode=0.

        standard_cauchy(size=None)
        """
        return cont0_array(self.internal_state, rk_standard_cauchy, size)

    def standard_t(self, df, size=None):
        """Standard Student's t distribution with df degrees of freedom.

        standard_t(df, size=None)
        """
        cdef ndarray odf
        odf = <ndarray> PyArray_FROM_OTF(df, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(odf, 0.0)):        
            raise ValueError("df <= 0")
        return cont1_array(self.internal_state, rk_standard_t, size, odf)

    def vonmises(self, mu, kappa, size=None):
        """von Mises circular distribution with mode mu and dispersion parameter
        kappa on [-pi, pi].

        vonmises(mu, kappa, size=None)
        """
        cdef ndarray omu
        cdef ndarray okappa
        omu = <ndarray> PyArray_FROM_OTF(mu, NPY_DOUBLE, NPY_ALIGNED)
        okappa = <ndarray> PyArray_FROM_OTF(kappa, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(okappa, 0.0)):        
            raise ValueError("kappa < 0")
        return cont2_array(self.internal_state, rk_vonmises, size, omu, okappa)

    def pareto(self, a, size=None):
        """Pareto distribution.

        pareto(a, size=None)
        """
        cdef ndarray oa
        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oa, 0.0)):        
            raise ValueError("a <= 0")
        return cont1_array(self.internal_state, rk_pareto, size, oa)

    def weibull(self, double a, size=None):
        """Weibull distribution.

        weibull(a, size=None)
        """
        cdef ndarray oa
        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oa, 0.0)):                
            raise ValueError("a <= 0")
        return cont1_array(self.internal_state, rk_weibull, size, oa)

    def power(self, double a, size=None):
        """Power distribution.

        power(a, size=None)
        """
        cdef ndarray oa
        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oa, 0.0)):       
            raise ValueError("a <= 0")
        return cont1_array(self.internal_state, rk_power, size, oa)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """Laplace distribution.
        
        laplace(loc=0.0, scale=1.0, size=None)
        """
        cdef ndarray oloc
        cdef ndarray oscale
        oloc = PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oscale, 0.0)):        
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_laplace, size, oloc, oscale)
    
    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """Gumbel distribution.
        
        gumbel(loc=0.0, scale=1.0, size=None)
        """
        cdef ndarray oloc
        cdef ndarray oscale
        oloc = PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oscale, 0.0)):        
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_gumbel, size, oloc, oscale)
    
    def logistic(self, loc=0.0, scale=1.0, size=None):
        """Logistic distribution.
        
        logistic(loc=0.0, scale=1.0, size=None)
        """
        cdef ndarray oloc
        cdef ndarray oscale
        oloc = PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oscale, 0.0)):        
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_logistic, size, oloc, oscale)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """Log-normal distribution.
        
        Note that the mean parameter is not the mean of this distribution, but 
        the underlying normal distribution.
        
            lognormal(mean, sigma) <=> exp(normal(mean, sigma))
        
        lognormal(mean=0.0, sigma=1.0, size=None)
        """
        cdef ndarray omean
        cdef ndarray osigma
        omean = PyArray_FROM_OTF(mean, NPY_DOUBLE, NPY_ALIGNED)
        osigma = PyArray_FROM_OTF(sigma, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(osigma, 0.0)):
            raise ValueError("sigma <= 0.0")
        return cont2_array(self.internal_state, rk_lognormal, size, omean, osigma)
    
    def rayleigh(self, scale=1.0, size=None):
        """Rayleigh distribution.
        
        rayleigh(scale=1.0, size=None)
        """
        cdef ndarray oscale
        oscale = <ndarray>PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oscale, 0.0)):        
            raise ValueError("scale <= 0.0")
        return cont1_array(self.internal_state, rk_rayleigh, size, oscale)
            
    def wald(self, mean, scale, size=None):
        """Wald (inverse Gaussian) distribution.
        
        wald(mean, scale, size=None)
        """
        cdef ndarray omean
        cdef ndarray oscale
        omean = PyArray_FROM_OTF(mean, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(omean,0.0)):
            raise ValueError("mean <= 0.0")
        elif _sp.any(_sp.less_equal(oscale,0.0)):
            raise ValueError("scale <= 0.0")
        return cont2_array(self.internal_state, rk_wald, size, omean, oscale)

    def triangular(self, left, mode, right, size=None):
        """Triangular distribution starting at left, peaking at mode, and 
        ending at right (left <= mode <= right).
        
        triangular(left, mode, right, size=None)
        """
        cdef ndarray oleft
        cdef ndarray omode
        cdef ndarray oright
        oleft = <ndarray>PyArray_FROM_OTF(left, NPY_DOUBLE, NPY_ALIGNED)
        omode = <ndarray>PyArray_FROM_OTF(mode, NPY_DOUBLE, NPY_ALIGNED)
        oright = <ndarray>PyArray_FROM_OTF(right, NPY_DOUBLE, NPY_ALIGNED)        
        
        if _sp.any(_sp.greater(oleft, omode)):
            raise ValueError("left > mode")
        if _sp.any(_sp.greater(omode, oright)):
            raise ValueError("mode > right")
        if _sp.any(_sp.equal(oleft, oright)):
            raise ValueError("left == right")
        return cont3_array(self.internal_state, rk_triangular, size, oleft, 
            omode, oright)

    # Complicated, discrete distributions:
    def binomial(self, n, p, size=None):
        """Binomial distribution of n trials and p probability of success.

        binomial(n, p, size=None) -> random values
        """
        cdef ndarray on
        cdef ndarray op
        on = <ndarray>PyArray_FROM_OTF(n, NPY_LONG, NPY_ALIGNED)
        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)        
        if _sp.any(_sp.less_equal(n, 0)):
            raise ValueError("n <= 0")
        if _sp.any(_sp.less(p, 0)):
            raise ValueError("p < 0")
        if _sp.any(_sp.greater(p, 1)):
            raise ValueError("p > 1")
        return discnp_array(self.internal_state, rk_binomial, size, on, op)

    def negative_binomial(self, n, p, size=None):
        """Negative Binomial distribution.

        negative_binomial(n, p, size=None) -> random values
        """
        cdef ndarray on
        cdef ndarray op
        on = <ndarray>PyArray_FROM_OTF(n, NPY_LONG, NPY_ALIGNED)
        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)        
        if _sp.any(_sp.less_equal(n, 0)):
            raise ValueError("n <= 0")
        if _sp.any(_sp.less(p, 0)):
            raise ValueError("p < 0")
        if _sp.any(_sp.greater(p, 1)):
            raise ValueError("p > 1")
        return discnp_array(self.internal_state, rk_negative_binomial, size,
                            on, op)

    def poisson(self, lam=1.0, size=None):
        """Poisson distribution.

        poisson(lam=1.0, size=None) -> random values
        """
        cdef ndarray olam
        olam = <ndarray>PyArray_FROM_OTF(lam, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less(olam, 0)):
            raise ValueError("lam < 0")
        return discd_array(self.internal_state, rk_poisson, size, olam)

    def zipf(self, a, size=None):
        """Zipf distribution.
        
        zipf(a, size=None)
        """
        cdef ndarray oa
        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less_equal(oa, 1.0)):
            raise ValueError("a <= 1.0")
        return discd_array(self.internal_state, rk_zipf, size, oa)
    
    def geometric(self, p, size=None):
        """Geometric distribution with p being the probability of "success" on
        an individual trial.
        
        geometric(p, size=None)
        """
        cdef ndarray op
        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less(op, 0.0)):        
            raise ValueError("p < 0.0")
        if _sp.any(_sp.greater(op, 1.0)):
            raise ValueError("p > 1.0")
        return discd_array(self.internal_state, rk_geometric, size, op)
    
    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """Hypergeometric distribution.
        
        Consider an urn with ngood "good" balls and nbad "bad" balls. If one 
        were to draw nsample balls from the urn without replacement, then 
        the hypergeometric distribution describes the distribution of "good" 
        balls in the sample.
        
        hypergeometric(ngood, nbad, nsample, size=None)        
        """
        cdef ndarray ongood
        cdef ndarray onbad
        cdef ndarray onsample

        ongood = <ndarray>PyArray_FROM_OTF(ngood, NPY_LONG, NPY_ALIGNED)
        onbad = <ndarray>PyArray_FROM_OTF(nbad, NPY_LONG, NPY_ALIGNED)
        onsample = <ndarray>PyArray_FROM_OTF(nsample, NPY_LONG, NPY_ALIGNED)
        if _sp.any(_sp.less(ongood, 1)):
            raise ValueError("ngood < 1")
        if _sp.any(_sp.less(onbad, 1)):
            raise ValueError("nbad < 1")
        if _sp.any(_sp.less(_sp.add(ongood, onbad),onsample)):
            raise ValueError("ngood + nbad < nsample")
        if _sp.any(_sp.less(onsample, 1)):
            raise ValueError("nsample < 1")
        return discnmN_array(self.internal_state, rk_hypergeometric, size,
            ongood, onbad, onsample)

    def logseries(self, p, size=None):
        """Logarithmic series distribution.
        
        logseries(p, size=None)
        """
        cdef ndarray op
        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)
        if _sp.any(_sp.less(op, 0.0)):        
            raise ValueError("p < 0.0")
        if _sp.any(_sp.greater(op, 1.0)):
            raise ValueError("p > 1.0")
        return discd_array(self.internal_state, rk_logseries, size, op)

    # Multivariate distributions:
    def multivariate_normal(self, mean, cov, size=None):
        """Return an array containing multivariate normally distributed random numbers
        with specified mean and covariance.

        multivariate_normal(mean, cov) -> random values
        multivariate_normal(mean, cov, [m, n, ...]) -> random values

        mean must be a 1 dimensional array. cov must be a square two dimensional
        array with the same number of rows and columns as mean has elements.

        The first form returns a single 1-D array containing a multivariate
        normal.

        The second form returns an array of shape (m, n, ..., cov.shape[0]).
        In this case, output[i,j,...,:] is a 1-D array containing a multivariate
        normal.
        """
        # Check preconditions on arguments
        mean = _sp.array(mean)
        cov = _sp.array(cov)
        if size is None:
            shape = []
        else:
            shape = size
        if len(mean.shape) != 1:
               raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
               raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
               raise ValueError("mean and cov must have same length")
        # Compute shape of output
        if isinstance(shape, int):
            shape = [shape]
        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])
        # Create a matrix of independent standard normally distributed random
        # numbers. The matrix has rows with the same length as mean and as
        # many rows are necessary to form a matrix of shape final_shape.
        x = standard_normal(_sp.multiply.reduce(final_shape))
        x.shape = (_sp.multiply.reduce(final_shape[0:len(final_shape)-1]),
                   mean.shape[0])
        # Transform matrix of standard normals into matrix where each row
        # contains multivariate normals with the desired covariance.
        # Compute A such that dot(transpose(A),A) == cov.
        # Then the matrix products of the rows of x and A has the desired
        # covariance. Note that sqrt(s)*v where (u,s,v) is the singular value
        # decomposition of cov is such an A.
        
        from numpy.dual import svd
        # XXX: we really should be doing this by Cholesky decomposition
        (u,s,v) = svd(cov)
        x = _sp.dot(x*_sp.sqrt(s),v)
        # The rows of x now have the correct covariance but mean 0. Add
        # mean to each row. Then each row will have mean mean.
        _sp.add(mean,x,x)
        x.shape = tuple(final_shape)
        return x

    def multinomial(self, long n, object pvals, size=None):
        """Multinomial distribution.
        
        multinomial(n, pvals, size=None) -> random values

        pvals is a sequence of probabilities that should sum to 1 (however, the
        last element is always assumed to account for the remaining probability
        as long as sum(pvals[:-1]) <= 1).
        """
        cdef long d
        cdef ndarray parr "arrayObject_parr", mnarr "arrayObject_mnarr"
        cdef double *pix
        cdef long *mnix
        cdef long i, j, dn
        cdef double Sum

        d = len(pvals)
        parr = <ndarray>PyArray_ContiguousFromObject(pvals, NPY_DOUBLE, 1, 1)
        pix = <double*>parr.data

        if kahan_sum(pix, d-1) > 1.0:
            raise ValueError("sum(pvals) > 1.0")

        if size is None:
            shape = (d,)
        elif type(size) is int:
            shape = (size, d)
        else:
            shape = size + (d,)

        multin = _sp.zeros(shape, int)
        mnarr = <ndarray>multin
        mnix = <long*>mnarr.data
        i = 0
        while i < PyArray_SIZE(mnarr):
            Sum = 1.0
            dn = n
            for j from 0 <= j < d-1:
                mnix[i+j] = rk_binomial(self.internal_state, dn, pix[j]/Sum)
                dn = dn - mnix[i+j]
                if dn <= 0:
                    break
                Sum = Sum - pix[j]
            if dn > 0:
                mnix[i+d-1] = dn

            i = i + d

        return multin

    # Shuffling and permutations:
    def shuffle(self, object x):
        """Modify a sequence in-place by shuffling its contents.
        
        shuffle(x)
        """
        cdef long i, j

        # adaptation of random.shuffle()
        i = len(x) - 1
        while i > 0:
            j = rk_interval(i, self.internal_state)
            x[i], x[j] = x[j], x[i]
            i = i - 1
                
    def permutation(self, object x):
        """Given an integer, return a shuffled sequence of integers >= 0 and 
        < x; given a sequence, return a shuffled array copy.

        permutation(x)
        """
        if type(x) is int:
            arr = _sp.arange(x)
        else:
            arr = _sp.array(x)
        self.shuffle(arr)
        return arr    

_rand = RandomState()
seed = _rand.seed
get_state = _rand.get_state
set_state = _rand.set_state
random_sample = _rand.random_sample
randint = _rand.randint
bytes = _rand.bytes
uniform = _rand.uniform
rand = _rand.rand
randn = _rand.randn
random_integers = _rand.random_integers
standard_normal = _rand.standard_normal
normal = _rand.normal
beta = _rand.beta
exponential = _rand.exponential
standard_exponential = _rand.standard_exponential
standard_gamma = _rand.standard_gamma
gamma = _rand.gamma
f = _rand.f
noncentral_f = _rand.noncentral_f
chisquare = _rand.chisquare
noncentral_chisquare = _rand.noncentral_chisquare
standard_cauchy = _rand.standard_cauchy
standard_t = _rand.standard_t
vonmises = _rand.vonmises
pareto = _rand.pareto
weibull = _rand.weibull
power = _rand.power
laplace = _rand.laplace
gumbel = _rand.gumbel
logistic = _rand.logistic
lognormal = _rand.lognormal
rayleigh = _rand.rayleigh
wald = _rand.wald
triangular = _rand.triangular

binomial = _rand.binomial
negative_binomial = _rand.negative_binomial
poisson = _rand.poisson
zipf = _rand.zipf
geometric = _rand.geometric
hypergeometric = _rand.hypergeometric
logseries = _rand.logseries

multivariate_normal = _rand.multivariate_normal
multinomial = _rand.multinomial

shuffle = _rand.shuffle
permutation = _rand.permutation
