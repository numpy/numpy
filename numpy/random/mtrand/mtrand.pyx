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
        int has_gauss
        double gauss

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
    long rk_negative_binomial(rk_state *state, double n, double p)
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
ctypedef long (* rk_discdd)(rk_state *state, double n, double p)
ctypedef long (* rk_discnmN)(rk_state *state, long n, long m, long N)
ctypedef long (* rk_discd)(rk_state *state, double a)


cdef extern from "initarray.h":
   void init_by_array(rk_state *self, unsigned long *init_key,
                      unsigned long key_length)

# Initialize numpy
import_array()

import numpy as np

cdef object cont0_array(rk_state *state, rk_cont0 func, object size):
    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state)
    else:
        array = <ndarray>np.empty(size, np.float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state)
        return array


cdef object cont1_array_sc(rk_state *state, rk_cont1 func, object size, double a):
    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, a)
    else:
        array = <ndarray>np.empty(size, np.float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a)
        return array

cdef object cont1_array(rk_state *state, rk_cont1 func, object size, ndarray oa):
    cdef double *array_data
    cdef double *oa_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef flatiter itera
    cdef broadcast multi

    if size is None:
        array = <ndarray>PyArray_SimpleNew(oa.nd, oa.dimensions, NPY_DOUBLE)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        itera = <flatiter>PyArray_IterNew(<object>oa)
        for i from 0 <= i < length:
            array_data[i] = func(state, (<double *>(itera.dataptr))[0])
            PyArray_ITER_NEXT(itera)
    else:
        array = <ndarray>np.empty(size, np.float64)
        array_data = <double *>array.data
        multi = <broadcast>PyArray_MultiIterNew(2, <void *>array,
                                                <void *>oa)
        if (multi.size != PyArray_SIZE(array)):
            raise ValueError("size is not compatible with inputs")
        for i from 0 <= i < multi.size:
            oa_data = <double *>PyArray_MultiIter_DATA(multi, 1)
            array_data[i] = func(state, oa_data[0])
            PyArray_MultiIter_NEXTi(multi, 1)
    return array

cdef object cont2_array_sc(rk_state *state, rk_cont2 func, object size, double a,
                           double b):
    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, a, b)
    else:
        array = <ndarray>np.empty(size, np.float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a, b)
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

    if size is None:
        multi = <broadcast> PyArray_MultiIterNew(2, <void *>oa, <void *>ob)
        array = <ndarray> PyArray_SimpleNew(multi.nd, multi.dimensions, NPY_DOUBLE)
        array_data = <double *>array.data
        for i from 0 <= i < multi.size:
            oa_data = <double *>PyArray_MultiIter_DATA(multi, 0)
            ob_data = <double *>PyArray_MultiIter_DATA(multi, 1)
            array_data[i] = func(state, oa_data[0], ob_data[0])
            PyArray_MultiIter_NEXT(multi)
    else:
        array = <ndarray>np.empty(size, np.float64)
        array_data = <double *>array.data
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

cdef object cont3_array_sc(rk_state *state, rk_cont3 func, object size, double a,
                           double b, double c):

    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, a, b, c)
    else:
        array = <ndarray>np.empty(size, np.float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a, b, c)
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

    if size is None:
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
        array = <ndarray>np.empty(size, np.float64)
        array_data = <double *>array.data
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
        array = <ndarray>np.empty(size, int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state)
        return array

cdef object discnp_array_sc(rk_state *state, rk_discnp func, object size, long n, double p):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, n, p)
    else:
        array = <ndarray>np.empty(size, int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, n, p)
        return array

cdef object discnp_array(rk_state *state, rk_discnp func, object size, ndarray on, ndarray op):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef double *op_data
    cdef long *on_data
    cdef broadcast multi

    if size is None:
        multi = <broadcast> PyArray_MultiIterNew(2, <void *>on, <void *>op)
        array = <ndarray> PyArray_SimpleNew(multi.nd, multi.dimensions, NPY_LONG)
        array_data = <long *>array.data
        for i from 0 <= i < multi.size:
            on_data = <long *>PyArray_MultiIter_DATA(multi, 0)
            op_data = <double *>PyArray_MultiIter_DATA(multi, 1)
            array_data[i] = func(state, on_data[0], op_data[0])
            PyArray_MultiIter_NEXT(multi)
    else:
        array = <ndarray>np.empty(size, int)
        array_data = <long *>array.data
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

cdef object discdd_array_sc(rk_state *state, rk_discdd func, object size, double n, double p):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, n, p)
    else:
        array = <ndarray>np.empty(size, int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, n, p)
        return array

cdef object discdd_array(rk_state *state, rk_discdd func, object size, ndarray on, ndarray op):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef double *op_data
    cdef double *on_data
    cdef broadcast multi

    if size is None:
        multi = <broadcast> PyArray_MultiIterNew(2, <void *>on, <void *>op)
        array = <ndarray> PyArray_SimpleNew(multi.nd, multi.dimensions, NPY_LONG)
        array_data = <long *>array.data
        for i from 0 <= i < multi.size:
            on_data = <double *>PyArray_MultiIter_DATA(multi, 0)
            op_data = <double *>PyArray_MultiIter_DATA(multi, 1)
            array_data[i] = func(state, on_data[0], op_data[0])
            PyArray_MultiIter_NEXT(multi)
    else:
        array = <ndarray>np.empty(size, int)
        array_data = <long *>array.data
        multi = <broadcast>PyArray_MultiIterNew(3, <void*>array, <void *>on, <void *>op)
        if (multi.size != PyArray_SIZE(array)):
            raise ValueError("size is not compatible with inputs")
        for i from 0 <= i < multi.size:
            on_data = <double *>PyArray_MultiIter_DATA(multi, 1)
            op_data = <double *>PyArray_MultiIter_DATA(multi, 2)
            array_data[i] = func(state, on_data[0], op_data[0])
            PyArray_MultiIter_NEXTi(multi, 1)
            PyArray_MultiIter_NEXTi(multi, 2)

    return array

cdef object discnmN_array_sc(rk_state *state, rk_discnmN func, object size,
    long n, long m, long N):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, n, m, N)
    else:
        array = <ndarray>np.empty(size, int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, n, m, N)
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

    if size is None:
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
        array = <ndarray>np.empty(size, int)
        array_data = <long *>array.data
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

cdef object discd_array_sc(rk_state *state, rk_discd func, object size, double a):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, a)
    else:
        array = <ndarray>np.empty(size, int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a)
        return array

cdef object discd_array(rk_state *state, rk_discd func, object size, ndarray oa):
    cdef long *array_data
    cdef double *oa_data
    cdef ndarray array "arrayObject"
    cdef npy_intp length
    cdef npy_intp i
    cdef broadcast multi
    cdef flatiter itera

    if size is None:
        array = <ndarray>PyArray_SimpleNew(oa.nd, oa.dimensions, NPY_LONG)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        itera = <flatiter>PyArray_IterNew(<object>oa)
        for i from 0 <= i < length:
            array_data[i] = func(state, (<double *>(itera.dataptr))[0])
            PyArray_ITER_NEXT(itera)
    else:
        array = <ndarray>np.empty(size, int)
        array_data = <long *>array.data
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
    """
    RandomState(seed=None)

    Container for the Mersenne Twister PRNG.

    `RandomState` exposes a number of methods for generating random numbers
    drawn from a variety of probability distributions. In addition to the
    distribution-specific arguments, each method takes a keyword argument
    `size` that defaults to ``None``. If `size` is ``None``, then a single
    value is generated and returned. If `size` is an integer, then a 1-D
    numpy array filled with generated values is returned. If size is a tuple,
    then a numpy array with that shape is filled and returned.

    Parameters
    ----------
    seed : {None, int, array-like}
        Random seed initializing the PRNG.
        Can be an integer, an array (or other sequence) of integers of
        any length, or ``None``.
        If `seed` is ``None``, then `RandomState` will try to read data from
        ``/dev/urandom`` (or the Windows analogue) if available or seed from
        the clock otherwise.

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
        """
        seed(seed=None)

        Seed the generator.

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
        elif isinstance(seed, np.integer):
            iseed = int(seed)
            rk_seed(iseed, self.internal_state)
        else:
            obj = <ndarray>PyArray_ContiguousFromObject(seed, NPY_LONG, 1, 1)
            init_by_array(self.internal_state, <unsigned long *>(obj.data),
                obj.dimensions[0])

    def get_state(self):
        """
        get_state()

        Return a tuple representing the internal state of the generator::

            ('MT19937', int key[624], int pos, int has_gauss, float cached_gaussian)

        """
        cdef ndarray state "arrayObject_state"
        state = <ndarray>np.empty(624, np.uint)
        memcpy(<void*>(state.data), <void*>(self.internal_state.key), 624*sizeof(long))
        state = <ndarray>np.asarray(state, np.uint32)
        return ('MT19937', state, self.internal_state.pos,
            self.internal_state.has_gauss, self.internal_state.gauss)

    def set_state(self, state):
        """
        set_state(state)

        Set the state from a tuple.

        state = ('MT19937', int key[624], int pos, int has_gauss, float cached_gaussian)

        For backwards compatibility, the following form is also accepted
        although it is missing some information about the cached Gaussian value.

        state = ('MT19937', int key[624], int pos)

        """
        cdef ndarray obj "arrayObject_obj"
        cdef int pos
        algorithm_name = state[0]
        if algorithm_name != 'MT19937':
            raise ValueError("algorithm must be 'MT19937'")
        key, pos = state[1:3]
        if len(state) == 3:
            has_gauss = 0
            cached_gaussian = 0.0
        else:
            has_gauss, cached_gaussian = state[3:5]
        try:
            obj = <ndarray>PyArray_ContiguousFromObject(key, NPY_ULONG, 1, 1)
        except TypeError:
            # compatibility -- could be an older pickle
            obj = <ndarray>PyArray_ContiguousFromObject(key, NPY_LONG, 1, 1)
        if obj.dimensions[0] != 624:
            raise ValueError("state must be 624 longs")
        memcpy(<void*>(self.internal_state.key), <void*>(obj.data), 624*sizeof(long))
        self.internal_state.pos = pos
        self.internal_state.has_gauss = has_gauss
        self.internal_state.gauss = cached_gaussian

    # Pickling support:
    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)

    def __reduce__(self):
        return (np.random.__RandomState_ctor, (), self.get_state())

    # Basic distributions:
    def random_sample(self, size=None):
        """
        random_sample(size=None)

        Return random floats in the half-open interval [0.0, 1.0).

        """
        return cont0_array(self.internal_state, rk_double, size)

    def tomaxint(self, size=None):
        """
        tomaxint(size=None)

        Uniformly sample discrete random integers `x` such that
        ``0 <= x <= sys.maxint``.

        Parameters
        ----------
        size : tuple of ints, int, optional
            Shape of output.  If the given size is, for example, (m,n,k),
            m*n*k samples are generated.  If no shape is specified, a single sample
            is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Uniform sampling over a given half-open interval of integers.
        random_integers : Uniform sampling over a given closed interval of
            integers.

        """
        return disc0_array(self.internal_state, rk_long, size)

    def randint(self, low, high=None, size=None):
        """
        randint(low, high=None, size=None)

        Return random integers x such that low <= x < high.

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
            return <long>rk_interval(diff, self.internal_state) + lo
        else:
            array = <ndarray>np.empty(size, int)
            length = PyArray_SIZE(array)
            array_data = <long *>array.data
            for i from 0 <= i < length:
                array_data[i] = lo + <long>rk_interval(diff, self.internal_state)
            return array

    def bytes(self, unsigned int length):
        """
        bytes(length)

        Return random bytes.

        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : str
            String of length `N`.

        Examples
        --------
        >>> np.random.bytes(10)
        ' eh\\x85\\x022SZ\\xbf\\xa4' #random

        """
        cdef void *bytes
        bytestring = PyString_FromStringAndSize(NULL, length)
        bytes = PyString_AS_STRING(bytestring)
        rk_fill(bytes, length, self.internal_state)
        return bytestring

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        uniform(low=0.0, high=1.0, size=1)

        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        Parameters
        ----------
        low : float, optional
            Lower boundary of the output interval.  All values generated will be
            greater than or equal to low.  The default value is 0.
        high : float
            Upper boundary of the output interval.  All values generated will be
            less than high.  The default value is 1.0.
        size : tuple of ints, int, optional
            Shape of output.  If the given size is, for example, (m,n,k),
            m*n*k samples are generated.  If no shape is specified, a single sample
            is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Discrete uniform distribution, yielding integers.
        random_integers : Discrete uniform distribution over the closed interval
                          ``[low, high]``.
        random_sample : Floats uniformly distributed over ``[0, 1)``.
        random : Alias for `random_sample`.
        rand : Convenience function that accepts dimensions as input, e.g.,
               ``rand(2,2)`` would generate a 2-by-2 array of floats, uniformly
               distributed over ``[0, 1)``.

        Notes
        -----
        The probability density function of the uniform distribution is

        .. math:: p(x) = \\frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.

        Examples
        --------
        Draw samples from the distribution:

        >>> s = np.random.uniform(-1,0,1000)

        All values are within the given interval:

        >>> np.all(s >= -1)
        True

        >>> np.all(s < 0)
        True

        Display the histogram of the samples, along with the
        probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 15, normed=True)
        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef ndarray olow, ohigh, odiff
        cdef double flow, fhigh
        cdef object temp

        flow = PyFloat_AsDouble(low)
        fhigh = PyFloat_AsDouble(high)
        if not PyErr_Occurred():
            return cont2_array_sc(self.internal_state, rk_uniform, size, flow, fhigh-flow)
        PyErr_Clear()
        olow = <ndarray>PyArray_FROM_OTF(low, NPY_DOUBLE, NPY_ALIGNED)
        ohigh = <ndarray>PyArray_FROM_OTF(high, NPY_DOUBLE, NPY_ALIGNED)
        temp = np.subtract(ohigh, olow)
        Py_INCREF(temp) # needed to get around Pyrex's automatic reference-counting
                        #  rules because EnsureArray steals a reference
        odiff = <ndarray>PyArray_EnsureArray(temp)
        return cont2_array(self.internal_state, rk_uniform, size, olow, odiff)

    def rand(self, *args):
        """
        rand(d0, d1, ..., dn)

        Random values in a given shape.

        Create an array of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        d0, d1, ..., dn : int
            Shape of the output.

        Returns
        -------
        out : ndarray, shape ``(d0, d1, ..., dn)``
            Random values.

        See Also
        --------
        random

        Notes
        -----
        This is a convenience function. If you want an interface that
        takes a shape-tuple as the first argument, refer to
        `random`.

        Examples
        --------
        >>> np.random.rand(3,2)
        array([[ 0.14022471,  0.96360618],  #random
               [ 0.37601032,  0.25528411],  #random
               [ 0.49313049,  0.94909878]]) #random

        """
        if len(args) == 0:
            return self.random_sample()
        else:
            return self.random_sample(size=args)

    def randn(self, *args):
        """
        randn(d0, d1, ..., dn)

        Returns zero-mean, unit-variance Gaussian random numbers in an
        array of shape (d0, d1, ..., dn).

        Note:  This is a convenience function. If you want an
                    interface that takes a tuple as the first argument
                    use numpy.random.standard_normal(shape_tuple).

        """
        if len(args) == 0:
            return self.standard_normal()
        else:
            return self.standard_normal(args)

    def random_integers(self, low, high=None, size=None):
        """
        random_integers(low, high=None, size=None)

        Return random integers x such that low <= x <= high.

        If high is None, then 1 <= x <= low.

        """
        if high is None:
            high = low
            low = 1
        return self.randint(low, high+1, size)

    # Complicated, continuous distributions:
    def standard_normal(self, size=None):
        """
        standard_normal(size=None)

        Standard Normal distribution (mean=0, stdev=1).

        """
        return cont0_array(self.internal_state, rk_gauss, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        normal(loc=0.0, scale=1.0, size=None)

        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first
        derived by De Moivre and 200 years later by both Gauss and Laplace
        independently [2]_, is often called the bell curve because of
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature.  For example, it
        describes the commonly occurring distribution of samples influenced
        by a large number of tiny, random disturbances, each with its own
        unique distribution [2]_.

        Parameters
        ----------
        loc : float
            Mean ("centre") of the distribution.
        scale : float
            Standard deviation (spread or "width") of the distribution.
        size : tuple of ints
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        See Also
        --------
        scipy.stats.distributions.norm : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gaussian distribution is

        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard deviation.
        The square of the standard deviation, :math:`\\sigma^2`, is called the
        variance.

        The function has its peak at the mean, and its "spread" increases with
        the standard deviation (the function reaches 0.607 times its maximum at
        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
        `numpy.random.normal` is more likely to return samples lying close to the
        mean, rather than those far away.

        References
        ----------
        .. [1] Wikipedia, "Normal distribution",
               http://en.wikipedia.org/wiki/Normal_distribution
        .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability, Random
               Variables and Random Signal Principles", 4th ed., 2001,
               pp. 51, 51, 125.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 0, 0.1 # mean and standard deviation
        >>> s = np.random.normal(mu, sigma, 1000)

        Verify the mean and the variance:

        >>> abs(mu - np.mean(s)) < 0.01
        True

        >>> abs(sigma - np.std(s, ddof=1)) < 0.01
        True

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        """
        cdef ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont2_array_sc(self.internal_state, rk_normal, size, floc, fscale)

        PyErr_Clear()

        oloc = <ndarray>PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = <ndarray>PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oscale, 0)):
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_normal, size, oloc, oscale)

    def beta(self, a, b, size=None):
        """
        beta(a, b, size=None)

        The Beta distribution over ``[0, 1]``.

        The Beta distribution is a special case of the Dirichlet distribution,
        and is related to the Gamma distribution.  It has the probability
        distribution function

        .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                         (1 - x)^{\\beta - 1},

        where the normalisation, B, is the beta function,

        .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                     (1 - t)^{\\beta - 1} dt.

        It is often seen in Bayesian inference and order statistics.

        Parameters
        ----------
        a : float
            Alpha, non-negative.
        b : float
            Beta, non-negative.
        size : tuple of ints, optional
            The number of samples to draw.  The ouput is packed according to
            the size given.

        Returns
        -------
        out : ndarray
            Array of the given shape, containing values drawn from a
            Beta distribution.

        """
        cdef ndarray oa, ob
        cdef double fa, fb

        fa = PyFloat_AsDouble(a)
        fb = PyFloat_AsDouble(b)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            if fb <= 0:
                raise ValueError("b <= 0")
            return cont2_array_sc(self.internal_state, rk_beta, size, fa, fb)

        PyErr_Clear()

        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        ob = <ndarray>PyArray_FROM_OTF(b, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oa, 0)):
            raise ValueError("a <= 0")
        if np.any(np.less_equal(ob, 0)):
            raise ValueError("b <= 0")
        return cont2_array(self.internal_state, rk_beta, size, oa, ob)

    def exponential(self, scale=1.0, size=None):
        """
        exponential(scale=1.0, size=None)

        Exponential distribution.

        Its probability density function is

        .. math:: f(x; \\lambda) = \\lambda \\exp(-\\lambda x),

        for ``x > 0`` and 0 elsewhere.  :math:`lambda` is
        known as the rate parameter.

        The exponential distribution is a continuous analogue of the
        geometric distribution.  It describes many common situations, such as
        the size of raindrops measured over many rainstorms [1]_, or the time
        between page requests to Wikipedia [2]_.

        Parameters
        ----------
        scale : float
            The rate parameter, :math:`\\lambda`.
        size : tuple of ints
            Number of samples to draw.  The output is shaped
            according to `size`.

        References
        ----------
        .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
               Random Signal Principles", 4th ed, 2001, p. 57.
        .. [2] "Poisson Process", Wikipedia,
               http://en.wikipedia.org/wiki/Poisson_process

        """
        cdef ndarray oscale
        cdef double fscale

        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont1_array_sc(self.internal_state, rk_exponential, size, fscale)

        PyErr_Clear()

        oscale = <ndarray> PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return cont1_array(self.internal_state, rk_exponential, size, oscale)

    def standard_exponential(self, size=None):
        """
        standard_exponential(size=None)

        Standard exponential distribution (scale=1).

        """
        return cont0_array(self.internal_state, rk_standard_exponential, size)

    def standard_gamma(self, shape, size=None):
        """
        standard_gamma(shape, size=None)

        Standard Gamma distribution.

        """
        cdef ndarray oshape
        cdef double fshape

        fshape = PyFloat_AsDouble(shape)
        if not PyErr_Occurred():
            if fshape <= 0:
                raise ValueError("shape <= 0")
            return cont1_array_sc(self.internal_state, rk_standard_gamma, size, fshape)

        PyErr_Clear()
        oshape = <ndarray> PyArray_FROM_OTF(shape, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oshape, 0.0)):
            raise ValueError("shape <= 0")
        return cont1_array(self.internal_state, rk_standard_gamma, size, oshape)

    def gamma(self, shape, scale=1.0, size=None):
        """
        gamma(shape, scale=1.0, size=None)

        Gamma distribution.

        """
        cdef ndarray oshape, oscale
        cdef double fshape, fscale

        fshape = PyFloat_AsDouble(shape)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fshape <= 0:
                raise ValueError("shape <= 0")
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont2_array_sc(self.internal_state, rk_gamma, size, fshape, fscale)

        PyErr_Clear()
        oshape = <ndarray>PyArray_FROM_OTF(shape, NPY_DOUBLE, NPY_ALIGNED)
        oscale = <ndarray>PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oshape, 0.0)):
            raise ValueError("shape <= 0")
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_gamma, size, oshape, oscale)

    def f(self, dfnum, dfden, size=None):
        """
        f(dfnum, dfden, size=None)

        F distribution.

        """
        cdef ndarray odfnum, odfden
        cdef double fdfnum, fdfden

        fdfnum = PyFloat_AsDouble(dfnum)
        fdfden = PyFloat_AsDouble(dfden)
        if not PyErr_Occurred():
            if fdfnum <= 0:
                raise ValueError("shape <= 0")
            if fdfden <= 0:
                raise ValueError("scale <= 0")
            return cont2_array_sc(self.internal_state, rk_f, size, fdfnum, fdfden)

        PyErr_Clear()

        odfnum = <ndarray>PyArray_FROM_OTF(dfnum, NPY_DOUBLE, NPY_ALIGNED)
        odfden = <ndarray>PyArray_FROM_OTF(dfden, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(odfnum, 0.0)):
            raise ValueError("dfnum <= 0")
        if np.any(np.less_equal(odfden, 0.0)):
            raise ValueError("dfden <= 0")
        return cont2_array(self.internal_state, rk_f, size, odfnum, odfden)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """
        noncentral_f(dfnum, dfden, nonc, size=None)

        Noncentral F distribution.

        """
        cdef ndarray odfnum, odfden, ononc
        cdef double fdfnum, fdfden, fnonc

        fdfnum = PyFloat_AsDouble(dfnum)
        fdfden = PyFloat_AsDouble(dfden)
        fnonc = PyFloat_AsDouble(nonc)
        if not PyErr_Occurred():
            if fdfnum <= 1:
                raise ValueError("dfnum <= 1")
            if fdfden <= 0:
                raise ValueError("dfden <= 0")
            if fnonc < 0:
                raise ValueError("nonc < 0")
            return cont3_array_sc(self.internal_state, rk_noncentral_f, size,
                                  fdfnum, fdfden, fnonc)

        PyErr_Clear()

        odfnum = <ndarray>PyArray_FROM_OTF(dfnum, NPY_DOUBLE, NPY_ALIGNED)
        odfden = <ndarray>PyArray_FROM_OTF(dfden, NPY_DOUBLE, NPY_ALIGNED)
        ononc = <ndarray>PyArray_FROM_OTF(nonc, NPY_DOUBLE, NPY_ALIGNED)

        if np.any(np.less_equal(odfnum, 1.0)):
            raise ValueError("dfnum <= 1")
        if np.any(np.less_equal(odfden, 0.0)):
            raise ValueError("dfden <= 0")
        if np.any(np.less(ononc, 0.0)):
            raise ValueError("nonc < 0")
        return cont3_array(self.internal_state, rk_noncentral_f, size, odfnum,
            odfden, ononc)

    def chisquare(self, df, size=None):
        """
        chisquare(df, size=None)

        Draw samples from a chi-square distribution.

        When `df` independent random variables, each with standard
        normal distributions (mean 0, variance 1), are squared and summed,
        the resulting distribution is chi-square (see Notes).  This
        distribution is often used in hypothesis testing.

        Parameters
        ----------
        df : int
             Number of degrees of freedom.
        size : tuple of ints, int, optional
             Size of the returned array.  By default, a scalar is
             returned.

        Returns
        -------
        output : ndarray
            Samples drawn from the distribution, packed in a `size`-shaped
            array.

        Raises
        ------
        ValueError
            When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
            is given.

        Notes
        -----
        The variable obtained by summing the squares of `df` independent,
        standard normally distributed random variables:

        .. math:: Q = \\sum_{i=0}^{\\mathtt{df}} X^2_i

        is chi-square distributed, denoted

        .. math:: Q \\sim \\chi^2_k.

        The probability density function of the chi-squared distribution is

        .. math:: p(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}
                         x^{k/2 - 1} e^{-x/2},

        where :math:`\\Gamma` is the gamma function,

        .. math:: \\Gamma(x) = \\int_0^{-\\infty} t^{x - 1} e^{-t} dt.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods,
               http://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm
        .. [2] Wikipedia, "Chi-square distribution",
               http://en.wikipedia.org/wiki/Chi-square_distribution

        Examples
        --------
        >>> np.random.chisquare(2,4)
        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272])

        """
        cdef ndarray odf
        cdef double fdf

        fdf = PyFloat_AsDouble(df)
        if not PyErr_Occurred():
            if fdf <= 0:
                raise ValueError("df <= 0")
            return cont1_array_sc(self.internal_state, rk_chisquare, size, fdf)

        PyErr_Clear()

        odf = <ndarray>PyArray_FROM_OTF(df, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(odf, 0.0)):
            raise ValueError("df <= 0")
        return cont1_array(self.internal_state, rk_chisquare, size, odf)

    def noncentral_chisquare(self, df, nonc, size=None):
        """
        noncentral_chisquare(df, nonc, size=None)

        Draw samples from a noncentral chi-square distribution.

        The noncentral :math:`\\chi^2` distribution is a generalisation of
        the :math:`\\chi^2` distribution.

        Parameters
        ----------
        df : int
            Degrees of freedom.
        nonc : float
            Non-centrality.
        size : tuple of ints
            Shape of the output.

        """
        cdef ndarray odf, ononc
        cdef double fdf, fnonc
        fdf = PyFloat_AsDouble(df)
        fnonc = PyFloat_AsDouble(nonc)
        if not PyErr_Occurred():
            if fdf <= 1:
                raise ValueError("df <= 0")
            if fnonc <= 0:
                raise ValueError("nonc <= 0")
            return cont2_array_sc(self.internal_state, rk_noncentral_chisquare,
                                  size, fdf, fnonc)

        PyErr_Clear()

        odf = <ndarray>PyArray_FROM_OTF(df, NPY_DOUBLE, NPY_ALIGNED)
        ononc = <ndarray>PyArray_FROM_OTF(nonc, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(odf, 0.0)):
            raise ValueError("df <= 1")
        if np.any(np.less_equal(ononc, 0.0)):
            raise ValueError("nonc < 0")
        return cont2_array(self.internal_state, rk_noncentral_chisquare, size,
            odf, ononc)

    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Standard Cauchy with mode=0.

        """
        return cont0_array(self.internal_state, rk_standard_cauchy, size)

    def standard_t(self, df, size=None):
        """
        standard_t(df, size=None)

        Standard Student's t distribution with df degrees of freedom.

        """
        cdef ndarray odf
        cdef double fdf

        fdf = PyFloat_AsDouble(df)
        if not PyErr_Occurred():
            if fdf <= 0:
                raise ValueError("df <= 0")
            return cont1_array_sc(self.internal_state, rk_standard_t, size, fdf)

        PyErr_Clear()

        odf = <ndarray> PyArray_FROM_OTF(df, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(odf, 0.0)):
            raise ValueError("df <= 0")
        return cont1_array(self.internal_state, rk_standard_t, size, odf)

    def vonmises(self, mu, kappa, size=None):
        """
        vonmises(mu=0.0, kappa=1.0, size=None)

        Draw samples from a von Mises distribution.

        Samples are drawn from a von Mises distribution with specified mode (mu)
        and dispersion (kappa), on the interval [-pi, pi].

        The von Mises distribution (also known as the circular normal
        distribution) is a continuous probability distribution on the circle. It
        may be thought of as the circular analogue of the normal distribution.

        Parameters
        ----------
        mu : float
            Mode ("center") of the distribution.
        kappa : float, >= 0.
            Dispersion of the distribution.
        size : {tuple, int}
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        Returns
        -------
        samples : {ndarray, scalar}
            The returned samples live on the unit circle [-\\pi, \\pi].

        See Also
        --------
        scipy.stats.distributions.vonmises : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the von Mises distribution is

        .. math:: p(x) = \\frac{e^{\\kappa cos(x-\\mu)}}{2\\pi I_0(\\kappa)},

        where :math:`\\mu` is the mode and :math:`\\kappa` the dispersion,
        and :math:`I_0(\\kappa)` is the modified Bessel function of order 0.

        The von Mises, named for Richard Edler von Mises, born in
        Austria-Hungary, in what is now the Ukraine. He fled to the United
        States in 1939 and became a professor at Harvard. He worked in
        probability theory, aerodynamics, fluid mechanics, and philosophy of
        science.

        References
        ----------
        .. [1] Abramowitz, M. and Stegun, I. A. (ed.), Handbook of Mathematical
               Functions, National Bureau of Standards, 1964; reprinted Dover
               Publications, 1965.
        .. [2] von Mises, Richard, 1964, Mathematical Theory of Probability
               and Statistics (New York: Academic Press).
        .. [3] Wikipedia, "Von Mises distribution",
               http://en.wikipedia.org/wiki/Von_Mises_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, kappa = 0.0, 4.0 # mean and dispersion
        >>> s = np.random.vonmises(mu, kappa, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        >>> count, bins, ignored = plt.hist(s, 50, normed=True)
        >>> x = arange(-pi, pi, 2*pi/50.)
        >>> y = -np.exp(kappa*np.cos(x-mu))/(2*pi*sps.jn(0,kappa))
        >>> plt.plot(x, y/max(y), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef ndarray omu, okappa
        cdef double fmu, fkappa

        fmu = PyFloat_AsDouble(mu)
        fkappa = PyFloat_AsDouble(kappa)
        if not PyErr_Occurred():
            if fkappa < 0:
                raise ValueError("kappa < 0")
            return cont2_array_sc(self.internal_state, rk_vonmises, size, fmu, fkappa)

        PyErr_Clear()

        omu = <ndarray> PyArray_FROM_OTF(mu, NPY_DOUBLE, NPY_ALIGNED)
        okappa = <ndarray> PyArray_FROM_OTF(kappa, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less(okappa, 0.0)):
            raise ValueError("kappa < 0")
        return cont2_array(self.internal_state, rk_vonmises, size, omu, okappa)

    def pareto(self, a, size=None):
        """
        pareto(a, size=None)

        Draw samples from a Pareto distribution with specified shape.

        This is a simplified version of the Generalized Pareto distribution
        (available in SciPy), with the scale set to one and the location set to
        zero. Most authors default the location to one.

        The Pareto distribution must be greater than zero, and is unbounded above.
        It is also known as the "80-20 rule".  In this distribution, 80 percent of
        the weights are in the lowest 20 percent of the range, while the other 20
        percent fill the remaining 80 percent of the range.

        Parameters
        ----------
        shape : float, > 0.
            Shape of the distribution.
        size : tuple of ints
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        See Also
        --------
        scipy.stats.distributions.genpareto.pdf : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Pareto distribution is

        .. math:: p(x) = \\frac{am^a}{x^{a+1}}

        where :math:`a` is the shape and :math:`m` the location

        The Pareto distribution, named after the Italian economist Vilfredo Pareto,
        is a power law probability distribution useful in many real world problems.
        Outside the field of economics it is generally referred to as the Bradford
        distribution. Pareto developed the distribution to describe the
        distribution of wealth in an economy.  It has also found use in insurance,
        web page access statistics, oil field sizes, and many other problems,
        including the download frequency for projects in Sourceforge [1].  It is
        one of the so-called "fat-tailed" distributions.


        References
        ----------
        .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
               Sourceforge projects.
        .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
        .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
               Values, Birkhauser Verlag, Basel, pp 23-30.
        .. [4] Wikipedia, "Pareto distribution",
               http://en.wikipedia.org/wiki/Pareto_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a, m = 3., 1. # shape and mode
        >>> s = np.random.pareto(a, 1000) + m

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 100, normed=True, align='center')
        >>> fit = a*m**a/bins**(a+1)
        >>> plt.plot(bins, max(count)*fit/max(fit),linewidth=2, color='r')
        >>> plt.show()

        """
        cdef ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            return cont1_array_sc(self.internal_state, rk_pareto, size, fa)

        PyErr_Clear()

        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oa, 0.0)):
            raise ValueError("a <= 0")
        return cont1_array(self.internal_state, rk_pareto, size, oa)

    def weibull(self, a, size=None):
        """
        weibull(a, size=None)

        Weibull distribution.

        Draw samples from a 1-parameter Weibull distribution with the given
        shape parameter.

        .. math:: X = (-ln(U))^{1/a}

        Here, U is drawn from the uniform distribution over (0,1].

        The more common 2-parameter Weibull, including a scale parameter
        :math:`\\lambda` is just :math:`X = \\lambda(-ln(U))^{1/a}`.

        The Weibull (or Type III asymptotic extreme value distribution for smallest
        values, SEV Type III, or Rosin-Rammler distribution) is one of a class of
        Generalized Extreme Value (GEV) distributions used in modeling extreme
        value problems.  This class includes the Gumbel and Frechet distributions.

        Parameters
        ----------
        a : float
            Shape of the distribution.
        size : tuple of ints
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        See Also
        --------
        scipy.stats.distributions.weibull : probability density function,
            distribution or cumulative density function, etc.

        gumbel, scipy.stats.distributions.genextreme

        Notes
        -----
        The probability density for the Weibull distribution is

        .. math:: p(x) = \\frac{a}
                         {\\lambda}(\\frac{x}{\\lambda})^{a-1}e^{-(x/\\lambda)^a},

        where :math:`a` is the shape and :math:`\\lambda` the scale.

        The function has its peak (the mode) at
        :math:`\\lambda(\\frac{a-1}{a})^{1/a}`.

        When ``a = 1``, the Weibull distribution reduces to the exponential
        distribution.

        References
        ----------
        .. [1] Waloddi Weibull, Professor, Royal Technical University, Stockholm,
               1939 "A Statistical Theory Of The Strength Of Materials",
               Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,
               Generalstabens Litografiska Anstalts Forlag, Stockholm.
        .. [2] Waloddi Weibull, 1951 "A Statistical Distribution Function of Wide
               Applicability",  Journal Of Applied Mechanics ASME Paper.
        .. [3] Wikipedia, "Weibull distribution",
               http://en.wikipedia.org/wiki/Weibull_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 5. # shape
        >>> s = np.random.weibull(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> def weib(x,n,a):
        ...     return (a/n)*(x/n)**(a-1)*exp(-(x/n)**a)

        >>> count, bins, ignored = plt.hist(numpy.random.weibull(5.,1000))
        >>> scale = count.max()/weib(x, 1., 5.).max()
        >>> x = arange(1,100.)/50.
        >>> plt.plot(x, weib(x, 1., 5.)*scale)
        >>> plt.show()

        """
        cdef ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            return cont1_array_sc(self.internal_state, rk_weibull, size, fa)

        PyErr_Clear()

        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oa, 0.0)):
            raise ValueError("a <= 0")
        return cont1_array(self.internal_state, rk_weibull, size, oa)

    def power(self, a, size=None):
        """
        power(a, size=None)

        Power distribution.

        """
        cdef ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            return cont1_array_sc(self.internal_state, rk_power, size, fa)

        PyErr_Clear()

        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oa, 0.0)):
            raise ValueError("a <= 0")
        return cont1_array(self.internal_state, rk_power, size, oa)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """
        laplace(loc=0.0, scale=1.0, size=None)

        Laplace or double exponential distribution.

        It has the probability density function

        .. math:: f(x; \\mu, \\lambda) = \\frac{1}{2\\lambda}
                                       \\exp\\left(-\\frac{|x - \\mu|}{\\lambda}\\right).

        The Laplace distribution is similar to the Gaussian/normal distribution,
        but is sharper at the peak and has fatter tails.

        Parameters
        ----------
        loc : float
            The position, :math:`\\mu`, of the distribution peak.
        scale : float
            :math:`\\lambda`, the exponential decay.

        """
        cdef ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont2_array_sc(self.internal_state, rk_laplace, size, floc, fscale)

        PyErr_Clear()
        oloc = PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_laplace, size, oloc, oscale)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """
        gumbel(loc=0.0, scale=1.0, size=None)

        Gumbel distribution.

        Draw samples from a Gumbel distribution with specified location (or mean)
        and scale (or standard deviation).

        The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme Value
        Type I) distribution is one of a class of Generalized Extreme Value (GEV)
        distributions used in modeling extreme value problems.  The Gumbel is a
        special case of the Extreme Value Type I distribution for maximums from
        distributions with "exponential-like" tails, it may be derived by
        considering a Gaussian process of measurements, and generating the pdf for
        the maximum values from that set of measurements (see examples).

        Parameters
        ----------
        loc : float
            The location of the mode of the distribution.
        scale : float
            The scale parameter of the distribution.
        size : tuple of ints
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        See Also
        --------
        scipy.stats.gumbel : probability density function,
            distribution or cumulative density function, etc.
        weibull, scipy.stats.genextreme

        Notes
        -----
        The probability density for the Gumbel distribution is

        .. math:: p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta} e^{ -e^{-(x - \\mu)/
                  \\beta}},

        where :math:`\\mu` is the mode, a location parameter, and :math:`\\beta`
        is the scale parameter.

        The Gumbel (named for German mathematician Emil Julius Gumbel) was used
        very early in the hydrology literature, for modeling the occurrence of
        flood events. It is also used for modeling maximum wind speed and rainfall
        rates.  It is a "fat-tailed" distribution - the probability of an event in
        the tail of the distribution is larger than if one used a Gaussian, hence
        the surprisingly frequent occurrence of 100-year floods. Floods were
        initially modeled as a Gaussian process, which underestimated the frequency
        of extreme events.

        It is one of a class of extreme value distributions, the Generalized
        Extreme Value (GEV) distributions, which also includes the Weibull and
        Frechet.

        The function has a mean of :math:`\\mu + 0.57721\\beta` and a variance of
        :math:`\\frac{\\pi^2}{6}\\beta^2`.

        References
        ----------
        .. [1] Gumbel, E.J. (1958). Statistics of Extremes. Columbia University
               Press.
        .. [2] Reiss, R.-D. and Thomas M. (2001), Statistical Analysis of Extreme
               Values, from Insurance, Finance, Hydrology and Other Fields,
               Birkhauser Verlag, Basel: Boston : Berlin.
        .. [3] Wikipedia, "Gumbel distribution",
               http://en.wikipedia.org/wiki/Gumbel_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, beta = 0, 0.1 # location and scale
        >>> s = np.random.gumbel(mu, beta, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)*
        ...          np.exp( -np.exp( -(bins - mu) /beta) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Show how an extreme value distribution can arise from a Gaussian process
        and compare to a Gaussian:

        >>> means = []
        >>> maxima = []
        >>> for i in range(0,1000) :
        ...    a = np.random.normal(mu, beta, 1000)
        ...    means.append(a.mean())
        ...    maxima.append(a.max())
        >>> count, bins, ignored = plt.hist(maxima, 30, normed=True)
        >>> beta = np.std(maxima)*np.pi/np.sqrt(6)
        >>> mu = np.mean(maxima) - 0.57721*beta
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)*
        ...          np.exp( -np.exp( -(bins - mu) /beta) ),
        ...          linewidth=2, color='r')
        >>> plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi)) *
        ...          np.exp( - (bins - mu)**2 / (2 * beta**2) ),
        ...          linewidth=2, color='g')
        >>> plt.show()

        """
        cdef ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont2_array_sc(self.internal_state, rk_gumbel, size, floc, fscale)

        PyErr_Clear()
        oloc = PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_gumbel, size, oloc, oscale)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """
        logistic(loc=0.0, scale=1.0, size=None)

        Logistic distribution.

        """
        cdef ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont2_array_sc(self.internal_state, rk_logistic, size, floc, fscale)

        PyErr_Clear()
        oloc = PyArray_FROM_OTF(loc, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return cont2_array(self.internal_state, rk_logistic, size, oloc, oscale)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """
        lognormal(mean=0.0, sigma=1.0, size=None)

        Log-normal distribution.

        Draw samples from a log-normal distribution with specified mean, standard
        deviation, and shape. Note that the mean and standard deviation are not the
        values for the distribution itself, but of the underlying normal
        distribution it is derived from.


        Parameters
        ----------
        mean : float
            Mean value of the underlying normal distribution
        sigma : float, >0.
            Standard deviation of the underlying normal distribution
        size : tuple of ints
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        See Also
        --------
        scipy.stats.lognorm : probability density function, distribution,
            cumulative density function, etc.

        Notes
        -----
        A variable `x` has a log-normal distribution if `log(x)` is normally
        distributed.

        The probability density function for the log-normal distribution is

        .. math:: p(x) = \\frac{1}{\\sigma x \\sqrt{2\\pi}}
                         e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}

        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation
        of the normally distributed logarithm of the variable.

        A log normal distribution results if a random variable is the *product* of
        a large number of independent, identically-distributed variables in the
        same way that a normal distribution results if the variable is the *sum*
        of a large number of independent, identically-distributed variables
        (see the last example). It is one of the so-called "fat-tailed"
        distributions.

        The log-normal distribution is commonly used to model the lifespan of units
        with fatigue-stress failure modes. Since this includes
        most mechanical systems, the lognormal distribution has widespread
        application.

        It is also commonly used to model oil field sizes, species abundance, and
        latent periods of infectious diseases.

        References
        ----------
        .. [1] Eckhard Limpert, Werner A. Stahel, and Markus Abbt, "Log-normal
               Distributions across the Sciences: Keys and Clues", May 2001
               Vol. 51 No. 5 BioScience
               http://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
        .. [2] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
               Values, Birkhauser Verlag, Basel, pp 31-32.
        .. [3] Wikipedia, "Lognormal distribution",
               http://en.wikipedia.org/wiki/Lognormal_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 3., 1. # mean and standard deviation
        >>> s = np.random.lognormal(mu, sigma, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 100, normed=True, align='center')

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, linewidth=2, color='r')
        >>> plt.axis('tight')
        >>> plt.show()

        Demonstrate that taking the products of random samples from a uniform
        distribution can be fit well by a log-normal pdf.

        >>> # Generate a thousand samples: each is the product of 100 random
        >>> # values, drawn from a normal distribution.
        >>> b = []
        >>> for i in range(1000):
        ...    a = 10. + np.random.random(100)
        ...    b.append(np.product(a))

        >>> b = np.array(b) / np.min(b) # scale values to be positive

        >>> count, bins, ignored = plt.hist(b, 100, normed=True, align='center')

        >>> sigma = np.std(np.log(b))
        >>> mu = np.mean(np.log(b))

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, color='r', linewidth=2)
        >>> plt.show()

        """
        cdef ndarray omean, osigma
        cdef double fmean, fsigma

        fmean = PyFloat_AsDouble(mean)
        fsigma = PyFloat_AsDouble(sigma)

        if not PyErr_Occurred():
            if fsigma <= 0:
                raise ValueError("sigma <= 0")
            return cont2_array_sc(self.internal_state, rk_lognormal, size, fmean, fsigma)

        PyErr_Clear()

        omean = PyArray_FROM_OTF(mean, NPY_DOUBLE, NPY_ALIGNED)
        osigma = PyArray_FROM_OTF(sigma, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(osigma, 0.0)):
            raise ValueError("sigma <= 0.0")
        return cont2_array(self.internal_state, rk_lognormal, size, omean, osigma)

    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        Rayleigh distribution.

        """
        cdef ndarray oscale
        cdef double fscale

        fscale = PyFloat_AsDouble(scale)

        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont1_array_sc(self.internal_state, rk_rayleigh, size, fscale)

        PyErr_Clear()

        oscale = <ndarray>PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0.0")
        return cont1_array(self.internal_state, rk_rayleigh, size, oscale)

    def wald(self, mean, scale, size=None):
        """
        wald(mean, scale, size=None)

        Wald (inverse Gaussian) distribution.

        """
        cdef ndarray omean, oscale
        cdef double fmean, fscale

        fmean = PyFloat_AsDouble(mean)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fmean <= 0:
                raise ValueError("mean <= 0")
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return cont2_array_sc(self.internal_state, rk_wald, size, fmean, fscale)

        PyErr_Clear()
        omean = PyArray_FROM_OTF(mean, NPY_DOUBLE, NPY_ALIGNED)
        oscale = PyArray_FROM_OTF(scale, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(omean,0.0)):
            raise ValueError("mean <= 0.0")
        elif np.any(np.less_equal(oscale,0.0)):
            raise ValueError("scale <= 0.0")
        return cont2_array(self.internal_state, rk_wald, size, omean, oscale)



    def triangular(self, left, mode, right, size=None):
        """
        triangular(left, mode, right, size=None)

        Triangular distribution starting at left, peaking at mode, and
        ending at right (left <= mode <= right).

        """
        cdef ndarray oleft, omode, oright
        cdef double fleft, fmode, fright

        fleft = PyFloat_AsDouble(left)
        fright = PyFloat_AsDouble(right)
        fmode = PyFloat_AsDouble(mode)
        if not PyErr_Occurred():
            if fleft > fmode:
                raise ValueError("left > mode")
            if fmode > fright:
                raise ValueError("mode > right")
            if fleft == fright:
                raise ValueError("left == right")
            return cont3_array_sc(self.internal_state, rk_triangular, size, fleft,
                                  fmode, fright)

        PyErr_Clear()
        oleft = <ndarray>PyArray_FROM_OTF(left, NPY_DOUBLE, NPY_ALIGNED)
        omode = <ndarray>PyArray_FROM_OTF(mode, NPY_DOUBLE, NPY_ALIGNED)
        oright = <ndarray>PyArray_FROM_OTF(right, NPY_DOUBLE, NPY_ALIGNED)

        if np.any(np.greater(oleft, omode)):
            raise ValueError("left > mode")
        if np.any(np.greater(omode, oright)):
            raise ValueError("mode > right")
        if np.any(np.equal(oleft, oright)):
            raise ValueError("left == right")
        return cont3_array(self.internal_state, rk_triangular, size, oleft,
            omode, oright)

    # Complicated, discrete distributions:
    def binomial(self, n, p, size=None):
        """
        binomial(n, p, size=None)

        Binomial distribution of n trials and p probability of success.

        """
        cdef ndarray on, op
        cdef long ln
        cdef double fp

        fp = PyFloat_AsDouble(p)
        ln = PyInt_AsLong(n)
        if not PyErr_Occurred():
            if ln <= 0:
                raise ValueError("n <= 0")
            if fp < 0:
                raise ValueError("p < 0")
            elif fp > 1:
                raise ValueError("p > 1")
            return discnp_array_sc(self.internal_state, rk_binomial, size, ln, fp)

        PyErr_Clear()

        on = <ndarray>PyArray_FROM_OTF(n, NPY_LONG, NPY_ALIGNED)
        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(n, 0)):
            raise ValueError("n <= 0")
        if np.any(np.less(p, 0)):
            raise ValueError("p < 0")
        if np.any(np.greater(p, 1)):
            raise ValueError("p > 1")
        return discnp_array(self.internal_state, rk_binomial, size, on, op)

    def negative_binomial(self, n, p, size=None):
        """
        negative_binomial(n, p, size=None)

        Negative Binomial distribution.

        """
        cdef ndarray on
        cdef ndarray op
        cdef double fn
        cdef double fp

        fp = PyFloat_AsDouble(p)
        fn = PyFloat_AsDouble(n)
        if not PyErr_Occurred():
            if fn <= 0:
                raise ValueError("n <= 0")
            if fp < 0:
                raise ValueError("p < 0")
            elif fp > 1:
                raise ValueError("p > 1")
            return discdd_array_sc(self.internal_state, rk_negative_binomial,
                                   size, fn, fp)

        PyErr_Clear()

        on = <ndarray>PyArray_FROM_OTF(n, NPY_DOUBLE, NPY_ALIGNED)
        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(n, 0)):
            raise ValueError("n <= 0")
        if np.any(np.less(p, 0)):
            raise ValueError("p < 0")
        if np.any(np.greater(p, 1)):
            raise ValueError("p > 1")
        return discdd_array(self.internal_state, rk_negative_binomial, size,
                            on, op)

    def poisson(self, lam=1.0, size=None):
        """
        poisson(lam=1.0, size=None)

        Poisson distribution.

        """
        cdef ndarray olam
        cdef double flam
        flam = PyFloat_AsDouble(lam)
        if not PyErr_Occurred():
            if lam < 0:
                raise ValueError("lam < 0")
            return discd_array_sc(self.internal_state, rk_poisson, size, flam)

        PyErr_Clear()

        olam = <ndarray>PyArray_FROM_OTF(lam, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less(olam, 0)):
            raise ValueError("lam < 0")
        return discd_array(self.internal_state, rk_poisson, size, olam)

    def zipf(self, a, size=None):
        """
        zipf(a, size=None)

        Draw samples from a Zipf distribution.

        Samples are drawn from a Zipf distribution with specified parameter (a),
        where a > 1.

        The zipf distribution (also known as the zeta
        distribution) is a continuous probability distribution that satisfies
        Zipf's law, where the frequency of an item is inversely proportional to
        its rank in a frequency table.

        Parameters
        ----------
        a : float
            parameter, > 1.
        size : {tuple, int}
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.

        Returns
        -------
        samples : {ndarray, scalar}
            The returned samples are greater than or equal to one.

        See Also
        --------
        scipy.stats.distributions.zipf : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Zipf distribution is

        .. math:: p(x) = \\frac{x^{-a}}{\\zeta(a)},

        where :math:`\\zeta` is the Riemann Zeta function.

        Named after the American linguist George Kingsley Zipf, who noted that
        the frequency of any word in a sample of a language is inversely
        proportional to its rank in the frequency table.


        References
        ----------
        .. [1] Weisstein, Eric W. "Zipf Distribution." From MathWorld--A Wolfram
               Web Resource. http://mathworld.wolfram.com/ZipfDistribution.html
        .. [2] Wikipedia, "Zeta distribution",
               http://en.wikipedia.org/wiki/Zeta_distribution
        .. [3] Wikipedia, "Zipf's Law",
               http://en.wikipedia.org/wiki/Zipf%27s_law
        .. [4] Zipf, George Kingsley (1932): Selected Studies of the Principle
               of Relative Frequency in Language. Cambridge (Mass.).

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 2. # parameter
        >>> s = np.random.zipf(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        Truncate s values at 50 so plot is interesting
        >>> count, bins, ignored = plt.hist(s[s<50], 50, normed=True)
        >>> x = arange(1., 50.)
        >>> y = x**(-a)/sps.zetac(a)
        >>> plt.plot(x, y/max(y), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 1.0:
                raise ValueError("a <= 1.0")
            return discd_array_sc(self.internal_state, rk_zipf, size, fa)

        PyErr_Clear()

        oa = <ndarray>PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less_equal(oa, 1.0)):
            raise ValueError("a <= 1.0")
        return discd_array(self.internal_state, rk_zipf, size, oa)

    def geometric(self, p, size=None):
        """
        geometric(p, size=None)

        Draw samples from the geometric distribution.

        Bernoulli trials are experiments with one of two outcomes:
        success or failure (an example of such an experiment is flipping
        a coin).  The geometric distribution models the number of trials
        that must be run in order to achieve success.  It is therefore
        supported on the positive integers, ``k = 1, 2, ...``.

        The probability mass function of the geometric distribution is

        .. math:: f(k) = (1 - p)^{k - 1} p

        where `p` is the probability of success of an individual trial.

        Parameters
        ----------
        p : float
            The probability of success of an individual trial.
        size : tuple of ints
            Number of values to draw from the distribution.  The output
            is shaped according to `size`.

        Returns
        -------
        out : ndarray
            Samples from the geometric distribution, shaped according to
            `size`.

        Examples
        --------
        Draw ten thousand values from the geometric distribution,
        with the probability of an individual success equal to 0.35:

        >>> z = np.random.geometric(p=0.35, size=10000)

        How many trials succeeded after a single run?

        >>> (z == 1).sum() / 10000.
        0.34889999999999999 #random

        """
        cdef ndarray op
        cdef double fp

        fp = PyFloat_AsDouble(p)
        if not PyErr_Occurred():
            if fp < 0.0:
                raise ValueError("p < 0.0")
            if fp > 1.0:
                raise ValueError("p > 1.0")
            return discd_array_sc(self.internal_state, rk_geometric, size, fp)

        PyErr_Clear()


        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less(op, 0.0)):
            raise ValueError("p < 0.0")
        if np.any(np.greater(op, 1.0)):
            raise ValueError("p > 1.0")
        return discd_array(self.internal_state, rk_geometric, size, op)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """
        hypergeometric(ngood, nbad, nsample, size=None)

        Hypergeometric distribution.

        Consider an urn with ngood "good" balls and nbad "bad" balls. If one
        were to draw nsample balls from the urn without replacement, then
        the hypergeometric distribution describes the distribution of "good"
        balls in the sample.

        """
        cdef ndarray ongood, onbad, onsample
        cdef long lngood, lnbad, lnsample

        lngood = PyInt_AsLong(ngood)
        lnbad = PyInt_AsLong(nbad)
        lnsample = PyInt_AsLong(nsample)
        if not PyErr_Occurred():
            if ngood < 1:
                raise ValueError("ngood < 1")
            if nbad < 1:
                raise ValueError("nbad < 1")
            if nsample < 1:
                raise ValueError("nsample < 1")
            if ngood + nbad < nsample:
                raise ValueError("ngood + nbad < nsample")
            return discnmN_array_sc(self.internal_state, rk_hypergeometric, size,
                                    lngood, lnbad, lnsample)


        PyErr_Clear()

        ongood = <ndarray>PyArray_FROM_OTF(ngood, NPY_LONG, NPY_ALIGNED)
        onbad = <ndarray>PyArray_FROM_OTF(nbad, NPY_LONG, NPY_ALIGNED)
        onsample = <ndarray>PyArray_FROM_OTF(nsample, NPY_LONG, NPY_ALIGNED)
        if np.any(np.less(ongood, 1)):
            raise ValueError("ngood < 1")
        if np.any(np.less(onbad, 1)):
            raise ValueError("nbad < 1")
        if np.any(np.less(onsample, 1)):
            raise ValueError("nsample < 1")
        if np.any(np.less(np.add(ongood, onbad),onsample)):
            raise ValueError("ngood + nbad < nsample")
        return discnmN_array(self.internal_state, rk_hypergeometric, size,
            ongood, onbad, onsample)

    def logseries(self, p, size=None):
        """
        logseries(p, size=None)

        Logarithmic series distribution.

        """
        cdef ndarray op
        cdef double fp

        fp = PyFloat_AsDouble(p)
        if not PyErr_Occurred():
            if fp < 0.0:
                raise ValueError("p < 0.0")
            if fp > 1.0:
                raise ValueError("p > 1.0")
            return discd_array_sc(self.internal_state, rk_logseries, size, fp)

        PyErr_Clear()

        op = <ndarray>PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ALIGNED)
        if np.any(np.less(op, 0.0)):
            raise ValueError("p < 0.0")
        if np.any(np.greater(op, 1.0)):
            raise ValueError("p > 1.0")
        return discd_array(self.internal_state, rk_logseries, size, op)

    # Multivariate distributions:
    def multivariate_normal(self, mean, cov, size=None):
        """
        multivariate_normal(mean, cov[, size])

        Draw random samples from a multivariate normal distribution.

        The multivariate normal, multinormal or Gaussian distribution is a
        generalisation of the one-dimensional normal distribution to higher
        dimensions.

        Such a distribution is specified by its mean and covariance matrix,
        which are analogous to the mean (average or "centre") and variance
        (standard deviation squared or "width") of the one-dimensional normal
        distribution.

        Parameters
        ----------
        mean : (N,) ndarray
            Mean of the N-dimensional distribution.
        cov : (N,N) ndarray
            Covariance matrix of the distribution.
        size : tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.  Because each
            sample is N-dimensional, the output shape is (m,n,k,N).  If no
            shape is specified, a single sample is returned.

        Returns
        -------
        out : ndarray
            The drawn samples, arranged according to `size`.  If the
            shape given is (m,n,...), then the shape of `out` is is
            (m,n,...,N).

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Notes
        -----
        The mean is a coordinate in N-dimensional space, which represents the
        location where samples are most likely to be generated.  This is
        analogous to the peak of the bell curve for the one-dimensional or
        univariate normal distribution.

        Covariance indicates the level to which two variables vary together.
        From the multivariate normal distribution, we draw N-dimensional
        samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
        element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
        The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
        "spread").

        Instead of specifying the full covariance matrix, popular
        approximations include:

          - Spherical covariance (`cov` is a multiple of the identity matrix)
          - Diagonal covariance (`cov` has non-negative elements, and only on
            the diagonal)

        This geometrical property can be seen in two dimensions by plotting
        generated data-points:

        >>> mean = [0,0]
        >>> cov = [[1,0],[0,100]] # diagonal covariance, points lie on x or y-axis

        >>> import matplotlib.pyplot as plt
        >>> x,y = np.random.multivariate_normal(mean,cov,5000).T
        >>> plt.plot(x,y,'x'); plt.axis('equal'); plt.show()

        Note that the covariance matrix must be non-negative definite.

        References
        ----------
        .. [1] A. Papoulis, "Probability, Random Variables, and Stochastic
               Processes," 3rd ed., McGraw-Hill Companies, 1991
        .. [2] R.O. Duda, P.E. Hart, and D.G. Stork, "Pattern Classification,"
               2nd ed., Wiley, 2001.

        Examples
        --------
        >>> mean = (1,2)
        >>> cov = [[1,0],[1,0]]
        >>> x = np.random.multivariate_normal(mean,cov,(3,3))
        >>> x.shape
        (3, 3, 2)

        The following is probably true, given that 0.6 is roughly twice the
        standard deviation:

        >>> print list( (x[0,0,:] - mean) < 0.6 )
        [True, True]

        """
        # Check preconditions on arguments
        mean = np.array(mean)
        cov = np.array(cov)
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
        x = self.standard_normal(np.multiply.reduce(final_shape))
        x.shape = (np.multiply.reduce(final_shape[0:len(final_shape)-1]),
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
        x = np.dot(x*np.sqrt(s),v)
        # The rows of x now have the correct covariance but mean 0. Add
        # mean to each row. Then each row will have mean mean.
        np.add(mean,x,x)
        x.shape = tuple(final_shape)
        return x

    def multinomial(self, long n, object pvals, size=None):
        """
        multinomial(n, pvals, size=None)

        Draw samples from a multinomial distribution.

        The multinomial distribution is a multivariate generalisation of the
        binomial distribution.  Take an experiment with one of ``p``
        possible outcomes.  An example of such an experiment is throwing a dice,
        where the outcome can be 1 through 6.  Each sample drawn from the
        distribution represents `n` such experiments.  Its values,
        ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the outcome
        was ``i``.

        Parameters
        ----------
        n : int
            Number of experiments.
        pvals : sequence of floats, length p
            Probabilities of each of the ``p`` different outcomes.  These
            should sum to 1 (however, the last element is always assumed to
            account for the remaining probability, as long as
            ``sum(pvals[:-1]) <= 1)``.
        size : tuple of ints
            Given a `size` of ``(M, N, K)``, then ``M*N*K`` samples are drawn,
            and the output shape becomes ``(M, N, K, p)``, since each sample
            has shape ``(p,)``.

        Examples
        --------
        Throw a dice 20 times:

        >>> np.random.multinomial(20, [1/6.]*6, size=1)
        array([[4, 1, 7, 5, 2, 1]])

        It landed 4 times on 1, once on 2, etc.

        Now, throw the dice 20 times, and 20 times again:

        >>> np.random.multinomial(20, [1/6.]*6, size=2)
        array([[3, 4, 3, 3, 4, 3],
               [2, 4, 3, 4, 0, 7]])

        For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
        we threw 2 times 1, 4 times 2, etc.

        A loaded dice is more likely to land on number 6:

        >>> np.random.multinomial(100, [1/7.]*5)
        array([13, 16, 13, 16, 42])

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

        if kahan_sum(pix, d-1) > (1.0 + 1e-12):
            raise ValueError("sum(pvals[:-1]) > 1.0")

        if size is None:
            shape = (d,)
        elif type(size) is int:
            shape = (size, d)
        else:
            shape = size + (d,)

        multin = np.zeros(shape, int)
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

    def dirichlet(self, object alpha, size=None):
        """
        dirichlet(alpha, size=None)

        Draw samples from the Dirichlet distribution.

        Draw `size` samples of dimension k from a Dirichlet distribution. A
        Dirichlet-distributed random variable can be seen as a multivariate
        generalization of a Beta distribution. Dirichlet pdf is the conjugate
        prior of a multinomial in Bayesian inference.

        Parameters
        ----------
        alpha : array
            Parameter of the distribution (k dimension for sample of
            dimension k).
        size : array
            Number of samples to draw.

        Notes
        -----
        .. math:: X \\approx \\prod_{i=1}^{k}{x^{\\alpha_i-1}_i}

        Uses the following property for computation: for each dimension,
        draw a random sample y_i from a standard gamma generator of shape
        `alpha_i`, then
        :math:`X = \\frac{1}{\\sum_{i=1}^k{y_i}} (y_1, \\ldots, y_n)` is
        Dirichlet distributed.

        References
        ----------
        .. [1] David McKay, "Information Theory, Inference and Learning
               Algorithms," chapter 23,
               http://www.inference.phy.cam.ac.uk/mackay/

        """

        #=================
        # Pure python algo
        #=================
        #alpha   = N.atleast_1d(alpha)
        #k       = alpha.size

        #if n == 1:
        #    val = N.zeros(k)
        #    for i in range(k):
        #        val[i]   = sgamma(alpha[i], n)
        #    val /= N.sum(val)
        #else:
        #    val = N.zeros((k, n))
        #    for i in range(k):
        #        val[i]   = sgamma(alpha[i], n)
        #    val /= N.sum(val, axis = 0)
        #    val = val.T

        #return val

        cdef long       k
        cdef long       totsize
        cdef ndarray    alpha_arr, val_arr
        cdef double     *alpha_data, *val_data
        cdef long       i, j
        cdef double     acc, invacc

        k           = len(alpha)
        alpha_arr   = <ndarray>PyArray_ContiguousFromObject(alpha, NPY_DOUBLE, 1, 1)
        alpha_data  = <double*>alpha_arr.data

        if size is None:
            shape = (k,)
        elif type(size) is int:
            shape = (size, k)
        else:
            shape = size + (k,)

        diric   = np.zeros(shape, np.float64)
        val_arr = <ndarray>diric
        val_data= <double*>val_arr.data

        i = 0
        totsize = PyArray_SIZE(val_arr)
        while i < totsize:
            acc = 0.0
            for j from 0 <= j < k:
                val_data[i+j]   = rk_standard_gamma(self.internal_state, alpha_data[j])
                acc             = acc + val_data[i+j]
            invacc  = 1/acc
            for j from 0 <= j < k:
                val_data[i+j]   = val_data[i+j] * invacc
            i = i + k

        return diric

    # Shuffling and permutations:
    def shuffle(self, object x):
        """
        shuffle(x)

        Modify a sequence in-place by shuffling its contents.

        """
        cdef long i, j
        cdef int copy

        i = len(x) - 1
        try:
            j = len(x[0])
        except:
            j = 0

        if (j == 0):
            # adaptation of random.shuffle()
            while i > 0:
                j = rk_interval(i, self.internal_state)
                x[i], x[j] = x[j], x[i]
                i = i - 1
        else:
            # make copies
            copy = hasattr(x[0], 'copy')
            if copy:
                while(i > 0):
                    j = rk_interval(i, self.internal_state)
                    x[i], x[j] = x[j].copy(), x[i].copy()
                    i = i - 1
            else:
                while(i > 0):
                    j = rk_interval(i, self.internal_state)
                    x[i], x[j] = x[j][:], x[i][:]
                    i = i - 1

    def permutation(self, object x):
        """
        permutation(x)

        Randomly permute a sequence, or return a permuted range.

        Parameters
        ----------
        x : int or array_like
            If `x` is an integer, randomly permute ``np.arange(x)``.
            If `x` is an array, make a copy and shuffle the elements
            randomly.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        Examples
        --------
        >>> np.random.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])

        >>> np.random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12])

        """
        if isinstance(x, (int, np.integer)):
            arr = np.arange(x)
        else:
            arr = np.array(x)
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
dirichlet = _rand.dirichlet

shuffle = _rand.shuffle
permutation = _rand.permutation
