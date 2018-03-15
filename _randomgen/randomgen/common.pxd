from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t, intptr_t,
                          uintptr_t)
from libc.math cimport sqrt

from distributions cimport brng_t 
import numpy as np
cimport numpy as np

cdef double POISSON_LAM_MAX
cdef uint64_t MAXSIZE

cdef enum ConstraintType:
    CONS_NONE
    CONS_NON_NEGATIVE
    CONS_POSITIVE
    CONS_BOUNDED_0_1
    CONS_BOUNDED_0_1_NOTNAN
    CONS_GT_1
    CONS_GTE_1
    CONS_POISSON

ctypedef ConstraintType constraint_type

cdef int check_constraint(double val, object name, constraint_type cons) except -1
cdef int check_array_constraint(np.ndarray val, object name, constraint_type cons) except -1

cdef extern from "src/aligned_malloc/aligned_malloc.h":
    cdef void *PyArray_realloc_aligned(void *p, size_t n);
    cdef void *PyArray_malloc_aligned(size_t n);
    cdef void *PyArray_calloc_aligned(size_t n, size_t s);
    cdef void PyArray_free_aligned(void *p);

ctypedef double (*random_double_fill)(brng_t *state, np.npy_intp count, double* out) nogil
ctypedef double (*random_double_0)(brng_t *state) nogil
ctypedef double (*random_double_1)(brng_t *state, double a) nogil
ctypedef double (*random_double_2)(brng_t *state, double a, double b) nogil
ctypedef double (*random_double_3)(brng_t *state, double a, double b, double c) nogil

ctypedef float (*random_float_0)(brng_t *state) nogil
ctypedef float (*random_float_1)(brng_t *state, float a) nogil

ctypedef int64_t (*random_uint_0)(brng_t *state) nogil
ctypedef int64_t (*random_uint_d)(brng_t *state, double a) nogil
ctypedef int64_t (*random_uint_dd)(brng_t *state, double a, double b) nogil
ctypedef int64_t (*random_uint_di)(brng_t *state, double a, uint64_t b) nogil
ctypedef int64_t (*random_uint_i)(brng_t *state, int64_t a) nogil
ctypedef int64_t (*random_uint_iii)(brng_t *state, int64_t a, int64_t b, int64_t c) nogil

ctypedef uint32_t (*random_uint_0_32)(brng_t *state) nogil
ctypedef uint32_t (*random_uint_1_i_32)(brng_t *state, uint32_t a) nogil

ctypedef int32_t (*random_int_2_i_32)(brng_t *state, int32_t a, int32_t b) nogil
ctypedef int64_t (*random_int_2_i)(brng_t *state, int64_t a, int64_t b) nogil

cdef double kahan_sum(double *darr, np.npy_intp n)

cdef inline double uint64_to_double(uint64_t rnd) nogil:
    return (rnd >> 11) * (1.0 / 9007199254740992.0)

cdef object double_fill(void *func, brng_t *state, object size, object lock, object out)

cdef object float_fill(void *func, brng_t *state, object size, object lock, object out)

cdef object float_fill_from_double(void *func, brng_t *state, object size, object lock, object out)

cdef np.ndarray int_to_array(object value, object name, object bits, object uint_size)

cdef object cont(void *func, brng_t *state, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint,
                 object out)

cdef object disc(void *func, brng_t *state, object size, object lock,
                 int narg_double, int narg_int64,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint)

cdef object cont_f(void *func, brng_t *state, object size, object lock,
                   object a, object a_name, constraint_type a_constraint,
                   object out)

cdef object cont_broadcast_3(void *func, brng_t *state, object size, object lock,
                             np.ndarray a_arr, object a_name, constraint_type a_constraint,
                             np.ndarray b_arr, object b_name, constraint_type b_constraint,
                             np.ndarray c_arr, object c_name, constraint_type c_constraint)

cdef object discrete_broadcast_iii(void *func, brng_t *state, object size, object lock,
                                  np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                  np.ndarray b_arr, object b_name, constraint_type b_constraint,
                                  np.ndarray c_arr, object c_name, constraint_type c_constraint)

cdef inline void compute_complex(double *rv_r, double *rv_i, double loc_r,
                                 double loc_i, double var_r, double var_i, double rho) nogil:
    cdef double scale_c, scale_i, scale_r

    scale_c = sqrt(1 - rho * rho)
    scale_r = sqrt(var_r)
    scale_i = sqrt(var_i)

    rv_i[0] = loc_i + scale_i * (rho * rv_r[0]  + scale_c * rv_i[0])
    rv_r[0] = loc_r + scale_r * rv_r[0]
