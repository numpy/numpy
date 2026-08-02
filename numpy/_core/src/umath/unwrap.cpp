/**
 * This module provides the inner loops for the unwrap ufunc
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <type_traits>

#include "numpy/halffloat.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"
#include "array_method.h"
#include "dispatching.h"
#include "dtypemeta.h"

#include "unwrap.h"


/*
 * `unwrap` is a scan (each output depends on the running phase correction of
 * the whole prefix along the core axis), so it cannot be a plain element-wise
 * ufunc.  Each loop performs the diff / modular-wrap / cumulative-correction in
 * a single pass over the core dimension, without allocating any temporaries.
 *
 * The operands are, in order, the values `p` (core `(n)`), the scalars
 * `discont` and `period`, and the output (core `(n)`).  The Python wrapper in
 * numpy/lib/_function_base_impl.py fills the `discont`/`period` defaults,
 * resolves the common dtype and casts the scalars to it.
 *
 * gufunc strided-loop layout for signature (n),(),()->(n):
 *   dimensions[0]      outer (broadcast) loop count
 *   dimensions[1]      core length n
 *   strides[0..3]      outer strides for p, discont, period, out
 *   strides[4]         core stride of p
 *   strides[5]         core stride of out
 */

/* floor-modulo matching numpy's `mod`, same as @TYPE@_remainder in loops.c.src */
static inline npy_float
floor_mod(npy_float a, npy_float b)
{
    return npy_remainderf(a, b);
}
static inline npy_double
floor_mod(npy_double a, npy_double b)
{
    return npy_remainder(a, b);
}
#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE
static inline npy_longdouble
floor_mod(npy_longdouble a, npy_longdouble b)
{
    return npy_remainderl(a, b);
}
#endif

/* no integer npy_remainder to call, so mirror @TYPE@_remainder in
 * loops_modulo.dispatch.c.src, including the T_MIN / -1 guard. that guard
 * doesn't set a floating point status there either, since only the
 * quotient overflows and this only ever computes the remainder */
template <typename T>
static inline std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, T>
floor_mod(T a, T b)
{
    if (b == 0) {
        npy_set_floatstatus_divbyzero();
        return 0;
    }
    if (b == -1) {
        return 0;
    }
    T rem = a % b;
    if ((a > 0) == (b > 0) || rem == 0) {
        return rem;
    }
    return rem + b;
}

/*
 * One loop per real dtype. T is the storage/compute type, and discont is
 * read as T itself for the float dtypes, npy_double for the integer ones.
 * npy_half has no native arithmetic, so it gets its own loop below.
 */
template <typename T>
static int
unwrap_loop(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
            npy_intp const dimensions[], npy_intp const strides[],
            NpyAuxData *NPY_UNUSED(auxdata))
{
    using D = std::conditional_t<std::is_floating_point_v<T>, T, npy_double>;

    npy_intp n_outer = dimensions[0];
    npy_intp n = dimensions[1];
    if (n == 0) {
        return 0;
    }
    npy_intp s_p = strides[0], s_disc = strides[1];
    npy_intp s_per = strides[2], s_out = strides[3];
    npy_intp ip_step = strides[4];
    npy_intp op_step = strides[5];

    char *p_o = data[0], *disc_o = data[1], *per_o = data[2], *out_o = data[3];

    for (npy_intp k = 0; k < n_outer; k++,
            p_o += s_p, disc_o += s_disc, per_o += s_per, out_o += s_out) {
        D discont = *(const D *)disc_o;
        T period = *(const T *)per_o;
        T interval_high;
        bool boundary_ambiguous;
        if constexpr (std::is_integral_v<T>) {
            /* floor((period)/2), matching python's `divmod(period, 2)` */
            T rem = floor_mod(period, (T)2);
            interval_high = (period - rem) / 2;
            boundary_ambiguous = (rem == 0);
        }
        else {
            interval_high = period / 2;
            boundary_ambiguous = true;
        }
        T interval_low = -interval_high;

        const char *ip = p_o;
        char *op = out_o;
        T prev = *(const T *)ip;
        *(T *)op = prev;
        T cum = 0;
        for (npy_intp i = 1; i < n; i++) {
            ip += ip_step;
            op += op_step;
            T cur = *(const T *)ip;
            T dd = cur - prev;
            /* wrap the raw diff into [interval_low, interval_low + period) */
            T ddmod = (T)(floor_mod((T)(dd - interval_low), period) + interval_low);
            if (boundary_ambiguous && ddmod == interval_low && dd > 0) {
                ddmod = interval_high;
            }
            T corr = ddmod - dd;
            D adiff = (D)(dd < 0 ? (T)-dd : dd);
            if (adiff < discont) {
                corr = 0;
            }
            cum += corr;
            *(T *)op = (T)(cur + cum);
            prev = cur;
        }
    }
    return 0;
}

/* rounds to half after every step, like numpy's own HALF_remainder loop.
 * rounding only at the end instead would change the result */
static inline float h2f(npy_half h) { return npy_half_to_float(h); }
static inline npy_half f2h(float f) { return npy_float_to_half(f); }

static int
unwrap_half_loop(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp n_outer = dimensions[0];
    npy_intp n = dimensions[1];
    if (n == 0) {
        return 0;
    }
    npy_intp s_p = strides[0], s_disc = strides[1];
    npy_intp s_per = strides[2], s_out = strides[3];
    npy_intp ip_step = strides[4];
    npy_intp op_step = strides[5];

    char *p_o = data[0], *disc_o = data[1], *per_o = data[2], *out_o = data[3];

    for (npy_intp k = 0; k < n_outer; k++,
            p_o += s_p, disc_o += s_disc, per_o += s_per, out_o += s_out) {
        double discont = h2f(*(const npy_half *)disc_o);
        npy_half period = *(const npy_half *)per_o;
        npy_half interval_high = f2h(h2f(period) / 2.0f);
        npy_half interval_low = f2h(-h2f(interval_high));

        const char *ip = p_o;
        char *op = out_o;
        npy_half prev = *(const npy_half *)ip;
        *(npy_half *)op = prev;
        npy_half cum = f2h(0.0f);
        for (npy_intp i = 1; i < n; i++) {
            ip += ip_step;
            op += op_step;
            npy_half cur = *(const npy_half *)ip;
            npy_half dd = f2h(h2f(cur) - h2f(prev));
            npy_half t1 = f2h(h2f(dd) - h2f(interval_low));
            npy_half t2 = f2h(floor_mod(h2f(t1), h2f(period)));
            npy_half ddmod = f2h(h2f(t2) + h2f(interval_low));
            if (ddmod == interval_low && h2f(dd) > 0) {
                ddmod = interval_high;
            }
            npy_half corr = f2h(h2f(ddmod) - h2f(dd));
            double adiff = h2f(dd) < 0 ? -(double)h2f(dd) : (double)h2f(dd);
            if (adiff < discont) {
                corr = f2h(0.0f);
            }
            cum = f2h(h2f(cum) + h2f(corr));
            *(npy_half *)op = f2h(h2f(cur) + h2f(cum));
            prev = cur;
        }
    }
    return 0;
}


static int
add_unwrap_loop(PyObject *ufunc, int typenum, PyArrayMethod_StridedLoop *loop)
{
    PyArray_DTypeMeta *dt = PyArray_DTypeFromTypeNum(typenum);
    if (dt == NULL) {
        return -1;
    }
    /* discont is read at typenum's own precision for float dtypes and at
     * double for the integer ones, matching unwrap_loop's own D derivation */
    int discont_typenum = PyTypeNum_ISFLOAT(typenum) ? typenum : NPY_DOUBLE;
    PyArray_DTypeMeta *disc_dt = PyArray_DTypeFromTypeNum(discont_typenum);
    if (disc_dt == NULL) {
        Py_DECREF(dt);
        return -1;
    }
    /* p, out and period share the loop dtype, discont has its own */
    PyArray_DTypeMeta *dtypes[4] = {dt, disc_dt, dt, dt};
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, (void *)loop},
        {0, NULL}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "unwrap_loop";
    spec.nin = 3;
    spec.nout = 1;
    spec.casting = NPY_NO_CASTING;
    spec.flags = (NPY_ARRAYMETHOD_FLAGS)0;
    spec.dtypes = dtypes;
    spec.slots = slots;

    int ret = PyUFunc_AddLoopFromSpec_int(ufunc, &spec, 1);
    Py_DECREF(dt);
    Py_DECREF(disc_dt);
    return ret;
}

NPY_NO_EXPORT int
init_unwrap_ufunc(PyObject *umath)
{
    PyObject *name = PyUnicode_FromString("_unwrap");
    if (name == NULL) {
        return -1;
    }
    PyObject *ufunc = PyObject_GetItem(umath, name);
    Py_DECREF(name);
    if (ufunc == NULL) {
        return -1;
    }

    struct Loop {
        int typenum;
        PyArrayMethod_StridedLoop *loop;
    };

    Loop loops[] = {
        {NPY_HALF, unwrap_half_loop},
        {NPY_FLOAT, unwrap_loop<npy_float>},
        {NPY_DOUBLE, unwrap_loop<npy_double>},
        {NPY_LONGDOUBLE, unwrap_loop<npy_longdouble>},
        {NPY_BYTE, unwrap_loop<npy_byte>},
        {NPY_SHORT, unwrap_loop<npy_short>},
        {NPY_INT, unwrap_loop<npy_int>},
        {NPY_LONG, unwrap_loop<npy_long>},
        {NPY_LONGLONG, unwrap_loop<npy_longlong>},
        /* object dtype falls back to the python implementation instead */
        // {NPY_OBJECT, object_unwrap_loop},
    };

    for (const auto& entry : loops) {
        if (add_unwrap_loop(ufunc, entry.typenum, entry.loop) < 0) {
            Py_DECREF(ufunc);
            return -1;
        }
    }

    Py_DECREF(ufunc);
    return 0;
}
