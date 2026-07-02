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
 * resolves the common dtype and casts the scalars to it (rounding `discont` up
 * for integer dtypes so that the `|dd| < discont` threshold keeps its meaning).
 *
 * gufunc strided-loop layout for signature (n),(),()->(n):
 *   dimensions[0]      outer (broadcast) loop count
 *   dimensions[1]      core length n
 *   strides[0..3]      outer strides for p, discont, period, out
 *   strides[4]         core stride of p
 *   strides[5]         core stride of out
 */

/*
 * floor-modulo matching numpy's `mod`, picked by overload on the npy floating
 * types.  The long double overload is only defined when it is a distinct type
 * from double (otherwise the double overload already covers npy_longdouble).
 */
static inline npy_float
floor_mod(npy_float a, npy_float b)
{
    npy_float m;
    npy_divmodf(a, b, &m);
    return m;
}
static inline npy_double
floor_mod(npy_double a, npy_double b)
{
    npy_double m;
    npy_divmod(a, b, &m);
    return m;
}
#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE
static inline npy_longdouble
floor_mod(npy_longdouble a, npy_longdouble b)
{
    npy_longdouble m;
    npy_divmodl(a, b, &m);
    return m;
}
#endif

/*
 * One strided loop per real dtype.  `T` is both the storage and the compute
 * type; the integer vs. floating-point parts of the algorithm (floor division
 * for the interval, the even-period boundary, integer floor-modulo) select with
 * `if constexpr` on `T`.  `discont` is only a threshold (never part of the wrap
 * arithmetic), so it is always a plain double operand -- it keeps its value
 * exactly and never drags the result dtype (e.g. integer input stays integer).
 *
 * `npy_half` has no native arithmetic, so it gets its own loop below that
 * computes in float; every other real dtype uses this template.
 */
template <typename T>
static int
unwrap_loop(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
            npy_intp const dimensions[], npy_intp const strides[],
            NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp n_outer = dimensions[0];
    npy_intp n = dimensions[1];
    npy_intp s_p = strides[0], s_disc = strides[1];
    npy_intp s_per = strides[2], s_out = strides[3];
    npy_intp ip_step = strides[4];
    npy_intp op_step = strides[5];

    char *p_o = data[0], *disc_o = data[1], *per_o = data[2], *out_o = data[3];

    for (npy_intp k = 0; k < n_outer; k++,
            p_o += s_p, disc_o += s_disc, per_o += s_per, out_o += s_out) {
        if (n <= 0) {
            continue;
        }
        npy_double discont = *(const npy_double *)disc_o;
        T period = *(const T *)per_o;
        /* floor division for ints, exact half for floats */
        T interval_high = period / 2;
        T interval_low = -interval_high;
        bool boundary_ambiguous;
        if constexpr (std::is_integral_v<T>) {
            /* only an even period has a representable +/-period/2 boundary */
            boundary_ambiguous = (period % 2) == 0;
        }
        else {
            boundary_ambiguous = true;
        }

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
            T ddmod;
            if constexpr (std::is_integral_v<T>) {
                /* floor-modulo into [0, period); period > 0 so a single fixup */
                T m = (T)((dd - interval_low) % period);
                if (m < 0) {
                    m += period;
                }
                ddmod = m + interval_low;
            }
            else {
                ddmod = floor_mod(dd - interval_low, period) + interval_low;
            }
            if (boundary_ambiguous && ddmod == interval_low && dd > 0) {
                ddmod = interval_high;
            }
            T corr = ddmod - dd;
            T adiff = dd < 0 ? (T)-dd : dd;
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

/* Half is stored as npy_half but has no native arithmetic, so it loads/stores
 * npy_half and does the whole scan in float. */
static int
unwrap_half_loop(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp n_outer = dimensions[0];
    npy_intp n = dimensions[1];
    npy_intp s_p = strides[0], s_disc = strides[1];
    npy_intp s_per = strides[2], s_out = strides[3];
    npy_intp ip_step = strides[4];
    npy_intp op_step = strides[5];

    char *p_o = data[0], *disc_o = data[1], *per_o = data[2], *out_o = data[3];

    for (npy_intp k = 0; k < n_outer; k++,
            p_o += s_p, disc_o += s_disc, per_o += s_per, out_o += s_out) {
        if (n <= 0) {
            continue;
        }
        npy_double discont = *(const npy_double *)disc_o;
        float period = npy_half_to_float(*(const npy_half *)per_o);
        float interval_high = period / 2;
        float interval_low = -interval_high;

        const char *ip = p_o;
        char *op = out_o;
        float prev = npy_half_to_float(*(const npy_half *)ip);
        *(npy_half *)op = npy_float_to_half(prev);
        float cum = 0;
        for (npy_intp i = 1; i < n; i++) {
            ip += ip_step;
            op += op_step;
            float cur = npy_half_to_float(*(const npy_half *)ip);
            float dd = cur - prev;
            float ddmod = floor_mod(dd - interval_low, period) + interval_low;
            if (ddmod == interval_low && dd > 0) {
                ddmod = interval_high;
            }
            float corr = ddmod - dd;
            float adiff = dd < 0 ? -dd : dd;
            if (adiff < discont) {
                corr = 0;
            }
            cum += corr;
            *(npy_half *)op = npy_float_to_half(cur + cum);
            prev = cur;
        }
    }
    return 0;
}


static int
add_unwrap_loop(PyObject *ufunc, int typenum,
                PyArrayMethod_StridedLoop *loop)
{
    PyArray_DTypeMeta *dt = PyArray_DTypeFromTypeNum(typenum);
    if (dt == NULL) {
        return -1;
    }
    PyArray_DTypeMeta *dbl = PyArray_DTypeFromTypeNum(NPY_DOUBLE);
    if (dbl == NULL) {
        Py_DECREF(dt);
        return -1;
    }
    /* p, out and period share the loop dtype; discont is always double. */
    PyArray_DTypeMeta *dtypes[4] = {dt, dbl, dt, dt};
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

    if (PyUFunc_AddLoopFromSpec_int(ufunc, &spec, 1)) {
        Py_DECREF(dt);
        Py_DECREF(dbl);
        return -1;
    }
    Py_DECREF(dbl);

    Py_DECREF(dt);
    return 0;
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
        // {NPY_OBJECT, object_unwrap_loop},
    };

    int res = 0;
    for (const auto& entry : loops) {
        if (add_unwrap_loop(ufunc, entry.typenum, entry.loop) < 0) {
            res = -1;
            break;
        }
    }

    Py_DECREF(ufunc);
    return res;
}
