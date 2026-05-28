#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
extern "C" {
#include "npy_argparse.h"
}
#include "common.h"

#include "histogram.h"

template <typename T>
static inline void
accum(T &out, const T *w, npy_intp i)
{
    if (w != nullptr)
        out += w[i];
    else
        out++;
}

static inline void
accum(npy_cfloat &out, const npy_cfloat *w, npy_intp i)
{
    npy_csetrealf(&out, npy_crealf(out) + npy_crealf(w[i]));
    npy_csetimagf(&out, npy_cimagf(out) + npy_cimagf(w[i]));
}

static inline void
accum(npy_cdouble &out, const npy_cdouble *w, npy_intp i)
{
    npy_csetreal(&out, npy_creal(out) + npy_creal(w[i]));
    npy_csetimag(&out, npy_cimag(out) + npy_cimag(w[i]));
}


// Helper function to find in which (indiced) uniform bins the data lies
template <typename FP, typename T>
static void
digitize_uniform(
        const FP *a, const T *weights, npy_intp len,
        FP first_edge, FP last_edge, npy_intp n_bins,
        const FP *bin_edges,
        T *out)
{
    // we only calculate the range outside the loop for overflow reasons
    const FP range = last_edge - first_edge;
    for (npy_intp i = 0; i < len; i++)
    {
        const FP v = a[i];
        if (!((v >= first_edge) & (v <= last_edge))) continue;

        npy_intp idx = (npy_intp)((v - first_edge) / range * (FP)n_bins);
        // If the value lies on the last edge, substract one
        if (NPY_UNLIKELY(idx >= n_bins)) idx = n_bins - 1;

        if (v < bin_edges[idx]) --idx;
        else if (idx < n_bins - 1 && v >= bin_edges[idx + 1]) ++idx;
        accum(out[idx], weights, i);
    }
}

// Helper function to run digitize_uniform over the data
template <typename FP, typename T>
static PyObject *
make_weighted_digitize(PyArrayObject *a, const FP *bin_edges,
                   PyArrayObject *weights, npy_intp n_bins)
{
    const npy_intp len_a = PyArray_SIZE(a);

    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(
            1, &n_bins, PyArray_TYPE(weights), 0);
    if (out == NULL) return NULL;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS_THRESHOLDED(len_a);
    digitize_uniform(
            (const FP *)PyArray_DATA(a),
            (const T *)PyArray_DATA(weights), len_a,
            bin_edges[0], bin_edges[n_bins], n_bins, bin_edges,
            (T *)PyArray_DATA(out));
    NPY_END_THREADS;
    return (PyObject *)out;
}

template <typename FP>
static PyObject *
histogram_uniform_impl(PyArrayObject *a, PyArrayObject *bin_edges_obj,
                       PyArrayObject *weights, npy_intp n_bins)
{
    const FP *bin_edges = (const FP *)PyArray_DATA(bin_edges_obj);

    if (weights == NULL)
    {
        const npy_intp len_a = PyArray_SIZE(a);
        PyArrayObject *out =
                (PyArrayObject *)PyArray_ZEROS(1, &n_bins, NPY_INTP, 0);
        if (out == NULL) return NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(len_a);
        digitize_uniform<FP, npy_intp>(
                (const FP *)PyArray_DATA(a),
                (const npy_intp *)nullptr, len_a,
                bin_edges[0], bin_edges[n_bins], n_bins, bin_edges,
                (npy_intp *)PyArray_DATA(out));
        NPY_END_THREADS;
        return (PyObject *)out;
    }

    switch (PyArray_TYPE(weights))
    {
        case NPY_BOOL:    return make_weighted_digitize<FP, npy_bool>   (a, bin_edges, weights, n_bins);
        case NPY_INT8:    return make_weighted_digitize<FP, npy_int8>   (a, bin_edges, weights, n_bins);
        case NPY_INT16:   return make_weighted_digitize<FP, npy_int16>  (a, bin_edges, weights, n_bins);
        case NPY_INT32:   return make_weighted_digitize<FP, npy_int32>  (a, bin_edges, weights, n_bins);
        case NPY_INT64:   return make_weighted_digitize<FP, npy_int64>  (a, bin_edges, weights, n_bins);
        case NPY_UINT8:   return make_weighted_digitize<FP, npy_uint8>  (a, bin_edges, weights, n_bins);
        case NPY_UINT16:  return make_weighted_digitize<FP, npy_uint16> (a, bin_edges, weights, n_bins);
        case NPY_UINT32:  return make_weighted_digitize<FP, npy_uint32> (a, bin_edges, weights, n_bins);
        case NPY_UINT64:  return make_weighted_digitize<FP, npy_uint64> (a, bin_edges, weights, n_bins);
        case NPY_HALF:    return make_weighted_digitize<FP, npy_half>   (a, bin_edges, weights, n_bins);
        case NPY_FLOAT:   return make_weighted_digitize<FP, npy_float>  (a, bin_edges, weights, n_bins);
        case NPY_DOUBLE:  return make_weighted_digitize<FP, npy_double> (a, bin_edges, weights, n_bins);
        case NPY_CFLOAT:  return make_weighted_digitize<FP, npy_cfloat> (a, bin_edges, weights, n_bins);
        case NPY_CDOUBLE: return make_weighted_digitize<FP, npy_cdouble>(a, bin_edges, weights, n_bins);
        default:
            PyErr_SetString(PyExc_TypeError, "unsupported weights dtype");
            return NULL;
    }
}

NPY_NO_EXPORT PyObject *
arr_histogram_uniform(PyObject *NPY_UNUSED(self), PyObject *const *args,
                      Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *a_obj = NULL;
    PyObject *n_equal_bins_obj = NULL, *bin_edges_obj = NULL;
    PyObject *weights_obj = Py_None;
    PyArrayObject *a = NULL, *bin_edges = NULL, *weights = NULL;
    npy_intp n_bins;
    int use_ld, dtype;
    const int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("_histogram_uniform", args, len_args, kwnames,
                {"a",            NULL, &a_obj},
                {"n_equal_bins", NULL, &n_equal_bins_obj},
                {"bin_edges",    NULL, &bin_edges_obj},
                {"weights",     NULL, &weights_obj}) < 0)
    {
        return NULL;
    }

    use_ld = (PyArray_Check(bin_edges_obj) &&
              PyArray_TYPE((PyArrayObject *)bin_edges_obj) == NPY_LONGDOUBLE);
    dtype = use_ld ? NPY_LONGDOUBLE : NPY_DOUBLE;

    a = (PyArrayObject *)PyArray_FROMANY(a_obj, dtype, 1, 1, flags);
    if (a == NULL) goto fail;

    bin_edges = (PyArrayObject *)PyArray_FROMANY(bin_edges_obj, dtype, 1, 1, flags);
    if (bin_edges == NULL) goto fail;

    if (weights_obj != Py_None)
    {
        weights = (PyArrayObject *)PyArray_FROM_OF(weights_obj, flags);
        if (weights == NULL) goto fail;
    }

    n_bins = PyArray_PyIntAsIntp(n_equal_bins_obj);
    if (error_converting(n_bins)) goto fail;

    if (n_bins <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "n_equal_bins must be at least 1");
        goto fail;
    }
    if (PyArray_SIZE(bin_edges) != n_bins + 1)
    {
        PyErr_SetString(PyExc_ValueError, "length of bin_edges must be n_bins + 1");
        goto fail;
    }

    if (PyArray_SIZE(a) == 0)
    {
        int out_typenum = (weights == NULL) ? NPY_INTP : PyArray_TYPE(weights);
        PyObject *out = PyArray_ZEROS(1, &n_bins, out_typenum, 0);
        Py_DECREF(a);
        Py_DECREF(bin_edges);
        Py_XDECREF(weights);
        return out;
    }

    {
        PyObject *out;
        if (use_ld)
            out = histogram_uniform_impl<npy_longdouble>(a, bin_edges, weights, n_bins);
        else
            out = histogram_uniform_impl<npy_double>(a, bin_edges, weights, n_bins);
        Py_DECREF(a);
        Py_DECREF(bin_edges);
        Py_XDECREF(weights);
        return out;
    }

fail:
    Py_XDECREF(a);
    Py_XDECREF(bin_edges);
    Py_XDECREF(weights);
    return NULL;
}
