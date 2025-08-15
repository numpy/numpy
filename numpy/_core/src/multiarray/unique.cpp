#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define HASH_TABLE_INITIAL_BUCKETS 1024
#include <Python.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_set>
#include <vector>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "gil_utils.h"
extern "C" {
    #include "fnv.h"
    #include "npy_argparse.h"
    #include "numpy/npy_math.h"
    #include "numpy/halffloat.h"
}

// This is to use RAII pattern to handle cpp exceptions while avoiding memory leaks.
// Adapted from https://stackoverflow.com/a/25510879/2536294
template <typename F>
struct FinalAction {
    FinalAction(F f) : clean_{f} {}
    ~FinalAction() { clean_(); }
  private:
    F clean_;
};
template <typename F>
FinalAction<F> finally(F f) {
    return FinalAction<F>(f);
}

template <typename T>
inline int equal_default(T lhs, T rhs) {
    return lhs == rhs;
}

// As npy_isnan is a macro, we need to define a wrapper function to use it in a template.
template <typename T>
inline int npy_isnan_wrapper(T value) {
    return npy_isnan(value);
}

template <typename T>
inline size_t hash_nonan(const T *value, npy_bool equal_nan) {
    return std::hash<T>{}(*value);
}

template <
    typename T,
    int (*isnan)(T) = npy_isnan_wrapper<T>
>
size_t hash_maybenan(const T *value, npy_bool equal_nan) {
    if (equal_nan && isnan(*value)) {
        return 0;
    }
    return hash_nonan(value, equal_nan);
}

template <typename S, typename T, S (*real)(T), S (*imag)(T)>
size_t hash_complex(const T *value, npy_bool equal_nan) {
    S value_real = real(*value);
    S value_imag = imag(*value);
    int isnan = npy_isnan_wrapper<S>(value_real) || npy_isnan_wrapper<S>(value_imag);
    if (equal_nan && isnan) {
        return 0;
    }

    return hash_nonan<S>(&value_real, equal_nan) ^ hash_nonan<S>(&value_imag, equal_nan);
}

template <typename T>
inline int equal_nonan(const T *lhs, const T *rhs, npy_bool equal_nan) {
    return equal_default(*lhs, *rhs);
}

template <
    typename T,
    int (*isnan)(T) = npy_isnan_wrapper<T>,
    int (*iseq_nonan)(T, T) = equal_default<T>
>
int equal_maybenan(const T *lhs, const T *rhs, npy_bool equal_nan) {
    int lhs_isnan = isnan(*lhs);
    int rhs_isnan = isnan(*rhs);
    if (lhs_isnan && rhs_isnan) {
        return equal_nan;
    }
    if (lhs_isnan || rhs_isnan) {
        return false;
    }
    return iseq_nonan(*lhs, *rhs);
}

template <typename S, typename T, S (*real)(T), S (*imag)(T)>
int equal_complex(const T *lhs, const T *rhs, npy_bool equal_nan) {
    S lhs_real = real(*lhs);
    S lhs_imag = imag(*lhs);
    int lhs_isnan = npy_isnan(lhs_real) || npy_isnan(lhs_imag);
    S rhs_real = real(*rhs);
    S rhs_imag = imag(*rhs);
    int rhs_isnan = npy_isnan(rhs_real) || npy_isnan(rhs_imag);

    if (lhs_isnan && rhs_isnan) {
        return equal_nan;
    }
    if (lhs_isnan || rhs_isnan) {
        return false;
    }
    return equal_nonan<S>(&lhs_real, &rhs_real, equal_nan) &&
           equal_nonan<S>(&lhs_imag, &rhs_imag, equal_nan);
}

template <typename T>
void copy_real(char *data, T *value) {
    std::copy_n(value, 1, (T *)data);
    return;
}

template <
    typename S,
    typename T,
    S (*real)(T),
    S (*imag)(T),
    void (*setreal)(T *, const S),
    void (*setimag)(T *, const S)
>
void copy_complex(char *data, T *value) {
    setreal((T *)data, real(*value));
    setimag((T *)data, imag(*value));
    return;
}

template <
    typename T,
    size_t (*hash_func)(const T *, npy_bool),
    int (*equal_func)(const T *, const T *, npy_bool),
    void (*copy_func)(char *, T *)
>
static PyObject*
unique_numeric(PyArrayObject *self, npy_bool equal_nan)
{
    /*
    * Returns a new NumPy array containing the unique values of the input array of numeric (integer, float or complex).
    * This function uses hashing to identify uniqueness efficiently.
    */
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
    NPY_DISABLE_C_API;

    PyThreadState *_save1 = PyEval_SaveThread();

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);

    auto hash = [equal_nan](const T *value) -> size_t {
        return hash_func(value, equal_nan);
    };
    auto equal = [equal_nan](const T *lhs, const T *rhs) -> bool {
        return equal_func(lhs, rhs, equal_nan);
    };

    // Reserve hashset capacity in advance to minimize reallocations and collisions.
    // We use min(isize, HASH_TABLE_INITIAL_BUCKETS) as the initial bucket count:
    // - Reserving for all elements (isize) may over-allocate when there are few unique values.
    // - Using a moderate upper bound HASH_TABLE_INITIAL_BUCKETS(1024) keeps memory usage reasonable (4 KiB for pointers).
    // See discussion: https://github.com/numpy/numpy/pull/28767#discussion_r2064267631
    std::unordered_set<T *, decltype(hash), decltype(equal)> hashset(
        std::min(isize, (npy_intp)HASH_TABLE_INITIAL_BUCKETS), hash, equal
    );

    // Input array is one-dimensional, enabling efficient iteration using strides.
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        hashset.insert((T *)idata);
    }

    npy_intp length = hashset.size();

    PyEval_RestoreThread(_save1);
    NPY_ALLOW_C_API;
    PyObject *res_obj = PyArray_NewFromDescr(
        &PyArray_Type,
        descr,
        1, // ndim
        &length, // shape
        NULL, // strides
        NULL, // data
        // This flag is needed to be able to call .sort on it.
        NPY_ARRAY_WRITEABLE, // flags
        NULL // obj
    );

    if (res_obj == NULL) {
        return NULL;
    }
    NPY_DISABLE_C_API;
    PyThreadState *_save2 = PyEval_SaveThread();
    auto save2_dealloc = finally([&]() {
        PyEval_RestoreThread(_save2);
    });

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];
    // Output array is one-dimensional, enabling efficient iteration using strides.
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        copy_func(odata, *it);
    }

    return res_obj;
}

template <typename T>
static PyObject*
unique_string(PyArrayObject *self, npy_bool equal_nan)
{
    /*
    * Returns a new NumPy array containing the unique values of the input array of fixed size strings.
    * This function uses hashing to identify uniqueness efficiently.
    */
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
    NPY_DISABLE_C_API;

    PyThreadState *_save1 = PyEval_SaveThread();

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);

    // variables for the string
    npy_intp itemsize = descr->elsize;
    npy_intp num_chars = itemsize / sizeof(T);
    auto hash = [num_chars](const T *value) -> size_t {
        return npy_fnv1a(value, num_chars * sizeof(T));
    };
    auto equal = [itemsize](const T *lhs, const T *rhs) -> bool {
        return std::memcmp(lhs, rhs, itemsize) == 0;
    };

    // Reserve hashset capacity in advance to minimize reallocations and collisions.
    // We use min(isize, HASH_TABLE_INITIAL_BUCKETS) as the initial bucket count:
    // - Reserving for all elements (isize) may over-allocate when there are few unique values.
    // - Using a moderate upper bound HASH_TABLE_INITIAL_BUCKETS(1024) keeps memory usage reasonable (4 KiB for pointers).
    // See discussion: https://github.com/numpy/numpy/pull/28767#discussion_r2064267631
    std::unordered_set<T *, decltype(hash), decltype(equal)> hashset(
        std::min(isize, (npy_intp)HASH_TABLE_INITIAL_BUCKETS), hash, equal
    );

    // Input array is one-dimensional, enabling efficient iteration using strides.
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        hashset.insert((T *)idata);
    }

    npy_intp length = hashset.size();

    PyEval_RestoreThread(_save1);
    NPY_ALLOW_C_API;
    PyObject *res_obj = PyArray_NewFromDescr(
        &PyArray_Type,
        descr,
        1, // ndim
        &length, // shape
        NULL, // strides
        NULL, // data
        // This flag is needed to be able to call .sort on it.
        NPY_ARRAY_WRITEABLE, // flags
        NULL // obj
    );

    if (res_obj == NULL) {
        return NULL;
    }
    NPY_DISABLE_C_API;
    PyThreadState *_save2 = PyEval_SaveThread();
    auto save2_dealloc = finally([&]() {
        PyEval_RestoreThread(_save2);
    });

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];
    // Output array is one-dimensional, enabling efficient iteration using strides.
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        std::memcpy(odata, *it, itemsize);
    }

    return res_obj;
}

static PyObject*
unique_vstring(PyArrayObject *self, npy_bool equal_nan)
{
    /*
    * Returns a new NumPy array containing the unique values of the input array.
    * This function uses hashing to identify uniqueness efficiently.
    */
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
    NPY_DISABLE_C_API;

    PyThreadState *_save1 = PyEval_SaveThread();

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);

    // variables for the vstring
    npy_string_allocator *in_allocator = NpyString_acquire_allocator((PyArray_StringDTypeObject *)descr);
    auto hash = [equal_nan](const npy_static_string *value) -> size_t {
        if (value->buf == NULL) {
            if (equal_nan) {
                return 0;
            } else {
                return std::hash<const npy_static_string *>{}(value);
            }
        }
        return npy_fnv1a(value->buf, value->size * sizeof(char));
    };
    auto equal = [equal_nan](const npy_static_string *lhs, const npy_static_string *rhs) -> bool {
        if (lhs->buf == NULL && rhs->buf == NULL) {
            if (equal_nan) {
                return true;
            } else {
                return lhs == rhs;
            }
        }
        if (lhs->buf == NULL || rhs->buf == NULL) {
            return false;
        }
        if (lhs->size != rhs->size) {
            return false;
        }
        return std::memcmp(lhs->buf, rhs->buf, lhs->size) == 0;
    };

    // Reserve hashset capacity in advance to minimize reallocations and collisions.
    // We use min(isize, HASH_TABLE_INITIAL_BUCKETS) as the initial bucket count:
    // - Reserving for all elements (isize) may over-allocate when there are few unique values.
    // - Using a moderate upper bound HASH_TABLE_INITIAL_BUCKETS(1024) keeps memory usage reasonable (4 KiB for pointers).
    // See discussion: https://github.com/numpy/numpy/pull/28767#discussion_r2064267631
    std::unordered_set<npy_static_string *, decltype(hash), decltype(equal)> hashset(
        std::min(isize, (npy_intp)HASH_TABLE_INITIAL_BUCKETS), hash, equal
    );

    // Input array is one-dimensional, enabling efficient iteration using strides.
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    // unpacked_strings need to be allocated outside of the loop because of the lifetime problem.
    std::vector<npy_static_string> unpacked_strings(isize, {0, NULL});
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        npy_packed_static_string *packed_string = (npy_packed_static_string *)idata;
        int is_null = NpyString_load(in_allocator, packed_string, &unpacked_strings[i]);
        if (is_null == -1) {
            npy_gil_error(PyExc_RuntimeError,
                "Failed to load string from packed static string. ");
            return NULL;
        }
        hashset.insert(&unpacked_strings[i]);
    }

    NpyString_release_allocator(in_allocator);

    npy_intp length = hashset.size();

    PyEval_RestoreThread(_save1);
    NPY_ALLOW_C_API;
    PyObject *res_obj = PyArray_NewFromDescr(
        &PyArray_Type,
        descr,
        1, // ndim
        &length, // shape
        NULL, // strides
        NULL, // data
        // This flag is needed to be able to call .sort on it.
        NPY_ARRAY_WRITEABLE, // flags
        NULL // obj
    );
    if (res_obj == NULL) {
        return NULL;
    }
    PyArray_Descr *res_descr = PyArray_DESCR((PyArrayObject *)res_obj);
    Py_INCREF(res_descr);
    NPY_DISABLE_C_API;

    PyThreadState *_save2 = PyEval_SaveThread();
    auto save2_dealloc = finally([&]() {
        PyEval_RestoreThread(_save2);
    });

    npy_string_allocator *out_allocator = NpyString_acquire_allocator((PyArray_StringDTypeObject *)res_descr);
    auto out_allocator_dealloc = finally([&]() {
        NpyString_release_allocator(out_allocator);
    });

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];
    // Output array is one-dimensional, enabling efficient iteration using strides.
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        npy_packed_static_string *packed_string = (npy_packed_static_string *)odata;
        int pack_status = 0;
        if ((*it)->buf == NULL) {
            pack_status = NpyString_pack_null(out_allocator, packed_string);
        } else {
            pack_status = NpyString_pack(out_allocator, packed_string, (*it)->buf, (*it)->size);
        }
        if (pack_status == -1) {
            // string packing failed
            return NULL;
        }
    }

    return res_obj;
}


// this map contains the functions used for each item size.
typedef std::function<PyObject *(PyArrayObject *, npy_bool)> function_type;
std::unordered_map<int, function_type> unique_funcs = {
    {NPY_BYTE, unique_numeric<npy_byte, hash_nonan<npy_byte>, equal_nonan<npy_byte>, copy_real<npy_byte>>},
    {NPY_UBYTE, unique_numeric<npy_ubyte, hash_nonan<npy_ubyte>, equal_nonan<npy_ubyte>, copy_real<npy_ubyte>>},
    {NPY_SHORT, unique_numeric<npy_short, hash_nonan<npy_short>, equal_nonan<npy_short>, copy_real<npy_short>>},
    {NPY_USHORT, unique_numeric<npy_ushort, hash_nonan<npy_ushort>, equal_nonan<npy_ushort>, copy_real<npy_ushort>>},
    {NPY_INT, unique_numeric<npy_int, hash_nonan<npy_int>, equal_nonan<npy_int>, copy_real<npy_int>>},
    {NPY_UINT, unique_numeric<npy_uint, hash_nonan<npy_uint>, equal_nonan<npy_uint>, copy_real<npy_uint>>},
    {NPY_LONG, unique_numeric<npy_long, hash_nonan<npy_long>, equal_nonan<npy_long>, copy_real<npy_long>>},
    {NPY_ULONG, unique_numeric<npy_ulong, hash_nonan<npy_ulong>, equal_nonan<npy_ulong>, copy_real<npy_ulong>>},
    {NPY_LONGLONG, unique_numeric<npy_longlong, hash_nonan<npy_longlong>, equal_nonan<npy_longlong>, copy_real<npy_longlong>>},
    {NPY_ULONGLONG, unique_numeric<npy_ulonglong, hash_nonan<npy_ulonglong>, equal_nonan<npy_ulonglong>, copy_real<npy_ulonglong>>},
    {NPY_HALF, unique_numeric<
        npy_half,
        hash_maybenan<npy_half, npy_half_isnan>,
        equal_maybenan<npy_half, npy_half_isnan, npy_half_eq_nonan>,
        copy_real<npy_half>
        >
    },
    {NPY_FLOAT, unique_numeric<npy_float, hash_maybenan<npy_float>, equal_maybenan<npy_float>, copy_real<npy_float>>},
    {NPY_DOUBLE, unique_numeric<npy_double, hash_maybenan<npy_double>, equal_maybenan<npy_double>, copy_real<npy_double>>},
    {NPY_LONGDOUBLE, unique_numeric<npy_longdouble, hash_maybenan<npy_longdouble>, equal_maybenan<npy_longdouble>, copy_real<npy_longdouble>>},
    {NPY_CFLOAT, unique_numeric<
        npy_cfloat,
        hash_complex<float, npy_cfloat, npy_crealf, npy_cimagf>,
        equal_complex<float, npy_cfloat, npy_crealf, npy_cimagf>,
        copy_complex<float, npy_cfloat, npy_crealf, npy_cimagf, npy_csetrealf, npy_csetimagf>
        >
    },
    {NPY_CDOUBLE, unique_numeric<
        npy_cdouble,
        hash_complex<double, npy_cdouble, npy_creal, npy_cimag>,
        equal_complex<double, npy_cdouble, npy_creal, npy_cimag>,
        copy_complex<double, npy_cdouble, npy_creal, npy_cimag, npy_csetreal, npy_csetimag>
        >
    },
    {NPY_CLONGDOUBLE, unique_numeric<
        npy_clongdouble,
        hash_complex<npy_longdouble, npy_clongdouble, npy_creall, npy_cimagl>,
        equal_complex<npy_longdouble, npy_clongdouble, npy_creall, npy_cimagl>,
        copy_complex<npy_longdouble, npy_clongdouble, npy_creall, npy_cimagl, npy_csetreall, npy_csetimagl>
        >
    },
    {NPY_INT8, unique_numeric<npy_int8, hash_nonan<npy_int8>, equal_nonan<npy_int8>, copy_real<npy_int8>>},
    {NPY_INT16, unique_numeric<npy_int16, hash_nonan<npy_int16>, equal_nonan<npy_int16>, copy_real<npy_int16>>},
    {NPY_INT32, unique_numeric<npy_int32, hash_nonan<npy_int32>, equal_nonan<npy_int32>, copy_real<npy_int32>>},
    {NPY_INT64, unique_numeric<npy_int64, hash_nonan<npy_int64>, equal_nonan<npy_int64>, copy_real<npy_int64>>},
    {NPY_UINT8, unique_numeric<npy_uint8, hash_nonan<npy_uint8>, equal_nonan<npy_uint8>, copy_real<npy_uint8>>},
    {NPY_UINT16, unique_numeric<npy_uint16, hash_nonan<npy_uint16>, equal_nonan<npy_uint16>, copy_real<npy_uint16>>},
    {NPY_UINT32, unique_numeric<npy_uint32, hash_nonan<npy_uint32>, equal_nonan<npy_uint32>, copy_real<npy_uint32>>},
    {NPY_UINT64, unique_numeric<npy_uint64, hash_nonan<npy_uint64>, equal_nonan<npy_uint64>, copy_real<npy_uint64>>},
    {NPY_DATETIME, unique_numeric<npy_uint64, hash_nonan<npy_uint64>, equal_nonan<npy_uint64>, copy_real<npy_uint64>>},
    {NPY_FLOAT16, unique_numeric<
        npy_float16,
        hash_maybenan<npy_float16, npy_half_isnan>,
        equal_maybenan<npy_float16, npy_half_isnan, npy_half_eq_nonan>,
        copy_real<npy_float16>
        >
    },
    {NPY_FLOAT32, unique_numeric<npy_float32, hash_maybenan<npy_float32>, equal_maybenan<npy_float32>, copy_real<npy_float32>>},
    {NPY_FLOAT64, unique_numeric<npy_float64, hash_maybenan<npy_float64>, equal_maybenan<npy_float64>, copy_real<npy_float64>>},
    {NPY_COMPLEX64, unique_numeric<
        npy_complex64,
        hash_complex<float, npy_complex64, npy_crealf, npy_cimagf>,
        equal_complex<float, npy_complex64, npy_crealf, npy_cimagf>,
        copy_complex<float, npy_complex64, npy_crealf, npy_cimagf, npy_csetrealf, npy_csetimagf>
        >
    },
    {NPY_COMPLEX128, unique_numeric<
        npy_complex128,
        hash_complex<double, npy_complex128, npy_creal, npy_cimag>,
        equal_complex<double, npy_complex128, npy_creal, npy_cimag>,
        copy_complex<double, npy_complex128, npy_creal, npy_cimag, npy_csetreal, npy_csetimag>
        >
    },
    {NPY_STRING, unique_string<npy_byte>},
    {NPY_UNICODE, unique_string<npy_ucs4>},
    {NPY_VSTRING, unique_vstring},
};


/**
 * Python exposed implementation of `_unique_hash`.
 *
 * This is a C only function wrapping code that may cause C++ exceptions into
 * try/catch.
 *
 * @param arr NumPy array to find the unique values of.
 * @return Base-class NumPy array with unique values, `NotImplemented` if the
 * type is unsupported or `NULL` with an error set.
 */
extern "C" NPY_NO_EXPORT PyObject *
array__unique_hash(PyObject *NPY_UNUSED(module),
                   PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArrayObject *arr = NULL;
    npy_bool equal_nan = NPY_TRUE;  // default to True

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("_unique_hash", args, len_args, kwnames,
                            "arr", &PyArray_Converter, &arr,
                            "|equal_nan",  &PyArray_BoolConverter, &equal_nan,
                            NULL, NULL, NULL
                            ) < 0
    ) {
        return NULL;
    }

    try {
        auto type = PyArray_TYPE(arr);
        // we only support data types present in our unique_funcs map
        if (unique_funcs.find(type) == unique_funcs.end()) {
            Py_RETURN_NOTIMPLEMENTED;
        }

        return unique_funcs[type](arr, equal_nan);
    }
    catch (const std::bad_alloc &e) {
        PyErr_NoMemory();
        return NULL;
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
