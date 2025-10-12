#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define HASH_TABLE_INITIAL_BUCKETS 1024
#include <Python.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "gil_utils.h"
extern "C" {
    #include "fnv.h"
    #include "npy_argparse.h"
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
static PyObject*
unique_integer(PyArrayObject *self, npy_bool equal_nan)
{
    /*
    * Returns a new NumPy array containing the unique values of the input array of integer.
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

    // Reserve hashset capacity in advance to minimize reallocations and collisions.
    // We use min(isize, HASH_TABLE_INITIAL_BUCKETS) as the initial bucket count:
    // - Reserving for all elements (isize) may over-allocate when there are few unique values.
    // - Using a moderate upper bound HASH_TABLE_INITIAL_BUCKETS(1024) keeps memory usage reasonable (4 KiB for pointers).
    // See discussion: https://github.com/numpy/numpy/pull/28767#discussion_r2064267631
    std::unordered_set<T> hashset(std::min(isize, (npy_intp)HASH_TABLE_INITIAL_BUCKETS));

    // Input array is one-dimensional, enabling efficient iteration using strides.
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        hashset.insert(*(T *)idata);
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
        *(T *)odata = *it;
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
    {NPY_BYTE, unique_integer<npy_byte>},
    {NPY_UBYTE, unique_integer<npy_ubyte>},
    {NPY_SHORT, unique_integer<npy_short>},
    {NPY_USHORT, unique_integer<npy_ushort>},
    {NPY_INT, unique_integer<npy_int>},
    {NPY_UINT, unique_integer<npy_uint>},
    {NPY_LONG, unique_integer<npy_long>},
    {NPY_ULONG, unique_integer<npy_ulong>},
    {NPY_LONGLONG, unique_integer<npy_longlong>},
    {NPY_ULONGLONG, unique_integer<npy_ulonglong>},
    {NPY_INT8, unique_integer<npy_int8>},
    {NPY_INT16, unique_integer<npy_int16>},
    {NPY_INT32, unique_integer<npy_int32>},
    {NPY_INT64, unique_integer<npy_int64>},
    {NPY_UINT8, unique_integer<npy_uint8>},
    {NPY_UINT16, unique_integer<npy_uint16>},
    {NPY_UINT32, unique_integer<npy_uint32>},
    {NPY_UINT64, unique_integer<npy_uint64>},
    {NPY_DATETIME, unique_integer<npy_uint64>},
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
