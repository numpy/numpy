#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <cstring>
#include <functional>
#include <unordered_set>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"

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

// function to caluculate the hash of a string
template <typename T>
size_t str_hash(const T *str, npy_intp num_chars) {
    // https://www.boost.org/doc/libs/1_88_0/libs/container_hash/doc/html/hash.html#notes_hash_combine
    size_t h = 0;
    for (npy_intp i = 0; i < num_chars; ++i) {
        h ^= std::hash<T>{}(str[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

template <typename T>
static PyObject*
unique_integer(PyArrayObject *self)
{
    /*
    * Returns a new NumPy array containing the unique values of the input array of integer.
    * This function uses hashing to identify uniqueness efficiently.
    */
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    // NumPy API calls and Python object manipulations require holding the GIL.
    Py_INCREF(descr);
    NPY_DISABLE_C_API;

    // release the GIL
    PyThreadState *_save1 = PyEval_SaveThread();

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);

    // Reserve hashset capacity in advance to minimize reallocations.
    std::unordered_set<T> hashset(isize * 2);

    // Input array is one-dimensional, enabling efficient iteration using strides.
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        hashset.insert(*(T *)idata);
    }

    npy_intp length = hashset.size();

    PyEval_RestoreThread(_save1);
    NPY_ALLOW_C_API;
    // NumPy API calls and Python object manipulations require holding the GIL.
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

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];
    // Output array is one-dimensional, enabling efficient iteration using strides.
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        *(T *)odata = *it;
    }

    PyEval_RestoreThread(_save2);
    return res_obj;
}

template <typename T>
static PyObject*
unique_string(PyArrayObject *self)
{
    /*
    * Returns a new NumPy array containing the unique values of the input array of fixed size strings.
    * This function uses hashing to identify uniqueness efficiently.
    */
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    // NumPy API calls and Python object manipulations require holding the GIL.
    Py_INCREF(descr);
    NPY_DISABLE_C_API;

    // release the GIL
    PyThreadState *_save1 = PyEval_SaveThread();

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);

    // variables for the string
    npy_intp itemsize = descr->elsize;
    npy_intp num_chars = itemsize / sizeof(T);
    auto hash = [num_chars](const T *value) -> size_t {
        return str_hash(value, num_chars);
    };
    auto equal = [num_chars](const T *lhs, const T *rhs) -> bool {
        return std::memcmp(lhs, rhs, num_chars) == 0;
    };

    // Reserve hashset capacity in advance to minimize reallocations.
    std::unordered_set<T *, decltype(hash), decltype(equal)> hashset(
        isize * 2, hash, equal
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
    // NumPy API calls and Python object manipulations require holding the GIL.
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

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];
    // Output array is one-dimensional, enabling efficient iteration using strides.
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        std::memcpy(odata, *it, itemsize);
    }

    PyEval_RestoreThread(_save2);
    return res_obj;
}

static PyObject*
unique_vstring(PyArrayObject *self)
{
    /*
    * Returns a new NumPy array containing the unique values of the input array.
    * This function uses hashing to identify uniqueness efficiently.
    */
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    // NumPy API calls and Python object manipulations require holding the GIL.
    Py_INCREF(descr);
    NPY_DISABLE_C_API;

    // release the GIL
    PyThreadState *_save1 = PyEval_SaveThread();

    // number of elements in the input array
    npy_intp isize = PyArray_SIZE(self);

    // variables for the vstring
    npy_string_allocator *allocator = NpyString_acquire_allocator((PyArray_StringDTypeObject *)descr);
    auto allocator_dealloc = finally([&]() {
        NpyString_release_allocator(allocator);
    });
    // unpacked_strings need to be allocated outside of the loop because of lifetime.
    std::vector<npy_static_string> unpacked_strings(isize, {0, NULL});
    auto hash = [](const npy_static_string *value) -> size_t {
        if (value->buf == NULL) {
            return 0;
        }
        return str_hash(value->buf, value->size);
    };
    auto equal = [](const npy_static_string *lhs, const npy_static_string *rhs) -> bool {
        if (lhs->buf == NULL && rhs->buf == NULL) {
            return true;
        }
        if (lhs->buf == NULL || rhs->buf == NULL) {
            return false;
        }
        if (lhs->size != rhs->size) {
            return false;
        }
        return std::memcmp(lhs->buf, rhs->buf, lhs->size) == 0;
    };

    // Reserve hashset capacity in advance to minimize reallocations.
    std::unordered_set<npy_static_string *, decltype(hash), decltype(equal)> hashset(
        isize * 2, hash, equal
    );

    // Input array is one-dimensional, enabling efficient iteration using strides.
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        npy_packed_static_string *packed_string = (npy_packed_static_string *)idata;
        NpyString_load(allocator, packed_string, &unpacked_strings[i]);
        hashset.insert(&unpacked_strings[i]);
    }

    npy_intp length = hashset.size();

    PyEval_RestoreThread(_save1);
    NPY_ALLOW_C_API;
    // NumPy API calls and Python object manipulations require holding the GIL.
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

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];
    // Output array is one-dimensional, enabling efficient iteration using strides.
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        npy_packed_static_string *packed_string = (npy_packed_static_string *)odata;
        if ((*it)->buf == NULL) {
            NpyString_pack_null(allocator, packed_string);
        } else {
            NpyString_pack(allocator, packed_string, (*it)->buf, (*it)->size);
        }
    }

    PyEval_RestoreThread(_save2);
    return res_obj;
}


// this map contains the functions used for each item size.
typedef std::function<PyObject *(PyArrayObject *)> function_type;
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
array__unique_hash(PyObject *NPY_UNUSED(module), PyObject *arr_obj)
{
    if (!PyArray_Check(arr_obj)) {
        PyErr_SetString(PyExc_TypeError,
                "_unique_hash() requires a NumPy array input.");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)arr_obj;

    try {
        auto type = PyArray_TYPE(arr);
        // we only support data types present in our unique_funcs map
        if (unique_funcs.find(type) == unique_funcs.end()) {
            Py_RETURN_NOTIMPLEMENTED;
        }

        return unique_funcs[type](arr);
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
