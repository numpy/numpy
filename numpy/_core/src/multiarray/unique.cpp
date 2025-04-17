#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <cstring>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <vector>

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

template <typename T>
struct VectorHash {
    size_t operator()(const std::vector<T>& v) const {
        // https://www.boost.org/doc/libs/1_88_0/libs/container_hash/doc/html/hash.html#notes_hash_combine
        size_t h = v.size();
        for (const T& x : v) {
            h ^= std::hash<T>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

template <typename T>
struct VectorEqual {
    bool operator()(const std::vector<T>& lhs, const std::vector<T>& rhs) const {
        return lhs == rhs;
    }
};

template<typename T>
static PyObject*
unique_int(PyArrayObject *self)
{
    /* This function takes a numpy array and returns a numpy array containing
    the unique values.

    It assumes the numpy array includes data that can be viewed as unsigned integers
    of a certain size (sizeof(T)).

    It doesn't need to know the actual type, since it needs to find unique values
    among binary representations of the input data. This means it won't apply to
    custom or complicated dtypes or string values.
    */
    NPY_ALLOW_C_API_DEF;
    std::unordered_set<T> hashset;

    NpyIter *iter = NpyIter_New(self, NPY_ITER_READONLY |
                                      NPY_ITER_EXTERNAL_LOOP |
                                      NPY_ITER_REFS_OK |
                                      NPY_ITER_ZEROSIZE_OK |
                                      NPY_ITER_GROWINNER,
                                NPY_KEEPORDER, NPY_NO_CASTING,
                                NULL);
    // Making sure the iterator is deallocated when the function returns, with
    // or w/o an exception
    auto iter_dealloc = finally([&]() { NpyIter_Deallocate(iter); });
    if (iter == NULL) {
        return NULL;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        return NULL;
    }
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // release the GIL
    PyThreadState *_save;
    _save = PyEval_SaveThread();
    // Making sure the GIL is re-acquired when the function returns, with
    // or w/o an exception
    auto grab_gil = finally([&]() { PyEval_RestoreThread(_save); });
    // first we put the data in a hash map

    if (NpyIter_GetIterSize(iter) > 0) {
        do {
            char* data = *dataptr;
            npy_intp stride = *strideptr;
            npy_intp count = *innersizeptr;

            while (count--) {
                hashset.insert(*((T *) data));
                data += stride;
            }
        } while (iternext(iter));
    }

    npy_intp length = hashset.size();

    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
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
    NPY_DISABLE_C_API;

    if (res_obj == NULL) {
        return NULL;
    }

    // then we iterate through the map's keys to get the unique values
    T* data = (T *)PyArray_DATA((PyArrayObject *)res_obj);
    auto it = hashset.begin();
    size_t i = 0;
    for (; it != hashset.end(); it++, i++) {
        data[i] = *it;
    }

    return res_obj;
}


template<typename T>
static PyObject*
unique_string(PyArrayObject *self)
{
    /* This function takes a numpy array and returns a numpy array containing
    the unique values.

    It assumes the numpy array includes data that can be viewed as fixed-size
    strings of a certain size (itemsize / sizeof(T)).
    */
    NPY_ALLOW_C_API_DEF;
    std::unordered_set<std::vector<T>, VectorHash<T>, VectorEqual<T>> hashset;

    NpyIter *iter = NpyIter_New(self, NPY_ITER_READONLY |
                                      NPY_ITER_EXTERNAL_LOOP |
                                      NPY_ITER_REFS_OK |
                                      NPY_ITER_ZEROSIZE_OK |
                                      NPY_ITER_GROWINNER,
                                NPY_KEEPORDER, NPY_NO_CASTING,
                                NULL);
    // Making sure the iterator is deallocated when the function returns, with
    // or w/o an exception
    auto iter_dealloc = finally([&]() { NpyIter_Deallocate(iter); });
    if (iter == NULL) {
        return NULL;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        return NULL;
    }
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // release the GIL
    PyThreadState *_save;
    _save = PyEval_SaveThread();
    // Making sure the GIL is re-acquired when the function returns, with
    // or w/o an exception
    auto grab_gil = finally([&]() { PyEval_RestoreThread(_save); });
    // first we put the data in a hash map

    // item size for the string type
    npy_intp itemsize = PyArray_ITEMSIZE(self);

    // the number of characters (For Unicode, itemsize / 4 for UCS4)
    npy_intp num_chars = itemsize / sizeof(T);

    if (NpyIter_GetIterSize(iter) > 0) {
        do {
            char* data = *dataptr;
            npy_intp stride = *strideptr;
            npy_intp count = *innersizeptr;

            while (count--) {
                hashset.emplace((T *) data, (T *) data + num_chars);
                data += stride;
            }
        } while (iternext(iter));
    }

    npy_intp length = hashset.size();

    NPY_ALLOW_C_API;
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
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
    NPY_DISABLE_C_API;

    if (res_obj == NULL) {
        return NULL;
    }

    // then we iterate through the map's keys to get the unique values
    char* data = PyArray_BYTES((const PyArrayObject *)res_obj);
    auto it = hashset.begin();
    size_t i = 0;
    for (; it != hashset.end(); it++, i++) {
        size_t byte_to_copy = it->size() * sizeof(T);
        std::memcpy(data + i * itemsize, it->data(), byte_to_copy);
        if (byte_to_copy < (size_t)itemsize) {
            std::memset(data + i * itemsize + byte_to_copy, 0, itemsize - byte_to_copy);
        }
    }

    return res_obj;
}


// this map contains the functions used for each item size.
typedef std::function<PyObject *(PyArrayObject *)> function_type;
std::unordered_map<int, function_type> unique_funcs = {
    {NPY_BYTE, unique_int<npy_byte>},
    {NPY_UBYTE, unique_int<npy_ubyte>},
    {NPY_SHORT, unique_int<npy_short>},
    {NPY_USHORT, unique_int<npy_ushort>},
    {NPY_INT, unique_int<npy_int>},
    {NPY_UINT, unique_int<npy_uint>},
    {NPY_LONG, unique_int<npy_long>},
    {NPY_ULONG, unique_int<npy_ulong>},
    {NPY_LONGLONG, unique_int<npy_longlong>},
    {NPY_ULONGLONG, unique_int<npy_ulonglong>},
    {NPY_INT8, unique_int<npy_int8>},
    {NPY_INT16, unique_int<npy_int16>},
    {NPY_INT32, unique_int<npy_int32>},
    {NPY_INT64, unique_int<npy_int64>},
    {NPY_UINT8, unique_int<npy_uint8>},
    {NPY_UINT16, unique_int<npy_uint16>},
    {NPY_UINT32, unique_int<npy_uint32>},
    {NPY_UINT64, unique_int<npy_uint64>},
    {NPY_DATETIME, unique_int<npy_uint64>},
    {NPY_STRING, unique_string<npy_byte>},
    {NPY_UNICODE, unique_string<npy_ucs4>},
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
