#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <iostream>
#include <unordered_set>
#include <functional>
#include <string>
#include <utility>

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

template<typename T>
static PyObject*
unique_integer(PyArrayObject *self)
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
    std::unordered_set<std::basic_string<T>> hashset;
    // reserve the hashset to avoid reallocations
    // reallocations are expensive, especially for string arrays
    hashset.reserve(PyArray_SIZE(self) * 2);

    NpyIter *iter = NpyIter_New(self, NPY_ITER_READONLY |
                                      NPY_ITER_EXTERNAL_LOOP |
                                      NPY_ITER_REFS_OK |
                                      NPY_ITER_ZEROSIZE_OK |
                                      NPY_ITER_GROWINNER,
                                NPY_KEEPORDER, NPY_NO_CASTING,
                                NULL);
    if (iter == NULL) {
        return NULL;
    }
    // Making sure the iterator is deallocated when the function returns, with
    // or w/o an exception
    auto iter_dealloc = finally([&]() { NpyIter_Deallocate(iter); });

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

    NPY_ALLOW_C_API;
    // size of each entries
    npy_intp itemsize = PyArray_ITEMSIZE(self);
    NPY_DISABLE_C_API;

    // the number of characters of each entries
    // (For Unicode, itemsize / 4 for UCS4)
    npy_intp num_chars = itemsize / sizeof(T);

    // first we put the data in a hash map
    if (NpyIter_GetIterSize(iter) > 0) {
        do {
            char* data = *dataptr;
            npy_intp stride = *strideptr;
            npy_intp count = *innersizeptr;

            while (count--) {
                T * sdata = reinterpret_cast<T *>(data);
                std::basic_string<T> sdata_str(sdata, num_chars);
                hashset.emplace(std::move(sdata_str));
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

    if (res_obj == NULL) {
        return NULL;
    }

    // then we iterate through the map's keys to get the unique values
    auto it = hashset.begin();
    size_t i = 0;
    for (; it != hashset.end(); it++, i++) {
        char* data = (char *)PyArray_GETPTR1((PyArrayObject *)res_obj, i);
        size_t byte_to_copy = it->size() * sizeof(T);
        memcpy(data, it->c_str(), byte_to_copy);
        if (byte_to_copy < (size_t)itemsize) {
            memset(data + byte_to_copy, 0, itemsize - byte_to_copy);
        }
    }
    NPY_DISABLE_C_API;

    return res_obj;
}


static PyObject*
unique_vstring(PyArrayObject *self)
{
    /* This function takes a numpy array and returns a numpy array containing
    the unique values.

    It assumes the numpy array includes data that can be viewed as variable width
    strings (StringDType).
    */
    NPY_ALLOW_C_API_DEF;
    std::unordered_set<std::string> hashset;
    // reserve the hashset to avoid reallocations
    // reallocations are expensive, especially for string arrays
    hashset.reserve(PyArray_SIZE(self) * 2);

    NpyIter *iter = NpyIter_New(self, NPY_ITER_READONLY |
                                      NPY_ITER_EXTERNAL_LOOP |
                                      NPY_ITER_REFS_OK |
                                      NPY_ITER_ZEROSIZE_OK |
                                      NPY_ITER_GROWINNER,
                                NPY_KEEPORDER, NPY_NO_CASTING,
                                NULL);
    if (iter == NULL) {
        return NULL;
    }
    // Making sure the iterator is deallocated when the function returns, with
    // or w/o an exception
    auto iter_dealloc = finally([&]() { NpyIter_Deallocate(iter); });

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

    NPY_ALLOW_C_API;
    // https://numpy.org/doc/stable/reference/c-api/strings.html#loading-a-string
    PyArray_Descr *descr = PyArray_DESCR(self);
    Py_INCREF(descr);
    NPY_DISABLE_C_API;

    npy_string_allocator *allocator = NpyString_acquire_allocator(
        (PyArray_StringDTypeObject *)descr);
    auto allocator_dealloc = finally([&]() {
        NpyString_release_allocator(allocator);
    });

    // whether the array contains null values
    bool contains_null = false;

    // first we put the data in a hash map
    if (NpyIter_GetIterSize(iter) > 0) {
        do {
            char* data = *dataptr;
            npy_intp stride = *strideptr;
            npy_intp count = *innersizeptr;

            while (count--) {
                npy_static_string sdata = {0, NULL};
                npy_packed_static_string *packed_string = (npy_packed_static_string *)data;
                int is_null = NpyString_load(allocator, packed_string, &sdata);

                if (is_null == -1) {
                    return NULL;
                }
                else if (is_null) {
                    contains_null = true;
                }
                else {
                    std::string sdata_str(sdata.buf, sdata.size);
                    hashset.emplace(std::move(sdata_str));
                }
                data += stride;
            }
        } while (iternext(iter));
    }

    npy_intp length = hashset.size();
    if (contains_null) {
        length++;
    }

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

    // then we iterate through the map's keys to get the unique values
    auto it = hashset.begin();
    size_t i = 0;
    if (contains_null) {
        // insert null if original array contains null
        char* data = (char *)PyArray_GETPTR1((PyArrayObject *)res_obj, i);
        npy_packed_static_string *packed_string = (npy_packed_static_string *)data;
        if (NpyString_pack_null(allocator, packed_string) == -1) {
            return NULL;
        }
        i++;
    }
    for (; it != hashset.end(); it++, i++) {
        char* data = (char *)PyArray_GETPTR1((PyArrayObject *)res_obj, i);
        npy_packed_static_string *packed_string = (npy_packed_static_string *)data;
        if (NpyString_pack(allocator, packed_string, it->c_str(), it->size()) == -1) {
            return NULL;
        }
    }
    NPY_DISABLE_C_API;

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
    {NPY_STRING, unique_string<char>},
    {NPY_UNICODE, unique_string<char32_t>},
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
