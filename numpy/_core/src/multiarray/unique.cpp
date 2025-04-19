#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <algorithm>
#include <unordered_set>
#include <functional>
#include <optional>
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

    // release the GIL
    PyThreadState *_save1 = PyEval_SaveThread();

    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];

    std::unordered_set<T> hashset;
    // reserve the hashset to avoid reallocations
    // reallocations are expensive, especially for string arrays
    hashset.reserve(isize * 2);

    // As input is 1d, we can use the strides to iterate through the array.
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        hashset.insert(*((T *) idata));
    }

    npy_intp length = hashset.size();

    PyEval_RestoreThread(_save1);
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
    NPY_DISABLE_C_API;
    PyThreadState *_save2 = PyEval_SaveThread();

    // then we iterate through the map's keys to get the unique values
    T* data = (T *)PyArray_DATA((PyArrayObject *)res_obj);
    auto it = hashset.begin();
    size_t i = 0;
    for (; it != hashset.end(); it++, i++) {
        data[i] = *it;
    }

    PyEval_RestoreThread(_save2);
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

    // release the GIL
    PyThreadState *_save1 = PyEval_SaveThread();

    // size of each entries
    npy_intp itemsize = PyArray_ITEMSIZE(self);
    // the number of characters of each entries
    // (For Unicode, itemsize / 4 for UCS4)
    npy_intp num_chars = itemsize / sizeof(typename T::value_type);

    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];

    std::unordered_set<T> hashset;
    // reserve the hashset to avoid reallocations
    // reallocations are expensive, especially for string arrays
    hashset.reserve(isize * 2);

    // As input is 1d, we can use the strides to iterate through the array.
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        typename T::value_type *sdata = reinterpret_cast<typename T::value_type *>(idata);
        size_t byte_to_copy = std::find(sdata, sdata + num_chars, 0) - sdata;
        T sdata_str(sdata, byte_to_copy);
        hashset.emplace(std::move(sdata_str));
    }

    npy_intp length = hashset.size();

    PyEval_RestoreThread(_save1);
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
    NPY_DISABLE_C_API;
    PyThreadState *_save2 = PyEval_SaveThread();

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];

    memset(odata, 0, itemsize * length);
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        size_t byte_to_copy = it->size() * sizeof(typename T::value_type);
        memcpy(odata, it->c_str(), byte_to_copy);
    }

    PyEval_RestoreThread(_save2);
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

    PyArray_Descr *descr = PyArray_DESCR(self);
    // this macro requires the GIL to be held
    Py_INCREF(descr);

    // release the GIL
    PyThreadState *_save1 = PyEval_SaveThread();

    npy_string_allocator *allocator = NpyString_acquire_allocator(
        (PyArray_StringDTypeObject *)descr);
    auto allocator_dealloc = finally([&]() {
        NpyString_release_allocator(allocator);
    });

    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];

    std::unordered_set<std::optional<std::string>> hashset;
    // reserve the hashset to avoid reallocations
    // reallocations are expensive, especially for string arrays
    hashset.reserve(isize * 2);

    // As input is 1d, we can use the strides to iterate through the array.
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        // https://numpy.org/doc/stable/reference/c-api/strings.html#loading-a-string
        npy_static_string sdata = {0, NULL};
        npy_packed_static_string *packed_string = (npy_packed_static_string *)idata;
        int is_null = NpyString_load(allocator, packed_string, &sdata);

        if (is_null == -1) {
            return NULL;
        }
        else if (is_null) {
            hashset.emplace(std::nullopt);
        }
        else {
            std::string sdata_str(sdata.buf, sdata.size);
            hashset.emplace(std::move(sdata_str));
        }
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

    char *odata = PyArray_BYTES((PyArrayObject *)res_obj);
    npy_intp ostride = PyArray_STRIDES((PyArrayObject *)res_obj)[0];
    
    for (auto it = hashset.begin(); it != hashset.end(); it++, odata += ostride) {
        npy_packed_static_string *packed_string = (npy_packed_static_string *)odata;
        if (it->has_value()) {
            std::string str = it->value();
            if (NpyString_pack(allocator, packed_string, str.c_str(), str.size()) == -1) {
                return NULL;
            }
        } else {
            if (NpyString_pack_null(allocator, packed_string) == -1) {
                return NULL;
            }
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
    {NPY_STRING, unique_string<std::string>},
    {NPY_UNICODE, unique_string<std::u32string>},
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
