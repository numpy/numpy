#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
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

template <typename T>
T read_integer(char *idata, npy_intp num_chars, npy_string_allocator *allocator) {
    return *(T *)idata;
};

template <typename T>
int write_integer(char *odata, const T &value, npy_intp itemsize, npy_string_allocator *allocator) {
    *reinterpret_cast<T *>(odata) = value;
    return 0;
};

template <typename T>
T read_string(char *idata, npy_intp num_chars, npy_string_allocator *allocator) {
    typename T::value_type *sdata = reinterpret_cast<typename T::value_type *>(idata);
    size_t byte_to_copy = std::find(sdata, sdata + num_chars, 0) - sdata;
    return T(sdata, sdata + byte_to_copy);
};

template <typename T>
int write_string(char *odata, const T &value, npy_intp itemsize, npy_string_allocator *allocator) {
    size_t byte_to_copy = value.size() * sizeof(typename T::value_type);
    memcpy(odata, value.c_str(), byte_to_copy);
    if (byte_to_copy < (size_t)itemsize) {
        memset(odata + byte_to_copy, 0, itemsize - byte_to_copy);
    }
    return 0;
};

template <typename T>
T read_vstring(char *idata, npy_intp num_chars, npy_string_allocator *allocator) {
    // https://numpy.org/doc/stable/reference/c-api/strings.html#loading-a-string
    npy_static_string sdata = {0, NULL};
    npy_packed_static_string *packed_string = (npy_packed_static_string *)idata;
    int is_null = NpyString_load(allocator, packed_string, &sdata);

    if (is_null == -1 || is_null) {
        return std::nullopt;
    }
    else {
        return std::make_optional<std::string>(sdata.buf, sdata.buf + sdata.size);
    }
};

template <typename T>
int write_vstring(char *odata, const T &value, npy_intp itemsize, npy_string_allocator *allocator) {
    npy_packed_static_string *packed_string = (npy_packed_static_string *)odata;
    if (value.has_value()) {
        std::string str = value.value();
        if (NpyString_pack(allocator, packed_string, str.c_str(), str.size()) == -1) {
            return -1;
        }
    } else {
        if (NpyString_pack_null(allocator, packed_string) == -1) {
            return -1;
        }
    }
    return 0;
};

template <typename T, auto read, auto write>
static PyObject*
unique(PyArrayObject *self)
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

    int dtype = PyArray_TYPE(self);

    // For NPY_STRING and NPY_UNICODE arrays,
    // retrieve item size and character count per entry.
    npy_intp itemsize = 0, num_chars = 0;
    if (dtype == NPY_STRING || dtype == NPY_UNICODE) {
        itemsize = PyArray_ITEMSIZE(self);
        if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::u32string>) {
            // (For Unicode, itemsize / 4 for UCS4)
            num_chars = itemsize / sizeof(typename T::value_type);
        }
    }

    // This is for NPY_VSTRING
    npy_string_allocator *allocator = nullptr;
    if (dtype == NPY_VSTRING) {
        allocator = NpyString_acquire_allocator(
            (PyArray_StringDTypeObject *)descr);
    }
    // Ensure that the allocator is properly released upon function exit.
    auto allocator_dealloc = finally([&]() {
        if (allocator != nullptr) {
            NpyString_release_allocator(allocator);
        }
    });

    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];

    std::unordered_set<T> hashset;
    // Reserve hashset capacity in advance to minimize reallocations,
    // which are particularly costly for string-based arrays.
    hashset.reserve(isize * 2);

    // Input array is one-dimensional, enabling efficient iteration using strides.
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        T value = read(idata, num_chars, allocator);
        hashset.emplace(std::move(value));
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
        if (write(odata, *it, itemsize, allocator) == -1) {
            return NULL;
        }
    }

    PyEval_RestoreThread(_save2);
    return res_obj;
}


// this map contains the functions used for each item size.
typedef std::function<PyObject *(PyArrayObject *)> function_type;
std::unordered_map<int, function_type> unique_funcs = {
    {NPY_BYTE, unique<npy_byte, read_integer<npy_byte>, write_integer<npy_byte>>},
    {NPY_UBYTE, unique<npy_ubyte, read_integer<npy_ubyte>, write_integer<npy_ubyte>>},
    {NPY_SHORT, unique<npy_short, read_integer<npy_short>, write_integer<npy_short>>},
    {NPY_USHORT, unique<npy_ushort, read_integer<npy_ushort>, write_integer<npy_ushort>>},
    {NPY_INT, unique<npy_int, read_integer<npy_int>, write_integer<npy_int>>},
    {NPY_UINT, unique<npy_uint, read_integer<npy_uint>, write_integer<npy_uint>>},
    {NPY_LONG, unique<npy_long, read_integer<npy_long>, write_integer<npy_long>>},
    {NPY_ULONG, unique<npy_ulong, read_integer<npy_ulong>, write_integer<npy_ulong>>},
    {NPY_LONGLONG, unique<npy_longlong, read_integer<npy_longlong>, write_integer<npy_longlong>>},
    {NPY_ULONGLONG, unique<npy_ulonglong, read_integer<npy_ulonglong>, write_integer<npy_ulonglong>>},
    {NPY_INT8, unique<npy_int8, read_integer<npy_int8>, write_integer<npy_int8>>},
    {NPY_INT16, unique<npy_int16, read_integer<npy_int16>, write_integer<npy_int16>>},
    {NPY_INT32, unique<npy_int32, read_integer<npy_int32>, write_integer<npy_int32>>},
    {NPY_INT64, unique<npy_int64, read_integer<npy_int64>, write_integer<npy_int64>>},
    {NPY_UINT8, unique<npy_uint8, read_integer<npy_uint8>, write_integer<npy_uint8>>},
    {NPY_UINT16, unique<npy_uint16, read_integer<npy_uint16>, write_integer<npy_uint16>>},
    {NPY_UINT32, unique<npy_uint32, read_integer<npy_uint32>, write_integer<npy_uint32>>},
    {NPY_UINT64, unique<npy_uint64, read_integer<npy_uint64>, write_integer<npy_uint64>>},
    {NPY_DATETIME, unique<npy_uint64, read_integer<npy_uint64>, write_integer<npy_uint64>>},
    {NPY_STRING, unique<std::string, read_string<std::string>, write_string<std::string>>},
    {NPY_UNICODE, unique<std::u32string, read_string<std::u32string>, write_string<std::u32string>>},
    {NPY_VSTRING, unique<std::optional<std::string>, read_vstring<std::optional<std::string>>, write_vstring<std::optional<std::string>>>},
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
