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

template <typename T>
class IReadWrite {
public:
    virtual T read(char *idata) const = 0;
    virtual int write(char *odata, const T &value) const = 0;
    virtual ~IReadWrite() = default;
};

template <typename T>
class IntegerReadWrite : public IReadWrite<T> {
public:
    IntegerReadWrite(PyArray_Descr *descr) {}

    T read(char *idata) const override {
        return *(T *)idata;
    }

    int write(char *odata, const T &value) const override {
        *(T *)(odata) = value;
        return 0;
    }
};

// T is a string type (std::string or std::u32string)
template <typename T>
class StringReadWrite : public IReadWrite<T> {
private:
    npy_intp itemsize_;
    npy_intp num_chars_;

public:
    StringReadWrite(PyArray_Descr *descr) {
        itemsize_ = descr->elsize;
        num_chars_ = itemsize_ / sizeof(typename T::value_type);
    }

    T read(char *idata) const override {
        typename T::value_type *sdata = reinterpret_cast<typename T::value_type *>(idata);
        size_t byte_to_copy = std::find(sdata, sdata + num_chars_, 0) - sdata;
        return T(sdata, sdata + byte_to_copy);
    }

    int write(char *odata, const T &value) const override {
        size_t byte_to_copy = value.size() * sizeof(typename T::value_type);
        memcpy(odata, value.c_str(), byte_to_copy);
        if (byte_to_copy < (size_t)itemsize_) {
            memset(odata + byte_to_copy, 0, itemsize_ - byte_to_copy);
        }
        return 0;
    }
};

class VStringReadWrite : public IReadWrite<std::optional<std::string>> {
private:
    npy_string_allocator *allocator_;

public:
    VStringReadWrite(PyArray_Descr *descr) {
        allocator_ = NpyString_acquire_allocator(
            (PyArray_StringDTypeObject *)descr);
    }

    ~VStringReadWrite() {
        NpyString_release_allocator(allocator_);
    }

    std::optional<std::string> read(char *idata) const override {
        // https://numpy.org/doc/stable/reference/c-api/strings.html#loading-a-string
        npy_static_string sdata = {0, nullptr};
        npy_packed_static_string *packed_string = (npy_packed_static_string *)(idata);
        int is_null = NpyString_load(allocator_, packed_string, &sdata);

        if (is_null == -1 || is_null) {
            return std::nullopt;
        }
        else {
            return std::make_optional<std::string>(sdata.buf, sdata.buf + sdata.size);
        }
    }

    int write(char *odata, const std::optional<std::string> &value) const override {
        npy_packed_static_string *packed_string = (npy_packed_static_string *)(odata);
        if (value.has_value()) {
            const std::string &str = value.value();
            return NpyString_pack(allocator_, packed_string, str.c_str(), str.size());
        } else {
            return NpyString_pack_null(allocator_, packed_string);
        }
    }
};

template <typename T, typename ReadWrite>
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

    // ReadWrite is a template class that provides read and write methods
    ReadWrite read_write = ReadWrite(descr);

    npy_intp isize = PyArray_SIZE(self);
    char *idata = PyArray_BYTES(self);
    npy_intp istride = PyArray_STRIDES(self)[0];

    std::unordered_set<T> hashset;
    // Reserve hashset capacity in advance to minimize reallocations,
    // which are particularly costly for string-based arrays.
    hashset.reserve(isize * 2);

    // Input array is one-dimensional, enabling efficient iteration using strides.
    for (npy_intp i = 0; i < isize; i++, idata += istride) {
        T value = read_write.read(idata);
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
        if (read_write.write(odata, *it) == -1) {
            return NULL;
        }
    }

    PyEval_RestoreThread(_save2);
    return res_obj;
}


// this map contains the functions used for each item size.
typedef std::function<PyObject *(PyArrayObject *)> function_type;
std::unordered_map<int, function_type> unique_funcs = {
    {NPY_BYTE, unique<npy_byte, IntegerReadWrite<npy_byte>>},
    {NPY_UBYTE, unique<npy_ubyte, IntegerReadWrite<npy_ubyte>>},
    {NPY_SHORT, unique<npy_short, IntegerReadWrite<npy_short>>},
    {NPY_USHORT, unique<npy_ushort, IntegerReadWrite<npy_ushort>>},
    {NPY_INT, unique<npy_int, IntegerReadWrite<npy_int>>},
    {NPY_UINT, unique<npy_uint, IntegerReadWrite<npy_uint>>},
    {NPY_LONG, unique<npy_long, IntegerReadWrite<npy_long>>},
    {NPY_ULONG, unique<npy_ulong, IntegerReadWrite<npy_ulong>>},
    {NPY_LONGLONG, unique<npy_longlong, IntegerReadWrite<npy_longlong>>},
    {NPY_ULONGLONG, unique<npy_ulonglong, IntegerReadWrite<npy_ulonglong>>},
    {NPY_INT8, unique<npy_int8, IntegerReadWrite<npy_int8>>},
    {NPY_INT16, unique<npy_int16, IntegerReadWrite<npy_int16>>},
    {NPY_INT32, unique<npy_int32, IntegerReadWrite<npy_int32>>},
    {NPY_INT64, unique<npy_int64, IntegerReadWrite<npy_int64>>},
    {NPY_UINT8, unique<npy_uint8, IntegerReadWrite<npy_uint8>>},
    {NPY_UINT16, unique<npy_uint16, IntegerReadWrite<npy_uint16>>},
    {NPY_UINT32, unique<npy_uint32, IntegerReadWrite<npy_uint32>>},
    {NPY_UINT64, unique<npy_uint64, IntegerReadWrite<npy_uint64>>},
    {NPY_DATETIME, unique<npy_uint64, IntegerReadWrite<npy_uint64>>},
    {NPY_STRING, unique<std::string, StringReadWrite<std::string>>},
    {NPY_UNICODE, unique<std::u32string, StringReadWrite<std::u32string>>},
    {NPY_VSTRING, unique<std::optional<std::string>, VStringReadWrite>},
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
