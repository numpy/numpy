#ifndef NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_

#ifndef _MULTIARRAYMODULE
typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject_fields;
#endif


typedef struct {
        PyObject_HEAD
        signed char obval;
} PyByteScalarObject_fields;


typedef struct {
        PyObject_HEAD
        short obval;
} PyShortScalarObject_fields;


typedef struct {
        PyObject_HEAD
        int obval;
} PyIntScalarObject_fields;


typedef struct {
        PyObject_HEAD
        long obval;
} PyLongScalarObject_fields;


typedef struct {
        PyObject_HEAD
        npy_longlong obval;
} PyLongLongScalarObject_fields;


typedef struct {
        PyObject_HEAD
        unsigned char obval;
} PyUByteScalarObject_fields;


typedef struct {
        PyObject_HEAD
        unsigned short obval;
} PyUShortScalarObject_fields;


typedef struct {
        PyObject_HEAD
        unsigned int obval;
} PyUIntScalarObject_fields;


typedef struct {
        PyObject_HEAD
        unsigned long obval;
} PyULongScalarObject_fields;


typedef struct {
        PyObject_HEAD
        npy_ulonglong obval;
} PyULongLongScalarObject_fields;


typedef struct {
        PyObject_HEAD
        npy_half obval;
} PyHalfScalarObject_fields;


typedef struct {
        PyObject_HEAD
        float obval;
} PyFloatScalarObject_fields;


typedef struct {
        PyObject_HEAD
        double obval;
} PyDoubleScalarObject_fields;


typedef struct {
        PyObject_HEAD
        npy_longdouble obval;
} PyLongDoubleScalarObject_fields;


typedef struct {
        PyObject_HEAD
        npy_cfloat obval;
} PyCFloatScalarObject_fields;


typedef struct {
        PyObject_HEAD
        npy_cdouble obval;
} PyCDoubleScalarObject_fields;


typedef struct {
        PyObject_HEAD
        npy_clongdouble obval;
} PyCLongDoubleScalarObject_fields;


typedef struct {
        PyObject_HEAD
        PyObject * obval;
} PyObjectScalarObject_fields;

typedef struct {
        PyObject_HEAD
        npy_datetime obval;
        PyArray_DatetimeMetaData obmeta;
} PyDatetimeScalarObject_fields;

typedef struct {
        PyObject_HEAD
        npy_timedelta obval;
        PyArray_DatetimeMetaData obmeta;
} PyTimedeltaScalarObject_fields;

typedef struct {
        PyObject_HEAD
        char obval;
} PyScalarObject_fields;

// PyUnicodeObject and PyBytesObject are not exposed in the limited API
#ifndef Py_LIMITED_API

#define PyStringScalarObject PyBytesObject

typedef struct {
        /* note that the PyObject_HEAD macro lives right here */
        PyUnicodeObject base;
        Py_UCS4 *obval;
    #if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
        char *buffer_fmt;
    #endif
} PyUnicodeScalarObject_fields;

#endif // Py_LIMITED_API

typedef struct {
        PyObject_VAR_HEAD
        char *obval;
#if defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD
        /* Internally use the subclass to allow accessing names/fields */
        _PyArray_LegacyDescr *descr;
#else
        PyArray_Descr *descr;
#endif
        int flags;
        PyObject *base;
    #if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
        void *_buffer_info;  /* private buffer info, tagged to allow warning */
    #endif
} PyVoidScalarObject_fields;

/* The exposed objects.
 *
 * A future NumPy release will likely make these objects
 * opaque and require accessing the fields via PyArrayScalar_VAL
 * and other acesors defined for specific types above.
 *
 */

#if defined(NPY_INTERNAL_BUILD) && (NPY_INTERNAL_BUILD)
// all these structs are opaque internally
#ifndef _MULTIARRAYMODULE
typedef struct PyBoolScalarObject PyBoolScalarObject;
#endif
typedef struct PyByteScalarObject PyByteScalarObject;
typedef struct PyShortScalarObject PyShortScalarObject;
typedef struct PyIntScalarObject PyIntScalarObject;
typedef struct PyLongScalarObject PyLongScalarObject;
typedef struct PyLongLongScalarObject PyLongLongScalarObject;
typedef struct PyUByteScalarObject PyUByteScalarObject;
typedef struct PyUShortScalarObject PyUShortScalarObject;
typedef struct PyUIntScalarObject PyUIntScalarObject;
typedef struct PyULongScalarObject PyULongScalarObject;
typedef struct PyULongLongScalarObject PyULongLongScalarObject;
typedef struct PyHalfScalarObject PyHalfScalarObject;
typedef struct PyFloatScalarObject PyFloatScalarObject;
typedef struct PyDoubleScalarObject PyDoubleScalarObject;
typedef struct PyLongDoubleScalarObject PyLongDoubleScalarObject;
typedef struct PyCFloatScalarObject PyCFloatScalarObject;
typedef struct PyCDoubleScalarObject PyCDoubleScalarObject;
typedef struct PyCLongDoubleScalarObject PyCLongDoubleScalarObject;
typedef struct PyObjectScalarObject PyObjectScalarObject;
typedef struct PyDatetimeScalarObject PyDatetimeScalarObject;
typedef struct PyTimedeltaScalarObject PyTimedeltaScalarObject;
typedef struct PyScalarObject PyScalarObject;
#ifndef Py_LIMITED_API
typedef struct PyUnicodeScalarObject PyUnicodeScalarObject;
#endif
typedef struct PyVoidScalarObject PyVoidScalarObject;
#else
#ifndef _MULTIARRAYMODULE
typedef PyBoolScalarObject_fields PyBoolScalarObject;
#endif
typedef PyByteScalarObject_fields PyByteScalarObject;
typedef PyShortScalarObject_fields PyShortScalarObject;
typedef PyIntScalarObject_fields PyIntScalarObject;
typedef PyLongScalarObject_fields PyLongScalarObject;
typedef PyLongLongScalarObject_fields PyLongLongScalarObject;
typedef PyUByteScalarObject_fields PyUByteScalarObject;
typedef PyUShortScalarObject_fields PyUShortScalarObject;
typedef PyUIntScalarObject_fields PyUIntScalarObject;
typedef PyULongScalarObject_fields PyULongScalarObject;
typedef PyULongLongScalarObject_fields PyULongLongScalarObject;
typedef PyHalfScalarObject_fields PyHalfScalarObject;
typedef PyFloatScalarObject_fields PyFloatScalarObject;
typedef PyDoubleScalarObject_fields PyDoubleScalarObject;
typedef PyLongDoubleScalarObject_fields PyLongDoubleScalarObject;
typedef PyCFloatScalarObject_fields PyCFloatScalarObject;
typedef PyCDoubleScalarObject_fields PyCDoubleScalarObject;
typedef PyCLongDoubleScalarObject_fields PyCLongDoubleScalarObject;
typedef PyObjectScalarObject_fields PyObjectScalarObject;
typedef PyDatetimeScalarObject_fields PyDatetimeScalarObject;
typedef PyTimedeltaScalarObject_fields PyTimedeltaScalarObject;
typedef PyScalarObject_fields PyScalarObject;
#ifndef Py_LIMITED_API
typedef PyUnicodeScalarObject_fields PyUnicodeScalarObject;
#endif
typedef PyVoidScalarObject_fields PyVoidScalarObject;


#endif

/* Macros
     Py<Cls><bitsize>ScalarObject
     Py<Cls><bitsize>ArrType_Type
   are defined in ndarrayobject.h
*/

#define PyArrayScalar_False ((PyObject *)(&(_PyArrayScalar_BoolValues[0])))
#define PyArrayScalar_True ((PyObject *)(&(_PyArrayScalar_BoolValues[1])))
#define PyArrayScalar_FromLong(i) \
        ((PyObject *)(&(_PyArrayScalar_BoolValues[((i)!=0)])))
#define PyArrayScalar_RETURN_BOOL_FROM_LONG(i) do {     \
        PyObject *obj = PyArrayScalar_FromLong(i);      \
        Py_INCREF(obj);                                 \
        return obj;                                     \
} while (0)
#define PyArrayScalar_RETURN_FALSE              \
        return Py_INCREF(PyArrayScalar_False),  \
                PyArrayScalar_False
#define PyArrayScalar_RETURN_TRUE               \
        return Py_INCREF(PyArrayScalar_True),   \
                PyArrayScalar_True

#define PyArrayScalar_New(cls) \
        Py##cls##ArrType_Type.tp_alloc(&Py##cls##ArrType_Type, 0)
#ifndef Py_LIMITED_API
/* For the limited API, use PyArray_ScalarAsCtype instead */
#define PyArrayScalar_VAL(obj, cls)             \
        ((Py##cls##ScalarObject_fields *)obj)->obval
#define PyArrayScalar_ASSIGN(obj, cls, val) \
        PyArrayScalar_VAL(obj, cls) = val
#endif

static inline PyVoidScalarObject_fields *
PyVoidScalar_GET_ITEM_DATA(PyVoidScalarObject *scalar)
{
    return (PyVoidScalarObject_fields *)scalar;
}

#if defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD
static inline _PyArray_LegacyDescr *
#else
static inline PyArray_Descr *
#endif
PyVoidScalar_DESCR(PyVoidScalarObject *scalar)
{
    return PyVoidScalar_GET_ITEM_DATA(scalar)->descr;
}

static inline int
PyVoidScalar_FLAGS(PyVoidScalarObject *scalar)
{
    return PyVoidScalar_GET_ITEM_DATA(scalar)->flags;
}

static inline PyObject *
PyVoidScalar_BASE(PyVoidScalarObject *scalar)
{
    return PyVoidScalar_GET_ITEM_DATA(scalar)->base;
}

static inline PyDatetimeScalarObject_fields *
PyDatetimeScalar_GET_ITEM_DATA(PyDatetimeScalarObject *scalar)
{
    return (PyDatetimeScalarObject_fields *)scalar;
}

static inline PyArray_DatetimeMetaData *
PyDatetimeScalar_OBMETA(PyDatetimeScalarObject *scalar) {
    return &(PyDatetimeScalar_GET_ITEM_DATA(scalar)->obmeta);
}

static inline PyTimedeltaScalarObject_fields *
PyTimedeltaScalar_GET_ITEM_DATA(PyTimedeltaScalarObject *scalar)
{
    return (PyTimedeltaScalarObject_fields *)scalar;
}

static inline PyArray_DatetimeMetaData *
PyTimedeltaScalar_OBMETA(PyTimedeltaScalarObject *scalar) {
    return &(PyTimedeltaScalar_GET_ITEM_DATA(scalar)->obmeta);
}

#ifndef Py_LIMITED_API

static inline PyUnicodeScalarObject_fields *
PyUnicodeScalar_GET_ITEM_DATA(PyUnicodeScalarObject *scalar)
{
    return (PyUnicodeScalarObject_fields *)scalar;
}

#if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
static inline char *
PyUnicodeScalar_BUFFER_FMT(PyUnicodeScalarObject *scalar)
{
    return PyUnicodeScalar_GET_ITEM_DATA(scalar)->buffer_fmt;
}

#endif

#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_ */
