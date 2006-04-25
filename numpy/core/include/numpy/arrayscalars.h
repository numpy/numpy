#ifndef _MULTIARRAYMODULE
typedef struct {
	PyObject_HEAD
	Bool obval;
} PyBoolScalarObject;
#endif


typedef struct {
	PyObject_HEAD
	signed char obval;
} PyByteScalarObject;


typedef struct {
	PyObject_HEAD
	short obval;
} PyShortScalarObject;


typedef struct {
	PyObject_HEAD
	int obval;
} PyIntScalarObject;


typedef struct {
	PyObject_HEAD
	long obval;
} PyLongScalarObject;


typedef struct {
	PyObject_HEAD
	longlong obval;
} PyLongLongScalarObject;


typedef struct {
	PyObject_HEAD
	unsigned char obval;
} PyUByteScalarObject;


typedef struct {
	PyObject_HEAD
	unsigned short obval;
} PyUShortScalarObject;


typedef struct {
	PyObject_HEAD
	unsigned int obval;
} PyUIntScalarObject;


typedef struct {
	PyObject_HEAD
	unsigned long obval;
} PyULongScalarObject;


typedef struct {
	PyObject_HEAD
	ulonglong obval;
} PyULongLongScalarObject;


typedef struct {
	PyObject_HEAD
	float obval;
} PyFloatScalarObject;


typedef struct {
	PyObject_HEAD
	double obval;
} PyDoubleScalarObject;


typedef struct {
	PyObject_HEAD
	longdouble obval;
} PyLongDoubleScalarObject;


typedef struct {
	PyObject_HEAD
	cfloat obval;
} PyCFloatScalarObject;


typedef struct {
	PyObject_HEAD
	cdouble obval;
} PyCDoubleScalarObject;


typedef struct {
	PyObject_HEAD
	clongdouble obval;
} PyCLongDoubleScalarObject;


typedef struct {
	PyObject_HEAD
	PyObject * obval;
} PyObjectScalarObject;


typedef struct {
	PyObject_HEAD
	char obval;
} PyScalarObject;

#define PyStringScalarObject PyStringObject
#define PyUnicodeScalarObject PyUnicodeObject

typedef struct {
	PyObject_VAR_HEAD
	char *obval;
	PyArray_Descr *descr;
	int flags;
	PyObject *base;
} PyVoidScalarObject;


#define PyArrayScalar_False ((PyObject *)&_PyArrayScalar_BoolValues[0])
#define PyArrayScalar_True ((PyObject *)&_PyArrayScalar_BoolValues[1])
#define PyArrayScalar_RETURN_BOOL_FROM_LONG(i)			\
	return Py_INCREF(&_PyArrayScalar_BoolValues[((i)!=0)]),	\
		(PyObject *)&_PyArrayScalar_BoolValues[((i)!=0)]
#define PyArrayScalar_RETURN_FALSE		\
	return Py_INCREF(PyArrayScalar_False),	\
		PyArrayScalar_False
#define PyArrayScalar_RETURN_TRUE		\
	return Py_INCREF(PyArrayScalar_True),	\
		PyArrayScalar_True

#define PyArrayScalar_New(cls) \
	Py##cls##ArrType_Type.tp_alloc(&Py##cls##ArrType_Type, 0)
#define PyArrayScalar_VAL(obj, cls)		\
	((Py##cls##ScalarObject *)obj)->obval
