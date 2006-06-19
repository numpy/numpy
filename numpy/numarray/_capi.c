
#include <Python.h>

#define _libnumarray_MODULE

#include "Python.h"
#include "pystate.h"
#include "libnumarray.h"
#include <stdio.h>
#include <float.h>

static PyObject *pNDArrayModule;
static PyObject *pNDArrayMDict;
static PyObject *pNDArrayClass;

static PyObject *pNumArrayModule;
static PyObject *pNumArrayMDict;
static PyObject *pNumArrayClass;
static PyObject *pNumArrayNewFunc;
static PyObject *pNumArrayArrayFunc;

static PyObject *pNumericTypesModule;
static PyObject *pNumericTypesMDict;
static PyObject *pNumericTypeClass;
static PyObject *pNumericTypesTDict;

static PyObject *pUfuncModule;
static PyObject *pUfuncMDict;
static PyObject *pUfuncClass;

static PyObject *pCfuncClass;

static PyObject *pConverterModule;
static PyObject *pConverterMDict;
static PyObject *pConverterClass;

static PyObject *pOperatorModule;
static PyObject *pOperatorMDict;
static PyObject *pOperatorClass;

static PyObject *pNewMemoryFunc;
static PyObject *pHandleErrorFunc;

static PyObject *pNumType[nNumarrayType];

static PyTypeObject CfuncType;

static PyObject *pEmptyDict;
static PyObject *pEmptyTuple;

static PyObject *dealloc_list;  /* list of global references to DECREF at
				 unload time, i.e.  when the module dict is
				 destructed. */

enum {
	BOOL_SCALAR,
	INT_SCALAR,
	LONG_SCALAR,
	FLOAT_SCALAR,
	COMPLEX_SCALAR
};

static int initialized = 0;

/* custom init function generally unuseable due to circular references */
static int 
libnumarray_init(void) 
{
	PyObject *m, *d;
	initialized = 0;
	if (!(dealloc_list = PyList_New(0)))
		return -1;
	if (!(m = PyImport_ImportModule("numarray.libnumarray")))
		return -1;
	d = PyModule_GetDict(m);
	if (PyDict_SetItemString(d, "_dealloc_list", dealloc_list) < 0)
		return -1;
	Py_DECREF(dealloc_list);
	Py_DECREF(m);		       				       
	return 0; 
}

static PyObject *
init_module(char *modulename, PyObject **pMDict)
{
	PyObject *pModule = PyImport_ImportModule(modulename);
	if (!pModule) return NULL;
	PyList_Append(dealloc_list, pModule);
	Py_DECREF(pModule);
	*pMDict = PyModule_GetDict(pModule);
	PyList_Append(dealloc_list, *pMDict);
	return pModule;
}

static PyObject *
init_object(char *objectname, PyObject *pMDict)
{
	PyObject *object = PyDict_GetItemString(pMDict, objectname);
	if (!object) return NULL;
	PyList_Append(dealloc_list, object);
	return object;
}

static int
init_module_class(char *modulename, PyObject **pModule, 
		  PyObject **pMDict, 
		  char *classname,  PyObject **pClass)
{
	if ((*pModule = init_module(modulename, pMDict)))
		*pClass = init_object(classname, *pMDict);
	else
		return -1;
	return 0;
}


extern void *libnumarray_API[];

 
static int 
deferred_libnumarray_init(void)
{
	int i;

	if (initialized) return 0;

	import_libtc();

	if (init_module_class("numarray.generic", &pNDArrayModule, 
			      &pNDArrayMDict,
			      "NDArray", &pNDArrayClass) < 0)
		goto _fail;
	
	if (init_module_class("numarray", &pNumArrayModule, 
			      &pNumArrayMDict,
			      "NumArray", &pNumArrayClass) < 0)
		goto _fail;
	
	if (init_module_class("numarray.numerictypes", &pNumericTypesModule, 
			      &pNumericTypesMDict, 
			      "NumericType", &pNumericTypeClass) < 0)
		goto _fail;
	
	if (init_module_class("numarray._ufunc", &pUfuncModule, 
			      &pUfuncMDict,
			      "_ufunc", &pUfuncClass) < 0)
		goto _fail;

	pCfuncClass = (PyObject *) &CfuncType;
	Py_INCREF(pCfuncClass);
	
	if (init_module_class("numarray._operator", &pOperatorModule, 
			      &pOperatorMDict,
			      "_operator", &pOperatorClass) < 0)
		goto _fail;
	
	if (init_module_class("numarray._converter", &pConverterModule, 
			      &pConverterMDict,
			      "_converter", &pConverterClass) < 0)
		goto _fail;

	if (!(pNumArrayNewFunc = PyObject_GetAttrString(
		      pNumArrayClass, "__new__")))
		goto _fail;
	
	if (!(pNumArrayArrayFunc = init_object( "array", pNumArrayMDict)))
		goto _fail;

	if (!(pNumericTypesTDict = init_object( "typeDict", pNumericTypesMDict)))
		goto _fail;

	pNewMemoryFunc = NA_initModuleGlobal("numarray.memory","new_memory");
	if (!pNewMemoryFunc) goto _fail;

	pHandleErrorFunc = 
		NA_initModuleGlobal("numarray.ufunc", "handleError");
	if (!pHandleErrorFunc) goto _fail;
		
	
	/* Set up table of type objects */
	for(i=0; i<ELEM(pNumType); i++) {
		PyObject *typeobj = init_object(NA_typeNoToName(i),
						pNumericTypesTDict);
		if (!typeobj) return -1;
		if (typeobj) {
			Py_INCREF(typeobj);
			pNumType[i] = typeobj;
		} else {
			pNumType[i] = NULL;
		}
	}

	/* Set up _get/_set descriptor hooks for numerical types */
	for(i=0; i<nNumarrayType; i++) {
		PyArray_Descr *ptr;
		switch(i) {
		case tAny: case tObject: 
			break;
		default: 
			ptr = NA_DescrFromType( i );
			if (!ptr) {
				PyErr_Format(PyExc_RuntimeError, 
					     "error initializing array descriptors");
				goto _fail;
			}
			ptr->_get = NA_getPythonScalar;
			ptr->_set = NA_setFromPythonScalar;
			break;
		}
	}

	libnumarray_API[ 0 ]  = (void *) pNumArrayClass;

	pEmptyDict = PyDict_New();
	if (!pEmptyDict) goto _fail;

	pEmptyTuple = PyTuple_New(0);
	if (!pEmptyTuple) goto _fail;

	/* _exit: */
	initialized = 1;
	return 0;

  _fail:
	initialized = 0;
	return -1;
}


/* Finalize this module. */
void 
fini_module_class(PyObject *module, PyObject *mdict, PyObject *class)
{
	Py_DECREF(module);
	Py_DECREF(mdict);
	Py_DECREF(class);
}

static void
NA_Done(void)
{
	int i;

	fini_module_class(pNDArrayModule, pNDArrayMDict, pNDArrayClass);

	fini_module_class(pNumArrayModule, pNumArrayMDict, pNumArrayClass);
	Py_DECREF(pNumArrayArrayFunc);

	fini_module_class(pOperatorModule, pOperatorMDict, pOperatorClass);

	fini_module_class(pConverterModule, pConverterMDict, pConverterClass);

	fini_module_class(pUfuncModule, pUfuncMDict, pUfuncClass);
	Py_DECREF(pCfuncClass);
	
	fini_module_class(pNumericTypesModule, pNumericTypesMDict, 
			  pNumericTypeClass);
	Py_DECREF(pNumericTypesTDict);

	for(i=0; i<ELEM(pNumType); i++) {
		Py_DECREF(pNumType[i]);
	}
}

#ifdef MS_WIN32
#pragma warning(once : 4244)
#endif

#define ELEM(x) (sizeof(x)/sizeof(x[0]))

typedef struct
{
	char *name;
	int typeno;
} NumarrayTypeNameMapping;

static PyArray_Descr descriptors[ ] = {
	{ tAny,       0,                 '*'}, 
	
	{ tBool,      sizeof(Bool),      '?'},

	{ tInt8,      sizeof(Int8),      '1'},
	{ tUInt8,     sizeof(UInt8),     'b'},

	{ tInt16,     sizeof(Int16),     's'},
	{ tUInt16,    sizeof(UInt16),    'w'},

	{ tInt32,     sizeof(Int32),     'i'},
	{ tUInt32,    sizeof(UInt32),    'u'},

	{ tInt64,     sizeof(Int64),     'N'},
	{ tUInt64,    sizeof(UInt64),    'U'},

	{ tFloat32,   sizeof(Float32),   'f'},
	{ tFloat64,   sizeof(Float64),   'd'},

	{ tComplex32, sizeof(Complex32), 'F'},
	{ tComplex64, sizeof(Complex64), 'D'}
};

static PyArray_Descr *
NA_DescrFromType(int type)
{
	if ((type >= tAny) && (type <= tComplex64)) {
		return &descriptors[ type ];
	} else {
		int i;
		for(i=0; i<ELEM(descriptors); i++)
			if (descriptors[i].type == type)
				return &descriptors[i];
	}
	PyErr_Format(
		PyExc_TypeError, 
		"NA_DescrFromType: unknown type: %d", type);
	return NULL;
}

static NumarrayTypeNameMapping NumarrayTypeNameMap[] = {
	{"Any", tAny},
	{"Bool", tBool},
	{"Int8", tInt8},
	{"UInt8", tUInt8},
	{"Int16", tInt16},
	{"UInt16", tUInt16},
	{"Int32", tInt32},
	{"UInt32", tUInt32},
	{"Int64", tInt64},
	{"UInt64", tUInt64},
	{"Float32", tFloat32},
	{"Float64", tFloat64},
	{"Complex32", tComplex32},
	{"Complex64", tComplex64},
	{"Object", tObject},
	{"Long", tLong},
};

typedef struct 
{
	NumarrayType type_num;
	char suffix[5];
	int  itemsize;
} scipy_typestr;

static scipy_typestr scipy_descriptors[ ] = {
	{ tAny,       ""}, 
	
	{ tBool,      "b1", 1},

	{ tInt8,      "i1", 1},
	{ tUInt8,     "u1", 1},

	{ tInt16,     "i2", 2},
	{ tUInt16,    "u2", 2},

	{ tInt32,     "i4", 4},
	{ tUInt32,    "u4", 4},

	{ tInt64,     "i8", 8},
	{ tUInt64,    "u8", 8},

	{ tFloat32,   "f4", 4},
	{ tFloat64,   "f8", 8},

	{ tComplex32, "c8", 8},
	{ tComplex64, "c16", 16}
};

static PyObject *
setTypeException(int type)
{
	/* Check if it is a printable character */
	if ((type >= 32) && (type <= 126)) 
		PyErr_Format(_Error, 
			     "Type object lookup returned"
			     " NULL for type \'%c\'", type);
	else
		PyErr_Format(_Error,
			     "Type object lookup returned"
			     " NULL for type %d", type);
	return NULL;
}

static PyObject *
getTypeObject(NumarrayType type) 
{
        char strcharcode[2];
        PyObject *typeobj;
	
	if (deferred_libnumarray_init() < 0) return NULL;

        if ((type >= tAny) && (type <= tObject)) {
		return pNumType[type];
	} else  {
	      /* Test if it is a Numeric charcode */
		strcharcode[0] = type; strcharcode[1] = 0;
		typeobj = PyDict_GetItemString(
			pNumericTypesTDict, strcharcode);
		return typeobj ? typeobj : setTypeException(type);
	}
}

static PyObject *
NA_getType( PyObject *type)
{
	PyObject *typeobj = NULL;
	if (deferred_libnumarray_init() < 0) goto _exit;
	if (!type) goto _exit;
	if (PyObject_IsInstance(type, pNumericTypeClass)) {
		Py_INCREF(type);
		typeobj = type;
		goto _exit;
	}
	if ((typeobj = PyDict_GetItem(pNumericTypesTDict, type))) {
		Py_INCREF(typeobj);
	} else {
	       PyErr_Format(PyExc_ValueError, "NA_getType: unknown type.");
	}
  _exit:
	return typeobj;
}

/* Look up the NumarrayType which corresponds to typename */

static int 
NA_nameToTypeNo(char *typename)
{
	int i;
	for(i=0; i<ELEM(NumarrayTypeNameMap); i++)
		if (!strcmp(typename, NumarrayTypeNameMap[i].name))
			return NumarrayTypeNameMap[i].typeno;
	return -1;
}

/* Convert NumarrayType 'typeno' into the string of the type's name. */

static char *
NA_typeNoToName(int typeno) 
{
	int i;
	PyObject *typeObj;
	int typeno2;

	for(i=0; i<ELEM(NumarrayTypeNameMap); i++)
		if (typeno == NumarrayTypeNameMap[i].typeno)
			return NumarrayTypeNameMap[i].name;

	/* Handle Numeric typecodes */
	typeObj = NA_typeNoToTypeObject(typeno);
	if (!typeObj) return 0;
	typeno2 = NA_typeObjectToTypeNo(typeObj);
	Py_DECREF(typeObj);

	return NA_typeNoToName(typeno2);
}

static PyObject *
NA_typeNoToTypeObject(int typeno)
{
	PyObject *o;
	o = getTypeObject(typeno);	
	if (o) Py_INCREF(o);
	return o;
}

static int 
NA_typeObjectToTypeNo(PyObject *typeObj)
{
	int i;
	if (deferred_libnumarray_init() < 0) return -1;
	for(i=0; i<ELEM(pNumType); i++)
		if (pNumType[i] == typeObj)
			break;
	if (i == ELEM(pNumType)) i = -1;
	return i;
}

static NumarrayType
_scipy_typekind_to_typeNo(char typekind, int itemsize)
{
	int i;
	for(i=0; i<ELEM(scipy_descriptors); i++) {
		scipy_typestr *ts = &scipy_descriptors[i];
		if ((typekind == ts->suffix[0]) && 
		    (itemsize == ts->itemsize))
			return i;
	}
	PyErr_Format(PyExc_TypeError,
		     "Unknown __array_struct__ typekind");
	return  -1;
}

static int
NA_scipy_typestr(NumarrayType t, int byteorder, char *typestr)
{
	int i;
	if (byteorder) 
		strcpy(typestr, ">");
	else
		strcpy(typestr, "<");
	for(i=0; i<ELEM(scipy_descriptors); i++) {
		scipy_typestr *ts = &scipy_descriptors[i];
		if (ts->type_num == t) {
			strncat(typestr, ts->suffix, 4);
			return 0;
		}
	}
	return -1;
}

static long 
NA_isIntegerSequence(PyObject *sequence)
{
	PyObject *o;
	long i, size, isInt = 1;
	if (!sequence) { 
		isInt = -1; 
		goto _exit; 
	}
	if (!PySequence_Check(sequence)) {
		isInt = 0;
		goto _exit;
	}
	if ((size = PySequence_Length(sequence)) < 0) {
		isInt = -1;
		goto _exit;
	}
	for(i=0; i<size; i++) {
		o = PySequence_GetItem(sequence, i);
		if (!PyInt_Check(o) && !PyLong_Check(o)) {
			isInt = 0;
			Py_XDECREF(o);
			goto _exit;
		}
		Py_XDECREF(o);
	}
  _exit:
	return isInt;
}

static PyObject *
NA_intTupleFromMaybeLongs(int len, maybelong *Longs)
{
	int i;
	PyObject *intTuple = PyTuple_New(len);
	if (!intTuple) goto _exit;
	for(i=0; i<len; i++) {
		PyObject *o = PyInt_FromLong(Longs[i]);
		if (!o) {
			Py_DECREF(intTuple);
			intTuple = NULL;
			goto _exit;
		}
		PyTuple_SET_ITEM(intTuple, i, o);
	}
  _exit:
	return intTuple;
}

static long
NA_maybeLongsFromIntTuple(int len, maybelong *arr, PyObject *sequence)
{
	long i, size = -1;
	if (!PySequence_Check(sequence)) {
		PyErr_Format(PyExc_TypeError, 
			     "NA_maybeLongsFromIntTuple: must be a sequence of integers.");
		goto _exit;
	}
	size = PySequence_Length(sequence);
	if (size < 0) {
		PyErr_Format(PyExc_RuntimeError, 
			     "NA_maybeLongsFromIntTuple: error getting sequence length.");
		size = -1;
		goto _exit;
	}
	if (size > len) {
		PyErr_Format(PyExc_ValueError, 
			     "NA_maybeLongsFromIntTuple: sequence is too long");
		size = -1;
		goto _exit;
	}
	for(i=0; i<size; i++) {
		PyObject *o = PySequence_GetItem(sequence, i);
		long value;
		if (!o || !(PyInt_Check(o) || PyLong_Check(o))) {
			PyErr_Format(PyExc_TypeError, 
				     "NA_maybeLongsFromIntTuple: non-integer in sequence.");
			Py_XDECREF(o);
			size = -1;
			goto _exit;
		}
		arr[i] = value = PyInt_AsLong(o);
		if (arr[i] != value) {
			PyErr_Format(PyExc_ValueError, 
				     "NA_maybeLongsFromIntTuple: integer value too large: %ld",
				     value);
			size = -1;
			goto _exit;
		}
		if (PyErr_Occurred()) {
			Py_DECREF(o);
			size = -1;
			goto _exit;
		}
		Py_DECREF(o);
	}
  _exit:
	return size;
}

static int
NA_intTupleProduct(PyObject  *shape, long *prod)
{
	int i, nshape, rval = -1;

	if (!PySequence_Check(shape)) {
		PyErr_Format(PyExc_TypeError, 
		     "NA_intSequenceProduct: object is not a sequence.");
		goto _exit;
	}
	nshape = PySequence_Size(shape);
	
	for(i=0, *prod=1;  i<nshape; i++) {
		PyObject *obj = PySequence_GetItem(shape, i);
		if (!obj || !(PyInt_Check(obj) || PyLong_Check(obj))) {
			PyErr_Format(PyExc_TypeError, 
			     "NA_intTupleProduct: non-integer in shape.");
			Py_XDECREF(obj);
			goto _exit;
		}
		*prod *= PyInt_AsLong(obj);
		Py_DECREF(obj);
		if (PyErr_Occurred())
			goto _exit;
	}
	rval = 0;
  _exit:
	return rval;
}

/* NA_updateDataPtr updates the "working" data buffer pointer from the array's
buffer object.  Since objects which meet the buffer API can potentially
relocate their data as a result of executing arbitrary Python code
(i.e. array.resize), NA_updateDataPtr needs to be called each time control
flow returns to C, prior to accessing and numarray data.

_data  points to an object which must meet the buffer API
data   points to the array data gotten from _data via the buffer API

*/

static PyArrayObject *
NA_updateDataPtr(PyArrayObject *me)
{
	if (!me) return me;

	if (me->_data != Py_None) {

		if (getReadBufferDataPtr (me->_data, 
					  (void **) &me->data) < 0) {
			return (PyArrayObject *) PyErr_Format(
				_Error, 
				"NA_updateDataPtr: error getting read buffer data ptr");
		}
		if (isBufferWriteable( me->_data ))
			me->flags |= WRITABLE;
		else
			me->flags &= ~WRITABLE;
	} else {
		me->data = NULL;
	}

	me->data += me->byteoffset;

	return me;
}

/* Count the number of elements in a 1D static array. */
#define ELEM(x) (sizeof(x)/sizeof(x[0]))

static int 
NA_ByteOrder(void)
{
	unsigned long byteorder_test;
	byteorder_test = 1;
	if (*((char *) &byteorder_test))
		return NUM_LITTLE_ENDIAN;
	else
		return NUM_BIG_ENDIAN;
}

/* Create a new numarray specifying all attribute values and using an object which
   meets the buffer API to store the array data.
*/
static void
_stridesFromShape(PyArrayObject *self)
{
    int i;
    if (self->nd > 0) {
	for(i=0; i<self->nd; i++)
	    self->strides[i] = self->bytestride;
	for(i=self->nd-2; i>=0; i--)
	    self->strides[i] = 
		self->strides[i+1]*self->dimensions[i+1];
	self->nstrides = self->nd;
    } else 
	self->nstrides = 0;
}

 
static PyArrayObject *
NA_NewAllFromBuffer(int ndim, maybelong *shape, NumarrayType type,
		    PyObject *bufferObject, maybelong byteoffset, maybelong bytestride,
		    int byteorder, int aligned, int writeable)
{
	PyObject *typeObject;
	PyArrayObject *self = NULL;
	long i;

	if (deferred_libnumarray_init() < 0) goto _fail;

	if (type == tAny)
		type = tDefault;
	
	if (ndim > MAXDIM) goto _fail;


	{
		PyTypeObject *typ = (PyTypeObject *) pNumArrayClass;
		self = (PyArrayObject *) typ->tp_new(
			typ, pEmptyTuple, pEmptyDict);
		if (!self) goto _fail;
	}

	typeObject = getTypeObject(type);
        if (!typeObject) {
		setTypeException(type);
		goto _fail;
	}
	if (!(self->descr = NA_DescrFromType(type))) {
		goto _fail;
	}

	self->nd = self->nstrides = ndim;
	for(i=0; i<ndim; i++) {
		self->dimensions[i] = shape[i];
	}
	if (bytestride == 0)
		self->bytestride = self->descr->elsize;
	else
		self->bytestride = bytestride;
	_stridesFromShape(self);

	self->byteoffset = byteoffset;
	self->byteorder = byteorder;
	self->itemsize = self->descr->elsize;

	Py_XDECREF(self->_data);
	if ((bufferObject == Py_None) || (bufferObject == NULL)) {
		long size = self->descr->elsize;
		for(i=0; i<self->nd; i++) {
			size *= self->dimensions[i];
		}
		self->_data = PyObject_CallFunction(
			pNewMemoryFunc, "(l)", size);
		if (!self->_data) goto _fail;
	} else {
		self->_data = bufferObject;
		Py_INCREF(self->_data);
	}

	if (!NA_updateDataPtr(self))
		goto _fail;
	NA_updateStatus(self);
	return self;

  _fail:
	Py_XDECREF(self);
	return NULL;
}

/* Create a new numarray specifying all attribute values but with a C-array as buffer
   which will be copied to a Python buffer. 
*/
static PyArrayObject *
NA_NewAll(int ndim, maybelong *shape, NumarrayType type, 
	  void *buffer, maybelong byteoffset, maybelong bytestride, 
	  int byteorder, int aligned, int writeable)
{
	PyArrayObject *result = NA_NewAllFromBuffer(
		ndim, shape, type, Py_None, byteoffset, bytestride,
		byteorder, aligned, writeable);
	
	if (result) {
		if (!NA_NumArrayCheck((PyObject *) result)) {
		       PyErr_Format( PyExc_TypeError,
				     "NA_NewAll: non-NumArray result");
		       result = NULL;
		} else {
			if (buffer) {
				memcpy(result->data, buffer, NA_NBYTES(result));
			} else {
				memset(result->data, 0, NA_NBYTES(result));
			}
		}
	}
	return  result;
}

static PyArrayObject *
NA_NewAllStrides(int ndim, maybelong *shape, maybelong *strides, 
		 NumarrayType type, void *buffer, maybelong byteoffset, 
		 int byteorder, int aligned, int writeable)
{
	int i;
	PyArrayObject *result = NA_NewAll(ndim, shape, type, buffer, 
					  byteoffset, 0,
					  byteorder, aligned, writeable);
	for(i=0; i<ndim; i++)
		result->strides[i] = strides[i];
	result->nstrides = ndim;
	return result;
}

static PyArrayObject *
NA_FromDimsStridesDescrAndData(int nd, maybelong *d, maybelong *s, PyArray_Descr *descr, char *data) 
{
	maybelong i, nelements, breadth, bsize, boffset, dimensions[MAXDIM], strides[MAXDIM];
	PyArrayObject *a;
	PyObject *buf;
	
	if (!descr) return NULL;
	
	if (nd < 0) {
		PyErr_SetString(PyExc_ValueError, 
				"number of dimensions must be >= 0");
		return NULL;
	}
	
	if (nd > MAXDIM) {
		return (PyArrayObject *) PyErr_Format(PyExc_ValueError,
				    "too many dimensions: %d", nd);
	}

	if (!s) { /* no strides specified so assume minimal contiguous array 
		     and compute */
		if (nd) {
			for(i=0; i<nd; i++)
				strides[i] = descr->elsize;
			for(i=nd-2; i>=0; i--)
				strides[i] = strides[i+1]*d[i+1];
		}
	} else {
		for(i=0; i<nd; i++)
			strides[i] = s[i];
	}

	bsize = descr->elsize;   /* find buffer size implied by array 
				    dimensions and strides */
	boffset = 0;
	for(i=0; i<nd; i++) {
		breadth = llabs(strides[i]) * d[i];
		bsize = MAX(bsize, breadth);
		if (strides[i] < 0)
			boffset += llabs(strides[i]) * (d[i]-1);
	}
	
	nelements = 1;
	for(i=0; i<nd; i++) {
		dimensions[i] = d[i];
		nelements *= d[i];
	}
	
	if (data) {
		buf = PyBuffer_FromReadWriteMemory(data-boffset, bsize);
		if (!buf) return NULL;
	} else {
		buf = Py_None;
	}
	
	a = NA_NewAllFromBuffer( nd, dimensions, descr->type_num, buf, 
				 boffset, descr->elsize, NA_ByteOrder(), 1, 1);
	if (!a) return NULL;
	
	for(i=0; i<nd; i++)
		a->strides[i] = strides[i];

	if (!data && !s) {
		memset(a->data, 0, bsize);
	}

	NA_updateStatus(a);

	return a;
}

static PyArrayObject *
NA_FromDimsTypeAndData(int nd, maybelong *d, int type, char *data) 
{
    PyArray_Descr *descr = NA_DescrFromType(type);
    return NA_FromDimsStridesDescrAndData(nd, d, NULL, descr, data);
}

static PyArrayObject *
NA_FromDimsStridesTypeAndData(int nd, maybelong *shape, maybelong *strides, 
			     int type, char *data) 
{
    PyArray_Descr *descr = NA_DescrFromType(type);
    return NA_FromDimsStridesDescrAndData(nd, shape, strides, descr, data);
}

/* Create a new numarray which is initially a C_array, or which
references a C_array: aligned, !byteswapped, contiguous, ... 
Call with buffer==NULL to allocate storage.
*/
static PyArrayObject *
NA_vNewArray(void *buffer, NumarrayType type, int ndim, maybelong *shape)
{
	return (PyArrayObject *) NA_NewAll(ndim, shape, type, buffer, 0, 0, 
					   NA_ByteOrder(), 1, 1);
}

static PyArrayObject *
NA_NewArray(void *buffer, NumarrayType type, int ndim, ...)
{
	int i;
	maybelong shape[MAXDIM];
	va_list ap;
	va_start(ap, ndim);
	for(i=0; i<ndim; i++)
		shape[i] = va_arg(ap, int);  /* literals will still be ints */
	va_end(ap);
	return NA_vNewArray(buffer, type, ndim, shape);
}

/* Original deprecated versions of new array and empty array */
static PyArrayObject *
NA_New(void *buffer, NumarrayType type, int ndim, ...)
{
	int i;
	maybelong shape[MAXDIM];
	va_list ap;
	va_start(ap, ndim);
	for(i=0; i<ndim; i++)
		shape[i] = va_arg(ap, int);
	va_end(ap);
	return NA_NewAll(ndim, shape, type, buffer, 0, 0, 
			 NA_ByteOrder(), 1, 1);
}

static PyArrayObject *
NA_Empty(int ndim, maybelong *shape, NumarrayType type)
{
	return NA_NewAll(ndim, shape, type, NULL, 0, 0, 
			 NA_ByteOrder(), 1, 1);
}


/* getArray creates a new array of type 't' from the given array 'a'
using the specified 'method', probably 'new' or 'astype'. */

static PyArrayObject *
getArray(PyArrayObject *a, NumarrayType t, char *method)
{
	char *name;

	if (deferred_libnumarray_init() < 0) return NULL;

	if (t == tAny)
		t = a->descr->type_num;
	name = NA_typeNoToName(t);
	if (!name) return (PyArrayObject *) setTypeException(t);
	return (PyArrayObject *) 
		PyObject_CallMethod((PyObject *) a, method, "s", name);
}

static int 
getShape(PyObject *a, maybelong *shape, int dims)
{
	long slen;

	if (PyString_Check(a)) {
		PyErr_Format(PyExc_TypeError,
			     "getShape: numerical sequences can't contain strings.");
		return -1;
	}

	if (!PySequence_Check(a) || 
	    (NA_NDArrayCheck(a) && (PyArray(a)->nd == 0)))
		return dims;
	slen = PySequence_Length(a);
	if (slen < 0) {
		PyErr_Format(_Error,
			     "getShape: couldn't get sequence length.");
		return -1;
	}
	if (!slen) {
		*shape = 0;
		return dims+1;
	} else if (dims < MAXDIM) {
		PyObject *item0 = PySequence_GetItem(a, 0);
		if (item0) {
			*shape = PySequence_Length(a);
			dims = getShape(item0, ++shape, dims+1);
			Py_DECREF(item0);
		} else {
			PyErr_Format(_Error, 
				     "getShape: couldn't get sequence item.");
			return -1;
		}
	} else {		  
		PyErr_Format(_Error, 
		 "getShape: sequence object nested more than MAXDIM deep.");
		return -1;
	}
	return dims;
}

typedef enum {
  NOTHING,
  NUMBER,
  SEQUENCE
} SequenceConstraint;

static int 
setArrayFromSequence(PyArrayObject *a, PyObject *s, int dim, long offset)
{
	SequenceConstraint mustbe = NOTHING;
	int i, seqlen=-1, slen = PySequence_Length(s);

	if (dim > a->nd) {
		PyErr_Format(PyExc_ValueError, 
			     "setArrayFromSequence: sequence/array dimensions mismatch.");
		return -1;
	}

	if (slen != a->dimensions[dim]) {
		PyErr_Format(PyExc_ValueError,
			     "setArrayFromSequence: sequence/array shape mismatch.");
		return -1;
	}

	for(i=0; i<slen; i++) {
		PyObject *o = PySequence_GetItem(s, i);
		if (!o) {
			PyErr_SetString(_Error, 
 			   "setArrayFromSequence: Can't get a sequence item");
			return -1;
		} else if ((NA_isPythonScalar(o) || 
			    (NA_NumArrayCheck(o) && PyArray(o)->nd == 0)) && 
			   ((mustbe == NOTHING) || (mustbe == NUMBER))) {
			if (NA_setFromPythonScalar(a, offset, o) < 0)
				return -2;
			mustbe = NUMBER;
		} else if (PyString_Check(o)) {
			PyErr_SetString( PyExc_ValueError,
			"setArrayFromSequence: strings can't define numeric numarray.");
			return -3;
		} else if (PySequence_Check(o)) {
			if ((mustbe == NOTHING) || (mustbe == SEQUENCE)) {
				if (mustbe == NOTHING) {
					mustbe = SEQUENCE;
					seqlen = PySequence_Length(o);
				} else if (PySequence_Length(o) != seqlen) {
					PyErr_SetString(
						PyExc_ValueError, 
						"Nested sequences with different lengths.");
					return -5;
				}
				setArrayFromSequence(a, o, dim+1, offset);
			} else {
				PyErr_SetString(PyExc_ValueError,
						"Nested sequences with different lengths.");
				return -4;
			}
		} else {
			PyErr_SetString(PyExc_ValueError, "Invalid sequence.");
			return -6;
		}
		Py_DECREF(o);
		offset += a->strides[dim];
	}
	return 0;
}

static PyObject *
NA_setArrayFromSequence(PyArrayObject *a, PyObject *s)
{
	maybelong shape[MAXDIM];

	if (!PySequence_Check(s))
		return PyErr_Format( PyExc_TypeError, 
				     "NA_setArrayFromSequence: (array, seq) expected.");

	if (getShape(s, shape, 0) < 0)
		return NULL;
	
	if (!NA_updateDataPtr(a)) 
		return NULL;
	
	if (setArrayFromSequence(a, s, 0, 0) < 0)
		return NULL;
	
	Py_INCREF(Py_None);
	return Py_None;
}

static int
_NA_maxType(PyObject *seq, int limit)
{
	if (limit > MAXDIM) {
		PyErr_Format( PyExc_ValueError, 
			      "NA_maxType: sequence nested too deep." );
		return -1;
	}
	if (NA_NumArrayCheck(seq)) {
		switch(PyArray(seq)->descr->type_num) {
		case tBool:
			return BOOL_SCALAR;
		case tInt8:
		case tUInt8:
		case tInt16:
		case tUInt16:
		case tInt32:
		case tUInt32:
			return INT_SCALAR;
		case tInt64:
		case tUInt64:
			return LONG_SCALAR;
		case tFloat32:
		case tFloat64:
			return FLOAT_SCALAR;
		case tComplex32:
		case tComplex64:
			return COMPLEX_SCALAR;
		default:
			PyErr_Format(PyExc_TypeError, 
				     "Expecting a python numeric type, got something else.");
			return -1;
		}
	} else if (PySequence_Check(seq) && !PyString_Check(seq)) {
		long i, maxtype=BOOL_SCALAR, slen;

		slen = PySequence_Length(seq);		
		if (slen < 0) return -1;

		if (slen == 0) return INT_SCALAR;

		for(i=0; i<slen; i++) {
			long newmax;
			PyObject *o = PySequence_GetItem(seq, i);
			if (!o) return -1;
			newmax = _NA_maxType(o, limit+1);
			if (newmax  < 0) 
				return -1;
			else if (newmax > maxtype) {
				maxtype = newmax;
			}
			Py_DECREF(o);
		}
		return maxtype;
	} else {
#if PY_VERSION_HEX >= 0x02030000
		if (PyBool_Check(seq))
			return BOOL_SCALAR;
		else 
#endif
			if (PyInt_Check(seq))
			return INT_SCALAR;
		else if (PyLong_Check(seq))
			return LONG_SCALAR;
		else if (PyFloat_Check(seq))
			return FLOAT_SCALAR;
		else if (PyComplex_Check(seq))
			return COMPLEX_SCALAR;
		else {
			PyErr_Format(PyExc_TypeError, 
				     "Expecting a python numeric type, got something else.");
			return -1;
		}
	}
}

static int 
NA_maxType(PyObject *seq)
{
	int rval;
	rval = _NA_maxType(seq, 0);
	return rval;
}

NumarrayType
NA_NumarrayType(PyObject *seq)
{
	int maxtype = NA_maxType(seq);
	int rval;
	switch(maxtype) {
	case BOOL_SCALAR:
		rval = tBool;
		goto _exit;
	case INT_SCALAR:
	case LONG_SCALAR:
		rval = tLong; /* tLong corresponds to C long int,
				 not Python long int */
		goto _exit;
	case FLOAT_SCALAR:
		rval = tFloat64;
		goto _exit;
	case COMPLEX_SCALAR:
		rval = tComplex64;
		goto _exit;
	default:
		PyErr_Format(PyExc_TypeError,
			     "expecting Python numeric scalar value; got something else.");
		rval = -1;
	}
  _exit:
	return rval;
}

/* sequenceAsArray converts a python sequence (list or tuple)
into an array of the specified type and returns it.
*/
static PyArrayObject*
sequenceAsArray(PyObject *s, NumarrayType *t)
{
	maybelong shape[MAXDIM];
	int dims = getShape(s, shape, 0);
	PyArrayObject *array;
	
	if (dims < 0) return NULL;
	
	if (*t == tAny) {
		*t = NA_NumarrayType(s);
	}
	
	if (!(array = NA_vNewArray(NULL, *t, dims, shape))) 
		return NULL;
	
	if (setArrayFromSequence(array, s, 0, 0) < 0) {
		return (PyArrayObject *) PyErr_Format(
			_Error, 
			"sequenceAsArray: can't convert sequence to array");
	}
	return array;
}

/* satisfies ensures that 'a' meets a set of requirements and matches 
the specified type.
*/
static int
satisfies(PyArrayObject *a, int requirements, NumarrayType t)
{
	int type_ok = (a->descr->type_num == t) || (t == tAny);

	if (PyArray_ISCARRAY(a))
		return type_ok;
	if (PyArray_ISBYTESWAPPED(a) && (requirements & NUM_NOTSWAPPED)) 
		return 0;
	if (!PyArray_ISALIGNED(a) && (requirements & NUM_ALIGNED)) 
		return 0;
	if (!PyArray_ISCONTIGUOUS(a) && (requirements & NUM_CONTIGUOUS))
		return 0;
	if (!PyArray_ISWRITABLE(a) && (requirements & NUM_WRITABLE))
		return 0;
	if (requirements & NUM_COPY)
		return 0;
	return type_ok;
}

/* NA_InputArray is the main input conversion routine.  NA_InputArray
 converts input array 'a' as necessary to NumarrayType 't' also guaranteeing
 that either 'a' or the converted result is contigous, aligned, and not
 byteswapped. NA_InputArray returns a pointer to a Numarray for which
 C_array is 1, and fills in 'ainfo' with the array information of the return
 value.  The return value provides a means to later deallocate any temporary
 array created by NA_InputArray, while 'ainfo' provides direct access to the
 array's "metadata" from C.  Since the reference count of the input array 'a'
 is incremented when it is directly useable by C, the return value (either 'a'
 or a temporary) should always be passed to Py_XDECREF by the calller.  Note
 that at failed sequence conversion, getNumInfo, or getArray results in the
 value NULL being returned.

1. 'a' is already c-usesable.
2. 'a' is an array, but needs conversion to be c-useable.
3. 'a' is a numeric sequence, not an array.
4. 'a' is a numeric scalar, not an array.

The contents of the resulting array are either 'a' or 'a.astype(t)'.
The return value should always be DECREF'ed by the caller.

requires is a bitmask specifying a set of requirements on the converted array.
*/


static PyArrayObject *
NA_FromArrayStruct(PyObject *obj)
{
	PyArrayInterface *arrayif;
	maybelong i, shape[MAXDIM], strides[MAXDIM];
	NumarrayType t;
	PyObject *cobj;
	PyArrayObject *a;

	cobj = PyObject_GetAttrString(obj, "__array_struct__"); /* does Py_INCREF */
	if (!cobj) goto _fail;

	if (!PyCObject_Check(cobj)) {
		PyErr_Format(
			PyExc_TypeError, 
			"__array_struct__ returned non-CObject.");
		goto _fail;
	}

	arrayif = PyCObject_AsVoidPtr(cobj);
	if (arrayif->nd > MAXDIM) {
		PyErr_Format( PyExc_ValueError, 
			      "__array_struct__ too many dimensions: %d", 
			      arrayif->nd);
		goto _fail;
	}

	for(i=0; i<arrayif->nd; i++) {
		shape[i] = arrayif->shape[i];
		strides[i] = arrayif->strides[i];
	}

	t = _scipy_typekind_to_typeNo( arrayif->typekind, arrayif->itemsize );
	if (t < 0) goto _fail;

	a = NA_FromDimsStridesTypeAndData(arrayif->nd, shape, strides, t, arrayif->data);
	if (!a) goto _fail;

	a->base = cobj;

	return a;

  _fail:
	Py_XDECREF(cobj);
	return NULL;
}

static PyArrayObject*
NA_InputArray(PyObject *a, NumarrayType t, int requires)
{
	PyArrayObject *wa = NULL;
	if (NA_isPythonScalar(a)) {
		if (t == tAny) 
			t = NA_NumarrayType(a);
		if (t < 0) goto _exit;
		wa = NA_vNewArray( NULL, t, 0, NULL);
		if (!wa) goto _exit;
		if (NA_setFromPythonScalar(wa, 0, a) < 0) {
			Py_DECREF(wa);
			wa = NULL;
		}
		goto _exit;
	} else if (NA_NumArrayCheck(a)) {
		wa = (PyArrayObject *) a;
		Py_INCREF(a);
	} else if (PyObject_HasAttrString(a, "__array_struct__")) {
		wa = NA_FromArrayStruct(a);
	} else if (PyObject_HasAttrString(a, "__array_typestr__")) {
		wa = (PyArrayObject *) PyObject_CallFunction( pNumArrayArrayFunc, "(O)", a );
	} else {
		wa = sequenceAsArray(a, &t);
	}
	if (!wa) goto _exit;
	if (!satisfies(wa, requires, t)) {
		PyArrayObject *wa2 = getArray(wa, t, "astype");
		Py_DECREF(wa);
		wa = wa2;
	}
	NA_updateDataPtr(wa);
  _exit:
	return wa;
}

/* NA_OutputArray creates a C-usable temporary array similar to 'a' but of
type 't' as necessary.  If 'a' is already C-useable and of type 't', then 'a'
is returned.  In either case, 'ainfo' is filled in with the array information
for the return value.

The contents of the resulting array are undefined, assumed to be filled in by
the caller.
*/
static PyArrayObject *
NA_OutputArray(PyObject *a0, NumarrayType t, int requires)
{
	PyArrayObject *a = (PyArrayObject *) a0;

	if (!NA_NumArrayCheck(a0)  || !PyArray_ISWRITABLE(a)) {
		PyErr_Format(PyExc_TypeError, 
			     "NA_OutputArray: only writable NumArrays work for output.");
		a = NULL;
		goto _exit;
	}

	if (satisfies(a, requires, t)) {
		Py_INCREF(a0);
		NA_updateDataPtr(a);
		goto _exit;
	} else {
		PyArrayObject *shadow = getArray(a, t, "new");
		if (shadow) {
			Py_INCREF(a0);
			shadow->_shadows = a0;
		}
		a = shadow;
	}
  _exit:
	return a;
}

/* NA_OptionalOutputArray works like NA_ShadowOutput, but handles the case
where the output array 'optional' is omitted entirely at the python level,
resulting in 'optional'==Py_None.  When 'optional' is Py_None, the return
value is cloned (but with NumarrayType 't') from 'master', typically an input
array with the same shape as the output array.
*/
static PyArrayObject *
NA_OptionalOutputArray(PyObject *optional, NumarrayType t, int requires, 
		       PyArrayObject *master)
{
	if ((optional == Py_None) || (optional == NULL)) {
		PyArrayObject *rval;
		rval = getArray(master, t, "new");
		return rval;
	} else {
		return NA_OutputArray(optional, t, requires);
	}
}

/* NA_IoArray is a combination of NA_InputArray and NA_OutputArray.

Unlike NA_OutputArray, if a temporary is required it is initialized to a copy
of the input array.

Unlike NA_InputArray, deallocating any resulting temporary array results in a
copy from the temporary back to the original.
*/
static PyArrayObject *
NA_IoArray(PyObject *a, NumarrayType t, int requires)
{
	PyArrayObject *shadow = NA_InputArray(a, t, requires);

	if (!shadow) return NULL;

	/* Guard against non-writable, but otherwise satisfying requires. 
	   In this case,  shadow == a.
	*/
	if (!PyArray_ISWRITABLE(shadow)) {
		PyErr_Format(PyExc_TypeError,
			     "NA_IoArray: I/O numarray must be writable NumArrays.");
		Py_DECREF(shadow);
		shadow = NULL;
		goto _exit;
	}

	if ((shadow != (PyArrayObject *) a) && NA_NumArrayCheck(a)) {
		Py_INCREF(a);
		shadow->_shadows = a;
	}
  _exit:
	return shadow;
}

/* NA_ReturnOutput handles returning a possibly unspecified output array.  If
the array 'out' was specified on the original call to the Python wrapper
function, then the contents of any 'shadow' array are copied into 'out' as
required.  The function then returns Py_None.  If no output was specified in
the original call, then the 'shadow' array *becomes* the output and is
returned.  This results in extension functions which return Py_None when you
specify an output array, and return an array value otherwise.  These functions
also correctly handle data typing, alignment, byteswapping, and contiguity
issues. 
*/
static PyObject*
NA_ReturnOutput(PyObject *out, PyArrayObject *shadow)
{
	if ((out == Py_None) || (out == NULL)) { 
                /* default behavior: return shadow array as the result */
		return (PyObject *) shadow;  
	} else {  
		PyObject *rval;
                /* specified output behavior: return None */
		/* del(shadow) --> out.copyFrom(shadow) */
		Py_DECREF(shadow);
		Py_INCREF(Py_None);
		rval = Py_None;
		return rval;
	}
}

/* NA_ShapeEqual returns 1 if 'a' and 'b' have the same shape, 0 otherwise.
*/
static int
NA_ShapeEqual(PyArrayObject *a, PyArrayObject *b)
{
	int i;
	
	if (!NA_NDArrayCheck((PyObject *) a) || 
	    !NA_NDArrayCheck((PyObject*) b)) {
		PyErr_Format(
			PyExc_TypeError, 
			"NA_ShapeEqual: non-array as parameter.");
		return -1;
	}
	if (a->nd != b->nd)
		return 0;
	for(i=0; i<a->nd; i++)
		if (a->dimensions[i] != b->dimensions[i])
			return 0;
	return 1;
}

/* NA_ShapeLessThan returns 1 if a.shape[i] < b.shape[i] for all i, else 0.  
If they have a different number of dimensions, it compares the innermost
overlapping dimensions of each.
*/
static int
NA_ShapeLessThan(PyArrayObject *a, PyArrayObject *b)
{
        int i;
	int mindim, aoff, boff;
	if (!NA_NDArrayCheck((PyObject *) a) || 
	    !NA_NDArrayCheck((PyObject *) b)) {
		PyErr_Format(PyExc_TypeError, 
			     "NA_ShapeLessThan: non-array as parameter.");
		return -1;
	}
	mindim = MIN(a->nd, b->nd);
	aoff = a->nd - mindim;
	boff = b->nd - mindim;
	for(i=0; i<mindim; i++)
		if (a->dimensions[i+aoff] >=  b->dimensions[i+boff])
			return 0;
	return 1;
}

#define MakeChecker(name, classpointer)                    \
static int                                                 \
name##Exact(PyObject *op) {                                \
        return ((PyObject *) op->ob_type) == classpointer; \
}                                                          \
static int                                                 \
name(PyObject *op) {                                       \
      int rval = -1;                                       \
      if (deferred_libnumarray_init() < 0) goto _exit;     \
      rval = PyObject_IsInstance(op, classpointer);        \
  _exit:                                                   \
      return rval;                                         \
}

MakeChecker(NA_NDArrayCheck, pNDArrayClass)
MakeChecker(NA_NumArrayCheck, pNumArrayClass)
MakeChecker(NA_OperatorCheck, pOperatorClass)
MakeChecker(NA_ConverterCheck, pConverterClass)
MakeChecker(NA_UfuncCheck, pUfuncClass)
MakeChecker(NA_CfuncCheck, pCfuncClass)

static int
NA_ComplexArrayCheck(PyObject *a)
{
	int rval = NA_NumArrayCheck(a);
	if (rval > 0) {
		PyArrayObject *arr = (PyArrayObject *) a;
		switch(arr->descr->type_num) {
		case tComplex64: case tComplex32:
			return 1;
		default:
			return 0;
		}
	}
	return rval;
}

static PyObject * 
NA_Cast(PyArrayObject *a, int type)
{
	PyObject *rval = NULL;
	if (deferred_libnumarray_init() < 0) 
		goto _exit;
	rval = (PyObject *) getArray(a, type, "astype");
  _exit:
	return rval;
}

static int
NA_copyArray(PyArrayObject *to, const PyArrayObject *from)
{
	int rval = -1;
	PyObject *result;
	result = PyObject_CallMethod((PyObject *) to, 
				     "_copyFrom","(O)", from);
	if (!result) goto _exit;
	Py_DECREF(result);
	rval = 0;
  _exit:
	return rval;
}

static PyArrayObject *
NA_copy(PyArrayObject *from)
{
	PyArrayObject * rval;
	rval = (PyArrayObject *)
		PyObject_CallMethod((PyObject *) from, "copy", NULL);
	return rval;
}

static void
NA_stridesFromShape(int nshape, maybelong *shape, maybelong bytestride, 
		    maybelong *strides)
{
	int i;
	if (nshape > 0) {
		for(i=0; i<nshape; i++)
			strides[i] = bytestride;
		for(i=nshape-2; i>=0; i--)
			strides[i] = strides[i+1]*shape[i+1];
	} 
}

static int
NA_getByteOffset(PyArrayObject *array, int nindices, maybelong *indices, 
		 long *offset)
{
	int i;

        /* rank0 or _UBuffer */
	if ((array->nd == 0) || (array->nstrides < 0)) {
		*offset = array->byteoffset;
		return 0;
	}
	
	/* Check for indices/shape mismatch when not rank-0.
	 */
	if ((nindices > array->nd) && 
	    !((nindices == 1) && (array->nd == 0))) {
		PyErr_Format(PyExc_IndexError, "too many indices.");
		return -1;
	}

	*offset = array->byteoffset;
	for(i=0; i<nindices; i++) {
		long ix = indices[i];
		long limit = i < array->nd ? array->dimensions[i] : 0;
		if (ix < 0) ix += limit;
		if (ix < 0 || ix >= limit) {
			PyErr_Format(PyExc_IndexError, "Index out of range");
			return -1;
		}
		*offset += ix*array->strides[i];
	}
	return 0;
}

static int
NA_swapAxes(PyArrayObject *array, int x, int y)
{
	long temp;

	if (((PyObject *) array) == Py_None) return 0;

	if (array->nd < 2) return 0;

	if (x < 0) x += array->nd;
	if (y < 0) y += array->nd;

	if ((x < 0) || (x >= array->nd) || 
	    (y < 0) || (y >= array->nd)) {
		PyErr_Format(PyExc_ValueError, 
			     "Specified dimension does not exist");
		return -1;
	}

	temp = array->dimensions[x];
	array->dimensions[x] = array->dimensions[y];
	array->dimensions[y] = temp;
	
	temp = array->strides[x];
	array->strides[x] = array->strides[y];
	array->strides[y] = temp;

	NA_updateStatus(array);
	
	return 0;
}

static PyObject * 
NA_initModuleGlobal(char *modulename, char *globalname)
{
	PyObject *module, *dict, *global = NULL;
	module = PyImport_ImportModule(modulename);
	if (!module) {
	       PyErr_Format(PyExc_RuntimeError, 
			    "Can't import '%s' module", 
			    modulename);
	       goto _exit;
	}
	dict = PyModule_GetDict(module);
	global = PyDict_GetItemString(dict, globalname);
	if (!global) {
		PyErr_Format(PyExc_RuntimeError, 
			     "Can't find '%s' global in '%s' module.",
			     globalname, modulename);
		goto _exit;
	}
	Py_DECREF(module);
	Py_INCREF(global);
  _exit:
	return global;
}


static long
_isaligned(PyArrayObject *self)
{
	long i, ptr, alignment, aligned = 1;

	alignment = MAX(MIN(self->itemsize, MAX_ALIGN), 1);
	ptr = (long) self->data;
	aligned = (ptr % alignment) == 0;
	for (i=0; i <self->nd; i++)
		aligned &= ((self->strides[i] % alignment) == 0);
	return aligned != 0;
}

static void
NA_updateAlignment(PyArrayObject *self)
{
	if (_isaligned(self))
		self->flags |= ALIGNED;
	else
		self->flags &= ~ALIGNED;

}

static long
_is_contiguous(PyArrayObject *self, maybelong elements)
{
	long i, ndim, nstrides;

	ndim = self->nd;
	nstrides = self->nstrides;

	/* rank-0 numarray are always contiguous */
	if (ndim == 0) return 1;

	/* zero-length arrays are also contiguous */
	if  (elements == 0) return 1;

	/* Strides must be in decreasing order. ndim >= 1 */
	for(i=0; i<ndim-1; i++)
		if (self->strides[i] != 
		    self->strides[i+1]*self->dimensions[i+1])
			return 0;

	/* Broadcast numarray have 0 in some stride and are discontiguous */
	for(i=0; i<nstrides-1; i++)
		if (!self->strides[i])
			return 0;

	if ((self->strides[nstrides-1] == self->itemsize) &&
	    (self->bytestride == self->itemsize))
		return 1;

	if ((self->strides[nstrides-1] == 0) && (nstrides > 1))
		return 1;
	
	return 0;
}

static long
_is_fortran_contiguous(PyArrayObject *self, maybelong elements)
{
	long i, sd;

	/* rank-0 numarray are always fortran_contiguous */
	if (self->nd == 0) return 1;

	/* zero-length arrays are also fortran_contiguous */
	if  (elements == 0) return 1;

	sd = self->descr->elsize;
	for (i=0;i<self->nd;++i) { /* fortran == increasing order */
		if (self->dimensions[i] == 0) return 0;  /* broadcast array */
		if (self->strides[i] != sd) return 0; 
		sd *= self->dimensions[i];
	}

	return 1;
}

static void
NA_updateContiguous(PyArrayObject *self)
{
	maybelong elements = NA_elements(self);
	if (_is_contiguous(self, elements))
		self->flags |= CONTIGUOUS;
	else
		self->flags &= ~CONTIGUOUS;
	if (_is_fortran_contiguous(self, elements))
		self->flags |= FORTRAN_CONTIGUOUS;
	else
		self->flags &= ~FORTRAN_CONTIGUOUS;
}

static int 
_isbyteswapped(PyArrayObject *self)
{
	int syslittle = (NA_ByteOrder() == NUM_LITTLE_ENDIAN);
	int selflittle = (self->byteorder == NUM_LITTLE_ENDIAN);
	int byteswapped = (syslittle != selflittle);
	return byteswapped;
}

static void
NA_updateByteswap(PyArrayObject *self)
{
	if (!_isbyteswapped(self))
		self->flags |= NOTSWAPPED;
	else
		self->flags &= ~NOTSWAPPED;
}

static void 
NA_updateStatus(PyArrayObject *self)
{
	NA_updateAlignment(self);
	NA_updateContiguous(self);
	NA_updateByteswap(self);
}

static char *
NA_getArrayData(PyArrayObject *obj)
{
	if (!NA_NDArrayCheck((PyObject *) obj)) {
		PyErr_Format(PyExc_TypeError, 
			     "expected an NDArray");
	}
	if (!NA_updateDataPtr(obj))
		return NULL;
	return obj->data;
}

/* The following function has much platform dependent code since
** there is no platform-independent way of checking Floating Point
** status bits
*/

/*  OSF/Alpha (Tru64)  ---------------------------------------------*/
#if defined(__osf__) && defined(__alpha)

static int 
NA_checkFPErrors(void)
{
	unsigned long fpstatus;
	int retstatus;

#include <machine/fpu.h>   /* Should migrate to global scope */

	fpstatus = ieee_get_fp_control(); 
	/* clear status bits as well as disable exception mode if on */
	ieee_set_fp_control( 0 );
	retstatus = 
                  pyFPE_DIVIDE_BY_ZERO* (int)((IEEE_STATUS_DZE  & fpstatus) != 0)
		+ pyFPE_OVERFLOW      * (int)((IEEE_STATUS_OVF  & fpstatus) != 0)
		+ pyFPE_UNDERFLOW     * (int)((IEEE_STATUS_UNF  & fpstatus) != 0)
		+ pyFPE_INVALID       * (int)((IEEE_STATUS_INV  & fpstatus) != 0); 

	return retstatus;
}

/* MS Windows -----------------------------------------------------*/
#elif defined(_MSC_VER)

#include <float.h>

static int 
NA_checkFPErrors(void)
{
	int fpstatus = (int) _clear87();
	int retstatus = 
                  pyFPE_DIVIDE_BY_ZERO * ((SW_ZERODIVIDE & fpstatus) != 0)
		+ pyFPE_OVERFLOW       * ((SW_OVERFLOW & fpstatus)   != 0)
		+ pyFPE_UNDERFLOW      * ((SW_UNDERFLOW & fpstatus)  != 0)
		+ pyFPE_INVALID        * ((SW_INVALID & fpstatus)    != 0);


	return retstatus;
}

/* Solaris --------------------------------------------------------*/
/* --------ignoring SunOS ieee_flags approach, someone else can
**         deal with that! */
#elif defined(sun)
#include <ieeefp.h>

static int 
NA_checkFPErrors(void)
{
	int fpstatus;
	int retstatus;

	fpstatus = (int) fpgetsticky();
	retstatus = pyFPE_DIVIDE_BY_ZERO * ((FP_X_DZ  & fpstatus) != 0)
		+ pyFPE_OVERFLOW       * ((FP_X_OFL & fpstatus) != 0)
		+ pyFPE_UNDERFLOW      * ((FP_X_UFL & fpstatus) != 0)
		+ pyFPE_INVALID        * ((FP_X_INV & fpstatus) != 0);
	(void) fpsetsticky(0);

	return retstatus;
}

#elif defined(linux) || defined(darwin) || defined(__CYGWIN__)

#if defined(__GLIBC__) || defined(darwin) || defined(__MINGW32__)
#include <fenv.h>
#elif defined(__CYGWIN__)
#include <mingw/fenv.h>
#endif

static int 
NA_checkFPErrors(void)
{
	int fpstatus = (int) fetestexcept(
		FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
	int retstatus = 
		  pyFPE_DIVIDE_BY_ZERO * ((FE_DIVBYZERO  & fpstatus) != 0)
		+ pyFPE_OVERFLOW       * ((FE_OVERFLOW   & fpstatus) != 0)
		+ pyFPE_UNDERFLOW      * ((FE_UNDERFLOW  & fpstatus) != 0)
		+ pyFPE_INVALID        * ((FE_INVALID    & fpstatus) != 0);
	(void) feclearexcept(FE_DIVBYZERO | FE_OVERFLOW | 
			     FE_UNDERFLOW | FE_INVALID);
	return retstatus;
}

#else

static int 
NA_checkFPErrors(void)
{
	return 0;
}

#endif

static void
NA_clearFPErrors()
{
	NA_checkFPErrors();
}

static int
NA_checkAndReportFPErrors(char *name)
{
	int error = NA_checkFPErrors();
	if (error) {
		PyObject *ans;
		char msg[128];
		if (deferred_libnumarray_init() < 0) 
			return -1;
		strcpy(msg, " in ");
		strncat(msg, name, 100);
		ans = PyObject_CallFunction(pHandleErrorFunc, "(is)", error, msg);
		if (!ans) return -1;
		Py_DECREF(ans); /* Py_None */
	}
	return 0;
}

/**********************************************************************/
/*  Buffer Utility Functions                                          */
/**********************************************************************/

static PyObject *
getBuffer( PyObject *obj) 
{
	if (!obj) return PyErr_Format(PyExc_RuntimeError,
				      "NULL object passed to getBuffer()");
	if (obj->ob_type->tp_as_buffer == NULL) {
		return PyObject_CallMethod(obj, "__buffer__", NULL);
	} else {
		Py_INCREF(obj);  /* Since CallMethod returns a new object when it
				    succeeds, We'll need to DECREF later to free it.
				    INCREF ordinary buffers here so we don't have to
				    remember where the buffer came from at DECREF time.
				 */
		return obj;
	}
}

/* Either it defines the buffer API, or it is an instance which returns
a buffer when obj.__buffer__() is called */
static int 
isBuffer (PyObject *obj) 
{
	PyObject *buf = getBuffer(obj);
	int ans = 0;
	if (buf) {
		ans = buf->ob_type->tp_as_buffer != NULL;
		Py_DECREF(buf);
	} else {
		PyErr_Clear();
	}
	return ans;
}

/**********************************************************************/

static int 
getWriteBufferDataPtr(PyObject *buffobj, void **buff) 
{
  int rval = -1;
  PyObject *buff2;
  if ((buff2 = getBuffer(buffobj)))
  {
	if (buff2->ob_type->tp_as_buffer->bf_getwritebuffer)
		rval = buff2->ob_type->tp_as_buffer->bf_getwritebuffer(buff2, 
									 0, buff);
	Py_DECREF(buff2);
  }
  return rval;
}

/**********************************************************************/

static int 
isBufferWriteable (PyObject *buffobj) 
{
  void *ptr;
  int rval = -1;
  rval = getWriteBufferDataPtr(buffobj, &ptr);
  if (rval == -1)
    PyErr_Clear(); /* Since we're just "testing", it's not really an error */
  return rval != -1;
}

/**********************************************************************/

static int 
getReadBufferDataPtr(PyObject *buffobj, void **buff) 
{
  int rval = -1;
  PyObject *buff2;
  if ((buff2 = getBuffer(buffobj))) {
	  if (buff2->ob_type->tp_as_buffer->bf_getreadbuffer)
		  rval = buff2->ob_type->tp_as_buffer->bf_getreadbuffer(buff2, 
									0, buff);
	  Py_DECREF(buff2);
  }
  return rval;
}

/**********************************************************************/

static int 
getBufferSize(PyObject *buffobj) 
{
  int segcount, size=0;
  PyObject *buff2;
  if ((buff2 = getBuffer(buffobj))) 
    {
      segcount = buff2->ob_type->tp_as_buffer->bf_getsegcount(buff2, 
								&size);
      Py_DECREF(buff2);
    }
  else
    size = -1;
  return size;
}

static long NA_getBufferPtrAndSize(PyObject *buffobj, int readonly, void **ptr)
{
	long rval;
	if (readonly)
		rval = getReadBufferDataPtr(buffobj, ptr);
	else 
		rval = getWriteBufferDataPtr(buffobj, ptr);
	return rval;
}

static int NA_checkOneCBuffer(char *name, long niter, 
		     void *buffer, long bsize, size_t typesize)
{
	Int64 lniter = niter, ltypesize = typesize;

	if (lniter*ltypesize > bsize) {
		PyErr_Format(_Error, 
			     "%s: access out of buffer. niter=%d typesize=%d bsize=%d",
			     name, (int) niter, (int) typesize, (int) bsize);
		return -1;
	}
	if ((typesize <= sizeof(Float64)) && (((long) buffer) % typesize)) {
		PyErr_Format(_Error, 
			     "%s: buffer not aligned on %d byte boundary.",
			     name, (int) typesize);
		return -1;
	}
	return 0;
}

static int NA_checkIo(char *name, 
		      int wantIn, int wantOut, int gotIn, int gotOut)
{
	if (wantIn != gotIn) {
		PyErr_Format(_Error,
		   "%s: wrong # of input buffers. Expected %d.  Got %d.", 
		    name, wantIn, gotIn);
		return -1;
	}
	if (wantOut != gotOut) {
		PyErr_Format(_Error,
		   "%s: wrong # of output buffers. Expected %d.  Got %d.", 
		    name, wantOut, gotOut);
		return -1;
	}
	return 0;
}

static int NA_checkNCBuffers(char *name, int N, long niter, 
			    void **buffers, long *bsizes,
			    Int8 *typesizes, Int8 *iters)
{
	int i;
	for (i=0; i<N; i++)
		if (NA_checkOneCBuffer(name, iters[i] ? iters[i] : niter, 
			      buffers[i], bsizes[i], typesizes[i]))
			return -1;
	return 0;
}


#if 0
static void
_dump_hex(char *name, int ndata, maybelong *data)
{
	int i;
	fprintf(stderr, name);
	for(i=0; i<ndata; i++)
		fprintf(stderr, "%08x ", data[i]);
	fprintf(stderr, "\n");
	fflush(stderr);
}
#endif

static int NA_checkOneStriding(char *name, long dim, maybelong *shape,
	       long offset, maybelong *stride, long buffersize, long itemsize, 
	       int align)
{
  int i;
  long omin=offset, omax=offset;
  long alignsize = (itemsize <= sizeof(Float64) ? itemsize : sizeof(Float64));
  
  if (align && (offset % alignsize)) {
    PyErr_Format(_Error, 
		 "%s: buffer not aligned on %d byte boundary.",
		 name, (int) alignsize);
    return -1;
  }
  for(i=0; i<dim; i++) {
    long strideN = stride[i] * (shape[i]-1);
    long tmax = omax + strideN;
    long tmin = omin + strideN;
    if (shape[i]-1 >= 0) {  /* Skip dimension == 0. */
      omax = MAX(omax, tmax);
      omin = MIN(omin, tmin);
      if (align && (ABS(stride[i]) % alignsize)) {
	PyErr_Format(_Error, 
         "%s: stride %d not aligned on %d byte boundary.",
	     name, (int) stride[i], (int) alignsize);
	return -1;
      }
      if (omax + itemsize > buffersize) {
#if 0
	      _dump_hex("shape:", dim, shape);
	      _dump_hex("strides:", dim, stride);
#endif
	PyErr_Format(_Error, 
           "%s: access beyond buffer. offset=%d buffersize=%d",
		     name, (int) (omax+itemsize-1), (int) buffersize);
	return -1;
      }
      if (omin < 0) {
	PyErr_Format(_Error, 
	     "%s: access before buffer. offset=%d buffersize=%d",
		     name, (int) omin, (int) buffersize);
	return -1;
      }
    }
  }
  return 0;
}

Float64 NA_get_Float64(PyArrayObject *a, long offset)
{
	switch(a->descr->type_num) {
	case tBool:    
		return NA_GETP(a, Bool, (NA_PTR(a)+offset)) != 0;
	case tInt8:    
		return NA_GETP(a, Int8, (NA_PTR(a)+offset));
	case tUInt8:   
		return NA_GETP(a, UInt8, (NA_PTR(a)+offset));
	case tInt16:   
		return NA_GETP(a, Int16, (NA_PTR(a)+offset));
	case tUInt16:  
		return NA_GETP(a, UInt16, (NA_PTR(a)+offset));
	case tInt32:  
		return NA_GETP(a, Int32, (NA_PTR(a)+offset));
	case tUInt32:  
		return NA_GETP(a, UInt32, (NA_PTR(a)+offset));
	case tInt64:  
		return NA_GETP(a, Int64, (NA_PTR(a)+offset));
	#if HAS_UINT64
	case tUInt64:  
		return NA_GETP(a, UInt64, (NA_PTR(a)+offset));
	#endif
	case tFloat32:
		return NA_GETP(a, Float32, (NA_PTR(a)+offset));
	case tFloat64:
		return NA_GETP(a, Float64, (NA_PTR(a)+offset));
	case tComplex32:  /* Since real value is first */
		return NA_GETP(a, Float32, (NA_PTR(a)+offset));
	case tComplex64:  /* Since real value is first */
		return NA_GETP(a, Float64, (NA_PTR(a)+offset));
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unknown type %d in NA_get_Float64",
			      a->descr->type_num); 
	}
	return 0; /* suppress warning */
}

void NA_set_Float64(PyArrayObject *a, long offset, Float64 v)
{
	Bool b;

	switch(a->descr->type_num) {
	case tBool:
		b = (v != 0);
		NA_SETP(a, Bool, (NA_PTR(a)+offset), b);
		break;
	case tInt8:    NA_SETP(a, Int8, (NA_PTR(a)+offset), v);
		break;
	case tUInt8:   NA_SETP(a, UInt8, (NA_PTR(a)+offset), v);
		break;
	case tInt16:   NA_SETP(a, Int16, (NA_PTR(a)+offset), v);
		break;
	case tUInt16:  NA_SETP(a, UInt16, (NA_PTR(a)+offset), v);
		break;
	case tInt32:   NA_SETP(a, Int32, (NA_PTR(a)+offset), v);
		break;
	case tUInt32:   NA_SETP(a, UInt32, (NA_PTR(a)+offset), v);
		break;
	case tInt64:   NA_SETP(a, Int64, (NA_PTR(a)+offset), v);
		break;
        #if HAS_UINT64
	case tUInt64:   NA_SETP(a, UInt64, (NA_PTR(a)+offset), v);
		break;
	#endif
	case tFloat32: 
		NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
		break;
	case tFloat64: 
		NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
		break;
	case tComplex32: {
		NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
		NA_SETP(a, Float32, (NA_PTR(a)+offset+sizeof(Float32)), 0);
		break;
	}
	case tComplex64: {
		NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
		NA_SETP(a, Float64, (NA_PTR(a)+offset+sizeof(Float64)), 0);
		break;
	}
	default:
		PyErr_Format( PyExc_TypeError, 
			      "Unknown type %d in NA_set_Float64",
			      a->descr->type_num ); 
		PyErr_Print();
	}
}

static int 
NA_overflow(PyArrayObject *a, Float64 v)
{
	if ((a->flags & CHECKOVERFLOW) == 0) return 0;

	switch(a->descr->type_num) {
	case tBool:  
		return 0;
	case tInt8:     
		if ((v < -128) || (v > 127))      goto _fail;
		return 0;
	case tUInt8:    
		if ((v < 0) || (v > 255))         goto _fail;
		return 0;
	case tInt16:    
		if ((v < -32768) || (v > 32767))  goto _fail;
		return 0;
	case tUInt16:	
		if ((v < 0) || (v > 65535))       goto _fail;
		return 0;
	case tInt32:   	
		if ((v < -2147483648.) || 
		    (v > 2147483647.))           goto _fail;
		return 0;
	case tUInt32:  	
		if ((v < 0) || (v > 4294967295.)) goto _fail;
		return 0;
	case tInt64: 	
		if ((v < -9223372036854775808.) || 
		    (v > 9223372036854775807.))    goto _fail;
		return 0;
        #if HAS_UINT64
	case tUInt64:	
		if ((v < 0) || 
		    (v > 18446744073709551615.))    goto _fail;
		return 0;
	#endif
	case tFloat32: 
		if ((v < -FLT_MAX) || (v > FLT_MAX)) goto _fail;
		return 0;
	case tFloat64: 
		return 0;
	case tComplex32: 
		if ((v < -FLT_MAX) || (v > FLT_MAX)) goto _fail;
		return 0;
	case tComplex64: 
		return 0;
	default:
		PyErr_Format( PyExc_TypeError, 
			      "Unknown type %d in NA_overflow",
			      a->descr->type_num ); 
		PyErr_Print();
		return -1;
	}
  _fail:
	PyErr_Format(PyExc_OverflowError, "value out of range for array");
	return -1;
}

Complex64 NA_get_Complex64(PyArrayObject *a, long offset)
{
	Complex32 v0;
	Complex64 v;

	switch(a->descr->type_num) {
	case tComplex32: 
		v0 = NA_GETP(a, Complex32, (NA_PTR(a)+offset));
		v.r = v0.r;
		v.i = v0.i;
		break;
	case tComplex64:
		v = NA_GETP(a, Complex64, (NA_PTR(a)+offset));
		break;
	default:
		v.r = NA_get_Float64(a, offset);
		v.i = 0;
		break;
	}
	return v;
}

void NA_set_Complex64(PyArrayObject *a, long offset, Complex64 v)
{
	Complex32 v0;

	switch(a->descr->type_num) {
	case tComplex32:
		v0.r = v.r;
		v0.i = v.i;
		NA_SETP(a, Complex32, (NA_PTR(a)+offset), v0);
		break;
	case tComplex64:
		NA_SETP(a, Complex64, (NA_PTR(a)+offset), v);
		break;
	default:
		NA_set_Float64(a, offset, v.r);
		break;
	}
}

Int64 NA_get_Int64(PyArrayObject *a, long offset)
{
	switch(a->descr->type_num) {
	case tBool:    
		return NA_GETP(a, Bool, (NA_PTR(a)+offset)) != 0;
	case tInt8:    
		return NA_GETP(a, Int8, (NA_PTR(a)+offset));
	case tUInt8:   
		return NA_GETP(a, UInt8, (NA_PTR(a)+offset));
	case tInt16:   
		return NA_GETP(a, Int16, (NA_PTR(a)+offset));
	case tUInt16:  
		return NA_GETP(a, UInt16, (NA_PTR(a)+offset));
	case tInt32:  
		return NA_GETP(a, Int32, (NA_PTR(a)+offset));
	case tUInt32:  
		return NA_GETP(a, UInt32, (NA_PTR(a)+offset));
	case tInt64:  
		return NA_GETP(a, Int64, (NA_PTR(a)+offset));
	case tUInt64:  
		return NA_GETP(a, UInt64, (NA_PTR(a)+offset));
	case tFloat32:
		return NA_GETP(a, Float32, (NA_PTR(a)+offset));
	case tFloat64:
		return NA_GETP(a, Float64, (NA_PTR(a)+offset));
	case tComplex32:
		return NA_GETP(a, Float32, (NA_PTR(a)+offset));
	case tComplex64:
		return NA_GETP(a, Float64, (NA_PTR(a)+offset));
	default:
		PyErr_Format( PyExc_TypeError, 
			      "Unknown type %d in NA_get_Int64",
			      a->descr->type_num); 
		PyErr_Print();
	}
	return 0; /* suppress warning */
}

void NA_set_Int64(PyArrayObject *a, long offset, Int64 v)
{
	Bool b;

	switch(a->descr->type_num) {
	case tBool:
		b = (v != 0);
		NA_SETP(a, Bool, (NA_PTR(a)+offset), b);
		break;
	case tInt8:    NA_SETP(a, Int8, (NA_PTR(a)+offset), v);
		break;
	case tUInt8:   NA_SETP(a, UInt8, (NA_PTR(a)+offset), v);
		break;
	case tInt16:   NA_SETP(a, Int16, (NA_PTR(a)+offset), v);
		break;
	case tUInt16:  NA_SETP(a, UInt16, (NA_PTR(a)+offset), v);
		break;
	case tInt32:   NA_SETP(a, Int32, (NA_PTR(a)+offset), v);
		break;
	case tUInt32:   NA_SETP(a, UInt32, (NA_PTR(a)+offset), v);
		break;
	case tInt64:   NA_SETP(a, Int64, (NA_PTR(a)+offset), v);
		break;
	case tUInt64:   NA_SETP(a, UInt64, (NA_PTR(a)+offset), v);
		break;
	case tFloat32: 
		NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
		break;
	case tFloat64: 
		NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
		break;
	case tComplex32: 
		NA_SETP(a, Float32, (NA_PTR(a)+offset), v);
		NA_SETP(a, Float32, (NA_PTR(a)+offset+sizeof(Float32)), 0);
		break;
	case tComplex64: 
		NA_SETP(a, Float64, (NA_PTR(a)+offset), v);
		NA_SETP(a, Float64, (NA_PTR(a)+offset+sizeof(Float64)), 0);
		break;
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unknown type %d in NA_set_Int64",
			      a->descr->type_num); 
		PyErr_Print();
	}
}

/*  NA_get_offset computes the offset specified by the set of indices.
If N > 0, the indices are taken from the outer dimensions of the array.
If N < 0, the indices are taken from the inner dimensions of the array.
If N == 0, the offset is 0.
*/
long NA_get_offset(PyArrayObject *a, int N, ...)
{
	int i;
	long offset = 0;
	va_list ap;
	va_start(ap, N);
	if (N > 0) { /* compute offset of "outer" indices. */
		for(i=0; i<N; i++)
			offset += va_arg(ap, long) * a->strides[i];
	} else {   /* compute offset of "inner" indices. */
		N = -N;
		for(i=0; i<N; i++)
			offset += va_arg(ap, long) * a->strides[a->nd-N+i];
	}
	va_end(ap);
	return offset;
}

Float64 NA_get1_Float64(PyArrayObject *a, long i)
{
	long offset = i * a->strides[0];
	return NA_get_Float64(a, offset);
}

Float64 NA_get2_Float64(PyArrayObject *a, long i, long j)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1];
	return NA_get_Float64(a, offset);
}

Float64 NA_get3_Float64(PyArrayObject *a, long i, long j, long k)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1] 
		+ k * a->strides[2];
	return NA_get_Float64(a, offset);
}

void NA_set1_Float64(PyArrayObject *a, long i, Float64 v)
{
	long offset = i * a->strides[0];
	NA_set_Float64(a, offset, v);
}

void NA_set2_Float64(PyArrayObject *a, long i, long j, Float64 v)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1];
	NA_set_Float64(a, offset, v);
}

void NA_set3_Float64(PyArrayObject *a, long i, long j, long k, Float64 v)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1] 
		+ k * a->strides[2];
	NA_set_Float64(a, offset, v);
}

Complex64 NA_get1_Complex64(PyArrayObject *a, long i)
{
	long offset = i * a->strides[0];
	return NA_get_Complex64(a, offset);
}

Complex64 NA_get2_Complex64(PyArrayObject *a, long i, long j)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1];
	return NA_get_Complex64(a, offset);
}

Complex64 NA_get3_Complex64(PyArrayObject *a, long i, long j, long k)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1] 
		+ k * a->strides[2];
	return NA_get_Complex64(a, offset);
}

void NA_set1_Complex64(PyArrayObject *a, long i, Complex64 v)
{
	long offset = i * a->strides[0];
	NA_set_Complex64(a, offset, v);
}

void NA_set2_Complex64(PyArrayObject *a, long i, long j, Complex64 v)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1];
	NA_set_Complex64(a, offset, v);
}

void NA_set3_Complex64(PyArrayObject *a, long i, long j, long k, Complex64 v)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1] 
		+ k * a->strides[2];
	NA_set_Complex64(a, offset, v);
}

Int64 NA_get1_Int64(PyArrayObject *a, long i)
{
	long offset = i * a->strides[0];
	return NA_get_Int64(a, offset);
}

Int64 NA_get2_Int64(PyArrayObject *a, long i, long j)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1];
	return NA_get_Int64(a, offset);
}

Int64 NA_get3_Int64(PyArrayObject *a, long i, long j, long k)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1] 
		+ k * a->strides[2];
	return NA_get_Int64(a, offset);
}

void NA_set1_Int64(PyArrayObject *a, long i, Int64 v)
{
	long offset = i * a->strides[0];
	NA_set_Int64(a, offset, v);
}

void NA_set2_Int64(PyArrayObject *a, long i, long j, Int64 v)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1];
	NA_set_Int64(a, offset, v);
}

void NA_set3_Int64(PyArrayObject *a, long i, long j, long k, Int64 v)
{
	long offset  = i * a->strides[0] 
		+ j * a->strides[1] 
		+ k * a->strides[2];
	NA_set_Int64(a, offset, v);
}

/* SET_CMPLX could be made faster by factoring it into 3 seperate loops.
*/
#define NA_SET_CMPLX(a, type, base, cnt, in)                                  \
{                                                                             \
        int i;                                                                \
	int stride = a->strides[ a->nd - 1];                                  \
        NA_SET1D(a, type, base, cnt, in);                                     \
	base = NA_PTR(a) + offset + sizeof(type);                             \
	for(i=0; i<cnt; i++) {                                                \
		NA_SETP(a, Float32, base, 0);                                 \
		base += stride;                                               \
	}                                                                     \
}

static int
NA_get1D_Float64(PyArrayObject *a, long offset, int cnt, Float64*out)
{
	char *base = NA_PTR(a) + offset;

	switch(a->descr->type_num) {
	case tBool:
		NA_GET1D(a, Bool, base, cnt, out); 
		break;
	case tInt8:    
		NA_GET1D(a, Int8, base, cnt, out); 
		break;
	case tUInt8:   
		NA_GET1D(a, UInt8, base, cnt, out); 
		break;
	case tInt16:   
		NA_GET1D(a, Int16, base, cnt, out); 
		break;
	case tUInt16:  
		NA_GET1D(a, UInt16, base, cnt, out); 
		break;
	case tInt32:  
		NA_GET1D(a, Int32, base, cnt, out); 
		break;
	case tUInt32:  
		NA_GET1D(a, UInt32, base, cnt, out); 
		break;
	case tInt64:  
		NA_GET1D(a, Int64, base, cnt, out); 
		break;
        #if HAS_UINT64
	case tUInt64:  
		NA_GET1D(a, UInt64, base, cnt, out); 
		break;
        #endif
	case tFloat32: 
		NA_GET1D(a, Float32, base, cnt, out); 
		break;
	case tFloat64: 
		NA_GET1D(a, Float64, base, cnt, out); 
		break;
	case tComplex32:
		NA_GET1D(a, Float32, base, cnt, out);
		break;
	case tComplex64:
		NA_GET1D(a, Float64, base, cnt, out);
		break;
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unknown type %d in NA_get1D_Float64", 
			      a->descr->type_num); 
		PyErr_Print();
		return -1;
	}
	return 0;
}

static Float64 *
NA_alloc1D_Float64(PyArrayObject *a, long offset, int cnt)
{
	Float64 *result = PyMem_New(Float64, cnt);
	if (!result) return NULL;
	if (NA_get1D_Float64(a, offset, cnt, result) < 0) {
		PyMem_Free(result);
		return NULL;
	}
	return result;
}

static int
NA_set1D_Float64(PyArrayObject *a, long offset, int cnt, Float64*in)
{
	char *base = NA_PTR(a) + offset;

	switch(a->descr->type_num) {
	case tBool:
		NA_SET1D(a, Bool, base, cnt, in); 
		break;
	case tInt8:    
		NA_SET1D(a, Int8, base, cnt, in); 
		break;
	case tUInt8:   
		NA_SET1D(a, UInt8, base, cnt, in); 
		break;
	case tInt16:   
		NA_SET1D(a, Int16, base, cnt, in); 
		break;
	case tUInt16:  
		NA_SET1D(a, UInt16, base, cnt, in); 
		break;
	case tInt32:  
		NA_SET1D(a, Int32, base, cnt, in); 
		break;
	case tUInt32:  
		NA_SET1D(a, UInt32, base, cnt, in); 
		break;
	case tInt64:  
		NA_SET1D(a, Int64, base, cnt, in); 
		break;
        #if HAS_UINT64
	case tUInt64:  
		NA_SET1D(a, UInt64, base, cnt, in); 
		break;
	#endif
	case tFloat32: 
		NA_SET1D(a, Float32, base, cnt, in); 
		break;
	case tFloat64: 
		NA_SET1D(a, Float64, base, cnt, in); 
		break;
	case tComplex32:
		NA_SET_CMPLX(a, Float32, base, cnt, in);
		break;
	case tComplex64:
		NA_SET_CMPLX(a, Float64, base, cnt, in);
		break;
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unknown type %d in NA_set1D_Float64",
			      a->descr->type_num); 
		PyErr_Print();
		return -1;
	}
	return 0;
}

static int
NA_get1D_Int64(PyArrayObject *a, long offset, int cnt, Int64*out)
{
	char *base = NA_PTR(a) + offset;

	switch(a->descr->type_num) {
	case tBool:
		NA_GET1D(a, Bool, base, cnt, out); 
		break;
	case tInt8:    
		NA_GET1D(a, Int8, base, cnt, out); 
		break;
	case tUInt8:   
		NA_GET1D(a, UInt8, base, cnt, out); 
		break;
	case tInt16:   
		NA_GET1D(a, Int16, base, cnt, out); 
		break;
	case tUInt16:  
		NA_GET1D(a, UInt16, base, cnt, out); 
		break;
	case tInt32:  
		NA_GET1D(a, Int32, base, cnt, out); 
		break;
	case tUInt32:  
		NA_GET1D(a, UInt32, base, cnt, out); 
		break;
	case tInt64:  
		NA_GET1D(a, Int64, base, cnt, out); 
		break;
	case tUInt64:  
		NA_GET1D(a, UInt64, base, cnt, out); 
		break;
	case tFloat32: 
		NA_GET1D(a, Float32, base, cnt, out); 
		break;
	case tFloat64: 
		NA_GET1D(a, Float64, base, cnt, out); 
		break;
	case tComplex32:
		NA_GET1D(a, Float32, base, cnt, out);
		break;
	case tComplex64:
		NA_GET1D(a, Float64, base, cnt, out);
		break;
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unknown type %d in NA_get1D_Int64",
			      a->descr->type_num); 
		PyErr_Print();
		return -1;
	}
	return 0;
}

static Int64 *
NA_alloc1D_Int64(PyArrayObject *a, long offset, int cnt)
{
	Int64 *result = PyMem_New(Int64, cnt);
	if (!result) return NULL;
	if (NA_get1D_Int64(a, offset, cnt, result) < 0) {
		PyMem_Free(result);
		return NULL;
	}
	return result;
}

static int
NA_set1D_Int64(PyArrayObject *a, long offset, int cnt, Int64*in)
{
	char *base = NA_PTR(a) + offset;

	switch(a->descr->type_num) {
	case tBool:
		NA_SET1D(a, Bool, base, cnt, in); 
		break;
	case tInt8:    
		NA_SET1D(a, Int8, base, cnt, in); 
		break;
	case tUInt8:   
		NA_SET1D(a, UInt8, base, cnt, in); 
		break;
	case tInt16:   
		NA_SET1D(a, Int16, base, cnt, in); 
		break;
	case tUInt16:  
		NA_SET1D(a, UInt16, base, cnt, in); 
		break;
	case tInt32:  
		NA_SET1D(a, Int32, base, cnt, in); 
		break;
	case tUInt32:  
		NA_SET1D(a, UInt32, base, cnt, in); 
		break;
	case tInt64:  
		NA_SET1D(a, Int64, base, cnt, in); 
		break;
	case tUInt64:  
		NA_SET1D(a, UInt64, base, cnt, in); 
		break;
	case tFloat32: 
		NA_SET1D(a, Float32, base, cnt, in); 
		break;
	case tFloat64: 
		NA_SET1D(a, Float64, base, cnt, in); 
		break;
	case tComplex32:
		NA_SET_CMPLX(a, Float32, base, cnt, in);
		break;
	case tComplex64:
		NA_SET_CMPLX(a, Float64, base, cnt, in);
		break;
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unknown type %d in NA_set1D_Int64",
			      a->descr->type_num); 
		PyErr_Print();
		return -1;
	}
	return 0;
}

static int
NA_get1D_Complex64(PyArrayObject *a, long offset, int cnt, Complex64*out)
{
	char *base = NA_PTR(a) + offset;

	switch(a->descr->type_num) {
	case tComplex64:
		NA_GET1D(a, Complex64, base, cnt, out);
		break;
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unsupported type %d in NA_get1D_Complex64",
			      a->descr->type_num); 
		PyErr_Print();
		return -1;
	}
	return 0;
}

static int 
NA_set1D_Complex64(PyArrayObject *a, long offset, int cnt, Complex64*in)
{
	char *base = NA_PTR(a) + offset;

	switch(a->descr->type_num) {
	case tComplex64:
		NA_SET1D(a, Complex64, base, cnt, in);
		break;
	default:
		PyErr_Format( PyExc_TypeError,
			      "Unsupported type %d in NA_set1D_Complex64",
			      a->descr->type_num); 
		PyErr_Print();
		return -1;
	}
	return 0;
}

#if LP64
#define PlatBigInt PyInt_FromLong
#define PlatBigUInt PyLong_FromUnsignedLong
#else
#define PlatBigInt PyLong_FromLongLong
#define PlatBigUInt PyLong_FromUnsignedLongLong
#endif

static int 
_checkOffset(PyArrayObject *a, long offset)
{
	long finaloffset = a->byteoffset + offset;
	long size = getBufferSize(a->_data);	
	if (size < 0) {
		PyErr_Format(_Error,
			     "can't get buffer size");
		return -1;
	}
	if (finaloffset < 0 || finaloffset > size) {
		PyErr_Format(_Error,
			     "invalid buffer offset");
		return -1;
	}
	return 0;
}

static PyObject *
NA_getPythonScalar(PyArrayObject *a, long offset)
{
	int type = a->descr->type_num;
	PyObject *rval = NULL;

	if (_checkOffset(a, offset) < 0)
		goto _exit;

	switch(type) {
	case tBool:
        case tInt8:
	case tUInt8:
        case tInt16:
	case tUInt16:
	case tInt32: {
		Int64 v = NA_get_Int64(a, offset);
		rval = PyInt_FromLong(v);
		break;
	}
	case tUInt32: {
		Int64 v = NA_get_Int64(a, offset);
		rval = PlatBigUInt(v); 
		break;
	}
	case tInt64: {
		Int64 v = NA_get_Int64(a, offset);
		rval = PlatBigInt( v);
		break;
	}
	case tUInt64: {
		Int64 v = NA_get_Int64(a, offset);
		rval = PlatBigUInt( v);
		break;
	}
	case tFloat32:
	case tFloat64: {
		Float64 v = NA_get_Float64(a, offset);
		rval = PyFloat_FromDouble( v );
		break;
	}
	case tComplex32:
	case tComplex64: 
	{
		Complex64 v = NA_get_Complex64(a, offset);
		rval = PyComplex_FromDoubles(v.r, v.i);
		break;
	}
	default:
		rval = PyErr_Format(PyExc_TypeError, 
				    "NA_getPythonScalar: bad type %d\n", 
				    type);
	}
  _exit:
	return rval;
}

static int        
_setFromPythonScalarCore(PyArrayObject *a, long offset, PyObject*value, int entries)
{
	Int64 v;
	if (entries >= 100) {
		PyErr_Format(PyExc_RuntimeError, 
			     "NA_setFromPythonScalar: __tonumtype__ conversion chain too long");
		return -1;
	} else if (PyInt_Check(value)) {
		v = PyInt_AsLong(value);
		if (NA_overflow(a, v) < 0)
			return -1;
		NA_set_Int64(a, offset, v);
	} else if (PyLong_Check(value)) {  
		if (a->descr->type_num == tInt64) {
			v = (Int64) PyLong_AsLongLong( value );
		} else if (a->descr->type_num == tUInt64) {
			v = (UInt64) PyLong_AsUnsignedLongLong( value );
		} else if (a->descr->type_num == tUInt32) {
			v = PyLong_AsUnsignedLong(value);
		} else {
			v = PyLong_AsLongLong(value);
		}
		if (PyErr_Occurred())
			return -1;
		if (NA_overflow(a, v) < 0)
			return -1;
		NA_set_Int64(a, offset, v);
	} else if (PyFloat_Check(value)) {
		Float64 v = PyFloat_AsDouble(value);
		if (NA_overflow(a, v) < 0)
			return -1;
		NA_set_Float64(a, offset, v);
	} else if (PyComplex_Check(value)) {
		Complex64 vc;
		vc.r = PyComplex_RealAsDouble(value);
		vc.i = PyComplex_ImagAsDouble(value);
		if (NA_overflow(a, vc.r) < 0)
			return -1;
		if (NA_overflow(a, vc.i) < 0)
			return -1;
		NA_set_Complex64(a, offset, vc);
	} else if (PyObject_HasAttrString(value, "__tonumtype__")) {
		int rval;
		PyObject *type = NA_typeNoToTypeObject(a->descr->type_num);
		if (!type) return -1;
		value = PyObject_CallMethod(
			value, "__tonumtype__", "(N)", type);
		if (!value) return -1;
		rval = _setFromPythonScalarCore(a, offset, value, entries+1);
		Py_DECREF(value);
		return rval;
	} else if (PyString_Check(value)) {
		long size = PyString_Size(value);
		if ((size <= 0) || (size > 1)) {
			PyErr_Format( PyExc_ValueError, 
				      "NA_setFromPythonScalar: len(string) must be 1.");
			return -1;
		}
		NA_set_Int64(a, offset, *PyString_AsString(value));
	} else {
		PyErr_Format(PyExc_TypeError, 
			     "NA_setFromPythonScalar: bad value type.");
		return -1;
	}
	return 0;
}

static int
NA_setFromPythonScalar(PyArrayObject *a, long offset, PyObject *value)
{
	if (_checkOffset(a, offset) < 0)
		return -1;
	if (a->flags & WRITABLE)
		return _setFromPythonScalarCore(a, offset, value, 0);
	else {
		PyErr_Format(
			PyExc_ValueError, "NA_setFromPythonScalar: assigment to readonly array buffer");
		return -1;
	}
}

static int
NA_isPythonScalar(PyObject *o)
{
	int rval;
	rval =  PyInt_Check(o) || 
		PyLong_Check(o) || 
		PyFloat_Check(o) || 
		PyComplex_Check(o) ||
		(PyString_Check(o) && (PyString_Size(o) == 1));
	return rval;
}

static unsigned long 
NA_elements(PyArrayObject  *a)
{
	int i;
	unsigned long n = 1;
	for(i = 0; i<a->nd; i++)
		n *= a->dimensions[i];
	return n;
}

staticforward PyTypeObject CfuncType;

static void
cfunc_dealloc(PyObject* self)
{
	PyObject_Del(self);
}

static PyObject *
cfunc_repr(PyObject *self)
{
	char buf[256];
	CfuncObject *me = (CfuncObject *) self;
	sprintf(buf, "<cfunc '%s' at %08lx check-self:%d align:%d  io:(%d, %d)>", 
		 me->descr.name, (unsigned long ) me->descr.fptr, 
		 me->descr.chkself, me->descr.align, 
		 me->descr.wantIn, me->descr.wantOut);
	return PyString_FromString(buf);
}

/* Call a standard "stride" function
**
** Stride functions always take one input and one output array.
** They can handle n-dimensional data with arbitrary strides (of
** either sign) for both the input and output numarray. Typically
** these functions are used to copy data, byteswap, or align data.
**
**
** It expects the following arguments:
**
**   Number of iterations for each dimension as a tuple
**   Input Buffer Object
**   Offset in bytes for input buffer
**   Input strides (in bytes) for each dimension as a tuple
**   Output Buffer Object
**   Offset in bytes for output buffer
**   Output strides (in bytes) for each dimension as a tuple
**   An integer (Optional), typically the number of bytes to copy per
*       element.
**
** Returns None
**
** The arguments expected by the standard stride functions that this
** function calls are:
**
**   Number of dimensions to iterate over
**   Long int value (from the optional last argument to
**      callStrideConvCFunc)
**      often unused by the C Function
**   An array of long ints. Each is the number of iterations for each
**      dimension. NOTE: the previous argument as well as the stride
**      arguments are reversed in order with respect to how they are
**      used in Python. Fastest changing dimension is the first element
**      in the numarray!
**   A void pointer to the input data buffer.
**   The starting offset for the input data buffer in bytes (long int).
**   An array of long int input strides (in bytes) [reversed as with
**      the iteration array]
**   A void pointer to the output data buffer.
**   The starting offset for the output data buffer in bytes (long int).
**   An array of long int output strides (in bytes) [also reversed]
*/


static PyObject *
NA_callStrideConvCFuncCore(
	PyObject *self, int nshape, maybelong *shape,
	PyObject *inbuffObj,  long inboffset, 
	int ninbstrides, maybelong *inbstrides,
	PyObject *outbuffObj, long outboffset, 
	int noutbstrides, maybelong *outbstrides,
	long nbytes)
{
	CfuncObject *me = (CfuncObject *) self;
	CFUNC_STRIDE_CONV_FUNC funcptr;
	void *inbuffer, *outbuffer;
	long inbsize, outbsize;
	maybelong i, lshape[MAXDIM], in_strides[MAXDIM], out_strides[MAXDIM];
	maybelong shape_0, inbstr_0, outbstr_0;

	if (nshape == 0) {   /* handle rank-0 numarray. */
		nshape = 1;
		shape = &shape_0; 
		inbstrides = &inbstr_0;
		outbstrides = &outbstr_0;
		shape[0] = 1;
		inbstrides[0] = outbstrides[0] = 0;
	}

	for(i=0; i<nshape; i++) 
		lshape[i] = shape[nshape-1-i];
	for(i=0; i<nshape; i++) 
		in_strides[i] = inbstrides[nshape-1-i];
	for(i=0; i<nshape; i++) 
		out_strides[i] = outbstrides[nshape-1-i];
	
	if (!PyObject_IsInstance(self , (PyObject *) &CfuncType)
	    || me->descr.type != CFUNC_STRIDING)
		return PyErr_Format(PyExc_TypeError,
		      "NA_callStrideConvCFuncCore: problem with cfunc");
	
	if ((inbsize = NA_getBufferPtrAndSize(inbuffObj, 1, &inbuffer)) < 0)
		return PyErr_Format(_Error,
		      "%s: Problem with input buffer", me->descr.name);
	
	if ((outbsize = NA_getBufferPtrAndSize(outbuffObj, 0, &outbuffer)) < 0)
		return PyErr_Format(_Error,
		      "%s: Problem with output buffer (read only?)", 
				    me->descr.name);
	
	/* Check buffer alignment and bounds */
	if (NA_checkOneStriding(me->descr.name, nshape, lshape,
				inboffset, in_strides, inbsize,
				(me->descr.sizes[0] == -1) ? 
				nbytes : me->descr.sizes[0],
				me->descr.align) ||
	    NA_checkOneStriding(me->descr.name, nshape, lshape,
				outboffset, out_strides, outbsize,
				(me->descr.sizes[1] == -1) ? 
				nbytes : me->descr.sizes[1],
				me->descr.align))
		return NULL;
	
	/* Cast function pointer and perform stride operation */
	funcptr = (CFUNC_STRIDE_CONV_FUNC) me->descr.fptr;
	if ((*funcptr)(nshape-1, nbytes, lshape, 
		       inbuffer,  inboffset, in_strides,
		       outbuffer, outboffset, out_strides) == 0) {
		Py_INCREF(Py_None);
		return Py_None;
	} else {
		return NULL;
	}
}

static PyObject *
callStrideConvCFunc(PyObject *self, PyObject *args) {
    PyObject *inbuffObj, *outbuffObj, *shapeObj;
    PyObject *inbstridesObj, *outbstridesObj;
    CfuncObject *me = (CfuncObject *) self;
    int  nshape, ninbstrides, noutbstrides, nargs;
    maybelong shape[MAXDIM], inbstrides[MAXDIM], 
	    outbstrides[MAXDIM], *outbstrides1 = outbstrides;
    long inboffset, outboffset, nbytes=0;

    nargs = PyObject_Length(args);
    if (!PyArg_ParseTuple(args, "OOlOOlO|l",
            &shapeObj, &inbuffObj,  &inboffset, &inbstridesObj,
            &outbuffObj, &outboffset, &outbstridesObj,
            &nbytes)) {
        return PyErr_Format(_Error,
                 "%s: Problem with argument list",
		  me->descr.name);
    }

    nshape = NA_maybeLongsFromIntTuple(MAXDIM, shape, shapeObj);
    if (nshape < 0) return NULL;
    
    ninbstrides = NA_maybeLongsFromIntTuple(MAXDIM, inbstrides, inbstridesObj);
    if (ninbstrides < 0) return NULL;

    noutbstrides=  NA_maybeLongsFromIntTuple(MAXDIM, outbstrides, outbstridesObj);
    if (noutbstrides < 0) return NULL;

    if (nshape && (nshape != ninbstrides)) {
        return PyErr_Format(_Error,
            "%s: Missmatch between input iteration and strides tuples",
	    me->descr.name);
    }

    if (nshape && (nshape != noutbstrides)) {
	    if (noutbstrides < 1 || 
		outbstrides[ noutbstrides - 1 ])/* allow 0 for reductions. */
		    return PyErr_Format(_Error,
					"%s: Missmatch between output "
					"iteration and strides tuples",
					me->descr.name);
    }
    
#if 0    /* reductions slow mode hack...  wrong place to do it. */
    _dump_hex("shape: ", nshape, shape);
    _dump_hex("instrides: ", ninbstrides, inbstrides);
    _dump_hex("outstrides: ", noutbstrides, outbstrides);

    if (ninbstrides != noutbstrides) {
	    outbstrides1 = outbstrides + (noutbstrides - ninbstrides);
	    noutbstrides = ninbstrides;
    } 
#endif

    return NA_callStrideConvCFuncCore(
	    self, nshape, shape,
	    inbuffObj,  inboffset,  ninbstrides, inbstrides,
	    outbuffObj, outboffset, noutbstrides, outbstrides1, nbytes);
}

static int 
_NA_callStridingHelper(PyObject *aux, long dim, 
		       long nnumarray, PyArrayObject *numarray[], char *data[],
		       CFUNC_STRIDED_FUNC f)
{
	int i, j, status=0;
	dim -= 1;
	for(i=0; i<numarray[0]->dimensions[dim]; i++) {
		for (j=0; j<nnumarray; j++)
			data[j] += numarray[j]->strides[dim]*i;
		if (dim == 0)
			status |= f(aux, nnumarray, numarray, data);
		else
			status |= _NA_callStridingHelper(
				aux, dim, nnumarray, numarray, data, f);
		for (j=0; j<nnumarray; j++)
			data[j] -= numarray[j]->strides[dim]*i;
	}
	return status;
}


static PyObject *
callStridingCFunc(PyObject *self, PyObject *args) {
    CfuncObject *me = (CfuncObject *) self;
    PyObject *aux;
    PyArrayObject *numarray[MAXARRAYS];
    char *data[MAXARRAYS];
    CFUNC_STRIDED_FUNC f;
    int i;

    int nnumarray = PySequence_Length(args)-1;
    if ((nnumarray < 1) || (nnumarray > MAXARRAYS))
	    return PyErr_Format(_Error, "%s, too many or too few numarray.",
				me->descr.name);

    aux = PySequence_GetItem(args, 0);
    if (!aux)
	    return NULL;

    for(i=0; i<nnumarray; i++) {
	    PyObject *otemp = PySequence_GetItem(args, i+1);
	    if (!otemp)
		    return PyErr_Format(_Error, "%s couldn't get array[%d]", 
					me->descr.name, i);
	    if (!NA_NDArrayCheck(otemp))
		    return PyErr_Format(PyExc_TypeError, 
				 "%s arg[%d] is not an array.",
				 me->descr.name, i);
	    numarray[i] = (PyArrayObject *) otemp;
	    data[i] = numarray[i]->data;
	    Py_DECREF(otemp);
	    if (!NA_updateDataPtr(numarray[i]))
		    return NULL;
    }
	    
    /* Cast function pointer and perform stride operation */
    f = (CFUNC_STRIDED_FUNC) me->descr.fptr;
    
    if (_NA_callStridingHelper(aux, numarray[0]->nd, 
			       nnumarray, numarray, data, f)) {
	    return NULL;
    } else {
	    Py_INCREF(Py_None);
	    return Py_None;
    }
}

/* Convert a standard C numeric value to a Python numeric value.
**
** Handles both nonaligned and/or byteswapped C data.
**
** Input arguments are:
**
**   Buffer object that contains the C numeric value.
**   Offset (in bytes) into the buffer that the data is located at.
**   The size of the C numeric data item in bytes.
**   Flag indicating if the C data is byteswapped from the processor's
**     natural representation.
**
**   Returns a Python numeric value.
*/

static PyObject *
NumTypeAsPyValue(PyObject *self, PyObject *args) {
    PyObject *bufferObj;
    long offset, itemsize, byteswap, i, buffersize;
    Py_complex temp;  /* to hold copies of largest possible type */
    void *buffer;
    char *tempptr;
    CFUNCasPyValue funcptr;
    CfuncObject *me = (CfuncObject *) self;

    if (!PyArg_ParseTuple(args, "Olll", 
			  &bufferObj, &offset, &itemsize, &byteswap))
        return PyErr_Format(_Error,
		"NumTypeAsPyValue: Problem with argument list");

    if ((buffersize = NA_getBufferPtrAndSize(bufferObj, 1, &buffer)) < 0)
        return PyErr_Format(_Error,
                "NumTypeAsPyValue: Problem with array buffer");

    if (offset < 0)
	return PyErr_Format(_Error,
		"NumTypeAsPyValue: invalid negative offset: %d", (int) offset);

    /* Guarantee valid buffer pointer */
    if (offset+itemsize > buffersize)
	    return PyErr_Format(_Error,
		"NumTypeAsPyValue: buffer too small for offset and itemsize.");

    /* Do byteswapping.  Guarantee double alignment by using temp. */
    tempptr = (char *) &temp;
    if (!byteswap) {
        for (i=0; i<itemsize; i++)
            *(tempptr++) = *(((char *) buffer)+offset+i);
    } else {
        tempptr += itemsize-1;
        for (i=0; i<itemsize; i++)
            *(tempptr--) = *(((char *) buffer)+offset+i);
    }

    funcptr = (CFUNCasPyValue) me->descr.fptr;

    /* Call function to build PyObject.  Bad parameters to this function
       may render call meaningless, but "temp" guarantees that its safe.  */
    return (*funcptr)((void *)(&temp));
}

/* Convert a Python numeric value to a standard C numeric value.
**
** Handles both nonaligned and/or byteswapped C data.
**
** Input arguments are:
**
**   The Python numeric value to be converted.
**   Buffer object to contain the C numeric value.
**   Offset (in bytes) into the buffer that the data is to be copied to.
**   The size of the C numeric data item in bytes.
**   Flag indicating if the C data is byteswapped from the processor's
**     natural representation.
**
**   Returns None
*/

static PyObject *
NumTypeFromPyValue(PyObject *self, PyObject *args) {
    PyObject *bufferObj, *valueObj;
    long offset, itemsize, byteswap, i, buffersize;
    Py_complex temp;  /* to hold copies of largest possible type */
    void *buffer;
    char *tempptr;
    CFUNCfromPyValue funcptr;
    CfuncObject *me = (CfuncObject *) self;

    if (!PyArg_ParseTuple(args, "OOlll", 
		  &valueObj, &bufferObj, &offset, &itemsize, &byteswap)) 
        return PyErr_Format(_Error,
                 "%s: Problem with argument list", me->descr.name);

    if ((buffersize = NA_getBufferPtrAndSize(bufferObj, 0, &buffer)) < 0)
	    return PyErr_Format(_Error,
                "%s: Problem with array buffer (read only?)", me->descr.name);

    funcptr = (CFUNCfromPyValue) me->descr.fptr;

    /* Convert python object into "temp". Always safe. */
    if (!((*funcptr)(valueObj, (void *)( &temp))))
        return PyErr_Format(_Error,
		 "%s: Problem converting value", me->descr.name);

    /* Check buffer offset. */
    if (offset < 0)
	return PyErr_Format(_Error,
		"%s: invalid negative offset: %d", me->descr.name, (int) offset);

    if (offset+itemsize > buffersize)
	return PyErr_Format(_Error,
		"%s: buffer too small(%d) for offset(%d) and itemsize(%d)",
			me->descr.name, (int) buffersize, (int) offset, (int) itemsize);

    /* Copy "temp" to array buffer. */
    tempptr = (char *) &temp;
    if (!byteswap) {
        for (i=0; i<itemsize; i++)
            *(((char *) buffer)+offset+i) = *(tempptr++);
    } else {
        tempptr += itemsize-1;
        for (i=0; i<itemsize; i++)
            *(((char *) buffer)+offset+i) = *(tempptr--);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

/* Function to call standard C Ufuncs
**
** The C Ufuncs expect contiguous 1-d data numarray, input and output numarray
** iterate with standard increments of one data element over all numarray.
** (There are some exceptions like arrayrangexxx which use one or more of
** the data numarray as parameter or other sources of information and do not
** iterate over every buffer).
**
** Arguments:
**
**   Number of iterations (simple integer value).
**   Number of input numarray.
**   Number of output numarray.
**   Tuple of tuples, one tuple per input/output array. Each of these
**     tuples consists of a buffer object and a byte offset to start.
**
** Returns None
*/

static PyObject *
NA_callCUFuncCore(PyObject *self, 
		  long niter, long ninargs, long noutargs,
		  PyObject **BufferObj, long *offset)
{
	CfuncObject *me = (CfuncObject *) self;
	char *buffers[MAXARGS];
	long bsizes[MAXARGS];
	long i, pnargs = ninargs + noutargs;
	UFUNC ufuncptr;

	if (pnargs > MAXARGS) 
		return PyErr_Format(PyExc_RuntimeError, "NA_callCUFuncCore: too many parameters");

	if (!PyObject_IsInstance(self, (PyObject *) &CfuncType)
	    || me->descr.type != CFUNC_UFUNC)
		return PyErr_Format(PyExc_TypeError,
		       "NA_callCUFuncCore: problem with cfunc.");
	
	for (i=0; i<pnargs; i++) {
		int readonly = (i < ninargs);
		if (offset[i] < 0)
			return PyErr_Format(_Error,
					    "%s: invalid negative offset:%d for buffer[%d]", 
					    me->descr.name, (int) offset[i], (int) i);
		if ((bsizes[i] = NA_getBufferPtrAndSize(BufferObj[i], readonly, 
							(void *) &buffers[i])) < 0)
			return PyErr_Format(_Error,
					    "%s: Problem with %s buffer[%d].", 
					    me->descr.name, 
					    readonly ? "read" : "write", (int) i);
		buffers[i] += offset[i];
		bsizes[i]  -= offset[i]; /* "shorten" buffer size by offset. */
	}
	
	ufuncptr = (UFUNC) me->descr.fptr;
	
	/* If it's not a self-checking ufunc, check arg count match,
	   buffer size, and alignment for all buffers */
	if (!me->descr.chkself && 
	    (NA_checkIo(me->descr.name, 
			me->descr.wantIn, me->descr.wantOut, ninargs, noutargs) ||
	     NA_checkNCBuffers(me->descr.name, pnargs, 
			       niter, (void **) buffers, bsizes, 
			       me->descr.sizes, me->descr.iters)))
		return NULL;
	
	/* Since the parameters are valid, call the C Ufunc */
	if (!(*ufuncptr)(niter, ninargs, noutargs, (void **)buffers, bsizes)) {
		Py_INCREF(Py_None);
		return Py_None;
	} else {
		return NULL;
	}
}

static PyObject *
callCUFunc(PyObject *self, PyObject *args) {
	PyObject *DataArgs, *ArgTuple;
	long pnargs, ninargs, noutargs, niter, i;
	CfuncObject *me = (CfuncObject *) self;
	PyObject *BufferObj[MAXARGS];
	long     offset[MAXARGS];
	
	if (!PyArg_ParseTuple(args, "lllO",
			      &niter, &ninargs, &noutargs, &DataArgs))
		return PyErr_Format(_Error,
				    "%s: Problem with argument list", me->descr.name);
	
	/* check consistency of stated inputs/outputs and supplied buffers */
	pnargs = PyObject_Length(DataArgs);
	if ((pnargs != (ninargs+noutargs)) || (pnargs > MAXARGS)) 
		return PyErr_Format(_Error,
				    "%s: wrong buffer count for function", me->descr.name);
	
	/* Unpack buffers and offsets, get data pointers */
	for (i=0; i<pnargs; i++) {
		ArgTuple = PySequence_GetItem(DataArgs, i);
		Py_DECREF(ArgTuple);
		if (!PyArg_ParseTuple(ArgTuple, "Ol", &BufferObj[i], &offset[i]))
			return PyErr_Format(_Error,
					    "%s: Problem with buffer/offset tuple", me->descr.name);
	}
	return NA_callCUFuncCore(self, niter, ninargs, noutargs, BufferObj, offset);
}


/* Handle "calling" the cfunc object at the python level. 
   Dispatch the call to the appropriate python-c wrapper based
   on the cfunc type.  Do this dispatch to avoid having to
   check that python code didn't somehow create a mismatch between
   cfunc and wrapper.
*/
static PyObject *
cfunc_call(PyObject *self, PyObject *argsTuple, PyObject *argsDict)
{
	CfuncObject *me = (CfuncObject *) self;
	switch(me->descr.type) {
	case CFUNC_UFUNC:
		return callCUFunc(self, argsTuple);
		break;
	case CFUNC_STRIDING:
		return callStrideConvCFunc(self, argsTuple);
		break;
	case CFUNC_NSTRIDING:
		return callStridingCFunc(self, argsTuple);
	case CFUNC_FROM_PY_VALUE:
		return NumTypeFromPyValue(self, argsTuple);
		break;
	case CFUNC_AS_PY_VALUE:
		return NumTypeAsPyValue(self, argsTuple);
		break;
	default:
		return PyErr_Format( _Error,
		     "cfunc_call: Can't dispatch cfunc '%s' with type: %d.", 
		     me->descr.name, me->descr.type);
	}
}

static PyTypeObject CfuncType = {
    PyObject_HEAD_INIT(NULL)
    0,
    "Cfunc",
    sizeof(CfuncObject),
    0,
    cfunc_dealloc, /*tp_dealloc*/
    0,          /*tp_print*/
    0,          /*tp_getattr*/
    0,          /*tp_setattr*/
    0,          /*tp_compare*/
    cfunc_repr, /*tp_repr*/
    0,          /*tp_as_number*/
    0,          /*tp_as_sequence*/
    0,          /*tp_as_mapping*/
    0,          /*tp_hash */
    cfunc_call, /* tp_call */
};


/* CfuncObjects are created at the c-level only.  They ensure that each
cfunc is called via the correct python-c-wrapper as defined by its 
CfuncDescriptor.  The wrapper, in turn, does conversions and buffer size
and alignment checking.  Allowing these to be created at the python level
would enable them to be created *wrong* at the python level, and thereby
enable python code to *crash* python. 
*/ 
static PyObject*
NA_new_cfunc(CfuncDescriptor *cfd)
{
    CfuncObject* cfunc;
    
    CfuncType.ob_type = &PyType_Type;  /* Should be done once at init.
					  Do now since there is no init. */

    cfunc = PyObject_New(CfuncObject, &CfuncType);
    
    if (!cfunc) {
	    return PyErr_Format(_Error,
			       "NA_new_cfunc: failed creating '%s'",
			       cfd->name);
    }

    cfunc->descr = *cfd;

    return (PyObject*)cfunc;
}

static int NA_add_cfunc(PyObject *dict, char *keystr, CfuncDescriptor *descr) 
{
	PyObject *c = (PyObject *) NA_new_cfunc(descr);
	if (!c) return -1;
	return PyDict_SetItemString(dict, keystr, c);
}


#define WITHIN32(v, f) (((v) >= f##_MIN32) && ((v) <= f##_MAX32))
#define WITHIN64(v, f) (((v) >= f##_MIN64) && ((v) <= f##_MAX64))

static Bool 
NA_IeeeMask32( Float32 f, Int32 mask)
{
	Int32 category;
	UInt32 v = *(UInt32 *) &f;

	if (v & BIT(31)) {
		if (WITHIN32(v, NEG_NORMALIZED)) {
			category = MSK_NEG_NOR;
		} else if (WITHIN32(v, NEG_DENORMALIZED)) {
			category = MSK_NEG_DEN;
		} else if (WITHIN32(v, NEG_SIGNAL_NAN)) {
			category = MSK_NEG_SNAN;
		} else if (WITHIN32(v, NEG_QUIET_NAN)) {
			category = MSK_NEG_QNAN;
		} else if (v == NEG_INFINITY_MIN32) {
			category = MSK_NEG_INF;
		} else if (v == NEG_ZERO_MIN32) {
			category = MSK_NEG_ZERO;
		} else if (v == INDETERMINATE_MIN32) {
			category = MSK_INDETERM;
		} else {
			category = MSK_BUG;
		}
	} else {
		if (WITHIN32(v, POS_NORMALIZED)) {
			category = MSK_POS_NOR;
		} else if (WITHIN32(v, POS_DENORMALIZED)) {
			category = MSK_POS_DEN;
		} else if (WITHIN32(v, POS_SIGNAL_NAN)) {
			category = MSK_POS_SNAN;
		} else if (WITHIN32(v, POS_QUIET_NAN)) {
			category = MSK_POS_QNAN;
		} else if (v == POS_INFINITY_MIN32) {
			category = MSK_POS_INF;
		} else if (v == POS_ZERO_MIN32) {
			category = MSK_POS_ZERO;
		} else {
			category = MSK_BUG;
		}
	}	
	return (category & mask) != 0;	
}

static Bool 
NA_IeeeMask64( Float64 f, Int32 mask)
{
	Int32 category;
	UInt64 v = *(UInt64 *) &f;

	if (v & BIT(63)) {
		if (WITHIN64(v, NEG_NORMALIZED)) {
			category = MSK_NEG_NOR;
		} else if (WITHIN64(v, NEG_DENORMALIZED)) {
			category = MSK_NEG_DEN;
		} else if (WITHIN64(v, NEG_SIGNAL_NAN)) {
			category = MSK_NEG_SNAN;
		} else if (WITHIN64(v, NEG_QUIET_NAN)) {
			category = MSK_NEG_QNAN;
		} else if (v == NEG_INFINITY_MIN64) {
			category = MSK_NEG_INF;
		} else if (v == NEG_ZERO_MIN64) {
			category = MSK_NEG_ZERO;
		} else if (v == INDETERMINATE_MIN64) {
			category = MSK_INDETERM;
		} else {
			category = MSK_BUG;
		}
	} else {
		if (WITHIN64(v, POS_NORMALIZED)) {
			category = MSK_POS_NOR;
		} else if (WITHIN64(v, POS_DENORMALIZED)) {
			category = MSK_POS_DEN;
		} else if (WITHIN64(v, POS_SIGNAL_NAN)) {
			category = MSK_POS_SNAN;
		} else if (WITHIN64(v, POS_QUIET_NAN)) {
			category = MSK_POS_QNAN;
		} else if (v == POS_INFINITY_MIN64) {
			category = MSK_POS_INF;
		} else if (v == POS_ZERO_MIN64) {
			category = MSK_POS_ZERO;
		} else {
			category = MSK_BUG;
		}
	}	
	return (category & mask) != 0;	
}

static Bool 
NA_IeeeSpecial32( Float32 *f, Int32 *mask)
{
	return NA_IeeeMask32(*f, *mask);
}

static Bool 
NA_IeeeSpecial64( Float64 *f, Int32 *mask)
{
	return NA_IeeeMask64(*f, *mask);
}

static double numarray_zero = 0.0;

static double raiseDivByZero(void)
{
  return 1.0/numarray_zero;
}

static double raiseNegDivByZero(void)
{
  return -1.0/numarray_zero;
}

static double num_log(double x)
{
   if (x == 0.0)
       return raiseNegDivByZero();
   else
      return log(x);
}

static double num_log10(double x)
{
   if (x == 0.0)
     return raiseNegDivByZero();
   else
     return log10(x);
}

static double num_pow(double x, double y)
{
   int z = (int) y;
   if ((x < 0.0) && (y != z))
     return raiseDivByZero();
   else
     return pow(x, y);
}

/* Inverse hyperbolic trig functions from Numeric */
static double num_acosh(double x)
{
    return log(x + sqrt((x-1.0)*(x+1.0)));
}

static double num_asinh(double xx)
{
    double x;
    int sign;
    if (xx < 0.0) {
        sign = -1;
        x = -xx;
    }
    else {
        sign = 1;
        x = xx;
    }
    return sign*log(x + sqrt(x*x+1.0));
}

static double num_atanh(double x)
{
    return 0.5*log((1.0+x)/(1.0-x));
}

/* NUM_CROUND (in numcomplex.h) also calls num_round */
static double num_round(double x)
{
    return (x >= 0) ? floor(x+0.5) : ceil(x-0.5);
}

/* The following routine is used in the event of a detected integer *
** divide by zero so that a floating divide by zero is generated.   *
** This is done since numarray uses the floating point exception    *
** sticky bits to detect errors. The last bit is an attempt to      *
** prevent optimization of the divide by zero away, the input value *
** should always be 0                                               *
*/

static int int_dividebyzero_error(long value, long unused) {
    double dummy;
    dummy = 1./numarray_zero;
    if (dummy) /* to prevent optimizer from eliminating expression */
        return 0;
    else
        return 1;
}

/* Likewise for Integer overflows */
#if defined(linux)
static int int_overflow_error(Float64 value) { /* For x86_64 */
	feraiseexcept(FE_OVERFLOW);
	return (int) value;
}
#else
static int int_overflow_error(Float64 value) {
	double dummy;
	dummy = pow(1.e10, fabs(value/2));
	if (dummy) /* to prevent optimizer from eliminating expression */
		return (int) value;
	else
		return 1;
}
#endif

static int umult64_overflow(UInt64 a, UInt64 b)
{
        UInt64 ah, al, bh, bl, w, x, y, z;

	ah = (a >> 32);
	al = (a & 0xFFFFFFFFL);
	bh = (b >> 32);
	bl = (b & 0xFFFFFFFFL);

        /* 128-bit product:  z*2**64 + (x+y)*2**32 + w  */
	w = al*bl;
	x = bh*al;
	y = ah*bl;
	z = ah*bh;

	/* *c = ((x + y)<<32) + w; */
	return z || (x>>32) || (y>>32) ||
               (((x & 0xFFFFFFFFL) + (y & 0xFFFFFFFFL) + (w >> 32)) >> 32);
}

static int smult64_overflow(Int64 a0, Int64 b0)
{
	UInt64 a, b;
        UInt64 ah, al, bh, bl, w, x, y, z;

        /* Convert to non-negative quantities */
	if (a0 < 0) { a = -a0; } else { a = a0; }
	if (b0 < 0) { b = -b0; } else { b = b0; }

	ah = (a >> 32);
	al = (a & 0xFFFFFFFFL);
	bh = (b >> 32);
	bl = (b & 0xFFFFFFFFL);

	w = al*bl;
	x = bh*al;
	y = ah*bl;
	z = ah*bh;

        /* 
         UInt64 c = ((x + y)<<32) + w;
	 if ((a0 < 0) ^ (b0 < 0))
	    *c = -c;
	 else 
	    *c = c
	*/

	return z || (x>>31) || (y>>31) ||
               (((x & 0xFFFFFFFFL) + (y & 0xFFFFFFFFL) + (w >> 32)) >> 31);
}



static PyObject *_Error;

void *libnumarray_API[] = {
	(void*) getBuffer,
	(void*) isBuffer,
	(void*) getWriteBufferDataPtr,
	(void*) isBufferWriteable,
	(void*) getReadBufferDataPtr,
	(void*) getBufferSize,
	(void*) num_log,
	(void*) num_log10,
	(void*) num_pow,
	(void*) num_acosh,
	(void*) num_asinh,
	(void*) num_atanh,
	(void*) num_round,
	(void*) int_dividebyzero_error,
	(void*) int_overflow_error,
	(void*) umult64_overflow,
	(void*) smult64_overflow,
	(void*) NA_Done,
	(void*) NA_NewAll,
	(void*) NA_NewAllStrides,
	(void*) NA_New,
	(void*) NA_Empty,
	(void*) NA_NewArray,
	(void*) NA_vNewArray,
	(void*) NA_ReturnOutput,
	(void*) NA_getBufferPtrAndSize,
	(void*) NA_checkIo,
	(void*) NA_checkOneCBuffer,
	(void*) NA_checkNCBuffers,
	(void*) NA_checkOneStriding,
	(void*) NA_new_cfunc,
	(void*) NA_add_cfunc,
	(void*) NA_InputArray,
	(void*) NA_OutputArray,
	(void*) NA_IoArray,
	(void*) NA_OptionalOutputArray,
	(void*) NA_get_offset,
	(void*) NA_get_Float64,
	(void*) NA_set_Float64,
	(void*) NA_get_Complex64,
	(void*) NA_set_Complex64,
	(void*) NA_get_Int64,
	(void*) NA_set_Int64,
	(void*) NA_get1_Float64,
	(void*) NA_get2_Float64,
	(void*) NA_get3_Float64,
	(void*) NA_set1_Float64,
	(void*) NA_set2_Float64,
	(void*) NA_set3_Float64,
	(void*) NA_get1_Complex64,
	(void*) NA_get2_Complex64,
	(void*) NA_get3_Complex64,
	(void*) NA_set1_Complex64,
	(void*) NA_set2_Complex64,
	(void*) NA_set3_Complex64,
	(void*) NA_get1_Int64,
	(void*) NA_get2_Int64,
	(void*) NA_get3_Int64,
	(void*) NA_set1_Int64,
	(void*) NA_set2_Int64,
	(void*) NA_set3_Int64,
	(void*) NA_get1D_Float64,
	(void*) NA_set1D_Float64,
	(void*) NA_get1D_Int64,
	(void*) NA_set1D_Int64,
	(void*) NA_get1D_Complex64,
	(void*) NA_set1D_Complex64,
	(void*) NA_ShapeEqual,
	(void*) NA_ShapeLessThan,
	(void*) NA_ByteOrder,
	(void*) NA_IeeeSpecial32,
	(void*) NA_IeeeSpecial64,
	(void*) NA_updateDataPtr,
	(void*) NA_typeNoToName,
	(void*) NA_nameToTypeNo,
	(void*) NA_typeNoToTypeObject,
	(void*) NA_intTupleFromMaybeLongs,
	(void*) NA_maybeLongsFromIntTuple,
	(void*) NA_intTupleProduct,
	(void*) NA_isIntegerSequence,
	(void*) NA_setArrayFromSequence,
	(void*) NA_maxType,
	(void*) NA_isPythonScalar,
	(void*) NA_getPythonScalar,
	(void*) NA_setFromPythonScalar,
	(void*) NA_NDArrayCheck,
	(void*) NA_NumArrayCheck,
	(void*) NA_ComplexArrayCheck,
	(void*) NA_elements,
	(void*) NA_typeObjectToTypeNo,
	(void*) NA_copyArray,
	(void*) NA_copy,
	(void*) NA_getType,
	(void*) NA_callCUFuncCore,
	(void*) NA_callStrideConvCFuncCore,
	(void*) NA_stridesFromShape,
	(void*) NA_OperatorCheck,
	(void*) NA_ConverterCheck,
	(void*) NA_UfuncCheck,
	(void*) NA_CfuncCheck,
	(void*) NA_getByteOffset,
	(void*) NA_swapAxes,
	(void*) NA_initModuleGlobal,
	(void*) NA_NumarrayType,
	(void*) NA_NewAllFromBuffer,
	(void*) NA_alloc1D_Float64,
	(void*) NA_alloc1D_Int64,
	(void*) NA_updateAlignment,
	(void*) NA_updateContiguous,
	(void*) NA_updateStatus,
	(void*) NA_NumArrayCheckExact,
	(void*) NA_NDArrayCheckExact,
	(void*) NA_OperatorCheckExact,
	(void*) NA_ConverterCheckExact,
	(void*) NA_UfuncCheckExact,
	(void*) NA_CfuncCheckExact,
	(void*) NA_getArrayData,
	(void*) NA_updateByteswap,
	(void*) NA_DescrFromType,
	(void*) NA_Cast,
	(void*) NA_checkFPErrors,
	(void*) NA_clearFPErrors,
	(void*) NA_checkAndReportFPErrors,
	(void*) NA_IeeeMask32,
	(void*) NA_IeeeMask64,
	(void*) _NA_callStridingHelper,
	(void*) NA_FromDimsStridesDescrAndData,
	(void*) NA_FromDimsTypeAndData,
	(void*) NA_FromDimsStridesTypeAndData,
	(void*) NA_scipy_typestr,
	(void*) NA_FromArrayStruct
};

#if (!defined(METHOD_TABLE_EXISTS))
static PyMethodDef _libnumarrayMethods[] = {
    {NULL,      NULL}        /* Sentinel */
};
#endif

/* platform independent*/
#ifdef MS_WIN32
__declspec(dllexport)
#endif

/* boiler plate API init */
PyMODINIT_FUNC init_capi(void)
{
    PyObject *m = Py_InitModule("_capi", _libnumarrayMethods);
    PyObject *c_api_object;

    _Error = PyErr_NewException("numpy.numarray._capi.error", NULL, NULL);

    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)libnumarray_API, NULL);

    if (c_api_object != NULL) {
      /* Create a name for this object in the module's namespace */
      PyObject *d = PyModule_GetDict(m);

      PyDict_SetItemString(d, "_C_API", c_api_object);
      PyDict_SetItemString(d, "error", _Error);
      Py_DECREF(c_api_object);
    } else {
        return;
    }
    if (PyModule_AddObject(m, "__version__", 
                           PyString_FromString("0.9")) < 0) return;
    return;
}


