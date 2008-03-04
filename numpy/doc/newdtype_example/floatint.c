
#include "Python.h"
#include "structmember.h" /* for offsetof macro if needed */
#include "numpy/arrayobject.h"


/* Use a Python float as the cannonical type being added
*/

typedef struct _floatint {
    PyObject_HEAD
    npy_int32 first;
    npy_int32 last;
} PyFloatIntObject;

static PyTypeObject PyFloatInt_Type = {
        PyObject_HEAD_INIT(NULL)
        0,                                          /*ob_size*/
        "floatint.floatint",                         /*tp_name*/
        sizeof(PyFloatIntObject),                   /*tp_basicsize*/
};

static PyArray_ArrFuncs _PyFloatInt_Funcs;

#define _ALIGN(type) offsetof(struct {char c; type v;},v)

/* Need to inherit from scalar type... */

static PyArray_Descr _PyFloatInt_Dtype = {
    PyObject_HEAD_INIT(NULL)
    &PyFloatInt_Type,
    'f',
    '0',
    '=',
    0,
    0,
    sizeof(double),
    _ALIGN(double),
    NULL,
    NULL,
    NULL,
    &_PyFloatInt_Funcs
};

static void
twoint_copyswap(void *dst, void *src, int swap, void *arr)
{
    if (src != NULL) 
	memcpy(dst, src, sizeof(double));
    
    if (swap) {
	register char *a, *b, c;
	a = (char *)dst;
	b = a + 7;
	c = *a; *a++ = *b; *b-- = c;
	c = *a; *a++ = *b; *b-- = c;
	c = *a; *a++ = *b; *b-- = c;
	c = *a; *a++ = *b; *b   = c;	
    }
}

static PyObject *
twoint_getitem(char *ip, PyArrayObject *ap) {
    npy_int32 a[2];
 
    if ((ap==NULL) || PyArray_ISBEHAVED_RO(ap)) {
	a[0] = *((npy_int32 *)ip);
	a[1] = *((npy_int32 *)ip + 1);
    }
    else {
	ap->descr->f->copyswap(a, ip, !PyArray_ISNOTSWAPPED(ap),
			       ap);
    }
    return Py_BuildValue("(ii)", a[0], a[1]);
}

static int
twoint_setitem(PyObject *op, char *ov, PyArrayObject *ap) {
    npy_int32 a[2];
    
    if (!PyTuple_Check(op)) {
	PyErr_SetString(PyExc_TypeError, "must be a tuple");
	return -1;
    }
    if (!PyArg_ParseTuple(op, "ii", a, a+1)) return -1;

    if (ap == NULL || PyArray_ISBEHAVED(ap)) {
	memcpy(ov, a, sizeof(double));
    }
    else {
	ap->descr->f->copyswap(ov, a, !PyArray_ISNOTSWAPPED(ap),
			       ap);
    }
    return 0;
}

static PyArray_Descr * _register_dtype(void)
{
    int userval;
    PyArray_InitArrFuncs(&_PyFloatInt_Funcs); 
    /* Add copyswap, copyswapn,
       nonzero, getitem, setitem, cast */
    _PyFloatInt_Funcs.copyswap = twoint_copyswap;
    _PyFloatInt_Funcs.getitem = (PyArray_GetItemFunc *)twoint_getitem;
    _PyFloatInt_Funcs.setitem = (PyArray_SetItemFunc *)twoint_setitem; 
    _PyFloatInt_Dtype.ob_type = &PyArrayDescr_Type;

    userval = PyArray_RegisterDataType(&_PyFloatInt_Dtype);
    return PyArray_DescrFromType(userval);
}


/* Initialization function for the module (*must* be called init<name>) */

PyMODINIT_FUNC initfloatint(void) {
    PyObject *m, *d, *s;
    PyArray_Descr *dtype;

    /* Create the module and add the functions */
    m = Py_InitModule("floatint", NULL);

    /* Import the array objects */
    import_array();


    /* Initialize the new float type */
    
    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    if (PyType_Ready(&PyFloat_Type) < 0) return;
    PyFloatInt_Type.tp_base = &PyFloat_Type;
    if (PyType_Ready(&PyFloatInt_Type) < 0) return;

    dtype = _register_dtype();
    Py_XINCREF(dtype);
    if (dtype != NULL) {
	PyDict_SetItemString(d, "floatint_type", (PyObject *)dtype);
    }
    Py_INCREF(&PyFloatInt_Type);
    PyDict_SetItemString(d, "floatint", (PyObject *)&PyFloatInt_Type);
    return;
}
