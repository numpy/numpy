
__all__ = ['PythonCAPIType', 'PyTypeInterface']

from wrapper_base import *
from parser.api import CHAR_BIT, Module, declaration_type_spec, \
     TypeDecl, TypeStmt, Subroutine, Function, Integer, Real,\
     DoublePrecision, Complex, DoubleComplex, Logical, Character, \
     Byte

class PyTypeInterface:

    def __init__(self, typedecl):
        if isinstance(typedecl, TypeStmt):
            typedecl = typedecl.get_type_decl(typedecl.name)
        if isinstance(typedecl, TypeDecl):
            self.name = name = typedecl.name
            tname = 'f2py_type_%s_' % (name)
        else:
            if isinstance(typedecl,(Integer,Byte)):
                tname = 'npy_int'
            elif isinstance(typedecl,(Real, DoublePrecision)):
                tname = 'npy_float'
            elif isinstance(typedecl,(Complex, DoubleComplex)):
                tname = 'npy_complex'
            elif isinstance(typedecl,Logical):
                tname = 'f2py_bool'
            elif isinstance(typedecl,Character):
                tname = 'f2py_string'
            else:
                raise NotImplementedError,`typedecl.__class__`
        bitsize = typedecl.get_bit_size()
        self.ctype = ctype = '%s%s' % (tname,bitsize)
        self.bits = bitsize
        self.bytes = bitsize / CHAR_BIT

        if isinstance(typedecl, TypeDecl):
            self.otype = '%sObject' % (ctype)
            self.ftype = 'TYPE(%s)' % (name)
        return
    

class PythonCAPIType(WrapperBase):
    """
    Fortran type hooks.
    """
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        if isinstance(typedecl, tuple(declaration_type_spec)):
            if isinstance(typedecl, TypeStmt):
                type_decl = typedecl.get_type_decl(typedecl.name)
                assert type_decl is not None,"%s %s" % (typedecl,typedecl.name)
                PythonCAPIDerivedType(parent, type_decl)
            else:
                PythonCAPIIntrinsicType(parent, typedecl)
        elif isinstance(typedecl, TypeDecl):
            PythonCAPIDerivedType(parent, typedecl)
        else:
            raise NotImplementedError,`self.__class__,typedecl.__class__`
        return

class PythonCAPIIntrinsicType(WrapperBase):
    """
    Fortran intrinsic type hooks.
    """

    capi_code_template_scalar = '''
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
  PyObject* obj = PyArrayScalar_New(%(Cls)s);
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_from_%(ctype)s(value=%%"%(CTYPE)s_FMT")\\n",*value);
#endif
  if (obj==NULL) /* TODO: set exception */ return NULL;
  PyArrayScalar_ASSIGN(obj,%(Cls)s,*value);
  return obj;
}

static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (obj==NULL) ;
  else if (PyArray_IsScalar(obj,%(Cls)s)) {
    *value = PyArrayScalar_VAL(obj,%(Cls)s);
    return_value = 1;
  }
  else if (PySequence_Check(obj)) {
    if (PySequence_Size(obj)==1)
      return_value = pyobj_to_%(ctype)s(PySequence_GetItem(obj,0),value);
  } else {
    PyObject* sc = Py%(Cls)sArrType_Type.tp_new(
      &Py%(Cls)sArrType_Type,Py_BuildValue("(O)",obj),NULL);
    if (sc==NULL) ;
    else if (PyArray_IsScalar(sc, Generic))
      return_value = pyobj_to_%(ctype)s(sc,value);
    else
      return_value = pyobj_to_%(ctype)s(PyArray_ScalarFromObject(sc),value);
  }
  if (!return_value && !PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C %(ctype)s"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  if (PyErr_Occurred()) {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
    else
      fprintf(stderr,"pyobj_to_%(ctype)s: PyErr_Occurred()=%%p\\n", PyErr_Occurred());
  } else {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s: value=%%"%(CTYPE)s_FMT"\\n", *value);
    else
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
  }
#endif
  return return_value;
}
'''

    capi_code_template_complex_scalar = '''
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
  PyObject* obj = PyArrayScalar_New(%(Cls)s);
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_from_%(ctype)s(value=(%%"%(FCTYPE)s_FMT",%%"%(FCTYPE)s_FMT"))\\n",value->real, value->imag);
#endif
  if (obj==NULL) /* TODO: set exception */ return NULL;
  PyArrayScalar_ASSIGN(obj,%(Cls)s,*value);
  return obj;
}

static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (obj==NULL) ;
  else if (PyArray_IsScalar(obj,%(Cls)s)) {
    value->real = PyArrayScalar_VAL(obj,%(Cls)s).real;
    value->imag = PyArrayScalar_VAL(obj,%(Cls)s).imag;
    return_value = 1;
  }
  else if (PySequence_Check(obj)) {
    if (PySequence_Size(obj)==1)
      return_value = pyobj_to_%(ctype)s(PySequence_GetItem(obj,0),value);
    else if (PySequence_Size(obj)==2) {
      return_value = pyobj_to_%(fctype)s(PySequence_GetItem(obj,0),&(value->real))
                     && pyobj_to_%(fctype)s(PySequence_GetItem(obj,1),&(value->imag));
    }
  } else {
    PyObject* sc = Py%(Cls)sArrType_Type.tp_new(
      &Py%(Cls)sArrType_Type,Py_BuildValue("(O)",obj),NULL);
    if (sc==NULL) ;
    else if (PyArray_IsScalar(sc, Generic))
      return_value = pyobj_to_%(ctype)s(sc,value);
    else
      return_value = pyobj_to_%(ctype)s(PyArray_ScalarFromObject(sc),value);
  }
  if (!return_value && !PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C %(ctype)s"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  if (PyErr_Occurred()) {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
    else
      fprintf(stderr,"pyobj_to_%(ctype)s: PyErr_Occurred()=%%p\\n", PyErr_Occurred());
  } else {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s: value=(%%"%(FCTYPE)s_FMT",%%"%(FCTYPE)s_FMT")\\n",
      value->real, value->imag);
    else
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
  }
#endif
  return return_value;
}
'''

    capi_code_template_logical_scalar = '''
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_from_%(ctype)s(value=%%"%(ICTYPE)s_FMT")\\n",*value);
#endif
  if (*value) {
    PyArrayScalar_RETURN_TRUE;
  } else {
    PyArrayScalar_RETURN_FALSE;
  }
}
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (obj==NULL) ;
  else if (PyArray_IsScalar(obj,Bool)) {
    *value = PyArrayScalar_VAL(obj,Bool);
    return_value = 1;
  } else {
    switch (PyObject_IsTrue(obj)) {
      case 0: *value = 0; return_value = 1; break;
      case -1: break;
      default: *value = 1; return_value = 1;
    }
  }
  if (!return_value && !PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C %(ctype)s"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  if (PyErr_Occurred()) {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
    else
      fprintf(stderr,"pyobj_to_%(ctype)s: PyErr_Occurred()=%%p\\n", PyErr_Occurred());
  } else {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s: value=%%"%(ICTYPE)s_FMT"\\n", *value);
    else
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
  }
#endif
  return return_value;
}
'''
    capi_code_template_string_scalar = '''
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_from_%(ctype)s(value->data=\'%%s\')\\n",value->data);
#endif
  PyArray_Descr* descr = PyArray_DescrNewFromType(NPY_STRING);
  descr->elsize = %(bytes)s;
  PyObject* obj = PyArray_Scalar(value->data, descr, NULL);
  if (obj==NULL) /* TODO: set exception */ return NULL;
  return obj;
}

static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (PyString_Check(obj)) {
    int s = PyString_GET_SIZE(obj);
    memset(value->data, (int)\' \',%(bytes)s);
    return_value = !! strncpy(value->data,PyString_AS_STRING(obj),%(bytes)s);
    if (return_value && s<%(bytes)s) {
      memset(value->data + s, (int)\' \',%(bytes)s-s);
    }
  } else {
    return_value = pyobj_to_%(ctype)s(PyObject_Str(obj), value);
  }
  if (!return_value && !PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C %(ctype)s"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  if (PyErr_Occurred()) {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
    else
      fprintf(stderr,"pyobj_to_%(ctype)s: PyErr_Occurred()=%%p\\n", PyErr_Occurred());
  } else {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s: value->data=\'%%s\'\\n", value->data);
    else
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
  }
#endif
  return return_value;
}
'''
    capi_code_template_string0_scalar = '''
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_from_%(ctype)s(value->len=%%d, value->data=\'%%s\')\\n",value->len, value->data);
#endif
  PyArray_Descr* descr = PyArray_DescrNewFromType(NPY_STRING);
  descr->elsize = value->len;
  PyObject* obj = PyArray_Scalar(value->data, descr, NULL);
  if (obj==NULL) /* TODO: set exception */ return NULL;
  return obj;
}

static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (PyString_Check(obj)) {
    value->len = PyString_GET_SIZE(obj);
    value->data = malloc(value->len*sizeof(char));
    return_value = !! strncpy(value->data,PyString_AS_STRING(obj),value->len);
  } else {
    return_value = pyobj_to_%(ctype)s(PyObject_Str(obj), value);
  }
  if (!return_value && !PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C %(ctype)s"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  if (PyErr_Occurred()) {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
    else
      fprintf(stderr,"pyobj_to_%(ctype)s: PyErr_Occurred()=%%p\\n", PyErr_Occurred());
  } else {
    if (return_value)
      fprintf(stderr,"pyobj_to_%(ctype)s: value->len=%%d, value->data=\'%%s\'\\n", value->len, value->data);
    else
      fprintf(stderr,"pyobj_to_%(ctype)s:INCONSISTENCY with return_value=%%d and PyErr_Occurred()=%%p\\n",return_value, PyErr_Occurred());
  }
#endif
  return return_value;
}
''' 
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        self.name = name = typedecl.name
        ti = PyTypeInterface(typedecl)
        self.ctype = ctype = ti.ctype

        defined = parent.defined_types
        if ctype in defined:
            return
        defined.append(ctype)
        
        self.info('Generating interface for %s: %s' % (typedecl.__class__.__name__, ctype))
        self.parent = parent
        if isinstance(typedecl, (Integer,Byte,Real,DoublePrecision)):
            self.Cls = ctype[4].upper() + ctype[5:]
            self.capi_code_template = self.capi_code_template_scalar
        elif isinstance(typedecl, (Complex,DoubleComplex)):
            self.Cls = ctype[4].upper() + ctype[5:]
            PythonCAPIIntrinsicType(parent, typedecl.get_part_typedecl())
            ti1 = PyTypeInterface(typedecl.get_part_typedecl())
            self.fctype = ti1.ctype
            self.capi_code_template = self.capi_code_template_complex_scalar
        elif isinstance(typedecl, Logical):
            self.ictype = 'npy_int%s' % (typedecl.get_bit_size())
            self.header_template = '#define %(ctype)s %(ictype)s'
            self.capi_code_template = self.capi_code_template_logical_scalar
        elif isinstance(typedecl, Character):
            self.bits = bits = typedecl.get_bit_size()
            if bits:
                self.bytes = bits/CHAR_BIT
                self.header_template = '''
#include <string.h>
typedef struct { char data[%(bytes)s]; } %(ctype)s;
'''
                self.capi_code_template = self.capi_code_template_string_scalar
            else:
                self.header_template = '''
#include <string.h>
typedef struct { char* data; size_t len; } %(ctype)s;
'''
                self.capi_code_template = self.capi_code_template_string0_scalar
        else:
            raise NotImplementedError,`name,ctype`
        parent.apply_templates(self)
        return

class PythonCAPIDerivedType(WrapperBase):
    """
    Fortran 90 derived type hooks.
    """

    header_template_wrapper = '''\
#define %(otype)s_Check(obj) \\
    PyObject_TypeCheck((PyObject*)obj, &%(otype)sType)
#define %(init_func)s_f \\
    F_FUNC(%(init_func)s,%(INIT_FUNC)s)
'''

    typedef_template_wrapper = '''\
typedef void * %(ctype)s;
typedef struct {
  PyObject_HEAD
  %(ptrstruct_list)s
  %(ctype)s data;
} %(otype)s;
typedef void (*%(init_func)s_c_functype)(%(init_func_c_ctype_arg_clist)s);
'''

    typedef_template_importer = '''\
typedef void * %(ctype)s;
typedef struct {
  PyObject_HEAD
  %(ptrstruct_list)s
  %(ctype)s data;
} %(otype)s;
typedef int (*pyobj_to_%(ctype)s_inplace_functype)(PyObject*, %(otype)s** );
typedef int (*pyobj_to_%(ctype)s_functype)(PyObject*, %(otype)s* );
typedef PyObject* (*pyobj_from_%(ctype)s_functype)(%(ctype)s*);
#define %(otype)sType (*(PyTypeObject *)PyArray_API[0])
#define pyobj_from_%(ctype)s ((pyobj_from_%(ctype)s_functype)PyArray_API[1])
#define pyobj_to_%(ctype)s_inplace ((pyobj_to_%(ctype)s_inplace_functype)PyArray_API[2])
'''

    extern_template_wrapper = '''\
static PyTypeObject %(otype)sType;
extern void %(init_func)s_f(%(init_func)s_c_functype, void*, %(ctype)s);
'''

    objdecl_template_wrapper = '''\
static PyMethodDef %(otype)s_methods[] = {
    %(type_method_list)s
    {NULL}  /* Sentinel */
};

static PyGetSetDef %(otype)s_getseters[] = {
    %(type_getseters_list)s
    {NULL}  /* Sentinel */
};

static PyTypeObject %(otype)sType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "%(modulename)s.%(name)s",                /*tp_name*/
    sizeof(%(otype)s),    /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)%(otype)s_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    %(otype)s_repr,            /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    "Fortran derived type %(name)s objects",        /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    %(otype)s_methods,          /* tp_methods */
    0 /*%(otype)s_members*/,    /* tp_members */
    %(otype)s_getseters,       /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)%(otype)s_init,      /* tp_init */
    0,                         /* tp_alloc */
    %(otype)s_new,                 /* tp_new */
};

void *F2PY_%(otype)s_API[] = {
  (void *) &%(otype)sType,
  (void *) pyobj_from_%(ctype)s,
  (void *) pyobj_to_%(ctype)s_inplace
};
'''

    objdecl_template_importer = '''\
static void **F2PY_%(otype)s_API;
'''
    module_init_template_wrapper = '''\
if (PyType_Ready(&%(otype)sType) < 0) goto capi_err;
PyModule_AddObject(f2py_module, "%(name)s", (PyObject *)&%(otype)sType);
{
  PyObject* c_api = PyCObject_FromVoidPtr((void *)F2PY_%(otype)s_API, NULL);
  PyModule_AddObject(f2py_module, "_%(NAME)s_API", c_api);
  if (PyErr_Occurred()) goto capi_err;
}
'''
    module_init_template_importer = '''\
{
  PyObject *c_api = NULL;
  PyObject *wrappermodule = PyImport_ImportModule("%(wrappermodulename)s");
  if (wrappermodule == NULL) goto capi_%(name)s_err;
  c_api = PyObject_GetAttrString(wrappermodule, "_%(NAME)s_API");
  if (c_api == NULL) {Py_DECREF(wrappermodule); goto capi_%(name)s_err;}
  if (PyCObject_Check(c_api)) {
      F2PY_%(otype)s_API = (void **)PyCObject_AsVoidPtr(c_api);
  }
  Py_DECREF(c_api);
  Py_DECREF(wrappermodule);
  if (F2PY_%(otype)s_API != NULL) goto capi_%(name)s_ok;
capi_%(name)s_err:
  PyErr_Print();
  PyErr_SetString(PyExc_ImportError, "%(wrappermodulename)s failed to import");
  return;
capi_%(name)s_ok:
  c_api = PyCObject_FromVoidPtr((void *)F2PY_%(otype)s_API, NULL);
  PyModule_AddObject(f2py_module, "_%(NAME)s_API", c_api);
  if (PyErr_Occurred()) goto capi_err;
}
'''

    c_code_template_wrapper = '''\
static void %(init_func)s_c(
               %(init_func_c_arg_clist)s) {
  %(init_func_c_body_list)s
}
'''

    capi_code_template_wrapper = '''\
static void %(otype)s_dealloc(%(otype)s* self) {
  if (self->data)
    PyMem_Free(self->data);
  self->ob_type->tp_free((PyObject*)self);
}

static int pyobj_to_%(ctype)s_inplace(PyObject *obj,
                                      %(otype)s** value_ptr) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (%(otype)s_Check(obj)) {
    *value_ptr = (%(otype)s*)obj;
    return_value = 1;
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s: return_value=%%d, PyErr_Occurred()=%%p\\n", return_value, PyErr_Occurred());
#endif
  return return_value;
}

static int pyobj_to_%(ctype)s(PyObject *obj,
                                   %(ctype)s* value_ptr) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (%(otype)s_Check(obj)) {
    if (!memcpy(value_ptr,((%(otype)s *)obj)->data, %(bytes)s)) {
      PyErr_SetString(PyExc_MemoryError,
         "failed to copy %(name)s instance memory to %(ctype)s object.");
    } else {
      return_value = 1;
    }
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s: return_value=%%d, PyErr_Occurred()=%%p\\n", return_value, PyErr_Occurred());
#endif
  return return_value;
}

static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value_ptr) {
  %(otype)s* obj = (%(otype)s*)(%(otype)sType.tp_alloc(&%(otype)sType, 0));
  if (obj == NULL)
    return NULL;
  obj->data = PyMem_Malloc(%(bytes)s);
  if (obj->data == NULL) {
    Py_DECREF(obj);
    return PyErr_NoMemory();
  }
  if (value_ptr) {
    if (!memcpy(obj->data, value_ptr, %(bytes)s)) {
      PyErr_SetString(PyExc_MemoryError,
         "failed to copy %(ctype)s object memory to %(name)s instance.");
    }
  }
  %(init_func)s_f(%(init_func)s_c, obj, obj->data);
  return (PyObject*)obj;
}

static PyObject * %(otype)s_new(PyTypeObject *type,
                                PyObject *args, PyObject *kwds)
{
  return pyobj_from_%(ctype)s(NULL);
}

static int %(otype)s_init(%(otype)s *self,
                          PyObject *capi_args, PyObject *capi_kwds)
{
   int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"%(otype)s_init()\\n");
#endif
   if (!PyArg_ParseTuple(capi_args,"%(attr_format_elist)s"
                                   %(attr_init_clist)s))
      return_value = -1;
   
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"%(otype)s_init: return_value=%%d, PyErr_Occurred()=%%p\\n", return_value, PyErr_Occurred());
#endif
   return return_value;
}

static PyObject * %(otype)s_as_tuple(%(otype)s * self) {
  return Py_BuildValue("%(as_tuple_format_elist)s"
                        %(as_tuple_arg_clist)s);
}

static PyObject * %(otype)s_repr(PyObject * self) {
  PyObject* r = PyString_FromString("%(name)s(");
  PyString_ConcatAndDel(&r, PyObject_Repr(%(otype)s_as_tuple((%(otype)s*)self)));
  PyString_ConcatAndDel(&r, PyString_FromString(")"));
  return r;
}

%(getset_func_list)s
'''

    fortran_code_template_wrapper = '''\
      subroutine %(init_func)s(init_func_c, self, obj)
      %(use_stmt_list)s
      %(type_decl_list)s
      external init_func_c
!     self is %(otype)s
      external self
      %(ftype)s obj
      call init_func_c(%(init_func_f_arg_clist)s)
      end
'''

    #module_method_template = ''''''

    _defined = []
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        ti = PyTypeInterface(typedecl)
        self.ctype = ctype = ti.ctype
        defined = parent.defined_types
        if ctype in defined:
            return
        defined.append(ctype)



        implement_wrappers = True
        if isinstance(typedecl.parent,Module) and typedecl.parent.name!=parent.modulename:
            implement_wrappers = False
            self.info('Using api for %s.%s: %s' % (parent.modulename, typedecl.name, ctype))
            self.wrappermodulename = typedecl.parent.name
        else:
            self.info('Generating interface for %s.%s: %s' % (parent.modulename, typedecl.name, ctype))
        
        parent.isf90 = True
        self.parent = parent
        self.name = name = typedecl.name
        self.otype = otype = ti.otype
        self.ctype = ctype = ti.ctype
        self.ctype_ptrs = self.ctype + '_ptrs'
        self.ftype = ti.ftype
        self.bytes = bytes = ti.bytes

        if not implement_wrappers:
            self.typedef_template = self.typedef_template_importer
            self.objdecl_template = self.objdecl_template_importer
            self.module_init_template = self.module_init_template_importer
        else:
            self.header_template = self.header_template_wrapper
            self.typedef_template = self.typedef_template_wrapper
            self.extern_template = self.extern_template_wrapper
            self.objdecl_template = self.objdecl_template_wrapper
            self.module_init_template = self.module_init_template_wrapper
            self.c_code_template = self.c_code_template_wrapper
            self.capi_code_template = self.capi_code_template_wrapper
            self.fortran_code_template = self.fortran_code_template_wrapper
            WrapperCPPMacro(parent, 'F_FUNC')

        self.init_func_f_arg_list = ['self']
        self.init_func_c_arg_list = ['%s *self' % (otype)]
        self.init_func_c_ctype_arg_list = ['%s *' % (otype)]
        self.init_func_c_body_list = []
        self.ptrstruct_list = []
        self.attr_decl_list = []
        self.attr_format_list = []
        self.attr_init_list = []
        self.as_tuple_format_list = []
        self.as_tuple_arg_list = []
        self.getset_func_list = []
        self.type_getseters_list = []
        for n in typedecl.a.component_names:
            v = typedecl.a.components[n]
            t = v.get_typedecl()
            ti1 = PyTypeInterface(t)
            PythonCAPIType(parent, t)
            ct = ti1.ctype
            parent.add(t)
            self.ptrstruct_list.append('%s* %s_ptr;' % (ct, n))
            self.init_func_f_arg_list.append('obj %% %s' % (n))
            self.init_func_c_arg_list.append('\n%s * %s_ptr' % (ct, n))
            self.init_func_c_ctype_arg_list.append('\n%s *' % (ct))
            self.init_func_c_body_list.append('''\
if (!((void*)%(n)s_ptr >= self->data
      && (void*)%(n)s_ptr < self->data + %(bytes)s ))
  fprintf(stderr,"INCONSISTENCY IN %(name)s WRAPPER: "
                 "self->data=%%p <= %(n)s_ptr=%%p < self->data+%(bytes)s=%%p\\n",
                 self->data, %(n)s_ptr, self->data + %(bytes)s);
self->%(n)s_ptr = %(n)s_ptr;
''' % (locals()))
            self.attr_format_list.append('O&')
            self.attr_init_list.append('\npyobj_to_%s, self->%s_ptr' % (ct,n))
            self.as_tuple_format_list.append('O&')
            self.as_tuple_arg_list.append('\npyobj_from_%s, self->%s_ptr' % (ct, n))
            self.getset_func_list.append('''\
static PyObject * %(otype)s_get_%(n)s(%(otype)s *self,
                                      void *closure) {
  return pyobj_from_%(ct)s(self->%(n)s_ptr);
}
static int %(otype)s_set_%(n)s(%(otype)s *self,
                               PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Cannot delete %(name)s attribute %(n)s");
    return -1;
  }
  if (pyobj_to_%(ct)s(value, self->%(n)s_ptr))
    return 0;
  return -1;
}
''' % (locals()))
            self.type_getseters_list.append('{"%(n)s",(getter)%(otype)s_get_%(n)s, (setter)%(otype)s_set_%(n)s,\n "component %(n)s",NULL},' % (locals()))
        if self.attr_init_list: self.attr_init_list.insert(0,'')
        if self.as_tuple_arg_list: self.as_tuple_arg_list.insert(0,'')
        self.init_func = self.ctype + '_init'

        self.type_method_list = []
        self.type_method_list.append('{"as_tuple",(PyCFunction)%(otype)s_as_tuple,METH_NOARGS,\n "Return %(name)s components as tuple."},' % (self.__dict__))

        self.use_stmt_list = []
        self.type_decl_list = []
        if isinstance(typedecl.parent, Module):
            self.use_stmt_list.append('use %s' % (typedecl.parent.name))
        elif isinstance(typedecl.parent, (Subroutine, Function)):
            self.type_decl_list.append(typedecl.asfix())
        else:
            raise NotImplementedError,'types declared in '+typedecl.parent.__class__.__name__
        parent.apply_templates(self)
        return
