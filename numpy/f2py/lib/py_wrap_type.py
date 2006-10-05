
__all__ = ['PythonCAPIType']

from wrapper_base import *
from parser.api import CHAR_BIT, Module, declaration_type_spec, TypeDecl

class PythonCAPIType(WrapperBase):
    """
    Fortran type hooks.
    """
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        if isinstance(typedecl, tuple(declaration_type_spec)):
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
    _defined = []
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        self.name = name = typedecl.name
        if name in self._defined:
            return
        self._defined.append(name)
        self.info('Generating interface for %s: %s' % (typedecl.__class__, name))

        ctype = typedecl.get_c_type()

        if ctype.startswith('npy_'):
            WrapperCCode(parent, 'pyobj_from_%s' % (ctype))
            WrapperCCode(parent, 'pyobj_to_%s' % (ctype))
            return
        
        if not ctype.startswith('f2py_type_'):
            raise NotImplementedError,`name,ctype`
        return

class PythonCAPIDerivedType(WrapperBase):
    """
    Fortran 90 derived type hooks.
    """

    header_template = '''\
#define %(oname)sObject_Check(obj) \\
    PyObject_TypeCheck((PyObject*)obj, &%(oname)sType)
#define %(init_func)s_f \\
    F_FUNC(%(init_func)s,%(INIT_FUNC)s)
'''

    typedef_template = '''\
typedef void * %(ctype)s;
typedef struct {
  PyObject_HEAD
  %(ptrstruct_list)s
  %(ctype)s data;
} %(oname)sObject;
'''

    extern_template = '''\
static PyTypeObject %(oname)sType;
'''

    objdecl_template = '''\
static PyMethodDef %(oname)s_methods[] = {
    %(type_method_list)s
    {NULL}  /* Sentinel */
};

static PyGetSetDef %(oname)s_getseters[] = {
    %(type_getseters_list)s
    {NULL}  /* Sentinel */
};

static PyTypeObject %(oname)sType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "%(name)s",                /*tp_name*/
    sizeof(%(oname)sObject),    /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)%(oname)s_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    %(oname)s_repr,            /*tp_repr*/
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
    %(oname)s_methods,          /* tp_methods */
    0 /*%(oname)s_members*/,    /* tp_members */
    %(oname)s_getseters,       /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)%(oname)s_init,      /* tp_init */
    0,                         /* tp_alloc */
    %(oname)s_new,                 /* tp_new */
};
'''

    module_init_template = '''\
if (PyType_Ready(&%(oname)sType) < 0)
  return;
PyModule_AddObject(f2py_module, "%(name)s",
                                (PyObject *)&%(oname)sType);
'''

    c_code_template = '''\
static void %(init_func)s_c(
               %(init_func_c_arg_clist)s) {
  %(init_func_c_body_list)s
}
'''

    capi_code_template = '''\
static void %(oname)s_dealloc(%(oname)sObject* self) {
  if (self->data)
    PyMem_Free(self->data);
  self->ob_type->tp_free((PyObject*)self);
}

static int pyobj_to_%(ctype)s(PyObject *obj,
                              %(ctype)s* value_ptr) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (%(oname)sObject_Check(obj)) {
    if (!memcpy(value_ptr,((%(oname)sObject *)obj)->data, %(byte_size)s)) {
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
  %(oname)sObject* obj = (%(oname)sObject*)(%(oname)sType.tp_alloc(&%(oname)sType, 0));
  if (obj == NULL)
    return NULL;
  obj->data = PyMem_Malloc(%(byte_size)s);
  if (obj->data == NULL) {
    Py_DECREF(obj);
    return PyErr_NoMemory();
  }
  if (value_ptr) {
    if (!memcpy(obj->data, value_ptr, %(byte_size)s)) {
      PyErr_SetString(PyExc_MemoryError,
         "failed to copy %(ctype)s object memory to %(name)s instance.");
    }
  }
  %(init_func)s_f(%(init_func)s_c, obj, obj->data);
  return (PyObject*)obj;
}

static PyObject * %(oname)s_new(PyTypeObject *type,
                                PyObject *args, PyObject *kwds)
{
  return pyobj_from_%(ctype)s(NULL);
}

static int %(oname)s_init(%(oname)sObject *self,
                          PyObject *capi_args, PyObject *capi_kwds)
{
   int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"%(oname)s_init()\\n");
#endif
   if (!PyArg_ParseTuple(capi_args,"%(attr_format_elist)s"
                                   %(attr_init_clist)s))
      return_value = -1;                             
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"%(oname)s_init: return_value=%%d, PyErr_Occurred()=%%p\\n", return_value, PyErr_Occurred());
#endif
   return return_value;
}

static PyObject * %(oname)s_as_tuple(%(oname)sObject * self) {
  return Py_BuildValue("%(as_tuple_format_elist)s"
                        %(as_tuple_arg_clist)s);
}

static PyObject * %(oname)s_repr(PyObject * self) {
  PyObject* r = PyString_FromString("%(name)s(");
  PyString_ConcatAndDel(&r, PyObject_Repr(%(oname)s_as_tuple((%(oname)sObject*)self)));
  PyString_ConcatAndDel(&r, PyString_FromString(")"));
  return r;
}

%(getset_func_list)s
'''

    fortran_code_template = '''\
      subroutine %(init_func)s(init_func_c, self, obj)
      %(use_stmt_list)s
      external init_func_c
!     self is %(oname)sObject
      external self
      %(ftype)s obj
      call init_func_c(%(init_func_f_arg_clist)s)
      end
'''

    #module_method_template = ''''''

    _defined = []
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        name = typedecl.name
        if name in self._defined:
            return
        self._defined.append(name)
        self.info('Generating interface for %s: %s' % (typedecl.__class__, name))

        self.name = name
        self.oname = oname = 'f2py_' + name
        self.ctype = typedecl.get_c_type()
        self.ctype_ptrs = self.ctype + '_ptrs'
        self.ftype = typedecl.get_f_type()
        self.byte_size = byte_size = typedecl.get_bit_size() / CHAR_BIT
        WrapperCPPMacro(parent, 'F_FUNC')

        self.init_func_f_arg_list = ['self']
        self.init_func_c_arg_list = ['%sObject *self' % (self.oname)]
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
            PythonCAPIType(t)
            ct = t.get_c_type()
            on = 'f2py_' + t.name
            parent.add(t)
            self.ptrstruct_list.append('%s* %s_ptr;' % (ct, n))
            self.init_func_f_arg_list.append('obj %% %s' % (n))
            self.init_func_c_arg_list.append('\n%s * %s_ptr' % (ct, n))
            self.init_func_c_body_list.append('''\
if (!((void*)%(n)s_ptr >= self->data
      && (void*)%(n)s_ptr < self->data + %(byte_size)s ))
  fprintf(stderr,"INCONSISTENCY IN %(name)s WRAPPER: "
                 "self->data=%%p <= %(n)s_ptr=%%p < self->data+%(byte_size)s=%%p\\n",
                 self->data, %(n)s_ptr, self->data + %(byte_size)s);
self->%(n)s_ptr = %(n)s_ptr;
''' % (locals()))
            self.attr_format_list.append('O&')
            self.attr_init_list.append('\npyobj_to_%s, self->%s_ptr' % (ct,n))
            self.as_tuple_format_list.append('O&')
            self.as_tuple_arg_list.append('\npyobj_from_%s, self->%s_ptr' % (ct, n))
            self.getset_func_list.append('''\
static PyObject * %(oname)s_get_%(n)s(%(oname)sObject *self,
                                      void *closure) {
  return pyobj_from_%(ct)s(self->%(n)s_ptr);
}
static int %(oname)s_set_%(n)s(%(oname)sObject *self,
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
            self.type_getseters_list.append('{"%(n)s",(getter)%(oname)s_get_%(n)s, (setter)%(oname)s_set_%(n)s,\n "component %(n)s",NULL},' % (locals()))
        if self.attr_init_list: self.attr_init_list.insert(0,'')
        if self.as_tuple_arg_list: self.as_tuple_arg_list.insert(0,'')
        self.init_func = self.ctype + '_init'
        self.INIT_FUNC = self.init_func.upper()

        self.type_method_list = []
        self.type_method_list.append('{"as_tuple",(PyCFunction)%(oname)s_as_tuple,METH_NOARGS,\n "Return %(name)s components as tuple."},' % (self.__dict__))
        self.cname = typedecl.get_c_name()

        self.use_stmt_list = []
        if isinstance(typedecl.parent, Module):
            self.use_stmt_list.append('use %s' % (typedecl.parent.name))

        parent.apply_templates(self)


        return
