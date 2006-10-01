
__all__ = ['TypeWrapper']

import re
import os
import sys

from parser.api import *

#from block_statements import *
#from typedecl_statements import intrinsic_type_spec, Character
#from utils import CHAR_BIT

from wrapper_base import *

class PythonWrapperModule(WrapperBase):

    main_template = '''\
#ifdef __cplusplus
extern \"C\" {
#endif
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

%(header_list)s

%(typedef_list)s

%(extern_list)s

%(c_code_list)s

%(capi_code_list)s

%(objdecl_list)s

static PyObject *f2py_module;

static PyMethodDef f2py_module_methods[] = {
  %(module_method_list)s
  {NULL,NULL,0,NULL}
};

PyMODINIT_FUNC init%(modulename)s(void) {
  f2py_module = Py_InitModule("%(modulename)s", f2py_module_methods);
  import_array();
  %(module_init_list)s
  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_ImportError, "can\'t initialize module %(modulename)s");
    return;
  }
}
#ifdef __cplusplus
}
#endif
'''

    main_fortran_template = '''\
! -*- f90 -*-
%(fortran_code_list)s
'''
    def __init__(self, modulename):
        WrapperBase.__init__(self)
        self.modulename = modulename
        #self.include_list = []
        #self.cppmacro_list = []
        
        self.header_list = []
        self.typedef_list = []
        self.extern_list = []
        self.objdecl_list = []
        self.c_code_list = []
        self.capi_code_list = []

        self.module_method_list = []
        self.module_init_list = []

        self.fortran_code_list = []

        #self.defined_types = []
        #self.defined_macros = []
        #self.defined_c_functions = []
        #self.defined_typedefs = []

        self.list_names = ['header', 'typedef', 'extern', 'objdecl',
                           'c_code','capi_code','module_method','module_init',
                           'fortran_code']

        return

    def add(self, block):
        if isinstance(block, BeginSource):
            for name, moduleblock in block.a.module.items():
                self.add(moduleblock)
            #for name, subblock in block.a.external_subprogram.items():
            #    self.add(subblock)
        elif isinstance(block, (Subroutine, Function)):
            self.info('Generating interface for %s' % (block.name))
            f = PythonCAPIFunction(self, block)
            f.fill()
        elif isinstance(block, Module):
            for name,declblock in block.a.type_decls.items():
                self.add(declblock)
        elif isinstance(block, TypeDecl):
            PythonCAPIDerivedType(self, block)
        elif isinstance(block, tuple(declaration_type_spec)):
            PythonCAPIIntrinsicType(self, block)
        else:
            raise NotImplementedError,`block.__class__.__name__`
        return
    
    def c_code(self):
        return self.apply_attributes(self.main_template)
    def fortran_code(self):
        return self.apply_attributes(self.main_fortran_template)

    def add_subroutine(self, block):
        raise
        f = PythonCAPIFunction(self, block)
        f.fill()
        return



        
class PythonCAPIIntrinsicType(WrapperBase):
    """
    Fortran intrinsic type hooks.
    """
    _defined_types = []
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        self.name = name = typedecl.name
        if name in self._defined_types:
            return
        self._defined_types.append(name)

        self.ctype = ctype = typedecl.get_c_type()

        if ctype.startswith('npy_'):
            from generate_pyobj_tofrom_funcs import pyobj_to_npy_scalar
            d = pyobj_to_npy_scalar(ctype)
            for v in d.values():
                self.resolve_dependencies(parent, v)
            for k,v in d.items():
                l = getattr(parent, k+'_list')
                l.append(v)
            return
        
        if not ctype.startswith('f2py_type_'):
            raise NotImplementedError,`name,ctype`

        for n in parent.list_names:
            l = getattr(parent,n + '_list')
            l.append(self.apply_attributes(getattr(self, n+'_template','')))

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
  PyMem_Free(self->data);
  self->ob_type->tp_free((PyObject*)self);
}

static int pyobj_to_%(ctype)s(PyObject *obj,
                              %(ctype)s* value_ptr) {
  if (%(oname)sObject_Check(obj)) {
    if (!memcpy(value_ptr,((%(oname)sObject *)obj)->data, %(byte_size)s)) {
      PyErr_SetString(PyExc_MemoryError,
         "failed to copy %(name)s instance memory to %(ctype)s object.");
    }
    return 1;
  }
  return 0;
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
   return !PyArg_ParseTuple(capi_args,"%(attr_format_elist)s"
                                      %(attr_init_clist)s);
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

    _defined_types = []
    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        name = typedecl.name
        if name in self._defined_types:
            return
        self._defined_types.append(name)

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
            WrapperCCode(parent, 'pyobj_to_%s' % (ct))
            self.attr_init_list.append('\npyobj_to_%s, self->%s_ptr' % (ct,n))
            WrapperCCode(parent, 'pyobj_from_%s' % (ct))
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

        for n in parent.list_names:
            l = getattr(parent,n + '_list')
            l.append(self.apply_attributes(getattr(self, n+'_template','')))
        return

class PythonCAPIFunction(WrapperBase):
    capi_function_template = '''
static char f2py_doc_%(function_name)s[] = "%(function_doc)s";
static PyObject* f2py_%(function_name)s(PyObject *capi_self, PyObject *capi_args, PyObject *capi_keywds) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
  %(decl_list)s
  static char *capi_kwlist[] = {%(keyword_clist+optkw_clist+extrakw_clist+["NULL"])s};
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,
                                   "%(pyarg_format_elist)s",
                                   %(["capi_kwlist"]+pyarg_obj_clist)s))
     return NULL;
  %(frompyobj_list)s
  %(call_list)s
  f2py_success = !PyErr_Occurred();
  if (f2py_success) {
    %(pyobjfrom_list)s
    capi_buildvalue = Py_BuildValue(%(buildvalue_clist)s);
    %(clean_pyobjfrom_list)s
  }
  %(clean_frompyobj_list)s
  return capi_buildvalue;
}
'''    

    pymethoddef_template = '''\
{"%(function_name)s", (PyCFunction)f2py_%(function_name)s, METH_VARARGS | METH_KEYWORDS, f2py_doc_%(function_name)s},\
'''

    cppmacro_template = '''\
#define %(function_name)s_f F_FUNC(%(function_name)s,%(FUNCTION_NAME)s)
'''

    extdef_template = '''\
extern void %(function_name)s_f();\
'''

    def __init__(self, parent, block):
        WrapperBase.__init__(self)
        self.parent = parent
        self.block = block
        self.function_name = block.name
        self.FUNCTION_NAME = self.function_name.upper()
        self.function_doc = ''
        self.args_list = block.args
        self.decl_list = []
        self.keyword_list = []
        self.optkw_list = []
        self.extrakw_list = []
        self.frompyobj_list = []
        self.call_list = []
        self.pyobjfrom_list = []
        self.buildvalue_list = []
        self.clean_pyobjfrom_list = []
        self.clean_frompyobj_list = []
        self.pyarg_format_list = []
        self.pyarg_obj_list = []
        return

    def fill(self):
        for argname in self.args_list:
            var = self.block.a.variables[argname]
            argwrap = ArgumentWrapper(self, var)
            argwrap.fill()
        self.call_list.append('%s_f(%s);' % (self.function_name, ', '.join(['&'+a for a in self.args_list])))
        if not self.buildvalue_list:
            self.buildvalue_list.append('""')
        self.parent.capi_function_list.append(self.apply_attributes(self.capi_function_template))
        self.parent.module_method_list.append(self.apply_attributes(self.pymethoddef_template))
        self.parent.extern_list.append(self.apply_attributes(self.extdef_template))
        self.parent.add_cppmacro('F_FUNC')
        self.parent.cppmacro_list.append(self.apply_attributes(self.cppmacro_template))
        return

class ArgumentWrapper(WrapperBase):

    objdecl_template = '%(ctype)s %(name)s;'
    pyarg_obj_template = '\npyobj_to_%(ctype)s, &%(name)s'

    def __init__(self, parent, variable):
        WrapperBase.__init__(self)
        self.parent = parent
        self.grand_parent = parent.parent
        self.variable = variable
        self.typedecl = variable.typedecl
        self.name = variable.name
        self.ctype = self.typedecl.get_c_type()
        
    def fill(self):
        typename = self.grand_parent.add_type(self.typedecl)
        self.parent.decl_list.append(self.apply_attributes(self.objdecl_template))
        
        self.parent.pyarg_obj_list.append(self.apply_attributes(self.pyarg_obj_template))
        self.parent.pyarg_format_list.append('O&')
        self.parent.keyword_list.append('"%s"' % (self.name))

        return


class TypeDecl2(WrapperBase):
    cppmacro_template = '''\
#define initialize_%(typename)s_interface F_FUNC(initialize_%(typename)s_interface_f,INITIALIZE_%(TYPENAME)s_INTERFACE_F)\
'''
    typedef_template = '''\
typedef struct { char data[%(byte_size)s]; } %(ctype)s;
typedef %(ctype)s (*create_%(typename)s_functype)(void);
typedef void (*initialize_%(typename)s_interface_functype)(create_%(typename)s_functype);\
'''
    objdecl_template = '''\
static create_%(typename)s_functype create_%(typename)s_object;
'''
    funcdef_template = '''\
static void initialize_%(typename)s_interface_c(create_%(typename)s_functype create_object_f) {
  create_%(typename)s_object = create_object_f;
}
'''
    extdef_template = '''\
extern void initialize_%(typename)s_interface(initialize_%(typename)s_interface_functype);\
'''
    initcall_template = '''\
initialize_%(typename)s_interface(initialize_%(typename)s_interface_c);\
'''
    fortran_code_template = '''\
       function create_%(typename)s_object_f() result (obj)
         %(typedecl_list)s
         %(typedecl)s obj
!        %(initexpr)s
       end
       subroutine initialize_%(typename)s_interface_f(init_c)
         external create_%(typename)s_object_f
         call init_c(create_%(typename)s_object_f)
       end
'''
    pyobj_to_type_template = '''
    static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
      if (PyTuple_Check(obj)) {
        return 0;
      }
    return 0;
    }
'''

    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        self.parent = parent
        self.typedecl = typedecl.astypedecl()
        self.typedecl_list = []
        self.ctype = self.typedecl.get_c_type()
        self.byte_size = self.typedecl.get_byte_size()
        self.typename = self.typedecl.name.lower()
        self.TYPENAME = self.typedecl.name.upper()
        self.initexpr = self.typedecl.assign_expression('obj',self.typedecl.get_zero_value())
        return

    def fill(self):
        ctype =self.typedecl.get_c_type()
        if ctype.startswith('npy_') or ctype.startswith('f2py_string'):
            # wrappers are defined via pyobj_to_* functions
            self.parent.add_c_function('pyobj_to_%s' % (self.ctype))
            return
        if ctype.startswith('f2py_type'):
            return
            self.parent.add_typedef(ctype,
                                    self.apply_attributes('typedef struct { char data[%(byte_size)s]; } %(ctype)s;'))
            self.parent.add_c_function(self.apply_attributes('pyobj_to_%(ctype)s'),
                                       self.apply_attributes(self.pyobj_to_type_template)
                                       )
        else:
            self.parent.typedef_list.append(self.apply_attributes(self.typedef_template))
            self.parent.objdecl_list.append(self.apply_attributes(self.objdecl_template))
            self.parent.c_function_list.append(self.apply_attributes(self.funcdef_template))
            self.parent.extern_list.append(self.apply_attributes(self.extdef_template))
            self.parent.initialize_interface_list.append(self.apply_attributes(self.initcall_template))
            self.parent.fortran_code_list.append(self.apply_attributes(self.fortran_code_template))
            self.parent.add_cppmacro('F_FUNC')
            self.parent.cppmacro_list.append(self.apply_attributes(self.cppmacro_template))
        return




if __name__ == '__main__':
    #from utils import str2stmt, get_char_bit
    
    stmt = parse("""
    module rat
      integer :: i
      type info
        integer flag
      end type info
      type rational
        integer n
        integer d
        type(info) i
      end type rational
    end module rat
    subroutine foo(a)
    use rat
    type(rational) a
    end
    """)
    #stmt = stmt.content[-1].content[1]
    #print stmt
    #wrapgen = TypeWrapper(stmt)
    #print wrapgen.fortran_code()
    #print wrapgen.c_code()

    foo_code = """! -*- f90 -*-
      module rat
        type info
          integer flag
        end type info
        type rational
          integer n,d
          type(info) i
        end type rational
      end module rat
      subroutine foo(a,b)
        use rat
        integer a
        !character*5 b
        type(rational) b
        print*,'a=',a,b
      end
"""

    wm = PythonWrapperModule('foo')
    wm.add(parse(foo_code))
    #wm.add_fortran_code(foo_code)
    #wm.add_subroutine(str2stmt(foo_code))
    #print wm.c_code()

    c_code = wm.c_code()
    f_code = wm.fortran_code()

    f = open('foomodule.c','w')
    f.write(c_code)
    f.close()
    f = open('foo.f','w')
    f.write(foo_code)
    f.close()
    f = open('foo_wrap.f','w')
    f.write(f_code)
    f.close()
    f = open('foo_setup.py','w')
    f.write('''\
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('foopack',parent_package,top_path)
    config.add_library('foolib',
                       sources = ['foo.f','foo_wrap.f'])
    config.add_extension('foo',
                         sources=['foomodule.c'],
                         libraries = ['foolib'],
                         )
    return config
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
''')
    f.close()
    #print get_char_bit()
    os.system('python foo_setup.py config_fc --fcompiler=gnu95 build build_ext --inplace')
    import foo
    #print foo.__doc__
    #print dir(foo)
    #print foo.info.__doc__
    #print foo.rational.__doc__
    #print dir(foo.rational)
    i = foo.info(7)
    #print i #,i.as_tuple()
    #print 'i.flag=',i.flag
    r = foo.rational(2,3,i)
    print r
    j = r.i
    print 'r.i.flag=',(r.i).flag
    print 'j.flag=',j.flag
    #print 'r=',r
    sys.exit()
    n,d,ii = r.as_tuple()
    n += 1
    print n,d
    print r
    #foo.foo(2,r)
    print r.n, r.d
    r.n = 5
    print r
    r.n -= 1
    print r
