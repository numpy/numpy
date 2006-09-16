
__all__ = ['TypeWrapper']

import re
import os
import sys

from block_statements import *
#from typedecl_statements import intrinsic_type_spec, Character
from utils import CHAR_BIT

class WrapperBase:


    def __init__(self):
        self.srcdir = os.path.join(os.path.dirname(__file__),'src')
        return
    def warning(self, message):
        print >> sys.stderr, message
    def info(self, message):
        print >> sys.stderr, message

    def get_resource_content(self, name, ext):
        if name.startswith('pyobj_to_'):
            body = self.generate_pyobj_to_ctype_c(name[9:])
            if body is not None: return body
        generator_mth_name = 'generate_' + name + ext.replace('.','_')
        generator_mth = getattr(self, generator_mth_name, lambda : None)
        body = generator_mth()
        if body is not None:
            return body
        fn = os.path.join(self.srcdir,name+ext)
        if os.path.isfile(fn):
            f = open(fn,'r')
            body = f.read()
            f.close()
            return body
        self.warning('No such file: %r' % (fn))
        return

    def get_dependencies(self, code):
        l = []
        for uses in re.findall(r'(?<=depends:)([,\w\s.]+)', code, re.I):
            for use in uses.split(','):
                use = use.strip()
                if not use: continue
                l.append(use)
        return l

    def apply_attributes(self, template):
        """
        Apply instance attributes to template string.

        Replace rules for attributes:
        _list  - will be joined with newline
        _clist - _list will be joined with comma
        _elist - _list will be joined
        ..+.. - attributes will be added
        [..]  - will be evaluated
        """
        replace_names = set(re.findall(r'[ ]*%\(.*?\)s', template))
        d = {}
        for name in replace_names:
            tab = ' ' * (len(name)-len(name.lstrip()))
            name = name.lstrip()[2:-2]
            names = name.split('+')
            joinsymbol = '\n'
            attrs = None
            for n in names:
                realname = n.strip()
                if n.endswith('_clist'):
                    joinsymbol = ', '
                    realname = realname[:-6] + '_list'
                elif n.endswith('_elist'):
                    joinsymbol = ''
                    realname = realname[:-6] + '_list'
                if hasattr(self, realname):
                    attr = getattr(self, realname)
                elif realname.startswith('['):
                    attr = eval(realname)
                else:
                    self.warning('Undefined %r attribute: %r' % (self.__class__.__name__, realname))
                    continue
                if attrs is None:
                    attrs = attr
                else:
                    attrs += attr
            if isinstance(attrs, list):
                attrs = joinsymbol.join(attrs)
            d[name] = str(attrs).replace('\n','\n'+tab)
        return template % d

class PythonWrapperModule(WrapperBase):

    main_template = '''\
#ifdef __cplusplus
extern \"C\" {
#endif
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

%(include_list)s
%(cppmacro_list)s
%(typedef_list)s
%(objdecl_list)s
%(extern_list)s
%(c_function_list)s
%(capi_function_list)s
static PyObject *f2py_module;
static PyMethodDef f2py_module_methods[] = {
  %(module_method_list)s
  {NULL,NULL,0,NULL}
};
PyMODINIT_FUNC init%(modulename)s(void) {
  f2py_module = Py_InitModule("%(modulename)s", f2py_module_methods);
  %(initialize_interface_list)s
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
%(fortran_code_list)s
'''
    def __init__(self, modulename):
        WrapperBase.__init__(self)
        self.modulename = modulename
        self.include_list = []
        self.typedef_list = []
        self.cppmacro_list = []
        self.objdecl_list = []
        self.c_function_list = []
        self.extern_list = []
        self.capi_function_list = []
        self.module_method_list = []
        self.initialize_interface_list = []
        self.fortran_code_list = []

        self.defined_types = []
        self.defined_macros = []
        self.defined_c_functions = []
        self.defined_typedefs = []
        return

    def add(self, block):
        if isinstance(block, BeginSource):
            for name, subblock in block.a.external_subprogram.items():
                self.add(subblock)
        elif isinstance(block, (Subroutine, Function)):
            self.info('Generating interface for %s' % (block.name))
            f = PythonCAPIFunction(self, block)
            f.fill()            
        else:
            raise NotImplementedError,`block.__class__.__name__`
        return
    
    def c_code(self):
        return self.apply_attributes(self.main_template)
    def fortran_code(self):
        return self.apply_attributes(self.main_fortran_template)

    def add_c_function(self, name):
        if name not in self.defined_c_functions:
            body = self.get_resource_content(name,'.c')
            if body is None:
                self.warning('Failed to get C function %r content.' % (name))
                return
            for d in self.get_dependencies(body):
                if d.endswith('.cpp'):
                    self.add_cppmacro(d[:-4])
                elif d.endswith('.c'):
                    self.add_c_function(d[:-2])
                else:
                    self.warning('Unknown dependence: %r.' % (d))
            self.defined_c_functions.append(name)
            self.c_function_list.append(body)
        return

    def add_cppmacro(self, name):
        if name not in self.defined_macros:
            body = self.get_resource_content(name,'.cpp')
            if body is None:
                self.warning('Failed to get CPP macro %r content.' % (name))
                return
            for d in self.get_dependencies(body):
                if d.endswith('.cpp'):
                    self.add_cppmacro(d[:-4])
                elif d.endswith('.c'):
                    self.add_c_function(d[:-2])
                else:
                    self.warning('Unknown dependence: %r.' % (d))
            self.defined_macros.append(name)
            self.cppmacro_list.append(body)
        return

    def add_type(self, typedecl):
        typewrap = TypeDecl(self, typedecl)
        typename = typewrap.typename
        if typename not in self.defined_types:
            self.defined_types.append(typename)
            typewrap.fill()
        return typename

    def add_typedef(self, name, code):
        if name not in self.defined_typedefs:
            self.typedef_list.append(code)
            self.defined_types.append(name)
        return

    def add_include(self, include):
        if include not in self.include_list:
            self.include_list.append(include)
        return

    def add_subroutine(self, block):
        f = PythonCAPIFunction(self, block)
        f.fill()
        return

    def generate_pyobj_to_ctype_c(self, ctype):
        if ctype.startswith('npy_int'):
            ctype_bits = int(ctype[7:])
            return '''
/* depends: pyobj_to_long.c, pyobj_to_npy_longlong.c */
#if NPY_BITSOF_LONG == %(ctype_bits)s
#define pyobj_to_%(ctype)s pyobj_to_long
#else
#if NPY_BITSOF_LONG > %(ctype_bits)s
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  long tmp;
  if (pyobj_to_long(obj,&tmp)) {
    *value = (%(ctype)s)tmp;
    return 1;
  }
  return 0;
}
#else
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  npy_longlong tmp;
  if (pyobj_to_npy_longlong(obj,&tmp)) {
    *value = (%(ctype)s)tmp;
    return 1;
  }
  return 0;
}
#endif
#endif
''' % (locals())
        elif ctype.startswith('npy_float'):
            ctype_bits = int(ctype[9:])
            return '''
/* depends: pyobj_to_double.c */
#if NPY_BITSOF_DOUBLE == %(ctype_bits)s
#define pyobj_to_%(ctype)s pyobj_to_double
#else
#if NPY_BITSOF_DOUBLE > %(ctype_bits)s
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  double tmp;
  if (pyobj_to_double(obj,&tmp)) {
    *value = (%(ctype)s)tmp;
    return 1;
  }
  return 0;
}
#else
#error, "NOTIMPLEMENTED pyobj_to_%(ctype)s"
#endif
#endif
''' % (locals())
        elif ctype.startswith('npy_complex'):
            ctype_bits = int(ctype[11:])
            cfloat_bits = ctype_bits/2
            return '''
/* depends: pyobj_to_Py_complex.c */
#if NPY_BITSOF_DOUBLE >= %(cfloat_bits)s
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  Py_complex c;
  if (pyobj_to_Py_complex(obj,&c)) {
    (*value).real = (npy_float%(cfloat_bits)s)c.real;
    (*value).imag = (npy_float%(cfloat_bits)s)c.imag; 
    return 1;
  }
  return 0;
}
#else
#error, "NOTIMPLEMENTED pyobj_to_%(ctype)s"
#endif
''' % (locals())
        elif ctype.startswith('f2py_string'):
            ctype_bits = int(ctype[11:])
            ctype_bytes = ctype_bits / CHAR_BIT
            self.add_typedef('f2py_string','typedef char * f2py_string;')
            self.add_typedef(ctype,'typedef struct { char data[%s]; } %s;' % (ctype_bytes,ctype))
            self.add_include('#include <string.h>')
            return '''
/* depends: pyobj_to_string_len.c */
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  return pyobj_to_string_len(obj, (f2py_string*)value, %(ctype_bytes)s);
}
''' % (locals())

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

        self.grand_parent.add_c_function('pyobj_to_%s' % (self.ctype))
        return

class TypeDecl(WrapperBase):
    cppmacro_template = '''\
#define initialize_%(typename)s_interface F_FUNC(initialize_%(typename)s_interface_f,INITIALIZE_%(TYPENAME)s_INTERFACE_F)\
'''
    typedef_template = '''\
typedef struct { char data[%(byte_size)s] } %(ctype)s;
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
         %(typedecl)s obj
!         %(initexpr)s
       end
       subroutine initialize_%(typename)s_interface_f(init_c)
         external create_%(typename)s_object_f
         call init_c(create_%(typename)s_object_f)
       end
'''

    def __init__(self, parent, typedecl):
        WrapperBase.__init__(self)
        self.parent = parent
        self.typedecl = typedecl.astypedecl()
        self.ctype = self.typedecl.get_c_type()
        self.byte_size = self.typedecl.get_byte_size()
        self.typename = self.typedecl.name.lower()
        self.TYPENAME = self.typedecl.name.upper()
        self.initexpr = self.typedecl.assign_expression('obj',self.typedecl.get_zero_value())
        return

    def fill(self):
        ctype =self.typedecl.get_c_type()
        if ctype.startswith('npy_'):
            pass
        elif ctype.startswith('f2py_string'):
            pass
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
    from utils import str2stmt, get_char_bit

    stmt = str2stmt("""
    module rat
      integer :: i
      type rational
        integer n
        integer*8 d
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
        type rational
          integer d,n
        end type rational
      end module rat
      subroutine foo(a,b)
        use rat
        integer a
        character*5 b
        type(rational) c
        print*,'a=',a,b,c
      end
"""

    wm = PythonWrapperModule('foo')
    wm.add(str2stmt(foo_code))
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
    print get_char_bit()
    os.system('python foo_setup.py config_fc --fcompiler=gnu95 build build_ext --inplace')
    import foo
    print dir(foo)
    foo.foo(2,"abcdefg")
