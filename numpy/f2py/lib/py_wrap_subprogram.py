
__all__ = ['PythonCAPISubProgram']

import sys

from parser.api import TypeDecl, TypeStmt, Module
from wrapper_base import *
from py_wrap_type import *

class PythonCAPISubProgram(WrapperBase):
    """
    Fortran subprogram hooks.
    """

    header_template_f77 = '''\
#define %(name)s_f F_FUNC(%(name)s, %(NAME)s)
'''
    extern_template_f77 = '''\
extern void %(name)s_f(%(ctype_args_f_clist)s);
'''
    objdecl_template_doc = '''\
static char %(cname)s__doc[] = "";
'''
    module_method_template = '''\
{"%(pyname)s", (PyCFunction)%(cname)s, METH_VARARGS | METH_KEYWORDS, %(cname)s__doc},'''

    capi_code_template = '''\
static PyObject* %(cname)s(PyObject *capi_self, PyObject *capi_args, PyObject *capi_keywds) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
  %(decl_list)s
  static char *capi_kwlist[] = {%(kw_clist+optkw_clist+extrakw_clist+["NULL"])s};
  if (PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,
                                  "%(pyarg_format_elist)s",
                                   %(["capi_kwlist"]+pyarg_obj_clist)s)) {
  %(frompyobj_list)s
  %(call_list)s
  f2py_success = !PyErr_Occurred();
  if (f2py_success) {
    %(pyobjfrom_list)s
    capi_buildvalue = Py_BuildValue("%(return_format_elist)s"
                                    %(return_obj_clist)s);
    %(clean_pyobjfrom_list)s
  }
  %(clean_call_list)s
  %(clean_frompyobj_list)s
  }
  return capi_buildvalue;
}
'''

    header_template_module = '''
#define %(name)s_f (*%(name)s_func_ptr)
#define %(init_func)s_f F_FUNC(%(init_func)s, %(INIT_FUNC)s)
'''
    typedef_template_module = '''
typedef void (*%(name)s_functype)(%(ctype_args_f_clist)s);
typedef void (*%(init_func)s_c_functype)(%(name)s_functype);
'''
    extern_template_module = '''\
extern void %(init_func)s_f(%(init_func)s_c_functype);
static %(name)s_functype %(name)s_func_ptr;
'''
    objdecl_template_module = '''
'''
    fortran_code_template_module = '''
    subroutine %(init_func)s(init_func_c)
      use %(mname)s
      external init_func_c
      call init_func_c(%(name)s)
    end
'''
    c_code_template_module = '''
static void %(init_func)s_c(%(name)s_functype func_ptr) {
  %(name)s_func_ptr = func_ptr;
}
'''
    module_init_template_module = '''
%(init_func)s_f(%(init_func)s_c);
'''

    def __init__(self, parent, block):
        WrapperBase.__init__(self)
        self.name = name = pyname = block.name
        self.cname = cname = '%s_%s' % (parent.cname,name)

        defined = parent.defined_capi_codes
        if cname in defined:
            return
        defined.append(cname)
        
        self.info('Generating interface for %s %s: %s' % (parent.modulename, block.__class__.__name__, cname))
        self.parent = parent

        if pyname.startswith('f2pywrap_'):
            pyname = pyname[9:]
        self.pyname = pyname

        self.header_template = ''
        self.extern_template = ''
        self.module_init_template = ''
        self.typedef_template = ''
        self.c_code_template = ''
        self.objdecl_template =  ''
        self.fortran_code_template = ''

        WrapperCPPMacro(parent, 'F_FUNC')
        
        if isinstance(block.parent, Module):
            self.mname = block.parent.name
            self.init_func = '%s_init' % (name)
            self.typedef_template += self.typedef_template_module
            self.header_template += self.header_template_module
            self.fortran_code_template += self.fortran_code_template_module
            self.module_init_template += self.module_init_template_module
            self.objdecl_template += self.objdecl_template_module
            self.c_code_template += self.c_code_template_module
            self.extern_template += self.extern_template_module
        else:
            self.extern_template += self.extern_template_f77
            self.header_template += self.header_template_f77

        self.objdecl_template += self.objdecl_template_doc

        self.decl_list = []
        self.kw_list = []
        self.optkw_list = []
        self.extrakw_list = []
        self.pyarg_format_list = []
        self.pyarg_obj_list = []
        self.frompyobj_list = []
        self.call_list = []
        self.pyobjfrom_list = []
        self.return_format_list = []
        self.return_obj_list = []
        self.buildvalue_list = []
        self.clean_pyobjfrom_list = []
        self.clean_call_list = []
        self.clean_frompyobj_list = []

        args_f = []
        extra_args_f = []
        ctype_args_f = []
        extra_ctype_args_f = []
        argindex = -1
        for argname in block.args:
            argindex += 1
            var = block.a.variables[argname]
            typedecl = var.get_typedecl()
            PythonCAPIType(parent, typedecl)
            ti = PyTypeInterface(typedecl)
            if var.is_intent_in():
                self.kw_list.append('"%s"' % (argname))

            if var.is_scalar():
                if isinstance(typedecl, TypeStmt):
                    if var.is_intent_in():
                        self.pyarg_format_list.append('O&')
                        self.pyarg_obj_list.append('\npyobj_to_%s_inplace, &%s' % (ti.ctype, argname))
                    else:
                        self.frompyobj_list.append('%s = (%s*)pyobj_from_%s(NULL);' % (argname,ti.otype,ti.ctype))
                        if not var.is_intent_out():
                            self.clean_frompyobj_list.append('Py_DECREF(%s);' % (argname))
                    self.decl_list.append('%s* %s = NULL;' % (ti.otype, argname))
                    args_f.append('%s->data' % (argname)) # is_scalar
                    ctype_args_f.append(ti.ctype)
                else:
                    if var.is_intent_in():
                        self.pyarg_format_list.append('O&')
                        self.pyarg_obj_list.append('\npyobj_to_%s, &%s' % (ti.ctype, argname))
                    assert not isinstance(typedecl, TypeDecl)
                    if ti.ctype=='f2py_string0':
                        if not var.is_intent_in():
                            assert not var.is_intent_out(),'intent(out) not implemented for "%s"' % (var)
                        self.decl_list.append('%s %s = {NULL,0};' % (ti.ctype, argname))
                        args_f.append('%s.data' % argname)  # is_scalar
                        ctype_args_f.append('char*')
                        extra_ctype_args_f.append('int')
                        extra_args_f.append('%s.len' % argname)
                        self.clean_frompyobj_list.append(\
                        'if (%s.len) free(%s.data);' % (argname,argname))
                    else:
                        self.decl_list.append('%s %s;' % (ti.ctype, argname))
                        args_f.append('&'+argname) # is_scalar
                        ctype_args_f.append(ti.ctype+'*')
                if var.is_intent_out(): # and is_scalar
                    if isinstance(typedecl, TypeStmt):
                        self.return_format_list.append('N')
                        self.return_obj_list.append('\n%s' % (argname))
                    else:
                        self.return_format_list.append('O&')
                        self.return_obj_list.append('\npyobj_from_%s, &%s' % (ti.ctype, argname))
            else:
                print `ti,var.dimension,var.bounds`
                assert var.is_scalar(),'array support not implemented: "%s"' % (var)

        self.call_list.append('%s_f(%s);' % (name,', '.join(args_f+extra_args_f)))

        self.ctype_args_f_list = ctype_args_f + extra_ctype_args_f
        if not self.ctype_args_f_list:
            self.ctype_args_f_list.append('void')
        

        self.clean_pyobjfrom_list.reverse()
        self.clean_call_list.reverse()
        self.clean_frompyobj_list.reverse()

        if self.return_obj_list: self.return_obj_list.insert(0,'')

        parent.apply_templates(self)
        return
