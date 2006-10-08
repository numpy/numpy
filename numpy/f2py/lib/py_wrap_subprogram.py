
__all__ = ['PythonCAPISubProgram']

import sys

from wrapper_base import *
from py_wrap_type import *

class PythonCAPISubProgram(WrapperBase):
    """
    Fortran subprogram hooks.
    """

    header_template = '''\
#define %(name)s_f F_FUNC(%(name)s, %(NAME)s)
'''
    typedef_template = ''
    extern_template = '''\
extern void %(name)s_f();
'''
    objdecl_template = '''\
static char %(cname)s__doc[] = "";
'''
    module_init_template = ''
    module_method_template = '''\
{"%(name)s", (PyCFunction)%(cname)s, METH_VARARGS | METH_KEYWORDS, %(cname)s__doc},'''
    c_code_template = ''
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
    fortran_code_template = ''
    
    _defined = []
    def __init__(self, parent, block):
        WrapperBase.__init__(self)
        self.name = name = block.name
        self.cname = cname = '%s_%s' % (parent.cname,name)
        if cname in self._defined:
            return
        self._defined.append(cname)
        self.info('Generating interface for %s: %s' % (block.__class__, cname))


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
        for argname in block.args:
            var = block.a.variables[argname]
            typedecl = var.get_typedecl()
            PythonCAPIType(parent, typedecl)
            ctype = typedecl.get_c_type()
            if ctype=='f2py_string0':
                self.decl_list.append('%s %s = {NULL,0};' % (ctype, argname))
            else:
                self.decl_list.append('%s %s;' % (ctype, argname))
            self.kw_list.append('"%s"' % (argname))
            self.pyarg_format_list.append('O&')
            self.pyarg_obj_list.append('\npyobj_to_%s, &%s' % (ctype, argname))
            if 1: # is_scalar
                if ctype=='f2py_string0':
                    args_f.append('%s.data' % argname)
                    extra_args_f.append('%s.len' % argname)
                    self.clean_frompyobj_list.append(\
                        'if (%s.len) free(%s.data);' % (argname,argname))
                else:
                    args_f.append('&'+argname)
                
            else:
                args_f.append(argname)
            if var.is_intent_out(): # and is_scalar
                self.return_format_list.append('O&')
                self.return_obj_list.append('\npyobj_from_%s, &%s' % (ctype, argname))

        WrapperCPPMacro(parent, 'F_FUNC')
        self.call_list.append('%s_f(%s);' % (name,', '.join(args_f+extra_args_f)))

        self.clean_pyobjfrom_list.reverse()
        self.clean_call_list.reverse()
        self.clean_frompyobj_list.reverse()

        if self.return_obj_list: self.return_obj_list.insert(0,'')

        parent.apply_templates(self)
        return
