
__all__ = ['PythonCAPISubProgram']

import sys

from parser.api import TypeDecl, TypeStmt
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
{"%(pyname)s", (PyCFunction)%(cname)s, METH_VARARGS | METH_KEYWORDS, %(cname)s__doc},'''
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
        self.name = name = pyname = block.name
        self.cname = cname = '%s_%s' % (parent.cname,name)
        if cname in self._defined:
            return
        self._defined.append(cname)
        self.info('Generating interface for %s: %s' % (block.__class__, cname))

        if pyname.startswith('f2pywrap_'):
            pyname = pyname[9:]
        self.pyname = pyname

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
        argindex = -1
        for argname in block.args:
            argindex += 1
            var = block.a.variables[argname]
            assert var.is_scalar(),'array support not implemented: "%s"' % (var)
            typedecl = var.get_typedecl()
            PythonCAPIType(parent, typedecl)
            ti = PyTypeInterface(typedecl)
            if var.is_intent_in():
                self.kw_list.append('"%s"' % (argname))

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
            else:
                if var.is_intent_in():
                    self.pyarg_format_list.append('O&')
                    self.pyarg_obj_list.append('\npyobj_to_%s, &%s' % (ti.ctype, argname))
                assert not isinstance(typedecl, TypeDecl)
                if ti.ctype=='f2py_string0':
                    assert not var.is_intent_out(),'intent(out) not implemented for "%s"' % (var)
                    self.decl_list.append('%s %s = {NULL,0};' % (ti.ctype, argname))
                    args_f.append('%s.data' % argname)  # is_scalar
                    extra_args_f.append('%s.len' % argname)
                    self.clean_frompyobj_list.append(\
                        'if (%s.len) free(%s.data);' % (argname,argname))
                else:
                    self.decl_list.append('%s %s;' % (ti.ctype, argname))
                    args_f.append('&'+argname) # is_scalar

            if var.is_intent_out(): # and is_scalar
                if isinstance(typedecl, TypeStmt):
                    self.return_format_list.append('N')
                    self.return_obj_list.append('\n%s' % (argname))
                else:
                    self.return_format_list.append('O&')
                    self.return_obj_list.append('\npyobj_from_%s, &%s' % (ti.ctype, argname))

        WrapperCPPMacro(parent, 'F_FUNC')
        self.call_list.append('%s_f(%s);' % (name,', '.join(args_f+extra_args_f)))

        self.clean_pyobjfrom_list.reverse()
        self.clean_call_list.reverse()
        self.clean_frompyobj_list.reverse()

        if self.return_obj_list: self.return_obj_list.insert(0,'')

        parent.apply_templates(self)
        return
