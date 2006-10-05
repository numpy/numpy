__all__ = ['PythonWrapperModule']

import re
import os
import sys

from parser.api import *
from wrapper_base import *
from py_wrap_type import *
from py_wrap_subprogram import *

class PythonWrapperModule(WrapperBase):

    main_template = '''\
#ifdef __cplusplus
extern \"C\" {
#endif
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

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
%(fortran_code_list)s
'''
    def __init__(self, modulename):
        WrapperBase.__init__(self)
        self.modulename = modulename
        self.cname = 'f2py_' + modulename

        
        self.header_list = []
        self.typedef_list = []
        self.extern_list = []
        self.objdecl_list = []
        self.c_code_list = []
        self.capi_code_list = []

        self.module_method_list = []
        self.module_init_list = []

        self.fortran_code_list = []

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
            PythonCAPISubProgram(self, block)
        elif isinstance(block, Module):
            for name,declblock in block.a.type_decls.items():
                self.add(declblock)
        elif isinstance(block, tuple([TypeDecl]+declaration_type_spec)):
            PythonCAPIType(self, block)
        else:
            raise NotImplementedError,`block.__class__.__name__`
        return
    
    def c_code(self):
        return self.apply_attributes(self.main_template)
    def fortran_code(self):
        return self.apply_attributes(self.main_fortran_template)
