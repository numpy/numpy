from base_spec import base_specification
from scalar_spec import numeric_to_blitz_type_mapping
from Numeric import *
from types import *
import os
import standard_array_info

class array_specification(base_specification):
    _build_information = [standard_array_info.array_info()]
    
    def type_match(self,value):
        return type(value) is ArrayType

    def type_spec(self,name,value):
        # factory
        new_spec = array_specification()
        new_spec.name = name
        new_spec.numeric_type = value.typecode()
        # dims not used, but here for compatibility with blitz_spec
        new_spec.dims = len(shape(value))
        return new_spec

    def declaration_code(self,templatize = 0,inline=0):
        if inline:
            code = self.inline_decl_code()
        else:
            code = self.standard_decl_code()
        return code
    
    def inline_decl_code(self):
        type = numeric_to_blitz_type_mapping[self.numeric_type]
        name = self.name
        var_name = self.retrieve_py_variable(inline=1)
        templ = '// %(name)s array declaration\n' \
                'py_%(name)s= %(var_name)s;\n' \
                'PyArrayObject* %(name)s = py_to_numpy(py_%(name)s,"%(name)s");\n' \
                'int* _N%(name)s = %(name)s->dimensions;\n' \
                'int* _S%(name)s = %(name)s->strides;\n' \
                'int _D%(name)s = %(name)s->nd;\n' \
                '%(type)s* %(name)s_data = (%(type)s*) %(name)s->data;\n' 
        code = templ % locals()
        return code

    def standard_decl_code(self):    
        type = numeric_to_blitz_type_mapping[self.numeric_type]
        name = self.name
        templ = '// %(name)s array declaration\n' \
                'PyArrayObject* %(name)s = py_to_numpy(py_%(name)s,"%(name)s");\n' \
                'int* _N%(name)s = %(name)s->dimensions;\n' \
                'int* _S%(name)s = %(name)s->strides;\n' \
                'int _D%(name)s = %(name)s->nd;\n' \
                '%(type)s* %(name)s_data = (%(type)s*) %(name)s->data;\n' 
        code = templ % locals()
        return code
    #def c_function_declaration_code(self):
    #    """
    #        This doesn't pass the size through.  That info is gonna have to 
    #        be redone in the c function.
    #    """
    #    templ_dict = {}
    #    templ_dict['type'] = numeric_to_blitz_type_mapping[self.numeric_type]
    #    templ_dict['dims'] = self.dims
    #    templ_dict['name'] = self.name
    #    code = 'blitz::Array<%(type)s,%(dims)d> &%(name)s' % templ_dict
    #    return code
        
    def local_dict_code(self):
        code = '// for now, array "%s" is not returned as arryas are edited' \
               ' in place (should this change?)\n' % (self.name)        
        return code

    def cleanup_code(self):
        # could use Py_DECREF here I think and save NULL test.
        code = "Py_XDECREF(py_%s);\n" % self.name
        return code

    def __repr__(self):
        msg = "(array:: name: %s, type: %s)" % \
                  (self.name, self.numeric_type)
        return msg

    def __cmp__(self,other):
        #only works for equal
        return cmp(self.name,other.name) or  \
               cmp(self.numeric_type,other.numeric_type) or \
               cmp(self.dims, other.dims) or \
               cmp(self.__class__, other.__class__)

import ext_tools
standard_array_factories = [array_specification()] + ext_tools.default_type_factories

def test():
    from scipy_test import module_test
    module_test(__name__,__file__)

def test_suite():
    from scipy_test import module_test_suite
    return module_test_suite(__name__,__file__)    
