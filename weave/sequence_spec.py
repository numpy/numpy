import cxx_info
from base_spec import base_converter
from types import *
import os

class base_cxx_converter(base_converter):
    _build_information = [cxx_info.cxx_info()]
    def type_spec(self,name,value):
        # factory
        new_spec = self.__class__()
        new_spec.name = name        
        return new_spec
    def __repr__(self):
        msg = "(%s:: name: %s)" % (self.type_name,self.name)
        return msg
    def __cmp__(self,other):
        #only works for equal
        return cmp(self.name,other.name) or \
               cmp(self.__class__, other.__class__)
        
class string_converter(base_cxx_converter):
    type_name = 'string'
    def type_match(self,value):
        return type(value) in [StringType]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        code = 'Py::String %s = convert_to_string(%s,"%s");\n' % \
               (self.name,var_name,self.name)
        return code       
    def local_dict_code(self):
        code = 'local_dict["%s"] = %s;\n' % (self.name,self.name)        
        return code


class list_converter(base_cxx_converter):
    type_name = 'list'
    def type_match(self,value):
        return type(value) in [ListType]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        code = 'Py::List %s = convert_to_list(%s,"%s");\n' % \
               (self.name,var_name,self.name)
        return code       
    def local_dict_code(self):
        code = 'local_dict["%s"] = %s;\n' % (self.name,self.name)        
        return code

class dict_converter(base_cxx_converter):
    type_name = 'dict'
    def type_match(self,value):
        return type(value) in [DictType]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        code = 'Py::Dict %s = convert_to_dict(%s,"%s");\n' % \
               (self.name,var_name,self.name)               
        return code
               
    def local_dict_code(self):
        code = 'local_dict["%s"] = %s;\n' % (self.name,self.name)        
        return code

class tuple_converter(base_cxx_converter):
    type_name = 'tuple'
    def type_match(self,value):
        return type(value) in [TupleType]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        code = 'Py::Tuple %s = convert_to_tuple(%s,"%s");\n' % \
               (self.name,var_name,self.name)
        return code       
    def local_dict_code(self):
        code = 'local_dict["%s"] = %s;\n' % (self.name,self.name)        
        return code

def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)

