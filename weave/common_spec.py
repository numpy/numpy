from base_spec import base_converter
import common_info
from types import *
import os

class common_base_converter(base_converter):
    def type_spec(self,name,value):
        # factory
        new_spec = self.__class__()
        new_spec.name = name        
        return new_spec
    def __repr__(self):
        msg = "(file:: name: %s)" % self.name
        return msg
    def __cmp__(self,other):
        #only works for equal
        return cmp(self.name,other.name) or \
               cmp(self.__class__, other.__class__)
        
    
class file_converter(common_base_converter):
    type_name = 'file'
    _build_information = [common_info.file_info()]
    def type_match(self,value):
        return type(value) in [FileType]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        #code = 'PyObject* py_%s = %s;\n'   \
        #       'FILE* %s = convert_to_file(py_%s,"%s");\n' % \
        #       (self.name,var_name,self.name,self.name,self.name)
        code = 'PyObject* py_%s = %s;\n'   \
               'FILE* %s = convert_to_file(py_%s,"%s");\n' % \
               (self.name,var_name,self.name,self.name,self.name)
        return code       
    def cleanup_code(self):
        # could use Py_DECREF here I think and save NULL test.
        code = "Py_XDECREF(py_%s);\n" % self.name
        return code

class callable_converter(common_base_converter):
    type_name = 'callable'
    _build_information = [common_info.callable_info()]
    def type_match(self,value):
        # probably should test for callable classes here also.
        return type(value) in [FunctionType,MethodType,type(len)]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        code = 'PyObject* %s = convert_to_callable(%s,"%s");\n' % \
               (self.name,var_name,self.name)
        return code       

class instance_converter(common_base_converter):
    type_name = 'instance'
    _build_information = [common_info.instance_info()]
    def type_match(self,value):
        return type(value) in [InstanceType]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        code = 'PyObject* %s = convert_to_instance(%s,"%s");\n' % \
               (self.name,var_name,self.name)
        return code       

def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)
