import wx_info
import base_info
from base_spec import base_converter
from types import *
import os

wx_support_template = \
"""
static %(wx_class)s* convert_to_%(wx_class)s(PyObject* py_obj,char* name)
{
    %(wx_class)s *wx_ptr;
    
    // work on this error reporting...
    if (SWIG_GetPtrObj(py_obj,(void **) &wx_ptr,"_%(wx_class)s_p"))
        handle_conversion_error(py_obj,"%(wx_class)s", name);
    return wx_ptr;
}    

static %(wx_class)s* py_to_%(wx_class)s(PyObject* py_obj,char* name)
{
    %(wx_class)s *wx_ptr;
    
    // work on this error reporting...
    if (SWIG_GetPtrObj(py_obj,(void **) &wx_ptr,"_%(wx_class)s_p"))
        handle_bad_type(py_obj,"%(wx_class)s", name);
    return wx_ptr;
}    
"""        

class wx_converter(base_converter):
    _build_information = [wx_info.wx_info()]
    def __init__(self,class_name=None):
        self.type_name = 'unkown wx_object'
        if class_name:
            # customize support_code for whatever type I was handed.
            vals = {'wx_class': class_name}
            specialized_support = wx_support_template % vals
            custom = base_info.base_info()
            custom._support_code = [specialized_support]
            self._build_information = self._build_information + [custom]
            self.type_name = class_name

    def type_match(self,value):
        try:
            class_name = value.this.split('_')[-2]
            if class_name[:2] == 'wx':
                return 1
        except AttributeError:
            pass
        return 0
            
    def type_spec(self,name,value):
        # factory
        class_name = value.this.split('_')[-2]
        new_spec = self.__class__(class_name)
        new_spec.name = name        
        return new_spec
    def declaration_code(self,inline=0):
        type = self.type_name
        name = self.name
        var_name = self.retrieve_py_variable(inline)
        template = '%(type)s *%(name)s = '\
                   'convert_to_%(type)s(%(var_name)s,"%(name)s");\n'
        code = template % locals()
        return code
        
    def __repr__(self):
        msg = "(%s:: name: %s)" % (self.type_name,self.name)
        return msg
    def __cmp__(self,other):
        #only works for equal
        return cmp(self.name,other.name) or \
               cmp(self.__class__, other.__class__) or \
               cmp(self.type_name,other.type_name)

"""
# this should only be enabled on machines with access to a display device
# It'll cause problems otherwise.
def test():
    from scipy_test import module_test
    module_test(__name__,__file__)

def test_suite():
    from scipy_test import module_test_suite
    return module_test_suite(__name__,__file__)    
"""        