""" This converter works with classes protected by a namespace with
    SWIG pointers (Python strings).  To use it to wrap classes in
    a C++ namespace called "ft", use the following:
    
    class ft_converter(cpp_namespace_converter):
        namespace = 'ft::'        
"""

from weave import common_info
from weave import  base_info
from weave.base_spec import base_converter

cpp_support_template = \
"""
static %(cpp_struct)s* convert_to_%(cpp_clean_struct)s(PyObject* py_obj,char* name)
{
    %(cpp_struct)s *cpp_ptr = 0;
    char* str = PyString_AsString(py_obj);
    if (!str)
        handle_conversion_error(py_obj,"%(cpp_struct)s", name);
    // work on this error reporting...
    //std::cout << "in:" << name << " " py_obj << std::endl;
    if (SWIG_GetPtr(str,(void **) &cpp_ptr,"_%(cpp_struct)s_p"))
    {
        handle_conversion_error(py_obj,"%(cpp_struct)s", name);
    }
    //std::cout << "out:" << name << " " << str << std::endl;
    return cpp_ptr;
}    

static %(cpp_struct)s* py_to_%(cpp_clean_struct)s(PyObject* py_obj,char* name)
{
    %(cpp_struct)s *cpp_ptr;
    char* str = PyString_AsString(py_obj);
    if (!str)
        handle_conversion_error(py_obj,"%(cpp_struct)s", name);
    // work on this error reporting...
    if (SWIG_GetPtr(str,(void **) &cpp_ptr,"_%(cpp_struct)s_p"))
    {
        handle_conversion_error(py_obj,"%(cpp_struct)s", name);
    }
    return cpp_ptr;
}    

std::string %(cpp_clean_struct)s_to_py( %(cpp_struct)s* cpp_ptr)
{
    char ptr_string[%(ptr_string_len)s]; 
    SWIG_MakePtr(ptr_string, cpp_ptr, "_%(cpp_struct)s_p");
    return std::string(ptr_string);
}              

"""        

class cpp_namespace_converter(base_converter):
    _build_information = [common_info.swig_info()]
    def __init__(self,class_name=None):
        self.type_name = 'unkown cpp_object'
        self.name =  'no name'        
        if class_name:
            # customize support_code for whatever type I was handed.
            clean_name = class_name.replace('::','_')
            clean_name = clean_name.replace('<','_')
            clean_name = clean_name.replace('>','_')
            clean_name = clean_name.replace(' ','_')
            # should be enough for 64 bit machines
            str_len = len(clean_name) + 20 
            vals = {'cpp_struct': class_name,
                    'cpp_clean_struct': clean_name,
                    'ptr_string_len': str_len }
            specialized_support = cpp_support_template % vals
            custom = base_info.base_info()
            custom._support_code = [specialized_support]
            self._build_information = self._build_information + [custom]
            self.type_name = class_name

    def type_match(self,value):
        try:
            cpp_ident = value.split('_')[2]
            if cpp_ident.find(self.namespace) != -1:
                return 1
        except:
            pass
        return 0
            
    def type_spec(self,name,value):
        # factory
        ptr_fields = value.split('_')
        class_name = '_'.join(ptr_fields[2:-1])
        new_spec = self.__class__(class_name)
        new_spec.name = name        
        return new_spec
        
    def declaration_code(self,inline=0):
        type = self.type_name
        clean_type = type.replace('::','_')
        name = self.name
        var_name = self.retrieve_py_variable(inline)
        template = '%(type)s *%(name)s = '\
                   'convert_to_%(clean_type)s(%(var_name)s,"%(name)s");\n'
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
