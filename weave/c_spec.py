from types import *
from base_spec import base_converter
import base_info

#----------------------------------------------------------------------------
# C++ code template for converting code from python objects to C++ objects
#
# This is silly code.  There is absolutely no reason why these simple
# conversion functions should be classes.  However, some versions of 
# Mandrake Linux ship with broken C++ compilers (or libraries) that do not
# handle exceptions correctly when they are thrown from functions.  However,
# exceptions thrown from class methods always work, so we make everything
# a class method to solve this error.
#----------------------------------------------------------------------------

py_to_c_template = \
"""
class %(type_name)s_handler
{
public:    
    %(c_type)s convert_to_%(type_name)s(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        %(inc_ref_count)s
        if (!py_obj || !%(check_func)s(py_obj))
            handle_conversion_error(py_obj,"%(type_name)s", name);    
        return %(to_c_return)s;
    }
    
    %(c_type)s py_to_%(type_name)s(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !%(check_func)s(py_obj))
            handle_bad_type(py_obj,"%(type_name)s", name);    
        %(inc_ref_count)s
        return %(to_c_return)s;
    }
};

%(type_name)s_handler x__%(type_name)s_handler = %(type_name)s_handler();
#define convert_to_%(type_name)s(py_obj,name) \\
        x__%(type_name)s_handler.convert_to_%(type_name)s(py_obj,name)
#define py_to_%(type_name)s(py_obj,name) \\
        x__%(type_name)s_handler.py_to_%(type_name)s(py_obj,name)

"""

#----------------------------------------------------------------------------
# C++ code template for converting code from C++ objects to Python objects
#
#----------------------------------------------------------------------------

simple_c_to_py_template = \
"""
PyObject* %(type_name)s_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}

"""

class common_base_converter(base_converter):
    
    def __init__(self):
        self.init_info()
        self._build_information = [self.generate_build_info()]
    
    def init_info(self):
        self.matching_types = []
        self.headers = []
        self.include_dirs = []
        self.libraries = []
        self.library_dirs = []
        self.sources = []
        self.support_code = []
        self.module_init_code = []
        self.warnings = []
        self.define_macros = []
        self.use_ref_count = 1
        self.name = "no_name"
        self.c_type = 'PyObject*'
        self.to_c_return = 'py_obj'
    
    def info_object(self):
        return base_info.custom_info()
        
    def generate_build_info(self):
        info = self.info_object()
        for header in self.headers:
            info.add_header(header)
        for d in self.include_dirs:
            info.add_include_dir(d)
        for lib in self.libraries:
            info.add_library(lib)
        for d in self.library_dirs:
            info.add_library_dir(d)
        for source in self.sources:
            info.add_source(source)
        for code in self.support_code:
            info.add_support_code(code)
        info.add_support_code(self.py_to_c_code())
        info.add_support_code(self.c_to_py_code())
        for init_code in self.module_init_code:
            info.add_module_init_code(init_code)
        for macro in self.define_macros:
            info.add_define_macro(macro)
        for warning in self.warnings:
            info.add_warning(warning)
        return info

    def type_match(self,value):
        return type(value) in self.matching_types

    def get_var_type(self,value):
        return type(value)
        
    def type_spec(self,name,value):
        # factory
        new_spec = self.__class__()
        new_spec.name = name        
        new_spec.var_type = self.get_var_type(value)
        return new_spec

    def template_vars(self,inline=0):
        d = {}
        d['type_name'] = self.type_name
        d['check_func'] = self.check_func
        d['c_type'] = self.c_type
        d['to_c_return'] = self.to_c_return
        d['name'] = self.name
        d['py_var'] = self.py_variable()
        d['var_lookup'] = self.retrieve_py_variable(inline)
        code = 'convert_to_%(type_name)s(%(py_var)s,"%(name)s")' % d
        d['var_convert'] = code
        if self.use_ref_count:
            d['inc_ref_count'] = "Py_INCREF(py_obj);"
        else:
            d['inc_ref_count'] = ""
        return d

    def py_to_c_code(self):
        return py_to_c_template % self.template_vars()

    def c_to_py_code(self):
        return simple_c_to_py_template % self.template_vars()
        
    def declaration_code(self,templatize = 0,inline=0):
        code = '%(py_var)s = %(var_lookup)s;\n'   \
               '%(c_type)s %(name)s = %(var_convert)s;\n' %  \
               self.template_vars(inline=inline)
        return code       

    def cleanup_code(self):
        if self.use_ref_count:
            code = "Py_XDECREF(%(py_var)s);\n" % self.template_vars()
        else:
            code = ""    
        return code
    
    def __repr__(self):
        msg = "(file:: name: %s)" % self.name
        return msg
    def __cmp__(self,other):
        #only works for equal
        result = -1
        try:
            result = cmp(self.name,other.name) or \
                     cmp(self.__class__, other.__class__)
        except AttributeError:
            pass
        return result            

#----------------------------------------------------------------------------
# Module Converter
#----------------------------------------------------------------------------
class module_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'module'
        self.check_func = 'PyModule_Check'    
        # probably should test for callable classes here also.
        self.matching_types = [ModuleType]

#----------------------------------------------------------------------------
# Instance Converter
#----------------------------------------------------------------------------
class instance_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'instance'
        self.check_func = 'PyInstance_Check'    
        self.matching_types = [InstanceType]

#----------------------------------------------------------------------------
# Catchall Converter
#----------------------------------------------------------------------------
class catchall_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'catchall'
        self.check_func = ''    
    def type_match(self,value):
        return 1

#----------------------------------------------------------------------------
# String Converter
#----------------------------------------------------------------------------
class string_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'string'
        self.check_func = 'PyString_Check'    
        self.c_type = 'std::string'
        self.to_c_return = "std::string(PyString_AsString(py_obj))"
        self.matching_types = [StringType]
        self.headers.append('<string>')
    def c_to_py_code(self):
        # !! Need to dedent returned code.
        code = """
               PyObject* string_to_py(std::string s)
               {
                   return PyString_FromString(s.c_str());
               }
               """
        return code        

#----------------------------------------------------------------------------
# Unicode Converter
#----------------------------------------------------------------------------
class unicode_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'unicode'
        self.check_func = 'PyUnicode_Check'
        # This isn't supported by gcc 2.95.3 -- MSVC works fine with it.    
        #self.c_type = 'std::wstring'
        #self.to_c_return = "std::wstring(PyUnicode_AS_UNICODE(py_obj))"
        self.c_type = 'Py_UNICODE*'
        self.to_c_return = "PyUnicode_AS_UNICODE(py_obj)"
        self.matching_types = [UnicodeType]
        #self.headers.append('<string>')
#----------------------------------------------------------------------------
# File Converter
#----------------------------------------------------------------------------
class file_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'file'
        self.check_func = 'PyFile_Check'    
        self.c_type = 'FILE*'
        self.to_c_return = "PyFile_AsFile(py_obj)"
        self.headers = ['<stdio.h>']
        self.matching_types = [FileType]

    def c_to_py_code(self):
        # !! Need to dedent returned code.
        code = """
               PyObject* file_to_py(FILE* file, char* name, char* mode)
               {
                   PyObject* py_obj = NULL;
                   //extern int fclose(FILE *);
                   return (PyObject*) PyFile_FromFile(file, name, mode, fclose);
               }
               """
        return code        

#----------------------------------------------------------------------------
#
# Scalar Number Conversions
#
#----------------------------------------------------------------------------

# the following typemaps are for 32 bit platforms.  A way to do this
# general case? maybe ask numeric types how long they are and base
# the decisions on that.

#----------------------------------------------------------------------------
# Standard Python numeric --> C type maps
#----------------------------------------------------------------------------
num_to_c_types = {}
num_to_c_types[type(1)]  = 'int'
num_to_c_types[type(1.)] = 'double'
num_to_c_types[type(1.+1.j)] = 'std::complex<double> '
# !! hmmm. The following is likely unsafe...
num_to_c_types[type(1L)]  = 'int'

#----------------------------------------------------------------------------
# Numeric array Python numeric --> C type maps
#----------------------------------------------------------------------------
num_to_c_types['T'] = 'T' # for templates
num_to_c_types['F'] = 'std::complex<float> '
num_to_c_types['D'] = 'std::complex<double> '
num_to_c_types['f'] = 'float'
num_to_c_types['d'] = 'double'
num_to_c_types['1'] = 'char'
num_to_c_types['b'] = 'unsigned char'
num_to_c_types['s'] = 'short'
num_to_c_types['i'] = 'int'
# not strictly correct, but shoulld be fine fo numeric work.
# add test somewhere to make sure long can be cast to int before using.
num_to_c_types['l'] = 'int'

class scalar_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.warnings = ['disable: 4275', 'disable: 4101']
        self.headers = ['<complex>','<math.h>']
        self.use_ref_count = 0

class int_converter(scalar_converter):
    def init_info(self):
        scalar_converter.init_info(self)
        self.type_name = 'int'
        self.check_func = 'PyInt_Check'    
        self.c_type = 'int'
        self.to_c_return = "(int) PyInt_AsLong(py_obj)"
        self.matching_types = [IntType]

class long_converter(scalar_converter):
    def init_info(self):
        scalar_converter.init_info(self)
        # !! long to int conversion isn't safe!
        self.type_name = 'long'
        self.check_func = 'PyLong_Check'    
        self.c_type = 'int'
        self.to_c_return = "(int) PyLong_AsLong(py_obj)"
        self.matching_types = [LongType]

class float_converter(scalar_converter):
    def init_info(self):
        scalar_converter.init_info(self)
        # Not sure this is really that safe...
        self.type_name = 'float'
        self.check_func = 'PyFloat_Check'    
        self.c_type = 'double'
        self.to_c_return = "PyFloat_AsDouble(py_obj)"
        self.matching_types = [FloatType]

class complex_converter(scalar_converter):
    def init_info(self):
        scalar_converter.init_info(self)
        self.type_name = 'complex'
        self.check_func = 'PyComplex_Check'    
        self.c_type = 'std::complex<double>'
        self.to_c_return = "std::complex<double>(PyComplex_RealAsDouble(py_obj),"\
                                                "PyComplex_ImagAsDouble(py_obj))"
        self.matching_types = [ComplexType]

#----------------------------------------------------------------------------
#
# List, Tuple, and Dict converters.
#
# Based on SCXX by Gordon McMillan
#----------------------------------------------------------------------------
import os, c_spec # yes, I import myself to find out my __file__ location.
local_dir,junk = os.path.split(os.path.abspath(c_spec.__file__))   
scxx_dir = os.path.join(local_dir,'scxx')

class scxx_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.headers = ['"scxx/PWOBase.h"','"scxx/PWOSequence.h"',
                        '"scxx/PWOCallable.h"','"scxx/PWOMapping.h"',
                        '"scxx/PWOSequence.h"','"scxx/PWOMSequence.h"',
                        '"scxx/PWONumber.h"','<iostream>']
        self.include_dirs = [local_dir,scxx_dir]
        self.sources = [os.path.join(scxx_dir,'PWOImp.cpp'),]

class list_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'list'
        self.check_func = 'PyList_Check'    
        self.c_type = 'PWOList'
        self.to_c_return = 'PWOList(py_obj)'
        self.matching_types = [ListType]
        # ref counting handled by PWOList
        self.use_ref_count = 0

class tuple_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'tuple'
        self.check_func = 'PyTuple_Check'    
        self.c_type = 'PWOTuple'
        self.to_c_return = 'PWOTuple(py_obj)'
        self.matching_types = [TupleType]
        # ref counting handled by PWOTuple
        self.use_ref_count = 0

class dict_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.support_code.append("#define PWODict PWOMapping")
        self.type_name = 'dict'
        self.check_func = 'PyDict_Check'    
        self.c_type = 'PWODict'
        self.to_c_return = 'PWODict(py_obj)'
        self.matching_types = [DictType]
        # ref counting handled by PWODict
        self.use_ref_count = 0

#----------------------------------------------------------------------------
# Callable Converter
#----------------------------------------------------------------------------
class callable_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'callable'
        self.check_func = 'PyCallable_Check'    
        # probably should test for callable classes here also.
        self.matching_types = [FunctionType,MethodType,type(len)]
        self.c_type = 'PWOCallable'
        self.to_c_return = 'PWOCallable(py_obj)'
        # ref counting handled by PWOCallable
        self.use_ref_count = 0

def test(level=10):
    from scipy_test.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_test.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)

if __name__ == "__main__":
    x = list_converter().type_spec("x",1)
    print x.py_to_c_code()
    print
    print x.c_to_py_code()
    print
    print x.declaration_code(inline=1)
    print
    print x.cleanup_code()