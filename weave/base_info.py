"""
    base_info holds classes that define the information
    needed for building C++ extension modules for Python that
    handle different data types.  The information includes
    such as include files, libraries, and even code snippets.
    
    base_info -- base class for cxx_info, blitz_info, etc.                  
    info_list -- a handy list class for working with multiple
                 info classes at the same time.            
"""
import os
import UserList

class base_info:
    _warnings =[]
    _headers = []
    _include_dirs = []
    _libraries = []
    _library_dirs = []
    _support_code = []
    _module_init_code = []
    _sources = []
    _define_macros = []
    _undefine_macros = []
    _extra_compile_args = []
    _extra_link_args = []
    compiler = ''
    def set_compiler(self,compiler):
        self.check_compiler(compiler)
        self.compiler = compiler
    # it would probably be better to specify what the arguments are
    # to avoid confusion, but I don't think these classes will get
    # very complicated, and I don't really know the variety of things
    # that should be passed in at this point.
    def check_compiler(self,compiler):
        pass        
    def warnings(self):   
        return self._warnings
    def headers(self):   
        return self._headers
    def include_dirs(self):
        return self._include_dirs
    def libraries(self):
        return self._libraries
    def library_dirs(self):
        return self._library_dirs
    def support_code(self):
        return self._support_code
    def module_init_code(self):
        return self._module_init_code
    def sources(self):
        return self._sources
    def define_macros(self):
        return self._define_macros
    def undefine_macros(self):
        return self._undefine_macros
    def extra_compile_args(self):
        return self._extra_compile_args
    def extra_link_args(self):
        return self._extra_link_args        
        
class custom_info(base_info):
    def __init__(self):
        self._warnings =[]
        self._headers = []
        self._include_dirs = []
        self._libraries = []
        self._library_dirs = []
        self._support_code = []
        self._module_init_code = []
        self._sources = []
        self._define_macros = []
        self._undefine_macros = []
        self._extra_compile_args = []
        self._extra_link_args = []

    def add_warning(self,warning):
        self._warnings.append(warning)
    def add_header(self,header):
        self._headers.append(header)
    def add_include_dir(self,include_dir):
        self._include_dirs.append(include_dir)
    def add_library(self,library):
        self._libraries.append(library)
    def add_library_dir(self,library_dir):
        self._library_dirs.append(library_dir)
    def add_support_code(self,support_code):
        self._support_code.append(support_code)
    def add_module_init_code(self,module_init_code):
        self._module_init_code.append(module_init_code)
    def add_source(self,source):
        self._sources.append(source)
    def add_define_macro(self,define_macro):
        self._define_macros.append(define_macro)
    def add_undefine_macro(self,undefine_macro):
        self._undefine_macros.append(undefine_macro)    
    def add_extra_compile_arg(self,compile_arg):
        return self._extra_compile_args.append(compile_arg)
    def add_extra_link_arg(self,link_arg):
        return self._extra_link_args.append(link_arg)        

class info_list(UserList.UserList):
    def get_unique_values(self,attribute):
        all_values = []        
        for info in self:
            vals = eval('info.'+attribute+'()')
            all_values.extend(vals)
        return unique_values(all_values)

    def extra_compile_args(self):
        return self.get_unique_values('extra_compile_args')
    def extra_link_args(self):
        return self.get_unique_values('extra_link_args')
    def sources(self):
        return self.get_unique_values('sources')    
    def define_macros(self):
        return self.get_unique_values('define_macros')
    def sources(self):
        return self.get_unique_values('sources')
    def warnings(self):
        return self.get_unique_values('warnings')
    def headers(self):
        return self.get_unique_values('headers')
    def include_dirs(self):
        return self.get_unique_values('include_dirs')
    def libraries(self):
        return self.get_unique_values('libraries')
    def library_dirs(self):
        return self.get_unique_values('library_dirs')
    def support_code(self):
        return self.get_unique_values('support_code')
    def module_init_code(self):
        return self.get_unique_values('module_init_code')

def unique_values(lst):
    all_values = []        
    for value in lst:
        if value not in all_values or value == '-framework':
            all_values.append(value)
    return all_values

