class base_converter:
    """
        Properties:
        headers --  list of strings that name the header files needed by this
                    object.
        include_dirs -- list of directories where the header files can be found.
        libraries   -- list of libraries needed to link to when compiling
                       extension.
        library_dirs -- list of directories to search for libraries.

        support_code -- list of strings.  Each string is a subroutine needed
                        by the type.  Functions that are used in the conversion
                        between Python and C++ files are examples of these.

        Methods:

        type_match(value) returns 1 if this class is used to represent type
                          specification for value.
        type_spec(name, value)  returns a new object (of this class) that is
                                used to produce C++ code for value.
        declaration_code()    returns C++ code fragment for type declaration and
                              conversion of python object to C++ object.
        cleanup_code()    returns C++ code fragment for cleaning up after the
                          variable after main C++ code fragment has executed.

    """
    _build_information = []
    compiler = ''   
                
    def set_compiler(self,compiler):
        self.compiler = compiler
    def type_match(self,value):
        raise NotImplementedError, "You must override method in derived class"
    def build_information(self):
        return self._build_information
    def type_spec(self,name,value): 
        pass
    def declaration_code(self,templatize = 0):   
        return ""
    def local_dict_code(self): 
        return ""
    def cleanup_code(self): 
        return ""
    def retrieve_py_variable(self,inline=0):
        # this needs a little coordination in name choices with the
        # ext_inline_function class.
        if inline:
            vn = 'get_variable("%s",raw_locals,raw_globals)' % self.name
        else:
            vn = 'py_' + self.name   
        return vn
        
    def py_reference(self):
        return "&py_" + self.name
    def py_pointer(self):
        return "*py_" + self.name
    def py_variable(self):
        return "py_" + self.name
    def reference(self):
        return "&" + self.name
    def pointer(self):
        return "*" + self.name
    def init_flag(self):
        return self.name + "_used"
    
    def variable(self):
        return self.name
    def variable_as_string(self):
        return '"' + self.name + '"'

import UserList
import base_info

class arg_spec_list(UserList.UserList):    
    def build_information(self): 
        all_info = base_info.info_list()
        for i in self:
            all_info.extend(i.build_information())
        return all_info
        
    def py_references(self): 
        return map(lambda x: x.py_reference(),self)
    def py_pointers(self): 
        return map(lambda x: x.py_pointer(),self)
    def py_variables(self): 
        return map(lambda x: x.py_variable(),self)

    def references(self): 
        return map(lambda x: x.py_reference(),self)
    def pointers(self): 
        return map(lambda x: x.pointer(),self)    
    def variables(self): 
        return map(lambda x: x.variable(),self)
    def init_flags(self): 
        return map(lambda x: x.init_flag(),self)
    def variable_as_strings(self): 
        return map(lambda x: x.variable_as_string(),self)

    