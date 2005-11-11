""" Generic support code for: 
        error handling code found in every weave module      
        local/global dictionary access code for inline() modules
        swig pointer (old style) conversion support
        
"""

import base_info

module_support_code = \
"""

// global None value for use in functions.
namespace py {
object None = object(Py_None);
}

char* find_type(PyObject* py_obj)
{
    if(py_obj == NULL) return "C NULL value";
    if(PyCallable_Check(py_obj)) return "callable";
    if(PyString_Check(py_obj)) return "string";
    if(PyInt_Check(py_obj)) return "int";
    if(PyFloat_Check(py_obj)) return "float";
    if(PyDict_Check(py_obj)) return "dict";
    if(PyList_Check(py_obj)) return "list";
    if(PyTuple_Check(py_obj)) return "tuple";
    if(PyFile_Check(py_obj)) return "file";
    if(PyModule_Check(py_obj)) return "module";
    
    //should probably do more intergation (and thinking) on these.
    if(PyCallable_Check(py_obj) && PyInstance_Check(py_obj)) return "callable";
    if(PyInstance_Check(py_obj)) return "instance"; 
    if(PyCallable_Check(py_obj)) return "callable";
    return "unkown type";
}

void throw_error(PyObject* exc, const char* msg)
{
 //printf("setting python error: %s\\n",msg);
  PyErr_SetString(exc, msg);
  //printf("throwing error\\n");
  throw 1;
}

void handle_bad_type(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);    
}

void handle_conversion_error(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"Conversion Error:, received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}

"""
#include "compile.h" /* Scary dangerous stuff */
#include "frameobject.h" /* Scary dangerous stuff */

class basic_module_info(base_info.base_info):
    _headers = ['"Python.h"','"compile.h"','"frameobject.h"']
    _support_code = [module_support_code]

#----------------------------------------------------------------------------
# inline() generated support code
#
# The following two function declarations handle access to variables in the 
# global and local dictionaries for inline functions.
#----------------------------------------------------------------------------

get_variable_support_code = \
"""
void handle_variable_not_found(char*  var_name)
{
    char msg[500];
    sprintf(msg,"Conversion Error: variable '%s' not found in local or global scope.",var_name);
    throw_error(PyExc_NameError,msg);
}
PyObject* get_variable(char* name,PyObject* locals, PyObject* globals)
{
    // no checking done for error -- locals and globals should
    // already be validated as dictionaries.  If var is NULL, the
    // function calling this should handle it.
    PyObject* var = NULL;
    var = PyDict_GetItemString(locals,name);
    if (!var)
    {
        var = PyDict_GetItemString(globals,name);
    }
    if (!var)
        handle_variable_not_found(name);
    return var;
}
"""

py_to_raw_dict_support_code = \
"""
PyObject* py_to_raw_dict(PyObject* py_obj, char* name)
{
    // simply check that the value is a valid dictionary pointer.
    if(!py_obj || !PyDict_Check(py_obj))
        handle_bad_type(py_obj, "dictionary", name);
    return py_obj;
}
"""

class inline_info(base_info.base_info):
    _support_code = [get_variable_support_code, py_to_raw_dict_support_code]


#----------------------------------------------------------------------------
# swig pointer support code
#
# The support code for swig is just slirped in from the swigptr.c file 
# from the *old* swig distribution.  The code from swigptr.c is now a string
# in swigptr.py to ease the process of incorporating it into py2exe 
# installations. New style swig pointers are not yet supported.
#----------------------------------------------------------------------------

import swigptr
swig_support_code = swigptr.swigptr_code

class swig_info(base_info.base_info):
    _support_code = [swig_support_code]
