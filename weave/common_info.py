""" Generic support code for handling standard Numeric arrays      
"""

import base_info

#############################################################
# Basic module support code
#############################################################

module_support_code = \
"""

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

void handle_bad_type(PyObject* py_obj, char* good_type, char*  var_name)
{
    char msg[500];
    sprintf(msg,"received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw Py::TypeError(msg);
}
"""

class basic_module_info(base_info.base_info):
    _headers = ['"Python.h"']
    _support_code = [module_support_code]

#############################################################
# File conversion support code
#############################################################

file_convert_code =  \
"""
FILE* py_to_file(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyFile_Check(py_obj))
        handle_bad_type(py_obj,"file", name);

    // Cleanup code should call DECREF
    Py_INCREF(py_obj);
    return PyFile_AsFile(py_obj);
}

PyObject* file_to_py(FILE* file, char* name, char* mode)
{
    PyObject* py_obj = NULL;
    //extern int fclose(FILE *);
    return (PyObject*) PyFile_FromFile(file, name, mode, fclose);
}

"""

class file_info(base_info.base_info):
    _headers = ['<stdio.h>']
    _support_code = [file_convert_code]

#############################################################
# Instance info code
#############################################################

instance_convert_code = \
"""
PyObject* py_to_instance(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyFile_Check(py_obj))
        handle_bad_type(py_obj,"instance", name);

    // Should I INCREF???
    // Py_INCREF(py_obj);
    // just return the raw python pointer.
    return py_obj;
}

PyObject* instance_to_py(PyObject* instance)
{
    // Don't think I need to do anything...
    return (PyObject*) instance;
}

"""
class instance_info(base_info.base_info):
    _support_code = [instance_convert_code]

#############################################################
# Callable info code
#############################################################

callable_convert_code = \
"""
PyObject* py_to_callable(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyCallable_Check(py_obj))
        handle_bad_type(py_obj,"callable", name);

    // Should I INCREF???
    // Py_INCREF(py_obj);
    // just return the raw python pointer.
    return py_obj;
}

PyObject* callable_to_py(PyObject* callable)
{
    // Don't think I need to do anything...
    return (PyObject*) callable;
}

"""
class callable_info(base_info.base_info):
    _support_code = [callable_convert_code]

#############################################################
# Module info code
#############################################################

module_convert_code = \
"""
PyObject* py_to_module(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyModule_Check(py_obj))
        handle_bad_type(py_obj,"module", name);

    // Should I INCREF???
    // Py_INCREF(py_obj);
    // just return the raw python pointer.
    return py_obj;
}

PyObject* module_to_py(PyObject* module)
{
    // Don't think I need to do anything...
    return (PyObject*) module;
}

"""
class module_info(base_info.base_info):
    _support_code = [module_convert_code]
