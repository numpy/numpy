""" C/C++ code strings needed for converting most non-sequence
    Python variables:
        module_support_code -- several routines used by most other code 
                               conversion methods.  It holds the only
                               CXX dependent code in this file.  The CXX
                               stuff is used for exceptions
        file_convert_code
        instance_convert_code
        callable_convert_code
        module_convert_code
        
        scalar_convert_code
        non_template_scalar_support_code               
            Scalar conversion covers int, float, double, complex,
            and double complex.  While Python doesn't support all these,
            Numeric does and so all of them are made available.
            Python longs are currently converted to C ints.  Any
            better way to handle this?
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

void throw_error(PyObject* exc, const char* msg)
{
  PyErr_SetString(exc, msg);
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

#############################################################
# File conversion support code
#############################################################

file_convert_code =  \
"""

class file_handler
{
public:
    FILE* convert_to_file(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyFile_Check(py_obj))
            handle_conversion_error(py_obj,"file", name);
    
        // Cleanup code should call DECREF
        Py_INCREF(py_obj);
        return PyFile_AsFile(py_obj);
    }
    
    FILE* py_to_file(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyFile_Check(py_obj))
            handle_bad_type(py_obj,"file", name);
    
        // Cleanup code should call DECREF
        Py_INCREF(py_obj);
        return PyFile_AsFile(py_obj);
    }
};

file_handler x__file_handler = file_handler();
#define convert_to_file(py_obj,name) x__file_handler.convert_to_file(py_obj,name)
#define py_to_file(py_obj,name) x__file_handler.py_to_file(py_obj,name)

PyObject* file_to_py(FILE* file, char* name, char* mode)
{
    PyObject* py_obj = NULL;
    //extern int fclose(FILE *);
    return (PyObject*) PyFile_FromFile(file, name, mode, fclose);
}

"""


#############################################################
# Instance conversion code
#############################################################

instance_convert_code = \
"""

class instance_handler
{
public:
    PyObject* convert_to_instance(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyFile_Check(py_obj))
            handle_conversion_error(py_obj,"instance", name);
    
        // Should I INCREF???
        // Py_INCREF(py_obj);
        // just return the raw python pointer.
        return py_obj;
    }
    
    PyObject* py_to_instance(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyFile_Check(py_obj))
            handle_bad_type(py_obj,"instance", name);
    
        // Should I INCREF???
        // Py_INCREF(py_obj);
        // just return the raw python pointer.
        return py_obj;
    }
};

instance_handler x__instance_handler = instance_handler();
#define convert_to_instance(py_obj,name) x__instance_handler.convert_to_instance(py_obj,name)
#define py_to_instance(py_obj,name) x__instance_handler.py_to_instance(py_obj,name)

PyObject* instance_to_py(PyObject* instance)
{
    // Don't think I need to do anything...
    return (PyObject*) instance;
}

"""

#############################################################
# Callable conversion code
#############################################################

callable_convert_code = \
"""

class callable_handler
{
public:    
    PyObject* convert_to_callable(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyCallable_Check(py_obj))
            handle_conversion_error(py_obj,"callable", name);
    
        // Should I INCREF???
        // Py_INCREF(py_obj);
        // just return the raw python pointer.
        return py_obj;
    }
    
    PyObject* py_to_callable(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyCallable_Check(py_obj))
            handle_bad_type(py_obj,"callable", name);
    
        // Should I INCREF???
        // Py_INCREF(py_obj);
        // just return the raw python pointer.
        return py_obj;
    }
};

callable_handler x__callable_handler = callable_handler();
#define convert_to_callable(py_obj,name) x__callable_handler.convert_to_callable(py_obj,name)
#define py_to_callable(py_obj,name) x__callable_handler.py_to_callable(py_obj,name)

PyObject* callable_to_py(PyObject* callable)
{
    // Don't think I need to do anything...
    return (PyObject*) callable;
}

"""

#############################################################
# Module conversion code
#############################################################

module_convert_code = \
"""
class module_handler
{
public:
    PyObject* convert_to_module(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyModule_Check(py_obj))
            handle_conversion_error(py_obj,"module", name);
    
        // Should I INCREF???
        // Py_INCREF(py_obj);
        // just return the raw python pointer.
        return py_obj;
    }
    
    PyObject* py_to_module(PyObject* py_obj, const char* name)
    {
        if (!py_obj || !PyModule_Check(py_obj))
            handle_bad_type(py_obj,"module", name);
    
        // Should I INCREF???
        // Py_INCREF(py_obj);
        // just return the raw python pointer.
        return py_obj;
    }
};

module_handler x__module_handler = module_handler();
#define convert_to_module(py_obj,name) x__module_handler.convert_to_module(py_obj,name)
#define py_to_module(py_obj,name) x__module_handler.py_to_module(py_obj,name)

PyObject* module_to_py(PyObject* module)
{
    // Don't think I need to do anything...
    return (PyObject*) module;
}

"""

#############################################################
# Scalar conversion code
#############################################################

# These non-templated version is now used for most scalar conversions.
scalar_support_code = \
"""

class scalar_handler
{
public:    
    int convert_to_int(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyInt_Check(py_obj))
            handle_conversion_error(py_obj,"int", name);
        return (int) PyInt_AsLong(py_obj);
    }
    long convert_to_long(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyLong_Check(py_obj))
            handle_conversion_error(py_obj,"long", name);
        return (long) PyLong_AsLong(py_obj);
    }

    double convert_to_float(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_conversion_error(py_obj,"float", name);
        return PyFloat_AsDouble(py_obj);
    }

    std::complex<double> convert_to_complex(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_conversion_error(py_obj,"complex", name);
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                    PyComplex_ImagAsDouble(py_obj));    
    }

    int py_to_int(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyInt_Check(py_obj))
            handle_bad_type(py_obj,"int", name);
        return (int) PyInt_AsLong(py_obj);
    }
    
    long py_to_long(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyLong_Check(py_obj))
            handle_bad_type(py_obj,"long", name);
        return (long) PyLong_AsLong(py_obj);
    }
    
    double py_to_float(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_bad_type(py_obj,"float", name);
        return PyFloat_AsDouble(py_obj);
    }
    
    std::complex<double> py_to_complex(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_bad_type(py_obj,"complex", name);
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                    PyComplex_ImagAsDouble(py_obj));    
    }

};

scalar_handler x__scalar_handler = scalar_handler();
#define convert_to_int(py_obj,name) x__scalar_handler.convert_to_int(py_obj,name)
#define py_to_int(py_obj,name) x__scalar_handler.py_to_int(py_obj,name)

#define convert_to_long(py_obj,name) x__scalar_handler.convert_to_long(py_obj,name)
#define py_to_long(py_obj,name) x__scalar_handler.py_to_long(py_obj,name)

#define convert_to_float(py_obj,name) x__scalar_handler.convert_to_float(py_obj,name)
#define py_to_float(py_obj,name) x__scalar_handler.py_to_float(py_obj,name)

#define convert_to_complex(py_obj,name) x__scalar_handler.convert_to_complex(py_obj,name)
#define py_to_complex(py_obj,name) x__scalar_handler.py_to_complex(py_obj,name)

"""    

