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

void handle_bad_type(PyObject* py_obj, char* good_type, char* var_name)
{
    char msg[500];
    sprintf(msg,"received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw Py::TypeError(msg);
}

void handle_conversion_error(PyObject* py_obj, char* good_type, char* var_name)
{
    char msg[500];
    sprintf(msg,"Conversion Error:, received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw Py::TypeError(msg);
}

"""

#############################################################
# File conversion support code
#############################################################

file_convert_code =  \
"""

FILE* convert_to_file(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyFile_Check(py_obj))
        handle_conversion_error(py_obj,"file", name);

    // Cleanup code should call DECREF
    Py_INCREF(py_obj);
    return PyFile_AsFile(py_obj);
}

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

#############################################################
# Instance conversion code
#############################################################

instance_convert_code = \
"""

PyObject* convert_to_instance(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyFile_Check(py_obj))
        handle_conversion_error(py_obj,"instance", name);

    // Should I INCREF???
    // Py_INCREF(py_obj);
    // just return the raw python pointer.
    return py_obj;
}

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

#############################################################
# Callable conversion code
#############################################################

callable_convert_code = \
"""

PyObject* convert_to_callable(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyCallable_Check(py_obj))
        handle_conversion_error(py_obj,"callable", name);

    // Should I INCREF???
    // Py_INCREF(py_obj);
    // just return the raw python pointer.
    return py_obj;
}

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

#############################################################
# Module conversion code
#############################################################

module_convert_code = \
"""
PyObject* convert_to_module(PyObject* py_obj, char* name)
{
    if (!py_obj || !PyModule_Check(py_obj))
        handle_conversion_error(py_obj,"module", name);

    // Should I INCREF???
    // Py_INCREF(py_obj);
    // just return the raw python pointer.
    return py_obj;
}

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

#############################################################
# Scalar conversion code
#############################################################

import base_info

# this code will not build with msvc...
scalar_support_code = \
"""
// conversion routines

template<class T> 
static T convert_to_scalar(PyObject* py_obj,char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int convert_to_scalar<int>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_conversion_error(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

template<>
static long convert_to_scalar<long>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_conversion_error(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

template<> 
static double convert_to_scalar<double>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_conversion_error(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

template<> 
static float convert_to_scalar<float>(PyObject* py_obj,char* name)
{
    return (float) convert_to_scalar<double>(py_obj,name);
}

// complex not checked.
template<> 
static std::complex<float> convert_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_ImagAsDouble(py_obj));    
}
template<> 
static std::complex<double> convert_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}

/////////////////////////////////
// standard translation routines

template<class T> 
static T py_to_scalar(PyObject* py_obj,char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int py_to_scalar<int>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_bad_type(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

template<>
static long py_to_scalar<long>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_bad_type(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

template<> 
static double py_to_scalar<double>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_bad_type(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

template<> 
static float py_to_scalar<float>(PyObject* py_obj,char* name)
{
    return (float) py_to_scalar<double>(py_obj,name);
}

// complex not checked.
template<> 
static std::complex<float> py_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_ImagAsDouble(py_obj));    
}
template<> 
static std::complex<double> py_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}
"""    

non_template_scalar_support_code = \
"""

// Conversion Errors

static int convert_to_int(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_conversion_error(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

static long convert_to_long(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_conversion_error(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

static double convert_to_float(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_conversion_error(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

// complex not checked.
static std::complex<double> convert_to_complex(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}

/////////////////////////////////////
// The following functions are used for scalar conversions in msvc
// because it doesn't handle templates as well.

static int py_to_int(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_bad_type(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

static long py_to_long(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_bad_type(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

static double py_to_float(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_bad_type(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

// complex not checked.
static std::complex<double> py_to_complex(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}
"""    
