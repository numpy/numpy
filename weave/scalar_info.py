""" support code and other things needed to compile support
    for numeric expressions in python.
    
    There are two sets of support code, one with templated
    functions and one without.  This is because msvc cannot
    handle the templated functions.  We need the templated
    versions for more complex support of numeric arrays with
    blitz. 
"""

import base_info

# this code will not build with msvc...
scalar_support_code = \
"""
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

# this code will not build with msvc...
non_template_scalar_support_code = \
"""
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

class scalar_info(base_info.base_info):
    _warnings = ['disable: 4275', 'disable: 4101']
    _headers = ['<complex>','<math.h>']
    def support_code(self):
        if self.compiler != 'msvc':
             # maybe this should only be for gcc...
            return [scalar_support_code,non_template_scalar_support_code]
        else:
            return [non_template_scalar_support_code]
            