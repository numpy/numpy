"""
    build_info holds classes that define the information
    needed for building C++ extension modules for Python that
    handle different data types.  The information includes
    such as include files, libraries, and even code snippets.
       
    array_info -- for building functions that use Python
                  Numeric arrays.
"""

import base_info

blitz_support_code =  \
"""

// This should be declared only if they are used by some function
// to keep from generating needless warnings. for now, we'll always
// declare them.

int _beg = blitz::fromStart;
int _end = blitz::toEnd;
blitz::Range _all = blitz::Range::all();

template<class T, int N>
static blitz::Array<T,N> convert_to_blitz(PyArrayObject* arr_obj,const char* name)
{

    //This is now handled externally (for now) to deal with exception/Abort issue
    //PyArrayObject* arr_obj = convert_to_numpy(py_obj,name);
    //conversion_numpy_check_size(arr_obj,N,name);
    //conversion_numpy_check_type(arr_obj,py_type<T>::code,name);
    
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    int stride_acc = 1;
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,        
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}

template<class T, int N>
static blitz::Array<T,N> py_to_blitz(PyArrayObject* arr_obj,const char* name)
{
    //This is now handled externally (for now) to deal with exception/Abort issue
    //PyArrayObject* arr_obj = py_to_numpy(py_obj,name);
    //numpy_check_size(arr_obj,N,name);
    //numpy_check_type(arr_obj,py_type<T>::code,name);
    
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    int stride_acc = 1;
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,        
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}
"""

# this code will not build with msvc...
# This is only used for blitz stuff now.  The non-templated
# version, defined further down, is now used for most code.
scalar_support_code = \
"""
// conversion routines

template<class T> 
static T convert_to_scalar(PyObject* py_obj,const char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int convert_to_scalar<int>(PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_conversion_error(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

template<>
static long convert_to_scalar<long>(PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_conversion_error(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

template<> 
static double convert_to_scalar<double>(PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_conversion_error(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

template<> 
static float convert_to_scalar<float>(PyObject* py_obj,const char* name)
{
    return (float) convert_to_scalar<double>(py_obj,name);
}

// complex not checked.
template<> 
static std::complex<float> convert_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              const char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_ImagAsDouble(py_obj));    
}
template<> 
static std::complex<double> convert_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}

/////////////////////////////////
// standard translation routines

template<class T> 
static T py_to_scalar(PyObject* py_obj,const char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int py_to_scalar<int>(PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_bad_type(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

template<>
static long py_to_scalar<long>(PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_bad_type(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

template<> 
static double py_to_scalar<double>(PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_bad_type(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

template<> 
static float py_to_scalar<float>(PyObject* py_obj,const char* name)
{
    return (float) py_to_scalar<double>(py_obj,name);
}

// complex not checked.
template<> 
static std::complex<float> py_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              const char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_ImagAsDouble(py_obj));    
}
template<> 
static std::complex<double> py_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,const char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}
"""    

import standard_array_info
import os, blitz_info
local_dir,junk = os.path.split(os.path.abspath(blitz_info.__file__))   
blitz_dir = os.path.join(local_dir,'blitz-20001213')

class array_info(base_info.base_info):
    _include_dirs = [blitz_dir]
    _headers = ['"blitz/array.h"','"Numeric/arrayobject.h"','<complex>','<math.h>']
    
    _support_code = [standard_array_info.array_convert_code,
                     standard_array_info.type_check_code,
                     standard_array_info.size_check_code,
                     scalar_support_code,
                     blitz_support_code,
                    ]
    _module_init_code = [standard_array_info.numeric_init_code]    
    
    # throw error if trying to use msvc compiler
    
    def check_compiler(self,compiler):        
        msvc_msg = 'Unfortunately, the blitz arrays used to support numeric' \
                   ' arrays will not compile with MSVC.' \
                   '  Please try using mingw32 (www.mingw.org).'
        if compiler == 'msvc':
            return ValueError, self.msvc_msg        