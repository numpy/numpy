module_header = \
"""    
// blitz must be first, or you get have issues with isnan defintion.
#include <blitz/array.h> 

#include "Python.h" 
#include "Numeric/arrayobject.h" 

// Use Exception stuff from SCXX
#include "PWOBase.h" 

#include <stdio.h> 
#include <math.h> 
#include <complex>

static PyArrayObject* obj_to_numpy(PyObject* py_obj, char* name, 
                                  int Ndims, int numeric_type )
{
    PyArrayObject* arr_obj = NULL;

    // Make sure input is an array.
    if (!PyArray_Check(py_obj))
        throw PWException(PyExc_TypeError,
                          "Input array *name* must be an array.");

    arr_obj = (PyArrayObject*) py_obj;
    
    // Make sure input has correct numeric type.
    if (arr_obj->descr->type_num != numeric_type)
    {
        // This should be more explicit:
        // Put the desired and actual type in the message.
        // printf("%d,%d",arr_obj->descr->type_num,numeric_type);
        throw PWException(PyExc_TypeError,
                          "Input array *name* is the wrong numeric type.");
    }
    
    // Make sure input has correct rank (defined as number of dimensions).
    // Currently, all arrays must have the same shape.
    // Broadcasting is not supported.
    // ...
    if (arr_obj->nd != Ndims)
    {
        // This should be more explicit:
        // Put the desired and actual dimensionality in message.
        throw PWException(PyExc_TypeError,
                         "Input array *name* has wrong number of dimensions.");
    }    
    // check the size of arrays.  Acutally, the size of the "views" really
    // needs checking -- not the arrays.
    // ...
    
    // Any need to deal with INC/DEC REFs?
    Py_INCREF(py_obj);
    return arr_obj;
}

// simple meta-program templates to specify python typecodes
// for each of the numeric types.
template<class T>
class py_type{public: enum {code = 100};};
class py_type<char>{public: enum {code = PyArray_CHAR};};
class py_type<unsigned char>{public: enum { code = PyArray_UBYTE};};
class py_type<short>{public:  enum { code = PyArray_SHORT};};
class py_type<int>{public: enum { code = PyArray_INT};};
class py_type<long>{public: enum { code = PyArray_LONG};};
class py_type<float>{public: enum { code = PyArray_FLOAT};};
class py_type<double>{public: enum { code = PyArray_DOUBLE};};
class py_type<complex<float> >{public: enum { code = PyArray_CFLOAT};};
class py_type<complex<double> >{public: enum { code = PyArray_CDOUBLE};};

template<class T, int N>
static blitz::Array<T,N> py_to_blitz(PyObject* py_obj,char* name)
{

    PyArrayObject* arr_obj = obj_to_numpy(py_obj,name,N,py_type<T>::code);
    
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

template<class T> 
static T py_to_scalar(PyObject* py_obj,char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int py_to_scalar<int>(PyObject* py_obj,char* name)
{
    return (int) PyLong_AsLong(py_obj);
}

template<>
static long py_to_scalar<long>(PyObject* py_obj,char* name)
{
    return (long) PyLong_AsLong(py_obj);
}
template<> 
static float py_to_scalar<float>(PyObject* py_obj,char* name)
{
    return (float) PyFloat_AsDouble(py_obj);
}
template<> 
static double py_to_scalar<double>(PyObject* py_obj,char* name)
{
    return PyFloat_AsDouble(py_obj);
}

// complex not checked.
template<> 
static std::complex<float> py_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              char* name)
{
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_RealAsDouble(py_obj));    
}
template<> 
static std::complex<double> py_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,char* name)
{
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_RealAsDouble(py_obj));    
}
"""    