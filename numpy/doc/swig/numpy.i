/* -*- C -*-  (not really, but good for syntax highlighting) */
%{
#ifndef SWIG_FILE_WITH_INIT
#  define NO_IMPORT_ARRAY
#endif
#include "stdio.h"
#include <numpy/arrayobject.h>

/* The following code originally appeared in enthought/kiva/agg/src/numeric.i,
 * author unknown.  It was translated from C++ to C by John Hunter.  Bill
 * Spotz has modified it slightly to fix some minor bugs, add some comments
 * and some functionality.
 */

/* Macros to extract array attributes.
 */
#define is_array(a)            ((a) && PyArray_Check((PyArrayObject *)a))
#define array_type(a)          (int)(PyArray_TYPE(a))
#define array_dimensions(a)    (((PyArrayObject *)a)->nd)
#define array_size(a,i)        (((PyArrayObject *)a)->dimensions[i])
#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS(a))

/* Given a PyObject, return a string describing its type.
 */
char* pytype_string(PyObject* py_obj) {
  if (py_obj == NULL          ) return "C NULL value";
  if (PyCallable_Check(py_obj)) return "callable"    ;
  if (PyString_Check(  py_obj)) return "string"      ;
  if (PyInt_Check(     py_obj)) return "int"         ;
  if (PyFloat_Check(   py_obj)) return "float"       ;
  if (PyDict_Check(    py_obj)) return "dict"        ;
  if (PyList_Check(    py_obj)) return "list"        ;
  if (PyTuple_Check(   py_obj)) return "tuple"       ;
  if (PyFile_Check(    py_obj)) return "file"        ;
  if (PyModule_Check(  py_obj)) return "module"      ;
  if (PyInstance_Check(py_obj)) return "instance"    ;

  return "unkown type";
}

/* Given a Numeric typecode, return a string describing the type.
 */
char* typecode_string(int typecode) {
  char* type_names[20] = {"char","unsigned byte","byte","short",
			  "unsigned short","int","unsigned int","long",
			  "float","double","complex float","complex double",
			  "object","ntype","unkown"};
  return type_names[typecode];
}

/* Make sure input has correct numeric type.  Allow character and byte
 * to match.  Also allow int and long to match.
 */
int type_match(int actual_type, int desired_type) {
  return PyArray_EquivTypenums(actual_type, desired_type);
}

/* Given a PyObject pointer, cast it to a PyArrayObject pointer if
 * legal.  If not, set the python error string appropriately and
 * return NULL./
 */
PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode) {
  PyArrayObject* ary = NULL;
  if (is_array(input) && (typecode == PyArray_NOTYPE || 
			  PyArray_EquivTypenums(array_type(input), 
						typecode))) {
        ary = (PyArrayObject*) input;
    }
    else if is_array(input) {
      char* desired_type = typecode_string(typecode);
      char* actual_type = typecode_string(array_type(input));
      PyErr_Format(PyExc_TypeError, 
		   "Array of type '%s' required.  Array of type '%s' given", 
		   desired_type, actual_type);
      ary = NULL;
    }
    else {
      char * desired_type = typecode_string(typecode);
      char * actual_type = pytype_string(input);
      PyErr_Format(PyExc_TypeError, 
		   "Array of type '%s' required.  A %s was given", 
		   desired_type, actual_type);
      ary = NULL;
    }
  return ary;
}

/* Convert the given PyObject to a Numeric array with the given
 * typecode.  On Success, return a valid PyArrayObject* with the
 * correct type.  On failure, the python error string will be set and
 * the routine returns NULL.
 */
PyArrayObject* obj_to_array_allow_conversion(PyObject* input, int typecode,
                                             int* is_new_object)
{
  PyArrayObject* ary = NULL;
  PyObject* py_obj;
  if (is_array(input) && (typecode == PyArray_NOTYPE || type_match(array_type(input),typecode))) {
    ary = (PyArrayObject*) input;
    *is_new_object = 0;
  }
  else {
    py_obj = PyArray_FromObject(input, typecode, 0, 0);
    /* If NULL, PyArray_FromObject will have set python error value.*/
    ary = (PyArrayObject*) py_obj;
    *is_new_object = 1;
  }
  return ary;
}

/* Given a PyArrayObject, check to see if it is contiguous.  If so,
 * return the input pointer and flag it as not a new object.  If it is
 * not contiguous, create a new PyArrayObject using the original data,
 * flag it as a new object and return the pointer.
 */
PyArrayObject* make_contiguous(PyArrayObject* ary, int* is_new_object,
                               int min_dims, int max_dims)
{
  PyArrayObject* result;
  if (array_is_contiguous(ary)) {
    result = ary;
    *is_new_object = 0;
  }
  else {
    result = (PyArrayObject*) PyArray_ContiguousFromObject((PyObject*)ary, 
							   array_type(ary), 
							   min_dims,
							   max_dims);
    *is_new_object = 1;
  }
  return result;
}

/* Convert a given PyObject to a contiguous PyArrayObject of the
 * specified type.  If the input object is not a contiguous
 * PyArrayObject, a new one will be created and the new object flag
 * will be set.
 */
PyArrayObject* obj_to_array_contiguous_allow_conversion(PyObject* input,
                                                        int typecode,
                                                        int* is_new_object) {
  int is_new1 = 0;
  int is_new2 = 0;
  PyArrayObject* ary2;
  PyArrayObject* ary1 = obj_to_array_allow_conversion(input, typecode, 
						      &is_new1);
  if (ary1) {
    ary2 = make_contiguous(ary1, &is_new2, 0, 0);
    if ( is_new1 && is_new2) {
      Py_DECREF(ary1);
    }
    ary1 = ary2;    
  }
  *is_new_object = is_new1 || is_new2;
  return ary1;
}

/* Test whether a python object is contiguous.  If array is
 * contiguous, return 1.  Otherwise, set the python error string and
 * return 0.
 */
int require_contiguous(PyArrayObject* ary) {
  int contiguous = 1;
  if (!array_is_contiguous(ary)) {
    PyErr_SetString(PyExc_TypeError, "Array must be contiguous.  A discontiguous array was given");
    contiguous = 0;
  }
  return contiguous;
}

/* Require the given PyArrayObject to have a specified number of
 * dimensions.  If the array has the specified number of dimensions,
 * return 1.  Otherwise, set the python error string and return 0.
 */
int require_dimensions(PyArrayObject* ary, int exact_dimensions) {
  int success = 1;
  if (array_dimensions(ary) != exact_dimensions) {
    PyErr_Format(PyExc_TypeError, 
		 "Array must be have %d dimensions.  Given array has %d dimensions", 
		 exact_dimensions, array_dimensions(ary));
    success = 0;
  }
  return success;
}

/* Require the given PyArrayObject to have one of a list of specified
 * number of dimensions.  If the array has one of the specified number
 * of dimensions, return 1.  Otherwise, set the python error string
 * and return 0.
 */
int require_dimensions_n(PyArrayObject* ary, int* exact_dimensions, int n) {
  int success = 0;
  int i;
  char dims_str[255] = "";
  char s[255];
  for (i = 0; i < n && !success; i++) {
    if (array_dimensions(ary) == exact_dimensions[i]) {
      success = 1;
    }
  }
  if (!success) {
    for (i = 0; i < n-1; i++) {
      sprintf(s, "%d, ", exact_dimensions[i]);                
      strcat(dims_str,s);
    }
    sprintf(s, " or %d", exact_dimensions[n-1]);            
    strcat(dims_str,s);
    PyErr_Format(PyExc_TypeError, 
		 "Array must be have %s dimensions.  Given array has %d dimensions",
		 dims_str, array_dimensions(ary));
  }
  return success;
}    

/* Require the given PyArrayObject to have a specified shape.  If the
 * array has the specified shape, return 1.  Otherwise, set the python
 * error string and return 0.
 */
int require_size(PyArrayObject* ary, int* size, int n) {
  int i;
  int success = 1;
  int len;
  char desired_dims[255] = "[";
  char s[255];
  char actual_dims[255] = "[";
  for(i=0; i < n;i++) {
    if (size[i] != -1 &&  size[i] != array_size(ary,i)) {
      success = 0;    
    }
  }
  if (!success) {
    for (i = 0; i < n; i++) {
      if (size[i] == -1) {
	sprintf(s, "*,");                
      }
      else
      {
	sprintf(s, "%d,", size[i]);                
      }    
      strcat(desired_dims,s);
    }
    len = strlen(desired_dims);
    desired_dims[len-1] = ']';
    for (i = 0; i < n; i++) {
      sprintf(s, "%d,", array_size(ary,i));                            
      strcat(actual_dims,s);
    }
    len = strlen(actual_dims);
    actual_dims[len-1] = ']';
    PyErr_Format(PyExc_TypeError, 
		 "Array must be have shape of %s.  Given array has shape of %s",
		 desired_dims, actual_dims);
  }
  return success;
}
/* End John Hunter translation (with modifications by Bill Spotz) */

%}

/* TYPEMAP_IN macros
 *
 * This family of typemaps allows pure input C arguments of the form
 *
 *     (type* IN_ARRAY1, int DIM1)
 *     (type* IN_ARRAY2, int DIM1, int DIM2)
 *
 * where "type" is any type supported by the Numeric module, to be
 * called in python with an argument list of a single array (or any
 * python object that can be passed to the Numeric.array constructor
 * to produce an arrayof te specified shape).  This can be applied to
 * a existing functions using the %apply directive:
 *
 *     %apply (double* IN_ARRAY1, int DIM1) {double* series, int length}
 *     %apply (double* IN_ARRAY2, int DIM1, int DIM2) {double* mx, int rows, int cols}
 *     double sum(double* series, int length);
 *     double max(double* mx, int rows, int cols);
 *
 * or with
 *
 *     double sum(double* IN_ARRAY1, int DIM1);
 *     double max(double* IN_ARRAY2, int DIM1, int DIM2);
 */

/* One dimensional input arrays */
%define TYPEMAP_IN1(type,typecode)
%typemap(in) (type* IN_ARRAY1, int DIM1)
             (PyArrayObject* array=NULL, int is_new_object) {
  int size[1] = {-1};
  array = obj_to_array_contiguous_allow_conversion($input, typecode, &is_new_object);
  if (!array || !require_dimensions(array,1) || !require_size(array,size,1)) SWIG_fail;
  $1 = (type*) array->data;
  $2 = array->dimensions[0];
}
%typemap(freearg) (type* IN_ARRAY1, int DIM1) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}
%enddef

/* Define concrete examples of the TYPEMAP_IN1 macros */
TYPEMAP_IN1(char,          PyArray_CHAR  )
TYPEMAP_IN1(unsigned char, PyArray_UBYTE )
TYPEMAP_IN1(signed char,   PyArray_SBYTE )
TYPEMAP_IN1(short,         PyArray_SHORT )
TYPEMAP_IN1(int,           PyArray_INT   )
TYPEMAP_IN1(long,          PyArray_LONG  )
TYPEMAP_IN1(float,         PyArray_FLOAT )
TYPEMAP_IN1(double,        PyArray_DOUBLE)
TYPEMAP_IN1(PyObject,      PyArray_OBJECT)

#undef TYPEMAP_IN1

 /* Two dimensional input arrays */
%define TYPEMAP_IN2(type,typecode)
  %typemap(in) (type* IN_ARRAY2, int DIM1, int DIM2)
               (PyArrayObject* array=NULL, int is_new_object) {
  int size[2] = {-1,-1};
  array = obj_to_array_contiguous_allow_conversion($input, typecode, &is_new_object);
  if (!array || !require_dimensions(array,2) || !require_size(array,size,1)) SWIG_fail;
  $1 = (type*) array->data;
  $2 = array->dimensions[0];
  $3 = array->dimensions[1];
}
%typemap(freearg) (type* IN_ARRAY2, int DIM1, int DIM2) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}
%enddef

/* Define concrete examples of the TYPEMAP_IN2 macros */
TYPEMAP_IN2(char,          PyArray_CHAR  )
TYPEMAP_IN2(unsigned char, PyArray_UBYTE )
TYPEMAP_IN2(signed char,   PyArray_SBYTE )
TYPEMAP_IN2(short,         PyArray_SHORT )
TYPEMAP_IN2(int,           PyArray_INT   )
TYPEMAP_IN2(long,          PyArray_LONG  )
TYPEMAP_IN2(float,         PyArray_FLOAT )
TYPEMAP_IN2(double,        PyArray_DOUBLE)
TYPEMAP_IN2(PyObject,      PyArray_OBJECT)

#undef TYPEMAP_IN2

/* TYPEMAP_INPLACE macros
 *
 * This family of typemaps allows input/output C arguments of the form
 *
 *     (type* INPLACE_ARRAY1, int DIM1)
 *     (type* INPLACE_ARRAY2, int DIM1, int DIM2)
 *
 * where "type" is any type supported by the Numeric module, to be
 * called in python with an argument list of a single contiguous
 * Numeric array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (double* INPLACE_ARRAY1, int DIM1) {double* series, int length}
 *     %apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {double* mx, int rows, int cols}
 *     void negate(double* series, int length);
 *     void normalize(double* mx, int rows, int cols);
 *     
 *
 * or with
 *
 *     void sum(double* INPLACE_ARRAY1, int DIM1);
 *     void sum(double* INPLACE_ARRAY2, int DIM1, int DIM2);
 */

 /* One dimensional input/output arrays */
%define TYPEMAP_INPLACE1(type,typecode)
%typemap(in) (type* INPLACE_ARRAY1, int DIM1) (PyArrayObject* temp=NULL) {
  int i;
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp  || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) temp->data;
  $2 = 1;
  for (i=0; i<temp->nd; ++i) $2 *= temp->dimensions[i];
}
%enddef

/* Define concrete examples of the TYPEMAP_INPLACE1 macro */
TYPEMAP_INPLACE1(char,          PyArray_CHAR  )
TYPEMAP_INPLACE1(unsigned char, PyArray_UBYTE )
TYPEMAP_INPLACE1(signed char,   PyArray_SBYTE )
TYPEMAP_INPLACE1(short,         PyArray_SHORT )
TYPEMAP_INPLACE1(int,           PyArray_INT   )
TYPEMAP_INPLACE1(long,          PyArray_LONG  )
TYPEMAP_INPLACE1(float,         PyArray_FLOAT )
TYPEMAP_INPLACE1(double,        PyArray_DOUBLE)
TYPEMAP_INPLACE1(PyObject,      PyArray_OBJECT)

#undef TYPEMAP_INPLACE1

 /* Two dimensional input/output arrays */
%define TYPEMAP_INPLACE2(type,typecode)
  %typemap(in) (type* INPLACE_ARRAY2, int DIM1, int DIM2) (PyArrayObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) temp->data;
  $2 = temp->dimensions[0];
  $3 = temp->dimensions[1];
}
%enddef

/* Define concrete examples of the TYPEMAP_INPLACE2 macro */
TYPEMAP_INPLACE2(char,          PyArray_CHAR  )
TYPEMAP_INPLACE2(unsigned char, PyArray_UBYTE )
TYPEMAP_INPLACE2(signed char,   PyArray_SBYTE )
TYPEMAP_INPLACE2(short,         PyArray_SHORT )
TYPEMAP_INPLACE2(int,           PyArray_INT   )
TYPEMAP_INPLACE2(long,          PyArray_LONG  )
TYPEMAP_INPLACE2(float,         PyArray_FLOAT )
TYPEMAP_INPLACE2(double,        PyArray_DOUBLE)
TYPEMAP_INPLACE2(PyObject,      PyArray_OBJECT)

#undef TYPEMAP_INPLACE2

/* TYPEMAP_ARGOUT macros
 *
 * This family of typemaps allows output C arguments of the form
 *
 *     (type* ARGOUT_ARRAY[ANY])
 *     (type* ARGOUT_ARRAY[ANY][ANY])
 *
 * where "type" is any type supported by the Numeric module, to be
 * called in python with an argument list of a single contiguous
 * Numeric array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (double* ARGOUT_ARRAY[ANY] {double series, int length}
 *     %apply (double* ARGOUT_ARRAY[ANY][ANY]) {double* mx, int rows, int cols}
 *     void negate(double* series, int length);
 *     void normalize(double* mx, int rows, int cols);
 *     
 *
 * or with
 *
 *     void sum(double* ARGOUT_ARRAY[ANY]);
 *     void sum(double* ARGOUT_ARRAY[ANY][ANY]);
 */

 /* One dimensional input/output arrays */
%define TYPEMAP_ARGOUT1(type,typecode)
%typemap(in,numinputs=0) type ARGOUT_ARRAY[ANY] {
  $1 = (type*) malloc($1_dim0*sizeof(type));
  if (!$1) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
    SWIG_fail;
  }
}
%typemap(argout) ARGOUT_ARRAY[ANY] {
  int dimensions[1] = {$1_dim0};
  PyObject* outArray = PyArray_FromDimsAndData(1, dimensions, typecode, (char*)$1);
}
%enddef

/* Define concrete examples of the TYPEMAP_ARGOUT1 macro */
TYPEMAP_ARGOUT1(char,          PyArray_CHAR  )
TYPEMAP_ARGOUT1(unsigned char, PyArray_UBYTE )
TYPEMAP_ARGOUT1(signed char,   PyArray_SBYTE )
TYPEMAP_ARGOUT1(short,         PyArray_SHORT )
TYPEMAP_ARGOUT1(int,           PyArray_INT   )
TYPEMAP_ARGOUT1(long,          PyArray_LONG  )
TYPEMAP_ARGOUT1(float,         PyArray_FLOAT )
TYPEMAP_ARGOUT1(double,        PyArray_DOUBLE)
TYPEMAP_ARGOUT1(PyObject,      PyArray_OBJECT)

#undef TYPEMAP_ARGOUT1

 /* Two dimensional input/output arrays */
%define TYPEMAP_ARGOUT2(type,typecode)
  %typemap(in) (type* ARGOUT_ARRAY2, int DIM1, int DIM2) (PyArrayObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) temp->data;
  $2 = temp->dimensions[0];
  $3 = temp->dimensions[1];
}
%enddef

/* Define concrete examples of the TYPEMAP_ARGOUT2 macro */
TYPEMAP_ARGOUT2(char,          PyArray_CHAR  )
TYPEMAP_ARGOUT2(unsigned char, PyArray_UBYTE )
TYPEMAP_ARGOUT2(signed char,   PyArray_SBYTE )
TYPEMAP_ARGOUT2(short,         PyArray_SHORT )
TYPEMAP_ARGOUT2(int,           PyArray_INT   )
TYPEMAP_ARGOUT2(long,          PyArray_LONG  )
TYPEMAP_ARGOUT2(float,         PyArray_FLOAT )
TYPEMAP_ARGOUT2(double,        PyArray_DOUBLE)
TYPEMAP_ARGOUT2(PyObject,      PyArray_OBJECT)

#undef TYPEMAP_ARGOUT2
