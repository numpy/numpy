/* -*- C -*-  (not really, but good for syntax highlighting) */
%{
#ifndef SWIG_FILE_WITH_INIT
#  define NO_IMPORT_ARRAY
#endif
#include "stdio.h"
#include <numpy/arrayobject.h>

/* The following code originally appeared in
 * enthought/kiva/agg/src/numeric.i, author unknown.  It was
 * translated from C++ to C by John Hunter.  Bill Spotz has modified
 * it slightly to fix some minor bugs, upgrade to numpy (all
 * versions), add some comments and some functionality.
 */

/* Macros to extract array attributes.
 */
#define is_array(a)            ((a) && PyArray_Check((PyArrayObject *)a))
#define array_type(a)          (int)(PyArray_TYPE(a))
#define array_dimensions(a)    (((PyArrayObject *)a)->nd)
#define array_size(a,i)        (((PyArrayObject *)a)->dimensions[i])
#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS(a))

/* Support older NumPy data type names
*/
#if NDARRAY_VERSION < 0x01000000
#define NPY_BOOL        PyArray_BOOL
#define NPY_BYTE        PyArray_BYTE
#define NPY_UBYTE       PyArray_UBYTE
#define NPY_SHORT       PyArray_SHORT
#define NPY_USHORT      PyArray_USHORT
#define NPY_INT         PyArray_INT
#define NPY_UINT        PyArray_UINT
#define NPY_LONG        PyArray_LONG
#define NPY_ULONG       PyArray_ULONG
#define NPY_LONGLONG    PyArray_LONGLONG
#define NPY_ULONGLONG   PyArray_ULONGLONG
#define NPY_FLOAT       PyArray_FLOAT
#define NPY_DOUBLE      PyArray_DOUBLE
#define NPY_LONGDOUBLE  PyArray_LONGDOUBLE
#define NPY_CFLOAT      PyArray_CFLOAT
#define NPY_CDOUBLE     PyArray_CDOUBLE
#define NPY_CLONGDOUBLE PyArray_CLONGDOUBLE
#define NPY_OBJECT      PyArray_OBJECT
#define NPY_STRING      PyArray_STRING
#define NPY_UNICODE     PyArray_UNICODE
#define NPY_VOID        PyArray_VOID
#define NPY_NTYPES      PyArray_NTYPES
#define NPY_NOTYPE      PyArray_NOTYPE
#define NPY_CHAR        PyArray_CHAR
#define NPY_USERDEF     PyArray_USERDEF
#define npy_intp        intp
#endif

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

/* Given a NumPy typecode, return a string describing the type.
 */
char* typecode_string(int typecode) {
  static char* type_names[24] = {"bool", "byte", "unsigned byte",
				 "short", "unsigned short", "int",
				 "unsigned int", "long", "unsigned long",
				 "long long", "unsigned long long",
				 "float", "double", "long double",
				 "complex float", "complex double",
				 "complex long double", "object",
				 "string", "unicode", "void", "ntypes",
				 "notype", "char"};
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
 * return NULL.
 */
PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode) {
  PyArrayObject* ary = NULL;
  if (is_array(input) && (typecode == NPY_NOTYPE ||
			  type_match(array_type(input), typecode))) {
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

/* Convert the given PyObject to a NumPy array with the given
 * typecode.  On Success, return a valid PyArrayObject* with the
 * correct type.  On failure, the python error string will be set and
 * the routine returns NULL.
 */
PyArrayObject* obj_to_array_allow_conversion(PyObject* input, int typecode,
                                             int* is_new_object) {
  PyArrayObject* ary = NULL;
  PyObject* py_obj;
  if (is_array(input) && (typecode == NPY_NOTYPE || type_match(array_type(input),typecode))) {
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
                               int min_dims, int max_dims) {
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
int require_size(PyArrayObject* ary, npy_intp* size, int n) {
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
/* End John Hunter translation (with modifications by Bill Spotz)
 */

%}

/* %numpy_typemaps() macro
 *
 * This macro defines a family of typemaps that allow pure input C
 * arguments of the form
 *
 *     (TYPE* IN_ARRAY1, int DIM1)
 *     (TYPE* IN_ARRAY2, int DIM1, int DIM2)
 *     (TYPE* INPLACE_ARRAY1, int DIM1)
 *     (TYPE* INPLACE_ARRAY2, int DIM1, int DIM2)
 *     (TYPE* ARGOUT_ARRAY1[ANY])
 *     (TYPE* ARGOUT_ARRAY2[ANY][ANY])
 *
 * where "TYPE" is any type supported by the NumPy module.  In python,
 * the dimensions will not need to be specified.  The IN_ARRAYs can be
 * a numpy array or any sequence that can be converted to a numpy
 * array of the specified type.  The INPLACE_ARRAYs must be numpy
 * arrays of the appropriate type.  The ARGOUT_ARRAYs will be returned
 * as numpy arrays of the appropriate type.

 * These typemaps can be applied to existing functions using the
 * %apply directive:
 *
 *     %apply (double* IN_ARRAY1, int DIM1) {double* series, int length};
 *     double sum(double* series, int length);
 *
 *     %apply (double* IN_ARRAY2, int DIM1, int DIM2) {double* mx, int rows, int cols};
 *     double max(double* mx, int rows, int cols);
 *
 *     %apply (double* INPLACE_ARRAY1, int DIM1) {double* series, int length};
 *     void negate(double* series, int length);
 *
 *     %apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {double* mx, int rows, int cols};
 *     void normalize(double* mx, int rows, int cols);
 *
 *     %apply (double* ARGOUT_ARRAY1[ANY] {double series, int length};
 *     void negate(double* series, int length);
 *
 *     %apply (double* ARGOUT_ARRAY2[ANY][ANY]) {double* mx, int rows, int cols};
 *     void normalize(double* mx, int rows, int cols);
 *
 * or directly with
 *
 *     double sum(double* IN_ARRAY1, int DIM1);
 *     double max(double* IN_ARRAY2, int DIM1, int DIM2);
 *     void sum(double* INPLACE_ARRAY1, int DIM1);
 *     void sum(double* INPLACE_ARRAY2, int DIM1, int DIM2);
 *     void sum(double* ARGOUT_ARRAY1[ANY]);
 *     void sum(double* ARGOUT_ARRAY2[ANY][ANY]);
 */

%define %numpy_typemaps(TYPE, TYPECODE)

/* Typemap suite for (TYPE* IN_ARRAY1, int DIM1)
 */
%typemap(in) (TYPE* IN_ARRAY1, int DIM1)
             (PyArrayObject* array=NULL, int is_new_object=0) {
  array = obj_to_array_contiguous_allow_conversion($input, TYPECODE, &is_new_object);
  npy_intp size[1] = {-1};
  if (!array || !require_dimensions(array, 1) || !require_size(array, size, 1)) SWIG_fail;
  $1 = (TYPE*) array->data;
  $2 = (int) array->dimensions[0];
}
%typemap(freearg) (TYPE* IN_ARRAY1, int DIM1) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}

/* Typemap suite for (TYPE* IN_ARRAY2, int DIM1, int DIM2)
 */
%typemap(in) (TYPE* IN_ARRAY2, int DIM1, int DIM2)
             (PyArrayObject* array=NULL, int is_new_object=0) {
  array = obj_to_array_contiguous_allow_conversion($input, TYPECODE, &is_new_object);
  npy_intp size[2] = {-1,-1};
  if (!array || !require_dimensions(array, 2) || !require_size(array, size, 1)) SWIG_fail;
  $1 = (TYPE*) array->data;
  $2 = (int) array->dimensions[0];
  $3 = (int) array->dimensions[1];
}
%typemap(freearg) (TYPE* IN_ARRAY2, int DIM1, int DIM2) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}

/* Typemap suite for (TYPE* INPLACE_ARRAY1, int DIM1)
 */
%typemap(in) (TYPE* INPLACE_ARRAY1, int DIM1) (PyArrayObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input, TYPECODE);
  if (!temp  || !require_contiguous(temp)) SWIG_fail;
  $1 = (TYPE*) temp->data;
  $2 = 1;
  for (int i=0; i<temp->nd; ++i) $2 *= temp->dimensions[i];
}

/* Typemap suite for (TYPE* INPLACE_ARRAY2, int DIM1, int DIM2)
 */
%typemap(in) (TYPE* INPLACE_ARRAY2, int DIM1, int DIM2) (PyArrayObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input, TYPECODE);
  if (!temp || !require_contiguous(temp)) SWIG_fail;
  $1 = (TYPE*) temp->data;
  $2 = (int) temp->dimensions[0];
  $3 = (int) temp->dimensions[1];
}

/* Typemap suite for (TYPE ARGOUT_ARRAY1[ANY])
 */
%typemap(in,numinputs=0) (TYPE ARGOUT_ARRAY1[ANY]) {
  $1 = (TYPE*) malloc($1_dim0*sizeof(TYPE));
  if (!$1) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
    SWIG_fail;
  }
}
%typemap(argout) (TYPE ARGOUT_ARRAY1[ANY]) {
  PyObject * obj = NULL;
  npy_intp dimensions[1] = { $1_dim0 };
  PyObject* outArray = PyArray_FromDimsAndData(1, dimensions, TYPECODE, (char*)$1);
  if ($result == Py_None) {
    Py_DECREF($result);
    $result = outArray;
  }
  else {
    if (!PyTuple_Check($result)) $result = Py_BuildValue("(O)", $result);
    obj = Py_Build_Value("(O)", outArray);
    $result = PySequence_Concat($result, obj);
  }
}

/* Typemap suite for (TYPE ARGOUT_ARRAY2[ANY][ANY])
 */
%typemap(in,numinputs=0) (TYPE ARGOUT_ARRAY2[ANY][ANY]) {
  $1 = (TYPE*) malloc($1_dim0 * $1_dim1 * sizeof(TYPE));
  if (!$1) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
    SWIG_fail;
  }
}
%typemap(argout) (TYPE ARGOUT_ARRAY1[ANY][ANY]) {
  PyObject * obj = NULL;
  npy_intp dimensions[2] = { $1_dim0, $1_dim1 };
  PyObject* outArray = PyArray_FromDimsAndData(1, dimensions, TYPECODE, (char*)$1);
  if ($result == Py_None) {
    Py_DECREF($result);
    $result = outArray;
  }
  else {
    if (!PyTuple_Check($result)) $result = Py_BuildValue("(O)", $result);
    obj = Py_Build_Value("(O)", outArray);
    $result = PySequence_Concat($result, obj);
  }
}

%enddef    /* %numpy_typemaps() macro */


/* Concrete instances of the %numpy_typemaps() macro: Each invocation
 * below applies all of the typemaps above to the specified data type.
 */
%numpy_typemaps(signed char,         NPY_BYTE     ) /**/
%numpy_typemaps(unsigned char,       NPY_UBYTE    ) /**/
%numpy_typemaps(short,               NPY_SHORT    ) /**/
%numpy_typemaps(unsigned short,      NPY_USHORT   ) /**/
%numpy_typemaps(int,                 NPY_INT      ) /**/
%numpy_typemaps(unsigned int,        NPY_UINT     ) /**/
%numpy_typemaps(long,                NPY_LONG     ) /**/
%numpy_typemaps(unsigned long,       NPY_ULONG    ) /**/
%numpy_typemaps(long long,           NPY_LONGLONG ) /**/
%numpy_typemaps(unsigned long long,  NPY_ULONGLONG) /**/
%numpy_typemaps(float,               NPY_FLOAT    ) /**/
%numpy_typemaps(double,              NPY_DOUBLE   ) /**/
%numpy_typemaps(PyObject,            NPY_OBJECT   )
%numpy_typemaps(char,                NPY_CHAR     )

/* ***************************************************************
 * The follow macro expansion does not work, because C++ bool is 4
 * bytes and NPY_BOOL is 1 byte
 */
/*%numpy_typemaps(bool, NPY_BOOL)
 */

/* ***************************************************************
 * On my Mac, I get the following warning for this macro expansion:
 * 'swig/python detected a memory leak of type 'long double *', no destructor found.'
 */
/*%numpy_typemaps(long double, NPY_LONGDOUBLE)
 */

/* ***************************************************************
 * Swig complains about a syntax error for the following macros
 * expansions:
 */
/*%numpy_typemaps(complex float,       NPY_CFLOAT     )
 */
/*%numpy_typemaps(complex double,      NPY_CDOUBLE    )
 */
/*%numpy_typemaps(complex long double, NPY_CLONGDOUBLE)
 */
