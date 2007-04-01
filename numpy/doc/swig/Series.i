// -*- c++ -*-
%module Series

%{
#define SWIG_FILE_WITH_INIT
#include "series.h"
%}

// Get the NumPy typemaps
%include "numpy.i"

%init %{
  import_array();
%}

%define %apply_numpy_typemaps(TYPE)

%apply (TYPE IN_ARRAY1[ANY]) {(TYPE vector[3])};
%apply (TYPE* IN_ARRAY1, int DIM1) {(TYPE* series, int size)};
%apply (int DIM1, TYPE* IN_ARRAY1) {(int size, TYPE* series)};

%apply (TYPE INPLACE_ARRAY1[ANY]) {(TYPE array[3])};
%apply (TYPE* INPLACE_ARRAY1, int DIM1) {(TYPE* array, int size)};
%apply (int DIM1, TYPE* INPLACE_ARRAY1) {(int size, TYPE* array)};

%apply (TYPE ARGOUT_ARRAY1[ANY]) {(TYPE even[3])};
%apply (TYPE ARGOUT_ARRAY1[ANY]) {(TYPE odd[ 3])};
%apply (TYPE* ARGOUT_ARRAY1, int DIM1) {(TYPE* twoVec, int size)};
%apply (int DIM1, TYPE* ARGOUT_ARRAY1) {(int size, TYPE* threeVec)};

%apply (TYPE IN_ARRAY2[ANY][ANY]) {(TYPE matrix[ANY][ANY])};
%apply (TYPE* IN_ARRAY2, int DIM1, int DIM2) {(TYPE* matrix, int rows, int cols)};
%apply (int DIM1, int DIM2, TYPE* IN_ARRAY2) {(int rows, int cols, TYPE* matrix)};

%apply (TYPE INPLACE_ARRAY2[ANY][ANY]) {(TYPE array[3][3])};
%apply (TYPE* INPLACE_ARRAY2, int DIM1, int DIM2) {(TYPE* array, int rows, int cols)};
%apply (int DIM1, int DIM2, TYPE* INPLACE_ARRAY2) {(int rows, int cols, TYPE* array)};

%apply (TYPE ARGOUT_ARRAY2[ANY][ANY]) {(TYPE lower[3][3])};
%apply (TYPE ARGOUT_ARRAY2[ANY][ANY]) {(TYPE upper[3][3])};

%enddef    /* %apply_numpy_typemaps() macro */

%apply_numpy_typemaps(signed char       )
%apply_numpy_typemaps(unsigned char     )
%apply_numpy_typemaps(short             )
%apply_numpy_typemaps(unsigned short    )
%apply_numpy_typemaps(int               )
%apply_numpy_typemaps(unsigned int      )
%apply_numpy_typemaps(long              )
%apply_numpy_typemaps(unsigned long     )
%apply_numpy_typemaps(long long         )
%apply_numpy_typemaps(unsigned long long)
%apply_numpy_typemaps(float             )
%apply_numpy_typemaps(double            )

// Include the header file to be wrapped
%include "series.h"
