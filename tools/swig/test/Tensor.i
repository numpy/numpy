// -*- c++ -*-
%module Tensor

%{
#define SWIG_FILE_WITH_INIT
#include "Tensor.h"
%}

// Get the NumPy typemaps
%include "../numpy.i"

%init %{
  import_array();
%}

%define %apply_numpy_typemaps(TYPE)

%apply (TYPE IN_ARRAY3[ANY][ANY][ANY]) {(TYPE tensor[ANY][ANY][ANY])};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* tensor, int slices, int rows, int cols)};
%apply (int DIM1, int DIM2, int DIM3, TYPE* IN_ARRAY3)
      {(int slices, int rows, int cols, TYPE* tensor)};

%apply (TYPE INPLACE_ARRAY3[ANY][ANY][ANY]) {(TYPE array[3][3][3])};
%apply (TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3)
      {(TYPE* array, int slices, int rows, int cols)};
%apply (int DIM1, int DIM2, int DIM3, TYPE* INPLACE_ARRAY3)
      {(int slices, int rows, int cols, TYPE* array)};

%apply (TYPE ARGOUT_ARRAY3[ANY][ANY][ANY]) {(TYPE lower[2][2][2])};
%apply (TYPE ARGOUT_ARRAY3[ANY][ANY][ANY]) {(TYPE upper[2][2][2])};

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
%include "Tensor.h"
