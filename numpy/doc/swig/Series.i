// -*- c++ -*-
%module Series

%{
#define SWIG_FILE_WITH_INIT
#include "series.h"
%}

// Get the Numeric typemaps
%include "numpy.i"

%init %{
  import_array();
%}

// Apply the Numeric typemaps for 1D input arrays
%apply (signed char*        IN_ARRAY1, int DIM1)
      {(signed char*        series,    int size)};
%apply (unsigned char*      IN_ARRAY1, int DIM1)
      {(unsigned char*      series,    int size)};
%apply (short*              IN_ARRAY1, int DIM1)
      {(short*              series,    int size)};
%apply (unsigned short*     IN_ARRAY1, int DIM1)
      {(unsigned short*     series,    int size)};
%apply (int*                IN_ARRAY1, int DIM1)
      {(int*                series,    int size)};
%apply (unsigned int*       IN_ARRAY1, int DIM1)
      {(unsigned int*       series,    int size)};
%apply (long*               IN_ARRAY1, int DIM1)
      {(long*               series,    int size)};
%apply (unsigned long*      IN_ARRAY1, int DIM1)
      {(unsigned long*      series,    int size)};
%apply (long long*          IN_ARRAY1, int DIM1)
      {(long long*          series,    int size)};
%apply (unsigned long long* IN_ARRAY1, int DIM1)
      {(unsigned long long* series,    int size)};
%apply (float*              IN_ARRAY1, int DIM1)
      {(float*              series,    int size)};
%apply (double*             IN_ARRAY1, int DIM1)
      {(double*             series,    int size)};
%apply (long double*        IN_ARRAY1, int DIM1)
      {(long double*        series,    int size)};

// Apply the Numeric typemaps for 1D input/output arrays
%apply (signed char*        INPLACE_ARRAY1, int DIM1)
      {(signed char*        array,          int size)};
%apply (unsigned char*      INPLACE_ARRAY1, int DIM1)
      {(unsigned char*      array,          int size)};
%apply (short*              INPLACE_ARRAY1, int DIM1)
      {(short*              array,          int size)};
%apply (unsigned short*     INPLACE_ARRAY1, int DIM1)
      {(unsigned short*     array,          int size)};
%apply (int*                INPLACE_ARRAY1, int DIM1)
      {(int*                array,          int size)};
%apply (unsigned int*       INPLACE_ARRAY1, int DIM1)
      {(unsigned int*       array,          int size)};
%apply (long*               INPLACE_ARRAY1, int DIM1)
      {(long*               array,          int size)};
%apply (unsigned long*      INPLACE_ARRAY1, int DIM1)
      {(unsigned long*      array,          int size)};
%apply (long long*          INPLACE_ARRAY1, int DIM1)
      {(long long*          array,          int size)};
%apply (unsigned long long* INPLACE_ARRAY1, int DIM1)
      {(unsigned long long* array,          int size)};
%apply (float*              INPLACE_ARRAY1, int DIM1)
      {(float*              array,          int size)};
%apply (double*             INPLACE_ARRAY1, int DIM1)
      {(double*             array,          int size)};
%apply (long double*        INPLACE_ARRAY1, int DIM1)
      {(long double*        array,          int size)};

// Apply the Numeric typemaps for 2D input arrays
%apply (signed char*        IN_ARRAY2, int DIM1, int DIM2)
      {(signed char*        matrix,    int rows, int cols)};
%apply (unsigned char*      IN_ARRAY2, int DIM1, int DIM2)
      {(unsigned char*      matrix,    int rows, int cols)};
%apply (short*              IN_ARRAY2, int DIM1, int DIM2)
      {(short*              matrix,    int rows, int cols)};
%apply (unsigned short*     IN_ARRAY2, int DIM1, int DIM2)
      {(unsigned short*     matrix,    int rows, int cols)};
%apply (int*                IN_ARRAY2, int DIM1, int DIM2)
      {(int*                matrix,    int rows, int cols)};
%apply (unsigned int*       IN_ARRAY2, int DIM1, int DIM2)
      {(unsigned int*       matrix,    int rows, int cols)};
%apply (long*               IN_ARRAY2, int DIM1, int DIM2)
      {(long*               matrix,    int rows, int cols)};
%apply (unsigned long*      IN_ARRAY2, int DIM1, int DIM2)
      {(unsigned long*      matrix,    int rows, int cols)};
%apply (long long*          IN_ARRAY2, int DIM1, int DIM2)
      {(long long*          matrix,    int rows, int cols)};
%apply (unsigned long long* IN_ARRAY2, int DIM1, int DIM2)
      {(unsigned long long* matrix,    int rows, int cols)};
%apply (float*              IN_ARRAY2, int DIM1, int DIM2)
      {(float*              matrix,    int rows, int cols)};
%apply (double*             IN_ARRAY2, int DIM1, int DIM2)
      {(double*             matrix,    int rows, int cols)};
%apply (long double*        IN_ARRAY2, int DIM1, int DIM2)
      {(long double*        matrix,    int rows, int cols)};

// Apply the Numeric typemaps for 2D input/output arrays
%apply (signed char*        INPLACE_ARRAY2, int DIM1, int DIM2)
      {(signed char*        array,          int rows, int cols)};
%apply (unsigned char*      INPLACE_ARRAY2, int DIM1, int DIM2)
      {(unsigned char*      array,          int rows, int cols)};
%apply (short*              INPLACE_ARRAY2, int DIM1, int DIM2)
      {(short*              array,          int rows, int cols)};
%apply (unsigned short*     INPLACE_ARRAY2, int DIM1, int DIM2)
      {(unsigned short*     array,          int rows, int cols)};
%apply (int*                INPLACE_ARRAY2, int DIM1, int DIM2)
      {(int*                array,          int rows, int cols)};
%apply (unsigned int*       INPLACE_ARRAY2, int DIM1, int DIM2)
      {(unsigned int*       array,          int rows, int cols)};
%apply (long*               INPLACE_ARRAY2, int DIM1, int DIM2)
      {(long*               array,          int rows, int cols)};
%apply (unsigned long*      INPLACE_ARRAY2, int DIM1, int DIM2)
      {(unsigned long*      array,          int rows, int cols)};
%apply (long long*          INPLACE_ARRAY2, int DIM1, int DIM2)
      {(long long*          array,          int rows, int cols)};
%apply (unsigned long long* INPLACE_ARRAY2, int DIM1, int DIM2)
      {(unsigned long long* array,          int rows, int cols)};
%apply (float*              INPLACE_ARRAY2, int DIM1, int DIM2)
      {(float*              array,          int rows, int cols)};
%apply (double*             INPLACE_ARRAY2, int DIM1, int DIM2)
      {(double*             array,          int rows, int cols)};
%apply (long double*        INPLACE_ARRAY2, int DIM1, int DIM2)
      {(long double*        array,          int rows, int cols)};

// Include the header file to be wrapped
%include "series.h"
