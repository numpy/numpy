// -*- c++ -*-

%module Array

%{
#define SWIG_FILE_WITH_INIT
#include "Array1.h"
#include "Array2.h"
%}

// Get the NumPy typemaps
%include "../numpy.i"

 // Get the STL typemaps
%include "stl.i"

// Handle standard exceptions
%include "exception.i"
%exception
{
  try
  {
    $action
  }
  catch (const std::invalid_argument& e)
  {
    SWIG_exception(SWIG_ValueError, e.what());
  }
  catch (const std::out_of_range& e)
  {
    SWIG_exception(SWIG_IndexError, e.what());
  }
}
%init %{
  import_array();
%}

// Global ignores
%ignore *::operator=;
%ignore *::operator[];

// Apply the 1D NumPy typemaps
%apply (int DIM1  , long* INPLACE_ARRAY1)
      {(int length, long* data          )};
%apply (long** ARGOUTVIEW_ARRAY1, int* DIM1  )
      {(long** data             , int* length)};

// Apply the 2D NumPy typemaps
%apply (int DIM1 , int DIM2 , long* INPLACE_ARRAY2)
      {(int nrows, int ncols, long* data          )};
%apply (int* DIM1 , int* DIM2 , long** ARGOUTVIEW_ARRAY2)
      {(int* nrows, int* ncols, long** data             )};

// Array1 support
%include "Array1.h"
%extend Array1
{
  void __setitem__(int i, long v)
  {
    self->operator[](i) = v;
  }

  long __getitem__(int i)
  {
    return self->operator[](i);
  }

  int __len__()
  {
    return self->length();
  }

  std::string __str__()
  {
    return self->asString();
  }
}

// Array2 support
%include "Array2.h"
%extend Array2
{
  void __setitem__(int i, Array1 & v)
  {
    self->operator[](i) = v;
  }

  Array1 & __getitem__(int i)
  {
    return self->operator[](i);
  }

  int __len__()
  {
    return self->nrows() * self->ncols();
  }

  std::string __str__()
  {
    return self->asString();
  }
}
