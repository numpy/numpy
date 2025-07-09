// -*- c++ -*-

%module Farray

%{
#define SWIG_FILE_WITH_INIT
#include "Farray.h"
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
%ignore *::operator();

// Apply the 2D NumPy typemaps
%apply (int* DIM1 , int* DIM2 , long** ARGOUTVIEW_FARRAY2)
      {(int* nrows, int* ncols, long** data              )};

// Farray support
%include "Farray.h"
%extend Farray
{
  PyObject * __setitem__(PyObject* index, long v)
  {
    int i, j;
    if (!PyArg_ParseTuple(index, "ii:Farray___setitem__",&i,&j)) return NULL;
    self->operator()(i,j) = v;
    return Py_BuildValue("");
  }

  PyObject * __getitem__(PyObject * index)
  {
    int i, j;
    if (!PyArg_ParseTuple(index, "ii:Farray___getitem__",&i,&j)) return NULL;
    return SWIG_From_long(self->operator()(i,j));
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
