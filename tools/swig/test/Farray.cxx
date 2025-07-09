#include "Farray.h"
#include <sstream>

// Size constructor
Farray::Farray(int nrows, int ncols) :
  _nrows(nrows), _ncols(ncols), _buffer(0)
{
  allocateMemory();
}

// Copy constructor
Farray::Farray(const Farray & source) :
  _nrows(source._nrows), _ncols(source._ncols)
{
  allocateMemory();
  *this = source;
}

// Destructor
Farray::~Farray()
{
  delete [] _buffer;
}

// Assignment operator
Farray & Farray::operator=(const Farray & source)
{
  int nrows = _nrows < source._nrows ? _nrows : source._nrows;
  int ncols = _ncols < source._ncols ? _ncols : source._ncols;
  for (int i=0; i < nrows; ++i)
  {
    for (int j=0; j < ncols; ++j)
    {
      (*this)(i,j) = source(i,j);
    }
  }
  return *this;
}

// Equals operator
bool Farray::operator==(const Farray & other) const
{
  if (_nrows != other._nrows) return false;
  if (_ncols != other._ncols) return false;
  for (int i=0; i < _nrows; ++i)
  {
    for (int j=0; j < _ncols; ++j)
    {
      if ((*this)(i,j) != other(i,j)) return false;
    }
  }
  return true;
}

// Length accessors
int Farray::nrows() const
{
  return _nrows;
}

int Farray::ncols() const
{
  return _ncols;
}

// Set item accessor
long & Farray::operator()(int i, int j)
{
  if (i < 0 || i > _nrows) throw std::out_of_range("Farray row index out of range");
  if (j < 0 || j > _ncols) throw std::out_of_range("Farray col index out of range");
  return _buffer[offset(i,j)];
}

// Get item accessor
const long & Farray::operator()(int i, int j) const
{
  if (i < 0 || i > _nrows) throw std::out_of_range("Farray row index out of range");
  if (j < 0 || j > _ncols) throw std::out_of_range("Farray col index out of range");
  return _buffer[offset(i,j)];
}

// String output
std::string Farray::asString() const
{
  std::stringstream result;
  result << "[ ";
  for (int i=0; i < _nrows; ++i)
  {
    if (i > 0) result << "  ";
    result << "[";
    for (int j=0; j < _ncols; ++j)
    {
      result << " " << (*this)(i,j);
      if (j < _ncols-1) result << ",";
    }
    result << " ]";
    if (i < _nrows-1) result << "," << std::endl;
  }
  result << " ]" << std::endl;
  return result.str();
}

// Get view
void Farray::view(int* nrows, int* ncols, long** data) const
{
  *nrows = _nrows;
  *ncols = _ncols;
  *data  = _buffer;
}

// Private methods
void Farray::allocateMemory()
{
  if (_nrows <= 0) throw std::invalid_argument("Farray nrows <= 0");
  if (_ncols <= 0) throw std::invalid_argument("Farray ncols <= 0");
  _buffer = new long[_nrows*_ncols];
}

inline int Farray::offset(int i, int j) const
{
  return i + j * _nrows;
}
