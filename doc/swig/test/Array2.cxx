#include "Array2.h"
#include <sstream>

// Default constructor
Array2::Array2() :
  _ownData(false), _nrows(0), _ncols(), _buffer(0), _rows(0)
{ }

// Size/array constructor
Array2::Array2(int nrows, int ncols, long* data) :
  _ownData(false), _nrows(0), _ncols(), _buffer(0), _rows(0)
{
  resize(nrows, ncols, data);
}

// Copy constructor
Array2::Array2(const Array2 & source) :
  _nrows(source._nrows), _ncols(source._ncols)
{
  _ownData = true;
  allocateMemory();
  *this = source;
}

// Destructor
Array2::~Array2()
{
  deallocateMemory();
}

// Assignment operator
Array2 & Array2::operator=(const Array2 & source)
{
  int nrows = _nrows < source._nrows ? _nrows : source._nrows;
  int ncols = _ncols < source._ncols ? _ncols : source._ncols;
  for (int i=0; i < nrows; ++i)
  {
    for (int j=0; j < ncols; ++j)
    {
      (*this)[i][j] = source[i][j];
    }
  }
  return *this;
}

// Equals operator
bool Array2::operator==(const Array2 & other) const
{
  if (_nrows != other._nrows) return false;
  if (_ncols != other._ncols) return false;
  for (int i=0; i < _nrows; ++i)
  {
    for (int j=0; j < _ncols; ++j)
    {
      if ((*this)[i][j] != other[i][j]) return false;
    }
  }
  return true;
}

// Length accessors
int Array2::nrows() const
{
  return _nrows;
}

int Array2::ncols() const
{
  return _ncols;
}

// Resize array
void Array2::resize(int nrows, int ncols, long* data)
{
  if (nrows < 0) throw std::invalid_argument("Array2 nrows less than 0");
  if (ncols < 0) throw std::invalid_argument("Array2 ncols less than 0");
  if (nrows == _nrows && ncols == _ncols) return;
  deallocateMemory();
  _nrows = nrows;
  _ncols = ncols;
  if (!data)
  {
    allocateMemory();
  }
  else
  {
    _ownData = false;
    _buffer  = data;
    allocateRows();
  }
}

// Set item accessor
Array1 & Array2::operator[](int i)
{
  if (i < 0 || i > _nrows) throw std::out_of_range("Array2 row index out of range");
  return _rows[i];
}

// Get item accessor
const Array1 & Array2::operator[](int i) const
{
  if (i < 0 || i > _nrows) throw std::out_of_range("Array2 row index out of range");
  return _rows[i];
}

// String output
std::string Array2::asString() const
{
  std::stringstream result;
  result << "[ ";
  for (int i=0; i < _nrows; ++i)
  {
    if (i > 0) result << "  ";
    result << (*this)[i].asString();
    if (i < _nrows-1) result << "," << std::endl;
  }
  result << " ]" << std::endl;
  return result.str();
}

// Get view
void Array2::view(int* nrows, int* ncols, long** data) const
{
  *nrows = _nrows;
  *ncols = _ncols;
  *data  = _buffer;
}

// Private methods
void Array2::allocateMemory()
{
  if (_nrows * _ncols == 0)
  {
    _ownData = false;
    _buffer  = 0;
    _rows    = 0;
  }
  else
  {
    _ownData = true;
    _buffer = new long[_nrows*_ncols];
    allocateRows();
  }
}

void Array2::allocateRows()
{
  _rows = new Array1[_nrows];
  for (int i=0; i < _nrows; ++i)
  {
    _rows[i].resize(_ncols, &_buffer[i*_ncols]);
  }
}

void Array2::deallocateMemory()
{
  if (_ownData && _nrows*_ncols && _buffer)
  {
    delete [] _rows;
    delete [] _buffer;
  }
  _ownData = false;
  _nrows   = 0;
  _ncols   = 0;
  _buffer  = 0;
  _rows    = 0;
}
