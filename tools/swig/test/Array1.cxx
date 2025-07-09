#include "Array1.h"
#include <iostream>
#include <sstream>

// Default/length/array constructor
Array1::Array1(int length, long* data) :
  _ownData(false), _length(0), _buffer(0)
{
  resize(length, data);
}

// Copy constructor
Array1::Array1(const Array1 & source) :
  _length(source._length)
{
  allocateMemory();
  *this = source;
}

// Destructor
Array1::~Array1()
{
  deallocateMemory();
}

// Assignment operator
Array1 & Array1::operator=(const Array1 & source)
{
  int len = _length < source._length ? _length : source._length;
  for (int i=0;  i < len; ++i)
  {
    (*this)[i] = source[i];
  }
  return *this;
}

// Equals operator
bool Array1::operator==(const Array1 & other) const
{
  if (_length != other._length) return false;
  for (int i=0; i < _length; ++i)
  {
    if ((*this)[i] != other[i]) return false;
  }
  return true;
}

// Length accessor
int Array1::length() const
{
  return _length;
}

// Resize array
void Array1::resize(int length, long* data)
{
  if (length < 0) throw std::invalid_argument("Array1 length less than 0");
  if (length == _length) return;
  deallocateMemory();
  _length = length;
  if (!data)
  {
    allocateMemory();
  }
  else
  {
    _ownData = false;
    _buffer  = data;
  }
}

// Set item accessor
long & Array1::operator[](int i)
{
  if (i < 0 || i >= _length) throw std::out_of_range("Array1 index out of range");
  return _buffer[i];
}

// Get item accessor
const long & Array1::operator[](int i) const
{
  if (i < 0 || i >= _length) throw std::out_of_range("Array1 index out of range");
  return _buffer[i];
}

// String output
std::string Array1::asString() const
{
  std::stringstream result;
  result << "[";
  for (int i=0; i < _length; ++i)
  {
    result << " " << _buffer[i];
    if (i < _length-1) result << ",";
  }
  result << " ]";
  return result.str();
}

// Get view
void Array1::view(long** data, int* length) const
{
  *data   = _buffer;
  *length = _length;
}

// Private methods
 void Array1::allocateMemory()
 {
   if (_length == 0)
   {
     _ownData = false;
     _buffer  = 0;
   }
   else
   {
     _ownData = true;
     _buffer = new long[_length];
   }
 }

 void Array1::deallocateMemory()
 {
   if (_ownData && _length && _buffer)
   {
     delete [] _buffer;
   }
   _ownData = false;
   _length  = 0;
   _buffer  = 0;
 }
