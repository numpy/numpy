#include "ArrayZ.h"
#include <iostream>
#include <sstream>

// Default/length/array constructor
ArrayZ::ArrayZ(int length, std::complex<double>* data) :
  _ownData(false), _length(0), _buffer(0)
{
  resize(length, data);
}

// Copy constructor
ArrayZ::ArrayZ(const ArrayZ & source) :
  _length(source._length)
{
  allocateMemory();
  *this = source;
}

// Destructor
ArrayZ::~ArrayZ()
{
  deallocateMemory();
}

// Assignment operator
ArrayZ & ArrayZ::operator=(const ArrayZ & source)
{
  int len = _length < source._length ? _length : source._length;
  for (int i=0;  i < len; ++i)
  {
    (*this)[i] = source[i];
  }
  return *this;
}

// Equals operator
bool ArrayZ::operator==(const ArrayZ & other) const
{
  if (_length != other._length) return false;
  for (int i=0; i < _length; ++i)
  {
    if ((*this)[i] != other[i]) return false;
  }
  return true;
}

// Length accessor
int ArrayZ::length() const
{
  return _length;
}

// Resize array
void ArrayZ::resize(int length, std::complex<double>* data)
{
  if (length < 0) throw std::invalid_argument("ArrayZ length less than 0");
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
std::complex<double> & ArrayZ::operator[](int i)
{
  if (i < 0 || i >= _length) throw std::out_of_range("ArrayZ index out of range");
  return _buffer[i];
}

// Get item accessor
const std::complex<double> & ArrayZ::operator[](int i) const
{
  if (i < 0 || i >= _length) throw std::out_of_range("ArrayZ index out of range");
  return _buffer[i];
}

// String output
std::string ArrayZ::asString() const
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
void ArrayZ::view(std::complex<double>** data, int* length) const
{
  *data   = _buffer;
  *length = _length;
}

// Private methods
 void ArrayZ::allocateMemory()
 {
   if (_length == 0)
   {
     _ownData = false;
     _buffer  = 0;
   }
   else
   {
     _ownData = true;
     _buffer = new std::complex<double>[_length];
   }
 }

 void ArrayZ::deallocateMemory()
 {
   if (_ownData && _length && _buffer)
   {
     delete [] _buffer;
   }
   _ownData = false;
   _length  = 0;
   _buffer  = 0;
 }
