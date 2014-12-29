#ifndef ARRAYZ_H
#define ARRAYZ_H

#include <stdexcept>
#include <string>
#include <complex>

class ArrayZ
{
public:

  // Default/length/array constructor
  ArrayZ(int length = 0, std::complex<double>* data = 0);

  // Copy constructor
  ArrayZ(const ArrayZ & source);

  // Destructor
  ~ArrayZ();

  // Assignment operator
  ArrayZ & operator=(const ArrayZ & source);

  // Equals operator
  bool operator==(const ArrayZ & other) const;

  // Length accessor
  int length() const;

  // Resize array
  void resize(int length, std::complex<double>* data = 0);

  // Set item accessor
  std::complex<double> & operator[](int i);

  // Get item accessor
  const std::complex<double> & operator[](int i) const;

  // String output
  std::string asString() const;

  // Get view
  void view(std::complex<double>** data, int* length) const;

private:
  // Members
  bool _ownData;
  int _length;
  std::complex<double> * _buffer;

  // Methods
  void allocateMemory();
  void deallocateMemory();
};

#endif
