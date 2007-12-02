#ifndef ARRAY1_H
#define ARRAY1_H

#include <stdexcept>
#include <string>

class Array1
{
public:

  // Default/length/array constructor
  Array1(int length = 0, long* data = 0);

  // Copy constructor
  Array1(const Array1 & source);

  // Destructor
  ~Array1();

  // Assignment operator
  Array1 & operator=(const Array1 & source);

  // Equals operator
  bool operator==(const Array1 & other) const;

  // Length accessor
  int length() const;

  // Resize array
  void resize(int length, long* data = 0);

  // Set item accessor
  long & operator[](int i);

  // Get item accessor
  const long & operator[](int i) const;

  // String output
  std::string asString() const;

  // Get view
  void view(long** data, int* length) const;

private:
  // Members
  bool _ownData;
  int _length;
  long * _buffer;

  // Methods
  void allocateMemory();
  void deallocateMemory();
};

#endif
