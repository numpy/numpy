#ifndef ARRAY2_H
#define ARRAY2_H

#include "Array1.h"
#include <stdexcept>
#include <string>

class Array2
{
public:

  // Default constructor
  Array2();

  // Size/array constructor
  Array2(int nrows, int ncols, long* data=0);

  // Copy constructor
  Array2(const Array2 & source);

  // Destructor
  ~Array2();

  // Assignment operator
  Array2 & operator=(const Array2 & source);

  // Equals operator
  bool operator==(const Array2 & other) const;

  // Length accessors
  int nrows() const;
  int ncols() const;

  // Resize array  
  void resize(int nrows, int ncols, long* data);
  void resize(int nrows, int ncols);
  
  // Set item accessor
  Array1 & operator[](int i);

  // Get item accessor
  const Array1 & operator[](int i) const;

  // String output
  std::string asString() const;

  // Get view
  void view(int* nrows, int* ncols, long** data) const;

private:
  // Members
  bool _ownData;
  int _nrows;
  int _ncols;
  long * _buffer;
  Array1 * _rows;

  // Methods
  void allocateMemory();
  void allocateRows();
  void deallocateMemory();
};

#endif
