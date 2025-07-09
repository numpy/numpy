#ifndef FARRAY_H
#define FARRAY_H

#include <stdexcept>
#include <string>

class Farray
{
public:

  // Size constructor
  Farray(int nrows, int ncols);

  // Copy constructor
  Farray(const Farray & source);

  // Destructor
  ~Farray();

  // Assignment operator
  Farray & operator=(const Farray & source);

  // Equals operator
  bool operator==(const Farray & other) const;

  // Length accessors
  int nrows() const;
  int ncols() const;

  // Set item accessor
  long & operator()(int i, int j);

  // Get item accessor
  const long & operator()(int i, int j) const;

  // String output
  std::string asString() const;

  // Get view
  void view(int* nrows, int* ncols, long** data) const;

private:
  // Members
  int _nrows;
  int _ncols;
  long * _buffer;

  // Default constructor: not implemented
  Farray();

  // Methods
  void allocateMemory();
  int  offset(int i, int j) const;
};

#endif
