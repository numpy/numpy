/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com

  modified for weave by eric jones
*********************************************/
#if !defined(NUMBER_H_INCLUDED_)
#define NUMBER_H_INCLUDED_

#include "object.h"
#include "sequence.h"

namespace py {
    
class number : public object
{
public:
  number() : object() {};
  number(int i) : object (PyInt_FromLong(i)) { LoseRef(_obj); }
  number(long i) : object (PyInt_FromLong(i)) { LoseRef(_obj); }
  number(unsigned long i) : object (PyLong_FromUnsignedLong(i)) { LoseRef(_obj); }
  number(double d) : object (PyFloat_FromDouble(d)) { LoseRef(_obj); }

  number(const number& other) : object(other) {};
  number(PyObject* obj) : object(obj) {
    _violentTypeCheck();
  };
  virtual ~number() {};

  virtual number& operator=(const number& other) {
    GrabRef(other);
    return *this;
  };
  /*virtual*/ number& operator=(const object& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyNumber_Check(_obj)) {
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a number");
    }
  };
  //PyNumber_Absolute
  number abs() const {
    PyObject* rslt = PyNumber_Absolute(_obj);
    if (rslt==0)
      Fail(PyExc_TypeError, "Failed to get absolute value");
    return LoseRef(rslt);
  };
  //PyNumber_Add
  number operator+(const number& rhs) const {
    PyObject*  rslt = PyNumber_Add(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for +");
    return LoseRef(rslt);
  };
  //PyNumber_And
  number operator&(const number& rhs) const {
    PyObject*  rslt = PyNumber_And(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for &");
    return LoseRef(rslt);
  };
  //PyNumber_Coerce
  //PyNumber_Divide
  number operator/(const number& rhs) const {
    PyObject*  rslt = PyNumber_Divide(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for /");
    return LoseRef(rslt);
  };
  //PyNumber_Divmod
  sequence divmod(const number& rhs) const {
    PyObject*  rslt = PyNumber_Divmod(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for divmod");
    return LoseRef(rslt);
  };
  //PyNumber_Float
    operator double () const {
  PyObject*  F = PyNumber_Float(_obj);
  if (F==0)
      Fail(PyExc_TypeError, "Cannot convert to double");
  double r = PyFloat_AS_DOUBLE(F);
  Py_DECREF(F);
  return r;
    };
  operator float () const {
    double rslt = (double) *this;
    //if (rslt > INT_MAX)
    //  Fail(PyExc_TypeError, "Cannot convert to a float");
    return (float) rslt;
  };
  //PyNumber_Int
    operator long () const {
  PyObject*  Int = PyNumber_Int(_obj);
  if (Int==0)
      Fail(PyExc_TypeError, "Cannot convert to long");
  long r = PyInt_AS_LONG(Int);
  Py_DECREF(Int);
  return r;
    };
  operator int () const {
    long rslt = (long) *this;
    if (rslt > INT_MAX)
      Fail(PyExc_TypeError, "Cannot convert to an int");
    return (int) rslt;
  };
  //PyNumber_Invert
  number operator~ () const {
    PyObject* rslt = PyNumber_Invert(_obj);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper type for ~");
    return LoseRef(rslt);
  };
  //PyNumber_Long
  //PyNumber_Lshift
  number operator<<(const number& rhs) const {
    PyObject*  rslt = PyNumber_Lshift(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for <<");
    return LoseRef(rslt);
  };
  //PyNumber_Multiply
  number operator*(const number& rhs) const {
    PyObject*  rslt = PyNumber_Multiply(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for *");
    return LoseRef(rslt);
  };
  //PyNumber_Negative
  number operator- () const {
    PyObject* rslt = PyNumber_Negative(_obj);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper type for unary -");
    return LoseRef(rslt);
  };
  //PyNumber_Or
  number operator|(const number& rhs) const {
    PyObject*  rslt = PyNumber_Or(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for |");
    return LoseRef(rslt);
  };
  //PyNumber_Positive
  number operator+ () const {
    PyObject* rslt = PyNumber_Positive(_obj);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper type for unary +");
    return LoseRef(rslt);
  };
  //PyNumber_Remainder
  number operator%(const number& rhs) const {
    PyObject*  rslt = PyNumber_Remainder(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for %");
    return LoseRef(rslt);
  };
  //PyNumber_Rshift
  number operator>>(const number& rhs) const {
    PyObject*  rslt = PyNumber_Rshift(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for >>");
    return LoseRef(rslt);
  };
  //PyNumber_Subtract
  number operator-(const number& rhs) const {
    PyObject*  rslt = PyNumber_Subtract(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for -");
    return LoseRef(rslt);
  };
  //PyNumber_Xor
  number operator^(const number& rhs) const {
    PyObject*  rslt = PyNumber_Xor(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for ^");
    return LoseRef(rslt);
  };
};

} // namespace py

#endif //NUMBER_H_INCLUDED_
