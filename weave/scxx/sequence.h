/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
  
  modified heavily for weave by eric jones
*********************************************/
#if !defined(SEQUENCE_H_INCLUDED_)
#define SEQUENCE_H_INCLUDED_

#include <string>
#include "object.h"

namespace py {
    
// This isn't being picked up out of object.h for some reason
void Fail(PyObject*, const char* msg);

// pre-declared.  needed by other include files
class str;
class tuple;
class list;

class sequence : public object
{
public:
  sequence() : object() {};
  sequence(const sequence& other) : object(other) {};
  sequence(PyObject* obj) : object(obj) {
    _violentTypeCheck();
  };
  virtual ~sequence() {}

  virtual sequence& operator=(const sequence& other) {
    GrabRef(other);
    return *this;
  };
  /*virtual*/ sequence& operator=(const object& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PySequence_Check(_obj)) {
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a sequence");
    }
  };
  //PySequence_Concat
  sequence operator+(const sequence& rhs) const {
    PyObject*  rslt = PySequence_Concat(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for +");
    return LoseRef(rslt);
  };

  //PySequence_Count
  int count(const object& value) const {
    int rslt = PySequence_Count(_obj, value);
    if (rslt == -1)
      Fail(PyExc_RuntimeError, "failure in count");
    return rslt;
  };

  int count(int value) const;
  int count(double value) const;  
  int count(char* value) const;
  int count(std::string value) const;
  
  //PySequence_GetItem  
  // ## lists - return list_member (mutable) 
  // ## tuples - return tuple_member (mutable)
  // ## otherwise just a object
  object operator [] (int i) const { //can't be virtual
    PyObject* o = PySequence_GetItem(_obj, i);
    if (o == 0)
      Fail(PyExc_IndexError, "index out of range");
    return LoseRef(o);
  };
  //PySequence_GetSlice
  //virtual sequence& operator [] (PWSlice& x) {...};
  sequence slice(int lo, int hi) const {
    PyObject* o = PySequence_GetSlice(_obj, lo, hi);
    if (o == 0)
      Fail(PyExc_IndexError, "could not obtain slice");
    return LoseRef(o);
  };
  
  //PySequence_In
  bool in(const object& value) const {
    int rslt = PySequence_In(_obj, value);
    if (rslt==-1)
      Fail(PyExc_RuntimeError, "problem in in");
    return (rslt==1);
  };
  
  bool in(int value);    
  bool in(double value);
  bool in(char* value);
  bool in(std::string value);
  
  //PySequence_Index
  int index(const object& value) const {
    int rslt = PySequence_Index(_obj, value);
    if (rslt==-1)
      Fail(PyExc_IndexError, "value not found");
    return rslt;
  };
  int index(int value) const;
  int index(double value) const;
  int index(char* value) const;
  int index(std::string value) const;    
  
  //PySequence_Length
  int len() const {
    return PySequence_Length(_obj);
  };
  // added length for compatibility with std::string.
  int length() const {
    return PySequence_Length(_obj);
  };
  //PySequence_Repeat
  sequence operator * (int count) const {
    PyObject* rslt = PySequence_Repeat(_obj, count);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "sequence repeat failed");
    return LoseRef(rslt);
  };
  //PySequence_Tuple
};

} // namespace py

#endif // PWOSEQUENCE_H_INCLUDED_
