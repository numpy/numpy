/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
*********************************************/
#if !defined(PWOSEQUENCE_H_INCLUDED_)
#define PWOSEQUENCE_H_INCLUDED_

#include "PWOBase.h"

// This isn't being picked up out of PWOBase.h for some reason
void Fail(PyObject*, const char* msg);

class PWOSequence : public PWOBase
{
public:
  PWOSequence() : PWOBase() {};
  PWOSequence(const PWOSequence& other) : PWOBase(other) {};
  PWOSequence(PyObject* obj) : PWOBase(obj) {
    _violentTypeCheck();
  };
  virtual ~PWOSequence() {}

  virtual PWOSequence& operator=(const PWOSequence& other) {
    GrabRef(other);
    return *this;
  };
  /*virtual*/ PWOSequence& operator=(const PWOBase& other) {
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
  PWOSequence operator+(const PWOSequence& rhs) const {
    PyObject*  rslt = PySequence_Concat(_obj, rhs);
    if (rslt==0)
      Fail(PyExc_TypeError, "Improper rhs for +");
    return LoseRef(rslt);
  };
  //PySequence_Count
  int count(const PWOBase& value) const {
    int rslt = PySequence_Count(_obj, value);
    if (rslt == -1)
      Fail(PyExc_RuntimeError, "failure in count");
    return rslt;
  };
  //PySequence_GetItem  ##lists - return PWOListMmbr (mutable) otherwise just a PWOBase
  PWOBase operator [] (int i) const { //can't be virtual
    PyObject* o = PySequence_GetItem(_obj, i);
    if (o == 0)
      Fail(PyExc_IndexError, "index out of range");
    return LoseRef(o);
  };
  //PySequence_GetSlice
  //virtual PWOSequence& operator [] (PWSlice& x) {...};
  PWOSequence getSlice(int lo, int hi) const {
    PyObject* o = PySequence_GetSlice(_obj, lo, hi);
    if (o == 0)
      Fail(PyExc_IndexError, "could not obtain slice");
    return LoseRef(o);
  };
  //PySequence_In
  bool in(const PWOBase& value) const {
    int rslt = PySequence_In(_obj, value);
    if (rslt==-1)
      Fail(PyExc_RuntimeError, "problem in in");
    return (rslt==1);
  };
  //PySequence_Index
  int index(const PWOBase& value) const {
    int rslt = PySequence_Index(_obj, value);
    if (rslt==-1)
      Fail(PyExc_IndexError, "value not found");
    return rslt;
  };
  //PySequence_Length
  int len() const {
    return PySequence_Length(_obj);
  };
  // added length for compatibility with std::string.
  int length() const {
    return PySequence_Length(_obj);
  };
  //PySequence_Repeat
  PWOSequence operator * (int count) const {
    PyObject* rslt = PySequence_Repeat(_obj, count);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "sequence repeat failed");
    return LoseRef(rslt);
  };
  //PySequence_Tuple
};

class PWOList;

class PWOTuple : public PWOSequence
{
public:
  PWOTuple(int sz=0) : PWOSequence (PyTuple_New(sz))  { LoseRef(_obj); }
  PWOTuple(const PWOTuple& other) : PWOSequence(other) { }
  PWOTuple(PyObject* obj) : PWOSequence(obj) { _violentTypeCheck(); }
  PWOTuple(const PWOList& list);
  virtual ~PWOTuple() {};

  virtual PWOTuple& operator=(const PWOTuple& other) {
    GrabRef(other);
    return *this;
  };
  /*virtual*/ PWOTuple& operator=(const PWOBase& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyTuple_Check(_obj)) {
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a Python Tuple");
    }
  };
  void setItem(int ndx, PWOBase& val) {
    int rslt = PyTuple_SetItem(_obj, ndx, val);
    val.disOwn(); //when using PyTuple_SetItem, he steals my reference
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };
};

class PWOString : public PWOSequence
{
public:
  PWOString() : PWOSequence() {};
  PWOString(const char* s)
    : PWOSequence(PyString_FromString((char* )s)) { LoseRef(_obj); }
  PWOString(const char* s, int sz)
    : PWOSequence(PyString_FromStringAndSize((char* )s, sz)) {  LoseRef(_obj); }
  PWOString(const PWOString& other)
    : PWOSequence(other) {};
  PWOString(PyObject* obj)
    : PWOSequence(obj) { _violentTypeCheck(); };
  PWOString(const PWOBase& other)
    : PWOSequence(other) { _violentTypeCheck(); };
  virtual ~PWOString() {};

  virtual PWOString& operator=(const PWOString& other) {
    GrabRef(other);
    return *this;
  };
  PWOString& operator=(const PWOBase& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyString_Check(_obj)) {
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a Python String");
    }
  };
  operator const char* () const {
    return PyString_AsString(_obj);
  };
  static PWOString format(const PWOString& fmt, PWOTuple& args){
    PyObject * rslt =PyString_Format(fmt, args);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "string format failed");
    return LoseRef(rslt);
  };
};
#endif // PWOSEQUENCE_H_INCLUDED_
