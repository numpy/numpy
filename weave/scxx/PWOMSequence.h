/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
*********************************************/
#if !defined(PWOMSEQUENCE_H_INCLUDED_)
#define PWOMSEQUENCE_H_INCLUDED_

#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000

#include "PWOBase.h"
#include "PWOSequence.h"
#include <string>


class PWOList;

class PWOListMmbr : public PWOBase
{
  PWOList& _parent;
  int _ndx;
public:
  PWOListMmbr(PyObject* obj, PWOList& parent, int ndx);
  virtual ~PWOListMmbr() {};
  PWOListMmbr& operator=(const PWOBase& other);
  PWOListMmbr& operator=(int other);
  PWOListMmbr& operator=(float other);
  PWOListMmbr& operator=(double other);
  PWOListMmbr& operator=(const char* other);
  PWOListMmbr& operator=(std::string other);
};

class PWOList : public PWOSequence
{
public:
  PWOList(int size=0) : PWOSequence (PyList_New(size)) { LoseRef(_obj); }
  PWOList(const PWOList& other) : PWOSequence(other) {};
  PWOList(PyObject* obj) : PWOSequence(obj) {
    _violentTypeCheck();
  };
  virtual ~PWOList() {};

  virtual PWOList& operator=(const PWOList& other) {
    GrabRef(other);
    return *this;
  };
  PWOList& operator=(const PWOBase& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyList_Check(_obj)) { //should probably check the sequence methods for non-0 setitem
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a mutable sequence");
    }
  };
  //PySequence_DelItem    ##lists
  bool delItem(int i) {
    int rslt = PySequence_DelItem(_obj, i);
    if (rslt == -1)
      Fail(PyExc_RuntimeError, "cannot delete item");
    return true;
  };
  //PySequence_DelSlice   ##lists
  bool delSlice(int lo, int hi) {
    int rslt = PySequence_DelSlice(_obj, lo, hi);
    if (rslt == -1)
      Fail(PyExc_RuntimeError, "cannot delete slice");
    return true;
  };
  //PySequence_GetItem    ##lists - return PWOListMmbr (mutable) otherwise just a PWOBase
  PWOListMmbr operator [] (int i) {       // can't be virtual
    //PyObject* o = PySequence_GetItem(_obj, i); assumes item is valid
    PyObject* o = PyList_GetItem(_obj, i);  // get a "borrowed" refcount
    //Py_XINCREF(o);
    //if (o == 0)
    //      Fail(PyExc_IndexError, "index out of range");
    return PWOListMmbr(o, *this, i); // this increfs
  };
  //PySequence_SetItem    ##Lists
  void setItem(int ndx, PWOBase& val) {
    //int rslt = PySequence_SetItem(_obj, ndx, val); - assumes old item is valid
    int rslt = PyList_SetItem(_obj, ndx, val);
    val.disOwn();   //when using PyList_SetItem, he steals my reference
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void setItem(int ndx, int val) {
    //int rslt = PySequence_SetItem(_obj, ndx, val); - assumes old item is valid
    int rslt = PyList_SetItem(_obj, ndx, PyInt_FromLong(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };
  
  void setItem(int ndx, double val) {
    //int rslt = PySequence_SetItem(_obj, ndx, val); - assumes old item is valid
    int rslt = PyList_SetItem(_obj, ndx, PyFloat_FromDouble(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void setItem(int ndx, char* val) {
    //int rslt = PySequence_SetItem(_obj, ndx, val); - assumes old item is valid
    int rslt = PyList_SetItem(_obj, ndx, PyString_FromString(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void setItem(int ndx, std::string val) {
    //int rslt = PySequence_SetItem(_obj, ndx, val); - assumes old item is valid
    int rslt = PyList_SetItem(_obj, ndx, PyString_FromString(val.c_str()));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  //PySequence_SetSlice   ##Lists
  void setSlice(int lo, int hi, const PWOSequence& slice) {
    int rslt = PySequence_SetSlice(_obj, lo, hi, slice);
    if (rslt==-1)
      Fail(PyExc_RuntimeError, "Error setting slice");
  };
  
  //PyList_Append
  PWOList& append(const PWOBase& other) {
    int rslt = PyList_Append(_obj, other);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error appending");
    };
    return *this;
  };
  PWOList& append(int other);
  PWOList& append(double other);
  PWOList& append(char* other);
  PWOList& append(std::string other);

  //PyList_AsTuple
  // problem with this is it's created on the heap
  //virtual PWOTuple& asTuple() const {
  //      PyObject* rslt = PyList_AsTuple(_obj);
  //      PWOTuple rtrn = new PWOTuple(rslt);
  //      Py_XDECREF(rslt);       //AsTuple set refcnt to 1, PWOTuple(rslt) increffed
  //      return *rtrn;
  //};
  //PyList_GetItem - inherited OK
  //PyList_GetSlice - inherited OK
  //PyList_Insert
  PWOList& insert(int ndx, PWOBase& other) {
    int rslt = PyList_Insert(_obj, ndx, other);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error inserting");
    };
    return *this;
  };
  PWOList& insert(int ndx, int other);
  PWOList& insert(int ndx, double other);
  PWOList& insert(int ndx, char* other);  
  PWOList& insert(int ndx, std::string other);

  //PyList_New
  //PyList_Reverse
  PWOList& reverse() {
    int rslt = PyList_Reverse(_obj);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error reversing");
    };
    return *this;   //HA HA - Guido can't stop me!!!
  };
  //PyList_SetItem - using abstract
  //PyList_SetSlice - using abstract
  //PyList_Size - inherited OK
  //PyList_Sort
  PWOList& sort() {
    int rslt = PyList_Sort(_obj);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error sorting");
    };
    return *this;   //HA HA - Guido can't stop me!!!
  };
};

#endif // PWOMSEQUENCE_H_INCLUDED_
