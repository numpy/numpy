/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com

  modified for weave by eric jones
*********************************************/
#if !defined(LIST_H_INCLUDED_)
#define LIST_H_INCLUDED_

// ej: not sure what this is about, but we'll leave it.
#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000

#include "object.h"
#include "sequence.h"
#include <string>

namespace py {
    
class list_member : public object
{
  list& _parent;
  int _ndx;
public:
  list_member(PyObject* obj, list& parent, int ndx);
  virtual ~list_member() {};
  list_member& operator=(const object& other);
  list_member& operator=(const list_member& other);
  list_member& operator=(int other);
  list_member& operator=(double other);
  list_member& operator=(const char* other);
  list_member& operator=(std::string other);
};

class list : public sequence
{
public:
  list(int size=0) : sequence (PyList_New(size)) { LoseRef(_obj); }
  list(const list& other) : sequence(other) {};
  list(PyObject* obj) : sequence(obj) {
    _violentTypeCheck();
  };
  virtual ~list() {};

  virtual list& operator=(const list& other) {
    GrabRef(other);
    return *this;
  };
  list& operator=(const object& other) {
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
  bool del(int i) {
    int rslt = PySequence_DelItem(_obj, i);
    if (rslt == -1)
      Fail(PyExc_RuntimeError, "cannot delete item");
    return true;
  };
  //PySequence_DelSlice   ##lists
  bool del(int lo, int hi) {
    int rslt = PySequence_DelSlice(_obj, lo, hi);
    if (rslt == -1)
      Fail(PyExc_RuntimeError, "cannot delete slice");
    return true;
  };
  //PySequence_GetItem    ##lists - return list_member (mutable) otherwise just a object
  list_member operator [] (int i) {       // can't be virtual
    //PyObject* o = PySequence_GetItem(_obj, i); assumes item is valid
    PyObject* o = PyList_GetItem(_obj, i);  // get a "borrowed" refcount
    //Py_XINCREF(o);
    //if (o == 0)
    //      Fail(PyExc_IndexError, "index out of range");
    return list_member(o, *this, i); // this increfs
  };
  //PySequence_SetItem    ##Lists
  void set_item(int ndx, object& val) {
    //int rslt = PySequence_SetItem(_obj, ndx, val); - assumes old item is valid
    int rslt = PyList_SetItem(_obj, ndx, val);
    val.disown();   //when using PyList_SetItem, he steals my reference
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void set_item(int ndx, int val) {
    int rslt = PyList_SetItem(_obj, ndx, PyInt_FromLong(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };
  
  void set_item(int ndx, double val) {
    int rslt = PyList_SetItem(_obj, ndx, PyFloat_FromDouble(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void set_item(int ndx, char* val) {
    int rslt = PyList_SetItem(_obj, ndx, PyString_FromString(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void set_item(int ndx, std::string val) {
    int rslt = PyList_SetItem(_obj, ndx, PyString_FromString(val.c_str()));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  //PySequence_SetSlice   ##Lists
  void setSlice(int lo, int hi, const sequence& slice) {
    int rslt = PySequence_SetSlice(_obj, lo, hi, slice);
    if (rslt==-1)
      Fail(PyExc_RuntimeError, "Error setting slice");
  };

  //PyList_Append
  list& append(const object& other) {
    int rslt = PyList_Append(_obj, other);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error appending");
    }
    return *this;
  };

  list& append(int other) {
    PyObject* oth = PyInt_FromLong(other);
    int rslt = PyList_Append(_obj, oth); 
    Py_XDECREF(oth);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error appending");
    }
  return *this;
  };

  list& append(double other) {
    PyObject* oth = PyFloat_FromDouble(other);
    int rslt = PyList_Append(_obj, oth); 
    Py_XDECREF(oth);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error appending");
    }
  return *this;
  };

  list& append(char* other) {
    PyObject* oth = PyString_FromString(other);
    int rslt = PyList_Append(_obj, oth); 
    Py_XDECREF(oth);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error appending");
    }
  return *this;
  };

  list& append(std::string other) {
    PyObject* oth = PyString_FromString(other.c_str());
    int rslt = PyList_Append(_obj, oth); 
    Py_XDECREF(oth);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error appending");
    }
  return *this;
  };

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
  list& insert(int ndx, object& other) {
    int rslt = PyList_Insert(_obj, ndx, other);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error inserting");
    };
    return *this;
  };
  list& insert(int ndx, int other);
  list& insert(int ndx, double other);
  list& insert(int ndx, char* other);  
  list& insert(int ndx, std::string other);

  //PyList_New
  //PyList_Reverse
  list& reverse() {
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
  list& sort() {
    int rslt = PyList_Sort(_obj);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one 
      Fail(PyExc_RuntimeError, "Error sorting");
    };
    return *this;   //HA HA - Guido can't stop me!!!
  };
}; // class list

} // namespace py

#endif // LIST_H_INCLUDED_
