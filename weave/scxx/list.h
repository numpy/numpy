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
#include <complex>

namespace py {

class list : public sequence
{
    
public:
  //-------------------------------------------------------------------------
  // constructors
  //-------------------------------------------------------------------------
  list(int size=0) : sequence (PyList_New(size)) { lose_ref(_obj); }
  list(const list& other) : sequence(other) {};
  list(PyObject* obj) : sequence(obj) {
    _violentTypeCheck();
  };
  
  //-------------------------------------------------------------------------
  // descructor
  //-------------------------------------------------------------------------
  virtual ~list() {};

  //-------------------------------------------------------------------------
  // operator=
  //-------------------------------------------------------------------------
  virtual list& operator=(const list& other) {
    grab_ref(other);
    return *this;
  };
  list& operator=(const object& other) {
    grab_ref(other);
    _violentTypeCheck();
    return *this;
  };
  
  //-------------------------------------------------------------------------
  // type checking
  //-------------------------------------------------------------------------
  virtual void _violentTypeCheck() {
    if (!PyList_Check(_obj)) { 
      //should probably check the sequence methods for non-0 setitem
      grab_ref(0);
      fail(PyExc_TypeError, "Not a mutable sequence");
    }
  };
  
  //-------------------------------------------------------------------------
  // del -- remove the item at a given index from the list.
  //        also a two valued version for removing slices from a list.
  //-------------------------------------------------------------------------
  bool del(int i) {
    int rslt = PySequence_DelItem(_obj, i);
    if (rslt == -1)
      fail(PyExc_RuntimeError, "cannot delete item");
    return true;
  };
  bool del(int lo, int hi) {
    int rslt = PySequence_DelSlice(_obj, lo, hi);
    if (rslt == -1)
      fail(PyExc_RuntimeError, "cannot delete slice");
    return true;
  };

  //-------------------------------------------------------------------------
  // operator[] -- access/set elements in a list
  //-------------------------------------------------------------------------
  indexed_ref operator [] (int i) {
    PyObject* o = PyList_GetItem(_obj, i);  // get a "borrowed" refcount
    // don't throw error for when [] fails because it might be on left hand 
    // side (a[0] = 1).  If the list was just created, it will be filled 
    // with NULL values, and setting the values should be ok.  However, we
    // do want to catch index errors that might occur on the right hand side
    // (obj = a[4] when a has len==3).
    if (!o) {
      if (PyErr_ExceptionMatches(PyExc_IndexError))
        throw 1;
    }
    return indexed_ref(o, *this, i); // this increfs
  };

  //-------------------------------------------------------------------------
  // set_item -- set list entry at a given index to a new value.
  //-------------------------------------------------------------------------
  virtual void set_item(int ndx, object& val) {
    //int rslt = PySequence_SetItem(_obj, ndx, val); - assumes old item is valid
    int rslt = PyList_SetItem(_obj, ndx, val);
    val.disown();   //when using PyList_SetItem, he steals my reference
    if (rslt==-1)
      fail(PyExc_IndexError, "Index out of range");
  };

  //-------------------------------------------------------------------------
  // set_slice -- set slice to a new sequence of values
  //
  // !! NOT TESTED
  //-------------------------------------------------------------------------
  void set_slice(int lo, int hi, const sequence& slice) {
    int rslt = PySequence_SetSlice(_obj, lo, hi, slice);
    if (rslt==-1)
      fail(PyExc_RuntimeError, "Error setting slice");
  };

  //-------------------------------------------------------------------------
  // append -- add new item to end of list
  //           overloaded to accept all of the common weave types.
  //-------------------------------------------------------------------------
  list& append(const object& other) {
    int rslt = PyList_Append(_obj, other);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one
      fail(PyExc_RuntimeError, "Error appending");
    }
    return *this;
  };
  list& append(int other) {
    object oth = other;
    return append(oth);
  };
  list& append(double other) {
    object oth = other;
    return append(oth);
  };
  list& append(const std::complex<double>& other) {
    object oth = other;
    return append(oth);
  };
  list& append(const char* other) {
    object oth = other;
    return append(oth);
  };
  list& append(const std::string& other) {
    object oth = other;
    return append(oth);
  };
  
  //-------------------------------------------------------------------------
  // insert -- insert a new item before the given index.
  //           overloaded to accept all of the common weave types.
  //-------------------------------------------------------------------------
  list& insert(int ndx, object& other) {
    int rslt = PyList_Insert(_obj, ndx, other);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one
      fail(PyExc_RuntimeError, "Error inserting");
    };
    return *this;
  };
  list& list::insert(int ndx, int other) {
    object oth = other;
    return insert(ndx, oth);
  };
  list& list::insert(int ndx, double other) {
    object oth = other;
    return insert(ndx, oth);
  };
  list& list::insert(int ndx, std::complex<double>& other) {
    object oth = other;
    return insert(ndx, oth);
  };  
  list& list::insert(int ndx, const char* other) {
    object oth = other;
    return insert(ndx, oth);
  };
  list& list::insert(int ndx, const std::string& other) {
    object oth = other;
    return insert(ndx, oth);
  };

  //-------------------------------------------------------------------------
  // reverse -- reverse the order of items in the list.
  //-------------------------------------------------------------------------
  list& reverse() {
    int rslt = PyList_Reverse(_obj);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one
      fail(PyExc_RuntimeError, "Error reversing");
    };
    return *this;   //HA HA - Guido can't stop me!!!
  };

  //-------------------------------------------------------------------------
  // sort -- sort the items in the list.
  //-------------------------------------------------------------------------
  list& sort() {
    int rslt = PyList_Sort(_obj);
    if (rslt==-1) {
      PyErr_Clear();  //Python sets one
      fail(PyExc_RuntimeError, "Error sorting");
    };
    return *this;   //HA HA - Guido can't stop me!!!
  };
}; // class list

} // namespace py

#endif // LIST_H_INCLUDED_
