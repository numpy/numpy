#if !defined(TUPLE_H_INCLUDED_)
#define TUPLE_H_INCLUDED_

#include "sequence.h"
#include <string>

namespace py {


// added to make tuples mutable.
class tuple_member : public object
{
  tuple& _parent;
  int _ndx;
public:
  tuple_member(PyObject* obj, tuple& parent, int ndx);
  virtual ~tuple_member() {};
  tuple_member& operator=(const object& other);
  tuple_member& operator=(const tuple_member& other);
  tuple_member& operator=(int other);
  tuple_member& operator=(double other);
  tuple_member& operator=(const char* other);
  tuple_member& operator=(std::string other);
};
    
class tuple : public sequence
{
public:
  tuple(int sz=0) : sequence (PyTuple_New(sz))  { LoseRef(_obj); }
  tuple(const tuple& other) : sequence(other) { }
  tuple(PyObject* obj) : sequence(obj) { _violentTypeCheck(); }
  tuple(const list& list);
  virtual ~tuple() {};

  virtual tuple& operator=(const tuple& other) {
    GrabRef(other);
    return *this;
  };
  /*virtual*/ tuple& operator=(const object& other) {
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
  void set_item(int ndx, object& val) {
    int rslt = PyTuple_SetItem(_obj, ndx, val);
    val.disown(); //when using PyTuple_SetItem, he steals my reference
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };
  
  // ej: additions
  void set_item(int ndx, int val) {
    int rslt = PyTuple_SetItem(_obj, ndx, PyInt_FromLong(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void set_item(int ndx, double val) {
    int rslt = PyTuple_SetItem(_obj, ndx, PyFloat_FromDouble(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void set_item(int ndx, char* val) {
    int rslt = PyTuple_SetItem(_obj, ndx, PyString_FromString(val));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  void set_item(int ndx, std::string val) {
    int rslt = PyTuple_SetItem(_obj, ndx, PyString_FromString(val.c_str()));
    if (rslt==-1)
      Fail(PyExc_IndexError, "Index out of range");
  };

  tuple_member operator [] (int i) {       // can't be virtual
    //PyObject* o = PySequence_GetItem(_obj, i); assumes item is valid
    PyObject* o = PyTuple_GetItem(_obj, i);  // get a "borrowed" refcount
    //Py_XINCREF(o);
    //if (o == 0)
    //      Fail(PyExc_IndexError, "index out of range");
    return tuple_member(o, *this, i); // this increfs
  };
  // ej: end additions
  
};// class tuple

} // namespace

#endif // TUPLE_H_INCLUDED_
