/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com

  modified heavily for weave by eric jones.
*********************************************/

#if !defined(OBJECT_H_INCLUDED_)
#define OBJECT_H_INCLUDED_

#include <Python.h>
#include <limits.h>
#include <string>

namespace py {

void Fail(PyObject*, const char* msg);

// used in method call specification.
class tuple;
class dict;
    
class object  
{
protected:
  PyObject* _obj;

    // incref new owner, decref old owner, and adjust to new owner
  void GrabRef(PyObject* newObj);
    // decrease reference count without destroying the object
  static PyObject* LoseRef(PyObject* o)
    { if (o != 0) --(o->ob_refcnt); return o; }

private:
  PyObject* _own; // set to _obj if we "own" a reference to _obj, else zero

public:
  object()
    : _obj (0), _own (0) { }
  object(const object& other)
    : _obj (0), _own (0) { GrabRef(other); }
  object(PyObject* obj)
    : _obj (0), _own (0) { GrabRef(obj); }

  virtual ~object()
    { Py_XDECREF(_own); }
  
  object& operator=(const object& other) {
    GrabRef(other);
    return *this;
  };
  operator PyObject* () const {
    return _obj;
  };
  
  operator int () const {
    if (!PyInt_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to integer");
    return PyInt_AsLong(_obj);
  };  
  operator float () const {
    if (!PyFloat_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to double");
    return (float) PyFloat_AsDouble(_obj);
  };  
  operator double () const {
    if (!PyFloat_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to double");
    return PyFloat_AsDouble(_obj);
  };  

  operator std::string () const {
    if (!PyString_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to std::string");
    return std::string(PyString_AsString(_obj));
  };  
  
  int print(FILE *f, int flags) const {
    return PyObject_Print(_obj, f, flags);
  };
  bool hasattr(const char* nm) const {
    return PyObject_HasAttrString(_obj, (char*) nm) == 1;
  };

  // Need to change return type?
  object attr(const char* nm) const {    
    PyObject* val = PyObject_GetAttrString(_obj, (char*) nm);
    if (!val)
        throw 1;
    return object(LoseRef(val));    
  };

  object attr(std::string nm) const {
    return attr(nm.c_str());
  };

  object attr(const object& nm) const {
    PyObject* val = PyObject_GetAttr(_obj, nm);
    if (!val)
        throw 1;
    return object(LoseRef(val));    
  };  
  
  void set_attr(const char* nm, object& val) {
    int res = PyObject_SetAttrString(_obj, (char*) nm, val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, object& val) {
    int res = PyObject_SetAttr(_obj, nm, val);
    if (res == -1)
        throw 1;
  };

  object mcall(const char* nm);
  object mcall(const char* nm, tuple& args);
  object mcall(const char* nm, tuple& args, dict& kwargs);

  object mcall(std::string nm) {
    return mcall(nm.c_str());
  }
  object mcall(std::string nm, tuple& args) {
    return mcall(nm.c_str(),args);
  }
  object mcall(std::string nm, tuple& args, dict& kwargs) {
    return mcall(nm.c_str(),args,kwargs);
  }

  object call() const;
  object call(tuple& args) const;
  object call(tuple& args, dict& kws) const;

  int del(const char* nm) {
    return PyObject_DelAttrString(_obj, (char*) nm);
  };
  int del(const object& nm) {
    return PyObject_DelAttr(_obj, nm);
  };
  
  int cmp(const object& other) const {
    int rslt = 0;
    int rc = PyObject_Cmp(_obj, other, &rslt);
    if (rc == -1)
      Fail(PyExc_TypeError, "cannot make the comparison");
    return rslt;
  };
  bool operator == (const object& other) const {
    return cmp(other) == 0;
  };
  bool operator != (const object& other) const {
    return cmp(other) != 0;
  };
  bool operator > (const object& other) const {
    return cmp(other) > 0;
  };
  bool operator < (const object& other) const {
    return cmp(other) < 0;
  };
  bool operator >= (const object& other) const {
    return cmp(other) >= 0;
  };
  bool operator <= (const object& other) const {
    return cmp(other) <= 0;
  };
      
  PyObject* repr() const {
    return LoseRef(PyObject_Repr(_obj));
  };
  /*
  PyObject* str() const {
    return LoseRef(PyObject_Str(_obj));
  };
  */
  bool is_callable() const {
    return PyCallable_Check(_obj) == 1;
  };
  int hash() const {
    return PyObject_Hash(_obj);
  };
  bool is_true() const {
    return PyObject_IsTrue(_obj) == 1;
  };
  PyObject* type() const {
    return LoseRef(PyObject_Type(_obj));
  };
  PyObject* disown() {
    _own = 0;
    return _obj;
  };
};

} // namespace

#endif // !defined(OBJECT_H_INCLUDED_)
