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
#include <complex>

// for debugging
#include <iostream>

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
    : _obj (0), _own (0) { 
        //std::cout << "construct before: (own,ref)" << (int)_own << " " << obj->ob_refcnt << std::endl;
        GrabRef(obj); 
        //std::cout << "construct after: (own,ref)" << (int)_own << " " << obj->ob_refcnt << std::endl;
        }

  //-------------------------------------------------------------------------
  //  Numeric constructors
  //-------------------------------------------------------------------------
  /*
  object(bool val) : _obj (0), _own (0) { 
    GrabRef(PyInt_FromLong((int)val)); 
    LoseRef(_obj); 
  };
  */
  object(int val) : _obj (0), _own (0) { 
    GrabRef(PyInt_FromLong(val)); 
    LoseRef(_obj); 
  };
  object(long val) : _obj (0), _own (0) { 
    GrabRef(PyInt_FromLong(val)); 
    LoseRef(_obj); 
  };  
  object(unsigned long val) : _obj (0), _own (0) { 
    GrabRef(PyLong_FromUnsignedLong(val)); 
    LoseRef(_obj); 
  };  
  object(double val) : _obj (0), _own (0) { 
    GrabRef(PyFloat_FromDouble(val)); 
    LoseRef(_obj); 
  };
  object(std::complex<double>& val) : _obj (0), _own (0) { 
    GrabRef(PyComplex_FromDoubles(val.real(),val.imag())); 
    LoseRef(_obj); 
  };
  
  //-------------------------------------------------------------------------
  // string constructors
  //-------------------------------------------------------------------------
  object(char* val) : _obj (0), _own (0) { 
    GrabRef(PyString_FromString(val)); 
    LoseRef(_obj); 
  };
  object(std::string& val) : _obj (0), _own (0) { 
    GrabRef(PyString_FromString((char*)val.c_str())); 
    LoseRef(_obj); 
  };
  
  //-------------------------------------------------------------------------
  // destructor
  //-------------------------------------------------------------------------
  virtual ~object()
    { 
        //std::cout << "destruct: (own,ref)" << (int)_own << " " << _obj->ob_refcnt << std::endl;
        Py_XDECREF(_own); 
        //std::cout << "destruct: (own,ref)" << (int)_own << " " << _obj->ob_refcnt << std::endl;
    }  
  
  //-------------------------------------------------------------------------
  // casting operators
  //-------------------------------------------------------------------------
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
        Fail(PyExc_TypeError, "cannot convert value to float");
    return (float) PyFloat_AsDouble(_obj);
  };  
  operator double () const {
    if (!PyFloat_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to double");
    return PyFloat_AsDouble(_obj);
  };  
  operator std::complex<double> () const {
    if (!PyComplex_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to complex");
    return std::complex<double>(PyComplex_RealAsDouble(_obj),
                                PyComplex_ImagAsDouble(_obj));
  };  
  operator std::string () const {
    if (!PyString_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to std::string");
    return std::string(PyString_AsString(_obj));
  };  
  operator char* () const {
    if (!PyString_Check(_obj))
        Fail(PyExc_TypeError, "cannot convert value to std::string");
    return PyString_AsString(_obj);
  };  
  
  //-------------------------------------------------------------------------
  // equal operator
  //-------------------------------------------------------------------------
  object& operator=(const object& other) {
    GrabRef(other);
    return *this;
  };
  
  //-------------------------------------------------------------------------
  // printing
  //
  // !! UNTESTED
  //-------------------------------------------------------------------------
  int print(FILE *f, int flags) const {
    return PyObject_Print(_obj, f, flags);
  };

  //-------------------------------------------------------------------------
  // hasattr -- test if object has specified attribute
  //-------------------------------------------------------------------------  
  int hasattr(const char* nm) const {
    return PyObject_HasAttrString(_obj, (char*) nm) == 1;
  };
  int hasattr(std::string nm) const {
    return PyObject_HasAttrString(_obj, (char*) nm.c_str()) == 1;
  };

  //-------------------------------------------------------------------------
  // attribute access
  //
  // should this return a reference?  Need to think about this.
  //-------------------------------------------------------------------------
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
  
  //-------------------------------------------------------------------------
  // setting attributes
  // !! NOT TESTED
  //-------------------------------------------------------------------------
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

  //-------------------------------------------------------------------------
  // calling methods
  //-------------------------------------------------------------------------
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

  //-------------------------------------------------------------------------
  // calling callable objects
  //-------------------------------------------------------------------------
  object call() const;
  object call(tuple& args) const;
  object call(tuple& args, dict& kws) const;

  //-------------------------------------------------------------------------
  // sequence methods
  // !! NOT TESTED
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  // iter methods
  // !! NOT TESTED
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  // del objects
  // !! NOT TESTED
  //-------------------------------------------------------------------------
  int del(const char* nm) {
    return PyObject_DelAttrString(_obj, (char*) nm);
  };
  int del(const object& nm) {
    return PyObject_DelAttr(_obj, nm);
  };
  
  //-------------------------------------------------------------------------
  // comparison
  // !! NOT TESTED
  //-------------------------------------------------------------------------
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
