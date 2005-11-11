/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com

  modified for weave by eric jones.
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

void fail(PyObject*, const char* msg);

//---------------------------------------------------------------------------
// py::object -- A simple C++ interface to Python objects.
//
// This is the basic type from which all others are derived from.  It is 
// also quite useful on its own.  The class is very light weight as far as
// data contents, carrying around only two python pointers.
//---------------------------------------------------------------------------
    
class object  
{
protected:

  //-------------------------------------------------------------------------
  // _obj is the underlying pointer to the real python object.
  //-------------------------------------------------------------------------
  PyObject* _obj;

  //-------------------------------------------------------------------------
  // grab_ref (rename to grab_ref)
  //
  // incref new owner, decref old owner, and adjust to new owner
  //-------------------------------------------------------------------------
  void grab_ref(PyObject* newObj) {
      // be careful to incref before decref if old is same as new
      Py_XINCREF(newObj);
      Py_XDECREF(_own);
      _own = _obj = newObj;
  };

  //-------------------------------------------------------------------------
  // lose_ref (rename to lose_ref)
  //
  // decrease reference count without destroying the object.
  //-------------------------------------------------------------------------
  static PyObject* lose_ref(PyObject* o)
    { if (o != 0) --(o->ob_refcnt); return o; }

private:
  //-------------------------------------------------------------------------
  // _own is set to _obj if we "own" a reference to _obj, else zero
  //-------------------------------------------------------------------------
  PyObject* _own; 

public:
  //-------------------------------------------------------------------------
  // forward declaration of reference obj returned when [] used as an lvalue.
  //-------------------------------------------------------------------------
  class keyed_ref;  

  object()
    : _obj (0), _own (0) { };
  object(const object& other)
    : _obj (0), _own (0) { grab_ref(other); };
  object(PyObject* obj)
    : _obj (0), _own (0) { grab_ref(obj);   };

  //-------------------------------------------------------------------------
  //  Numeric constructors
  //-------------------------------------------------------------------------
  object(bool val) { 
    _obj = _own = PyInt_FromLong((int)val); 
  };
  object(int val) { 
    _obj = _own = PyInt_FromLong((int)val); 
  };
  object(unsigned int val) { 
    _obj = _own = PyLong_FromUnsignedLong(val); 
  };  
  object(long val) { 
    _obj = _own = PyInt_FromLong((int)val); 
  };  
  object(unsigned long val) { 
    _obj = _own = PyLong_FromUnsignedLong(val); 
  };  
  object(double val) {
    _obj = _own = PyFloat_FromDouble(val); 
  };
  object(const std::complex<double>& val) { 
    _obj = _own = PyComplex_FromDoubles(val.real(),val.imag()); 
  };
  
  //-------------------------------------------------------------------------
  // string constructors
  //-------------------------------------------------------------------------
  object(const char* val) {
    _obj = _own = PyString_FromString((char*) val); 
  };
  object(const std::string& val) : _obj (0), _own (0) { 
    _obj = _own = PyString_FromString((char*)val.c_str()); 
  };
  
  //-------------------------------------------------------------------------
  // destructor
  //-------------------------------------------------------------------------
  virtual ~object() { 
    Py_XDECREF(_own); 
  };
  
  //-------------------------------------------------------------------------
  // casting operators
  //-------------------------------------------------------------------------
  operator PyObject* () const {
    return _obj;
  };
  
  operator int () const {
    if (!PyInt_Check(_obj))
        fail(PyExc_TypeError, "cannot convert value to integer");
    return PyInt_AsLong(_obj);
  };  
  operator float () const {
    if (!PyFloat_Check(_obj))
        fail(PyExc_TypeError, "cannot convert value to float");
    return (float) PyFloat_AsDouble(_obj);
  };  
  operator double () const {
    if (!PyFloat_Check(_obj))
        fail(PyExc_TypeError, "cannot convert value to double");
    return PyFloat_AsDouble(_obj);
  };  
  operator std::complex<double> () const {
    if (!PyComplex_Check(_obj))
        fail(PyExc_TypeError, "cannot convert value to complex");
    return std::complex<double>(PyComplex_RealAsDouble(_obj),
                                PyComplex_ImagAsDouble(_obj));
  };  
  operator std::string () const {
    if (!PyString_Check(_obj))
        fail(PyExc_TypeError, "cannot convert value to std::string");
    return std::string(PyString_AsString(_obj));
  };  
  operator char* () const {
    if (!PyString_Check(_obj))
        fail(PyExc_TypeError, "cannot convert value to char*");
    return PyString_AsString(_obj);
  };  
  
  //-------------------------------------------------------------------------
  // equal operator
  //-------------------------------------------------------------------------
  object& operator=(const object& other) {
    grab_ref(other);
    return *this;
  };
  
  //-------------------------------------------------------------------------
  // printing
  //
  // This needs to become more sophisticated and handle objects that 
  // implement the file protocol.
  //-------------------------------------------------------------------------
  void print(FILE* f, int flags=0) const {
    int res = PyObject_Print(_obj, f, flags);
    if (res == -1)
        throw 1;
  };

  void print(object f, int flags=0) const {
    int res = PyFile_WriteObject(_obj, f, flags);
    if (res == -1)
        throw 1;
  };

  //-------------------------------------------------------------------------
  // hasattr -- test if object has specified attribute
  //-------------------------------------------------------------------------  
  int hasattr(const char* nm) const {
    return PyObject_HasAttrString(_obj, (char*) nm) == 1;
  };
  int hasattr(const std::string& nm) const {
    return PyObject_HasAttrString(_obj, (char*) nm.c_str()) == 1;
  };
  int hasattr(object& nm) const {
    return PyObject_HasAttr(_obj, nm) == 1;
  };
  

  //-------------------------------------------------------------------------
  // attr -- retreive attribute/method from object
  //-------------------------------------------------------------------------
  object attr(const char* nm) const {    
    PyObject* val = PyObject_GetAttrString(_obj, (char*) nm);
    if (!val)
        throw 1;
    return object(lose_ref(val));    
  };

  object attr(const std::string& nm) const {
    return attr(nm.c_str());
  };

  object attr(const object& nm) const {
    PyObject* val = PyObject_GetAttr(_obj, nm);
    if (!val)
        throw 1;
    return object(lose_ref(val));    
  };  
  
  //-------------------------------------------------------------------------
  // setting attributes
  //
  // There is a combinatorial explosion here of function combinations.
  // perhaps there is a casting fix someone can suggest.
  //-------------------------------------------------------------------------
  void set_attr(const char* nm, object& val) {
    int res = PyObject_SetAttrString(_obj, (char*) nm, val);
    if (res == -1)
        throw 1;
  };

  void set_attr(const std::string& nm, object& val) {
    int res = PyObject_SetAttrString(_obj, (char*) nm.c_str(), val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, object& val) {
    int res = PyObject_SetAttr(_obj, nm, val);
    if (res == -1)
        throw 1;
  };

  //////////////     int      //////////////
  void set_attr(const char* nm, int val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm, _val);
    if (res == -1)
        throw 1;
  };

  void set_attr(const std::string& nm, int val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm.c_str(), _val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, int val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttr(_obj, nm, _val);
    if (res == -1)
        throw 1;
  };
  
  ////////////// unsigned long //////////////
  void set_attr(const char* nm, unsigned long val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm, _val);
    if (res == -1)
        throw 1;
  };

  void set_attr(const std::string& nm, unsigned long val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm.c_str(), _val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, unsigned long val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttr(_obj, nm, _val);
    if (res == -1)
        throw 1;
  };

  ////////////// double //////////////
  void set_attr(const char* nm, double val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm, _val);
    if (res == -1)
        throw 1;
  };

  void set_attr(const std::string& nm, double val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm.c_str(), _val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, double val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttr(_obj, nm, _val);
    if (res == -1)
        throw 1;
  };

  ////////////// complex //////////////
  void set_attr(const char* nm, const std::complex<double>& val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm, _val);
    if (res == -1)
        throw 1;
  };

  void set_attr(const std::string& nm, const std::complex<double>& val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm.c_str(), _val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, const std::complex<double>& val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttr(_obj, nm, _val);
    if (res == -1)
        throw 1;
  };
  
  ////////////// char* //////////////
  void set_attr(const char* nm, const char* val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm, _val);
    if (res == -1)
        throw 1;
  };

  void set_attr(const std::string& nm, const char* val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm.c_str(), _val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, const char* val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttr(_obj, nm, _val);
    if (res == -1)
        throw 1;
  };

  ////////////// std::string //////////////
  void set_attr(const char* nm, const std::string& val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm, _val);
    if (res == -1)
        throw 1;
  };

  void set_attr(const std::string& nm, const std::string& val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttrString(_obj, (char*) nm.c_str(), _val);
    if (res == -1)
        throw 1;
  };
  
  void set_attr(const object& nm, const std::string& val) {
    py::object _val = py::object(val);
    int res = PyObject_SetAttr(_obj, nm, _val);
    if (res == -1)
        throw 1;
  };
  
  //-------------------------------------------------------------------------
  // del attributes/methods from object
  //-------------------------------------------------------------------------
  void del(const char* nm) {
    int result = PyObject_DelAttrString(_obj, (char*) nm);
    if (result == -1)
        throw 1;
  };
  void del(const std::string& nm) {
    int result = PyObject_DelAttrString(_obj, (char*) nm.c_str());
    if (result == -1)
        throw 1;
  };
  void del(const object& nm) {
    int result = PyObject_DelAttr(_obj, nm);
    if (result ==-1)
        throw 1;
  };
  
    //-------------------------------------------------------------------------
  // comparison
  // !! NOT TESTED
  //-------------------------------------------------------------------------
  int cmp(const object& other) const {
    int rslt = 0;
    int rc = PyObject_Cmp(_obj, other, &rslt);
    if (rc == -1)
      fail(PyExc_TypeError, "cannot make the comparison");
    return rslt;
  };  
  int cmp(int other) const {
    object _other = object(other);
    return cmp(_other);
  };  
  int cmp(unsigned long other) const {
    object _other = object(other);
    return cmp(_other);
  };
  int cmp(double other) const {
    object _other = object(other);
    return cmp(_other);
  };
  int cmp(const std::complex<double>& other) const {
    object _other = object(other);
    return cmp(_other);
  };
  
  int cmp(const char* other) const {
    object _other = object((char*)other);
    return cmp(_other);
  };
  
  int cmp(const std::string& other) const {
    object _other = object(other);
    return cmp(_other);
  };
    
  bool operator == (const object& other) const {
    return cmp(other) == 0;
  };
  bool operator == (int other) const {
    return cmp(other) == 0;
  };
  bool operator == (unsigned long other) const {
    return cmp(other) == 0;
  };
  bool operator == (double other) const {
    return cmp(other) == 0;
  };
  bool operator == (const std::complex<double>& other) const {
    return cmp(other) == 0;
  };
  bool operator == (const std::string& other) const {
    return cmp(other) == 0;
  };
  bool operator == (const char* other) const {
    return cmp(other) == 0;
  };

  bool operator != (const object& other) const {
    return cmp(other) != 0;
  };
  bool operator != (int other) const {
    return cmp(other) != 0;
  };
  bool operator != (unsigned long other) const {
    return cmp(other) != 0;
  };
  bool operator != (double other) const {
    return cmp(other) != 0;
  };
  bool operator != (const std::complex<double>& other) const {
    return cmp(other) != 0;
  };
  bool operator != (const std::string& other) const {
    return cmp(other) != 0;
  };
  bool operator != (const char* other) const {
    return cmp(other) != 0;
  };
    
  bool operator < (const object& other) const {
    return cmp(other) < 0;
  };
  bool operator < (int other) const {
    return cmp(other) < 0;
  };
  bool operator < (unsigned long other) const {
    return cmp(other) < 0;
  };
  bool operator < (double other) const {
    return cmp(other) < 0;
  };
  bool operator < (const std::complex<double>& other) const {
    return cmp(other) < 0;
  };
  bool operator < (const std::string& other) const {
    return cmp(other) < 0;
  };
  bool operator < (const char* other) const {
    return cmp(other) < 0;
  };
  
  bool operator > (const object& other) const {
    return cmp(other) > 0;
  };
  bool operator > (int other) const {
    return cmp(other) > 0;
  };
  bool operator > (unsigned long other) const {
    return cmp(other) > 0;
  };
  bool operator > (double other) const {
    return cmp(other) > 0;
  };
  bool operator > (const std::complex<double>& other) const {
    return cmp(other) > 0;
  };
  bool operator > (const std::string& other) const {
    return cmp(other) > 0;
  };
  bool operator > (const char* other) const {
    return cmp(other) > 0;
  };

  bool operator >= (const object& other) const {
    return cmp(other) >= 0;
  };
  bool operator >= (int other) const {
    return cmp(other) >= 0;
  };
  bool operator >= (unsigned long other) const {
    return cmp(other) >= 0;
  };
  bool operator >= (double other) const {
    return cmp(other) >= 0;
  };
  bool operator >= (const std::complex<double>& other) const {
    return cmp(other) >= 0;
  };
  bool operator >= (const std::string& other) const {
    return cmp(other) >= 0;
  };
  bool operator >= (const char* other) const {
    return cmp(other) >= 0;
  };
  
  bool operator <= (const object& other) const {
    return cmp(other) <= 0;
  };
  bool operator <= (int other) const {
    return cmp(other) <= 0;
  };
  bool operator <= (unsigned long other) const {
    return cmp(other) <= 0;
  };
  bool operator <= (double other) const {
    return cmp(other) <= 0;
  };
  bool operator <= (const std::complex<double>& other) const {
    return cmp(other) <= 0;
  };
  bool operator <= (const std::string& other) const {
    return cmp(other) <= 0;
  };
  bool operator <= (const char* other) const {
    return cmp(other) <= 0;
  };

  //-------------------------------------------------------------------------
  // string representations
  //
  //-------------------------------------------------------------------------
  std::string repr() const {    
    object result = PyObject_Repr(_obj);
    if (!(PyObject*)result)
        throw 1;
    return std::string(PyString_AsString(result));
  };
  
  std::string str() const {
    object result = PyObject_Str(_obj);
    if (!(PyObject*)result)
        throw 1;
    return std::string(PyString_AsString(result));
  };

  // !! Not Tested  
  object unicode() const {
    object result = PyObject_Unicode(_obj);
    if (!(PyObject*)result)
        throw 1;
    lose_ref(result);    
    return result;
  };
  
  //-------------------------------------------------------------------------
  // calling methods on object
  //
  // Note: I changed args_tup from a tuple& to a object& so that it could
  //       be inlined instead of implemented i weave_imp.cpp.  This 
  //       provides less automatic type checking, but is potentially faster.
  //-------------------------------------------------------------------------
  object object::mcall(const char* nm) {
    object method = attr(nm);
    PyObject* result = PyEval_CallObjectWithKeywords(method,NULL,NULL);
    if (!result)
      throw 1; // signal exception has occured.
    return object(lose_ref(result));
  }
  
  object object::mcall(const char* nm, object& args_tup) {
    object method = attr(nm);
    PyObject* result = PyEval_CallObjectWithKeywords(method,args_tup,NULL);
    if (!result)
      throw 1; // signal exception has occured.
    return object(lose_ref(result));
  }
  
  object object::mcall(const char* nm, object& args_tup, object& kw_dict) {
    object method = attr(nm);
    PyObject* result = PyEval_CallObjectWithKeywords(method,args_tup,kw_dict);
    if (!result)
      throw 1; // signal exception has occured.
    return object(lose_ref(result));
  }

  object mcall(const std::string& nm) {
    return mcall(nm.c_str());
  }
  object mcall(const std::string& nm, object& args_tup) {
    return mcall(nm.c_str(),args_tup);
  }
  object mcall(const std::string& nm, object& args_tup, object& kw_dict) {
    return mcall(nm.c_str(),args_tup,kw_dict);
  }

  //-------------------------------------------------------------------------
  // calling callable objects
  //
  // Note: see not on mcall()
  //-------------------------------------------------------------------------
  object object::call() const {
    PyObject *rslt = PyEval_CallObjectWithKeywords(*this, NULL, NULL);
    if (rslt == 0)
      throw 1;
    return object(lose_ref(rslt));
  }
  object object::call(object& args_tup) const {
    PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args_tup, NULL);
    if (rslt == 0)
      throw 1;
    return object(lose_ref(rslt));
  }
  object object::call(object& args_tup, object& kw_dict) const {
    PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args_tup, kw_dict);
    if (rslt == 0)
      throw 1;
    return object(lose_ref(rslt));
  }

  //-------------------------------------------------------------------------
  // check if object is callable
  //-------------------------------------------------------------------------
  bool is_callable() const {
    return PyCallable_Check(_obj) == 1;
  };

  //-------------------------------------------------------------------------
  // retreive the objects hash value
  //-------------------------------------------------------------------------
  int hash() const {
    int result = PyObject_Hash(_obj);
    if (result == -1 && PyErr_Occurred())
        throw 1;
    return result;        
  };
  
  //-------------------------------------------------------------------------
  // test whether object is true
  //-------------------------------------------------------------------------
  bool is_true() const {
    return PyObject_IsTrue(_obj) == 1;
  };

  //-------------------------------------------------------------------------
  // test for null
  //-------------------------------------------------------------------------
  bool is_null() { 
    return (_obj == NULL);
  }  
  
  /*
   * //-------------------------------------------------------------------------
  // test whether object is not true
  //-------------------------------------------------------------------------
#if defined(__GNUC__) && __GNUC__ < 3
  bool not() const {
#else
  bool operator not() const {
#endif
    return PyObject_Not(_obj) == 1;
  };
  */

  //-------------------------------------------------------------------------
  // return the variable type for the object
  //-------------------------------------------------------------------------
  object type() const {
    PyObject* result = PyObject_Type(_obj);
    if (!result)
        throw 1;
    return lose_ref(result);
  };

  object is_int() const {
    return PyInt_Check(_obj) == 1;
  };

  object is_float() const {
    return PyFloat_Check(_obj) == 1;
  };

  object is_complex() const {
    return PyComplex_Check(_obj) == 1;
  };
  
  object is_list() const {
    return PyList_Check(_obj) == 1;
  };
  
  object is_tuple() const {
    return PyDict_Check(_obj) == 1;
  };
  
  object is_dict() const {
    return PyDict_Check(_obj) == 1;
  };
  
  object is_string() const {
    return PyString_Check(_obj) == 1;
  };
  
  //-------------------------------------------------------------------------
  // size, len, and length are all synonyms.
  // 
  // length() is useful because it allows the same code to work with STL 
  // strings as works with py::objects.
  //-------------------------------------------------------------------------
  int size() const {
    int result = PyObject_Size(_obj);
    if (result == -1)
        throw 1;
    return result;
  };
  int len() const {
    return size();
  };
  int length() const {
    return size();
  };

  //-------------------------------------------------------------------------
  // set_item 
  //
  // To prevent a combonatorial explosion, only objects are allowed for keys.
  // Users are encouraged to use the [] interface for setting values.
  //-------------------------------------------------------------------------  
  virtual void set_item(const object& key, const object& val) {
    int rslt = PyObject_SetItem(_obj, key, val);
    if (rslt==-1)
      throw 1;
  };

  //-------------------------------------------------------------------------
  // operator[]  
  //-------------------------------------------------------------------------
  // !! defined in weave_imp.cpp
  // !! I'd like to refactor things so that they can be defined here.
  keyed_ref operator [] (object& key);
  keyed_ref operator [] (const char* key);
  keyed_ref operator [] (const std::string& key);
  keyed_ref operator [] (int key);
  keyed_ref operator [] (double key);
  keyed_ref operator [] (const std::complex<double>& key);
  
  //-------------------------------------------------------------------------
  // iter methods
  // !! NOT TESTED
  //-------------------------------------------------------------------------
  
  //-------------------------------------------------------------------------
  //  iostream operators
  //-------------------------------------------------------------------------
  friend std::ostream& operator <<(std::ostream& os, py::object& obj);
  //-------------------------------------------------------------------------
  //  refcount utilities 
  //-------------------------------------------------------------------------
        
  PyObject* disown() {
    _own = 0;
    return _obj;
  };
  
  int refcount() {
    return _obj->ob_refcnt;
  }
};


//---------------------------------------------------------------------------
// keyed_ref
//
// Provides a reference value when operator[] returns an lvalue.  The 
// reference has to keep track of its parent object and its key in the parent
// object so that it can insert a new value into the parent at the 
// appropriate place when a new value is assigned to the keyed_ref object.
//
// The keyed_ref class is also used by the py::dict class derived from 
// py::object
// !! Note: Need to check ref counting on key and parent here.
//---------------------------------------------------------------------------
class object::keyed_ref : public object
{
  object& _parent;
  object _key;
public:
  keyed_ref(object obj, object& parent, object& key)
      : object(obj), _parent(parent), _key(key) {};  
  virtual ~keyed_ref() {};

  keyed_ref& operator=(const object& other) {
    grab_ref(other);
    _parent.set_item(_key, other);
    return *this;
  }
  keyed_ref& operator=(int other) {
    object _other = object(other);
    return operator=(_other);
  }  
  keyed_ref& operator=(double other) {
    object _other = object(other);
    return operator=(_other);
  }
  keyed_ref& operator=(const std::complex<double>& other) {
    object _other = object(other);
    return operator=(_other);
  }
  keyed_ref& operator=(const char* other) {
    object _other = object(other);
    return operator=(_other);
  }
  keyed_ref& operator=(const std::string& other) {
    object _other = object(other);
    return operator=(_other);
  }
};
} // namespace


#endif // !defined(OBJECT_H_INCLUDED_)
