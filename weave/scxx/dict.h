/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
  
  modified for weave by eric jones
*********************************************/

#if !defined(DICT_H_INCLUDED_)
#define DICT_H_INCLUDED_
#include <string>
#include "object.h"
#include "list.h"

namespace py {


class dict : public object
{
public:

  //-------------------------------------------------------------------------
  // constructors
  //-------------------------------------------------------------------------
  dict() : object (PyDict_New()) { lose_ref(_obj); }
  dict(const dict& other) : object(other) {};
  dict(PyObject* obj) : object(obj) {
    _violentTypeCheck();
  };
  
  //-------------------------------------------------------------------------
  // destructor
  //-------------------------------------------------------------------------
  virtual ~dict() {};

  //-------------------------------------------------------------------------
  // operator= 
  //-------------------------------------------------------------------------
  virtual dict& operator=(const dict& other) {
    grab_ref(other);
    return *this;
  };
  dict& operator=(const object& other) {
    grab_ref(other);
    _violentTypeCheck();
    return *this;
  };

  //-------------------------------------------------------------------------
  // type checking
  //-------------------------------------------------------------------------
  virtual void _violentTypeCheck() {
    if (!PyDict_Check(_obj)) {
      grab_ref(0);
      fail(PyExc_TypeError, "Not a dictionary");
    }
  };
  
  //-------------------------------------------------------------------------
  // get -- object, numeric, and string versions
  //-------------------------------------------------------------------------  
  object get (object& key) {
    object rslt = PyDict_GetItem(_obj, key);
    return rslt;
  };
  object get (int key) {
    object _key = object(key);
    return get(_key);
  };
  object get (double key) {
    object _key = object(key);
    return get(_key);
  };
  object get (const std::complex<double>& key) {
    object _key = object(key);
    return get(_key);
  };  
  object get (const char* key) {
    object rslt = PyDict_GetItemString(_obj, (char*) key);
    return rslt;
  };
  object get (const std::string& key) {
    return get(key.c_str());
  };
  object get (char key) {
    return get(&key);
  };
  
  //-------------------------------------------------------------------------
  // operator[] -- object and numeric versions
  //-------------------------------------------------------------------------  
  keyed_ref operator [] (object& key) {
    object rslt = PyDict_GetItem(_obj, key);
    if (!(PyObject*)rslt)
        PyErr_Clear(); // Ignore key errors
    return keyed_ref(rslt, *this, key);
  };
  keyed_ref operator [] (int key) {
    object _key = object(key);
    return operator [](_key);
  };
  keyed_ref operator [] (double key) {
    object _key = object(key);
    return operator [](_key);
  };
  keyed_ref operator [] (const std::complex<double>& key) {
    object _key = object(key);
    return operator [](_key);
  };
  
  //-------------------------------------------------------------------------
  // operator[] non-const -- string versions
  //-------------------------------------------------------------------------
  keyed_ref operator [] (const char* key) {
    object rslt = PyDict_GetItemString(_obj, (char*) key);
    if (!(PyObject*)rslt)
        PyErr_Clear(); // Ignore key errors
    object _key = key;    
    return keyed_ref(rslt, *this, _key);
  };
  keyed_ref operator [] (const std::string& key) {
    return operator [](key.c_str());
  };

  keyed_ref operator [] (char key) {
    return operator [](&key);
  };

  //-------------------------------------------------------------------------
  // has_key -- object and numeric versions
  //-------------------------------------------------------------------------  
  bool has_key(object& key) const {
    return PyMapping_HasKey(_obj, key)==1;
  };
  bool has_key(int key) const {
    object _key = key;    
    return has_key(_key);
  };
  bool has_key(double key) const {
    object _key = key;    
    return has_key(_key);
  };
  bool has_key(const std::complex<double>& key) const {
    object _key = key;    
    return has_key(_key);
  };

  //-------------------------------------------------------------------------
  // has_key -- string versions
  //-------------------------------------------------------------------------
  bool has_key(const char* key) const {
    return PyMapping_HasKeyString(_obj, (char*) key)==1;
  };
  bool has_key(const std::string& key) const {
    return has_key(key.c_str());
  };
  bool has_key(char key) const {
    return has_key(&key);
  };

  //-------------------------------------------------------------------------
  // len and length methods
  //-------------------------------------------------------------------------  
  int len() const {
    return PyDict_Size(_obj);
  }  
  int length() const {
    return PyDict_Size(_obj);
  };

  //-------------------------------------------------------------------------
  // set_item
  //-------------------------------------------------------------------------
  virtual void set_item(const char* key, object& val) {
    int rslt = PyDict_SetItemString(_obj, (char*) key, val);
    if (rslt==-1)
      fail(PyExc_RuntimeError, "Cannot add key / value");
  };

  virtual void set_item(object& key, object& val) const {
    int rslt = PyDict_SetItem(_obj, key, val);
    if (rslt==-1)
      fail(PyExc_KeyError, "Key must be hashable");
  };

  //-------------------------------------------------------------------------
  // clear
  //-------------------------------------------------------------------------  
  void clear() {
    PyDict_Clear(_obj);
  };
  
  //-------------------------------------------------------------------------
  // update
  //-------------------------------------------------------------------------  
#if PY_VERSION_HEX >= 0x02020000
  void update(dict& other) {
    PyDict_Merge(_obj,other,1);
  };
#endif
  //-------------------------------------------------------------------------
  // del -- remove key from dictionary
  //        overloaded to take all common weave types
  //-------------------------------------------------------------------------
  void del(object& key) {
    int rslt = PyDict_DelItem(_obj, key);
    if (rslt==-1)
      fail(PyExc_KeyError, "Key not found");
  };
  void del(int key) {
    object _key = key;
    del(_key);
  };
  void del(double key) {
    object _key = key;
    del(_key);
  };
  void del(const std::complex<double>& key) {
    object _key = key;
    del(_key);
  };
  void del(const char* key) {
    int rslt = PyDict_DelItemString(_obj, (char*) key);
    if (rslt==-1)
      fail(PyExc_KeyError, "Key not found");
  };
  void del(const std::string key) {
    del(key.c_str());
  };

  //-------------------------------------------------------------------------
  // items, keys, and values
  //-------------------------------------------------------------------------
  list items() const {
    PyObject* rslt = PyDict_Items(_obj);
    if (rslt==0)
      fail(PyExc_RuntimeError, "failed to get items");
    return lose_ref(rslt);
  };

  list keys() const {
    PyObject* rslt = PyDict_Keys(_obj);
    if (rslt==0)
      fail(PyExc_RuntimeError, "failed to get keys");
    return lose_ref(rslt);
  };

  list values() const {
    PyObject* rslt = PyDict_Values(_obj);
    if (rslt==0)
      fail(PyExc_RuntimeError, "failed to get values");
    return lose_ref(rslt);
  };
};

} // namespace
#endif // DICT_H_INCLUDED_
