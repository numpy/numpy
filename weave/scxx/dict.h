/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
  
  modified for weave by eric jones
*********************************************/

#if !defined(DICT_H_INCLUDED_)
#define DICT_H_INCLUDED_

#include "object.h"
#include "number.h"
#include "list.h"
#include "str.h"
#include <string>

namespace py {

class dict;
    
class dict_member : public object
{
  dict& _parent;
  PyObject* _key;
public:
  dict_member(PyObject* obj, dict& parent, PyObject* key)
    : object(obj), _parent(parent), _key(key)
  {
    Py_XINCREF(_key);
  };
  virtual ~dict_member() {
    Py_XDECREF(_key);
  };
  dict_member& operator=(const object& other);
  dict_member& operator=(int other);
  dict_member& operator=(double other);
  dict_member& operator=(const char* other);
  dict_member& operator=(std::string other);
};

class dict : public object
{
public:
  dict() : object (PyDict_New()) { LoseRef(_obj); }
  dict(const dict& other) : object(other) {};
  dict(PyObject* obj) : object(obj) {
    _violentTypeCheck();
  };
  virtual ~dict() {};

  virtual dict& operator=(const dict& other) {
    GrabRef(other);
    return *this;
  };
  dict& operator=(const object& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyMapping_Check(_obj)) {
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a mapping");
    }
  };

  //PyMapping_GetItemString
  //PyDict_GetItemString
  dict_member operator [] (const char* key) {
    PyObject* rslt = PyMapping_GetItemString(_obj, (char*) key);
    if (rslt==0)
      PyErr_Clear();
    // ?? why do I need py:: here?
    str _key(key);
    return dict_member(rslt, *this, _key);
  };

  dict_member operator [] (std::string key) {
    PyObject* rslt = PyMapping_GetItemString(_obj, (char*) key.c_str());
    if (rslt==0)
      PyErr_Clear();
    str _key(key.c_str());
    return dict_member(rslt, *this, _key);
  };

  //PyDict_GetItem
  dict_member operator [] (PyObject* key) {
    PyObject* rslt = PyDict_GetItem(_obj, key);
    //if (rslt==0)
    //  Fail(PyExc_KeyError, "Key not found");
    return dict_member(rslt, *this, key);
  };

  //PyDict_GetItem
  dict_member operator [] (int key) {
    number _key = number(key);
    PyObject* rslt = PyDict_GetItem(_obj, _key);
    //if (rslt==0)
    //  Fail(PyExc_KeyError, "Key not found");
    return dict_member(rslt, *this, _key);
  };
  
  dict_member operator [] (double key) {
    number _key = number(key);
    PyObject* rslt = PyDict_GetItem(_obj, _key);
    //if (rslt==0)
    //  Fail(PyExc_KeyError, "Key not found");
    return dict_member(rslt, *this, _key);
  };
  
  //PyMapping_HasKey
  bool has_key(PyObject* key) const {
    return PyMapping_HasKey(_obj, key)==1;
  };
  //PyMapping_HasKeyString
  bool has_key(const char* key) const {
    return PyMapping_HasKeyString(_obj, (char*) key)==1;
  };
  //PyMapping_Length
  //PyDict_Size
  int len() const {
    return PyMapping_Length(_obj);
  }  
  int length() const {
    return PyMapping_Length(_obj);
  };
  //PyMapping_SetItemString
  //PyDict_SetItemString
  void set_item(const char* key, PyObject* val) {
    int rslt = PyMapping_SetItemString(_obj, (char*) key, val);
    if (rslt==-1)
      Fail(PyExc_RuntimeError, "Cannot add key / value");
  };
  //PyDict_SetItem
  void set_item(PyObject* key, PyObject* val) const {
    int rslt = PyDict_SetItem(_obj, key, val);
    if (rslt==-1)
      Fail(PyExc_KeyError, "Key must be hashable");
  };
  //PyDict_Clear
  void clear() {
    PyDict_Clear(_obj);
  };
  //PyDict_DelItem
  void del(PyObject* key) {
    int rslt = PyMapping_DelItem(_obj, key);
    if (rslt==-1)
      Fail(PyExc_KeyError, "Key not found");
  };
  //PyDict_DelItemString
  void del(const char* key) {
    int rslt = PyDict_DelItemString(_obj, (char*) key);
    if (rslt==-1)
      Fail(PyExc_KeyError, "Key not found");
  };
  //PyDict_Items
  list items() const {
    PyObject* rslt = PyMapping_Items(_obj);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "Failed to get items");
    return LoseRef(rslt);
  };
  //PyDict_Keys
  list keys() const {
    PyObject* rslt = PyMapping_Keys(_obj);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "Failed to get keys");
    return LoseRef(rslt);
  };
  //PyDict_New - default constructor
  //PyDict_Next
  //PyDict_Values
  list values() const {
    PyObject* rslt = PyMapping_Values(_obj);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "Failed to get values");
    return LoseRef(rslt);
  };
};

} // namespace
#endif // DICT_H_INCLUDED_
