/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
*********************************************/
#if !defined(PWOMAPPING_H_INCLUDED_)
#define PWOMAPPING_H_INCLUDED_

#include "PWOBase.h"
#include "PWOMSequence.h"

class PWOMapping;

class PWOMappingMmbr : public PWOBase
{
  PWOMapping& _parent;
  PyObject* _key;
public:
  PWOMappingMmbr(PyObject* obj, PWOMapping& parent, PyObject* key)
    : PWOBase(obj), _parent(parent), _key(key)
  {
    Py_XINCREF(_key);
  };
  virtual ~PWOMappingMmbr() {
    Py_XDECREF(_key);
  };
  PWOMappingMmbr& operator=(const PWOBase& other);
};

class PWOMapping : public PWOBase
{
public:
  PWOMapping() : PWOBase (PyDict_New()) { LoseRef(_obj); }
  PWOMapping(const PWOMapping& other) : PWOBase(other) {};
  PWOMapping(PyObject* obj) : PWOBase(obj) {
    _violentTypeCheck();
  };
  virtual ~PWOMapping() {};

  virtual PWOMapping& operator=(const PWOMapping& other) {
    GrabRef(other);
    return *this;
  };
  PWOMapping& operator=(const PWOBase& other) {
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
  PWOMappingMmbr operator [] (const char* key) {
    PyObject* rslt = PyMapping_GetItemString(_obj, (char*) key);
    if (rslt==0)
      PyErr_Clear();
    PWOString _key(key);
    return PWOMappingMmbr(rslt, *this, _key);
  };
  //PyDict_GetItem
  PWOMappingMmbr operator [] (PyObject* key) {
    PyObject* rslt = PyDict_GetItem(_obj, key);
    //if (rslt==0)
    //  Fail(PyExc_KeyError, "Key not found");
    return PWOMappingMmbr(rslt, *this, key);
  };
  //PyMapping_HasKey
  bool hasKey(PyObject* key) const {
    return PyMapping_HasKey(_obj, key)==1;
  };
  //PyMapping_HasKeyString
  bool hasKey(const char* key) const {
    return PyMapping_HasKeyString(_obj, (char*) key)==1;
  };
  //PyMapping_Length
  //PyDict_Size
  int len() const {
    return PyMapping_Length(_obj);
  };
  //PyMapping_SetItemString
  //PyDict_SetItemString
  void setItem(const char* key, PyObject* val) {
    int rslt = PyMapping_SetItemString(_obj, (char*) key, val);
    if (rslt==-1)
      Fail(PyExc_RuntimeError, "Cannot add key / value");
  };
  //PyDict_SetItem
  void setItem(PyObject* key, PyObject* val) const {
    int rslt = PyDict_SetItem(_obj, key, val);
    if (rslt==-1)
      Fail(PyExc_KeyError, "Key must be hashable");
  };
  //PyDict_Clear
  void clear() {
    PyDict_Clear(_obj);
  };
  //PyDict_DelItem
  void delItem(PyObject* key) {
    int rslt = PyMapping_DelItem(_obj, key);
    if (rslt==-1)
      Fail(PyExc_KeyError, "Key not found");
  };
  //PyDict_DelItemString
  void delItem(const char* key) {
    int rslt = PyDict_DelItemString(_obj, (char*) key);
    if (rslt==-1)
      Fail(PyExc_KeyError, "Key not found");
  };
  //PyDict_Items
  PWOList items() const {
    PyObject* rslt = PyMapping_Items(_obj);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "Failed to get items");
    return LoseRef(rslt);
  };
  //PyDict_Keys
  PWOList keys() const {
    PyObject* rslt = PyMapping_Keys(_obj);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "Failed to get keys");
    return LoseRef(rslt);
  };
  //PyDict_New - default constructor
  //PyDict_Next
  //PyDict_Values
  PWOList values() const {
    PyObject* rslt = PyMapping_Values(_obj);
    if (rslt==0)
      Fail(PyExc_RuntimeError, "Failed to get values");
    return LoseRef(rslt);
  };
};

#endif // PWOMAPPING_H_INCLUDED_
