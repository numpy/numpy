/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
*********************************************/
#include "PWOSequence.h"
#include "PWOMSequence.h"
#include "PWOMapping.h"
#include "PWOCallable.h"
#include "PWONumber.h"

  // incref new owner, and decref old owner, and adjust to new owner
void PWOBase::GrabRef(PyObject* newObj)
{
    // be careful to incref before decref if old is same as new
  Py_XINCREF(newObj);
  Py_XDECREF(_own);
  _own = _obj = newObj;
}

bool PWOSequence::in(int value) {
  PWOBase val = PWONumber(value);
  return in(val);
};
  
bool PWOSequence::in(double value) {
  PWOBase val = PWONumber(value);
  return in(val);
};

bool PWOSequence::in(char* value) {
  PWOBase val = PWOString(value);
  return in(val);
};

bool PWOSequence::in(std::string value) {
  PWOBase val = PWOString(value.c_str());
  return in(val);
};
  
int PWOSequence::count(int value) const {
  PWONumber val = PWONumber(value);
  return count(val);
};

int PWOSequence::count(double value) const {
  PWONumber val = PWONumber(value);
  return count(val);
};

int PWOSequence::count(char* value) const {
  PWOString val = PWOString(value);
  return count(val);
};

int PWOSequence::count(std::string value) const {
  PWOString val = PWOString(value.c_str());
  return count(val);
};

int PWOSequence::index(int value) const {
  PWONumber val = PWONumber(value);
  return index(val);
};

int PWOSequence::index(double value) const {
  PWONumber val = PWONumber(value);
  return index(val);
};
int PWOSequence::index(char* value) const {
  PWOString val = PWOString(value);
  return index(val);
};

int PWOSequence::index(std::string value) const {
  PWOString val = PWOString(value.c_str());
  return index(val);
};

PWOTuple::PWOTuple(const PWOList& list)
  : PWOSequence (PyList_AsTuple(list)) { LoseRef(_obj); }
  
PWOList& PWOList::insert(int ndx, int other) {
  PWONumber oth = PWONumber(other);
  return insert(ndx, oth);
};

PWOList& PWOList::insert(int ndx, double other) {
  PWONumber oth = PWONumber(other);
  return insert(ndx, oth);
};

PWOList& PWOList::insert(int ndx, char* other) {
  PWOString oth = PWOString(other);
  return insert(ndx, oth);
};

PWOList& PWOList::insert(int ndx, std::string other) {
  PWOString oth = PWOString(other.c_str());
  return insert(ndx, oth);
};

PWOListMmbr::PWOListMmbr(PyObject* obj, PWOList& parent, int ndx) 
  : PWOBase(obj), _parent(parent), _ndx(ndx) { }

PWOListMmbr& PWOListMmbr::operator=(const PWOBase& other) {
  GrabRef(other);
  //Py_XINCREF(_obj); // this one is for setItem to steal
  _parent.setItem(_ndx, *this);
  return *this;
}

PWOListMmbr& PWOListMmbr::operator=(const PWOListMmbr& other) {
  GrabRef(other);
  //Py_XINCREF(_obj); // this one is for setItem to steal
  _parent.setItem(_ndx, *this);
  return *this;
}

PWOListMmbr& PWOListMmbr::operator=(int other) {
  GrabRef(PWONumber(other));
  _parent.setItem(_ndx, *this);
  return *this;
}

PWOListMmbr& PWOListMmbr::operator=(double other) {
  GrabRef(PWONumber(other));
  _parent.setItem(_ndx, *this);
  return *this;
}

PWOListMmbr& PWOListMmbr::operator=(const char* other) {
  GrabRef(PWOString(other));
  _parent.setItem(_ndx, *this);
  return *this;
}

PWOListMmbr& PWOListMmbr::operator=(std::string other) {
  GrabRef(PWOString(other.c_str()));
  _parent.setItem(_ndx, *this);
  return *this;
}

PWOMappingMmbr& PWOMappingMmbr::operator=(const PWOBase& other) {
  GrabRef(other);
  _parent.setItem(_key, *this);
  return *this;
}

PWOMappingMmbr& PWOMappingMmbr::operator=(int other) {
  GrabRef(PWONumber(other));
  _parent.setItem(_key, *this);
  return *this;
}

PWOMappingMmbr& PWOMappingMmbr::operator=(double other) {
  GrabRef(PWONumber(other));
  _parent.setItem(_key, *this);
  return *this;
}

PWOMappingMmbr& PWOMappingMmbr::operator=(const char* other) {
  GrabRef(PWOString(other));
  _parent.setItem(_key, *this);
  return *this;
}

PWOMappingMmbr& PWOMappingMmbr::operator=(std::string other) {
  GrabRef(PWOString(other.c_str()));
  _parent.setItem(_key, *this);
  return *this;
}

PWOBase PWOCallable::call() const {
  static PWOTuple _empty;
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, _empty, NULL);
  if (rslt == 0)
    throw 1;
  return rslt;
}
PWOBase PWOCallable::call(PWOTuple& args) const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, NULL);
  if (rslt == 0)
    throw 1;
  return rslt;
}
PWOBase PWOCallable::call(PWOTuple& args, PWOMapping& kws) const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, kws);
  if (rslt == 0)
    throw 1;
  return rslt;
}

void Fail(PyObject* exc, const char* msg)
{
  PyErr_SetString(exc, msg);
  throw 1;
}
