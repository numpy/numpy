/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
*********************************************/
#include "sequence.h"
#include "list.h"
#include "tuple.h"
#include "str.h"
#include "dict.h"
#include "callable.h"
#include "number.h"

using namespace py;
    
//---------------------------------------------------------------------------
// object
//---------------------------------------------------------------------------

// incref new owner, and decref old owner, and adjust to new owner
void object::GrabRef(PyObject* newObj)
{
    // be careful to incref before decref if old is same as new
  Py_XINCREF(newObj);
  Py_XDECREF(_own);
  _own = _obj = newObj;
}

object object::mcall(const char* nm)
{
  object method = attr(nm);
  PyObject* result = PyEval_CallObjectWithKeywords(method,NULL,NULL);
  if (!result)
    throw 1; // signal exception has occured.
  return object(LoseRef(result));
}

object object::mcall(const char* nm, tuple& args)
{
  object method = attr(nm);
  PyObject* result = PyEval_CallObjectWithKeywords(method,args,NULL);
  if (!result)
    throw 1; // signal exception has occured.
  return object(LoseRef(result));
}

object object::mcall(const char* nm, tuple& args, dict& kwargs)
{
  object method = attr(nm);
  PyObject* result = PyEval_CallObjectWithKeywords(method,args,kwargs);
  if (!result)
    throw 1; // signal exception has occured.
  return object(LoseRef(result));
}

object object::call() const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, NULL, NULL);
  if (rslt == 0)
    throw 1;
  return object(LoseRef(rslt));
}
object object::call(tuple& args) const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, NULL);
  if (rslt == 0)
    throw 1;
  return object(LoseRef(rslt));
}
object object::call(tuple& args, dict& kws) const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, kws);
  if (rslt == 0)
    throw 1;
  return object(LoseRef(rslt));
}

//---------------------------------------------------------------------------
// sequence
//---------------------------------------------------------------------------

bool sequence::in(int value) {
  object val = number(value);
  return in(val);
};
  
bool sequence::in(double value) {
  object val = number(value);
  return in(val);
};

bool sequence::in(char* value) {
  object val = str(value);
  return in(val);
};

bool sequence::in(std::string value) {
  object val = str(value.c_str());
  return in(val);
};
  
int sequence::count(int value) const {
  number val = number(value);
  return count(val);
};

int sequence::count(double value) const {
  number val = number(value);
  return count(val);
};

int sequence::count(char* value) const {
  str val = str(value);
  return count(val);
};

int sequence::count(std::string value) const {
  str val = str(value.c_str());
  return count(val);
};

int sequence::index(int value) const {
  number val = number(value);
  return index(val);
};

int sequence::index(double value) const {
  number val = number(value);
  return index(val);
};
int sequence::index(char* value) const {
  str val = str(value);
  return index(val);
};

int sequence::index(std::string value) const {
  str val = str(value.c_str());
  return index(val);
};

//---------------------------------------------------------------------------
// tuple
//---------------------------------------------------------------------------

tuple::tuple(const list& lst)
  : sequence (PyList_AsTuple(lst)) { LoseRef(_obj); }

//---------------------------------------------------------------------------
// tuple_member
//---------------------------------------------------------------------------

tuple_member::tuple_member(PyObject* obj, tuple& parent, int ndx) 
  : object(obj), _parent(parent), _ndx(ndx) { }

tuple_member& tuple_member::operator=(const object& other) {
  GrabRef(other);
  //Py_XINCREF(_obj); // this one is for set_item to steal
  _parent.set_item(_ndx, *this);
  return *this;
}

tuple_member& tuple_member::operator=(const tuple_member& other) {
  GrabRef(other);
  //Py_XINCREF(_obj); // this one is for set_item to steal
  _parent.set_item(_ndx, *this);
  return *this;
}

tuple_member& tuple_member::operator=(int other) {
  GrabRef(number(other));
  _parent.set_item(_ndx, *this);
  return *this;
}

tuple_member& tuple_member::operator=(double other) {
  GrabRef(number(other));
  _parent.set_item(_ndx, *this);
  return *this;
}

tuple_member& tuple_member::operator=(const char* other) {
  GrabRef(str(other));
  _parent.set_item(_ndx, *this);
  return *this;
}

tuple_member& tuple_member::operator=(std::string other) {
  GrabRef(str(other.c_str()));
  _parent.set_item(_ndx, *this);
  return *this;
}
//---------------------------------------------------------------------------
// list
//---------------------------------------------------------------------------
  
list& list::insert(int ndx, int other) {
  number oth = number(other);
  return insert(ndx, oth);
};

list& list::insert(int ndx, double other) {
  number oth = number(other);
  return insert(ndx, oth);
};

list& list::insert(int ndx, char* other) {
  str oth = str(other);
  return insert(ndx, oth);
};

list& list::insert(int ndx, std::string other) {
  str oth = str(other.c_str());
  return insert(ndx, oth);
};

//---------------------------------------------------------------------------
// list_member
//---------------------------------------------------------------------------

list_member::list_member(PyObject* obj, list& parent, int ndx) 
  : object(obj), _parent(parent), _ndx(ndx) { }

list_member& list_member::operator=(const object& other) {
  GrabRef(other);
  //Py_XINCREF(_obj); // this one is for set_item to steal
  _parent.set_item(_ndx, *this);
  return *this;
}

list_member& list_member::operator=(const list_member& other) {
  GrabRef(other);
  //Py_XINCREF(_obj); // this one is for set_item to steal
  _parent.set_item(_ndx, *this);
  return *this;
}

list_member& list_member::operator=(int other) {
  GrabRef(number(other));
  _parent.set_item(_ndx, *this);
  return *this;
}

list_member& list_member::operator=(double other) {
  GrabRef(number(other));
  _parent.set_item(_ndx, *this);
  return *this;
}

list_member& list_member::operator=(const char* other) {
  GrabRef(str(other));
  _parent.set_item(_ndx, *this);
  return *this;
}

list_member& list_member::operator=(std::string other) {
  GrabRef(str(other.c_str()));
  _parent.set_item(_ndx, *this);
  return *this;
}

//---------------------------------------------------------------------------
// dict_member
//---------------------------------------------------------------------------

dict_member& dict_member::operator=(const object& other) {
  GrabRef(other);
  _parent.set_item(_key, *this);
  return *this;
}

dict_member& dict_member::operator=(int other) {
  GrabRef(number(other));
  _parent.set_item(_key, *this);
  return *this;
}

dict_member& dict_member::operator=(double other) {
  GrabRef(number(other));
  _parent.set_item(_key, *this);
  return *this;
}

dict_member& dict_member::operator=(const char* other) {
  GrabRef(str(other));
  _parent.set_item(_key, *this);
  return *this;
}

dict_member& dict_member::operator=(std::string other) {
  GrabRef(str(other.c_str()));
  _parent.set_item(_key, *this);
  return *this;
}

//---------------------------------------------------------------------------
// callable
//---------------------------------------------------------------------------

object callable::call() const {
  static tuple _empty;
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, _empty, NULL);
  if (rslt == 0)
    throw 1;
  return rslt;
}
object callable::call(tuple& args) const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, NULL);
  if (rslt == 0)
    throw 1;
  return rslt;
}
object callable::call(tuple& args, dict& kws) const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, kws);
  if (rslt == 0)
    throw 1;
  return rslt;
}

void py::Fail(PyObject* exc, const char* msg)
{
  PyErr_SetString(exc, msg);
  throw 1;
}