import base_info, common_info

string_support_code = \
"""
class string_handler
{
public:
    static Py::String convert_to_string(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyString_Check(py_obj))
            handle_conversion_error(py_obj,"string", name);
        return Py::String(py_obj);
    }
    Py::String py_to_string(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyString_Check(py_obj))
            handle_bad_type(py_obj,"string", name);
        return Py::String(py_obj);
    }
};

string_handler x__string_handler = string_handler();

#define convert_to_string(py_obj,name) x__string_handler.convert_to_string(py_obj,name)
#define py_to_string(py_obj,name) x__string_handler.py_to_string(py_obj,name)
"""

list_support_code = \
"""

class list_handler
{
public:
    Py::List convert_to_list(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyList_Check(py_obj))
            handle_conversion_error(py_obj,"list", name);
        return Py::List(py_obj);
    }
    Py::List py_to_list(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyList_Check(py_obj))
            handle_bad_type(py_obj,"list", name);
        return Py::List(py_obj);
    }
};

list_handler x__list_handler = list_handler();
#define convert_to_list(py_obj,name) x__list_handler.convert_to_list(py_obj,name)
#define py_to_list(py_obj,name) x__list_handler.py_to_list(py_obj,name)

"""

dict_support_code = \
"""
class dict_handler
{
public:
    Py::Dict convert_to_dict(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyDict_Check(py_obj))
            handle_conversion_error(py_obj,"dict", name);
        return Py::Dict(py_obj);
    }
    Py::Dict py_to_dict(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyDict_Check(py_obj))
            handle_bad_type(py_obj,"dict", name);
        return Py::Dict(py_obj);
    }
};

dict_handler x__dict_handler = dict_handler();
#define convert_to_dict(py_obj,name) x__dict_handler.convert_to_dict(py_obj,name)
#define py_to_dict(py_obj,name) x__dict_handler.py_to_dict(py_obj,name)

"""

tuple_support_code = \
"""
class tuple_handler
{
public:
    Py::Tuple convert_to_tuple(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_conversion_error(py_obj,"tuple", name);
        return Py::Tuple(py_obj);
    }
    Py::Tuple py_to_tuple(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_bad_type(py_obj,"tuple", name);
        return Py::Tuple(py_obj);
    }
};

tuple_handler x__tuple_handler = tuple_handler();
#define convert_to_tuple(py_obj,name) x__tuple_handler.convert_to_tuple(py_obj,name)
#define py_to_tuple(py_obj,name) x__tuple_handler.py_to_tuple(py_obj,name)
"""

import os, cxx_info
local_dir,junk = os.path.split(os.path.abspath(cxx_info.__file__))   
cxx_dir = os.path.join(local_dir,'CXX')

class cxx_info(base_info.base_info):
    _headers = ['"CXX/Objects.hxx"','"CXX/Extensions.hxx"','<algorithm>']
    _include_dirs = [local_dir]

    # should these be built to a library??
    _sources = [os.path.join(cxx_dir,'cxxsupport.cxx'),
                os.path.join(cxx_dir,'cxx_extensions.cxx'),
                os.path.join(cxx_dir,'IndirectPythonInterface.cxx'),
                os.path.join(cxx_dir,'cxxextensions.c')]
    _support_code = [string_support_code,list_support_code, dict_support_code,
                     tuple_support_code]
