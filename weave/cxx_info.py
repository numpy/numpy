import base_info, common_info

string_support_code = \
"""
class string_handler
{
    static Py::String convert_to_string(PyObject* py_obj,char* name)
    {
        if (!py_obj || !PyString_Check(py_obj))
            handle_conversion_error(py_obj,"string", name);
        return Py::String(py_obj);
    }
};
static Py::String py_to_string(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyString_Check(py_obj))
        handle_bad_type(py_obj,"string", name);
    return Py::String(py_obj);
}

"""

list_support_code = \
"""

class list_handler
{
    Py::List convert_to_list(PyObject* py_obj,char* name)
    {
        if (!py_obj || !PyList_Check(py_obj))
            handle_conversion_error(py_obj,"list", name);
        return Py::List(py_obj);
    }
};

static Py::List py_to_list(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyList_Check(py_obj))
        handle_bad_type(py_obj,"list", name);
    return Py::List(py_obj);
}
"""

dict_support_code = \
"""
class dict_handler
{
    Py::Dict convert_to_dict(PyObject* py_obj,char* name)
    {
        if (!py_obj || !PyDict_Check(py_obj))
            handle_conversion_error(py_obj,"dict", name);
        return Py::Dict(py_obj);
    }
}

static Py::Dict py_to_dict(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyDict_Check(py_obj))
        handle_bad_type(py_obj,"dict", name);
    return Py::Dict(py_obj);
}
"""

tuple_support_code = \
"""
class tuple_handler
{
    Py::Tuple convert_to_tuple(PyObject* py_obj,char* name)
    {
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_conversion_error(py_obj,"tuple", name);
        return Py::Tuple(py_obj);
    }
};

static Py::Tuple py_to_tuple(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyTuple_Check(py_obj))
        handle_bad_type(py_obj,"tuple", name);
    return Py::Tuple(py_obj);
}
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
