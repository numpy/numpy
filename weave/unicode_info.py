import base_info

unicode_support_code = \
"""
class unicode_handler
{
public:
    Py_UNICODE* convert_to_unicode(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_conversion_error(py_obj,"unicode", name);
        Py_INCREF(py_obj);
        return PyUnicode_AS_UNICODE(py_obj);        
    }
    Py_UNICODE* py_to_unicode(PyObject* py_obj,const char* name)
    {
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_bad_type(py_obj,"string", name);
        Py_INCREF(py_obj);
        return PyUnicode_AS_UNICODE(py_obj);        
    }
};

unicode_handler x__unicode_handler = unicode_handler();

#define convert_to_unicode(py_obj,name) x__unicode_handler.convert_to_unicode(py_obj,name)
#define py_to_unicode(py_obj,name) x__unicode_handler.py_to_unicode(py_obj,name)
"""

class unicode_info(base_info.base_info):
    _support_code = [unicode_support_code]
