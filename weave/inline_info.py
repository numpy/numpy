get_variable_support_code = \
"""
void handle_variable_not_found(char*  var_name)
{
    char msg[500];
    sprintf(msg,"variable '%s' not found in local or global scope.",var_name);
    throw_error(PyExc_NameError,msg);
}
PyObject* get_variable(char* name,PyObject* locals, PyObject* globals)
{
    // no checking done for error -- locals and globals should
    // already be validated as dictionaries.  If var is NULL, the
    // function calling this should handle it.
    PyObject* var = NULL;
    var = PyDict_GetItemString(locals,name);
    if (!var)
    {
        var = PyDict_GetItemString(globals,name);
    }
    if (!var)
        handle_variable_not_found(name);
    return var;
}
"""

py_to_raw_dict_support_code = \
"""
PyObject* py_to_raw_dict(PyObject* py_obj, char* name)
{
    // simply check that the value is a valid dictionary pointer.
    if(!py_obj || !PyDict_Check(py_obj))
        handle_bad_type(py_obj, "dictionary", name);
    return py_obj;
}
"""

import base_info
class inline_info(base_info.base_info):
    _support_code = [get_variable_support_code, py_to_raw_dict_support_code]
