from types import UnicodeType
import unicode_info
from common_spec import common_base_converter

class unicode_converter(common_base_converter):
    type_name = 'unicode'
    _build_information = [unicode_info.unicode_info()]
    def type_match(self,value):
        return type(value) in [UnicodeType]

    def declaration_code(self,templatize = 0,inline=0):
        var_name = self.retrieve_py_variable(inline)
        code = 'Py_UNICODE* %s = convert_to_unicode(py_%s,"%s");\n' \
               'int _N%s = PyUnicode_GET_SIZE(py_%s);\n'   %     \
               (self.name,self.name,self.name,self.name,self.name)
        return code
    def cleanup_code(self):
        # could use Py_DECREF here I think and save NULL test.
        code = "Py_XDECREF(py_%s);\n" % self.name
        return code
