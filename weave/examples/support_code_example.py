import sys
sys.path.insert(0,'..')
import inline_tools


support_code = """
               PyObject* length(Py::String a)
               {
                   int l = a.length();
                   return Py::new_reference_to(Py::Int(l));
               }
               """
a='some string'
val = inline_tools.inline("return_val = length(a);",['a'],
                          support_code=support_code)
print val

               