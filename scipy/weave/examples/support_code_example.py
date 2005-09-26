import sys
sys.path.insert(0,'..')
import inline_tools


support_code = """
               PyObject* length(std::string a)
               {
                   int l = a.length();
                   return PyInt_FromLong(l);
               }
               """
a='some string'
val = inline_tools.inline("return_val = length(a);",['a'],
                          support_code=support_code)
print val

               