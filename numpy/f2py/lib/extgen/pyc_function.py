
from base import Base

class PyCFunction(Base):

    """
    PyCFunction(<name>, *components, provides=..)
    
    """

    container_options = dict(FuncDoc=dict(separator='"\n"', prefix='"', suffix='"'),
                             Args = dict(),
                             Decl = dict(default='<KILLLINE>', use_indent=True),
                             KWList = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True),
                             PyArgFormat = dict(separator=''),
                             PyArgObj = dict(separator=', ', prefix=', ', skip_prefix_when_empty=True),
                             FromPyObj = dict(default='<KILLLINE>', use_indent=True),
                             Exec = dict(default='<KILLLINE>', use_indent=True),
                             PyObjFrom = dict(default='<KILLLINE>', use_indent=True),
                             RetFormat = dict(separator=''),
                             RetObj = dict(separator=', ', prefix=', ', skip_prefix_when_empty=True),
                             CleanPyObjFrom = dict(default='<KILLLINE>', reverse=True, use_indent=True),
                             CleanExec = dict(default='<KILLLINE>', reverse=True, use_indent=True),
                             CleanFromPyObj = dict(default='<KILLLINE>', reverse=True, use_indent=True),
                             )

    component_container_map = dict(CCode = 'Exec',
                                   PyCArgument = 'Args')

    template = '''
static char %(pyc_name)s_doc[] = %(FuncDoc)s;

static PyObject*
%(pyc_name)s
(PyObject *pyc_self, PyObject *pyc_args, PyObject *pyc_keywds) {
  PyObject * volatile pyc_buildvalue = NULL;
  volatile int capi_success = 1;
  %(Decl)s
  static char *capi_kwlist[] = {%(KWList)sNULL};
  if (PyArg_ParseTupleAndKeywords(pyc_args, pyc_keywds,"%(PyArgFormat)s", capi_kwlist%(PyArgObj)s)) {
    %(FromPyObj)s
    %(Exec)s
    capi_success = !PyErr_Occurred();
    if (capi_success) {
      %(PyObjFrom)s
      pyc_buildvalue = Py_BuildValue("%(RetFormat)s"%(RetObj)s);
      %(CleanPyObjFrom)s
    }
    %(CleanExec)s
    %(CleanFromPyObj)s
  }
  return pyc_buildvalue;
}
'''

    def initialize(self, name, *components, **options):
        self.name = name
        self.pyc_name = 'pyc_function_'+name
        self._provides = options.get('provides',
                                     '%s_%s' % (self.__class__.__name__, name))
        map(self.add, components)

    def init_containers(self):
        # set header to FuncDoc, for example.
        FuncDoc = self.get_container('FuncDoc')
        FuncDoc.add(self.name)
        return

    def update_containers(self, params=None):
        ModuleMethod = self.get_container('ModuleMethod')
        t = '{"%(name)s", (PyCFunction)%(pyc_name)s,\n METH_VARARGS | METH_KEYWORDS, %(pyc_name)s_doc}'
        ModuleMethod.add(self.evaluate(t), self.name)
        return



    
