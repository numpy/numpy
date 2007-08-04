
from base import Base

class PyCFunction(Base):

    """
    PyCFunction(<name>, *components, provides=..,title=.., description=..)

    >>> from __init__ import * #doctest: +ELLIPSIS
    Ignoring...
    >>> f = PyCFunction('hello', title='A function.', description='\\nFirst line.\\n2nd line.')
    >>> a1_in_doc = '''First line.\\nSecond line.'''
    >>> a1_out_doc = '''Single line.'''
    >>> f += PyCArgument('a1',output_intent='return', input_title='a Python object',
    ...    input_description=a1_in_doc, output_description=a1_out_doc)
    >>> f += PyCArgument('c1',input_intent='extra')
    >>> f += PyCArgument('b1',input_intent='optional')
    >>> f += PyCArgument('d2',input_intent='hide', output_intent='return')
    >>> f += PyCArgument('a2',input_intent='required')
    >>> f += PyCArgument('c2',input_intent='extra')
    >>> f += PyCArgument('b2',input_intent='optional')
    >>> m = ExtensionModule('foo', f)
    >>> foo = m.build() #doctest: +ELLIPSIS
    exec_command...
    >>> print foo.hello.__doc__
      hello(a1, a2 [, b1, b2, c1, c2]) -> (a1, d2)
    <BLANKLINE>
    A function.
    <BLANKLINE>
    Required arguments:
      a1 - a Python object
        First line.
        Second line.
      a2 - None
    <BLANKLINE>
    Optional arguments:
      b1 - None
      b2 - None
    <BLANKLINE>
    Extra optional arguments:
      c1 - None
      c2 - None
    <BLANKLINE>
    Return values:
      a1 - None
        Single line.
      d2 - None
    <BLANKLINE>
    Description:
    <BLANKLINE>
      First line.
      2nd line.
    """

    container_options = dict(\
        Args = dict(separator=', '),

        ReqArgs = dict(separator=', '),
        OptArgs = dict(separator=', '),
        ExtArgs = dict(separator=', '),
        RetArgs = dict(separator=', ', prefix='(', suffix=')', default = 'None',
                       skip_prefix_when_empty=True, skip_suffix_when_empty=True),
        
        OptExtArgs = dict(separator=', ', prefix=' [, ', skip_prefix_when_empty=True,
                          suffix=']', skip_suffix_when_empty=True),

        FuncTitle = dict(default='<KILLLINE>',prefix='"\\n\\n',suffix='"',separator='\\n"\n"  ',
                         skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                         use_firstline_indent=True),
        FuncDescr = dict(default='<KILLLINE>',prefix='"\\n\\nDescription:\\n"\n"  ',
                         suffix='"',separator='\\n"\n"  ',
                         skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                         use_firstline_indent=True),
        ReqArgsDoc = dict(default='<KILLLINE>', prefix='"\\n\\nRequired arguments:\\n"\n"  ',
                          separator='\\n"\n"  ', suffix='"',
                          skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                          use_firstline_indent=True),
        OptArgsDoc = dict(default='<KILLLINE>', prefix='"\\n\\nOptional arguments:\\n"\n"  ',
                          separator='\\n"\n"  ', suffix='"',
                          skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                          use_firstline_indent=True),
        ExtArgsDoc = dict(default='<KILLLINE>', prefix='"\\n\\nExtra optional arguments:\\n"\n"  ',
                          separator='\\n"\n"  ', suffix='"',
                          skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                          use_firstline_indent=True),
        RetDoc = dict(default='"Return value:\\n  None\\n"', prefix='"\\n\\nReturn values:\\n"\n"  ',
                    separator='\\n"\n"  ', suffix='"',
                      skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                      use_firstline_indent=True),
        
        Decl = dict(default='<KILLLINE>', use_indent=True),
        
        ReqKWList = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True),
        OptKWList = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True),
        ExtKWList = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True),
        
        ReqPyArgFmt = dict(separator=''),
        OptPyArgFmt = dict(separator=''),
        ExtPyArgFmt = dict(separator=''),
        
        ReqPyArgObj = dict(separator=', ', prefix=', ', skip_prefix_when_empty=True),
        OptPyArgObj = dict(separator=', ', prefix=', ', skip_prefix_when_empty=True),
        ExtPyArgObj = dict(separator=', ', prefix=', ', skip_prefix_when_empty=True),
        
        FromPyObj = dict(default='<KILLLINE>', use_indent=True),
        Exec = dict(default='<KILLLINE>', use_indent=True),
        PyObjFrom = dict(default='<KILLLINE>', use_indent=True),
        
        RetFmt = dict(separator=''),
        RetObj = dict(separator=', ', prefix=', ', skip_prefix_when_empty=True),

        CleanPyObjFrom = dict(default='<KILLLINE>', reverse=True, use_indent=True),
        CleanExec = dict(default='<KILLLINE>', reverse=True, use_indent=True),
        CleanFromPyObj = dict(default='<KILLLINE>', reverse=True, use_indent=True),
        )

    component_container_map = dict(CCode = 'Exec',
                                   PyCArgument = 'Args')

    template = '''
static char %(pyc_name)s_doc[] =
"  %(name)s(%(ReqArgs)s%(OptExtArgs)s) -> %(RetArgs)s"
%(FuncTitle)s
%(ReqArgsDoc)s
%(OptArgsDoc)s
%(ExtArgsDoc)s
%(RetDoc)s
%(FuncDescr)s
;

static PyObject*
%(pyc_name)s
(PyObject *pyc_self, PyObject *pyc_args, PyObject *pyc_keywds) {
  PyObject * volatile pyc_buildvalue = NULL;
  volatile int capi_success = 1;
  %(Decl)s
  static char *capi_kwlist[] = {%(ReqKWList)s%(OptKWList)s%(ExtKWList)sNULL};
  if (PyArg_ParseTupleAndKeywords(pyc_args, pyc_keywds,"%(ReqPyArgFmt)s%(OptPyArgFmt)s%(ExtPyArgFmt)s",
                                  capi_kwlist%(ReqPyArgObj)s%(OptPyArgObj)s%(ExtPyArgObj)s)) {
    %(FromPyObj)s
    %(Exec)s
    capi_success = !PyErr_Occurred();
    if (capi_success) {
      %(PyObjFrom)s
      pyc_buildvalue = Py_BuildValue("%(RetFmt)s"%(RetObj)s);
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
        self.title = options.get('title', None)
        self.description = options.get('description', None)
        map(self.add, components)

    def init_containers(self):
        return

    def update_containers(self):
        evaluate = self.evaluate

        # get containers
        FuncTitle = self.container_FuncTitle
        FuncDescr = self.container_FuncDescr
        ReqArgs = self.container_ReqArgs
        OptArgs = self.container_OptArgs
        ExtArgs = self.container_ExtArgs
        OptExtArgs = self.container_OptExtArgs
        ModuleMethod = self.container_ModuleMethod
        ModuleFuncDoc = self.container_ModuleFuncDoc

        # update ExtensionModule containers:
        t = '{"%(name)s", (PyCFunction)%(pyc_name)s,\n METH_VARARGS | METH_KEYWORDS, %(pyc_name)s_doc}'
        ModuleMethod.add(evaluate(t), self.name)

        # update local containers:
        OptExtArgs += OptArgs + ExtArgs
        ModuleFuncDoc += evaluate('%(name)s(%(ReqArgs)s%(OptExtArgs)s) -> %(RetArgs)s')
        if self.title is not None:
            FuncTitle += self.title.replace('\n','\\n')
            ModuleFuncDoc += '  ' + self.title.replace('\n','\\n')
        if self.description is not None:
            FuncDescr += self.description.replace('\n','\\n')
        return

    
def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()
