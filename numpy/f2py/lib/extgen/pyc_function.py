
from base import Component

class PyCFunction(Component):

    """
    PyCFunction(<name>, *components, provides=..,title=.., description=..)

    >>> from __init__ import * #doctest: +ELLIPSIS
    Ignoring...
    >>> f = PyCFunction('hello', title='A function.', description='\\nFirst line.\\n2nd line.')
    >>> a1_in_doc = '''First line.\\nSecond line.'''
    >>> a1_out_doc = '''Single line.'''
    >>> f += PyCArgument('a1',output_intent='return', input_title='anything',
    ...    input_description=a1_in_doc, output_description=a1_out_doc)
    >>> f += PyCArgument('c1',input_intent='extra')
    >>> f += PyCArgument('b1',input_intent='optional')
    >>> f += PyCArgument('d2',input_intent='hide', output_intent='return')
    >>> f += PyCArgument('a2',input_intent='required')
    >>> f += PyCArgument('c2',input_intent='extra')
    >>> f += PyCArgument('b2',input_intent='optional')
    >>> m = ExtensionModule('test_PyCFunction', f)
    >>> foo = m.build() #doctest: +ELLIPSIS
    exec_command...
    >>> print foo.hello.__doc__
      hello(a1, a2 [, b1, b2, c1, c2]) -> (a1, d2)
    <BLANKLINE>
    A function.
    <BLANKLINE>
    Required arguments:
      a1 - a python object, anything
        First line.
        Second line.
      a2 - a python object
    <BLANKLINE>
    Optional arguments:
      b1 - a python object
      b2 - a python object
    <BLANKLINE>
    Extra optional arguments:
      c1 - a python object
      c2 - a python object
    <BLANKLINE>
    Return values:
      a1 - a python object
        Single line.
      d2 - a python object
    <BLANKLINE>
    Description:
    <BLANKLINE>
      First line.
      2nd line.
    >>> print foo.hello(1, 2)
    (1, None)
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
                         use_firstline_indent=True, replace_map={'\n':'\\n'}),
        FuncDescr = dict(default='<KILLLINE>',prefix='"\\n\\nDescription:\\n"\n"  ',
                         suffix='"',separator='\\n"\n"  ',
                         skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                         use_firstline_indent=True, replace_map={'\n':'\\n'}),
        ReqArgsDoc = dict(default='<KILLLINE>', prefix='"\\n\\nRequired arguments:\\n"\n"  ',
                          separator='\\n"\n"  ', suffix='"',
                          skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                          use_firstline_indent=True, replace_map={'\n':'\\n'}),
        OptArgsDoc = dict(default='<KILLLINE>', prefix='"\\n\\nOptional arguments:\\n"\n"  ',
                          separator='\\n"\n"  ', suffix='"',
                          skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                          use_firstline_indent=True, replace_map={'\n':'\\n'}),
        ExtArgsDoc = dict(default='<KILLLINE>', prefix='"\\n\\nExtra optional arguments:\\n"\n"  ',
                          separator='\\n"\n"  ', suffix='"',
                          skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                          use_firstline_indent=True, replace_map={'\n':'\\n'}),
        RetDoc = dict(default='"Return value:\\n  None\\n"', prefix='"\\n\\nReturn values:\\n"\n"  ',
                    separator='\\n"\n"  ', suffix='"',
                      skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                      use_firstline_indent=True, replace_map={'\n':'\\n'}),
        
        Decl = dict(default='<KILLLINE>', use_indent=True),
        
        ReqKWList = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True),
        OptKWList = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True),
        ExtKWList = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True),
        
        ReqPyArgFmt = dict(separator=''),
        OptPyArgFmt = dict(separator=''),
        ExtPyArgFmt = dict(separator=''),
        OptExtPyArgFmt = dict(separator='', prefix='|', skip_prefix_when_empty=True),
        
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
                                   PyCArgument = 'Args',
                                   CDecl = 'Decl')

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
  if (PyArg_ParseTupleAndKeywords(pyc_args, pyc_keywds,"%(ReqPyArgFmt)s%(OptExtPyArgFmt)s",
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
        self._provides = options.pop('provides',
                                     '%s_%s' % (self.__class__.__name__, name))
        self.title = options.pop('title', None)
        self.description = options.pop('description', None)

        if options: self.warning('%s unused options: %s\n' % (self.__class__.__name__, options))
        
        map(self.add, components)
        return self

    def init_containers(self):
        return

    def update_containers(self):
        evaluate = self.evaluate

        # update ExtensionModule containers:
        t = '{"%(name)s", (PyCFunction)%(pyc_name)s,\n METH_VARARGS | METH_KEYWORDS, %(pyc_name)s_doc}'
        self.container_ModuleMethod.add(evaluate(t), self.name)

        # update local containers:
        self.container_OptExtArgs += self.container_OptArgs + self.container_ExtArgs
        self.container_OptExtPyArgFmt += self.container_OptPyArgFmt + self.container_ExtPyArgFmt
        self.container_ModuleFuncDoc += evaluate('%(name)s(%(ReqArgs)s%(OptExtArgs)s) -> %(RetArgs)s')
        if self.title is not None:
            self.container_FuncTitle += self.title
            self.container_ModuleFuncDoc += '  ' + self.title
        if self.description is not None:
            self.container_FuncDescr += self.description

        # resolve dependencies
        sorted_arguments = []
        sorted_names = []
        comp_map = {}
        dep_map = {}
        for (c,l) in self.components:
            if not isinstance(c, Component.PyCArgument):
                continue
            d = [n for n in c.depends if n not in sorted_names]
            if not d:
                sorted_arguments.append((c,l))
                sorted_names.append(c.name)
            else:
                comp_map[c.name] = (c,l)
                dep_map[c.name] = d

        while dep_map:
            dep_map_copy = dep_map.copy()
            for name, deps in dep_map.items():
                d = [n for n in deps if dep_map.has_key(n)]
                if not d:
                    sorted_arguments.append(comp_map[name])
                    del dep_map[name]
                else:
                    dep_map[name] = d
            if dep_map_copy==dep_map:
                self.warnign('%s: detected cyclic dependencies in %r, incorrect behavior is expected.\n'\
                             % (self.provides, dep_map))
                sorted_arguments += dep_map.values()
                break

        for c, l in sorted_arguments:
            old_parent = c.parent
            c.parent = self
            c.ctype.set_converters(c)
            c.parent = old_parent
        
        return

    
def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()
