
__all__ = ['PySource', 'PyCFunction', 'PyCModule', 'PyCTypeSpec', 'PyCArgument', 'PyCReturn']

import os
import sys
from base import Component
from utils import *
from c_support import *

class PySource(FileSource):

    template_py_header = '''\
#!/usr/bin/env python
# This file %(path)r is generated using ExtGen tool
# from NumPy version %(numpy_version)s.
# ExtGen is developed by Pearu Peterson <pearu.peterson@gmail.com>.
# For more information see http://www.scipy.org/ExtGen/ .'''

    container_options = dict(
        Content = dict(default='',
                       prefix = template_py_header + '\n',
                       suffix = '\n',
                       use_indent=True)
        )

    pass

class PyCModule(CSource):

    """
    >>> m = PyCModule('PyCModule_test', title='This is first line.\\nSecond line.', description='This is a module.\\nYes, it is.')
    >>> mod = m.build()
    >>> print mod.__doc__ #doctest: +ELLIPSIS
    This module 'PyCModule_test' is generated with ExtGen from NumPy version ...
    <BLANKLINE>
    This is first line.
    Second line.
    <BLANKLINE>
    This is a module.
    Yes, it is.
    """

    template = CSource.template_c_header + '''
#ifdef __cplusplus
extern \"C\" {
#endif
#include "Python.h"
%(CHeader)s
%(CTypeDef)s
%(CProto)s
%(CDefinition)s
%(CAPIDefinition)s
%(CDeclaration)s
%(PyCModuleCDeclaration)s
%(CMainProgram)s
#ifdef __cplusplus
}
#endif
'''

    container_options = CSource.container_options.copy()
    container_options.update(CAPIDefinition=container_options['CDefinition'],
                             PyCModuleCDeclaration=dict(default='<KILLLINE>',
                                                        ignore_empty_content=True),
                             )

    component_container_map = dict(
        PyCModuleInitFunction = 'CMainProgram',
        PyCModuleCDeclaration = 'PyCModuleCDeclaration',
        PyCFunction = 'CAPIDefinition',
        )

    def initialize(self, pyname, *components, **options):
        self.pyname = pyname
        self.title = options.pop('title', None)
        self.description = options.pop('description', None)

        self = CSource.initialize(self, '%smodule.c' % (pyname), **options)
        self.need_numpy_support = False

        self.cdecl = PyCModuleCDeclaration(pyname)
        self += self.cdecl

        self.main = PyCModuleInitFunction(pyname)
        self += self.main
        map(self.add, components)
        return self

    def update_parent(self, parent):
        if isinstance(parent, Component.SetupPy):
            self.update_SetupPy(parent)

    def update_SetupPy(self, parent):
        parent.setup_py += self.evaluate('    config.add_extension(%(pyname)r, sources = ["%(extmodulesrc)s"])',
                                         extmodulesrc = self.path)
        parent.init_py += 'import %s' % (self.pyname)

    def finalize(self):
        if self.need_numpy_support:
            self.add(CCode('''
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
'''), 'CHeader')
            self.main.add(CCode('''
import_array();
if (PyErr_Occurred()) {
  PyErr_SetString(PyExc_ImportError, "failed to load NumPy array module.");
  goto capi_error;
}
'''),'CBody')
        CSource.finalize(self)

    def build(self, build_dir=None, clean_at_exit=None):
        """ build(build_dir=None, clean_at_exit=None)

        A convenience function to build, import, an return
        an extension module object.
        """
        if build_dir is None:
            import tempfile
            import time
            packagename = 'extgen_' + str(hex(int(time.time()*10000000)))[2:]
            build_dir = os.path.join(tempfile.gettempdir(), packagename)
            clean_at_exit = True

        setup = Component.SetupPy(build_dir)
        setup += self
        s,o = setup.execute('build_ext','--inplace')
        if s:
            self.info('return status=%s' % (s))
            self.info(o)
            raise RuntimeError('failed to build extension module %r,'\
                               ' the build is located in %r directory'\
                               % (self.pyname, build_dir))

        if clean_at_exit:
            import atexit
            import shutil
            atexit.register(lambda d=build_dir: shutil.rmtree(d))
            self.info('directory %r will be removed at exit from python.' % (build_dir))

        sys.path.insert(0, os.path.dirname(build_dir))
        packagename = os.path.basename(build_dir)
        try:
            p = __import__(packagename)
            m = getattr(p, self.pyname)
        except:
            del sys.path[0]
            raise
        else:
            del sys.path[0]
        return m

class PyCModuleCDeclaration(Component):

    template = '''\
static PyObject* extgen_module;
static
PyMethodDef extgen_module_methods[] = {
  %(PyMethodDef)s
  {NULL,NULL,0,NULL}
};
static
char extgen_module_doc[] =
"This module %(pyname)r is generated with ExtGen from NumPy version %(numpy_version)s."
%(Title)s
%(Description)s
%(FunctionSignature)s
;'''
    container_options = dict(
        PyMethodDef = dict(suffix=',', skip_suffix_when_empty=True,separator=',\n',
                           default='<KILLLINE>', use_indent=True, ignore_empty_content=True),
        FunctionSignature = dict(prefix='"\\n\\n:Functions:\\n"\n"  ', skip_prefix_when_empty=True, use_indent=True,
                                 ignore_empty_content=True, default='<KILLLINE>',
                                 separator = '"\n"  ', suffix='"', skip_suffix_when_empty=True,
                                 ),
        Title = dict(default='<KILLLINE>',prefix='"\\n\\n',suffix='"',separator='\\n"\n"',
                         skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                         use_firstline_indent=True, replace_map={'\n':'\\n'}),
        Description = dict(default='<KILLLINE>',prefix='"\\n\\n"\n"',
                         suffix='"',separator='\\n"\n"',
                         skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                         use_firstline_indent=True, replace_map={'\n':'\\n'}),
        )

    default_component_class_name = 'Line'

    def initialize(self, pyname):
        self.pyname = pyname
        return self

    def update_parent(self, parent):
        if isinstance(parent, PyCModule):
            self.update_PyCModule(parent)

    def update_PyCModule(self, parent):
        if parent.title:
            self.add(parent.title, 'Title')
        if parent.description:
            self.add(parent.description, 'Description')


class PyCModuleInitFunction(CFunction):

    """
    >>> f = PyCModuleInitFunction('test_PyCModuleInitFunction')
    >>> print f.generate()
    PyMODINIT_FUNC
    inittest_PyCModuleInitFunction(void) {
      PyObject* extgen_module_dict = NULL;
      PyObject* extgen_str_obj = NULL;
      extgen_module = Py_InitModule(\"test_PyCModuleInitFunction\", extgen_module_methods);
      if ((extgen_module_dict = PyModule_GetDict(extgen_module))==NULL) goto capi_error;
      if ((extgen_str_obj = PyString_FromString(extgen_module_doc))==NULL) goto capi_error;
      PyDict_SetItemString(extgen_module_dict, \"__doc__\", extgen_str_obj);
      Py_DECREF(extgen_str_obj);
      if ((extgen_str_obj = PyString_FromString(\"restructuredtext\"))==NULL) goto capi_error;
      PyDict_SetItemString(extgen_module_dict, \"__docformat__\", extgen_str_obj);
      Py_DECREF(extgen_str_obj);
      return;
    capi_error:
      if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, \"failed to initialize 'test_PyCModuleInitFunction' module.\");
      }
      return;
    }
    """

    template = '''\
%(CSpecifier)s
%(CTypeSpec)s
%(name)s(void) {
  PyObject* extgen_module_dict = NULL;
  PyObject* extgen_str_obj = NULL;
  %(CDeclaration)s
  extgen_module = Py_InitModule("%(pyname)s", extgen_module_methods);
  if ((extgen_module_dict = PyModule_GetDict(extgen_module))==NULL) goto capi_error;
  if ((extgen_str_obj = PyString_FromString(extgen_module_doc))==NULL) goto capi_error;
  PyDict_SetItemString(extgen_module_dict, "__doc__", extgen_str_obj);
  Py_DECREF(extgen_str_obj);
  if ((extgen_str_obj = PyString_FromString("restructuredtext"))==NULL) goto capi_error;
  PyDict_SetItemString(extgen_module_dict, "__docformat__", extgen_str_obj);
  Py_DECREF(extgen_str_obj);
  %(CBody)s
  return;
capi_error:
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError, "failed to initialize %(pyname)r module.");
  }
  return;
}'''

    def initialize(self, pyname, *components, **options):
        self.pyname = pyname
        self.title = options.pop('title', None)
        self.description = options.pop('description', None)
        self = CFunction.initialize(self, 'init'+pyname, 'PyMODINIT_FUNC', *components, **options)
        return self

#helper classes for PyCFunction
class KWListBase(Word): parent_container_options = dict(separator=', ', suffix=', ', skip_suffix_when_empty=True)
class ReqKWList(KWListBase): pass
class OptKWList(KWListBase): pass
class ExtKWList(KWListBase): pass
class ArgBase(Word): parent_container_options = dict(separator=', ')
class ReqArg(ArgBase): pass
class OptArg(ArgBase): pass
class ExtArg(ArgBase): pass
class RetArg(ArgBase):
    parent_container_options = dict(separator=', ', prefix='(', suffix=')', default = 'None',
                                    skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                                    skip_prefix_suffix_when_single=True)
class OptExtArg(ArgBase):
    parent_container_options = dict(separator=', ', prefix=' [, ', skip_prefix_when_empty=True,
                                    suffix=']', skip_suffix_when_empty=True)
class ArgDocBase(Word):
    parent_container_options = dict(default='<KILLLINE>', prefix='"\\n\\nArguments:\\n"\n"  ',
                                    separator='\\n"\n"  ', suffix='"',
                                    skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                                    use_firstline_indent=True, replace_map={'\n':'\\n'})
class ReqArgDoc(ArgDocBase):
    parent_container_options = ArgDocBase.parent_container_options.copy()
    parent_container_options.update(prefix='"\\n\\n:Parameters:\\n"\n"  ')
class OptArgDoc(ArgDocBase):
    parent_container_options = ArgDocBase.parent_container_options.copy()
    parent_container_options.update(prefix='"\\n\\n:Optional parameters:\\n"\n"  ')
class ExtArgDoc(ArgDocBase):
    parent_container_options = ArgDocBase.parent_container_options.copy()
    parent_container_options.update(prefix='"\\n\\n:Extra parameters:\\n"\n"  ')
class RetArgDoc(ArgDocBase):
    parent_container_options = ArgDocBase.parent_container_options.copy()
    parent_container_options.update(prefix='"\\n\\n:Returns:\\n"\n"  ',
                                    default='"\\n\\n:Returns:\\n  None"')
class ArgFmtBase(Word): parent_container_options = dict(separator='')
class ReqArgFmt(ArgFmtBase): pass
class OptArgFmt(ArgFmtBase): pass
class ExtArgFmt(ArgFmtBase): pass
class RetArgFmt(ArgFmtBase): pass
class OptExtArgFmt(ArgFmtBase):
    parent_container_options = dict(separator='', prefix='|', skip_prefix_when_empty=True)
class ArgObjBase(Word): parent_container_options = dict(separator=', ', prefix=', ', skip_prefix_when_empty=True)
class ReqArgObj(ArgObjBase): pass
class OptArgObj(ArgObjBase): pass
class ExtArgObj(ArgObjBase): pass
class RetArgObj(ArgObjBase): pass

class FunctionSignature(Component):
    template = '%(name)s(%(ReqArg)s%(OptExtArg)s) -> %(RetArg)s'
    parent_container_options = dict()
    container_options = dict(
        ReqArg = ReqArg.parent_container_options,
        OptArg = OptArg.parent_container_options,
        ExtArg = ExtArg.parent_container_options,
        RetArg = RetArg.parent_container_options,
        OptExtArg = OptExtArg.parent_container_options,
        )
    def initialize(self, name, *components, **options):
        self.name = name
        map(self.add, components)
        return self
    def update_containers(self):
        self.container_OptExtArg += self.container_OptArg + self.container_ExtArg

class PyCFunction(CFunction):

    """
    >>> from __init__ import *
    >>> f = PyCFunction('foo')
    >>> print f.generate()
    static
    char pyc_function_foo_doc[] =
    \"  foo() -> None\"
    \"\\n\\n:Returns:\\n  None\"
    ;
    static
    PyObject*
    pyc_function_foo(PyObject *pyc_self, PyObject *pyc_args, PyObject *pyc_keywds) {
      PyObject * volatile pyc_buildvalue = NULL;
      volatile int capi_success = 1;
      static char *capi_kwlist[] = {NULL};
      if (PyArg_ParseTupleAndKeywords(pyc_args, pyc_keywds,"",
                                      capi_kwlist)) {
        capi_success = !PyErr_Occurred();
        if (capi_success) {
          pyc_buildvalue = Py_BuildValue("");
        }
      }
      return pyc_buildvalue;
    }
    >>> f = PyCFunction('foo', title='  Function title.\\nSecond line.', description=' This is a function.\\n2nd line.')
    >>> e = PyCModule('PyCFunction_test', f)
    >>> mod = e.build()
    >>> print mod.foo.__doc__
      foo() -> None
    <BLANKLINE>
      Function title.
      Second line.
    <BLANKLINE>
     This is a function.
     2nd line.
    <BLANKLINE>
    :Returns:
      None
    """

    template = '''\
static
char %(name)s_doc[] =
"  %(FunctionSignature)s"
%(Title)s
%(Description)s
%(ReqArgDoc)s
%(RetArgDoc)s
%(OptArgDoc)s
%(ExtArgDoc)s
;
static
PyObject*
%(name)s(PyObject *pyc_self, PyObject *pyc_args, PyObject *pyc_keywds) {
  PyObject * volatile pyc_buildvalue = NULL;
  volatile int capi_success = 1;
  %(CDeclaration)s
  static char *capi_kwlist[] = {%(ReqKWList)s%(OptKWList)s%(ExtKWList)sNULL};
  if (PyArg_ParseTupleAndKeywords(pyc_args, pyc_keywds,"%(ReqArgFmt)s%(OptExtArgFmt)s",
                                  capi_kwlist%(ReqArgObj)s%(OptArgObj)s%(ExtArgObj)s)) {
    %(FromPyObj)s
    %(CBody)s
    capi_success = !PyErr_Occurred();
    if (capi_success) {
      %(PyObjFrom)s
      pyc_buildvalue = Py_BuildValue("%(RetArgFmt)s"%(RetArgObj)s);
      %(CleanPyObjFrom)s
    }
    %(CleanCBody)s
    %(CleanFromPyObj)s
  }
  return pyc_buildvalue;
}'''

    container_options = CFunction.container_options.copy()

    container_options.update(\

        TMP = dict(),

        ReqArg = ReqArg.parent_container_options,
        OptArg = OptArg.parent_container_options,
        ExtArg = ExtArg.parent_container_options,
        RetArg = RetArg.parent_container_options,

        FunctionSignature = FunctionSignature.parent_container_options,

        OptExtArg = OptExtArg.parent_container_options,

        Title = dict(default='<KILLLINE>',prefix='"\\n\\n',suffix='"',separator='\\n"\n"',
                     skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                     use_firstline_indent=True, replace_map={'\n':'\\n'}),
        Description = dict(default='<KILLLINE>',prefix='"\\n\\n"\n"',
                           suffix='"',separator='\\n"\n"',
                           skip_prefix_when_empty=True, skip_suffix_when_empty=True,
                           use_firstline_indent=True, replace_map={'\n':'\\n'}),

        ReqArgDoc = ReqArgDoc.parent_container_options,
        OptArgDoc = OptArgDoc.parent_container_options,
        ExtArgDoc = ExtArgDoc.parent_container_options,
        RetArgDoc = RetArgDoc.parent_container_options,

        ReqKWList = ReqKWList.parent_container_options,
        OptKWList = OptKWList.parent_container_options,
        ExtKWList = ExtKWList.parent_container_options,

        ReqArgFmt = ReqArgFmt.parent_container_options,
        OptArgFmt = OptArgFmt.parent_container_options,
        ExtArgFmt = ExtArgFmt.parent_container_options,
        OptExtArgFmt = OptExtArgFmt.ExtArgFmt.parent_container_options,
        RetArgFmt = ExtArgFmt.parent_container_options,

        ReqArgObj = ReqArgObj.parent_container_options,
        OptArgObj = OptArgObj.parent_container_options,
        ExtArgObj = ExtArgObj.parent_container_options,
        RetArgObj = RetArgObj.parent_container_options,

        FromPyObj = CCode.parent_container_options,
        PyObjFrom = CCode.parent_container_options,

        CleanPyObjFrom = dict(default='<KILLLINE>', reverse=True, use_indent=True, ignore_empty_content=True),
        CleanCBody = dict(default='<KILLLINE>', reverse=True, use_indent=True, ignore_empty_content=True),
        CleanFromPyObj = dict(default='<KILLLINE>', reverse=True, use_indent=True, ignore_empty_content=True),

        )

    default_component_class_name = 'CCode'

    component_container_map = CFunction.component_container_map.copy()
    component_container_map.update(
        PyCArgument = 'TMP',
        CCode = 'CBody',
        )

    def initialize(self, pyname, *components, **options):
        self.pyname = pyname
        self.title = options.pop('title', None)
        self.description = options.pop('description', None)
        self = CFunction.initialize(self, 'pyc_function_'+pyname, 'PyObject*', **options)
        self.signature = FunctionSignature(pyname)
        self += self.signature
        if self.title:
            self.add(self.title, 'Title')
        if self.description:
            self.add(self.description, 'Description')
        map(self.add, components)
        return self

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.pyname]+[c for (c,l) in self.components])))

    def update_parent(self, parent):
        if isinstance(parent, PyCModule):
            self.update_PyCModule(parent)

    def update_PyCModule(self, parent):
        t = '  {"%(pyname)s", (PyCFunction)%(name)s, METH_VARARGS | METH_KEYWORDS, %(name)s_doc}'
        parent.cdecl.add(self.evaluate(t),'PyMethodDef')
        parent.cdecl.add(self.signature,'FunctionSignature')

    def update_containers(self):
        self.container_OptExtArg += self.container_OptArg + self.container_ExtArg
        self.container_OptExtArgFmt += self.container_OptArgFmt + self.container_ExtArgFmt

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


class PyCArgument(Component):

    """
    >>> from __init__ import *
    >>> a = PyCArgument('a')
    >>> print a
    PyCArgument('a', PyCTypeSpec('object'))
    >>> print a.generate()
    a
    >>> f = PyCFunction('foo')
    >>> f += a
    >>> f += PyCArgument('b')
    >>> m = PyCModule('PyCArgument_test')
    >>> m += f
    >>> #print m.generate()
    >>> mod = m.build()
    >>> print mod.__doc__ #doctest: +ELLIPSIS
    This module 'PyCArgument_test' is generated with ExtGen from NumPy version ...
    <BLANKLINE>
    :Functions:
      foo(a, b) -> None

    """

    container_options = dict(
        TMP = dict()
        )

    component_container_map = dict(
        PyCTypeSpec = 'TMP'
        )

    template = '%(name)s'

    def initialize(self, name, ctype = object, *components, **options):
        self.input_intent = options.pop('input_intent','required') # 'optional', 'extra', 'hide'
        self.output_intent = options.pop('output_intent','hide')   # 'return'
        self.input_title = options.pop('input_title', None)
        self.output_title = options.pop('output_title', None)
        self.input_description = options.pop('input_description', None)
        self.output_description = options.pop('output_description', None)
        self.depends = options.pop('depends', [])
        title = options.pop('title', None)
        description = options.pop('description', None)
        if title is not None:
            if self.input_intent!='hide':
                if self.input_title is None:
                    self.input_title = title
            elif self.output_intent!='hide':
                if self.output_title is None:
                    self.output_title = title
        if description is not None:
            if self.input_intent!='hide':
                if self.input_description is None:
                    self.input_description = description
            elif self.output_intent!='hide':
                if self.output_description is None:
                    self.output_description = description
        if options: self.warning('%s unused options: %s\n' % (self.__class__.__name__, options))

        self.name = name
        self.ctype = ctype = PyCTypeSpec(ctype)
        self += ctype

        self.cvar = name
        self.pycvar = None
        self.retpycvar = None

        retfmt = ctype.get_pyret_fmt(self)
        if isinstance(ctype, PyCTypeSpec):
            if retfmt and retfmt in 'SON':
                if self.output_intent == 'return':
                    if self.input_intent=='hide':
                        self.retpycvar = name
                    else:
                        self.pycvar = name
                        self.retpycvar = name + '_return'
                elif self.input_intent!='hide':
                    self.pycvar = name
            else:
                self.pycvar = name
                self.retpycvar = name
        else:
            self.pycvar = name + '_pyc'
            self.retpycvar = name + '_pyc_r'

        ctype.set_titles(self)

        map(self.add, components)
        return self

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.name]+[c for (c,l) in self.components])))

    def update_parent(self, parent):
        if isinstance(parent, PyCFunction):
            self.update_PyCFunction(parent)

    def update_PyCFunction(self, parent):
        ctype = self.ctype

        input_doc_title = '%s : %s' % (self.name, self.input_title)
        output_doc_title = '%s : %s' % (self.name, self.output_title)
        if self.input_description is not None:
            input_doc_descr = '  %s' % (self.input_description)
        else:
            input_doc_descr = None
        if self.output_description is not None:
            output_doc_descr = '  %s' % (self.output_description)
        else:
            output_doc_descr = None

        # add components to parent:
        parent += ctype.get_decl(self, parent)
        if self.input_intent=='required':
            parent += ReqArg(self.name)
            parent.signature += ReqArg(self.name)
            parent += ReqKWList('"' + self.name + '"')
            parent += ReqArgFmt(ctype.get_pyarg_fmt(self))
            parent += ReqArgObj(ctype.get_pyarg_obj(self))
            parent += ReqArgDoc(input_doc_title)
            parent += ReqArgDoc(input_doc_descr)
        elif self.input_intent=='optional':
            parent += OptArg(self.name)
            parent.signature += OptArg(self.name)
            parent += OptKWList('"' + self.name + '"')
            parent += OptArgFmt(ctype.get_pyarg_fmt(self))
            parent += OptArgObj(ctype.get_pyarg_obj(self))
            parent += OptArgDoc(input_doc_title)
            parent += OptArgDoc(input_doc_descr)
        elif self.input_intent=='extra':
            parent += ExtArg(self.name)
            parent.signature += ExtArg(self.name)
            parent += ExtKWList('"' + self.name + '"')
            parent += ExtArgFmt(ctype.get_pyarg_fmt(self))
            parent += ExtArgObj(ctype.get_pyarg_obj(self))
            parent += ExtArgDoc(input_doc_title)
            parent += ExtArgDoc(input_doc_descr)
        elif self.input_intent=='hide':
            pass
        else:
            raise NotImplementedError('input_intent=%r' % (self.input_intent))

        if self.output_intent=='return':
            parent += RetArg(self.name)
            parent.signature += RetArg(self.name)
            parent += RetArgFmt(ctype.get_pyret_fmt(self))
            parent += RetArgObj(ctype.get_pyret_obj(self))
            parent += RetArgDoc(output_doc_title)
            parent += RetArgDoc(output_doc_descr)
        elif self.output_intent=='hide':
            pass
        else:
            raise NotImplementedError('output_intent=%r' % (self.output_intent))

class PyCReturn(PyCArgument):

    def initialize(self, name, ctype = object, *components, **options):
        return PyCArgument(name, ctype, input_intent='hide', output_intent='return', *components, **options)

class PyCTypeSpec(CTypeSpec):

    """
    >>> s = PyCTypeSpec(object)
    >>> print s
    PyCTypeSpec('object')
    >>> print s.generate()
    PyObject*

    >>> from __init__ import *
    >>> m = PyCModule('test_PyCTypeSpec')
    >>> f = PyCFunction('func')
    >>> f += PyCArgument('i', int, output_intent='return')
    >>> f += PyCArgument('l', long, output_intent='return')
    >>> f += PyCArgument('f', float, output_intent='return')
    >>> f += PyCArgument('c', complex, output_intent='return')
    >>> f += PyCArgument('s', str, output_intent='return')
    >>> f += PyCArgument('u', unicode, output_intent='return')
    >>> f += PyCArgument('t', tuple, output_intent='return')
    >>> f += PyCArgument('lst', list, output_intent='return')
    >>> f += PyCArgument('d', dict, output_intent='return')
    >>> f += PyCArgument('set', set, output_intent='return')
    >>> f += PyCArgument('o1', object, output_intent='return')
    >>> f += PyCArgument('o2', object, output_intent='return')
    >>> m += f
    >>> b = m.build() #doctest: +ELLIPSIS
    >>> b.func(23, 23l, 1.2, 1+2j, 'hello', u'hei', (2,'a'), [-2], {3:4}, set([1,2]), 2, '15')
    (23, 23L, 1.2, (1+2j), 'hello', u'hei', (2, 'a'), [-2], {3: 4}, set([1, 2]), 2, '15')
    >>> print b.func.__doc__
      func(i, l, f, c, s, u, t, lst, d, set, o1, o2) -> (i, l, f, c, s, u, t, lst, d, set, o1, o2)
    <BLANKLINE>
    :Parameters:
      i : a python int object
      l : a python long object
      f : a python float object
      c : a python complex object
      s : a python str object
      u : a python unicode object
      t : a python tuple object
      lst : a python list object
      d : a python dict object
      set : a python set object
      o1 : a python object
      o2 : a python object
    <BLANKLINE>
    :Returns:
      i : a python int object
      l : a python long object
      f : a python float object
      c : a python complex object
      s : a python str object
      u : a python unicode object
      t : a python tuple object
      lst : a python list object
      d : a python dict object
      set : a python set object
      o1 : a python object
      o2 : a python object

    >>> m = PyCModule('test_PyCTypeSpec_c')
    >>> f = PyCFunction('func_c_int')
    >>> f += PyCArgument('i1', 'c_char', output_intent='return')
    >>> f += PyCArgument('i2', 'c_short', output_intent='return')
    >>> f += PyCArgument('i3', 'c_int', output_intent='return')
    >>> f += PyCArgument('i4', 'c_long', output_intent='return')
    >>> f += PyCArgument('i5', 'c_long_long', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_unsigned_int')
    >>> f += PyCArgument('i1', 'c_unsigned_char', output_intent='return')
    >>> f += PyCArgument('i2', 'c_unsigned_short', output_intent='return')
    >>> f += PyCArgument('i3', 'c_unsigned_int', output_intent='return')
    >>> f += PyCArgument('i4', 'c_unsigned_long', output_intent='return')
    >>> f += PyCArgument('i5', 'c_unsigned_long_long', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_float')
    >>> f += PyCArgument('f1', 'c_float', output_intent='return')
    >>> f += PyCArgument('f2', 'c_double', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_complex')
    >>> f += PyCArgument('c1', 'c_Py_complex', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_string')
    >>> f += PyCArgument('s1', 'c_const_char_ptr', output_intent='return')
    >>> f += PyCArgument('s2', 'c_const_char_ptr', output_intent='return')
    >>> f += PyCArgument('s3', 'c_Py_UNICODE', output_intent='return')
    >>> f += PyCArgument('s4', 'c_char1', output_intent='return')
    >>> m += f
    >>> b = m.build()
    >>> b.func_c_int(2,3,4,5,6)
    (2, 3, 4, 5, 6L)
    >>> b.func_c_unsigned_int(-1,-1,-1,-1,-1)
    (255, 65535, 4294967295, 18446744073709551615L, 18446744073709551615L)
    >>> b.func_c_float(1.2,1.2)
    (1.2000000476837158, 1.2)
    >>> b.func_c_complex(1+2j)
    (1+2j)
    >>> b.func_c_string('hei', None, u'tere', 'b')
    ('hei', None, u'tere', 'b')

    >>> import numpy
    >>> m = PyCModule('test_PyCTypeSpec_numpy')
    >>> f = PyCFunction('func_int')
    >>> f += PyCArgument('i1', numpy.int8, output_intent='return')
    >>> f += PyCArgument('i2', numpy.int16, output_intent='return')
    >>> f += PyCArgument('i3', numpy.int32, output_intent='return')
    >>> f += PyCArgument('i4', numpy.int64, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_uint')
    >>> f += PyCArgument('i1', numpy.uint8, output_intent='return')
    >>> f += PyCArgument('i2', numpy.uint16, output_intent='return')
    >>> f += PyCArgument('i3', numpy.uint32, output_intent='return')
    >>> f += PyCArgument('i4', numpy.uint64, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_float')
    >>> f += PyCArgument('f1', numpy.float32, output_intent='return')
    >>> f += PyCArgument('f2', numpy.float64, output_intent='return')
    >>> f += PyCArgument('f3', numpy.float128, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_complex')
    >>> f += PyCArgument('c1', numpy.complex64, output_intent='return')
    >>> f += PyCArgument('c2', numpy.complex128, output_intent='return')
    >>> f += PyCArgument('c3', numpy.complex256, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_array')
    >>> f += PyCArgument('a1', numpy.ndarray, output_intent='return')
    >>> m += f
    >>> b = m.build()
    >>> b.func_int(numpy.int8(-2), numpy.int16(-3), numpy.int32(-4), numpy.int64(-5))
    (-2, -3, -4, -5)
    >>> b.func_uint(numpy.uint8(-1), numpy.uint16(-1), numpy.uint32(-1), numpy.uint64(-1))
    (255, 65535, 4294967295, 18446744073709551615)
    >>> b.func_float(numpy.float32(1.2),numpy.float64(1.2),numpy.float128(1.2))
    (1.20000004768, 1.2, 1.19999999999999995559)
    >>> b.func_complex(numpy.complex64(1+2j),numpy.complex128(1+2j),numpy.complex256(1+2j))
    ((1+2j), (1+2j), (1.0+2.0j))
    >>> b.func_array(numpy.array([1,2]))
    array([1, 2])
    >>> b.func_array(numpy.array(2))
    array(2)
    >>> b.func_array(2)
    Traceback (most recent call last):
    ...
    TypeError: argument 1 must be numpy.ndarray, not int
    >>> b.func_array(numpy.int8(2))
    Traceback (most recent call last):
    ...
    TypeError: argument 1 must be numpy.ndarray, not numpy.int8
    """

    typeinfo_map = dict(
        int = ('PyInt_Type', 'PyIntObject*', 'O!', 'N', 'NULL'),
        long = ('PyLong_Type', 'PyLongObject*', 'O!', 'N', 'NULL'),
        float = ('PyFloat_Type', 'PyFloatObject*', 'O!', 'N', 'NULL'),
        complex = ('PyComplex_Type', 'PyComplexObject*', 'O!', 'N', 'NULL'),
        str = ('PyString_Type', 'PyStringObject*', 'S', 'N', 'NULL'),
        unicode = ('PyUnicode_Type', 'PyUnicodeObject*', 'U', 'N', 'NULL'),
        buffer = ('PyBuffer_Type', 'PyBufferObject*', 'O!', 'N', 'NULL'),
        tuple = ('PyTuple_Type', 'PyTupleObject*', 'O!', 'N', 'NULL'),
        list = ('PyList_Type', 'PyListObject*', 'O!', 'N', 'NULL'),
        dict = ('PyDict_Type', 'PyDictObject*', 'O!', 'N', 'NULL'),
        file = ('PyFile_Type', 'PyFileObject*', 'O!', 'N', 'NULL'),
        instance = ('PyInstance_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        function = ('PyFunction_Type', 'PyFunctionObject*', 'O!', 'N', 'NULL'),
        method = ('PyMethod_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        module = ('PyModule_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        iter = ('PySeqIter_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        property = ('PyProperty_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        slice = ('PySlice_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        cell = ('PyCell_Type', 'PyCellObject*', 'O!', 'N', 'NULL'),
        generator = ('PyGen_Type', 'PyGenObject*', 'O!', 'N', 'NULL'),
        set = ('PySet_Type', 'PySetObject*', 'O!', 'N', 'NULL'),
        frozenset = ('PyFrozenSet_Type', 'PySetObject*', 'O!', 'N', 'NULL'),
        cobject = (None, 'PyCObject*', 'O', 'N', 'NULL'),
        type = ('PyType_Type', 'PyTypeObject*', 'O!', 'N', 'NULL'),
        object = (None, 'PyObject*', 'O', 'N', 'NULL'),
        numpy_ndarray = ('PyArray_Type', 'PyArrayObject*', 'O!', 'N', 'NULL'),
        numpy_descr = ('PyArrayDescr_Type','PyArray_Descr', 'O!', 'N', 'NULL'),
        numpy_ufunc = ('PyUFunc_Type', 'PyUFuncObject*', 'O!', 'N', 'NULL'),
        numpy_iter = ('PyArrayIter_Type', 'PyArrayIterObject*', 'O!', 'N', 'NULL'),
        numpy_multiiter = ('PyArrayMultiIter_Type', 'PyArrayMultiIterObject*', 'O!', 'N', 'NULL'),
        numpy_int8 = ('PyInt8ArrType_Type', 'PyInt8ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int16 = ('PyInt16ArrType_Type', 'PyInt16ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int32 = ('PyInt32ArrType_Type', 'PyInt32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int64 = ('PyInt64ArrType_Type', 'PyInt64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int128 = ('PyInt128ArrType_Type', 'PyInt128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint8 = ('PyUInt8ArrType_Type', 'PyUInt8ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint16 = ('PyUInt16ArrType_Type', 'PyUInt16ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint32 = ('PyUInt32ArrType_Type', 'PyUInt32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint64 = ('PyUInt64ArrType_Type', 'PyUInt64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint128 = ('PyUInt128ArrType_Type', 'PyUInt128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float16 = ('PyFloat16ArrType_Type', 'PyFloat16ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float32 = ('PyFloat32ArrType_Type', 'PyFloat32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float64 = ('PyFloat64ArrType_Type', 'PyFloat64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float80 = ('PyFloat80ArrType_Type', 'PyFloat80ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float96 = ('PyFloat96ArrType_Type', 'PyFloat96ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float128 = ('PyFloat128ArrType_Type', 'PyFloat128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex32 = ('PyComplex32ArrType_Type', 'PyComplex32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex64 = ('PyComplex64ArrType_Type', 'PyComplex64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex128 = ('PyComplex128ArrType_Type', 'PyComplex128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex160 = ('PyComplex160ArrType_Type', 'PyComplex160ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex192 = ('PyComplex192ArrType_Type', 'PyComplex192ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex256 = ('PyComplex256ArrType_Type', 'PyComplex256ScalarObject*', 'O!', 'N', 'NULL'),
        numeric_array = ('PyArray_Type', 'PyArrayObject*', 'O!', 'N', 'NULL'),
        c_char = (None, 'char', 'b', 'b', '0'),
        c_unsigned_char = (None, 'unsigned char', 'B', 'B', '0'),
        c_short = (None, 'short int', 'h', 'h', '0'),
        c_unsigned_short = (None, 'unsigned short int', 'H', 'H', '0'),
        c_int = (None,'int', 'i', 'i', '0'),
        c_unsigned_int = (None,'unsigned int', 'I', 'I', '0'),
        c_long = (None,'long', 'l', 'l', '0'),
        c_unsigned_long = (None,'unsigned long', 'k', 'k', '0'),
        c_long_long = (None,'PY_LONG_LONG', 'L', 'L', '0'),
        c_unsigned_long_long = (None,'unsigned PY_LONG_LONG', 'K', 'K', '0'),
        c_Py_ssize_t = (None,'Py_ssize_t', 'n', 'n', '0'),
        c_char1 = (None,'char', 'c', 'c', '"\\0"'),
        c_float = (None,'float', 'f', 'f', '0.0'),
        c_double = (None,'double', 'd', 'd', '0.0'),
        c_Py_complex = (None,'Py_complex', 'D', 'D', '{0.0, 0.0}'),
        c_const_char_ptr = (None,'const char *', 'z', 'z', 'NULL'),
        c_Py_UNICODE = (None,'Py_UNICODE*','u','u', 'NULL'),
        )

    def initialize(self, typeobj):
        if isinstance(typeobj, self.__class__):
            return typeobj

        m = self.typeinfo_map

        key = None
        if isinstance(typeobj, type):
            if typeobj.__module__=='__builtin__':
                key = typeobj.__name__
                if key=='array':
                    key = 'numeric_array'
            elif typeobj.__module__=='numpy':
                key = 'numpy_' + typeobj.__name__
        elif isinstance(typeobj, str):
            key = typeobj
            if key.startswith('numpy_'):
                k = key[6:]
                named_scalars = ['byte','short','int','long','longlong',
                                 'ubyte','ushort','uint','ulong','ulonglong',
                                 'intp','uintp',
                                 'float_','double',
                                 'longfloat','longdouble',
                                 'complex_',
                                 ]
                if k in named_scalars:
                    import numpy
                    key = 'numpy_' + getattr(numpy, k).__name__

        try: item = m[key]
        except KeyError:
            raise NotImplementedError('%s: need %s support' % (self.__class__.__name__, typeobj))

        self.typeobj_name = key
        self.ctypeobj = item[0]
        self.line = item[1]
        self.arg_fmt = item[2]
        self.ret_fmt = item[3]
        self.cinit_value = item[4]

        self.need_numpy_support = False
        if key.startswith('numpy_'):
            self.need_numpy_support = True
            #self.add(Component.get('arrayobject.h'), 'CHeader')
            #self.add(Component.get('import_array'), 'ModuleInit')
        if key.startswith('numeric_'):
            raise NotImplementedError(self.__class__.__name__ + ': Numeric support')

        return self

    def finalize(self):
        if self.need_numpy_support:
            self.component_PyCModule.need_numpy_support = True

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join([repr(self.typeobj_name)]+[repr(c) for (c,l) in self.components]))

    def get_pyarg_fmt(self, arg):
        if arg.input_intent=='hide': return None
        return self.arg_fmt

    def get_pyarg_obj(self, arg):
        if arg.input_intent=='hide': return None
        if self.arg_fmt=='O!':
            return '&%s, &%s' % (self.ctypeobj, arg.pycvar)
        return '&' + arg.pycvar

    def get_pyret_fmt(self, arg):
        if arg.output_intent=='hide': return None
        return self.ret_fmt

    def get_pyret_obj(self, arg):
        if arg.output_intent=='return':
            if self.get_pyret_fmt(arg)=='D':
                return '&' + arg.retpycvar
            return arg.retpycvar
        return

    def get_init_value(self, arg):
        return self.cinit_value

    def set_titles(self, arg):
        if self.typeobj_name == 'object':
            tn = 'a python ' + self.typeobj_name
        else:
            if self.typeobj_name.startswith('numpy_'):
                tn = 'a numpy.' + self.typeobj_name[6:] + ' object'
            elif self.typeobj_name.startswith('c_'):
                n = self.typeobj_name[2:]
                if not n.startswith('Py_'):
                    n = ' '.join(n.split('_'))
                tn = 'a to C ' + n + ' convertable object'
            else:
                tn = 'a python ' + self.typeobj_name + ' object'
        if arg.input_intent!='hide':
            r = ''
            if arg.input_title: r = ', ' + arg.input_title
            arg.input_title = tn + r
        if arg.output_intent!='hide':
            r = ''
            if arg.output_title: r = ', ' + arg.output_title
            arg.output_title = tn + r

    def get_decl(self, arg, func):
        init_value = self.get_init_value(arg)
        if init_value:
            init =  ' = %s' % (init_value)
        else:
            init = ''
        if arg.pycvar and arg.pycvar==arg.retpycvar:
            func += CDeclaration(self, '%s%s' % (arg.pycvar, init))
        else:
            if self.get_pyret_obj(arg) is None:
                if self.get_pyret_obj(arg) is not None:
                    func += CDeclaration(self, '%s%s' % (arg.pycvar, init))
            elif self.get_pyarg_obj(arg) is not None:
                func += CDeclaration(self, '%s%s' % (arg.pycvar, init))
                func += CDeclaration(self,'%s%s' % (arg.retpycvar, init))
            else:
                func += CDeclaration(self, '%s%s' % (arg.retpycvar, init))
        return

    def set_converters(self, arg):
        """
        Notes for user:
          if arg is intent(optional, in, out) and not specified
          as function argument then function may created but
          it must then have *new reference* (ie use Py_INCREF
          unless it is a new reference already).
        """
        # this method is called from PyCFunction.update_containers(),
        # note that self.parent is None put arg.parent is PyCFunction
        # instance.
        eval_a = arg.evaluate
        FromPyObj = arg.container_FromPyObj
        PyObjFrom = arg.container_PyObjFrom

        argfmt = self.get_pyarg_fmt(arg)
        retfmt = self.get_pyret_fmt(arg)
        if arg.output_intent=='return':
            if arg.input_intent in ['optional', 'extra']:
                if retfmt in 'SON':
                    FromPyObj += eval_a('''\
if (!(%(pycvar)s==NULL)) {
  /* make %(pycvar)r a new reference */
  %(retpycvar)s = %(pycvar)s;
  Py_INCREF((PyObject*)%(retpycvar)s);
}
''')
                    PyObjFrom += eval_a('''\
if (%(retpycvar)s==NULL) {
  /* %(pycvar)r was not specified */
  if (%(pycvar)s==NULL) {
    %(retpycvar)s = Py_None;
    Py_INCREF((PyObject*)%(retpycvar)s);
  } else {
    %(retpycvar)s = %(pycvar)s;
    /* %(pycvar)r must be a new reference or expect a core dump. */
  }
} elif (!(%(retpycvar)s == %(pycvar)s)) {
  /* a new %(retpycvar)r was created, undoing %(pycvar)s new reference */
  Py_DECREF((PyObject*)%(pycvar)s);
}
''')
            elif arg.input_intent=='hide':
                if retfmt in 'SON':
                    PyObjFrom += eval_a('''\
if (%(retpycvar)s==NULL) {
  %(retpycvar)s = Py_None;
  Py_INCREF((PyObject*)%(retpycvar)s);
} /* else %(retpycvar)r must be a new reference or expect a core dump. */
''')
            elif arg.input_intent=='required':
                if retfmt in 'SON':
                    FromPyObj += eval_a('''\
/* make %(pycvar)r a new reference */
%(retpycvar)s = %(pycvar)s;
Py_INCREF((PyObject*)%(retpycvar)s);
''')
                    PyObjFrom += eval_a('''\
if (!(%(retpycvar)s==%(pycvar)s)) {
  /* a new %(retpycvar)r was created, undoing %(pycvar)r new reference */
  /* %(retpycvar)r must be a new reference or expect a core dump. */
  Py_DECREF((PyObject*)%(pycvar)s);
}
''')


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
