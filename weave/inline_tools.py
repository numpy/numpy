# should re-write compiled functions to take a local and global dict
# as input.
import sys,os
import ext_tools
import string
import catalog
import common_info

# not an easy way for the user_path_list to come in here.
# the PYTHONCOMPILED environment variable offers the most hope.

function_catalog = catalog.catalog()


class inline_ext_function(ext_tools.ext_function):
    # Some specialization is needed for inline extension functions
    def function_declaration_code(self):
       code  = 'static PyObject* %s(PyObject*self, PyObject* args)\n{\n'
       return code % self.name

    def template_declaration_code(self):
        code = 'template<class T>\n' \
               'static PyObject* %s(PyObject*self, PyObject* args)\n{\n'
        return code % self.name

    def parse_tuple_code(self):
        """ Create code block for PyArg_ParseTuple.  Variable declarations
            for all PyObjects are done also.

            This code got a lot uglier when I added local_dict...
        """
        declare_return = 'py::object return_val;\n'    \
                         'int exception_occured = 0;\n'    \
                         'PyObject *py__locals = NULL;\n' \
                         'PyObject *py__globals = NULL;\n'

        py_objects = ', '.join(self.arg_specs.py_pointers())
        if py_objects:
            declare_py_objects = 'PyObject ' + py_objects +';\n'
        else:
            declare_py_objects = ''

        py_vars = ' = '.join(self.arg_specs.py_variables())
        if py_vars:
            init_values = py_vars + ' = NULL;\n\n'
        else:
            init_values = ''

        parse_tuple = 'if(!PyArg_ParseTuple(args,"OO:compiled_func",'\
                                           '&py__locals,'\
                                           '&py__globals))\n'\
                      '    return NULL;\n'

        return   declare_return + declare_py_objects + \
                 init_values + parse_tuple

    def arg_declaration_code(self):
        arg_strings = []
        for arg in self.arg_specs:
            arg_strings.append(arg.declaration_code(inline=1))
        code = string.join(arg_strings,"")
        return code

    def arg_cleanup_code(self):
        arg_strings = []
        for arg in self.arg_specs:
            arg_strings.append(arg.cleanup_code())
        code = string.join(arg_strings,"")
        return code

    def arg_local_dict_code(self):
        arg_strings = []
        for arg in self.arg_specs:
            arg_strings.append(arg.local_dict_code())
        code = string.join(arg_strings,"")
        return code


    def function_code(self):
        from ext_tools import indent
        decl_code = indent(self.arg_declaration_code(),4)
        cleanup_code = indent(self.arg_cleanup_code(),4)
        function_code = indent(self.code_block,4)
        #local_dict_code = indent(self.arg_local_dict_code(),4)

        try_code =    'try                              \n' \
                      '{                                \n' \
                      '    PyObject* raw_locals = py_to_raw_dict('       \
                                             'py__locals,"_locals");\n'  \
                      '    PyObject* raw_globals = py_to_raw_dict('      \
                                          'py__globals,"_globals");\n' + \
                      '    /* argument conversion code */     \n' + \
                           decl_code                               + \
                      '    /* inline code */                   \n' + \
                           function_code                           + \
                      '    /*I would like to fill in changed    '   \
                              'locals and globals here...*/   \n'   \
                      '\n}                                       \n'
        catch_code =  "catch(...)                        \n"   \
                      "{                                 \n" + \
                      "    return_val =  py::object();   \n"   \
                      "    exception_occured = 1;        \n"   \
                      "}                                 \n"   
        return_code = "    /* cleanup code */                   \n" + \
                           cleanup_code                             + \
                      "    if(!(PyObject*)return_val && !exception_occured)\n"   \
                      "    {\n                                  \n"   \
                      "        return_val = Py_None;            \n"   \
                      "    }\n                                  \n"   \
                      "    return return_val.disown();          \n"           \
                      "}                                \n"

        all_code = self.function_declaration_code()         + \
                       indent(self.parse_tuple_code(),4)    + \
                       indent(try_code,4)                   + \
                       indent(catch_code,4)                 + \
                       return_code

        return all_code

    def python_function_definition_code(self):
        args = (self.name, self.name)
        function_decls = '{"%s",(PyCFunction)%s , METH_VARARGS},\n' % args
        return function_decls

class inline_ext_module(ext_tools.ext_module):
    def __init__(self,name,compiler=''):
        ext_tools.ext_module.__init__(self,name,compiler)
        self._build_information.append(common_info.inline_info())

function_cache = {}
def inline(code,arg_names=[],local_dict = None, global_dict = None,
           force = 0,
           compiler='',
           verbose = 0,
           support_code = None,
           headers = [],
           customize=None,
           type_converters = None,
           auto_downcast=1,
           **kw):
    """ Inline C/C++ code within Python scripts.

        inline() compiles and executes C/C++ code on the fly.  Variables
        in the local and global Python scope are also available in the
        C/C++ code.  Values are passed to the C/C++ code by assignment
        much like variables passed are passed into a standard Python
        function.  Values are returned from the C/C++ code through a
        special argument called return_val.  Also, the contents of
        mutable objects can be changed within the C/C++ code and the
        changes remain after the C code exits and returns to Python.

        inline has quite a few options as listed below.  Also, the keyword
        arguments for distutils extension modules are accepted to
        specify extra information needed for compiling.

        code -- string. A string of valid C++ code.  It should not specify a
                return statement.  Instead it should assign results that
                need to be returned to Python in the return_val.
        arg_names -- optional. list of strings. A list of Python variable names 
                     that should be transferred from Python into the C/C++ 
                     code.  It defaults to an empty string.
        local_dict -- optional. dictionary. If specified, it is a dictionary
                      of values that should be used as the local scope for the
                      C/C++ code.  If local_dict is not specified the local
                      dictionary of the calling function is used.
        global_dict -- optional. dictionary.  If specified, it is a dictionary
                       of values that should be used as the global scope for
                       the C/C++ code.  If global_dict is not specified the
                       global dictionary of the calling function is used.
        force --      optional. 0 or 1. default 0.  If 1, the C++ code is
                      compiled every time inline is called.  This is really
                      only useful for debugging, and probably only useful if
                      your editing support_code a lot.
        compiler --   optional. string.  The name of compiler to use when
                      compiling.  On windows, it understands 'msvc' and 'gcc'
                      as well as all the compiler names understood by
                      distutils.  On Unix, it'll only understand the values
                      understood by distutils. ( I should add 'gcc' though
                      to this).

                      On windows, the compiler defaults to the Microsoft C++
                      compiler.  If this isn't available, it looks for mingw32
                      (the gcc compiler).

                      On Unix, it'll probably use the same compiler that was
                      used when compiling Python. Cygwin's behavior should be
                      similar.
        verbose --    optional. 0,1, or 2. defualt 0.  Speficies how much
                      much information is printed during the compile phase
                      of inlining code.  0 is silent (except on windows with
                      msvc where it still prints some garbage). 1 informs
                      you when compiling starts, finishes, and how long it
                      took.  2 prints out the command lines for the compilation
                      process and can be useful if your having problems
                      getting code to work.  Its handy for finding the name
                      of the .cpp file if you need to examine it.  verbose has
                      no affect if the compilation isn't necessary.
        support_code -- optional. string.  A string of valid C++ code declaring
                        extra code that might be needed by your compiled
                        function.  This could be declarations of functions,
                        classes, or structures.
        headers      -- optional. list of strings.  A list of strings specifying
                        header files to use when compiling the code.  The list 
                        might look like ["<vector>","'my_header'"].  Note that 
                        the header strings need to be in a form than can be 
                        pasted at the end of a #include statement in the 
                        C++ code.
        customize --   optional. base_info.custom_info object. An alternative
                       way to specify support_code, headers, etc. needed by
                       the function see the compiler.base_info module for more
                       details. (not sure this'll be used much).
        type_converters -- optional. list of type converters. These
                          guys are what convert Python data types to C/C++ data
                          types.  If you'd like to use a different set of type
                          conversions than the default, specify them here. Look
                          in the type conversions section of the main
                          documentation for examples.
        auto_downcast -- optional. 0 or 1. default 1.  This only affects
                         functions that have Numeric arrays as input variables.
                         Setting this to 1 will cause all floating point values
                         to be cast as float instead of double if all the
                         Numeric arrays are of type float.  If even one of the
                         arrays has type double or double complex, all
                         variables maintain there standard types.

        Distutils keywords.  These are cut and pasted from Greg Ward's
        distutils.extension.Extension class for convenience:

        sources : [string]
          list of source filenames, relative to the distribution root
          (where the setup script lives), in Unix form (slash-separated)
          for portability.  Source files may be C, C++, SWIG (.i),
          platform-specific resource files, or whatever else is recognized
          by the "build_ext" command as source for a Python extension.
          Note: The module_path file is always appended to the front of this
                list
        include_dirs : [string]
          list of directories to search for C/C++ header files (in Unix
          form for portability)
        define_macros : [(name : string, value : string|None)]
          list of macros to define; each macro is defined using a 2-tuple,
          where 'value' is either the string to define it to or None to
          define it without a particular value (equivalent of "#define
          FOO" in source or -DFOO on Unix C compiler command line)
        undef_macros : [string]
          list of macros to undefine explicitly
        library_dirs : [string]
          list of directories to search for C/C++ libraries at link time
        libraries : [string]
          list of library names (not filenames or paths) to link against
        runtime_library_dirs : [string]
          list of directories to search for C/C++ libraries at run time
          (for shared extensions, this is when the extension is loaded)
        extra_objects : [string]
          list of extra files to link with (eg. object files not implied
          by 'sources', static library that must be explicitly specified,
          binary resource files, etc.)
        extra_compile_args : [string]
          any extra platform- and compiler-specific information to use
          when compiling the source files in 'sources'.  For platforms and
          compilers where "command line" makes sense, this is typically a
          list of command-line arguments, but for other platforms it could
          be anything.
        extra_link_args : [string]
          any extra platform- and compiler-specific information to use
          when linking object files together to create the extension (or
          to create a new static Python interpreter).  Similar
          interpretation as for 'extra_compile_args'.
        export_symbols : [string]
          list of symbols to be exported from a shared extension.  Not
          used on all platforms, and not generally necessary for Python
          extensions, which typically export exactly one symbol: "init" +
          extension_name.
    """
    # this grabs the local variables from the *previous* call
    # frame -- that is the locals from the function that called
    # inline.
    global function_catalog

    call_frame = sys._getframe().f_back
    if local_dict is None:
        local_dict = call_frame.f_locals
    if global_dict is None:
        global_dict = call_frame.f_globals
    if force:
        module_dir = global_dict.get('__file__',None)
        func = compile_function(code,arg_names,local_dict,
                                global_dict,module_dir,
                                compiler=compiler,
                                verbose=verbose,
                                support_code = support_code,
                                headers = headers,
                                customize=customize,
                                type_converters = type_converters,
                                auto_downcast = auto_downcast,
                                **kw)

        function_catalog.add_function(code,func,module_dir)
        results = attempt_function_call(code,local_dict,global_dict)
    else:
        # 1. try local cache
        try:
            results = apply(function_cache[code],(local_dict,global_dict))
            return results
        except TypeError, msg: 
            msg = str(msg).strip()
            if msg[:16] == "Conversion Error":
                pass
            else:
                raise TypeError, msg
        except NameError, msg: 
            msg = str(msg).strip()
            if msg[:16] == "Conversion Error":
                pass
            else:
                raise NameError, msg
        except KeyError:
            pass
        # 2. try function catalog
        try:
            results = attempt_function_call(code,local_dict,global_dict)
        # 3. build the function
        except ValueError:
            # compile the library
            module_dir = global_dict.get('__file__',None)
            func = compile_function(code,arg_names,local_dict,
                                    global_dict,module_dir,
                                    compiler=compiler,
                                    verbose=verbose,
                                    support_code = support_code,
                                    headers = headers,
                                    customize=customize,
                                    type_converters = type_converters,
                                    auto_downcast = auto_downcast,
                                    **kw)

            function_catalog.add_function(code,func,module_dir)
            results = attempt_function_call(code,local_dict,global_dict)
    return results

def attempt_function_call(code,local_dict,global_dict):
    # we try 3 levels here -- a local cache first, then the
    # catalog cache, and then persistent catalog.
    #
    global function_cache
    # 2. try catalog cache.
    function_list = function_catalog.get_functions_fast(code)
    for func in function_list:
        try:
            results = apply(func,(local_dict,global_dict))
            function_catalog.fast_cache(code,func)
            function_cache[code] = func
            return results
        except TypeError, msg: # should specify argument types here.
            # This should really have its own error type, instead of
            # checking the beginning of the message, but I don't know
            # how to define that yet.
            msg = str(msg)
            if msg[:16] == "Conversion Error":
                pass
            else:
                raise TypeError, msg
        except NameError, msg: 
            msg = str(msg).strip()
            if msg[:16] == "Conversion Error":
                pass
            else:
                raise NameError, msg                
    # 3. try persistent catalog
    module_dir = global_dict.get('__file__',None)
    function_list = function_catalog.get_functions(code,module_dir)
    for func in function_list:
        try:
            results = apply(func,(local_dict,global_dict))
            function_catalog.fast_cache(code,func)
            function_cache[code] = func
            return results
        except: # should specify argument types here.
            pass
    # if we get here, the function wasn't found
    raise ValueError, 'function with correct signature not found'

def inline_function_code(code,arg_names,local_dict=None,
                         global_dict=None,auto_downcast = 1,
                         type_converters=None,compiler=''):
    call_frame = sys._getframe().f_back
    if local_dict is None:
        local_dict = call_frame.f_locals
    if global_dict is None:
        global_dict = call_frame.f_globals
    ext_func = inline_ext_function('compiled_func',code,arg_names,
                                   local_dict,global_dict,auto_downcast,
                                   type_converters = type_converters)
    import build_tools
    compiler = build_tools.choose_compiler(compiler)
    ext_func.set_compiler(compiler)
    return ext_func.function_code()

def compile_function(code,arg_names,local_dict,global_dict,
                     module_dir,
                     compiler='',
                     verbose = 0,
                     support_code = None,
                     headers = [],
                     customize = None,
                     type_converters = None,
                     auto_downcast=1,
                     **kw):
    # figure out where to store and what to name the extension module
    # that will contain the function.
    #storage_dir = catalog.intermediate_dir()
    module_path = function_catalog.unique_module_name(code,module_dir)
    storage_dir, module_name = os.path.split(module_path)
    mod = inline_ext_module(module_name,compiler)

    # create the function.  This relies on the auto_downcast and
    # type factories setting
    ext_func = inline_ext_function('compiled_func',code,arg_names,
                                   local_dict,global_dict,auto_downcast,
                                   type_converters = type_converters)
    mod.add_function(ext_func)

    # if customize (a custom_info object), then set the module customization.
    if customize:
        mod.customize = customize

    # add the extra "support code" needed by the function to the module.
    if support_code:
        mod.customize.add_support_code(support_code)
    
    # add the extra headers needed by the function to the module.
    for header in headers:
        mod.customize.add_header(header)
        
    # compile code in correct location, with the given compiler and verbosity
    # setting.  All input keywords are passed through to distutils
    mod.compile(location=storage_dir,compiler=compiler,
                verbose=verbose, **kw)

    # import the module and return the function.  Make sure
    # the directory where it lives is in the python path.
    try:
        sys.path.insert(0,storage_dir)
        exec 'import ' + module_name
        func = eval(module_name+'.compiled_func')
    finally:
        del sys.path[0]
    return func

