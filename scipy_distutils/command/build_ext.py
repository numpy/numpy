""" Modified version of build_ext that handles fortran source files and f2py 

    The f2py_sources() method is pretty much a copy of the swig_sources()
    method in the standard build_ext class , but modified to use f2py.  It
    also.
    
    slightly_modified_standard_build_extension() is a verbatim copy of
    the standard build_extension() method with a single line changed so that
    preprocess_sources() is called instead of just swig_sources().  This new
    function is a nice place to stick any source code preprocessing functions
    needed.
    
    build_extension() handles building any needed static fortran libraries
    first and then calls our slightly_modified_..._extenstion() to do the
    rest of the processing in the (mostly) standard way.    
"""

import os, string
from types import *

from distutils.dep_util import newer_group, newer
from distutils.command.build_ext import *
from distutils.command.build_ext import build_ext as old_build_ext

class build_ext (old_build_ext):
    user_options = old_build_ext.user_options + \
                   [('f2py-options=', None,
                      "command line arguments to f2py")]

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        self.f2py_options = None

    def finalize_options (self):        
        old_build_ext.finalize_options(self)
        if self.f2py_options is None:
            self.f2py_options = []
            
    def run (self):
        if self.distribution.has_f_libraries():
            build_flib = self.get_finalized_command('build_flib')
            self.libraries.extend(build_flib.get_library_names() or [])
            self.library_dirs.extend(build_flib.get_library_dirs() or [])
            #self.library_dirs.extend(build_flib.get_library_dirs() or [])
            #runtime_dirs = build_flib.get_runtime_library_dirs()
            #self.runtime_library_dirs.extend(runtime_dirs or [])
            
            #?? what is this ??
            self.library_dirs.append(build_flib.build_flib)
            
        old_build_ext.run(self)

    def preprocess_sources(self,sources,ext):
        sources = self.swig_sources(sources)
        if self.has_f2py_sources(sources):        
            sources = self.f2py_sources(sources,ext)
        return sources
        
    def extra_include_dirs(self,sources):
        if self.has_f2py_sources(sources):        
            import f2py2e
            d = os.path.dirname(f2py2e.__file__)
            return [os.path.join(d,'src')]
        else:
            return []    
        
    def build_extension(self, ext):
        # support for building static fortran libraries
        if self.distribution.has_f_libraries():
            build_flib = self.get_finalized_command('build_flib')
            moreargs = build_flib.fcompiler.get_extra_link_args()
            if moreargs != []:                
                if ext.extra_link_args is None:
                    ext.extra_link_args = moreargs
                else:
                    ext.extra_link_args += moreargs
            # be sure to include fortran runtime library directory names
            runtime_dirs = build_flib.get_runtime_library_dirs()
            ext.runtime_library_dirs.extend(runtime_dirs or [])
            linker_so = build_flib.fcompiler.get_linker_so()
            if linker_so is not None:
                self.compiler.linker_so = linker_so
        # end of fortran source support
        return old_build_ext.build_extension(self,ext)
        
        # f2py support handled slightly_modified..._extenstion.
        return self.slightly_modified_standard_build_extension(ext)
        
    def slightly_modified_standard_build_extension(self, ext):
        """
            This is pretty much a verbatim copy of the build_extension()
            function in distutils with a single change to make it possible
            to pre-process f2py as well as swig source files before 
            compilation.
        """
        sources = ext.sources
        if sources is None or type(sources) not in (ListType, TupleType):
            raise DistutilsSetupError, \
                  ("in 'ext_modules' option (extension '%s'), " +
                   "'sources' must be present and must be " +
                   "a list of source filenames") % ext.name
        sources = list(sources)

        fullname = self.get_ext_fullname(ext.name)
        if self.inplace:
            # ignore build-lib -- put the compiled extension into
            # the source tree along with pure Python modules

            modpath = string.split(fullname, '.')
            package = string.join(modpath[0:-1], '.')
            base = modpath[-1]

            build_py = self.get_finalized_command('build_py')
            package_dir = build_py.get_package_dir(package)
            ext_filename = os.path.join(package_dir,
                                        self.get_ext_filename(base))
        else:
            ext_filename = os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname))

        if not (self.force or newer_group(sources, ext_filename, 'newer')):
            self.announce("skipping '%s' extension (up-to-date)" %
                          ext.name)
            return
        else:
            self.announce("building '%s' extension" % ext.name)

        # I copied this hole stinken function just to change from
        # self.swig_sources to self.preprocess_sources...
        # ! must come before the next line!!
        include_dirs = self.extra_include_dirs(sources) + \
                      (ext.include_dirs or [])
        sources = self.preprocess_sources(sources, ext)
        
        # Next, compile the source code to object files.

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        # XXX and if we support CFLAGS, why not CC (compiler
        # executable), CPPFLAGS (pre-processor options), and LDFLAGS
        # (linker options) too?
        # XXX should we use shlex to properly parse CFLAGS?

        if os.environ.has_key('CFLAGS'):
            extra_args.extend(string.split(os.environ['CFLAGS']))

        objects = self.compiler.compile(sources,
                                        output_dir=self.build_temp,
                                        macros=macros,
                                        include_dirs=include_dirs,
                                        debug=self.debug,
                                        extra_postargs=extra_args)

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        self.compiler.link_shared_object(
            objects, ext_filename, 
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext), 
            debug=self.debug,
            build_temp=self.build_temp)

    def has_f2py_sources (self, sources):
        print sources
        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == ".pyf":             # f2py interface file
                return 1
        print 'no!'        
        return 0
                
    def f2py_sources (self, sources, ext):

        """Walk the list of source files in 'sources', looking for f2py
        interface (.pyf) files.  Run f2py on all that are found, and
        return a modified 'sources' list with f2py source files replaced
        by the generated C (or C++) files.
        """

        import f2py2e

        new_sources = []
        new_include_dirs = []
        f2py_sources = []
        f2py_targets = {}

        # XXX this drops generated C/C++ files into the source tree, which
        # is fine for developers who want to distribute the generated
        # source -- but there should be an option to put f2py output in
        # the temp dir.

        target_ext = 'module.c'
        target_dir = self.build_temp
        print 'target_dir', target_dir

        match_module = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                  re.I).match
        
        for source in sources:
            (base, source_ext) = os.path.splitext(source)
            (source_dir, base) = os.path.split(base)
            if source_ext == ".pyf":             # f2py interface file
                # get extension module name
                f = open(source)
                for line in f.xreadlines():
                    m = match_module(line)
                    if m:
                        base = m.group('name')
                        break
                f.close()
                if base != ext.name:
                    # XXX: Should we do here more than just warn?
                    print 'Warning: %s provides %s but this extension is %s' \
                          % (source,`base`,`ext`)

                target_file = os.path.join(target_dir,base+target_ext)
                new_sources.append(target_file)
                f2py_sources.append(source)
                f2py_targets[source] = new_sources[-1]
            else:
                new_sources.append(source)

        if not f2py_sources:
            return new_sources

        # a bit of a hack, but I think it'll work.  Just include one of
        # the fortranobject.c files that was copied into most 
        d = os.path.dirname(f2py2e.__file__)
        new_sources.append(os.path.join(d,'src','fortranobject.c'))
        self.include_dirs.append(os.path.join(d,'src'))

        f2py_opts = []
        f2py_options = ext.f2py_options + self.f2py_options
        for i in f2py_options:
            f2py_opts.append('--'+i)
        f2py_opts = ['--build-dir',target_dir] + f2py_opts
        
        # make sure the target dir exists
        from distutils.dir_util import mkpath
        mkpath(target_dir)

        for source in f2py_sources:
            target = f2py_targets[source]
            if newer(source,target):
                self.announce("f2py-ing %s to %s" % (source, target))
                self.announce("f2py-args: %s" % f2py_options)
                f2py2e.run_main(f2py_opts + [source])                
                #ext.sources.extend(pyf.data[ext.name].get('fsrc') or [])
                #self.distribution.fortran_sources_to_flib(ext)
        print new_sources
        return new_sources
    # f2py_sources ()

    def get_source_files (self):
        self.check_extensions_list(self.extensions)
        filenames = []

        # Get sources and any include files in the same directory.
        for ext in self.extensions:
            filenames.extend(ext.sources)
            filenames.extend(get_headers(get_directories(ext.sources)))

        return filenames

    def check_extensions_list (self, extensions):
        """
        Very slightly modified to add f2py_options as a flag... argh.
        
        Ensure that the list of extensions (presumably provided as a
        command option 'extensions') is valid, i.e. it is a list of
        Extension objects.  We also support the old-style list of 2-tuples,
        where the tuples are (ext_name, build_info), which are converted to
        Extension instances here.

        Raise DistutilsSetupError if the structure is invalid anywhere;
        just returns otherwise.
        """
        if type(extensions) is not ListType:
            raise DistutilsSetupError, \
                  "'ext_modules' option must be a list of Extension instances"
        
        for i in range(len(extensions)):
            ext = extensions[i]
            if isinstance(ext, Extension):
                continue                # OK! (assume type-checking done
                                        # by Extension constructor)

            (ext_name, build_info) = ext
            self.warn(("old-style (ext_name, build_info) tuple found in "
                       "ext_modules for extension '%s'" 
                       "-- please convert to Extension instance" % ext_name))
            if type(ext) is not TupleType and len(ext) != 2:
                raise DistutilsSetupError, \
                      ("each element of 'ext_modules' option must be an "
                       "Extension instance or 2-tuple")

            if not (type(ext_name) is StringType and
                    extension_name_re.match(ext_name)):
                raise DistutilsSetupError, \
                      ("first element of each tuple in 'ext_modules' "
                       "must be the extension name (a string)")

            if type(build_info) is not DictionaryType:
                raise DistutilsSetupError, \
                      ("second element of each tuple in 'ext_modules' "
                       "must be a dictionary (build info)")

            # OK, the (ext_name, build_info) dict is type-safe: convert it
            # to an Extension instance.
            ext = Extension(ext_name, build_info['sources'])

            # Easy stuff: one-to-one mapping from dict elements to
            # instance attributes.
            for key in ('include_dirs',
                        'library_dirs',
                        'libraries',
                        'extra_objects',
                        'extra_compile_args',
                        'extra_link_args',
                        'f2py_options'):
                val = build_info.get(key)
                if val is not None:
                    setattr(ext, key, val)

            # Medium-easy stuff: same syntax/semantics, different names.
            ext.runtime_library_dirs = build_info.get('rpath')
            if build_info.has_key('def_file'):
                self.warn("'def_file' element of build info dict "
                          "no longer supported")

            # Non-trivial stuff: 'macros' split into 'define_macros'
            # and 'undef_macros'.
            macros = build_info.get('macros')
            if macros:
                ext.define_macros = []
                ext.undef_macros = []
                for macro in macros:
                    if not (type(macro) is TupleType and
                            1 <= len(macro) <= 2):
                        raise DistutilsSetupError, \
                              ("'macros' element of build info dict "
                               "must be 1- or 2-tuple")
                    if len(macro) == 1:
                        ext.undef_macros.append(macro[0])
                    elif len(macro) == 2:
                        ext.define_macros.append(macro)

            extensions[i] = ext

        # for extensions

    # check_extensions_list ()
