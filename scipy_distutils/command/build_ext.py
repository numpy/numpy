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
import f2py2e

class build_ext (old_build_ext):
    def run (self):
        if self.distribution.has_f_libraries():
            build_flib = self.get_finalized_command('build_flib')
            self.libraries.extend(build_flib.get_library_names() or [])
            self.library_dirs.extend(build_flib.get_library_dirs() or [])
            self.library_dirs.extend(build_flib.get_library_dirs() or [])
            #runtime_dirs = build_flib.get_runtime_library_dirs()
            #self.runtime_library_dirs.extend(runtime_dirs or [])
            self.library_dirs.append(build_flib.build_flib)
            
        old_build_ext.run(self)

    def preprocess_sources(self,sources):
        sources = self.swig_sources(sources)        
        sources = self.f2py_sources(sources)
        return sources
        
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
        # end of fortran source support
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
        sources = self.preprocess_sources(sources)

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
                                        include_dirs=ext.include_dirs,
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

    def f2py_sources (self, sources):

        """Walk the list of source files in 'sources', looking for f2py
        interface (.i) files.  Run f2py on all that are found, and
        return a modified 'sources' list with f2py source files replaced
        by the generated C (or C++) files.
        """

        new_sources = []
        f2py_sources = []
        f2py_targets = {}

        # XXX this drops generated C/C++ files into the source tree, which
        # is fine for developers who want to distribute the generated
        # source -- but there should be an option to put f2py output in
        # the temp dir.

        target_ext = 'module.c'

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == ".pyf":             # f2py interface file
                new_sources.append(base + target_ext)
                f2py_sources.append(source)
                f2py_targets[source] = new_sources[-1]
            else:
                new_sources.append(source)

        if not f2py_sources:
            return new_sources

        # a bit of a hack, but I think it'll work.  Just include one of
        # the fortranobject.c files that was copied into most 
        d,f = os.path.split(f2py_sources[0])
        new_sources.append(os.path.join(d,'fortranobject.c'))

        f2py_opts = ['--no-wrap-functions', '--no-latex-doc',
                     '--no-makefile','-no-setup']
        for source in f2py_sources:
            target = f2py_targets[source]
            if newer(source,target):
                self.announce("f2py-ing %s to %s" % (source, target))
                f2py2e.run_main(f2py_opts + [source])

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
