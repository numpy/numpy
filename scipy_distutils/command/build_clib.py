""" Modified version of build_clib that handles fortran source files.
"""

import os
import string
import sys
import re
from glob import glob
from types import *
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.command.build_clib import show_compilers

from scipy_distutils import log, misc_util
from distutils.dep_util import newer_group
from scipy_distutils.misc_util import filter_sources, \
     has_f_sources, has_cxx_sources

def get_headers(directory_list):
    # get *.h files from list of directories
    headers = []
    for dir in directory_list:
        head = glob(os.path.join(dir,"*.h")) #XXX: *.hpp files??
        headers.extend(head)

    return headers

def get_directories(list_of_sources):
    # get unique directories from list of sources.
    direcs = []
    for file in list_of_sources:
        dir = os.path.split(file)
        if dir[0] != '' and not dir[0] in direcs:
            direcs.append(dir[0])
    return direcs

class build_clib(old_build_clib):

    description = "build C/C++/F libraries used by Python extensions"

    user_options = old_build_clib.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ]

    def initialize_options(self):
        old_build_clib.initialize_options(self)
        self.fcompiler = None
        return

    def finalize_options(self):
        old_build_clib.finalize_options(self)
        self.set_undefined_options('build_ext',
                                   ('fcompiler', 'fcompiler'))

        #XXX: This is hackish and probably unnecessary,
        #     could we get rid of this?
        from scipy_distutils import misc_util
        extra_includes = misc_util.get_environ_include_dirs()
        if extra_includes:
            print "XXX: are you sure you'll need PYTHONINCLUDES env. variable??"
        self.include_dirs.extend(extra_includes)

        return

    def have_f_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_f_sources(build_info.get('sources',[])):
                return 1
        return 0

    def have_cxx_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_cxx_sources(build_info.get('sources',[])):
                return 1
        return 0

    def run(self):
        if not self.libraries:
            return

        # Make sure that library sources are complete.
        for (lib_name, build_info) in self.libraries:
            if not misc_util.all_strings(build_info.get('sources',[])):
                raise TypeError,'Library "%s" sources contains unresolved'\
                      ' items (call build_src before built_clib).' % (lib_name)

        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=self.compiler,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution,need_cxx=self.have_cxx_sources())

        libraries = self.libraries
        self.libraries = None
        self.compiler.customize_cmd(self)
        self.libraries = libraries

        self.compiler.show_customization()

        if self.have_f_sources():
            from scipy_distutils.fcompiler import new_fcompiler
            self.fcompiler = new_fcompiler(compiler=self.fcompiler,
                                           verbose=self.verbose,
                                           dry_run=self.dry_run,
                                           force=self.force)
            self.fcompiler.customize(self.distribution)
    
            libraries = self.libraries
            self.libraries = None
            self.fcompiler.customize_cmd(self)
            self.libraries = libraries

            self.fcompiler.show_customization()

        self.build_libraries(self.libraries)
        return

    def get_source_files(self):
        from build_ext import is_local_src_dir
        self.check_library_list(self.libraries)
        filenames = []
        def visit_func(filenames,dirname,names):
            if os.path.basename(dirname) in ['CVS','.svn']:
                names[:] = []
                return
            for name in names:
                if name[-1] in "#~":
                    continue
                fullname = os.path.join(dirname,name)
                if os.path.isfile(fullname):
                    filenames.append(fullname)
        for (lib_name, build_info) in self.libraries:
            sources = build_info.get('sources',[])
            sources = filter(lambda s:type(s) is StringType,sources)
            filenames.extend(sources)
            filenames.extend(get_headers(get_directories(sources)))
            depends = build_info.get('depends',[])
            for d in depends:
                if is_local_src_dir(d):
                    os.path.walk(d,visit_func,filenames)
                elif os.path.isfile(d):
                    filenames.append(d)
        return filenames

    def build_libraries(self, libraries):

        compiler = self.compiler
        fcompiler = self.fcompiler

        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or type(sources) not in (ListType, TupleType):
                raise DistutilsSetupError, \
                      ("in 'libraries' option (library '%s'), " +
                       "'sources' must be present and must be " +
                       "a list of source filenames") % lib_name
            sources = list(sources)

            lib_file = compiler.library_filename(lib_name,
                                                 output_dir=self.build_clib)

            depends = sources + build_info.get('depends',[])
            if not (self.force or newer_group(depends, lib_file, 'newer')):
                log.debug("skipping '%s' library (up-to-date)", lib_name)
                continue
            else:
                log.info("building '%s' library", lib_name)

            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')
            extra_postargs = build_info.get('extra_compiler_args') or []

            c_sources, cxx_sources, f_sources, fmodule_sources \
                       = filter_sources(sources)

            if self.compiler.compiler_type=='msvc':
                # this hack works around the msvc compiler attributes
                # problem, msvc uses its own convention :(
                c_sources += cxx_sources
                cxx_sources = []

            if fmodule_sources:
                print 'XXX: Fortran 90 module support not implemented or tested'
                f_sources.extend(fmodule_sources)

            objects = []
            if c_sources:
                log.info("compiling C sources")
                objects = compiler.compile(c_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs)

            if cxx_sources:
                log.info("compiling C++ sources")
                old_compiler = self.compiler.compiler_so[0]
                self.compiler.compiler_so[0] = self.compiler.compiler_cxx[0]

                cxx_objects = compiler.compile(cxx_sources,
                                               output_dir=self.build_temp,
                                               macros=macros,
                                               include_dirs=include_dirs,
                                               debug=self.debug,
                                               extra_postargs=extra_postargs)
                objects.extend(cxx_objects)

                self.compiler.compiler_so[0] = old_compiler

            if f_sources:
                log.info("compiling Fortran sources")
                f_objects = fcompiler.compile(f_sources,
                                              output_dir=self.build_temp,
                                              macros=macros,
                                              include_dirs=include_dirs,
                                              debug=self.debug,
                                              extra_postargs=[])
                objects.extend(f_objects)

            self.compiler.create_static_lib(objects, lib_name,
                                            output_dir=self.build_clib,
                                            debug=self.debug)
        return
