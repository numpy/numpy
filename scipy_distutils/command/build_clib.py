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

from scipy_distutils import log
from distutils.dep_util import newer_group
from scipy_distutils.misc_util import filter_sources, has_f_sources

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

    def run(self):
        if not self.libraries:
            return
        old_build_clib.run(self)   # sets self.compiler
        if self.have_f_sources():
            from scipy_distutils.fcompiler import new_fcompiler
            self.fcompiler = new_fcompiler(compiler=self.fcompiler,
                                           verbose=self.verbose,
                                           dry_run=self.dry_run,
                                           force=self.force)
            self.fcompiler.customize(self.distribution)

        #XXX: C++ linker support, see build_ext2.py

        self.build_libraries2(self.libraries)
        return

    def build_libraries(self, libraries):
        # Hold on building libraries in old_build_clib.run()
        # until Fortran/C++ compilers are set. Building will be
        # carried out in build_libraries2()
        return

    def get_source_files(self):
        if sys.version[:3]>='2.2':
            filenames = old_build_clib.get_source_files(self)
        else:
            for (lib_name, build_info) in self.libraries:
                filenames.extend(build_info.get('sources',[]))
        filenames.extend(get_headers(get_directories(filenames)))
        return filenames

    def build_libraries2(self, libraries):

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

            c_sources, cxx_sources, f_sources, fmodule_sources \
                       = filter_sources(sources)

            if fmodule_sources:
                print 'XXX: Fortran 90 module support not implemented or tested'
                f_sources.extend(fmodule_sources)

            if cxx_sources:
                print 'XXX: C++ linker support not implemented or tested'
            objects = compiler.compile(c_sources+cxx_sources,
                                       output_dir=self.build_temp,
                                       macros=macros,
                                       include_dirs=include_dirs,
                                       debug=self.debug)

            if f_sources:
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
