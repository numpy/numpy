""" Modified version of build_ext that handles fortran source files.
"""

import os, string
from types import *

from distutils.dep_util import newer_group, newer
from distutils.command.build_ext import *
from distutils.command.build_ext import build_ext as old_build_ext

class build_ext (old_build_ext):

    def build_extension(self, ext):
        # support for building static fortran libraries
        need_f_libs = 0
        ext_name = string.split(ext.name,'.')[-1]
        if self.distribution.has_f_libraries():
            build_flib = self.get_finalized_command('build_flib')
            if build_flib.has_f_library(ext_name):
                need_f_libs = 1
            else:
                for lib_name in ext.libraries:
                    if build_flib.has_f_library(lib_name):
                        need_f_libs = 1
                        break
        print ext.name,ext_name,'needs fortran libraries',need_f_libs
        if need_f_libs:
            moreargs = build_flib.fcompiler.get_extra_link_args()
            if moreargs != []:
                if ext.extra_link_args is None:
                    ext.extra_link_args = moreargs
                else:
                    ext.extra_link_args += moreargs
            if build_flib.has_f_library(ext_name) and \
               ext_name not in ext.libraries:
                ext.libraries.append(ext_name)
            for lib_name in ext.libraries[:]:
                ext.libraries.extend(build_flib.get_library_names(lib_name))
                ext.library_dirs.extend(build_flib.get_library_dirs(lib_name))

            ext.libraries.extend(build_flib.get_fcompiler_library_names())
            ext.library_dirs.extend(build_flib.get_fcompiler_library_dirs())

            print ext.libraries
            
            ext.library_dirs.append(build_flib.build_flib)
            runtime_dirs = build_flib.get_runtime_library_dirs()
            ext.runtime_library_dirs.extend(runtime_dirs or [])
            linker_so = build_flib.fcompiler.get_linker_so()
            if linker_so is not None:
                self.compiler.linker_so = linker_so

        # end of fortran source support
        return old_build_ext.build_extension(self,ext)

    def get_source_files (self):
        self.check_extensions_list(self.extensions)
        filenames = []

        # Get sources and any include files in the same directory.
        for ext in self.extensions:
            filenames.extend(ext.sources)
            filenames.extend(get_headers(get_directories(ext.sources)))

        return filenames
