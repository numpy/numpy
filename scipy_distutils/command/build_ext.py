""" Modified version of build_ext that handles fortran source files.
"""

import os, string
from types import *

from distutils.dep_util import newer_group, newer
from distutils.command.build_ext import build_ext as old_build_ext

from scipy_distutils.command.build_clib import get_headers,get_directories
from scipy_distutils import misc_util


class build_ext (old_build_ext):

    def finalize_options (self):
        old_build_ext.finalize_options(self)
        extra_includes = misc_util.get_environ_include_dirs()
        self.include_dirs.extend(extra_includes)
        
    def build_extension(self, ext):
        
        # The MSVC compiler doesn't have a linker_so attribute.
        # Giving it a dummy one of None seems to do the trick.
        if not hasattr(self.compiler,'linker_so'):
            self.compiler.linker_so = None
            
        #XXX: anything else we need to save?
        save_linker_so = self.compiler.linker_so
        save_compiler_libs = self.compiler.libraries
        save_compiler_libs_dirs = self.compiler.library_dirs
        
        # support for building static fortran libraries
        need_f_libs = 0
        need_f_opts = getattr(ext,'need_fcompiler_opts',0)
        ext_name = string.split(ext.name,'.')[-1]
        flib_name = ext_name + '_f2py'

        if self.distribution.has_f_libraries():
            build_flib = self.get_finalized_command('build_flib')
            if build_flib.has_f_library(flib_name):
                need_f_libs = 1
            else:
                for lib_name in ext.libraries:
                    if build_flib.has_f_library(lib_name):
                        need_f_libs = 1
                        break
        elif need_f_opts:
            build_flib = self.get_finalized_command('build_flib')

        #self.announce('%s %s needs fortran libraries %s %s'%(\
        #    ext.name,ext_name,need_f_libs,need_f_opts))
        
        if need_f_libs:
            if build_flib.has_f_library(flib_name) and \
               flib_name not in ext.libraries:
                ext.libraries.insert(0,flib_name)
            for lib_name in ext.libraries[:]:
                ext.libraries.extend(build_flib.get_library_names(lib_name))
                ext.library_dirs.extend(build_flib.get_library_dirs(lib_name))
            d = build_flib.build_flib
            if d not in ext.library_dirs:
                ext.library_dirs.append(d)

        if need_f_libs or need_f_opts:
            moreargs = build_flib.fcompiler.get_extra_link_args()
            if moreargs != []:
                if ext.extra_link_args is None:
                    ext.extra_link_args = moreargs
                else:
                    ext.extra_link_args += moreargs

            runtime_dirs = build_flib.get_runtime_library_dirs()
            ext.runtime_library_dirs.extend(runtime_dirs or [])

            linker_so = build_flib.fcompiler.get_linker_so()

            if linker_so is not None:
                if linker_so is not save_linker_so:
                    self.announce('replacing linker_so %r with %r' %(\
                        ' '.join(save_linker_so),
                        ' '.join(linker_so)))
                    self.compiler.linker_so = linker_so
                    l = build_flib.get_fcompiler_library_names()
                    #l = self.compiler.libraries + l
                    self.compiler.libraries = l
                    l = ( self.compiler.library_dirs +
                          build_flib.get_fcompiler_library_dirs() )
                    self.compiler.library_dirs = l
            else:
                libs = build_flib.get_fcompiler_library_names()
                for lib in libs:
                    if lib not in self.compiler.libraries:
                        self.compiler.libraries.append(lib)

                lib_dirs = build_flib.get_fcompiler_library_dirs()
                for lib_dir in lib_dirs:
                    if lib_dir not in self.compiler.library_dirs:
                        self.compiler.library_dirs.append(lib_dir)
        # check for functions existence so that we can mix distutils
        # extension with scipy_distutils functions without breakage
        elif (hasattr(ext,'has_cxx_sources') and
              ext.has_cxx_sources()):

            if save_linker_so and save_linker_so[0]=='gcc':
                #XXX: need similar hooks that are in weave.build_tools.py
                #     Or more generally, implement cxx_compiler_class
                #     hooks similar to fortran_compiler_class.
                linker_so = ['g++'] + save_linker_so[1:]
                self.compiler.linker_so = linker_so
                self.announce('replacing linker_so %r with %r' %(\
                        ' '.join(save_linker_so),
                        ' '.join(linker_so)))

        # end of fortran source support
        res = old_build_ext.build_extension(self,ext)

        if save_linker_so is not self.compiler.linker_so:
            self.announce('restoring linker_so %r' % ' '.join(save_linker_so))
            self.compiler.linker_so = save_linker_so
            self.compiler.libraries = save_compiler_libs
            self.compiler.library_dirs = save_compiler_libs_dirs

        return res

    def get_source_files (self):
        self.check_extensions_list(self.extensions)
        filenames = []

        # Get sources and any include files in the same directory.
        for ext in self.extensions:
            filenames.extend(ext.sources)
            filenames.extend(get_headers(get_directories(ext.sources)))

        return filenames

    
