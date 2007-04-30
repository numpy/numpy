""" Modified version of build_ext that handles fortran source files.
"""

import os
import string
import sys
from glob import glob

from distutils.dep_util import newer_group
from distutils.command.build_ext import build_ext as old_build_ext
from distutils.errors import DistutilsFileError, DistutilsSetupError
from distutils.file_util import copy_file

from numpy.distutils import log
from numpy.distutils.exec_command import exec_command
from numpy.distutils.system_info import combine_paths
from numpy.distutils.misc_util import filter_sources, has_f_sources, \
     has_cxx_sources, get_ext_source_files, all_strings, \
     get_numpy_include_dirs, is_sequence


def ext_language(ext):
    return getattr(ext, 'language', 'c')

class build_ext (old_build_ext):

    description = "build C/C++/F extensions (compile/link to build directory)"

    user_options = old_build_ext.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ]

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        self.fcompiler = None

    def finalize_options(self):
        incl_dirs = self.include_dirs
        old_build_ext.finalize_options(self)
        if incl_dirs is not None:
            self.include_dirs.extend(self.distribution.include_dirs or [])
        self.set_undefined_options('config_fc',
                                   ('fcompiler', 'fcompiler'))
        self._fcompiler = None

    def initialize_fcompiler(self, build_clib):
        # Determine if Fortran compiler is needed.
        requiref77 = requiref90 = False
        if build_clib:
            lang = build_clib.languages()
            requiref77 = 'f77' in lang
            requiref90 = 'f90' in lang
        else:
            for ext in self.extensions:
                language = ext_language(ext)
                if language == 'f77':
                    requiref77 = True
                elif language == 'f90':
                    requiref90 = True
                elif has_f_sources(ext.sources):
                    # because we don't know any better, assume F77
                    requiref77 = True

        if not (requiref77 or requiref90):
            return

        if requiref90:
            self.fcompiler.need_f90()
        if requiref77:
            self.fcompiler.need_f77()

        fc = self.fcompiler.fortran(requiref90)
        if fc.get_version():
            fc.customize_cmd(self)
            fc.show_customization()
        else:
            self.warn('fcompiler=%s is not available.' % (
                fc.compiler_type,))
        self._fcompiler = fc

    def run(self):
        if not self.extensions:
            return

        # Make sure that extension sources are complete.
        self.run_command('build_src')

        # Not including C libraries to the list of
        # extension libraries automatically to prevent
        # bogus linking commands. Extensions must
        # explicitly specify the C libraries that they use.
        if self.distribution.has_c_libraries():
            self.run_command('build_clib')
            build_clib = self.get_finalized_command('build_clib')
            self.library_dirs.append(build_clib.build_clib)
        else:
            build_clib = None

        # Determine if C++ compiler is needed.
        need_cxx_compiler = False
        for ext in self.extensions:
            if has_cxx_sources(ext.sources):
                need_cxx_compiler = True
                break
            if getattr(ext,'language','c') == 'c++':
                need_cxx_compiler = True
                break

        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=self.compiler,
                                     verbose=self.verbose,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution,need_cxx=need_cxx_compiler)
        self.compiler.customize_cmd(self)
        self.compiler.show_customization()

        self.initialize_fcompiler(build_clib)

        # Build extensions
        self.build_extensions()

    def swig_sources(self, sources):
        # Do nothing. Swig sources have beed handled in build_src command.
        return sources

    def get_fortran_objects(self, ext, f_sources, fmodule_sources,
                            macros, include_dirs):
        if not f_sources and not fmodule_sources:
            return None, []

        fcompiler = self._fcompiler

        extra_postargs = []
        module_dirs = ext.module_dirs[:]

        macros = []

        if fmodule_sources:
            module_build_dir = os.path.join(
                self.build_temp,os.path.dirname(
                self.get_ext_filename(fullname)))

            self.mkpath(module_build_dir)
            if fcompiler.module_dir_switch is None:
                existing_modules = glob('*.mod')
            extra_postargs += fcompiler.module_options(\
                module_dirs,module_build_dir)

        f_objects = []
        if fmodule_sources:
            log.info("compiling Fortran 90 module sources")
            f_objects = fcompiler.compile(fmodule_sources,
                                          output_dir=self.build_temp,
                                          macros=macros,
                                          include_dirs=include_dirs,
                                          debug=self.debug,
                                          extra_postargs=extra_postargs,
                                          depends=ext.depends)

        if fmodule_sources and fcompiler.module_dir_switch is None:
            for f in glob('*.mod'):
                if f in existing_modules:
                    continue
                try:
                    self.move_file(f, module_build_dir)
                except DistutilsFileError:  # already exists in destination
                    os.remove(f)

        if f_sources:
            log.info("compiling Fortran sources")
            f_objects += fcompiler.compile(f_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs,
                                           depends=ext.depends)

        return fcompiler, f_objects

    def build_extension(self, ext):
        sources = ext.sources
        if sources is None or not is_sequence(sources):
            raise DistutilsSetupError, \
                  ("in 'ext_modules' option (extension '%s'), " +
                   "'sources' must be present and must be " +
                   "a list of source filenames") % ext.name
        sources = list(sources)

        if not sources:
            return

        fullname = self.get_ext_fullname(ext.name)
        if self.inplace:
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
        depends = sources + ext.depends

        if not (self.force or newer_group(depends, ext_filename, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        extra_args = ext.extra_compile_args or []
        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        clib_libraries = []
        clib_library_dirs = []
        if self.distribution.libraries:
            for libname,build_info in self.distribution.libraries:
                if libname in ext.libraries:
                    macros.extend(build_info.get('macros',[]))
                    clib_libraries.extend(build_info.get('libraries',[]))
                    clib_library_dirs.extend(build_info.get('library_dirs',[]))

        c_sources, cxx_sources, f_sources, fmodule_sources = \
                   filter_sources(ext.sources)
        if self.compiler.compiler_type=='msvc':
            if cxx_sources:
                # Needed to compile kiva.agg._agg extension.
                extra_args.append('/Zm1000')
            # this hack works around the msvc compiler attributes
            # problem, msvc uses its own convention :(
            c_sources += cxx_sources
            cxx_sources = []


        kws = {'depends':ext.depends}
        output_dir = self.build_temp

        include_dirs = ext.include_dirs + get_numpy_include_dirs()

        c_objects = []
        if c_sources:
            log.info("compiling C sources")
            c_objects = self.compiler.compile(c_sources,
                                              output_dir=output_dir,
                                              macros=macros,
                                              include_dirs=include_dirs,
                                              debug=self.debug,
                                              extra_postargs=extra_args,
                                              **kws)
        if cxx_sources:
            log.info("compiling C++ sources")

            cxx_compiler = self.compiler.cxx_compiler()

            c_objects += cxx_compiler.compile(cxx_sources,
                                              output_dir=output_dir,
                                              macros=macros,
                                              include_dirs=include_dirs,
                                              debug=self.debug,
                                              extra_postargs=extra_args,
                                              **kws)

        fcompiler, f_objects = self.get_fortran_objects(ext,
                                                        f_sources,
                                                        fmodule_sources,
                                                        macros, include_dirs)

        objects = c_objects + f_objects

        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        linker = self.compiler.link_shared_object

        use_fortran_linker = getattr(ext,'language','c') in ['f77','f90']
        c_libraries = []
        c_library_dirs = []
        if not use_fortran_linker and self.distribution.has_c_libraries():
            build_clib = self.get_finalized_command('build_clib')
            f_libs = []
            for (lib_name, build_info) in build_clib.libraries:
                if has_f_sources(build_info.get('sources',[])):
                    f_libs.append(lib_name)
                if lib_name in ext.libraries:
                    # XXX: how to determine if c_libraries contain
                    # fortran compiled sources?
                    c_libraries.extend(build_info.get('libraries',[]))
                    c_library_dirs.extend(build_info.get('library_dirs',[]))
            for l in ext.libraries:
                if l in f_libs:
                    use_fortran_linker = True
                    fcompiler = self.fcompiler.fortran()
                    break

        if use_fortran_linker and not fcompiler:
            fcompiler = self.fcompiler.fortran()

        # Always use system linker when using MSVC compiler.
        if self.compiler.compiler_type=='msvc' and use_fortran_linker:
            self._libs_with_msvc_and_fortran(c_libraries, c_library_dirs)
            use_fortran_linker = False

        if use_fortran_linker:
            if cxx_sources:
                # XXX: Which linker should be used, Fortran or C++?
                log.warn('mixing Fortran and C++ is untested')
            linker = fcompiler.link_shared_object
            language = ext.language or fcompiler.detect_language(f_sources)
        else:
            linker = self.compiler.link_shared_object
            if sys.version[:3]>='2.3':
                language = ext.language or self.compiler.detect_language(sources)
            else:
                language = ext.language
            if cxx_sources:
                linker = self.compiler.cxx_compiler().link_shared_object

        if sys.version[:3]>='2.3':
            kws = {'target_lang':language}
        else:
            kws = {}

        linker(objects, ext_filename,
               libraries=self.get_libraries(ext) + c_libraries + clib_libraries,
               library_dirs=ext.library_dirs+c_library_dirs+clib_library_dirs,
               runtime_library_dirs=ext.runtime_library_dirs,
               extra_postargs=extra_args,
               export_symbols=self.get_export_symbols(ext),
               debug=self.debug,
               build_temp=self.build_temp,**kws)

    def _libs_with_msvc_and_fortran(self, c_libraries, c_library_dirs):
        # Always use system linker when using MSVC compiler.
        f_lib_dirs = []
        fcompiler = self.fcompiler.fortran()
        for dir in fcompiler.library_dirs:
            # correct path when compiling in Cygwin but with normal Win
            # Python
            if dir.startswith('/usr/lib'):
                s,o = exec_command(['cygpath', '-w', dir], use_tee=False)
                if not s:
                    dir = o
            f_lib_dirs.append(dir)
        c_library_dirs.extend(f_lib_dirs)

        # make g77-compiled static libs available to MSVC
        lib_added = False
        for lib in fcompiler.libraries:
            if not lib.startswith('msvcr'):
                c_libraries.append(lib)
                p = combine_paths(f_lib_dirs, 'lib' + lib + '.a')
                if p:
                    dst_name = os.path.join(self.build_temp, lib + '.lib')
                    copy_file(p[0], dst_name)
                    if not lib_added:
                        c_library_dirs.append(self.build_temp)
                        lib_added = True

    def get_source_files (self):
        self.check_extensions_list(self.extensions)
        filenames = []
        for ext in self.extensions:
            filenames.extend(get_ext_source_files(ext))
        return filenames

    def get_outputs (self):
        self.check_extensions_list(self.extensions)

        outputs = []
        for ext in self.extensions:
            if not ext.sources:
                continue
            fullname = self.get_ext_fullname(ext.name)
            outputs.append(os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname)))
        return outputs
