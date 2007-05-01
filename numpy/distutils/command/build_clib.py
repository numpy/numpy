""" Modified version of build_clib that handles fortran source files.
"""

from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError

from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import filter_sources, has_f_sources,\
     has_cxx_sources, all_strings, get_lib_source_files, is_sequence

class build_clib(old_build_clib):

    description = "build C/C++/F libraries used by Python extensions"

    user_options = old_build_clib.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ]

    def initialize_options(self):
        old_build_clib.initialize_options(self)
        self.fcompiler = None

    def finalize_options(self):
        old_build_clib.finalize_options(self)
        self.set_undefined_options('build_ext',
                                   ('fcompiler', 'fcompiler'))

    def have_f_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_f_sources(build_info.get('sources',[])):
                return True
        return False

    def have_cxx_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_cxx_sources(build_info.get('sources',[])):
                return True
        return False

    def run(self):
        if not self.libraries:
            return

        # Make sure that library sources are complete.
        languages = []
        for (lib_name, build_info) in self.libraries:
            if not all_strings(build_info.get('sources',[])):
                self.run_command('build_src')
            l = build_info.get('language',None)
            if l and l not in languages: languages.append(l)

        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=self.compiler,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution,
                                need_cxx=self.have_cxx_sources())

        libraries = self.libraries
        self.libraries = None
        self.compiler.customize_cmd(self)
        self.libraries = libraries

        self.compiler.show_customization()

        if self.have_f_sources():
            from numpy.distutils.fcompiler import new_fcompiler
            self.fcompiler = new_fcompiler(compiler=self.fcompiler,
                                           verbose=self.verbose,
                                           dry_run=self.dry_run,
                                           force=self.force,
                                           requiref90='f90' in languages)
            self.fcompiler.customize(self.distribution)

            libraries = self.libraries
            self.libraries = None
            self.fcompiler.customize_cmd(self)
            self.libraries = libraries

            self.fcompiler.show_customization()

        self.build_libraries(self.libraries)

    def get_source_files(self):
        self.check_library_list(self.libraries)
        filenames = []
        for lib in self.libraries:
            filenames.extend(get_lib_source_files(lib))
        return filenames

    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            # default compilers
            compiler = self.compiler
            fcompiler = self.fcompiler

            sources = build_info.get('sources')
            if sources is None or not is_sequence(sources):
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


            config_fc = build_info.get('config_fc',{})
            if fcompiler is not None and config_fc:
                log.info('using additional config_fc from setup script '\
                         'for fortran compiler: %s' \
                         % (config_fc,))
                from numpy.distutils.fcompiler import new_fcompiler
                requiref90 = build_info.get('language','c')=='f90'
                fcompiler = new_fcompiler(compiler=self.fcompiler.compiler_type,
                                          verbose=self.verbose,
                                          dry_run=self.dry_run,
                                          force=self.force,
                                          requiref90=requiref90)
                dist = self.distribution
                base_config_fc = dist.get_option_dict('config_fc').copy()
                base_config_fc.update(config_fc)
                fcompiler.customize(base_config_fc)

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
                cxx_compiler = compiler.cxx_compiler()
                cxx_objects = cxx_compiler.compile(cxx_sources,
                                                   output_dir=self.build_temp,
                                                   macros=macros,
                                                   include_dirs=include_dirs,
                                                   debug=self.debug,
                                                   extra_postargs=extra_postargs)
                objects.extend(cxx_objects)

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

            clib_libraries = build_info.get('libraries',[])
            for lname, binfo in libraries:
                if lname in clib_libraries:
                    clib_libraries.extend(binfo[1].get('libraries',[]))
            if clib_libraries:
                build_info['libraries'] = clib_libraries
