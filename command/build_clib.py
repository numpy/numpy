""" Modified version of build_clib that handles fortran source files.
"""

from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError

from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import filter_sources, has_f_sources,\
     has_cxx_sources, all_strings, get_lib_source_files, is_sequence

try:
    set
except NameError:
    from sets import Set as set

# Fix Python distutils bug sf #1718574:
_l = old_build_clib.user_options
for _i in range(len(_l)):
    if _l[_i][0] in ['build-clib', 'build-temp']:
        _l[_i] = (_l[_i][0]+'=',)+_l[_i][1:]
#

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
        self._languages = None
        self.set_undefined_options('config_fc',
                                   ('fcompiler', 'fcompiler'))
        # we set this to the appropiate Fortran compiler object
        # (f77 or f90) in the .run() method
        self._fcompiler = None

    def languages(self):
        """Return a set of language names used in this library.
        Valid language names are 'c', 'f77', and 'f90'.
        """
        if self._languages is None:
            languages = set()
            for (lib_name, build_info) in self.libraries:
                l = build_info.get('language',None)
                if l:
                    languages.add(l)
            self._languages = languages
        return self._languages

    def have_f_sources(self):
        l = self.languages()
        return 'f90' in l or 'f77' in l

    def have_cxx_sources(self):
        l = self.languages()
        return 'c++' in l

    def run(self):
        if not self.libraries:
            return

        # Make sure that library sources are complete.
        self.run_command('build_src')

        languages = self.languages()

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
            if 'f90' in languages:
                fc = self.fcompiler.f90()
            else:
                fc = self.fcompiler.f77()
            libraries = self.libraries
            self.libraries = None
            fc.customize_cmd(self)
            self.libraries = libraries

            fc.show_customization()
            self._fcompiler = fc

        self.build_libraries(self.libraries)

    def get_source_files(self):
        self.check_library_list(self.libraries)
        filenames = []
        for lib in self.libraries:
            filenames.extend(get_lib_source_files(lib))
        return filenames

    def build_libraries(self, libraries):
        fcompiler = self._fcompiler
        compiler = self.compiler

        for (lib_name, build_info) in libraries:

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
