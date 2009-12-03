import os
import sys
import os.path
from os.path import join as pjoin, dirname as pdirname

from distutils.errors import DistutilsPlatformError
from distutils.errors import DistutilsExecError, DistutilsSetupError

from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.ccompiler import CCompiler, new_compiler
from numpy.distutils.fcompiler import FCompiler, new_fcompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils import log
from numpy.distutils.misc_util import is_bootstrapping, get_cmd
from numpy.distutils.misc_util import get_numpy_include_dirs as _incdir
from numpy.distutils.compat import get_exception

# A few notes:
#   - numscons is not mandatory to build numpy, so we cannot import it here.
#   Any numscons import has to happen once we check numscons is available and
#   is required for the build (call through setupscons.py or native numscons
#   build).
def get_scons_build_dir():
    """Return the top path where everything produced by scons will be put.

    The path is relative to the top setup.py"""
    from numscons import get_scons_build_dir
    return get_scons_build_dir()

def get_scons_pkg_build_dir(pkg):
    """Return the build directory for the given package (foo.bar).

    The path is relative to the top setup.py"""
    from numscons.core.utils import pkg_to_path
    return pjoin(get_scons_build_dir(), pkg_to_path(pkg))

def get_scons_configres_dir():
    """Return the top path where everything produced by scons will be put.

    The path is relative to the top setup.py"""
    from numscons import get_scons_configres_dir
    return get_scons_configres_dir()

def get_scons_configres_filename():
    """Return the top path where everything produced by scons will be put.

    The path is relative to the top setup.py"""
    from numscons import get_scons_configres_filename
    return get_scons_configres_filename()

def get_scons_local_path():
    """This returns the full path where scons.py for scons-local is located."""
    from numscons import get_scons_path
    return get_scons_path()

def _get_top_dir(pkg):
    # XXX: this mess is necessary because scons is launched per package, and
    # has no knowledge outside its build dir, which is package dependent. If
    # one day numscons does not launch one process/package, this will be
    # unnecessary.
    from numscons import get_scons_build_dir
    from numscons.core.utils import pkg_to_path
    scdir = pjoin(get_scons_build_dir(), pkg_to_path(pkg))
    n = scdir.count(os.sep)
    return os.sep.join([os.pardir for i in range(n+1)])

def get_distutils_libdir(cmd, pkg):
    """Returns the path where distutils install libraries, relatively to the
    scons build directory."""
    return pjoin(_get_top_dir(pkg), cmd.build_lib)

def get_distutils_clibdir(cmd, pkg):
    """Returns the path where distutils put pure C libraries."""
    return pjoin(_get_top_dir(pkg), cmd.build_clib)

def get_distutils_install_prefix(pkg, inplace):
    """Returns the installation path for the current package."""
    from numscons.core.utils import pkg_to_path
    if inplace == 1:
        return pkg_to_path(pkg)
    else:
        install_cmd = get_cmd('install').get_finalized_command('install')
        return pjoin(install_cmd.install_libbase, pkg_to_path(pkg))

def get_python_exec_invoc():
    """This returns the python executable from which this file is invocated."""
    # Do we  need to take into account the PYTHONPATH, in a cross platform way,
    # that is the string returned can be executed directly on supported
    # platforms, and the sys.path of the executed python should be the same
    # than the caller ? This may not be necessary, since os.system is said to
    # take into accound os.environ. This actually also works for my way of
    # using "local python", using the alias facility of bash.
    return sys.executable

def get_numpy_include_dirs(sconscript_path):
    """Return include dirs for numpy.

    The paths are relatively to the setup.py script path."""
    from numscons import get_scons_build_dir
    scdir = pjoin(get_scons_build_dir(), pdirname(sconscript_path))
    n = scdir.count(os.sep)

    dirs = _incdir()
    rdirs = []
    for d in dirs:
        rdirs.append(pjoin(os.sep.join([os.pardir for i in range(n+1)]), d))
    return rdirs

def dirl_to_str(dirlist):
    """Given a list of directories, returns a string where the paths are
    concatenated by the path separator.

    example: ['foo/bar', 'bar/foo'] will return 'foo/bar:bar/foo'."""
    return os.pathsep.join(dirlist)

def dist2sconscc(compiler):
    """This converts the name passed to distutils to scons name convention (C
    compiler). compiler should be a CCompiler instance.

    Example:
        --compiler=intel -> intelc"""
    compiler_type = compiler.compiler_type
    if compiler_type == 'msvc':
        return 'msvc'
    elif compiler_type == 'intel':
        return 'intelc'
    else:
        return compiler.compiler[0]

def dist2sconsfc(compiler):
    """This converts the name passed to distutils to scons name convention
    (Fortran compiler). The argument should be a FCompiler instance.

    Example:
        --fcompiler=intel -> ifort on linux, ifl on windows"""
    if compiler.compiler_type == 'intel':
        #raise NotImplementedError('FIXME: intel fortran compiler name ?')
        return 'ifort'
    elif compiler.compiler_type == 'gnu':
        return 'g77'
    elif compiler.compiler_type == 'gnu95':
        return 'gfortran'
    elif compiler.compiler_type == 'sun':
        return 'sunf77'
    else:
        # XXX: Just give up for now, and use generic fortran compiler
        return 'fortran'

def dist2sconscxx(compiler):
    """This converts the name passed to distutils to scons name convention
    (C++ compiler). The argument should be a Compiler instance."""
    if compiler.compiler_type == 'msvc':
        return compiler.compiler_type

    return compiler.compiler_cxx[0]

def get_compiler_executable(compiler):
    """For any give CCompiler instance, this gives us the name of C compiler
    (the actual executable).

    NOTE: does NOT work with FCompiler instances."""
    # Geez, why does distutils has no common way to get the compiler name...
    if compiler.compiler_type == 'msvc':
        # this is harcoded in distutils... A bit cleaner way would be to
        # initialize the compiler instance and then get compiler.cc, but this
        # may be costly: we really just want a string.
        # XXX: we need to initialize the compiler anyway, so do not use
        # hardcoded string
        #compiler.initialize()
        #print compiler.cc
        return 'cl.exe'
    else:
        return compiler.compiler[0]

def get_f77_compiler_executable(compiler):
    """For any give FCompiler instance, this gives us the name of F77 compiler
    (the actual executable)."""
    return compiler.compiler_f77[0]

def get_cxxcompiler_executable(compiler):
    """For any give CCompiler instance, this gives us the name of CXX compiler
    (the actual executable).

    NOTE: does NOT work with FCompiler instances."""
    # Geez, why does distutils has no common way to get the compiler name...
    if compiler.compiler_type == 'msvc':
        # this is harcoded in distutils... A bit cleaner way would be to
        # initialize the compiler instance and then get compiler.cc, but this
        # may be costly: we really just want a string.
        # XXX: we need to initialize the compiler anyway, so do not use
        # hardcoded string
        #compiler.initialize()
        #print compiler.cc
        return 'cl.exe'
    else:
        return compiler.compiler_cxx[0]

def get_tool_path(compiler):
    """Given a distutils.ccompiler.CCompiler class, returns the path of the
    toolset related to C compilation."""
    fullpath_exec = find_executable(get_compiler_executable(compiler))
    if fullpath_exec:
        fullpath = pdirname(fullpath_exec)
    else:
        raise DistutilsSetupError("Could not find compiler executable info for scons")
    return fullpath

def get_f77_tool_path(compiler):
    """Given a distutils.ccompiler.FCompiler class, returns the path of the
    toolset related to F77 compilation."""
    fullpath_exec = find_executable(get_f77_compiler_executable(compiler))
    if fullpath_exec:
        fullpath = pdirname(fullpath_exec)
    else:
        raise DistutilsSetupError("Could not find F77 compiler executable "\
                "info for scons")
    return fullpath

def get_cxx_tool_path(compiler):
    """Given a distutils.ccompiler.CCompiler class, returns the path of the
    toolset related to C compilation."""
    fullpath_exec = find_executable(get_cxxcompiler_executable(compiler))
    if fullpath_exec:
        fullpath = pdirname(fullpath_exec)
    else:
        raise DistutilsSetupError("Could not find compiler executable info for scons")
    return fullpath

def protect_path(path):
    """Convert path (given as a string) to something the shell will have no
    problem to understand (space, etc... problems)."""
    if path:
        # XXX: to this correctly, this is totally bogus for now (does not check for
        # already quoted path, for example).
        return '"' + path + '"'
    else:
        return '""'

def parse_package_list(pkglist):
    return pkglist.split(",")

def find_common(seq1, seq2):
    """Given two list, return the index of the common items.

    The index are relative to seq1.

    Note: do not handle duplicate items."""
    dict2 = dict([(i, None) for i in seq2])

    return [i for i in range(len(seq1)) if dict2.has_key(seq1[i])]

def select_packages(sconspkg, pkglist):
    """Given a list of packages in pkglist, return the list of packages which
    match this list."""
    common = find_common(sconspkg, pkglist)
    if not len(common) == len(pkglist):
        msg = "the package list contains a package not found in "\
              "the current list. The current list is %s" % sconspkg
        raise ValueError(msg)
    return common

def check_numscons(minver):
    """Check that we can use numscons.

    minver is a 3 integers tuple which defines the min version."""
    try:
        import numscons
    except ImportError:
        e = get_exception()
        raise RuntimeError("importing numscons failed (error was %s), using " \
                           "scons within distutils is not possible without "
                           "this package " % str(e))

    try:
        # version_info was added in 0.10.0
        from numscons import version_info
        # Stupid me used string instead of numbers in version_info in
        # dev versions of 0.10.0
        if isinstance(version_info[0], str):
            raise ValueError("Numscons %s or above expected " \
                             "(detected 0.10.0)" % str(minver))
        # Stupid me used list instead of tuple in numscons
        version_info = tuple(version_info)
        if version_info[:3] < minver:
            raise ValueError("Numscons %s or above expected (got %s) "
                             % (str(minver), str(version_info[:3])))
    except ImportError:
        raise RuntimeError("You need numscons >= %s to build numpy "\
                           "with numscons (imported numscons path " \
                           "is %s)." % (minver, numscons.__file__))

# XXX: this is a giantic mess. Refactor this at some point.
class scons(old_build_ext):
    # XXX: add an option to the scons command for configuration (auto/force/cache).
    description = "Scons builder"

    library_options = [
        ('with-perflib=', None,
         'Specify which performance library to use for BLAS/LAPACK/etc...' \
         'Examples: mkl/atlas/sunper/accelerate'),
        ('with-mkl-lib=', None, 'TODO'),
        ('with-mkl-include=', None, 'TODO'),
        ('with-mkl-libraries=', None, 'TODO'),
        ('with-atlas-lib=', None, 'TODO'),
        ('with-atlas-include=', None, 'TODO'),
        ('with-atlas-libraries=', None, 'TODO')
        ]
    user_options = [
        ('jobs=', 'j', "specify number of worker threads when executing" \
                       "scons"),
        ('inplace', 'i', 'If specified, build in place.'),
        ('import-env', 'e', 'If specified, import user environment into scons env["ENV"].'),
        ('bypass', 'b', 'Bypass distutils compiler detection (experimental).'),
        ('scons-tool-path=', None, 'specify additional path '\
        '(absolute) to look for scons tools'),
        ('silent=', None, 'specify whether scons output should less verbose'\
        '(1), silent (2), super silent (3) or not (0, default)'),
        ('log-level=', None, 'specify log level for numscons. Any value ' \
                             'valid for the logging python module is valid'),
        ('package-list=', None,
         'If specified, only run scons on the given '\
         'packages (example: --package-list=scipy.cluster). If empty, '\
         'no package is built'),
        ('fcompiler=', None, "specify the Fortran compiler type"),
        ('compiler=', None, "specify the C compiler type"),
        ('cxxcompiler=', None,
         "specify the C++ compiler type (same as C by default)"),
        ('debug', 'g',
         "compile/link with debugging information"),
         ] + library_options

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        self.build_clib = None

        self.debug = 0

        self.compiler = None
        self.cxxcompiler = None
        self.fcompiler = None

        self.jobs = None
        self.silent = 0
        self.import_env = 0
        self.scons_tool_path = ''
        # If true, we bypass distutils to find the c compiler altogether. This
        # is to be used in desperate cases (like incompatible visual studio
        # version).
        self._bypass_distutils_cc = False

        # scons compilers
        self.scons_compiler = None
        self.scons_compiler_path = None
        self.scons_fcompiler = None
        self.scons_fcompiler_path = None
        self.scons_cxxcompiler = None
        self.scons_cxxcompiler_path = None

        self.package_list = None
        self.inplace = 0
        self.bypass = 0

        # Only critical things
        self.log_level = 50

        # library options
        self.with_perflib = []
        self.with_mkl_lib = []
        self.with_mkl_include = []
        self.with_mkl_libraries = []
        self.with_atlas_lib = []
        self.with_atlas_include = []
        self.with_atlas_libraries = []

    def _init_ccompiler(self, compiler_type):
        # XXX: The logic to bypass distutils is ... not so logic.
        if compiler_type == 'msvc':
            self._bypass_distutils_cc = True
        try:
            distutils_compiler = new_compiler(compiler=compiler_type,
                          verbose=self.verbose,
                          dry_run=self.dry_run,
                          force=self.force)
            distutils_compiler.customize(self.distribution)
            # This initialization seems necessary, sometimes, for find_executable to work...
            if hasattr(distutils_compiler, 'initialize'):
                distutils_compiler.initialize()
            self.scons_compiler = dist2sconscc(distutils_compiler)
            self.scons_compiler_path = protect_path(get_tool_path(distutils_compiler))
        except DistutilsPlatformError:
            e = get_exception()
            if not self._bypass_distutils_cc:
                raise e
            else:
                self.scons_compiler = compiler_type

    def _init_fcompiler(self, compiler_type):
        self.fcompiler = new_fcompiler(compiler = compiler_type,
                                       verbose = self.verbose,
                                       dry_run = self.dry_run,
                                       force = self.force)

        if self.fcompiler is not None:
            self.fcompiler.customize(self.distribution)
            self.scons_fcompiler = dist2sconsfc(self.fcompiler)
            self.scons_fcompiler_path = protect_path(get_f77_tool_path(self.fcompiler))

    def _init_cxxcompiler(self, compiler_type):
        cxxcompiler = new_compiler(compiler = compiler_type,
                                   verbose = self.verbose,
                                   dry_run = self.dry_run,
                                   force = self.force)
        if cxxcompiler is not None:
            cxxcompiler.customize(self.distribution, need_cxx = 1)
            cxxcompiler.customize_cmd(self)
            self.cxxcompiler = cxxcompiler.cxx_compiler()
            try:
                get_cxx_tool_path(self.cxxcompiler)
            except DistutilsSetupError:
                self.cxxcompiler = None

            if self.cxxcompiler:
                self.scons_cxxcompiler = dist2sconscxx(self.cxxcompiler)
                self.scons_cxxcompiler_path = protect_path(get_cxx_tool_path(self.cxxcompiler))

    def finalize_options(self):
        old_build_ext.finalize_options(self)

        self.sconscripts = []
        self.pre_hooks = []
        self.post_hooks = []
        self.pkg_names = []
        self.pkg_paths = []

        if self.distribution.has_scons_scripts():
            for i in self.distribution.scons_data:
                self.sconscripts.append(i.scons_path)
                self.pre_hooks.append(i.pre_hook)
                self.post_hooks.append(i.post_hook)
                self.pkg_names.append(i.parent_name)
                self.pkg_paths.append(i.pkg_path)
            # This crap is needed to get the build_clib
            # directory
            build_clib_cmd = get_cmd("build_clib").get_finalized_command("build_clib")
            self.build_clib = build_clib_cmd.build_clib

        if not self.cxxcompiler:
            self.cxxcompiler = self.compiler

        # To avoid trouble, just don't do anything if no sconscripts are used.
        # This is  useful when for example f2py uses numpy.distutils, because
        # f2py does not pass compiler information to scons command, and the
        # compilation setup below can crash in some situation.
        if len(self.sconscripts) > 0:
            if self.bypass:
                self.scons_compiler = self.compiler
                self.scons_fcompiler = self.fcompiler
                self.scons_cxxcompiler = self.cxxcompiler
            else:
                # Try to get the same compiler than the ones used by distutils: this is
                # non trivial because distutils and scons have totally different
                # conventions on this one (distutils uses PATH from user's environment,
                # whereas scons uses standard locations). The way we do it is once we
                # got the c compiler used, we use numpy.distutils function to get the
                # full path, and add the path to the env['PATH'] variable in env
                # instance (this is done in numpy.distutils.scons module).

                self._init_ccompiler(self.compiler)
                self._init_fcompiler(self.fcompiler)
                self._init_cxxcompiler(self.cxxcompiler)

        if self.package_list:
            self.package_list = parse_package_list(self.package_list)

    def _call_scons(self, scons_exec, sconscript, pkg_name, pkg_path, bootstrapping):
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)

        # XXX: passing everything at command line may cause some trouble where
        # there is a size limitation ? What is the standard solution in thise
        # case ?

        cmd = [scons_exec, "-f", sconscript, '-I.']
        if self.jobs:
            cmd.append(" --jobs=%d" % int(self.jobs))
        if self.inplace:
            cmd.append("inplace=1")
        cmd.append('scons_tool_path="%s"' % self.scons_tool_path)
        cmd.append('src_dir="%s"' % pdirname(sconscript))
        cmd.append('pkg_path="%s"' % pkg_path)
        cmd.append('pkg_name="%s"' % pkg_name)
        cmd.append('log_level=%s' % self.log_level)
        #cmd.append('distutils_libdir=%s' % protect_path(pjoin(self.build_lib,
        #                                                    pdirname(sconscript))))
        cmd.append('distutils_libdir=%s' %
                     protect_path(get_distutils_libdir(self, pkg_name)))
        cmd.append('distutils_clibdir=%s' %
                     protect_path(get_distutils_clibdir(self, pkg_name)))
        prefix = get_distutils_install_prefix(pkg_name, self.inplace)
        cmd.append('distutils_install_prefix=%s' % protect_path(prefix))

        if not self._bypass_distutils_cc:
            cmd.append('cc_opt=%s' % self.scons_compiler)
        if self.scons_compiler_path:
            cmd.append('cc_opt_path=%s' % self.scons_compiler_path)
        else:
            cmd.append('cc_opt=%s' % self.scons_compiler)

        cmd.append('debug=%s' % self.debug)

        if self.scons_fcompiler:
            cmd.append('f77_opt=%s' % self.scons_fcompiler)
        if self.scons_fcompiler_path:
            cmd.append('f77_opt_path=%s' % self.scons_fcompiler_path)

        if self.scons_cxxcompiler:
            cmd.append('cxx_opt=%s' % self.scons_cxxcompiler)
        if self.scons_cxxcompiler_path:
            cmd.append('cxx_opt_path=%s' % self.scons_cxxcompiler_path)

        cmd.append('include_bootstrap=%s' % dirl_to_str(get_numpy_include_dirs(sconscript)))
        cmd.append('bypass=%s' % self.bypass)
        cmd.append('import_env=%s' % self.import_env)
        if self.silent:
            if int(self.silent) == 2:
                cmd.append('-Q')
            elif int(self.silent) == 3:
                cmd.append('-s')
        cmd.append('silent=%d' % int(self.silent))
        cmd.append('bootstrapping=%d' % bootstrapping)
        cmdstr = ' '.join(cmd)
        if int(self.silent) < 1:
            log.info("Executing scons command (pkg is %s): %s ", pkg_name, cmdstr)
        else:
            log.info("======== Executing scons command for pkg %s =========", pkg_name)
        st = os.system(cmdstr)
        if st:
            #print "status is %d" % st
            msg = "Error while executing scons command."
            msg += " See above for more information.\n"
            msg += """\
If you think it is a problem in numscons, you can also try executing the scons
command with --log-level option for more detailed output of what numscons is
doing, for example --log-level=0; the lowest the level is, the more detailed
the output it."""
            raise DistutilsExecError(msg)

    def run(self):
        if len(self.sconscripts) < 1:
            # nothing to do, just leave it here.
            return

        check_numscons(minver=(0, 11, 0))

        if self.package_list is not None:
            id = select_packages(self.pkg_names, self.package_list)
            sconscripts = [self.sconscripts[i] for i in id]
            pre_hooks = [self.pre_hooks[i] for i in id]
            post_hooks = [self.post_hooks[i] for i in id]
            pkg_names = [self.pkg_names[i] for i in id]
            pkg_paths = [self.pkg_paths[i] for i in id]
        else:
            sconscripts = self.sconscripts
            pre_hooks = self.pre_hooks
            post_hooks = self.post_hooks
            pkg_names = self.pkg_names
            pkg_paths = self.pkg_paths

        if is_bootstrapping():
            bootstrapping = 1
        else:
            bootstrapping = 0

        scons_exec = get_python_exec_invoc()
        scons_exec += ' ' + protect_path(pjoin(get_scons_local_path(), 'scons.py'))

        for sconscript, pre_hook, post_hook, pkg_name, pkg_path in zip(sconscripts,
                                                   pre_hooks, post_hooks,
                                                   pkg_names, pkg_paths):
            if pre_hook:
                pre_hook()

            if sconscript:
                self._call_scons(scons_exec, sconscript, pkg_name, pkg_path, bootstrapping)

            if post_hook:
                post_hook(**{'pkg_name': pkg_name, 'scons_cmd' : self})

