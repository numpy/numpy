import os
import os.path
from os.path import join as pjoin, dirname as pdirname

from distutils.errors import DistutilsPlatformError
from distutils.errors import DistutilsExecError, DistutilsSetupError

from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.ccompiler import CCompiler
from numpy.distutils.fcompiler import FCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils import log

def get_scons_build_dir():
    """Return the top path where everything produced by scons will be put.

    The path is relative to the top setup.py"""
    from numscons import get_scons_build_dir
    return get_scons_build_dir()

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

def get_distutils_libdir(cmd, sconscript_path):
    """Returns the path where distutils install libraries, relatively to the
    scons build directory."""
    from numscons import get_scons_build_dir
    scdir = pjoin(get_scons_build_dir(), pdirname(sconscript_path))
    n = scdir.count(os.sep)
    return pjoin(os.sep.join([os.pardir for i in range(n+1)]), cmd.build_lib)

def get_python_exec_invoc():
    """This returns the python executable from which this file is invocated."""
    # Do we  need to take into account the PYTHONPATH, in a cross platform way,
    # that is the string returned can be executed directly on supported
    # platforms, and the sys.path of the executed python should be the same
    # than the caller ? This may not be necessary, since os.system is said to
    # take into accound os.environ. This actually also works for my way of
    # using "local python", using the alias facility of bash.
    import sys
    return sys.executable

def get_numpy_include_dirs(sconscript_path):
    """Return include dirs for numpy.

    The paths are relatively to the setup.py script path."""
    from numpy.distutils.misc_util import get_numpy_include_dirs as _incdir
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
    elif compiler_type == 'mingw32':
        return 'mingw'
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
    # XXX: to this correctly, this is totally bogus for now (does not check for
    # already quoted path, for example).
    return '"' + path + '"'

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

def is_bootstrapping():
    import __builtin__
    try:
        __builtin__.__NUMPY_SETUP__
        return True
    except AttributeError:
        return False
        __NUMPY_SETUP__ = False

class scons(old_build_ext):
    # XXX: add an option to the scons command for configuration (auto/force/cache).
    description = "Scons builder"
    user_options = old_build_ext.user_options + \
            [('jobs=', None,
              "specify number of worker threads when executing scons"),
             ('scons-tool-path=', None, 'specify additional path '\
                                    '(absolute) to look for scons tools'),
             ('silent=', None, 'specify whether scons output should less verbose'\
                               '(1), silent (2), super silent (3) or not (0, default)'),
             ('package-list=', None, 'If specified, only run scons on the given '\
                 'packages (example: --package-list=scipy.cluster). If empty, '\
                 'no package is built')]

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        self.jobs = None
        self.silent = 0
        self.scons_tool_path = ''
        # If true, we bypass distutils to find the c compiler altogether. This
        # is to be used in desperate cases (like incompatible visual studio
        # version).
        self._bypass_distutils_cc = False
        self.scons_compiler = None
        self.scons_compiler_path = None
        self.scons_fcompiler = None

        self.package_list = None

    def finalize_options(self):
        old_build_ext.finalize_options(self)
        if self.distribution.has_scons_scripts():
            self.sconscripts = self.distribution.get_scons_scripts()
            self.pre_hooks = self.distribution.get_scons_pre_hooks()
            self.post_hooks = self.distribution.get_scons_post_hooks()
            self.pkg_names = self.distribution.get_scons_parent_names()
        else:
            self.sconscripts = []
            self.pre_hooks = []
            self.post_hooks = []
            self.pkg_names = []

        # Try to get the same compiler than the ones used by distutils: this is
        # non trivial because distutils and scons have totally different
        # conventions on this one (distutils uses PATH from user's environment,
        # whereas scons uses standard locations). The way we do it is once we
        # got the c compiler used, we use numpy.distutils function to get the
        # full path, and add the path to the env['PATH'] variable in env
        # instance (this is done in numpy.distutils.scons module).

        # XXX: The logic to bypass distutils is ... not so logic.
        compiler_type = self.compiler
        if compiler_type == 'msvc':
            self._bypass_distutils_cc = True
        from numpy.distutils.ccompiler import new_compiler
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
        except DistutilsPlatformError, e:
            if not self._bypass_distutils_cc:
                raise e
            else:
                self.scons_compiler = compiler_type

        # We do the same for the fortran compiler ...
        fcompiler_type = self.fcompiler
        from numpy.distutils.fcompiler import new_fcompiler
        self.fcompiler = new_fcompiler(compiler = fcompiler_type,
                                       verbose = self.verbose,
                                       dry_run = self.dry_run,
                                       force = self.force)
        if self.fcompiler is not None:
            self.fcompiler.customize(self.distribution)

        # And the C++ compiler
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

        if self.package_list:
            self.package_list = parse_package_list(self.package_list)

    def run(self):
        if len(self.sconscripts) > 0:
            try:
                import numscons
            except ImportError, e:
                raise RuntimeError("importing numscons failed (error was %s), using " \
                                   "scons within distutils is not possible without "
                                   "this package " % str(e))

            try:
                minver = "0.8.2"
                from numscons import get_version
                if get_version() < minver:
                    raise ValueError()
            except ImportError:
                raise RuntimeError("You need numscons >= %s to build numpy "\
                                   "with numscons (imported numscons path " \
                                   "is %s)." % (minver, numscons.__file__))
            except ValueError:
                raise RuntimeError("You need numscons >= %s to build numpy "\
                                   "with numscons (detected %s )" \
                                   % (minver, get_version()))
                
        else:
            # nothing to do, just leave it here.
            return

        print "is bootstrapping ? %s" % is_bootstrapping()
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)

        # XXX: passing everything at command line may cause some trouble where
        # there is a size limitation ? What is the standard solution in thise
        # case ?

        scons_exec = get_python_exec_invoc()
        scons_exec += ' ' + protect_path(pjoin(get_scons_local_path(), 'scons.py'))

        if self.package_list is not None:
            id = select_packages(self.pkg_names, self.package_list)
            sconscripts = [self.sconscripts[i] for i in id]
            pre_hooks = [self.pre_hooks[i] for i in id]
            post_hooks = [self.post_hooks[i] for i in id]
            pkg_names = [self.pkg_names[i] for i in id]
        else:
            sconscripts = self.sconscripts
            pre_hooks = self.pre_hooks
            post_hooks = self.post_hooks
            pkg_names = self.pkg_names

        if is_bootstrapping():
            bootstrap = 1
        else:
            bootstrap = 0

        for sconscript, pre_hook, post_hook, pkg_name in zip(sconscripts,
                                                   pre_hooks, post_hooks,
                                                   pkg_names):
            if pre_hook:
                pre_hook()

            cmd = [scons_exec, "-f", sconscript, '-I.']
            if self.jobs:
                cmd.append(" --jobs=%d" % int(self.jobs))
            cmd.append('scons_tool_path="%s"' % self.scons_tool_path)
            cmd.append('src_dir="%s"' % pdirname(sconscript))
            cmd.append('pkg_name="%s"' % pkg_name)
            #cmd.append('distutils_libdir=%s' % protect_path(pjoin(self.build_lib,
            #                                                    pdirname(sconscript))))
            cmd.append('distutils_libdir=%s' % 
                         protect_path(get_distutils_libdir(self, sconscript)))

            if not self._bypass_distutils_cc:
                cmd.append('cc_opt=%s' % self.scons_compiler)
                cmd.append('cc_opt_path=%s' % self.scons_compiler_path)
            else:
                cmd.append('cc_opt=%s' % self.scons_compiler)


            if self.fcompiler:
                cmd.append('f77_opt=%s' % dist2sconsfc(self.fcompiler))
                cmd.append('f77_opt_path=%s' % protect_path(get_f77_tool_path(self.fcompiler)))

            if self.cxxcompiler:
                cmd.append('cxx_opt=%s' % dist2sconscxx(self.cxxcompiler))
                cmd.append('cxx_opt_path=%s' % protect_path(get_cxx_tool_path(self.cxxcompiler)))

            cmd.append('include_bootstrap=%s' % dirl_to_str(get_numpy_include_dirs(sconscript)))
            if self.silent:
                if int(self.silent) == 2:
                    cmd.append('-Q')
                elif int(self.silent) == 3:
                    cmd.append('-s')
            cmd.append('silent=%d' % int(self.silent))
            cmd.append('bootstrapping=%d' % bootstrap)
            cmdstr = ' '.join(cmd)
            if int(self.silent) < 1:
                log.info("Executing scons command (pkg is %s): %s ", pkg_name, cmdstr)
            else:
                log.info("======== Executing scons command for pkg %s =========", pkg_name)
            st = os.system(cmdstr)
            if st:
                print "status is %d" % st
                raise DistutilsExecError("Error while executing scons command "\
                                         "%s (see above)" % cmdstr)
            if post_hook:
                post_hook()
