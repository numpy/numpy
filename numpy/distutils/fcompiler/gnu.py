import re
import os
import sys
import warnings

from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler
from numpy.distutils.exec_command import exec_command
from numpy.distutils.misc_util import msvc_runtime_library

compilers = ['GnuFCompiler', 'Gnu95FCompiler']

TARGET_R = re.compile("Target: ([a-zA-Z0-9_\-]*)")
class GnuFCompiler(FCompiler):
    compiler_type = 'gnu'
    compiler_aliases = ('g77',)
    description = 'GNU Fortran 77 compiler'

    def gnu_version_match(self, version_string):
        """Handle the different versions of GNU fortran compilers"""
        m = re.match(r'GNU Fortran', version_string)
        if not m:
            return None
        m = re.match(r'GNU Fortran\s+95.*?([0-9-.]+)', version_string)
        if m:
            return ('gfortran', m.group(1))
        m = re.match(r'GNU Fortran.*?([0-9-.]+)', version_string)
        if m:
            v = m.group(1)
            if v.startswith('0') or v.startswith('2') or v.startswith('3'):
                # the '0' is for early g77's
                return ('g77', v)
            else:
                # at some point in the 4.x series, the ' 95' was dropped
                # from the version string
                return ('gfortran', v)

    def version_match(self, version_string):
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'g77':
            return None
        return v[1]

    # 'g77 --version' results
    # SunOS: GNU Fortran (GCC 3.2) 3.2 20020814 (release)
    # Debian: GNU Fortran (GCC) 3.3.3 20040110 (prerelease) (Debian)
    #         GNU Fortran (GCC) 3.3.3 (Debian 20040401)
    #         GNU Fortran 0.5.25 20010319 (prerelease)
    # Redhat: GNU Fortran (GCC 3.2.2 20030222 (Red Hat Linux 3.2.2-5)) 3.2.2 20030222 (Red Hat Linux 3.2.2-5)
    # GNU Fortran (GCC) 3.4.2 (mingw-special)

    possible_executables = ['g77', 'f77']
    executables = {
        'version_cmd'  : [None, "--version"],
        'compiler_f77' : [None, "-g", "-Wall", "-fno-second-underscore"],
        'compiler_f90' : None, # Use --fcompiler=gnu95 for f90 codes
        'compiler_fix' : None,
        'linker_so'    : [None, "-g", "-Wall"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"],
        'linker_exe'   : [None, "-g", "-Wall"]
        }
    module_dir_switch = None
    module_include_switch = None

    # Cygwin: f771: warning: -fPIC ignored for target (all code is
    # position independent)
    if os.name != 'nt' and sys.platform != 'cygwin':
        pic_flags = ['-fPIC']

    # use -mno-cygwin for g77 when Python is not Cygwin-Python
    if sys.platform == 'win32':
        for key in ['version_cmd', 'compiler_f77', 'linker_so', 'linker_exe']:
            executables[key].append('-mno-cygwin')

    g2c = 'g2c'

    suggested_f90_compiler = 'gnu95'

    #def get_linker_so(self):
    #    # win32 linking should be handled by standard linker
    #    # Darwin g77 cannot be used as a linker.
    #    #if re.match(r'(darwin)', sys.platform):
    #    #    return
    #    return FCompiler.get_linker_so(self)

    def get_flags_linker_so(self):
        opt = self.linker_so[1:]
        if sys.platform=='darwin':
            # MACOSX_DEPLOYMENT_TARGET must be at least 10.3. This is
            # a reasonable default value even when building on 10.4 when using
            # the official Python distribution and those derived from it (when
            # not broken).
            target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
            if target is None or target == '':
                target = '10.3'
            major, minor = target.split('.')
            if int(minor) < 3:
                minor = '3'
                warnings.warn('Environment variable '
                    'MACOSX_DEPLOYMENT_TARGET reset to %s.%s' % (major, minor))
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '%s.%s' % (major,
                minor)

            opt.extend(['-undefined', 'dynamic_lookup', '-bundle'])
        else:
            opt.append("-shared")
        if sys.platform.startswith('sunos'):
            # SunOS often has dynamically loaded symbols defined in the
            # static library libg2c.a  The linker doesn't like this.  To
            # ignore the problem, use the -mimpure-text flag.  It isn't
            # the safest thing, but seems to work. 'man gcc' says:
            # ".. Instead of using -mimpure-text, you should compile all
            #  source code with -fpic or -fPIC."
            opt.append('-mimpure-text')
        return opt

    def get_libgcc_dir(self):
        status, output = exec_command(self.compiler_f77 +
                                      ['-print-libgcc-file-name'],
                                      use_tee=0)
        if not status:
            return os.path.dirname(output)
        return None

    def get_library_dirs(self):
        opt = []
        if sys.platform[:5] != 'linux':
            d = self.get_libgcc_dir()
            if d:
                # if windows and not cygwin, libg2c lies in a different folder
                if sys.platform == 'win32' and not d.startswith('/usr/lib'):
                    d = os.path.normpath(d)
                    if not os.path.exists(os.path.join(d, "lib%s.a" % self.g2c)):
                        d2 = os.path.abspath(os.path.join(d,
                                                          '../../../../lib'))
                        if os.path.exists(os.path.join(d2, "lib%s.a" % self.g2c)):
                            opt.append(d2)
                opt.append(d)
        return opt

    def get_libraries(self):
        opt = []
        d = self.get_libgcc_dir()
        if d is not None:
            g2c = self.g2c + '-pic'
            f = self.static_lib_format % (g2c, self.static_lib_extension)
            if not os.path.isfile(os.path.join(d,f)):
                g2c = self.g2c
        else:
            g2c = self.g2c

        if g2c is not None:
            opt.append(g2c)
        c_compiler = self.c_compiler
        if sys.platform == 'win32' and c_compiler and \
               c_compiler.compiler_type=='msvc':
            # the following code is not needed (read: breaks) when using MinGW
            # in case want to link F77 compiled code with MSVC
            opt.append('gcc')
            runtime_lib = msvc_runtime_library()
            if runtime_lib:
                opt.append(runtime_lib)
        if sys.platform == 'darwin':
            opt.append('cc_dynamic')
        return opt

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        if self.get_version()<='3.3.3':
            # With this compiler version building Fortran BLAS/LAPACK
            # with -O3 caused failures in lib.lapack heevr,syevr tests.
            opt = ['-O2']
        else:
            opt = ['-O3']
        opt.append('-funroll-loops')
        return opt

    def get_flags_arch(self):
        return []

class Gnu95FCompiler(GnuFCompiler):
    compiler_type = 'gnu95'
    compiler_aliases = ('gfortran',)
    description = 'GNU Fortran 95 compiler'

    def version_match(self, version_string):
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'gfortran':
            return None
        return v[1]

    # 'gfortran --version' results:
    # XXX is the below right?
    # Debian: GNU Fortran 95 (GCC 4.0.3 20051023 (prerelease) (Debian 4.0.2-3))
    #         GNU Fortran 95 (GCC) 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)
    # OS X: GNU Fortran 95 (GCC) 4.1.0
    #       GNU Fortran 95 (GCC) 4.2.0 20060218 (experimental)
    #       GNU Fortran (GCC) 4.3.0 20070316 (experimental)

    possible_executables = ['gfortran', 'f95']
    executables = {
        'version_cmd'  : ["<F90>", "--version"],
        'compiler_f77' : [None, "-Wall", "-ffixed-form",
                          "-fno-second-underscore"],
        'compiler_f90' : [None, "-Wall", "-fno-second-underscore"],
        'compiler_fix' : [None, "-Wall", "-ffixed-form",
                          "-fno-second-underscore"],
        'linker_so'    : ["<F90>", "-Wall"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"],
        'linker_exe'   : [None, "-Wall"]
        }

    # use -mno-cygwin flag for g77 when Python is not Cygwin-Python
    if sys.platform == 'win32':
        for key in ['version_cmd', 'compiler_f77', 'compiler_f90',
                    'compiler_fix', 'linker_so', 'linker_exe']:
            executables[key].append('-mno-cygwin')

    module_dir_switch = '-J'
    module_include_switch = '-I'

    g2c = 'gfortran'

    # Note that this is here instead of GnuFCompiler as gcc < 4 uses a
    # different output format (which isn't as useful) than gcc >= 4,
    # and we don't have to worry about g77 being universal (as it can't be).
    def target_architecture(self, extra_opts=()):
        """Return the architecture that the compiler will build for.
        This is most useful for detecting universal compilers in OS X."""
        extra_opts = list(extra_opts)
        status, output = exec_command(self.compiler_f90 + ['-v'] + extra_opts,
                                      use_tee=False)
        if status == 0:
            m = re.match(r'(?m)^Target: (.*)$', output)
            if m:
                return m.group(1)
        return None

    def is_universal_compiler(self):
        """Return True if this compiler can compile universal binaries
        (for OS X).

        Currently only checks for i686 and powerpc architectures (no 64-bit
        support yet).
        """
        if sys.platform != 'darwin':
            return False
        i686_arch = self.target_architecture(extra_opts=['-arch', 'i686'])
        if not i686_arch or not i686_arch.startswith('i686-'):
            return False
        ppc_arch = self.target_architecture(extra_opts=['-arch', 'ppc'])
        if not ppc_arch or not ppc_arch.startswith('powerpc-'):
            return False
        return True

    def _add_arches_for_universal_build(self, flags):
        if self.is_universal_compiler():
            flags[:0] = ['-arch', 'i686', '-arch', 'ppc']
        return flags

    def get_flags(self):
        flags = GnuFCompiler.get_flags(self)
        return self._add_arches_for_universal_build(flags)

    def get_flags_linker_so(self):
        flags = GnuFCompiler.get_flags_linker_so(self)
        return self._add_arches_for_universal_build(flags)

    def get_libraries(self):
        opt = GnuFCompiler.get_libraries(self)
        if sys.platform == 'darwin':
            opt.remove('cc_dynamic')
        return opt

    def get_target(self):
        status, output = exec_command(self.compiler_f77 +
                                      ['-v'],
                                      use_tee=0)
        if not status:
	    m = TARGET_R.search(output)
	    if m:
	        return m.group(1)	
        return ""

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    compiler = GnuFCompiler()
    compiler.customize()
    print compiler.get_version()
    raw_input('Press ENTER to continue...')
    try:
        compiler = Gnu95FCompiler()
        compiler.customize()
        print compiler.get_version()
    except Exception, msg:
        print msg
    raw_input('Press ENTER to continue...')
