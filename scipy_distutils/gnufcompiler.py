
import re
import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler
from exec_command import exec_command, find_executable

class GnuFCompiler(FCompiler):

    compiler_type = 'gnu'
    version_pattern = r'GNU Fortran ((\(GCC[^\)]*(\)\)|\)))|)\s*'\
                      '(?P<version>[^\s*\)]+)'

    # 'g77 --version' results
    # SunOS: GNU Fortran (GCC 3.2) 3.2 20020814 (release)
    # Debian: GNU Fortran (GCC) 3.3.3 20040110 (prerelease) (Debian)
    #         GNU Fortran (GCC) 3.3.3 (Debian 20040401)
    #         GNU Fortran 0.5.25 20010319 (prerelease)
    # Redhat: GNU Fortran (GCC 3.2.2 20030222 (Red Hat Linux 3.2.2-5)) 3.2.2 20030222 (Red Hat Linux 3.2.2-5)

    for fc_exe in map(find_executable,['g77','f77']):
        if os.path.isfile(fc_exe):
            break
    executables = {
        'version_cmd'  : [fc_exe,"--version"],
        'compiler_f77' : [fc_exe,"-Wall","-fno-second-underscore"],
        'compiler_f90' : None,
        'compiler_fix' : None,
        'linker_so'    : [fc_exe],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"],
        }
    module_dir_switch = None
    module_include_switch = None

    # Cygwin: f771: warning: -fPIC ignored for target (all code is position independent)
    if os.name != 'nt' and sys.platform!='cygwin':
        pic_flags = ['-fPIC']

    #def get_linker_so(self):
    #    # win32 linking should be handled by standard linker
    #    # Darwin g77 cannot be used as a linker.
    #    #if re.match(r'(darwin)', sys.platform):
    #    #    return
    #    return FCompiler.get_linker_so(self)

    def get_flags_linker_so(self):
        opt = []
        if sys.platform=='darwin':
            if os.path.realpath(sys.executable).startswith('/System'):
                # This is when Python is from Apple framework
                opt.extend(["-Wl,-framework","-Wl,Python"])
            #else we are running in Fink python.
            opt.extend(["-lcc_dynamic","-bundle"])
        else:
            opt.append("-shared")
        if sys.platform[:5]=='sunos':
            # SunOS often has dynamically loaded symbols defined in the
            # static library libg2c.a  The linker doesn't like this.  To
            # ignore the problem, use the -mimpure-text flag.  It isn't
            # the safest thing, but seems to work. 'man gcc' says:
            # ".. Instead of using -mimpure-text, you should compile all
            #  source code with -fpic or -fPIC."
            opt.append('-mimpure-text')
        return opt

    def get_libgcc_dir(self):
        status, output = exec_command('%s -print-libgcc-file-name' \
                                      % (self.compiler_f77[0]),use_tee=0)        
        if not status:
            return os.path.dirname(output)
        return

    def get_library_dirs(self):
        opt = []
        if sys.platform[:5] != 'linux':
            d = self.get_libgcc_dir()
            if d:
                opt.append(d)
        return opt

    def get_libraries(self):
        opt = []
        d = self.get_libgcc_dir()
        if d is not None:
            for g2c in ['g2c-pic','g2c']:
                f = self.static_lib_format % (g2c, self.static_lib_extension)
                if os.path.isfile(os.path.join(d,f)):
                    break
        else:
            g2c = 'g2c'
        if sys.platform=='win32':
            opt.extend(['gcc',g2c])
        else:
            opt.append(g2c)
        return opt

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        opt = ['-O3','-funroll-loops']
        return opt

    def get_flags_arch(self):
        opt = []
        if sys.platform=='darwin':
            if os.name != 'posix':
                # this should presumably correspond to Apple
                if cpu.is_ppc():
                    opt.append('-arch ppc')
                elif cpu.is_i386():
                    opt.append('-arch i386')
            for a in '601 602 603 603e 604 604e 620 630 740 7400 7450 750'\
                    '403 505 801 821 823 860'.split():
                if getattr(cpu,'is_ppc%s'%a)():
                    opt.append('-mcpu='+a)
                    opt.append('-mtune='+a)
                    break    
            return opt
        march_flag = 1
        # 0.5.25 corresponds to 2.95.x
        if self.get_version() == '0.5.26': # gcc 3.0
            if cpu.is_AthlonK6():
                opt.append('-march=k6')
            elif cpu.is_AthlonK7():
                opt.append('-march=athlon')
            else:
                march_flag = 0
        # Note: gcc 3.2 on win32 has breakage with -march specified
        elif self.get_version() >= '3.1.1' \
            and not sys.platform=='win32': # gcc >= 3.1.1
            if cpu.is_AthlonK6():
                opt.append('-march=k6')
            elif cpu.is_AthlonK6_2():
                opt.append('-march=k6-2')
            elif cpu.is_AthlonK6_3():
                opt.append('-march=k6-3')
            elif cpu.is_AthlonK7():
                opt.append('-march=athlon')
            elif cpu.is_AthlonMP():
                opt.append('-march=athlon-mp')
                # there's also: athlon-tbird, athlon-4, athlon-xp
            elif cpu.is_PentiumIV():
                opt.append('-march=pentium4')
            elif cpu.is_PentiumIII():
                opt.append('-march=pentium3')
            elif cpu.is_PentiumII():
                opt.append('-march=pentium2')
            else:
                march_flag = 0
            if self.get_version() >= '3.4' and not march_flag:
                march_flag = 1
                if cpu.is_Opteron():
                    opt.append('-march=opteron')
                elif cpu.is_Athlon64():
                    opt.append('-march=athlon64')
                else:
                    march_flag = 0
            if cpu.has_mmx(): opt.append('-mmmx')      
            if self.get_version() > '3.2.2':
                if cpu.has_sse2(): opt.append('-msse2')
                if cpu.has_sse(): opt.append('-msse')
            if self.get_version() >= '3.4':
                if cpu.has_sse3(): opt.append('-msse3')
            if cpu.has_3dnow(): opt.append('-m3dnow')
        else:
            march_flag = 0
        if march_flag:
            pass
        elif cpu.is_i686():
            opt.append('-march=i686')
        elif cpu.is_i586():
            opt.append('-march=i586')
        elif cpu.is_i486():
            opt.append('-march=i486')
        elif cpu.is_i386():
            opt.append('-march=i386')
        if cpu.is_Intel():
            opt.extend(['-malign-double','-fomit-frame-pointer'])
        return opt

if __name__ == '__main__':
    from scipy_distutils import log
    log.set_verbosity(2)
    from fcompiler import new_fcompiler
    #compiler = new_fcompiler(compiler='gnu')
    compiler = GnuFCompiler()
    compiler.customize()
    print compiler.get_version()
