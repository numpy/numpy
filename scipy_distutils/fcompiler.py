"""scipy_distutils.fcompiler

Contains FCompiler, an abstract base class that defines the interface
for the Scipy_istutils Fortran compiler abstraction model.

"""

import re
import os
import sys

from distutils.version import StrictVersion
from distutils.ccompiler import CCompiler
# distutils.ccompiler provides the following functions:
#   gen_preprocess_options(macros, include_dirs)
#   gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries)
from distutils.errors import DistutilsModuleError,DistutilsArgError
from distutils.core import Command
from distutils.util import split_quoted
from distutils.fancy_getopt import FancyGetopt
from distutils.version import LooseVersion
from distutils import log
from distutils.sysconfig import get_config_var

from exec_command import find_executable, exec_command


class FCompiler(CCompiler):
    """ Abstract base class to define the interface that must be implemented
    by real Fortran compiler classes.

    Methods that subclasses may redefine:

        get_version_cmd(), get_linker_so(), get_version()
        get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()
        get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),
        get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),
        get_flags_arch_f90(), get_flags_debug_f90(),
        get_flags_fix(), get_flags_linker_so(), get_flags_version()

    DON'T call these methods (except get_version) after
    constructing a compiler instance or inside any other method.
    All methods, except get_version_cmd() and get_flags_version(), may
    call get_version() method.

    After constructing a compiler instance, always call customize(dist=None)
    method that finalizes compiler construction and makes the following
    attributes available:
      compiler_f77
      compiler_f90
      compiler_fix
      linker_so
      archiver
      ranlib
      libraries
      library_dirs
    """
    # CCompiler defines the following attributes:
    #   compiler_type
    #   src_extensions
    #   obj_extension
    #   static_lib_extension
    #   shared_lib_extension
    #   static_lib_format
    #   shared_lib_format
    #   exe_extension
    #   language_map    ### REDEFINED
    #   language_order  ### REDEFINED
    # and the following public methods:
    #   set_executables(**args)
    #     set_executable(key,value)
    #   define_macro(name, value=None)
    #   undefine_macro(name)
    #   add_include_dir(dir)
    #   set_include_dirs(dirs)
    #   add_library(libname)
    #   set_libraries(libnames)
    #   add_library_dir(dir)
    #   set_library_dirs(dirs)
    #   add_runtime_library_dir(dir)
    #   set_runtime_library_dirs(dirs)
    #   add_link_object(object)
    #   set_link_objects(objects)
    #
    #   detect_language(sources)  ### USABLE
    #
    #   preprocess(source,output_file=None,macros=None,include_dirs=None,
    #              extra_preargs=None,extra_postargs=None)
    #   compile(sources, output_dir=None, macros=None,
    #           include_dirs=None, debug=0, extra_preargs=None,
    #           extra_postargs=None, depends=None)
    #   create_static_lib(objects,output_libname,output_dir=None,debug=0,target_lang=None):
    #   link(target_desc, objects, output_filename, output_dir=None,
    #        libraries=None, library_dirs=None, runtime_library_dirs=None,
    #        export_symbols=None, debug=0, extra_preargs=None, extra_postargs=None,
    #        build_temp=None, target_lang=None)
    #   link_shared_lib(objects, output_libname, output_dir=None,
    #                   libraries=None, library_dirs=None, runtime_library_dirs=None,
    #                   export_symbols=None, debug=0, extra_preargs=None,
    #                   extra_postargs=None, build_temp=None, target_lang=None)
    #   link_shared_object(objects,output_filename,output_dir=None,
    #                      libraries=None,library_dirs=None,runtime_library_dirs=None,
    #                      export_symbols=None,debug=0,extra_preargs=None,
    #                      extra_postargs=None,build_temp=None,target_lang=None)
    #   link_executable(objects,output_progname,output_dir=None,
    #                   libraries=None,library_dirs=None,runtime_library_dirs=None,
    #                   debug=0,extra_preargs=None,extra_postargs=None,target_lang=None)
    #
    #   library_dir_option(dir)
    #   runtime_library_dir_option(dir)
    #   library_option(lib)
    #   has_function(funcname,includes=None,include_dirs=None,
    #                libraries=None,library_dirs=None)
    #   find_library_file(dirs, lib, debug=0)
    #
    #   object_filenames(source_filenames, strip_dir=0, output_dir='')
    #   shared_object_filename(basename, strip_dir=0, output_dir='')
    #   executable_filenamee(basename, strip_dir=0, output_dir='')
    #   library_filename(libname, lib_type='static',strip_dir=0, output_dir=''):
    #
    #   announce(msg, level=1)
    #   debug_print(msg)
    #   warn(msg)
    #   execute(func, args, msg=None, level=1)
    #   spawn(cmd)
    #   move_file(src,dst)
    #   mkpath(name, mode=0777)
    #

    src_extensions = ['.for','.ftn','.f77','.f','.f90','.f95']

    language_map = {'.f':'f77',
                    '.for':'f77',
                    '.ftn':'f77',
                    '.f77':'f77',
                    '.f90':'f90',
                    '.f95':'f90'}
    language_order = ['f90','f77']

    version_pattern = None

    executables = {
        'version_cmd'  : ["f77","-v"],
        'compiler_f77' : ["f77"],
        'compiler_f90' : ["f90"],
        'compiler_fix' : ["f90","-fixed"],
        'linker_so'    : ["f90","-shared"],
        'archiver'     : ["ar","-cr"],
        'ranlib'       : None,
        }

    compile_switch = "-c " #  Ending space matters!
    object_switch = "-o " #  Ending space matters!
    object_extension = ".o"
    shared_lib_extension = get_config_var('SO')
    static_lib_extension = ".a"
    static_lib_format = "lib%s%s"
    shared_lib_format = "%s%s"

    def __init__(self,verbose=0,dry_run=0,force=0):
        # Set the following attributes (see ccompiler.py for explanations):
        #   output_dir
        #   macros
        #   include_dirs
        #   libraries
        #   library_dirs
        #   runtime_library_dirs
        #   objects
        # and call set_executables.
        CCompiler.__init__(self,verbose,dry_run,force)

    ## Methods that subclasses may redefine. But don't call these methods!
    ## They are private to FCompiler class and may return unexpected
    ## results if used elsewhere. So, you have been warned..

    def get_version_cmd(self):
        """ Compiler command to print out version information. """
        f77 = self.executables['compiler_f77']
        if f77 is not None:
            f77 = f77[0]
        cmd = self.executables['version_cmd']
        if cmd is not None:
            cmd = cmd[0]
            if cmd==f77:
                cmd = self.compiler_f77[0]
            else:
                f90 = self.executables['compiler_f90']
                if f90 is not None:
                    f90 = f90[0]
                if cmd==f90:
                    cmd = self.compiler_f90[0]
        return cmd

    def get_linker_so(self):
        """ Linker command to build shared libraries. """
        f77 = self.executables['compiler_f77']
        if f77 is not None:
            f77 = f77[0]
        ln = self.executables['linker_so']
        if ln is not None:
            ln = ln[0]
            if ln==f77:
                ln = self.compiler_f77[0]
            else:
                f90 = self.executables['compiler_f90']
                if f90 is not None:
                    f90 = f90[0]
                if ln==f90:
                    ln = self.compiler_f90[0]
        return ln

    def get_flags(self):
        """ List of flags common to all compiler types. """
        return []
    def get_flags_version(self):
        """ List of compiler flags to print out version information. """
        if self.executables['version_cmd']:
            return self.executables['version_cmd'][1:]
        return []
    def get_flags_f77(self):
        """ List of Fortran 77 specific flags. """
        if self.executables['compiler_f77']:
            return self.executables['compiler_f77'][1:]
        return []
    def get_flags_f90(self):
        """ List of Fortran 90 specific flags. """
        if self.executables['compiler_f90']:
            return self.executables['compiler_f90'][1:]
        return []
    def get_flags_fix(self):
        """ List of Fortran 90 fixed format specific flags. """
        if self.executables['compiler_fix']:
            return self.executables['compiler_fix'][1:]
        return []
    def get_flags_linker_so(self):
        """ List of linker flags to build a shared library. """
        if self.executables['linker_so']:
            return self.executables['linker_so'][1:]
        return []
    def get_flags_ar(self):
        """ List of archiver flags. """
        if self.executables['archiver']:
            return self.executables['archiver'][1:]
        return []
    def get_flags_opt(self):
        """ List of architecture independent compiler flags. """
        return []
    def get_flags_arch(self):
        """ List of architecture dependent compiler flags. """
        return []
    def get_flags_debug(self):
        """ List of compiler flags to compile with debugging information. """
        return []
    get_flags_opt_f77 = get_flags_opt_f90 = get_flags_opt
    get_flags_arch_f77 = get_flags_arch_f90 = get_flags_arch
    get_flags_debug_f77 = get_flags_debug_f90 = get_flags_debug

    def get_libraries(self):
        """ List of compiler libraries. """
        return self.libraries[:]
    def get_library_dirs(self):
        """ List of compiler library directories. """
        return self.library_dirs[:]

    def get_version(self, force=0, ok_status=[0]):
        """ Compiler version. Returns None if compiler is not available. """
        if not force and hasattr(self,'version'):
            return self.version

        cmd = ' '.join(self.version_cmd)
        status, output = self.exec_command(cmd,use_tee=0)
        version = None
        if status in ok_status:
            m = re.match(self.version_pattern,output)
            if m:
                version = m.group('version')
                assert version,`version`
                version = LooseVersion(version)
        self.version = version
        return version

    ############################################################

    ## Public methods:

    def customize(self, dist=None):
        """ Customize Fortran compiler.

        This method gets Fortran compiler specific information from
        (i) class definition, (ii) environment, (iii) distutils config
        files, and (iv) command line.

        This method should be always called after constructing a
        compiler instance. But not in __init__ because Distribution
        instance is needed for (iii) and (iv).
        """
        if dist is None:
            # These hooks are for testing only!
            from dist import Distribution
            dist = Distribution()
            dist.script_name = os.path.basename(sys.argv[0])
            dist.script_args = ['config_fc'] + sys.argv[1:]
            dist.cmdclass['config_fc'] = config_fc
            dist.parse_config_files()
            dist.parse_command_line()

        conf = dist.get_option_dict('config_fc')

        noopt = conf.get('noopt',[None,0])[1]
        noarch = conf.get('noarch',[None,noopt])[1]
        debug = conf.get('debug',[None,0])[1]

        f77 = self.__get_cmd('compiler_f77','F77',(conf,'f77_exec'))
        f90 = self.__get_cmd('compiler_f90','F90',(conf,'f90_exec'))
        # Temporarily setting f77,f90 compilers so that
        # version_cmd can use their executables.
        if f77:
            self.set_executables(compiler_f77=[f77])
        if f90:
            self.set_executables(compiler_f90=[f90])

        # Must set version_cmd before others as self.get_flags*
        # methods may call self.get_version.
        vers_cmd = self.__get_cmd(self.get_version_cmd)
        if vers_cmd:
            vflags = self.__get_flags(self.get_flags_version)
            self.set_executables(version_cmd=[vers_cmd]+vflags)

        if f77:
            f77flags = self.__get_flags(self.get_flags_f77,'F77FLAGS',
                                   (conf,'f77flags'))
        if f90:
            f90flags = self.__get_flags(self.get_flags_f90,'F90FLAGS',
                                       (conf,'f90flags'))

        # XXX Assuming that free format is default for f90 compiler.
        fix = self.__get_cmd('compiler_fix','F90',(conf,'f90_exec'))
        if fix:
            fixflags = self.__get_flags(self.get_flags_fix) + f90flags

        oflags,aflags,dflags = [],[],[]
        if not noopt:
            oflags = self.__get_flags(self.get_flags_opt,'FOPT',(conf,'opt'))
            if f77 and self.get_flags_opt is not self.get_flags_opt_f77:
                f77flags += self.__get_flags(self.get_flags_opt_f77)
            if f90 and self.get_flags_opt is not self.get_flags_opt_f90:
                f90flags += self.__get_flags(self.get_flags_opt_f90)
            if fix and self.get_flags_opt is not self.get_flags_opt_f90:
                fixflags += self.__get_flags(self.get_flags_opt_f90)
            if not noarch:
                aflags = self.__get_flags(self.get_flags_arch,'FARCH',
                                          (conf,'arch'))
                if f77 and self.get_flags_arch is not self.get_flags_arch_f77:
                    f77flags += self.__get_flags(self.get_flags_arch_f77)
                if f90 and self.get_flags_arch is not self.get_flags_arch_f90:
                    f90flags += self.__get_flags(self.get_flags_arch_f90)
                if fix and self.get_flags_arch is not self.get_flags_arch_f90:
                    fixflags += self.__get_flags(self.get_flags_arch_f90)
        if debug:
            dflags = self.__get_flags(self.get_flags_debug,'FDEBUG')
            if f77  and self.get_flags_debug is not self.get_flags_debug_f77:
                f77flags += self.__get_flags(self.get_flags_debug_f77)
            if f90  and self.get_flags_debug is not self.get_flags_debug_f90:
                f90flags += self.__get_flags(self.get_flags_debug_f90)
            if fix and self.get_flags_debug is not self.get_flags_debug_f90:
                fixflags += self.__get_flags(self.get_flags_debug_f90)

        fflags = self.__get_flags(self.get_flags,'FFLAGS') \
                 + dflags + oflags + aflags

        if f77:
            self.set_executables(compiler_f77=[f77]+f77flags+fflags)
        if f90:
            self.set_executables(compiler_f90=[f90]+f90flags+fflags)
        if fix:
            self.set_executables(compiler_fix=[fix]+fixflags+fflags)

        linker_so = self.__get_cmd(self.get_linker_so,'LDSHARED')
        if linker_so:
            linker_so_flags = self.__get_flags(self.get_flags_linker_so,'LDFLAGS')
            self.set_executables(linker_so=[linker_so]+linker_so_flags)

        ar = self.__get_cmd('archiver','AR')
        if ar:
            arflags = self.__get_flags(self.get_flags_ar,'ARFLAGS')
            self.set_executables(archiver=[ar]+arflags)

        ranlib = self.__get_cmd('ranlib','RANLIB')
        if ranlib:
            self.set_executables(ranlib=[ranlib])

        self.set_library_dirs(self.get_library_dirs())
        self.set_libraries(self.get_libraries())

        verbose = conf.get('verbose',[None,0])[1]
        if verbose:
            self.dump_properties()

    def dump_properties(self):
        """ Print out the attributes of a compiler instance. """
        props = []
        for key in self.executables.keys() + \
                ['version','libraries','library_dirs',
                 'object_switch','compile_switch']:
            if hasattr(self,key):
                v = getattr(self,key)
                props.append((key, None, '= '+`v`))
        props.sort()

        pretty_printer = FancyGetopt(props)
        for l in pretty_printer.generate_help("%s instance properties:" \
                                              % (self.__class__.__name__)):
            if l[:4]=='  --':
                l = '  ' + l[4:]
            print l

    def exec_command(self,*args,**kws):
        """ Return status,output of a command. """
        log.info('%s.exec_command(*%s,**%s)' % (self.__class__.__name__,args,kws))
        status, output = exec_command(*args,**kws)
        log.info('*****status:%s\n*****output:\n%s\n*****' % (status,output))
        return status, output

    ############################################################

    ## Private methods:

    def __get_cmd(self, command, envvar=None, confvar=None):
        if command is None:
            var = None
        elif type(command) is type(''):
            var = self.executables[command]
            if var is not None:
                var = var[0]
        else:
            var = command()
        if envvar is not None:
            var = os.environ.get(envvar, var)
        if confvar is not None:
            var = confvar[0].get(confvar[1], [None,var])[1]
        return var

    def __get_flags(self, command, envvar=None, confvar=None):
        if command is None:
            var = []
        elif type(command) is type(''):
            var = self.executables[command][1:]
        else:
            var = command()
        if envvar is not None:
            var = os.environ.get(envvar, var)
        if confvar is not None:
            var = confvar[0].get(confvar[1], [None,var])[1]
        if type(var) is type(''):
            var = split_quoted(var)
        return var

    ## class FCompiler

fcompiler_class = {'gnu':('gnufcompiler','GnuFCompiler',
                          "GNU Fortran Compiler"),
                   'pg':('pgfcompiler','PGroupFCompiler',
                         "Portland Group Fortran Compiler"),
                   'absoft':('absoftfcompiler','AbsoftFCompiler',
                             "Absoft Corp Fortran Compiler"),
                   'mips':('mipsfcompiler','MipsFCompiler',
                           "MIPSpro Fortran Compiler"),
                   'sun':('sunfcompiler','SunFCompiler',
                          "Sun|Forte Fortran 95 Compiler"),
                   'intel':('intelfcompiler','IntelFCompiler',
                            "Intel Fortran Compiler for 32-bit apps"),
                   'intelv':('intelfcompiler','IntelVisualFCompiler',
                             "Intel Visual Fortran Compiler for 32-bit apps"),
                   'intele':('intelfcompiler','IntelItaniumFCompiler',
                             "Intel Fortran Compiler for Itanium apps"),
                   'intelev':('intelfcompiler','IntelItaniumVisualFCompiler',
                              "Intel Visual Fortran Compiler for Itanium apps"),
                   'nag':('nagfcompiler','NAGFCompiler',
                          "NAGWare Fortran 95 Compiler"),
                   'compaq':('compaqfcompiler','CompaqFCompiler',
                             "Compaq Fortran Compiler"),
                   'compaqv':('compaqfcompiler','CompaqVisualFCompiler',
                             "DIGITAL|Compaq Visual Fortran Compiler"),
                   'vast':('vastfcompiler','VastFCompiler',
                           "Pacific-Sierra Research Fortran 90 Compiler"),
                   'hpux':('hpuxfcompiler','HPUXFCompiler',
                           "HP Fortran 90 Compiler"),
                   'lahey':('laheyfcompiler','LaheyFCompiler',
                            "Lahey/Fujitsu Fortran 95 Compiler"),
                   'f':('fcompiler','FFCompiler',
                        "Fortran Company/NAG F Compiler"),
                   }

_default_compilers = (
    # Platform mappings
    ('win32',('gnu','intelv','absoft','compaqv','intelitanium')),
    ('cygwin.*',('gnu','intelv','absoft','compaqv','intelitanium')),
    ('linux.*',('gnu','intel','lahey','pg','absoft','nag','vast','compaq',
                'intelitanium')),
    ('sunos.*',('forte','gnu','sun')),
    ('irix',('mips','gnu')),
    # OS mappings
    ('posix',('gnu',)),
    ('nt',('gnu',)),
    ('mac',('gnu',)),
    )

def get_default_fcompiler(osname=None, platform=None):
    """ Determine the default Fortran compiler to use for the given platform. """
    if osname is None:
        osname = os.name
    if platform is None:
        platform = sys.platform
    for pattern, compiler in _default_compilers:
        if re.match(pattern, platform) is not None or \
               re.match(pattern, osname) is not None:
            if type(compiler) is type(()):
                return compiler[0]
            return compiler
    return 'gnu'

def new_fcompiler(plat=None,
                  compiler=None,
                  verbose=0,
                  dry_run=0,
                  force=0):
    """ Generate an instance of some FCompiler subclass for the supplied
    platform/compiler combination.
    """
    if plat is None:
        plat = os.name
    try:
        if compiler is None:
            compiler = get_default_fcompiler(plat)
        (module_name, class_name, long_description) = fcompiler_class[compiler]
    except KeyError:
        msg = "don't know how to compile Fortran code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError, msg

    try:
        module_name = module_name
        __import__ (module_name)
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except ImportError:
        raise DistutilsModuleError, \
              "can't compile Fortran code: unable to load module '%s'" % \
              module_name
    except KeyError:
        raise DistutilsModuleError, \
              ("can't compile Fortran code: unable to find class '%s' " +
               "in module '%s'") % (class_name, module_name)

    return klass(None, dry_run, force)


def show_fcompilers(dist = None):
    """ Print list of available compilers (used by the "--help-fcompiler"
    option to "config_fc").
    """
    if dist is None:
        from dist import Distribution
        dist = Distribution()
        dist.script_name = os.path.basename(sys.argv[0])
        dist.script_args = ['config_fc'] + sys.argv[1:]
        dist.cmdclass['config_fc'] = config_fc
        dist.parse_config_files()
        dist.parse_command_line()

    compilers = []
    compilers_na = []
    compilers_ni = []
    for compiler in fcompiler_class.keys():
        v = 'N/A'
        try:
            c = new_fcompiler(compiler=compiler)
            c.customize(dist)
            v = c.get_version()
        except DistutilsModuleError:
            pass
        except:
            print sys.exc_info()[0],sys.exc_info()[1]
        if v is None:
            compilers_na.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2]))
        elif v=='N/A':
            compilers_ni.append(("fcompiler="+compiler, None,
                                 fcompiler_class[compiler][2]))
        else:
            compilers.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2] + ' (%s)' % v))
    compilers.sort()
    compilers_na.sort()
    pretty_printer = FancyGetopt(compilers)
    pretty_printer.print_help("List of available Fortran compilers:")
    pretty_printer = FancyGetopt(compilers_na)
    pretty_printer.print_help("List of unavailable Fortran compilers:")
    if compilers_ni:
        pretty_printer = FancyGetopt(compilers_ni)
        pretty_printer.print_help("List of unimplemented Fortran compilers:")
    print "For compiler details, run 'config_fc --verbose' setup command."

class config_fc(Command):
    """ Distutils command to hold user specified options
    to Fortran compilers.

    This is used in FCompiler.customize() method.
    """

    user_options = [
        ('fcompiler=',None,"specify Fortran compiler type"),
        ('f77-exec=', None, "specify F77 compiler command"),
        ('f90-exec=', None, "specify F90 compiler command"),
        ('f77flags=',None,"specify F77 compiler flags"),
        ('f90flags=',None,"specify F90 compiler flags"),
        ('opt=',None,"specify optimization flags"),
        ('arch=',None,"specify architecture specific optimization flags"),
        ('debug','g',"compile with debugging information"),
        ('noopt',None,"compile without optimization"),
        ('noarch',None,"compile without arch-dependent optimization"),
        ]

    boolean_options = ['debug','noopt','noarch']

    help_options = [
        ('help-fcompiler', None,
         "list available Fortran compilers", show_fcompilers),
        ]

def dummy_fortran_file():
    import tempfile
    dummy_name = tempfile.mktemp()+'__dummy'
    dummy = open(dummy_name+'.f','w')
    dummy.write("      subroutine dummy()\n      end\n")
    dummy.close()
    import atexit
    from distutils import log
    def rm_file(name=dummy_name,log_threshold=log._global_log.threshold):
        save_th = log._global_log.threshold
        log.set_threshold(log_threshold)
        try: os.remove(name+'.f'); log.debug('removed '+name+'.f')
        except OSError: pass
        try: os.remove(name+'.o'); log.debug('removed '+name+'.o')
        except OSError: pass
        log.set_threshold(save_th)
    atexit.register(rm_file)
    return dummy_name

is_f_file = re.compile(r'.*[.](for|ftn|f77|f)\Z',re.I).match
_has_f_header = re.compile(r'-[*]-\s*fortran\s*-[*]-',re.I).search
_has_f90_header = re.compile(r'-[*]-\s*f90\s*-[*]-',re.I).search
_free_f90_start = re.compile(r'[^c*][^\s\d\t]',re.I).match
def is_free_format(file):
    """Check if file is in free format Fortran."""
    # f90 allows both fixed and free format, assuming fixed unless
    # signs of free format are detected.
    result = 0
    f = open(file,'r')
    line = f.readline()
    n = 15 # the number of non-comment lines to scan for hints
    if _has_f_header(line):
        n = 0
    elif _has_f90_header(line):
        n = 0
        result = 1
    while n>0 and line:
        if line[0]!='!':
            n -= 1
            if _free_f90_start(line[:5]) or line[-2:-1]=='&':
                result = 1
                break
        line = f.readline()
    f.close()
    return result

if __name__ == '__main__':
    show_fcompilers()
