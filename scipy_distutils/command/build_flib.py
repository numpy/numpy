""" Implements the build_flib command which should go into Distutils
    at some point.
     
    Note:
    Right now, we're dynamically linking to the Fortran libraries on 
    some platforms (Sun for sure).  This is fine for local installations
    but a bad thing for redistribution because these libraries won't
    live on any machine that doesn't have a fortran compiler installed.
    It is pretty hard (impossible?) to get gcc to pass the right compiler
    flags on Sun to get the linker to use static libs for the fortran
    stuff.  Investigate further...

Bugs:
 ***  Options -e and -x have no effect when used with --help-compiler
       options. E.g. 
       ./setup.py build_flib --help-compiler -e g77-3.0
      finds g77-2.95.
      How to extract these options inside the show_compilers function?
 ***  compiler.is_available() method may not work correctly on nt
      because of lack of knowledge how to get exit status in
      run_command function. However, it may give reasonable results
      based on a version string.
 ***  Some vendors provide different compilers for F77 and F90
      compilations. Currently, checking the availability of these
      compilers is based on only checking the availability of the
      corresponding F77 compiler. If it exists, then F90 is assumed
      to exist also.
 ***  F compiler from Fortran Compiler is _not_ supported, though it
      is defined below. The reasons is that this F95 compiler is
      incomplete: it does not support external procedures
      that are needed to facilitate calling F90 module routines
      from C and subsequently from Python. See also
        http://cens.ioc.ee/pipermail/f2py-users/2002-May/000265.html

Open issues:
 ***  User-defined compiler flags. Do we need --fflags?

Fortran compilers (as to be used with --fcompiler= option):
      Absoft
      Sun
      SGI
      Intel
      Itanium
      NAG
      Compaq
      Digital
      Gnu
      VAST
      F       [unsupported]
"""

import distutils
import distutils.dep_util, distutils.dir_util
import os,sys,string
import commands,re
from types import *
from distutils.ccompiler import CCompiler,gen_preprocess_options
from distutils.command.build_clib import build_clib
from distutils.errors import *
from scipy_distutils.misc_util import red_text,green_text,yellow_text,\
     cyan_text

class FortranCompilerError (CCompilerError):
    """Some compile/link operation failed."""
class FortranCompileError (FortranCompilerError):
    """Failure to compile one or more Fortran source files."""
class FortranBuildError (FortranCompilerError):
    """Failure to build Fortran library."""

if os.name == 'nt':
    def run_command(command):
        """ not sure how to get exit status on nt. """
        in_pipe,out_pipe = os.popen4(command)
        in_pipe.close()
        text = out_pipe.read()
        return 0, text
else:
    run_command = commands.getstatusoutput

fcompiler_vendors = r'Absoft|Sun|SGI|Intel|Itanium|NAG|Compaq|Digital|Gnu|VAST|F'

def show_compilers():
    for compiler_class in all_compilers:
        compiler = compiler_class()
        if compiler.is_available():
            print cyan_text(compiler)

class build_flib (build_clib):

    description = "build f77/f90 libraries used by Python extensions"

    user_options = [
        ('build-flib', 'b',
         "directory to build f77/f90 libraries to"),
        ('build-temp', 't',
         "directory to put temporary build by-products"),
        ('debug', 'g',
         "compile with debugging information"),
        ('force', 'f',
         "forcibly build everything (ignore file timestamps)"),
        ('fcompiler=', 'c',
         "specify the compiler type"),
        ('fcompiler-exec=', 'e',
         "specify the path to F77 compiler"),
        ('f90compiler-exec=', 'x',
         "specify the path to F90 compiler"),
        ]

    boolean_options = ['debug', 'force']

    help_options = [
        ('help-compiler', None,
         "list available compilers", show_compilers),
        ]

    def initialize_options (self):

        self.build_flib = None
        self.build_temp = None

        self.fortran_libraries = None
        self.define = None
        self.undef = None
        self.debug = None
        self.force = 0
        self.fcompiler = os.environ.get('FC_VENDOR')
        if self.fcompiler \
           and not re.match(r'\A('+fcompiler_vendors+r')\Z',self.fcompiler):
            self.warn(red_text('Unknown FC_VENDOR=%s (expected %s)'\
                                   %(self.fcompiler,fcompiler_vendors)))
            self.fcompiler = None
        self.fcompiler_exec = os.environ.get('F77')
        self.f90compiler_exec = os.environ.get('F90')

    # initialize_options()

    def finalize_options (self):
        self.set_undefined_options('build',
                                   ('build_temp', 'build_flib'),
                                   ('build_temp', 'build_temp'),
                                   ('debug', 'debug'),
                                   ('force', 'force'))

        self.announce('running find_fortran_compiler')
        fc = find_fortran_compiler(self.fcompiler,
                                   self.fcompiler_exec,
                                   self.f90compiler_exec,
                                   verbose = self.verbose)
        if not fc:
            raise DistutilsOptionError, 'Fortran compiler not available: %s'\
                  % (self.fcompiler)
        else:
            self.announce('using '+cyan_text('%s Fortran compiler' % fc))
        self.fcompiler = fc
        if self.has_f_libraries():
            self.fortran_libraries = self.distribution.fortran_libraries
            self.check_library_list(self.fortran_libraries)
        
    # finalize_options()

    def has_f_libraries(self):
        return self.distribution.fortran_libraries \
               and len(self.distribution.fortran_libraries) > 0

    def run (self):
        if not self.has_f_libraries():
            return
        self.build_libraries(self.fortran_libraries)

    # run ()

    def has_f_library(self,name):
        if self.has_f_libraries():
            # If self.fortran_libraries is None at this point
            # then it means that build_flib was called before
            # build. Always call build before build_flib.
            for (lib_name, build_info) in self.fortran_libraries:
                if lib_name == name:
                    return 1
        
    def get_library_names(self, name=None):
        if not self.has_f_libraries():
            return None

        lib_names = []

        if name is None:
            for (lib_name, build_info) in self.fortran_libraries:
                lib_names.append(lib_name)

            if self.fcompiler is not None:
                lib_names.extend(self.fcompiler.get_libraries())
        else:
            for (lib_name, build_info) in self.fortran_libraries:
                if name != lib_name: continue
                for n in build_info.get('libraries',[]):
                    lib_names.append(n)
                    #XXX: how to catch recursive calls here?
                    lib_names.extend(self.get_library_names(n))
                break
        return lib_names

    def get_fcompiler_library_names(self):
        #if not self.has_f_libraries():
        #    return None
        if self.fcompiler is not None:
            return self.fcompiler.get_libraries()
        return []

    def get_fcompiler_library_dirs(self):
        #if not self.has_f_libraries():
        #    return None
        if self.fcompiler is not None:
            return self.fcompiler.get_library_dirs()
        return []

    # get_library_names ()

    def get_library_dirs(self, name=None):
        if not self.has_f_libraries():
            return []

        lib_dirs = [] 

        if name is None:
            if self.fcompiler is not None:
                lib_dirs.extend(self.fcompiler.get_library_dirs())
        else:
            for (lib_name, build_info) in self.fortran_libraries:
                if name != lib_name: continue
                lib_dirs.extend(build_info.get('library_dirs',[]))
                for n in build_info.get('libraries',[]):
                    lib_dirs.extend(self.get_library_dirs(n))
                break

        return lib_dirs

    # get_library_dirs ()

    def get_runtime_library_dirs(self):
        #if not self.has_f_libraries():
        #    return []

        lib_dirs = []

        if self.fcompiler is not None:
            lib_dirs.extend(self.fcompiler.get_runtime_library_dirs())
            
        return lib_dirs

    # get_library_dirs ()

    def get_source_files (self):
        if not self.has_f_libraries():
            return []

        self.check_library_list(self.fortran_libraries)
        filenames = []

        # Gets source files specified 
        for ext in self.fortran_libraries:
            filenames.extend(ext[1]['sources'])

        return filenames    
                
    def build_libraries (self, fortran_libraries):
        
        fcompiler = self.fcompiler
        
        for (lib_name, build_info) in fortran_libraries:
            self.announce(" building '%s' library" % lib_name)

            sources = build_info.get('sources')
            if sources is None or type(sources) not in (ListType, TupleType):
                raise DistutilsSetupError, \
                      ("in 'fortran_libraries' option (library '%s'), " +
                       "'sources' must be present and must be " +
                       "a list of source filenames") % lib_name
            sources = list(sources)
            module_dirs = build_info.get('module_dirs')
            module_files = build_info.get('module_files')


            include_dirs = build_info.get('include_dirs')

            if include_dirs:
                fcompiler.set_include_dirs(include_dirs)
            for n,v in build_info.get('define_macros') or []:
                fcompiler.define_macro(n,v)
            for n in build_info.get('undef_macros') or []:
                fcompiler.undefine_macro(n)

            if module_files:
                fcompiler.build_library(lib_name, module_files,
                                        temp_dir=self.build_temp)
                
            fcompiler.build_library(lib_name, sources,
                                    module_dirs, temp_dir=self.build_temp)

        # for loop

    # build_libraries ()

class fortran_compiler_base(CCompiler):

    vendor = None
    ver_match = None

    compiler_type = 'fortran'
    executables = {}

    compile_switch = ' -c '
    object_switch = ' -o '
    lib_prefix = 'lib'
    lib_suffix = '.a'
    lib_ar = 'ar -cur '
    lib_ranlib = 'ranlib '

    def __init__(self,verbose=0,dry_run=0,force=0):
        # Default initialization. Constructors of derived classes MUST
        # call this function.
        CCompiler.__init__(self,verbose,dry_run,force)

        self.version = None
        
        self.f77_switches = ''
        self.f77_opt = ''
        self.f77_debug = ''
        
        self.f90_switches = ''
        self.f90_opt = ''
        self.f90_debug = ''
        
        #self.libraries = []
        #self.library_dirs = []

        if self.vendor is None:
            raise DistutilsInternalError,\
                  '%s must define vendor attribute'%(self.__class__)
        if self.ver_match is None:
            raise DistutilsInternalError,\
                  '%s must define ver_match attribute'%(self.__class__)

    def to_object(self,
                  dirty_files,
                  module_dirs=None,
                  temp_dir=''):
        files = string.join(dirty_files)
        f90_files = get_f90_files(dirty_files)
        f77_files = get_f77_files(dirty_files)
        if f90_files != []:
            obj1 = self.f90_compile(f90_files,module_dirs,temp_dir = temp_dir)
        else:
            obj1 = []
        if f77_files != []:
            obj2 = self.f77_compile(f77_files, temp_dir = temp_dir)
        else:
            obj2 = []
        return obj1 + obj2

    def source_to_object_names(self,source_files, temp_dir=''):
        file_list = map(lambda x: os.path.basename(x),source_files)
        file_base_ext = map(lambda x: os.path.splitext(x),file_list)
        object_list = map(lambda x: x[0] +'.o',file_base_ext)
        object_files = map(lambda x,td=temp_dir: os.path.join(td,x),object_list)
        return object_files
        
    def source_and_object_pairs(self,source_files, temp_dir=''):
        object_files = self.source_to_object_names(source_files,temp_dir)
        file_pairs = zip(source_files,object_files)
        return file_pairs
 
    def f_compile(self,compiler,switches, source_files,
                  module_dirs=None, temp_dir=''):

        pp_opts = gen_preprocess_options(self.macros,self.include_dirs)

        switches = switches + ' ' + string.join(pp_opts,' ')

        module_switch = self.build_module_switch(module_dirs)
        file_pairs = self.source_and_object_pairs(source_files,temp_dir)
        object_files = []
        for source,object in file_pairs:
            if distutils.dep_util.newer(source,object):
                cmd =  compiler + ' ' + switches + ' '+\
                       module_switch + \
                       self.compile_switch + source + \
                       self.object_switch + object
                self.announce(yellow_text(cmd))
                failure = os.system(cmd)
                if failure:
                    raise FortranCompileError,\
                          'failure during compile (exit status = %s)' % failure
                object_files.append(object)
        return object_files
        #return all object files to make sure everything is archived 
        #return map(lambda x: x[1], file_pairs)

    def f90_compile(self,source_files,module_dirs=None, temp_dir=''):
        switches = string.join((self.f90_switches, self.f90_opt))
        return self.f_compile(self.f90_compiler,switches,
                              source_files, module_dirs,temp_dir)

    def f77_compile(self,source_files,module_dirs=None, temp_dir=''):
        switches = string.join((self.f77_switches, self.f77_opt))
        return self.f_compile(self.f77_compiler,switches,
                              source_files, module_dirs,temp_dir)

    def build_module_switch(self, module_dirs):
        return ''

    def create_static_lib(self, object_files, library_name,
                          output_dir='', debug=None, skip_ranlib=0):
        lib_file = os.path.join(output_dir,
                                self.lib_prefix+library_name+self.lib_suffix)
        objects = string.join(object_files)
        if objects:
            cmd = '%s%s %s' % (self.lib_ar,lib_file,objects)
            self.announce(yellow_text(cmd))
            failure = os.system(cmd)
            if failure:
                raise FortranBuildError,\
                      'failure during build (exit status = %s)'%failure 
            if self.lib_ranlib and not skip_ranlib:
                # Digital,MIPSPro compilers do not have ranlib.
                cmd = '%s %s' %(self.lib_ranlib,lib_file)
                self.announce(yellow_text(cmd))
                failure = os.system(cmd)
                if failure:
                    raise FortranBuildError,\
                          'failure during build (exit status = %s)'%failure 

    def build_library(self,library_name,source_list,module_dirs=None,
                      temp_dir = ''):
        #make sure the temp directory exists before trying to build files
        import distutils.dir_util
        distutils.dir_util.mkpath(temp_dir)

        #this compiles the files
        object_list = self.to_object(source_list,
                                     module_dirs,
                                     temp_dir)

        # actually we need to use all the object file names here to
        # make sure the library is always built.  It could occur that an
        # object file exists but hasn't been put in the archive. (happens
        # a lot when builds fail once and are restarted).
        object_list = self.source_to_object_names(source_list, temp_dir)

        if os.name == 'nt' or sys.platform[:4] == 'irix':
            # I (pearu) had the same problem on irix646 ...
            # I think we can make this "bunk" default as skip_ranlib
            # feature speeds things up.
            # XXX:Need to check if Digital compiler works here.

            # This is pure bunk...
            # Windows fails for long argument strings on the command line.
            # if objects is real long (> 2048 chars or so on my machine),
            # the command fails (cmd.exe /e:2048 on w2k).
            # for now we'll split linking into to steps which should work for
            objects = object_list[:]
            while objects:
                #obj,objects = objects[:20],objects[20:]
                i = 0
                obj = []
                while i<1900 and objects:
                    i = i + len(objects[0]) + 1
                    obj.append(objects[0])
                    objects = objects[1:]
                self.create_static_lib(obj,library_name,temp_dir,
                                       skip_ranlib = len(objects))
        else:
            self.create_static_lib(object_list,library_name,temp_dir)

    def dummy_fortran_files(self):
        import tempfile 
        d = tempfile.gettempdir()
        dummy_name = os.path.join(d,'__dummy.f')
        dummy = open(dummy_name,'w')
        dummy.write("      subroutine dummy()\n      end\n")
        dummy.close()
        return (os.path.join(d,'__dummy.f'),os.path.join(d,'__dummy.o'))
    
    def is_available(self):
        return self.get_version()
        
    def get_version(self):
        """Return the compiler version. If compiler is not available,
        return empty string."""
        # XXX: Are there compilers that have no version? If yes,
        #      this test will fail even if the compiler is available.
        if self.version is not None:
            # Finding version is expensive, so return previously found
            # version string.
            return self.version
        self.version = ''
        # works I think only for unix...
        #if self.verbose:
        self.announce('detecting %s Fortran compiler...'%(self.vendor))
        self.announce(yellow_text(self.ver_cmd))
        exit_status, out_text = run_command(self.ver_cmd)
        out_text2 = out_text.split('\n')[0]
        if not exit_status:
            self.announce('found %s' %(green_text(out_text2)))
            m = re.match(self.ver_match,out_text)
            if m:
                self.version = m.group('version')
        else:
            self.announce('%s: %s' % (exit_status,red_text(out_text2)))
        return self.version

    def get_libraries(self):
        return self.libraries
    def get_library_dirs(self):
        return self.library_dirs
    def get_extra_link_args(self):
        return []
    def get_runtime_library_dirs(self):
        return []
    def get_linker_so(self):
        """
        If a compiler requires specific linker then return a list
        containing a linker executable name and linker options.
        Otherwise, return None.
        """

    def __str__(self):
        return "%s %s" % (self.vendor, self.get_version())


class absoft_fortran_compiler(fortran_compiler_base):

    vendor = 'Absoft'
    ver_match = r'FORTRAN 77 Compiler (?P<version>[^\s*,]*).*?Absoft Corp'

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'f77'
        if f90c is None:
            f90c = 'f90'

        self.f77_compiler = fc
        self.f90_compiler = f90c

        # got rid of -B108 cause it was generating 2 underscores instead
        # of one on the newest version.  Now we use -YEXT_SFX=_ to 
        # specify the output format
        if os.name == 'nt':
            self.f90_switches = '-f fixed  -YCFRL=1 -YCOM_NAMES=LCS' \
                                ' -YCOM_PFX  -YEXT_PFX -YEXT_NAMES=LCS' \
                                ' -YCOM_SFX=_ -YEXT_SFX=_ -YEXT_NAMES=LCS'        
            self.f90_opt = '-O -Q100'
            self.f77_switches = '-N22 -N90 -N110'
            self.f77_opt = '-O -Q100'
            self.libraries = ['fio', 'f90math', 'fmath', 'COMDLG32']
        else:
            self.f90_switches = '-ffixed  -YCFRL=1 -YCOM_NAMES=LCS' \
                                ' -YCOM_PFX  -YEXT_PFX -YEXT_NAMES=LCS' \
                                ' -YCOM_SFX=_ -YEXT_SFX=_ -YEXT_NAMES=LCS'        
            self.f90_opt = '-O -B101'                            
            self.f77_switches = '-N22 -N90 -N110 -B108'
            self.f77_opt = '-O -B101'

            self.libraries = ['fio', 'f77math', 'f90math']
        
        try:
            dir = os.environ['ABSOFT'] 
            self.library_dirs = [os.path.join(dir,'lib')]
        except KeyError:
            self.library_dirs = []

        self.ver_cmd = self.f77_compiler + ' -V -c %s -o %s' % \
                       self.dummy_fortran_files()

    def build_module_switch(self,module_dirs):
        res = ''
        if module_dirs:
            for mod in module_dirs:
                res = res + ' -p' + mod
        return res

    def get_extra_link_args(self):
        return []
        # Couldn't get this to link for anything using gcc.
        #dr = "c:\\Absoft62\\lib"
        #libs = ['fio.lib', 'COMDLG32.lib','fmath.lib', 'f90math.lib','libcomdlg32.a' ]        
        #libs = map(lambda x,dr=dr:os.path.join(dr,x),libs)
        #return libs


class sun_fortran_compiler(fortran_compiler_base):
    """specify/detect settings for Sun's Forte compiler

    Recent Sun Fortran compilers are FORTRAN 90/95 beasts.  F77 support is
    handled by the same compiler, so even if you are asking for F77 you're
    getting a FORTRAN 95 compiler.  Since most (all?) the code currently
    being compiled for SciPy is FORTRAN 77 code, the list of libraries
    contains various F77-related libraries.  Not sure what would happen if
    you tried to actually compile and link FORTRAN 95 code with these
    settings.

    Note also that the 'Forte' name is passe.  Sun's latest compiler is
    named 'Sun ONE Studio 7, Compiler Collection'.  Heaven only knows what
    the version string for that baby will be.
    """
    
    vendor = 'Sun'

    # old compiler - any idea what the proper flags would be?
    #ver_match =  r'f77: (?P<version>[^\s*,]*)'

    ver_match = r'f90: (Forte Developer 7 Fortran 95|Sun) (?P<version>[^\s*,]*)'
    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'f77'
        if f90c is None:
            f90c = 'f90'

        self.f77_compiler = fc
        self.f77_switches = ' -pic'
        self.f77_opt = ' -fast -dalign -xtarget=generic -R/opt/SUNWspro/lib'

        self.f90_compiler = f90c
        # -fixed specifies fixed-format instead of free-format F90/95 code
        self.f90_switches = ' -pic'
        self.f90_opt = ' -fast -dalign -xtarget=generic -R/opt/SUNWspro/lib'

        self.ver_cmd = self.f90_compiler + ' -V'

        self.libraries = ['fsu', 'F77', 'M77', 'sunmath',
                          'mvec', 'f77compat', 'm']

        #threaded
        #self.libraries = ['f90', 'F77_mt', 'sunmath_mt', 'm', 'thread']
        #self.libraries = []
        if self.is_available():
            self.library_dirs = self.find_lib_dir()

    def build_module_switch(self,module_dirs):
        res = ''
        if module_dirs:
            for mod in module_dirs:
                res = res + ' -M' + mod
        return res

    def find_lib_dir(self):
        library_dirs = ["/opt/SUNWspro/prod/lib"]
        lib_match = r'### f90: Note: LD_RUN_PATH\s*= '\
                     '(?P<lib_paths>[^\s.]*).*'
        cmd = self.f90_compiler + ' -dryrun dummy.f'
        self.announce(yellow_text(cmd))
        exit_status, output = run_command(cmd)
        if not exit_status:
            libs = re.findall(lib_match,output)
            if libs and libs[0] == "(null)":
                del libs[0]
            if libs:
                library_dirs = string.split(libs[0],':')
                self.get_version() # force version calculation
                compiler_home = os.path.dirname(library_dirs[0])
                library_dirs.append(os.path.join(compiler_home,
                                                 self.version,'lib'))
        return library_dirs

    def get_extra_link_args(self):
        return ["-Bdynamic", "-G"]

    def get_linker_so(self):
        return [self.f77_compiler]


class mips_fortran_compiler(fortran_compiler_base):

    vendor = 'SGI'
    ver_match =  r'MIPSpro Compilers: Version (?P<version>[^\s*,]*)'
    lib_ranlib = '' # XXX: should we use `ar -s' here?

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'f77'
        if f90c is None:
            f90c = 'f90'

        self.f77_compiler = fc
        self.f77_switches = ' -n32 -KPIC '
        self.f77_opt = ' -O3 '

        self.f90_compiler = f90c
        self.f90_switches = ' -n32 -KPIC -fixedform ' # why fixed ???
        self.f90_opt = ' '                            

        self.ver_cmd = self.f77_compiler + ' -version '

        #self.libraries = ['fortran', 'ftn', 'm']
        # -lfortran is redundant with MIPSPro 7.30
        self.libraries = ['ftn', 'm']
        self.library_dirs = self.find_lib_dir()


    def build_module_switch(self,module_dirs):
        res = ''
        return res 
    def find_lib_dir(self):
        library_dirs = []
        return library_dirs
    def get_runtime_library_dirs(self):
	return self.find_lib_dir() 
    def get_extra_link_args(self):
	return []

class hpux_fortran_compiler(fortran_compiler_base):

    vendor = 'HP'
    ver_match =  r'HP F90 (?P<version>[^\s*,]*)'

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'f90'
        if f90c is None:
            f90c = 'f90'

        self.f77_compiler = fc
        self.f77_switches = ' +pic=long +ppu '
        self.f77_opt = ' -O3 '

        self.f90_compiler = f90c
        self.f90_switches = ' +pic=long +ppu '
        self.f90_opt = ' -O3 '

        self.ver_cmd = self.f77_compiler + ' +version '

        self.libraries = ['m']
        self.library_dirs = []

    def get_version(self):
        if self.version is not None:
            return self.version
        self.version = ''
        self.announce(yellow_text(self.ver_cmd))
        exit_status, out_text = run_command(self.ver_cmd)
        if self.verbose:
            out_text = out_text.split('\n')[0]
        if exit_status in [0,256]:
            # 256 seems to indicate success on HP-UX but keeping
            # also 0. Or does 0 exit status mean something different
            # in this platform?
            self.announce('found: '+green_text(out_text))
            m = re.match(self.ver_match,out_text)
            if m:
                self.version = m.group('version')
        else:
            self.announce('%s: %s' % (exit_status,red_text(out_text)))
        return self.version

class gnu_fortran_compiler(fortran_compiler_base):

    vendor = 'Gnu'
    ver_match = r'GNU Fortran (\(GCC\s*|)(?P<version>[^\s*\)]+)'
    gcc_lib_dir = None

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'g77'
        if f90c is None:
            f90c = fc

        self.f77_compiler = fc

        gcc_lib_dir = self.find_lib_directories()
        if gcc_lib_dir and \
           os.path.isfile(os.path.join(gcc_lib_dir[0],'libg2c-pic.a')):
            g2c = 'g2c-pic'
        else:
            g2c = 'g2c'
        if sys.platform == 'win32':
            self.libraries = ['gcc',g2c]
            self.library_dirs = gcc_lib_dir
        else:
            # On linux g77 does not need lib_directories to be specified.
            self.libraries = [g2c]

        switches = ' -Wall -fno-second-underscore '

        if os.name != 'nt':
            switches = switches + ' -fPIC '

        self.f77_switches = switches
        self.ver_cmd = self.f77_compiler + ' --version '

        self.f77_opt = self.get_opt()

    def get_opt(self):
        import cpuinfo
        cpu = cpuinfo.cpuinfo()
        opt = ' -O3 -funroll-loops '

        # only check for more optimization if g77 can handle it.
        if self.get_version():
            march_flag = 1
            if self.version == '0.5.26': # gcc 3.0
                if cpu.is_AthlonK6():
                    opt = opt + ' -march=k6 '
                elif cpu.is_AthlonK7():
                    opt = opt + ' -march=athlon '
                else:
                    march_flag = 0
            elif self.version >= '3.1.1': # gcc >= 3.1.1
                if cpu.is_AthlonK6():
                    opt = opt + ' -march=k6 '
                elif cpu.is_AthlonK7():
                    opt = opt + ' -march=athlon '
                elif cpu.is_PentiumIV():
                    opt = opt + ' -march=pentium4 '
                elif cpu.is_PentiumIII():
                    opt = opt + ' -march=pentium3 '
                elif cpu.is_PentiumII():
                    opt = opt + ' -march=pentium2 '
                else:
                    march_flag = 0
                if cpu.has_mmx(): opt = opt + ' -mmmx '
                if cpu.has_sse(): opt = opt + ' -msse '
                if cpu.has_sse2(): opt = opt + ' -msse2 '
                if cpu.has_3dnow(): opt = opt + ' -m3dnow '
            else:
                march_flag = 0
            if march_flag:
                pass
            elif cpu.is_i686():
                opt = opt + ' -march=i686 '
            elif cpu.is_i586():
                opt = opt + ' -march=i586 '
            elif cpu.is_i486():
                opt = opt + ' -march=i486 '
            elif cpu.is_i386():
                opt = opt + ' -march=i386 '
            if cpu.is_Intel():
                opt = opt + ' -malign-double -fomit-frame-pointer '     
        return opt
        
    def find_lib_directories(self):
        if self.gcc_lib_dir is not None:
            return self.gcc_lib_dir
        self.announce('running gnu_fortran_compiler.find_lib_directories')
        self.gcc_lib_dir = []
        lib_dir = []
        match = r'Reading specs from (.*)/specs'

        # works I think only for unix...
        cmd = '%s -v' % self.f77_compiler
        self.announce(yellow_text(cmd))
        exit_status, out_text = run_command(cmd)
        if not exit_status:
            m = re.findall(match,out_text)
            if m:
                assert len(m)==1,`m`
                self.gcc_lib_dir = m
        return self.gcc_lib_dir

    def get_linker_so(self):
        lnk = None
        # win32 linking should be handled by standard linker
        if ((sys.platform  != 'win32')  and
            (sys.platform  != 'cygwin') and
            (os.uname()[0] != 'Darwin')):
            lnk = [self.f77_compiler,'-shared']
        return lnk

    def get_extra_link_args(self):
        # SunOS often has dynamically loaded symbols defined in the
        # static library libg2c.a  The linker doesn't like this.  To
        # ignore the problem, use the -mimpure-text flag.  It isn't
        # the safest thing, but seems to work.
        args = []  
        if (hasattr(os,'uname') and (os.uname()[0] == 'SunOS')):
            args =  ['-mimpure-text']
        return args

    def f90_compile(self,source_files,module_files,temp_dir=''):
        raise DistutilsExecError, 'f90 not supported by Gnu'


#http://developer.intel.com/software/products/compilers/f60l/
class intel_ia32_fortran_compiler(fortran_compiler_base):

    vendor = 'Intel' # Intel(R) Corporation 
    ver_match = r'Intel\(R\) Fortran Compiler for 32-bit applications, '\
                'Version (?P<version>[^\s*]*)'

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'ifc'
        if f90c is None:
            f90c = fc

        self.f77_compiler = fc
        self.f90_compiler = f90c

        switches = ' -KPIC '

        import cpuinfo
        cpu = cpuinfo.cpuinfo()
        if cpu.has_fdiv_bug():
            switches = switches + ' -fdiv_check '
        if cpu.has_f00f_bug():
            switches = switches + ' -0f_check '
        self.f77_switches = self.f90_switches = switches
        self.f77_switches = self.f77_switches + ' -FI -w90 -w95 -cm -c '

        self.f77_opt = self.f90_opt = self.get_opt()
        
        debug = ' -g ' # usage of -C sometimes causes segfaults
        self.f77_debug =  self.f90_debug = debug

        #self.f77_switches = self.f77_switches + self.f77_debug
        self.ver_cmd = self.f77_compiler+' -FI -V -c %s -o %s' %\
                       self.dummy_fortran_files()

        #self.libraries = ['imf']

    def get_opt(self):
        import cpuinfo
        cpu = cpuinfo.cpuinfo()
        opt = ' -O3 -unroll '
        if cpu.is_PentiumPro() or cpu.is_PentiumII():
            opt = opt + ' -tpp6 -xi '
        elif cpu.is_PentiumIII():
            opt = opt + ' -tpp6 '
        elif cpu.is_Pentium():
            opt = opt + ' -tpp5 '
        elif cpu.is_PentiumIV():
            opt = opt + ' -tpp7 -xW '
        elif cpu.has_mmx():
            opt = opt + ' -xM '
        return opt

    def get_linker_so(self):
        return [self.f77_compiler,'-shared']


class intel_itanium_fortran_compiler(intel_ia32_fortran_compiler):

    vendor = 'Itanium'
    ver_match = r'Intel\(R\) Fortran 90 Compiler Itanium\(TM\) Compiler'\
                ' for the Itanium\(TM\)-based applications,'\
                ' Version (?P<version>[^\s*]*)'

    def __init__(self, fc=None, f90c=None, verbose=0):
        if fc is None:
            fc = 'efc'
        intel_ia32_fortran_compiler.__init__(self, fc, f90c, verbose=verbose)


class nag_fortran_compiler(fortran_compiler_base):

    vendor = 'NAG'
    ver_match = r'NAGWare Fortran 95 compiler Release (?P<version>[^\s]*)'

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'f95'
        if f90c is None:
            f90c = fc

        self.f77_compiler = fc
        self.f90_compiler = f90c

        switches = ''
        debug = ' -g -gline -g90 -nan -C '

        self.f77_switches = self.f90_switches = switches
        self.f77_switches = self.f77_switches + ' -fixed '
        self.f77_debug = self.f90_debug = debug
        self.f77_opt = self.f90_opt = self.get_opt()

        self.ver_cmd = self.f77_compiler+' -V '

    def get_opt(self):
        opt = ' -O4 -target=native '
        return opt

    def get_linker_so(self):
        return [self.f77_compiler,'-Wl,-shared']


# http://www.fortran.com/F/compilers.html
#
# We define F compiler here but it is quite useless
# because it does not support external procedures
# which are needed for calling F90 module routines
# through f2py generated wrappers.
class f_fortran_compiler(fortran_compiler_base):

    vendor = 'F'
    ver_match = r'Fortran Company/NAG F compiler Release (?P<version>[^\s]*)'

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'F'
        if f90c is None:
            f90c = 'F'

        self.f77_compiler = fc
        self.f90_compiler = f90c
        self.ver_cmd = self.f90_compiler+' -V '

        gnu = gnu_fortran_compiler('g77')
        if not gnu.is_available(): # F compiler requires gcc.
            self.version = ''
            return
        if not self.is_available():
            return

        if self.verbose:
            print red_text("""
WARNING: F compiler is unsupported due to its incompleteness.
         Send complaints to its vendor. For adding its support
         to scipy_distutils, it must support external procedures.
""")

        self.f90_switches = ''
        self.f90_debug = ' -g -gline -g90 -C '
        self.f90_opt = ' -O '

        #self.f77_switches = gnu.f77_switches
        #self.f77_debug = gnu.f77_debug
        #self.f77_opt = gnu.f77_opt

    def get_linker_so(self):
        return ['gcc','-shared']


class vast_fortran_compiler(fortran_compiler_base):

    vendor = 'VAST'
    ver_match = r'\s*Pacific-Sierra Research vf90 (Personal|Professional)'\
                '\s+(?P<version>[^\s]*)'

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'g77'
        if f90c is None:
            f90c = 'f90'

        self.f77_compiler = fc
        self.f90_compiler = f90c

        d,b = os.path.split(f90c)
        vf90 = os.path.join(d,'v'+b)
        self.ver_cmd = vf90+' -v '

        gnu = gnu_fortran_compiler(fc)
        if not gnu.is_available(): # VAST compiler requires g77.
            self.version = ''
            return
        if not self.is_available():
            return

        self.f77_switches = gnu.f77_switches
        self.f77_debug = gnu.f77_debug
        self.f77_opt = gnu.f77_opt        

        # XXX: need f90 switches, debug, opt

    def get_linker_so(self):
        return [self.f90_compiler,'-shared']


class compaq_fortran_compiler(fortran_compiler_base):

    vendor = 'Compaq'
    ver_match = r'Compaq Fortran (?P<version>[^\s]*)'

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'fort'
        if f90c is None:
            f90c = fc

        self.f77_compiler = fc
        self.f90_compiler = f90c

        switches = ' -assume no2underscore -nomixed_str_len_arg '
        debug = ' -g -check_bounds '

        self.f77_switches = self.f90_switches = switches
        self.f77_debug = self.f90_debug = debug
        self.f77_opt = self.f90_opt = self.get_opt()

        # XXX: uncomment if required
        #self.libraries = ' -lUfor -lfor -lFutil -lcpml -lots -lc '

        # XXX: fix the version showing flag
        self.ver_cmd = self.f77_compiler+' -V '

    def get_opt(self):
        opt = ' -O4 -align dcommons -arch host -assume bigarrays'\
              ' -assume nozsize -math_library fast -tune host '
        return opt

    def get_linker_so(self):
        return [self.f77_compiler,'-shared']


#http://www.compaq.com/fortran
class digital_fortran_compiler(fortran_compiler_base):

    vendor = 'Digital'
    ver_match = r'DIGITAL Visual Fortran Optimizing Compiler'\
                ' Version (?P<version>[^\s]*).*'

    compile_switch = ' /c '
    object_switch = ' /object:'
    lib_prefix = ''
    lib_suffix = '.lib'
    lib_ar = 'lib.exe /OUT:'
    lib_ranlib = ''

    def __init__(self, fc=None, f90c=None, verbose=0):
        fortran_compiler_base.__init__(self, verbose=verbose)

        if fc is None:
            fc = 'DF'
        if f90c is None:
            f90c = fc

        self.f77_compiler = fc
        self.f90_compiler = f90c
        self.ver_cmd = self.f77_compiler+' /what '

        if self.is_available():
            #XXX: is this really necessary???
            from distutils.msvccompiler import find_exe
            self.lib_ar = find_exe("lib.exe", self.version) + ' /OUT:'

        switches = ' /nologo /MD /W1 /iface:cref /iface=nomixed_str_len_arg '
        debug = ' '

        self.f77_switches = ' /f77rtl /fixed ' + switches
        self.f90_switches = switches
        self.f77_debug = self.f90_debug = debug
        self.f77_opt = self.f90_opt = self.get_opt()

    def get_opt(self):
        # XXX: use also /architecture, see gnu_fortran_compiler
        return ' /Ox '


def match_extension(files,ext):
    match = re.compile(r'.*[.]('+ext+r')\Z',re.I).match
    return filter(lambda x,match = match: match(x),files)

def get_f77_files(files):
    return match_extension(files,'for|f77|ftn|f')

def get_f90_files(files):
    return match_extension(files,'f90|f95')

def get_fortran_files(files):
    return match_extension(files,'f90|f95|for|f77|ftn|f')

def find_fortran_compiler(vendor=None, fc=None, f90c=None, verbose=0):
    for compiler_class in all_compilers:
        if vendor is not None and vendor != compiler_class.vendor:
            continue
        #print compiler_class
        compiler = compiler_class(fc,f90c,verbose = verbose)
        if compiler.is_available():
            return compiler
    return None

all_compilers = [absoft_fortran_compiler,
                 mips_fortran_compiler,
                 sun_fortran_compiler,
                 intel_ia32_fortran_compiler,
                 intel_itanium_fortran_compiler,
                 nag_fortran_compiler,
                 compaq_fortran_compiler,
                 digital_fortran_compiler,
                 vast_fortran_compiler,
                 hpux_fortran_compiler,
                 f_fortran_compiler,
                 gnu_fortran_compiler,
                 ]

if __name__ == "__main__":
    show_compilers()
