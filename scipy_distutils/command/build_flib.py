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
"""

import distutils
import distutils.dep_util, distutils.dir_util
import os,string
import commands,re
from types import *
from distutils.command.build_clib import build_clib

if os.name == 'nt':
    def run_command(command):
        """ not sure how to get exit status on nt. """
        in_pipe,out_pipe = os.popen4(command)
        in_pipe.close()
        text = out_pipe.read()
        return 0, text
else:
    run_command = commands.getstatusoutput
    
    
def show_compilers():
    for compiler in all_compilers:
        if compiler.is_available():
            print compiler

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
        self.fcompiler = None

    # initialize_options()

    def finalize_options (self):
        self.set_undefined_options('build',
                                   ('build_temp', 'build_flib'),
                                   ('build_temp', 'build_temp'),
                                   ('debug', 'debug'),
                                   ('force', 'force'))
        if self.fcompiler is None:
            self.fcompiler = find_fortran_compiler()
            if not self.fcompiler:
                raise ValueError, 'Fortran compiler not available'
            else:
                print self.fcompiler
        
        self.fortran_libraries = self.distribution.fortran_libraries
        if self.fortran_libraries:
            self.check_library_list(self.fortran_libraries)

    # finalize_options()

    def run (self):
        if not self.fortran_libraries:
            return

        self.build_libraries(self.fortran_libraries)

    # run ()

    def get_library_names(self):
        if not self.fortran_libraries:
            return None

        lib_names = [] 

        for (lib_name, build_info) in self.fortran_libraries:
            lib_names.append(lib_name)

        if self.fcompiler is not None:
            lib_names.extend(self.fcompiler.get_libraries())
            
        return lib_names

    # get_library_names ()

    def get_library_dirs(self):
        if not self.fortran_libraries:
            return []#None

        lib_dirs = [] 

        if self.fcompiler is not None:
            lib_dirs.extend(self.fcompiler.get_library_dirs())
            
        return lib_dirs

    # get_library_dirs ()

    def get_runtime_library_dirs(self):
        if not self.fortran_libraries:
            return []#None

        lib_dirs = [] 

        if self.fcompiler is not None:
            lib_dirs.extend(self.fcompiler.get_runtime_library_dirs())
            
        return lib_dirs

    # get_library_dirs ()

    def get_source_files (self):
        self.check_library_list(self.fortran_libraries)
        filenames = []

        # Gets source files specified 
        for ext in self.fortran_libraries:
            filenames.extend(ext[1]['sources'])

        return filenames    
                
    def build_libraries (self, fortran_libraries):
        
        fcompiler = self.fcompiler
        
        for (lib_name, build_info) in fortran_libraries:
            sources = build_info.get('sources')
            if sources is None or type(sources) not in (ListType, TupleType):
                raise DistutilsSetupError, \
                      ("in 'fortran_libraries' option (library '%s'), " +
                       "'sources' must be present and must be " +
                       "a list of source filenames") % lib_name
            sources = list(sources)
            module_dirs = build_info.get('module_dirs')
            module_files = build_info.get('module_files')
            self.announce(" building '%s' library" % lib_name)
            
            if module_files:
                fcompiler.build_library(lib_name, module_files,
                                        temp_dir=self.build_temp)
                
            fcompiler.build_library(lib_name, sources,
                                    module_dirs, temp_dir=self.build_temp)

        # for loop

    # build_libraries ()


class fortran_compiler_base:
    def to_object(self,dirty_files,module_dirs=None, temp_dir=''):
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
        module_switch = self.build_module_switch(module_dirs)
        file_pairs = self.source_and_object_pairs(source_files,temp_dir)
        object_files = []
        for source,object in file_pairs:
            if distutils.dep_util.newer(source,object):
                cmd =  compiler + ' ' + switches + \
                       module_switch + ' -c ' + source + ' -o ' + object 
                print cmd
                failure = os.system(cmd)
                if failure:
                    raise ValueError, 'failure during compile' 
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
                          output_dir='', debug=None):
        import string
        lib_file = os.path.join(output_dir,'lib'+library_name+'.a')
        newer = distutils.dep_util.newer
        # This doesn't work -- no way to know if the file is in the archive
        #object_files = filter(lambda o,lib=lib_file:\
        #                 distutils.dep_util.newer(o,lib),object_files)
        objects = string.join(object_files)
        if objects:
            cmd = 'ar -cur  %s %s' % (lib_file,objects)
            print cmd
            os.system(cmd)

    def build_library(self,library_name,source_list,module_dirs=None,
                      temp_dir = ''):
        #make sure the temp directory exists before trying to build files
        import distutils.dir_util
        distutils.dir_util.mkpath(temp_dir)
        #this compiles the files
        object_list = self.to_object(source_list,module_dirs,temp_dir)
        # actually we need to use all the object file names here to
        # make sure the library is always built.  It could occur that an
        # object file exists but hasn't been put in the archive. (happens
        # a lot when builds fail once and are restarted).
        object_list = self.source_to_object_names(source_list, temp_dir)
        #self.create_static_lib(object_list,library_name,temp_dir)           
        # This is pure bunk...
        # Windows fails for long argument strings on the command line.
        # if objects is real long (> 2048 chars or so on my machine),
        # the command fails (cmd.exe /e:2048 on w2k)
        # for now we'll split linking into to steps which should work for
        objects = object_list[:]
        while objects:
            obj,objects = objects[:20],objects[20:]
            self.create_static_lib(obj,library_name,temp_dir)

                   
    def dummy_fortran_files(self):
        import tempfile 
        d = tempfile.gettempdir()
        dummy_name = os.path.join(d,'__dummy.f')
        dummy = open(dummy_name,'w')
        dummy.write("\tsubroutine dummy()\n\tend\n")
        dummy.close()
        return (os.path.join(d,'__dummy.f'),os.path.join(d,'__dummy.o'))
    
    def is_available(self): 
        self.version = None
        # works I think only for unix...        
        #print 'command:', self.ver_cmd
        exit_status, out_text = run_command(self.ver_cmd)
        #print exit_status, out_text
        if not exit_status:
            m = re.match(self.ver_match,out_text)
            if m:
                self.version = m.group('version')           
        return self.version

    def get_libraries(self):
        return self.libraries
    def get_library_dirs(self):
        return self.library_dirs
    def get_extra_link_args(self):
        return []
    def get_runtime_library_dirs(self):
        return []

    def __str__(self):
        return "%s %s" % (self.vendor, self.version)

class absoft_fortran_compiler(fortran_compiler_base):
    def __init__(self):
        self.vendor = 'Absoft'
        self.version = '' 
        
        self.f90_compiler = 'f90'
        self.f77_compiler = 'f77'
        # got rid of -B108 cause it was generating 2 underscores instead
        # of one on the newest version.  Now we use -YEXT_SFX=_ to 
        # specify the output format
        if os.name == 'nt':
            self.f90_switches = '-f fixed  -YCFRL=1 -YCOM_NAMES=LCS'\
                                ' -YCOM_PFX  -YEXT_PFX -YEXT_NAMES=LCS' \
    			                ' -YCOM_SFX=_ -YEXT_SFX=_ -YEXT_NAMES=LCS'        
            self.f90_opt = '-O -Q100'
            self.f77_switches = '-N22 -N90 -N110'
            self.f77_opt = '-O -Q100'
            self.libraries = ['fio', 'fmath', 'f90math', 'COMDLG32']
        else:
            self.f90_switches = '-ffixed  -YCFRL=1 -YCOM_NAMES=LCS'\
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



        self.ver_cmd = 'f77 -V -c %s -o %s' % self.dummy_fortran_files()
        self.ver_match = r'FORTRAN 77 Compiler (?P<version>[^\s*,]*).*?Absoft Corp'
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
    def __init__(self):
        self.vendor = 'Sun'
        self.version = ''
        self.ver_cmd = 'f77 -V'
        self.ver_match =  r'f77: (?P<version>[^\s*,]*).*'

        self.f90_compiler = 'f90'
        self.f90_switches = ' -fixed '
        self.f90_opt = ' -fast -dalign '

        self.libraries = ['f90', 'F77', 'M77', 'sunmath', 'm']
        #threaded
        #self.libraries = ['f90', 'F77_mt', 'sunmath_mt', 'm', 'thread']
        #self.libraries = []
        self.library_dirs = self.find_lib_dir()
        #print 'sun:',self.library_dirs
        self.f77_compiler = 'f77'
        # not tested
        self.f77_switches = ' -pic '
        #self.f77_switches = ' -fixed '
        self.f77_opt = ' -fast -dalign '

    def build_module_switch(self,module_dirs):
        res = ''
        if module_dirs:
            for mod in module_dirs:
                res = res + ' -M' + mod
        return res

    def find_lib_dir(self):
        library_dirs = []
        lib_match = r'### f90: Note: LD_RUN_PATH\s*= '\
                     '(?P<lib_paths>[^\s.]*).*'
        cmd = self.f90_compiler + ' -dryrun dummy.f'
        exit_status, output = run_command(cmd)
        if not exit_status:
            libs = re.findall(lib_match,output)
            if libs:
                library_dirs = string.split(libs[0],':')
                self.is_available() # force version calculation
                compiler_home = os.path.dirname(library_dirs[0])
                library_dirs.append(os.path.join(compiler_home,
                                               self.version,'lib'))
        return library_dirs
    def get_runtime_library_dirs(self):
        return self.find_lib_dir()
    def get_extra_link_args(self):
        return ['-mimpure-text']

class mips_fortran_compiler(fortran_compiler_base):
    def __init__(self):
        self.vendor = 'SGI'
        self.version = '' 
        self.ver_cmd = 'f90 -version'
        self.ver_match =  r'MIPSpro Compilers: Version (?P<version>[^\s*,]*).*'
        
        self.f90_compiler = 'f90'
        self.f90_switches = ' -n32 -KPIC -fixedform '
        self.f90_opt = ' '                            
        
        self.libraries = ['fortran', 'ftn', 'm']
        #self.libraries = []
        #self.library_dirs = [] 
        self.library_dirs = self.find_lib_dir()
        
        self.f77_compiler = 'f77'
        # not tested
        self.f77_switches = ' -n32 -KPIC '
        self.f77_opt = ' -O3 '

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

class gnu_fortran_compiler(fortran_compiler_base):
    def __init__(self):
        self.vendor = 'Gnu'
        self.version = '' 
           
        self.f77_compiler = 'g77'
        if os.name == 'nt':
            self.f77_switches = ' -Wall'
        else:
            self.f77_switches = ' -fpic -Wall '  
        self.f77_opt = '-O3'
        self.libraries = ['g2c','gcc']
        self.library_dirs = self.find_lib_directories()
        self.ver_cmd = 'g77 -v'
        self.ver_match = r'g77 version (?P<version>[^\s*]*)'
 
    def f90_compile(self,source_files,module_files,temp_dir=''):
        raise ValueError, 'f90 not supported by gnu'
    
    def find_lib_directories(self):
        lib_dir = []
        match = r'Reading specs from (.*)/specs'

        # works I think only for unix...        
        exit_status, out_text = run_command('g77 -v')
        if not exit_status:
            m = re.findall(match,out_text)
            if m:
				lib_dir= m #m[0]          
        return lib_dir


def match_extension(files,ext):
    file_base_ext = map(lambda x: os.path.splitext(x),files)
    files = filter(lambda x,ext=ext: string.lower(x[1]) == ext,file_base_ext)
    files = map(lambda x: string.join(x,''), files)
    return files
    
def get_f77_files(files):
    return match_extension(files,'.f')

def get_f90_files(files):
    return match_extension(files,'.f90')

def get_fortran_files(files):    
    file_base_ext = map(lambda x: os.path.splitext(x),files)
    return get_f77_files(files) + get_f90_files(files)

def find_fortran_compiler():
    fcompiler = None
    for compiler in all_compilers:
        if compiler.is_available():
            fcompiler = compiler
            break
    return fcompiler

all_compilers = [absoft_fortran_compiler(),
                 mips_fortran_compiler(),
                 sun_fortran_compiler(),
                 gnu_fortran_compiler()] 


if __name__ == "__main__":
    for compiler in all_compilers:
        if compiler.is_available():
            break
    print compiler

