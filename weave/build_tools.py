""" Tools for compiling C/C++ code to extension modules

    The main function, build_extension(), takes the C/C++ file
    along with some other options and builds a Python extension.
    It uses distutils for most of the heavy lifting.
    
    choose_compiler() is also useful (mainly on windows anyway)
    for trying to determine whether MSVC++ or gcc is available.
    MSVC doesn't handle templates as well, so some of the code emitted
    by the python->C conversions need this info to choose what kind
    of code to create.
    
    The other main thing here is an alternative version of the MingW32
    compiler class.  The class makes it possible to build libraries with
    gcc even if the original version of python was built using MSVC.  It
    does this by converting a pythonxx.lib file to a libpythonxx.a file.
    Note that you need write access to the pythonxx/lib directory to do this.
"""

import sys,os,string,time
import tempfile
import exceptions
import commands

import platform_info

# If linker is 'gcc', this will convert it to 'g++'
# necessary to make sure stdc++ is linked in cross-platform way.
import distutils.sysconfig
import distutils.dir_util
old_init_posix = distutils.sysconfig._init_posix

def _init_posix():
    old_init_posix()
    ld = distutils.sysconfig._config_vars['LDSHARED']
    #distutils.sysconfig._config_vars['LDSHARED'] = ld.replace('gcc','g++')
    # FreeBSD names gcc as cc, so the above find and replace doesn't work.    
    # So, assume first entry in ld is the name of the linker -- gcc or cc or 
    # whatever.  This is a sane assumption, correct?
    # If the linker is gcc, set it to g++
    link_cmds = ld.split()    
    if gcc_exists(link_cmds[0]):
        link_cmds[0] = 'g++'
        ld = ' '.join(link_cmds)
    

    if (sys.platform == 'darwin'):
        # The Jaguar distributed python 2.2 has -arch i386 in the link line
        # which doesn't seem right.  It omits all kinds of warnings, so 
        # remove it.
        ld = ld.replace('-arch i386','')
        
        # The following line is a HACK to fix a problem with building the
        # freetype shared library under Mac OS X:
        ld += ' -framework AppKit'
        
        # 2.3a1 on OS X emits a ton of warnings about long double.  OPT
        # appears to not have all the needed flags set while CFLAGS does.
        cfg_vars = distutils.sysconfig._config_vars
        cfg_vars['OPT'] = cfg_vars['CFLAGS']        
    distutils.sysconfig._config_vars['LDSHARED'] = ld           
    
distutils.sysconfig._init_posix = _init_posix    
# end force g++


class CompileError(exceptions.Exception):
    pass


def create_extension(module_path, **kw):
    """ Create an Extension that can be buil by setup.py
        
        See build_extension for information on keyword arguments.
    """
    # some (most?) platforms will fail to link C++ correctly
    # unless scipy_distutils is used.
    try:
        from scipy_distutils.core import Extension
    except ImportError:
        from distutils.core import Extension
    
    # this is a screwy trick to get rid of a ton of warnings on Unix
    import distutils.sysconfig
    distutils.sysconfig.get_config_vars()
    if distutils.sysconfig._config_vars.has_key('OPT'):
        flags = distutils.sysconfig._config_vars['OPT']        
        flags = flags.replace('-Wall','')
        distutils.sysconfig._config_vars['OPT'] = flags
    
    # get the name of the module and the extension directory it lives in.  
    module_dir,cpp_name = os.path.split(os.path.abspath(module_path))
    module_name,ext = os.path.splitext(cpp_name)    
           
    # the business end of the function
    sources = kw.get('sources',[])
    kw['sources'] = [module_path] + sources        
        
    #--------------------------------------------------------------------
    # added access to environment variable that user can set to specify
    # where python (and other) include files are located.  This is 
    # very useful on systems where python is installed by the root, but
    # the user has also installed numerous packages in their own 
    # location.
    #--------------------------------------------------------------------
    if os.environ.has_key('PYTHONINCLUDE'):
        path_string = os.environ['PYTHONINCLUDE']        
        if sys.platform == "win32":
            extra_include_dirs = path_string.split(';')
        else:  
            extra_include_dirs = path_string.split(':')
        include_dirs = kw.get('include_dirs',[])
        kw['include_dirs'] = include_dirs + extra_include_dirs

    # SunOS specific
    # fix for issue with linking to libstdc++.a. see:
    # http://mail.python.org/pipermail/python-dev/2001-March/013510.html
    platform = sys.platform
    version = sys.version.lower()
    if platform[:5] == 'sunos' and version.find('gcc') != -1:
        extra_link_args = kw.get('extra_link_args',[])
        kw['extra_link_args'] = ['-mimpure-text'] +  extra_link_args
        
    ext = Extension(module_name, **kw)
    return ext    
                            
def build_extension(module_path,compiler_name = '',build_dir = None,
                    temp_dir = None, verbose = 0, **kw):
    """ Build the file given by module_path into a Python extension module.
    
        build_extensions uses distutils to build Python extension modules.
        kw arguments not used are passed on to the distutils extension
        module.  Directory settings can handle absoulte settings, but don't
        currently expand '~' or environment variables.
        
        module_path   -- the full path name to the c file to compile.  
                         Something like:  /full/path/name/module_name.c 
                         The name of the c/c++ file should be the same as the
                         name of the module (i.e. the initmodule() routine)
        compiler_name -- The name of the compiler to use.  On Windows if it 
                         isn't given, MSVC is used if it exists (is found).
                         gcc is used as a second choice. If neither are found, 
                         the default distutils compiler is used. Acceptable 
                         names are 'gcc', 'msvc' or any of the compiler names 
                         shown by distutils.ccompiler.show_compilers()
        build_dir     -- The location where the resulting extension module 
                         should be placed. This location must be writable.  If
                         it isn't, several default locations are tried.  If the 
                         build_dir is not in the current python path, a warning
                         is emitted, and it is added to the end of the path.
                         build_dir defaults to the current directory.
        temp_dir      -- The location where temporary files (*.o or *.obj)
                         from the build are placed. This location must be 
                         writable.  If it isn't, several default locations are 
                         tried.  It defaults to tempfile.gettempdir()
        verbose       -- 0, 1, or 2.  0 is as quiet as possible. 1 prints
                         minimal information.  2 is noisy.                 
        **kw          -- keyword arguments. These are passed on to the 
                         distutils extension module.  Most of the keywords
                         are listed below.

        Distutils keywords.  These are cut and pasted from Greg Ward's
        distutils.extension.Extension class for convenience:
        
        sources : [string]
          list of source filenames, relative to the distribution root
          (where the setup script lives), in Unix form (slash-separated)
          for portability.  Source files may be C, C++, SWIG (.i),
          platform-specific resource files, or whatever else is recognized
          by the "build_ext" command as source for a Python extension.
          Note: The module_path file is always appended to the front of this
                list                
        include_dirs : [string]
          list of directories to search for C/C++ header files (in Unix
          form for portability)          
        define_macros : [(name : string, value : string|None)]
          list of macros to define; each macro is defined using a 2-tuple,
          where 'value' is either the string to define it to or None to
          define it without a particular value (equivalent of "#define
          FOO" in source or -DFOO on Unix C compiler command line)          
        undef_macros : [string]
          list of macros to undefine explicitly
        library_dirs : [string]
          list of directories to search for C/C++ libraries at link time
        libraries : [string]
          list of library names (not filenames or paths) to link against
        runtime_library_dirs : [string]
          list of directories to search for C/C++ libraries at run time
          (for shared extensions, this is when the extension is loaded)
        extra_objects : [string]
          list of extra files to link with (eg. object files not implied
          by 'sources', static library that must be explicitly specified,
          binary resource files, etc.)
        extra_compile_args : [string]
          any extra platform- and compiler-specific information to use
          when compiling the source files in 'sources'.  For platforms and
          compilers where "command line" makes sense, this is typically a
          list of command-line arguments, but for other platforms it could
          be anything.
        extra_link_args : [string]
          any extra platform- and compiler-specific information to use
          when linking object files together to create the extension (or
          to create a new static Python interpreter).  Similar
          interpretation as for 'extra_compile_args'.
        export_symbols : [string]
          list of symbols to be exported from a shared extension.  Not
          used on all platforms, and not generally necessary for Python
          extensions, which typically export exactly one symbol: "init" +
          extension_name.
    """
    success = 0
    try:
        from scipy_distutils.core import setup, Extension
        from scipy_distutils.log import set_verbosity
        set_verbosity(-1)
    except ImportError:
        from distutils.core import setup, Extension
    
    # this is a screwy trick to get rid of a ton of warnings on Unix
    import distutils.sysconfig
    distutils.sysconfig.get_config_vars()
    if distutils.sysconfig._config_vars.has_key('OPT'):
        flags = distutils.sysconfig._config_vars['OPT']        
        flags = flags.replace('-Wall','')
        distutils.sysconfig._config_vars['OPT'] = flags
    
    # get the name of the module and the extension directory it lives in.  
    module_dir,cpp_name = os.path.split(os.path.abspath(module_path))
    module_name,ext = os.path.splitext(cpp_name)    
       
    # configure temp and build directories
    temp_dir = configure_temp_dir(temp_dir)    
    build_dir = configure_build_dir(module_dir)
    
    # dag. We keep having to add directories to the path to keep 
    # object files separated from each other.  gcc2.x and gcc3.x C++ 
    # object files are not compatible, so we'll stick them in a sub
    # dir based on their version.  This will add an md5 check sum
    # of the compiler binary to the directory name to keep objects
    # from different compilers in different locations.
    
    compiler_dir = platform_info.get_compiler_dir(compiler_name)
    temp_dir = os.path.join(temp_dir,compiler_dir)
    distutils.dir_util.mkpath(temp_dir)
    
    compiler_name = choose_compiler(compiler_name)
            
    configure_sys_argv(compiler_name,temp_dir,build_dir)
    
    # the business end of the function
    try:
        if verbose == 1:
            print 'Compiling code...'
            
        # set compiler verboseness 2 or more makes it output results
        if verbose > 1:
            verb = 1                
        else:
            verb = 0
        
        t1 = time.time()        
        ext = create_extension(module_path,**kw)
        # the switcheroo on SystemExit here is meant to keep command line
        # sessions from exiting when compiles fail.
        builtin = sys.modules['__builtin__']
        old_SysExit = builtin.__dict__['SystemExit']
        builtin.__dict__['SystemExit'] = CompileError
        
        # distutils for MSVC messes with the environment, so we save the
        # current state and restore them afterward.
        import copy
        environ = copy.deepcopy(os.environ)
        try:
            setup(name = module_name, ext_modules = [ext],verbose=verb)
        finally:
            # restore state
            os.environ = environ        
            # restore SystemExit
            builtin.__dict__['SystemExit'] = old_SysExit
        t2 = time.time()
        
        if verbose == 1:
            print 'finished compiling (sec): ', t2 - t1    
        success = 1
        configure_python_path(build_dir)
    except SyntaxError: #TypeError:
        success = 0    
            
    # restore argv after our trick...            
    restore_sys_argv()

    return success

old_argv = []
def configure_sys_argv(compiler_name,temp_dir,build_dir):
    # We're gonna play some tricks with argv here to pass info to distutils 
    # which is really built for command line use. better way??
    global old_argv
    old_argv = sys.argv[:]        
    sys.argv = ['','build_ext','--build-lib', build_dir,
                               '--build-temp',temp_dir]    
    if compiler_name == 'gcc':
        sys.argv.insert(2,'--compiler='+compiler_name)
    elif compiler_name:
        sys.argv.insert(2,'--compiler='+compiler_name)

def restore_sys_argv():
    sys.argv = old_argv
            
def configure_python_path(build_dir):    
    #make sure the module lives in a directory on the python path.
    python_paths = [os.path.abspath(x) for x in sys.path]
    if os.path.abspath(build_dir) not in python_paths:
        #print "warning: build directory was not part of python path."\
        #      " It has been appended to the path."
        sys.path.append(os.path.abspath(build_dir))

def choose_compiler(compiler_name=''):
    """ Try and figure out which compiler is gonna be used on windows.
        On other platforms, it just returns whatever value it is given.
        
        converts 'gcc' to 'mingw32' on win32
    """
    if sys.platform == 'win32':        
        if not compiler_name:
            # On Windows, default to MSVC and use gcc if it wasn't found
            # wasn't found.  If neither are found, go with whatever
            # the default is for distutils -- and probably fail...
            if msvc_exists():
                compiler_name = 'msvc'
            elif gcc_exists():
                compiler_name = 'mingw32'
        elif compiler_name == 'gcc':
                compiler_name = 'mingw32'
    else:
        # don't know how to force gcc -- look into this.
        if compiler_name == 'gcc':
                compiler_name = 'unix'                    
    return compiler_name
    
def gcc_exists(name = 'gcc'):
    """ Test to make sure gcc is found 
       
        Does this return correct value on win98???
    """
    result = 0
    cmd = '%s -v' % name
    try:
        w,r=os.popen4(cmd)
        w.close()
        str_result = r.read()
        #print str_result
        if string.find(str_result,'Reading specs') != -1:
            result = 1
    except:
        # This was needed because the msvc compiler messes with
        # the path variable. and will occasionlly mess things up
        # so much that gcc is lost in the path. (Occurs in test
        # scripts)
        result = not os.system(cmd)
    return result

def msvc_exists():
    """ Determine whether MSVC is available on the machine.
    """
    result = 0
    try:
        w,r=os.popen4('cl')
        w.close()
        str_result = r.read()
        #print str_result
        if string.find(str_result,'Microsoft') != -1:
            result = 1
    except:
        #assume we're ok if devstudio exists
        import distutils.msvccompiler
        version = distutils.msvccompiler.get_devstudio_version()
        if version:
            result = 1
    return result

if os.name == 'nt':
    def run_command(command):
        """ not sure how to get exit status on nt. """
        in_pipe,out_pipe = os.popen4(command)
        in_pipe.close()
        text = out_pipe.read()
        return 0, text
else:
    run_command = commands.getstatusoutput

        
def configure_temp_dir(temp_dir=None):
    if temp_dir is None:         
        temp_dir = tempfile.gettempdir()
    elif not os.path.exists(temp_dir) or not os.access(temp_dir,os.W_OK):
        print "warning: specified temp_dir '%s' does not exist " \
              "or is not writable. Using the default temp directory" % \
              temp_dir
        temp_dir = tempfile.gettempdir()

    # final check that that directories are writable.        
    if not os.access(temp_dir,os.W_OK):
        msg = "Either the temp or build directory wasn't writable. Check" \
              " these locations: '%s'" % temp_dir  
        raise ValueError, msg
    return temp_dir

def configure_build_dir(build_dir=None):
    # make sure build_dir exists and is writable
    if build_dir and (not os.path.exists(build_dir) or 
                      not os.access(build_dir,os.W_OK)):
        print "warning: specified build_dir '%s' does not exist " \
               "or is not writable. Trying default locations" % build_dir
        build_dir = None
        
    if build_dir is None:
        #default to building in the home directory of the given module.        
        build_dir = os.curdir
        # if it doesn't work use the current directory.  This should always
        # be writable.    
        if not os.access(build_dir,os.W_OK):
            print "warning:, neither the module's directory nor the "\
                  "current directory are writable.  Using the temporary"\
                  "directory."
            build_dir = tempfile.gettempdir()

    # final check that that directories are writable.
    if not os.access(build_dir,os.W_OK):
        msg = "The build directory wasn't writable. Check" \
              " this location: '%s'" % build_dir
        raise ValueError, msg
        
    return os.path.abspath(build_dir)        
    
if sys.platform == 'win32':
    import distutils.cygwinccompiler
    from distutils.version import StrictVersion
    from distutils.ccompiler import gen_preprocess_options, gen_lib_options
    from distutils.errors import DistutilsExecError, CompileError, UnknownFileError
    
    from distutils.unixccompiler import UnixCCompiler 
    
    # the same as cygwin plus some additional parameters
    class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
        """ A modified MingW32 compiler compatible with an MSVC built Python.
            
        """
    
        compiler_type = 'mingw32'
    
        def __init__ (self,
                      verbose=0,
                      dry_run=0,
                      force=0):
    
            distutils.cygwinccompiler.CygwinCCompiler.__init__ (self, 
                                                       verbose,dry_run, force)
            
            # we need to support 3.2 which doesn't match the standard
            # get_versions methods regex
            if self.gcc_version is None:
                import re
                out = os.popen('gcc' + ' -dumpversion','r')
                out_string = out.read()
                out.close()
                result = re.search('(\d+\.\d+)',out_string)
                if result:
                    self.gcc_version = StrictVersion(result.group(1))            

            # A real mingw32 doesn't need to specify a different entry point,
            # but cygwin 2.91.57 in no-cygwin-mode needs it.
            if self.gcc_version <= "2.91.57":
                entry_point = '--entry _DllMain@12'
            else:
                entry_point = ''
            if self.linker_dll == 'dllwrap':
                self.linker = 'dllwrap' + ' --driver-name g++'
            elif self.linker_dll == 'gcc':
                self.linker = 'g++'    

            # **changes: eric jones 4/11/01
            # 1. Check for import library on Windows.  Build if it doesn't exist.
            if not import_library_exists():
                build_import_library()
    
            # **changes: eric jones 4/11/01
            # 2. increased optimization and turned off all warnings
            # 3. also added --driver-name g++
            #self.set_executables(compiler='gcc -mno-cygwin -O2 -w',
            #                     compiler_so='gcc -mno-cygwin -mdll -O2 -w',
            #                     linker_exe='gcc -mno-cygwin',
            #                     linker_so='%s --driver-name g++ -mno-cygwin -mdll -static %s' 
            #                                % (self.linker, entry_point))
            if self.gcc_version <= "3.0.0":
                self.set_executables(compiler='gcc -mno-cygwin -O2 -w',
                                     compiler_so='gcc -mno-cygwin -mdll -O2 -w -Wstrict-prototypes',
                                     linker_exe='g++ -mno-cygwin',
                                     linker_so='%s -mno-cygwin -mdll -static %s' 
                                                % (self.linker, entry_point))
            else:            
                self.set_executables(compiler='gcc -mno-cygwin -O2 -w',
                                     compiler_so='gcc -O2 -w -Wstrict-prototypes',
                                     linker_exe='g++ ',
                                     linker_so='g++ -shared')
            # added for python2.3 support
            # we can't pass it through set_executables because pre 2.2 would fail
            self.compiler_cxx = ['g++']
            
            # Maybe we should also append -mthreads, but then the finished
            # dlls need another dll (mingwm10.dll see Mingw32 docs)
            # (-mthreads: Support thread-safe exception handling on `Mingw32')       
            
            # no additional libraries needed 
            self.dll_libraries=[]
            
        # __init__ ()

        def link(self,
                 target_desc,
                 objects,
                 output_filename,
                 output_dir,
                 libraries,
                 library_dirs,
                 runtime_library_dirs,
                 export_symbols=None, # export_symbols, we do this in our def-file
                 debug=0,
                 extra_preargs=None,
                 extra_postargs=None,
                 build_temp=None,
                 target_lang=None):
            if self.gcc_version < "3.0.0":
                distutils.cygwinccompiler.CygwinCCompiler.link(self,
                               target_desc,
                               objects,
                               output_filename,
                               output_dir,
                               libraries,
                               library_dirs,
                               runtime_library_dirs,
                               None, # export_symbols, we do this in our def-file
                               debug,
                               extra_preargs,
                               extra_postargs,
                               build_temp,
                               target_lang)
            else:
                UnixCCompiler.link(self,
                               target_desc,
                               objects,
                               output_filename,
                               output_dir,
                               libraries,
                               library_dirs,
                               runtime_library_dirs,
                               None, # export_symbols, we do this in our def-file
                               debug,
                               extra_preargs,
                               extra_postargs,
                               build_temp,
                               target_lang)

        
    # On windows platforms, we want to default to mingw32 (gcc)
    # because msvc can't build blitz stuff.
    # We should also check the version of gcc available...
    #distutils.ccompiler._default_compilers['nt'] = 'mingw32'
    #distutils.ccompiler._default_compilers = (('nt', 'mingw32'))
    # reset the Mingw32 compiler in distutils to the one defined above
    distutils.cygwinccompiler.Mingw32CCompiler = Mingw32CCompiler
    
    def import_library_exists():
        """ on windows platforms, make sure a gcc import library exists
        """
        if os.name == 'nt':
            lib_name = "libpython%d%d.a" % tuple(sys.version_info[:2])
            full_path = os.path.join(sys.prefix,'libs',lib_name)
            if not os.path.exists(full_path):
                return 0
        return 1
    
    def build_import_library():
        """ Build the import libraries for Mingw32-gcc on Windows
        """
        from scipy_distutils import lib2def
        #libfile, deffile = parse_cmd()
        #if deffile is None:
        #    deffile = sys.stdout
        #else:
        #    deffile = open(deffile, 'w')
        lib_name = "python%d%d.lib" % tuple(sys.version_info[:2])    
        lib_file = os.path.join(sys.prefix,'libs',lib_name)
        def_name = "python%d%d.def" % tuple(sys.version_info[:2])    
        def_file = os.path.join(sys.prefix,'libs',def_name)
        nm_cmd = '%s %s' % (lib2def.DEFAULT_NM, lib_file)
        nm_output = lib2def.getnm(nm_cmd)
        dlist, flist = lib2def.parse_nm(nm_output)
        lib2def.output_def(dlist, flist, lib2def.DEF_HEADER, open(def_file, 'w'))
        
        out_name = "libpython%d%d.a" % tuple(sys.version_info[:2])
        out_file = os.path.join(sys.prefix,'libs',out_name)
        dll_name = "python%d%d.dll" % tuple(sys.version_info[:2])
        args = (dll_name,def_file,out_file)
        cmd = 'dlltool --dllname %s --def %s --output-lib %s' % args
        success = not os.system(cmd)
        # for now, fail silently
        if not success:
            print 'WARNING: failed to build import library for gcc. Linking will fail.'
        #if not success:
        #    msg = "Couldn't find import library, and failed to build it."
        #    raise DistutilsPlatformError, msg
    
