""" Information about platform and python version and compilers

    This information is manly used to build directory names that
    keep the object files and shared libaries straight when
    multiple platforms share the same file system.
"""

import os, sys

import distutils
from distutils.sysconfig import customize_compiler


try:
    from scipy_distutils.ccompiler import new_compiler
    from scipy_distutils.core import Extension, setup
    from scipy_distutils.command.build_ext import build_ext
except ImportError:
    from distutils.ccompiler import new_compiler
    from distutils.core import Extension, setup
    from distutils.command.build_ext import build_ext

import distutils.bcppcompiler

#from scipy_distutils import mingw32_support

def dummy_dist():
    # create a dummy distribution.  It will look at any site configuration files
    # and parse the command line to pick up any user configured stuff.  The 
    # resulting Distribution object is returned from setup.
    # Setting _setup_stop_after prevents the any commands from actually executing.
    distutils.core._setup_stop_after = "commandline"
    dist = setup(name="dummy")
    distutils.core._setup_stop_after = None
    return dist

def create_compiler_instance(dist):    
    # build_ext is in charge of building C/C++ files.
    # We are using it and dist to parse config files, and command line 
    # configurations.  There may be other ways to handle this, but I'm
    # worried I may miss one of the steps in distutils if I do it my self.
    #ext_builder = build_ext(dist)
    #ext_builder.finalize_options ()
    
    # For some reason the build_ext stuff wasn't picking up the compiler 
    # setting, so we grab it manually from the distribution object instead.
    opts = dist.command_options.get('build_ext',None)
    compiler_name = ''
    if opts:
        comp = opts.get('compiler',('',''))
        compiler_name = comp[1]
        
    # Create a new compiler, customize it based on the build settings,
    # and return it. 
    if not compiler_name:
        compiler_name = None
    print compiler_name    
    compiler = new_compiler(compiler=compiler_name)
    customize_compiler(compiler)
    return compiler

def compiler_exe_name(compiler):    
    exe_name = ''
    # this is really ugly...  Why aren't the attribute names 
    # standardized and used in a consistent way?
    if hasattr(compiler, "compiler"):
        # standard unix format
        exe_name = compiler.compiler[0]
    elif hasattr(compiler, "cc"):
        exe_name = compiler.cc
    elif compiler.__class__ is distutils.bcppcompiler.BCPPCompiler:
        exe_name = 'brcc32'
    return exe_name

def compiler_exe_path(exe_name):
    exe_path = None
    if os.path.exists(exe_name):
        exe_path = exe_name
    else:
        path_string = os.environ['PATH']
        path_string = os.path.expandvars(path_string)
        path_string = os.path.expanduser(path_string)
        paths = path_string.split(os.pathsep)
        for path in paths:
            path = os.path.join(path,exe_name)
            if os.path.exists(path):
                exe_path = path
                break               
            # needed to catch gcc on mingw32 installations.    
            path = path + '.exe'    
            if os.path.exists(path):
                exe_path = path
                break
    return exe_path

def check_sum(file):
    
    import md5
    try:
        f = open(file,'r')
        bytes = f.read(-1)
    except IOError:
        bytes = ''    
    chk_sum = md5.md5(bytes)
    return chk_sum.hexdigest()

def get_compiler_dir(compiler_name):
    """ Try to figure out the compiler directory based on the
        input compiler name.  This is fragile and really should
        be done at the distutils level inside the compiler.  I
        think it is only useful on windows at the moment.
    """
    compiler_type = choose_compiler(compiler_name)
    #print compiler_type
    configure_sys_argv(compiler_type)
    #print sys.argv
    dist = dummy_dist()    
    compiler_obj = create_compiler_instance(dist)
    #print compiler_obj.__class__
    exe_name = compiler_exe_name(compiler_obj)
    exe_path = compiler_exe_path(exe_name)
    if not exe_path:
        raise ValueError, "The '%s' compiler was not found." % compiler_name
    chk_sum = check_sum(exe_path)    
    restore_sys_argv()
    
    return 'compiler_'+chk_sum

#----------------------------------------------------------------------------
# Not needed -- used for testing.
#----------------------------------------------------------------------------

def choose_compiler(compiler_name=''):
    """ Try and figure out which compiler is gonna be used on windows.
        On other platforms, it just returns whatever value it is given.
        
        converts 'gcc' to 'mingw32' on win32
    """
    if not compiler_name:
        compiler_name = ''
        
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

old_argv = []
def configure_sys_argv(compiler_name):
    # We're gonna play some tricks with argv here to pass info to distutils 
    # which is really built for command line use. better way??
    global old_argv
    old_argv = sys.argv[:]        
    sys.argv = ['','build_ext','--compiler='+compiler_name]

def restore_sys_argv():
    sys.argv = old_argv

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

        # There was a change to 'distutils.msvccompiler' between Python 2.2
        # and Python 2.3.
        #
        # In Python 2.2 the function is 'get_devstudio_versions'
        # In Python 2.3 the function is 'get_build_version'
        try:
            version = distutils.msvccompiler.get_devstudio_versions()
            
        except:
            version = distutils.msvccompiler.get_build_version()
            
        if version:
            result = 1
    return result

if __name__ == "__main__":
    """
    import time
    t1 = time.time()    
    dist = dummy_dist()    
    compiler_obj = create_compiler_instance(dist)
    exe_name = compiler_exe_name(compiler_obj)
    exe_path = compiler_exe_path(exe_name)
    chk_sum = check_sum(exe_path)    
    
    t2 = time.time()
    print 'compiler exe:', exe_path
    print 'check sum:', chk_sum
    print 'time (sec):', t2 - t1
    print
    """
    path = get_compiler_dir('gcc')
    print 'gcc path:', path
    print
    try: 
        path = get_compiler_dir('msvc')
        print 'gcc path:', path
    except ValueError:
        pass    
