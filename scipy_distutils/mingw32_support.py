"""
Support code for building Python extensions on Windows.

    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support) 
    # 3. Force windows to use g77

"""

import os, sys
import distutils.ccompiler

# I'd really like to pull this out of scipy and make it part of distutils...
import scipy_distutils.command.build_flib as build_flib


if sys.platform == 'win32':
    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support) 
    # 3. Force windows to use g77
    
    # 1.  Build libpython<version> from .lib and .dll if they don't exist.    
    def import_library_exists():
            """ on windows platforms, make sure a gcc import library exists
            """
            if sys.platform == 'win32':
                lib_name = "libpython%d%d.a" % tuple(sys.version_info[:2])
                full_path = os.path.join(sys.prefix,'libs',lib_name)
                #print full_path
                if not os.path.exists(full_path):
                    return 0
            return 1
        
    def build_import_library():
        """ Build the import libraries for Mingw32-gcc on Windows
        """
        # lib2def lives in weave
        sys.path.append(os.path.join('.','weave'))

        import lib2def
        #libfile, deffile = parse_cmd()
        #if deffile == None:
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
        print cmd
        success = not os.system(cmd)
        # for now, fail silently
        if not success:
            print "WARNING: failed to build import library for gcc. "\
                  "Linking will fail."
        #if not success:
        #    msg = "Couldn't find import library, and failed to build it."
        #    raise DistutilsPlatformError, msg
    
    def set_windows_compiler(compiler):
        distutils.ccompiler._default_compilers = (
        
            # Platform string mappings
        
            # on a cygwin built python we can use gcc like an ordinary UNIXish
            # compiler
            ('cygwin.*', 'unix'),
            
            # OS name mappings
            ('posix', 'unix'),
            ('nt', compiler),
            ('mac', 'mwerks'),
            
            )                
    def use_msvc():
        set_windows_compiler('msvc')
    
    def use_gcc(): 
        set_windows_compiler('mingw32')   
    
    standard_compiler_list = build_flib.all_compilers[:]
    def use_g77():
        build_flib.all_compilers = [build_flib.gnu_fortran_compiler]    
    
    def use_standard_fortran_compiler():
        build_flib.all_compilers = standard_compiler_list
        
    # 2. force the use of gcc on windows platform
    use_gcc()
    # 3. force the use of g77 on windows platform
    use_g77()
    if not import_library_exists():
        build_import_library()

    
