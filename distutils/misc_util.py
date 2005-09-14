
import os,sys,string
import re
import types
import glob

if sys.version[:3]<='2.1':
    from distutils import util
    util_get_platform = util.get_platform
    util.get_platform = lambda : util_get_platform().replace(' ','_')

def cyg2win32(path):
    if sys.platform=='cygwin' and path.startswith('/cygdrive'):
        path = path[10] + ':' + os.path.normcase(path[11:])
    return path

# Hooks for colored terminal output.
# See also http://www.livinglogic.de/Python/ansistyle
def terminal_has_colors():
    if sys.platform=='cygwin' and not os.environ.has_key('USE_COLOR'):
        # Avoid importing curses that causes illegal operation
        # with a message:
        #  PYTHON2 caused an invalid page fault in
        #  module CYGNURSES7.DLL as 015f:18bbfc28
        # Details: Python 2.3.3 [GCC 3.3.1 (cygming special)]
        #          ssh to Win32 machine from debian
        #          curses.version is 2.2
        #          CYGWIN_98-4.10, release 1.5.7(0.109/3/2))
        return 0
    if hasattr(sys.stdout,'isatty') and sys.stdout.isatty(): 
        try:
            import curses
            curses.setupterm()
            if (curses.tigetnum("colors") >= 0
                and curses.tigetnum("pairs") >= 0
                and ((curses.tigetstr("setf") is not None 
                      and curses.tigetstr("setb") is not None) 
                     or (curses.tigetstr("setaf") is not None
                         and curses.tigetstr("setab") is not None)
                     or curses.tigetstr("scp") is not None)):
                return 1
        except Exception,msg:
            pass
    return 0

if terminal_has_colors():
    def red_text(s): return '\x1b[31m%s\x1b[0m'%s
    def green_text(s): return '\x1b[32m%s\x1b[0m'%s
    def yellow_text(s): return '\x1b[33m%s\x1b[0m'%s
    def blue_text(s): return '\x1b[34m%s\x1b[0m'%s
    def cyan_text(s): return '\x1b[35m%s\x1b[0m'%s
else:
    def red_text(s): return s
    def green_text(s): return s
    def yellow_text(s): return s
    def cyan_text(s): return s
    def blue_text(s): return s

class PostponedException:
    """Postpone exception until an attempt is made to use a resource."""
    #Example usage:
    #  try: import foo
    #  except ImportError: foo = PostponedException()
    __all__ = []
    def __init__(self):
        self._info = sys.exc_info()[:2]
        self.__doc__ = '%s: %s' % tuple(self._info)
    def __getattr__(self,name):
        raise self._info[0],self._info[1]

def get_path(mod_name,parent_path=None):
    """ This function makes sure installation is done from the
        correct directory no matter if it is installed from the
        command line or from another package or run_setup function.
        
    """
    if mod_name == '__main__':
        d = os.path.abspath('.')
    elif mod_name == '__builtin__':
        #builtin if/then added by Pearu for use in core.run_setup.        
        d = os.path.dirname(os.path.abspath(sys.argv[0]))
    else:
        mod = __import__(mod_name)
        file = mod.__file__
        d = os.path.dirname(os.path.abspath(file))
    if parent_path is not None:
        pd = os.path.abspath(parent_path)
        if pd==d[:len(pd)]:
            d = d[len(pd)+1:]
    return d or '.'
    
def add_local_to_path(mod_name):
    local_path = get_path(mod_name)
    sys.path.insert(0,local_path)

def add_grandparent_to_path(mod_name):
    local_path = get_path(mod_name)
    gp_dir = os.path.split(local_path)[0]
    sys.path.insert(0,gp_dir)

def restore_path():
    del sys.path[0]

def append_package_dir_to_path(package_name):           
    """ Search for a directory with package_name and append it to PYTHONPATH
        
        The local directory is searched first and then the parent directory.
    """
    # first see if it is in the current path
    # then try parent.  If it isn't found, fail silently
    # and let the import error occur.
    
    # not an easy way to clean up after this...
    import os,sys
    if os.path.exists(package_name):
        sys.path.append(package_name)
    elif os.path.exists(os.path.join('..',package_name)):
        sys.path.append(os.path.join('..',package_name))

def get_package_config(package_name):
    """ grab the configuration info from the setup_xxx.py file
        in a package directory.  The package directory is searched
        from the current directory, so setting the path to the
        setup.py file directory of the file calling this is usually
        needed to get search the path correct.
    """
    append_package_dir_to_path(package_name)
    mod = __import__('setup_'+package_name)
    config = mod.configuration()
    return config

def package_config(primary,dependencies=[]):
    """ Create a configuration dictionary ready for setup.py from
        a list of primary and dependent package names.  Each
        package listed must have a directory with the same name
        in the current or parent working directory.  Further, it
        should have a setup_xxx.py module within that directory that
        has a configuration() function in it. 
    """
    config = []
    config.extend([get_package_config(x) for x in primary])
    config.extend([get_package_config(x) for x in dependencies])        
    config_dict = merge_config_dicts(config)
    return config_dict
        
list_keys = ['packages', 'ext_modules', 'data_files',
             'include_dirs', 'libraries', 'fortran_libraries',
             'headers', 'scripts']
dict_keys = ['package_dir']

def default_config_dict(name = None, parent_name = None, local_path=None):
    """ Return a configuration dictionary for usage in
    configuration() function defined in file setup_<name>.py.
    """
    d={}
    for key in list_keys: d[key] = []
    for key in dict_keys: d[key] = {}

    full_name = dot_join(parent_name,name)

    if full_name:
        # XXX: The following assumes that default_config_dict is called
        #      only from setup_<name>.configuration().
        #      Todo: implement check for this assumption.
        if local_path is None:
            frame = get_frame(1)
            caller_name = eval('__name__',frame.f_globals,frame.f_locals)
            local_path = get_path(caller_name)
        test_path = os.path.join(local_path,'tests')
        if 0 and name and parent_name is None:
            # Useful for local builds
            d['version'] = get_version(path=local_path)
        if os.path.exists(os.path.join(local_path,'__init__.py')):
            d['packages'].append(full_name)
            d['package_dir'][full_name] = local_path
        if os.path.exists(test_path):
            d['packages'].append(dot_join(full_name,'tests'))
            d['package_dir'][dot_join(full_name,'tests')] = test_path
        d['name'] = full_name
        if 0 and not parent_name:
            # Include scipy_distutils to local distributions
            for p in ['.','..']:
                dir_name = os.path.abspath(os.path.join(local_path,
                                                        p,'scipy_distutils'))
                if os.path.exists(dir_name):
                    d['packages'].append('scipy_distutils')
                    d['packages'].append('scipy_distutils.command')
                    d['package_dir']['scipy_distutils'] = dir_name
                    break
    return d

def get_frame(level=0):
    try:
        return sys._getframe(level+1)
    except AttributeError:
        frame = sys.exc_info()[2].tb_frame
        for i in range(level+1):
            frame = frame.f_back
        return frame

def merge_config_dicts(config_list):
    result = default_config_dict()
    for d in config_list:
        if not d: continue
        name = d.get('name',None)
        if name is not None:
            result['name'] = name
            break
    for d in config_list:
        if not d: continue
        for key in list_keys:
            result[key].extend(d.get(key,[]))
        for key in dict_keys:
            result[key].update(d.get(key,{}))
    return result

def dict_append(d,**kws):
    for k,v in kws.items():
        if d.has_key(k):
            d[k].extend(v)
        else:
            d[k] = v

def dot_join(*args):
    return string.join(filter(None,args),'.')

def fortran_library_item(lib_name,
                         sources,
                         **attrs
                         ):   #obsolete feature
    """ Helper function for creating fortran_libraries items. """
    build_info = {'sources':sources}
    known_attrs = ['module_files','module_dirs',
                   'libraries','library_dirs']
    for key,value in attrs.items():
        if key not in known_attrs:
            raise TypeError,\
                  "fortran_library_item() got an unexpected keyword "\
                  "argument '%s'" % key
        build_info[key] = value
    
    return (lib_name,build_info)

def get_environ_include_dirs():  #obsolete feature
    includes = []
    if os.environ.has_key('PYTHONINCLUDE'):
        includes = os.environ['PYTHONINCLUDE'].split(os.pathsep)
    return includes

def get_build_temp():
    from distutils.util import get_platform
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    return os.path.join('build','temp'+plat_specifier)

def get_build_platlib():
    from distutils.util import get_platform
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    return os.path.join('build','lib'+plat_specifier)

class SourceGenerator:  #obsolete feature
    """ SourceGenerator
    func    - creates target, arguments are (target,sources)+args
    sources - target source files
    args    - extra arguments to func

    If func is None then target must exist and it is touched whenever
    sources are newer.
    """
    def __init__(self,func,target,sources=[],*args):
        if not os.path.isabs(target) and func is not None:
            g = sys._getframe(1).f_globals
            fn = g.get('__file__',g.get('__name__'))
            if fn=='__main__': fn = sys.argv[0]
            caller_dir = os.path.abspath(os.path.dirname(fn))
            prefix = os.path.commonprefix([caller_dir,os.getcwd()])
            target_dir = caller_dir[len(prefix)+1:]
            target = os.path.join(get_build_temp(),target_dir,target)
        self.func = func
        self.target = target
        self.sources = sources
        self.args = args
    def __str__(self):
        return str(self.target)
    def generate(self):
        from distutils import dep_util,dir_util
        if dep_util.newer_group(self.sources,self.target):
            print 'Running generate',self.target
            dir_util.mkpath(os.path.dirname(self.target),verbose=1)
            if self.func is None:
                # Touch target
                os.utime(self.target,None)
            else:
                self.func(self.target,self.sources,*self.args)
        assert os.path.exists(self.target),`self.target`
        return self.target
    def __call__(self, extension, src_dir):
        return self.generate()

class SourceFilter:  #obsolete feature
    """ SourceFilter
    func    - implements criteria to filter sources
    sources - source files
    args    - extra arguments to func
    """
    def __init__(self,func,sources,*args):
        self.func = func
        self.sources = sources
        self.args = args
    def filter(self):
        return self.func(self.sources,*self.args)
    def __call__(self, extension, src_dir):
        return self.filter()

##

#XXX need support for .C that is also C++
cxx_ext_match = re.compile(r'.*[.](cpp|cxx|cc)\Z',re.I).match
fortran_ext_match = re.compile(r'.*[.](f90|f95|f77|for|ftn|f)\Z',re.I).match
f90_ext_match = re.compile(r'.*[.](f90|f95)\Z',re.I).match
f90_module_name_match = re.compile(r'\s*module\s*(?P<name>[\w_]+)',re.I).match
def get_f90_modules(source):
    """ Return a list of Fortran f90 module names that
    given source file defines.
    """
    if not f90_ext_match(source):
        return []
    modules = []
    f = open(source,'r')
    f_readlines = getattr(f,'xreadlines',f.readlines)
    for line in f_readlines():
        m = f90_module_name_match(line)
        if m:
            name = m.group('name')
            modules.append(name)
            # break  # XXX can we assume that there is one module per file?
    f.close()
    return modules

def all_strings(lst):
    """ Return True if all items in lst are string objects. """
    for item in lst:
        if type(item) is not types.StringType:
            return 0
    return 1

def has_f_sources(sources):
    """ Return True if sources contains Fortran files """
    for source in sources:
        if fortran_ext_match(source):
            return 1
    return 0

def has_cxx_sources(sources):
    """ Return True if sources contains C++ files """
    for source in sources:
        if cxx_ext_match(source):
            return 1
    return 0

def filter_sources(sources):
    """ Return four lists of filenames containing
    C, C++, Fortran, and Fortran 90 module sources,
    respectively.
    """
    c_sources = []
    cxx_sources = []
    f_sources = []
    fmodule_sources = []
    for source in sources:
        if fortran_ext_match(source):
            modules = get_f90_modules(source)
            if modules:
                fmodule_sources.append(source)
            else:
                f_sources.append(source)
        elif cxx_ext_match(source):
            cxx_sources.append(source)
        else:
            c_sources.append(source)            
    return c_sources, cxx_sources, f_sources, fmodule_sources

def compiler_to_string(compiler):
    props = []
    mx = 0
    keys = compiler.executables.keys()
    for key in ['version','libraries','library_dirs',
                'object_switch','compile_switch',
                'include_dirs','define','undef','rpath','link_objects']:
        if key not in keys:
            keys.append(key)
    for key in keys:
        if hasattr(compiler,key):
            v = getattr(compiler, key)
            mx = max(mx,len(key))
            props.append((key,`v`))
    lines = []
    format = '%-' +`mx+1`+ 's = %s'
    for prop in props:
        lines.append(format % prop)
    return '\n'.join(lines)

def _get_dirs_with_init((packages,path), dirname, names):
    """Internal: used by get_subpackages."""
    for bad in ['.svn','build']:
        if bad in names:
            del names[names.index(bad)]
    if os.path.isfile(os.path.join(dirname,'__init__.py')):
        if path==dirname: return
        package_name = '.'.join(dirname.split(os.sep)[len(path.split(os.sep)):])
        if package_name not in packages:
            packages.append(package_name)

def get_subpackages(path,
                    parent=None,
                    parent_path=None,
                    include_packages=[],
                    ignore_packages=[],
                    include_only=None,
                    recursive=None):

    """
    Return a list of configurations found in a tree of Python
    packages.

    It is assumed that each package xxx in path/xxx has file
    path/xxx/info_xxx.py that follows convention specified in
    scipy/DEVELOPERS.txt.

    Packages that do not define info_*.py files or should override
    options in info*_.py files can be specified in include_packages
    list.

    Unless a package xxx is specified standalone, it will be installed
    as parent.xxx.

    Specifying parent_path is recommended for reducing verbosity of
    compilations.

    Packages in ignore_packages list will be ignored unless they are
    also in include_packages.

    When include_only is True then only configurations of those
    packages are returned that are in include_packages list.

    If recursive is True then subpackages are searched recursively
    starting from the path and added to include_packages list.
    """

    config_list = []

    for info_file in glob.glob(os.path.join(path,'*','info_*.py')):
        package_name = os.path.basename(os.path.dirname(info_file))
        if package_name != os.path.splitext(os.path.basename(info_file))[0][5:]:
            print '  !! Mismatch of package name %r and %s' \
                  % (package_name, info_file)
            continue

        if package_name in ignore_packages:
            continue
        if include_only and package_name not in include_packages:
            continue

        sys.path.insert(0,os.path.dirname(info_file))
        try:
            exec 'import %s as info_module' \
                 % (os.path.splitext(os.path.basename(info_file))[0])
            if not getattr(info_module,'ignore',0):
                exec 'import setup_%s as setup_module' % (package_name)
                if getattr(info_module,'standalone',0) or not parent:
                    args = ('',)
                else:
                    args = (parent,)
                if setup_module.configuration.func_code.co_argcount>1:
                    args = args + (parent_path,)
                config = setup_module.configuration(*args)
                config_list.append(config)
        finally:
            del sys.path[0]

    if recursive:
        os.path.walk(path,_get_dirs_with_init, (include_packages,path))

    for package_name in include_packages:
        dirname = os.path.join(*([path]+package_name.split('.')))
        name = package_name.split('.')[-1]
        setup_name = 'setup_' + name
        setup_file = os.path.join(dirname, setup_name + '.py')
        ns = package_name.split('.')[:-1]
        if parent: ns.insert(0, parent)
        parent_name = '.'.join(ns)

        if not os.path.isfile(setup_file):
            print 'Assuming default configuration (%r was not found)' \
                  % (setup_file)

            config = default_config_dict(name, parent_name,
                                         local_path=dirname)
            config_list.append(config)
            continue
    
        sys.path.insert(0,dirname)
        try:
            exec 'import %s as setup_module' % (setup_name)
            args = (parent_name,)
            if setup_module.configuration.func_code.co_argcount>1:
                args = args + (parent_path,)
            config = setup_module.configuration(*args)
            config_list.append(config)
        finally:
            del sys.path[0]
    return config_list

def generate_config_py(extension, build_dir):
    """ Generate <package>/config.py file containing system_info
    information used during building the package.

    Usage:\
        ext = Extension(dot_join(config['name'],'config'),
                        sources=[generate_config_py])
        config['ext_modules'].append(ext)
    """
    from scipy_distutils.system_info import system_info
    from distutils.dir_util import mkpath
    target = os.path.join(*([build_dir]+extension.name.split('.'))) + '.py'
    mkpath(os.path.dirname(target))
    f = open(target,'w')
    f.write('# This file is generated by %s\n' % (os.path.abspath(sys.argv[0])))
    f.write('# It contains system_info results at the time of building this package.\n')
    f.write('__all__ = ["get_info","show"]\n\n')
    for k,i in system_info.saved_results.items():
        f.write('%s=%r\n' % (k,i))
    f.write('\ndef get_info(name): g=globals(); return g.get(name,g.get(name+"_info",{}))\n')
    f.write('''
def show():
    for name,info_dict in globals().items():
        if name[0]=="_" or type(info_dict) is not type({}): continue
        print name+":"
        if not info_dict:
            print "  NOT AVAILABLE"
        for k,v in info_dict.items():
            v = str(v)
            if k==\'sources\' and len(v)>200: v = v[:60]+\' ...\\n... \'+v[-60:]
            print \'    %s = %s\'%(k,v)
        print
    return
    ''')

    f.close()
    return target

def generate_svn_version_py(extension, build_dir):
    """ Generate __svn_version__.py file containing SVN
    revision number of a module.
    
    To use, add the following codelet to setup
    configuration(..) function

      ext = Extension(dot_join(config['name'],'__svn_version__'),
                      sources=[generate_svn_version_py])
      ext.local_path = local_path
      config['ext_modules'].append(ext)

    """
    from distutils import dep_util
    local_path = extension.local_path
    target = os.path.join(build_dir, '__svn_version__.py')
    entries = os.path.join(local_path,'.svn','entries')
    if os.path.isfile(entries):
        if not dep_util.newer(entries, target):
            return target
    elif os.path.isfile(target):
        return target

    revision = get_svn_revision(local_path)
    f = open(target,'w')
    f.write('revision=%s\n' % (revision))
    f.close()
    return target

def get_svn_revision(path):
    """ Return path's SVN revision number.
    """
    entries = os.path.join(path,'.svn','entries')
    revision = None
    if os.path.isfile(entries):
        f = open(entries)
        m = re.search(r'revision="(?P<revision>\d+)"',f.read())
        f.close()
        if m:
            revision = int(m.group('revision'))
    return revision

if __name__ == '__main__':
    print 'terminal_has_colors:',terminal_has_colors()
    print red_text("This is red text")
    print yellow_text("This is yellow text")
