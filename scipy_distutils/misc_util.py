import os,sys,string

# Hooks for colored terminal output.
# See also http://www.livinglogic.de/Python/ansistyle
def terminal_has_colors():
    if not hasattr(sys.stdout,'isatty') or not sys.stdout.isatty(): 
        return 0
    try:
        import curses
        curses.setupterm()
        return (curses.tigetnum("colors") >= 0
                and curses.tigetnum("pairs") >= 0
                and ((curses.tigetstr("setf") is not None 
                      and curses.tigetstr("setb") is not None) 
                     or (curses.tigetstr("setaf") is not None
                         and curses.tigetstr("setab") is not None)
                     or curses.tigetstr("scp") is not None))
    except: pass
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

def get_path(mod_name):
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
    return d
    
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
             'headers']
dict_keys = ['package_dir']

def default_config_dict(name = None, parent_name = None):
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
                         ):
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

def get_environ_include_dirs():
    includes = []
    if os.environ.has_key('PYTHONINCLUDE'):
        includes = os.environ['PYTHONINCLUDE'].split(os.pathsep)
    return includes

def get_build_temp():
    from distutils.util import get_platform
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    return os.path.join('build','temp'+plat_specifier)

class SourceGenerator:
    def __init__(self,func,target,sources=[]):
        if not os.path.isabs(target):
            caller_dir = os.path.dirname(sys._getframe(1).f_globals['__file__'])
            prefix = os.path.commonprefix([caller_dir,os.getcwd()])
            target_dir = caller_dir[len(prefix)+1:]
            target = os.path.join(get_build_temp(),target_dir,target)
        self.func = func
        self.target = target
        self.sources = sources
    def __str__(self):
        return str(self.target)
    def generate(self):
        from distutils import dep_util,dir_util
        if dep_util.newer_group(self.sources,self.target):
            print 'Running generate',self.target
            dir_util.mkpath(os.path.dirname(self.target),verbose=1)
            self.func(self.target,self.sources)
        assert os.path.exists(self.target),`self.target`
        return self.target
