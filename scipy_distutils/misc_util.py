import os,sys,string

def update_version(release_level='alpha',
                   path='.',
                   version_template = \
                   '%(major)d.%(minor)d.%(micro)d-%(release_level)s-%(serial)d',
                   major=None,
                   overwrite_version_py = 1):
    """
    Return version string calculated from CVS/Entries file(s) starting
    at <path>. If the version information is different from the one
    found in the <path>/__version__.py file, update_version updates
    the file automatically. The version information will be always
    increasing in time.
    If CVS tree does not exist (e.g. as in distribution packages),
    return the version string found from  <path>/__version__.py.
    If no version information is available, return None.

    Default version string is in the form

      <major>.<minor>.<micro>-<release_level>-<serial>

    The items have the following meanings:

      serial - shows cumulative changes in all files in the CVS
               repository
      micro  - a number that is equivalent to the number of files
      minor  - indicates the changes in micro value (files are added
               or removed)
      release_level - is alpha, beta, canditate, or final
      major  - indicates changes in release_level.

    """
    # Issues:
    # *** Recommend or not to add __version__.py file to CVS
    #     repository? If it is in CVS, then when commiting, the
    #     version information will change, but __version__.py
    #     is commited with the old version information. To get
    #     __version__.py also up to date, a second commit of the
    #     __version__.py file is required after you re-run
    #     update_version(..). To summarize:
    #     1) cvs commit ...
    #     2) python setup.py  # that should call update_version
    #     3) cvs commit -m "updating version" __version__.py

    release_level_map = {'alpha':0,
                         'beta':1,
                         'canditate':2,
                         'final':3}
    release_level_value = release_level_map.get(release_level)
    if release_level_value is None:
        print 'Warning: release_level=%s is not %s'\
              % (release_level,
                 string.join(release_level_map.keys(),','))

    cwd = os.getcwd()
    os.chdir(path)
    try:
        version_module = __import__('__version__')
        reload(version_module)
        old_version_info = version_module.version_info
        old_version = version_module.version
    except:
        print sys.exc_value
        old_version_info = None
        old_version = None
    os.chdir(cwd)

    cvs_revs = get_cvs_revision(path)
    if cvs_revs is None:
        return old_version

    minor = 1
    micro,serial = cvs_revs
    if old_version_info is not None:
        minor = old_version_info[1]
        old_release_level_value = release_level_map.get(old_version_info[3])
        if micro != old_version_info[2]: # files have beed added or removed
            minor = minor + 1
        if major is None:
            major = old_version_info[0]
            if old_release_level_value is not None:
                if old_release_level_value > release_level_value:
                    major = major + 1
    if major is None:
        major = 0

    version_info = (major,minor,micro,release_level,serial)
    version_dict = {'major':major,'minor':minor,'micro':micro,
                    'release_level':release_level,'serial':serial
                    }
    version = version_template % version_dict

    if version_info != old_version_info:
        print 'version increase detected: %s -> %s'%(old_version,version)
        version_file = os.path.join(path,'__version__.py')
        if not overwrite_version_py:
            print 'keeping %s with old version, returing new version' \
                  % (version_file)
            return version
        print 'updating version in %s' % version_file
        version_file = os.path.abspath(version_file)
        f = open(version_file,'w')
        f.write('# This file is automatically updated with update_version\n'\
                '# function from scipy_distutils.misc_util.py\n'\
                'version = %s\n'\
                'version_info = %s\n'%(repr(version),version_info))
        f.close()
    return version

def get_version(release_level='alpha',
                path='.',
                version_template = \
                '%(major)d.%(minor)d.%(micro)d-%(release_level)s-%(serial)d',
                major=None,
                ):
    """
    Return version string calculated from CVS/Entries file(s) starting
    at <path>. Does not change <path>/__version__.py.
    See also update_version(..) function.
    """
    return update_version(release_level = release_level,path = path,
                          version_template = version_template,
                          major = major,overwrite_version_py = 0)


def get_cvs_revision(path):
    """
    Return two last cumulative revision numbers of a CVS tree starting
    at <path>. The first number shows the number of files in the CVS
    tree (this is often true, but not always) and the second number
    characterizes the changes in these files.
    If <path>/CVS/Entries is not existing then return None.
    """
    entries_file = os.path.join(path,'CVS','Entries')
    if os.path.exists(entries_file):
        rev1,rev2 = 0,0
        for line in open(entries_file).readlines():
            items = string.split(line,'/')
            if items[0]=='D' and len(items)>1:
                try:
                    d1,d2 = get_cvs_revision(os.path.join(path,items[1]))
                except:
                    d1,d2 = 0,0
            elif items[0]=='' and len(items)>3 and items[1]!='__version__.py':
		last_numbers = map(eval,string.split(items[2],'.')[-2:])
		if len(last_numbers)==2:
		    d1,d2 = last_numbers
		else: # this is when 'cvs add' but not yet 'cvs commit'
		    d1,d2 = 0,0
            else:
                continue
            rev1,rev2 = rev1+d1,rev2+d2
        return rev1,rev2

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
        #import scipy_distutils.setup
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
        has a configuration() file in it. 
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
    d={}
    for key in list_keys: d[key] = []
    for key in dict_keys: d[key] = {}

    full_name = dot_join(parent_name,name)

    if full_name:
        # XXX: The following assumes that default_config_dict is called
        #      only from setup_<name>.configuration().
        frame = sys._getframe(1)
        caller_name = eval('__name__',frame.f_globals,frame.f_locals)
        local_path = get_path(caller_name)
        if name and parent_name is None:
            # Useful for local builds
            d['version'] = get_version(path=local_path)

        if os.path.exists(os.path.join(local_path,'__init__.py')):
            d['packages'].append(full_name)
            d['package_dir'][full_name] = local_path
        d['name'] = full_name
        if not parent_name:
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

def merge_config_dicts(config_list):
    result = default_config_dict()
    for d in config_list:
        for key in list_keys:
            result[key].extend(d.get(key,[]))
        for key in dict_keys:
            result[key].update(d.get(key,{}))
    return result

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
