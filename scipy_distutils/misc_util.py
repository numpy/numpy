import os,sys

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

def default_config_dict():
    d={}
    for key in list_keys: d[key] = []
    for key in dict_keys: d[key] = {}
    return d

def merge_config_dicts(config_list):
    result = default_config_dict()    
    for d in config_list:
        for key in list_keys:
            result[key].extend(d.get(key,[]))
        for key in dict_keys:
            result[key].update(d.get(key,{}))
    return result