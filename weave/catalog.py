""" Track relationships between compiled extension functions & code fragments

    catalog keeps track of which compiled(or even standard) functions are 
    related to which code fragments.  It also stores these relationships
    to disk so they are remembered between Python sessions.  When 
    
        a = 1
        compiler.inline('printf("printed from C: %d",a);',['a'] )
     
    is called, inline() first looks to see if it has seen the code 
    'printf("printed from C");' before.  If not, it calls 
    
        catalog.get_functions('printf("printed from C: %d", a);')
    
    which returns a list of all the function objects that have been compiled
    for the code fragment.  Multiple functions can occur because the code
    could be compiled for different types for 'a' (although not likely in
    this case). The catalog first looks in its cache and quickly returns
    a list of the functions if possible.  If the cache lookup fails, it then
    looks through possibly multiple catalog files on disk and fills its
    cache with all the functions that match the code fragment.  
    
    In case where the code fragment hasn't been compiled, inline() compiles
    the code and then adds it to the catalog:
    
        function = <code to compile function>
        catalog.add_function('printf("printed from C: %d", a);',function)
           
    add_function() adds function to the front of the cache.  function,
    along with the path information to its module, are also stored in a
    persistent catalog for future use by python sessions.    
"""       

import os,sys,string
import pickle
import tempfile

try:
    import dbhash
    import shelve
    dumb = 0
except ImportError:
    import dumb_shelve as shelve
    dumb = 1

#For testing...
#import dumb_shelve as shelve
#dumb = 1

#import shelve
#dumb = 0
    
def getmodule(object):
    """ Discover the name of the module where object was defined.
    
        This is an augmented version of inspect.getmodule that can discover 
        the parent module for extension functions.
    """
    import inspect
    value = inspect.getmodule(object)
    if value is None:
        #walk trough all modules looking for function
        for name,mod in sys.modules.items():
            # try except used because of some comparison failures
            # in wxPoint code.  Need to review this
            try:
                if mod and object in mod.__dict__.values():
                    value = mod
                    # if it is a built-in module, keep looking to see
                    # if a non-builtin also has it.  Otherwise quit and
                    # consider the module found. (ain't perfect, but will 
                    # have to do for now).
                    if string.find('(built-in)',str(mod)) is -1:
                        break
                    
            except (TypeError, KeyError, ImportError):
                pass        
    return value

def expr_to_filename(expr):
    """ Convert an arbitrary expr string to a valid file name.
    
        The name is based on the md5 check sum for the string and
        Something that was a little more human readable would be 
        nice, but the computer doesn't seem to care.
    """
    import md5
    base = 'sc_'
    return base + md5.new(expr).hexdigest()

def unique_file(d,expr):
    """ Generate a unqiue file name based on expr in directory d
    
        This is meant for use with building extension modules, so
        a file name is considered unique if none of the following
        extension '.cpp','.o','.so','module.so','.py', or '.pyd'
        exists in directory d.  The fully qualified path to the
        new name is returned.  You'll need to append your own
        extension to it before creating files.
    """
    files = os.listdir(d)
    #base = 'scipy_compile'
    base = expr_to_filename(expr)
    for i in range(1000000):
        fname = base + `i`
        if not (fname+'.cpp' in files or
                fname+'.o' in files or
                fname+'.so' in files or
                fname+'module.so' in files or
                fname+'.py' in files or
                fname+'.pyd' in files):
            break
    return os.path.join(d,fname)

def create_dir(p):
    """ Create a directory and any necessary intermediate directories."""
    if not os.path.exists(p):
        try:
            os.mkdir(p)
        except OSError:
            # perhaps one or more intermediate path components don't exist
            # try to create them
            base,dir = os.path.split(p)
            create_dir(base)
            # don't enclose this one in try/except - we want the user to
            # get failure info
            os.mkdir(p)

def is_writable(dir):
    dummy = os.path.join(dir, "dummy")
    try:
        open(dummy, 'w')
    except IOError:
        return 0
    os.unlink(dummy)
    return 1

def whoami():
    """return a string identifying the user."""
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"

def default_dir():
    """ Return a default location to store compiled files and catalogs.
        
        XX is the Python version number in all paths listed below
        On windows, the default location is the temporary directory
        returned by gettempdir()/pythonXX.
        
        On Unix, ~/.pythonXX_compiled is the default location.  If it doesn't
        exist, it is created.  The directory is marked rwx------.
        
        If for some reason it isn't possible to build a default directory
        in the user's home, /tmp/<uid>_pythonXX_compiled is used.  If it 
        doesn't exist, it is created.  The directory is marked rwx------
        to try and keep people from being able to sneak a bad module
        in on you.        
    """
    python_name = "python%d%d_compiled" % tuple(sys.version_info[:2])    
    if sys.platform != 'win32':
        try:
            path = os.path.join(os.environ['HOME'],'.' + python_name)
        except KeyError:
            temp_dir = `os.getuid()` + '_' + python_name
            path = os.path.join(tempfile.gettempdir(),temp_dir)        
        
        # add a subdirectory for the OS.
        # It might be better to do this at a different location so that
        # it wasn't only the default directory that gets this behavior.    
        #path = os.path.join(path,sys.platform)
    else:
        path = os.path.join(tempfile.gettempdir(),"%s"%whoami(),python_name)
        
    if not os.path.exists(path):
        create_dir(path)
        os.chmod(path,0700) # make it only accessible by this user.
    if not is_writable(path):
        print 'warning: default directory is not write accessible.'
        print 'default:', path
    return path

def intermediate_dir():
    """ Location in temp dir for storing .cpp and .o  files during
        builds.
    """
    python_name = "python%d%d_intermediate" % tuple(sys.version_info[:2])    
    path = os.path.join(tempfile.gettempdir(),"%s"%whoami(),python_name)
    if not os.path.exists(path):
        create_dir(path)
    return path
    
def default_temp_dir():
    path = os.path.join(default_dir(),'temp')
    if not os.path.exists(path):
        create_dir(path)
        os.chmod(path,0700) # make it only accessible by this user.
    if not is_writable(path):
        print 'warning: default directory is not write accessible.'
        print 'default:', path
    return path

    
def os_dependent_catalog_name():
    """ Generate catalog name dependent on OS and Python version being used.
    
        This allows multiple platforms to have catalog files in the
        same directory without stepping on each other.  For now, it 
        bases the name of the value returned by sys.platform and the
        version of python being run.  If this isn't enough to descriminate
        on some platforms, we can try to add other info.  It has 
        occured to me that if we get fancy enough to optimize for different
        architectures, then chip type might be added to the catalog name also.
    """
    version = '%d%d' % sys.version_info[:2]
    return sys.platform+version+'compiled_catalog'
    
def catalog_path(module_path):
    """ Return the full path name for the catalog file in the given directory.
    
        module_path can either be a file name or a path name.  If it is a 
        file name, the catalog file name in its parent directory is returned.
        If it is a directory, the catalog file in that directory is returned.

        If module_path doesn't exist, None is returned.  Note though, that the
        catalog file does *not* have to exist, only its parent.  '~', shell
        variables, and relative ('.' and '..') paths are all acceptable.
        
        catalog file names are os dependent (based on sys.platform), so this 
        should support multiple platforms sharing the same disk space 
        (NFS mounts). See os_dependent_catalog_name() for more info.
    """
    module_path = os.path.expanduser(module_path)
    module_path = os.path.expandvars(module_path)
    module_path = os.path.abspath(module_path)
    if not os.path.exists(module_path):
        catalog_file = None
    elif not os.path.isdir(module_path):
        module_path,dummy = os.path.split(module_path)
        catalog_file = os.path.join(module_path,os_dependent_catalog_name())
    else:    
        catalog_file = os.path.join(module_path,os_dependent_catalog_name())
    return catalog_file

def get_catalog(module_path,mode='r'):
    """ Return a function catalog (shelve object) from the path module_path

        If module_path is a directory, the function catalog returned is
        from that directory.  If module_path is an actual module_name,
        then the function catalog returned is from its parent directory.
        mode uses the standard 'c' = create, 'n' = new, 'r' = read, 
        'w' = write file open modes available for anydbm databases.
        
        Well... it should be.  Stuck with dumbdbm for now and the modes
        almost don't matter.  We do some checking for 'r' mode, but that
        is about it.
        
        See catalog_path() for more information on module_path.
    """
    if mode not in ['c','r','w','n']:
        msg = " mode must be 'c', 'n', 'r', or 'w'.  See anydbm for more info"
        raise ValueError, msg
    catalog_file = catalog_path(module_path)
    if (dumb and os.path.exists(catalog_file+'.dat')) \
           or os.path.exists(catalog_file):
        sh = shelve.open(catalog_file,mode)
    else:
        if mode=='r':
            sh = None
        else:
            sh = shelve.open(catalog_file,mode)
    return sh

class catalog:
    """ Stores information about compiled functions both in cache and on disk.
    
        catalog stores (code, list_of_function) pairs so that all the functions
        that have been compiled for code are available for calling (usually in
        inline or blitz).
        
        catalog keeps a dictionary of previously accessed code values cached 
        for quick access.  It also handles the looking up of functions compiled 
        in previously called Python sessions on disk in function catalogs. 
        catalog searches the directories in the PYTHONCOMPILED environment 
        variable in order loading functions that correspond to the given code 
        fragment.  A default directory is also searched for catalog functions. 
        On unix, the default directory is usually '~/.pythonxx_compiled' where 
        xx is the version of Python used. On windows, it is the directory 
        returned by temfile.gettempdir().  Functions closer to the front are of 
        the variable list are guaranteed to be closer to the front of the 
        function list so that they will be called first.  See 
        get_cataloged_functions() for more info on how the search order is 
        traversed.
        
        Catalog also handles storing information about compiled functions to
        a catalog.  When writing this information, the first writable catalog
        file in PYTHONCOMPILED path is used.  If a writable catalog is not
        found, it is written to the catalog in the default directory.  This
        directory should always be writable.
    """
    def __init__(self,user_path_list=None):
        """ Create a catalog for storing/searching for compiled functions. 
        
            user_path_list contains directories that should be searched 
            first for function catalogs.  They will come before the path
            entries in the PYTHONCOMPILED environment varilable.
        """
        if type(user_path_list) == type('string'):
            self.user_path_list = [user_path_list]
        elif user_path_list:
            self.user_path_list = user_path_list
        else:
            self.user_path_list = []
        self.cache = {}
        self.module_dir = None
        self.paths_added = 0
        
    def set_module_directory(self,module_dir):
        """ Set the path that will replace 'MODULE' in catalog searches.
        
            You should call clear_module_directory() when your finished
            working with it.
        """
        self.module_dir = module_dir
    def get_module_directory(self):
        """ Return the path used to replace the 'MODULE' in searches.
        """
        return self.module_dir
    def clear_module_directory(self):
        """ Reset 'MODULE' path to None so that it is ignored in searches.        
        """
        self.module_dir = None
        
    def get_environ_path(self):
        """ Return list of paths from 'PYTHONCOMPILED' environment variable.
        
            On Unix the path in PYTHONCOMPILED is a ':' separated list of
            directories.  On Windows, a ';' separated list is used. 
        """
        paths = []
        if os.environ.has_key('PYTHONCOMPILED'):
            path_string = os.environ['PYTHONCOMPILED'] 
            if sys.platform == 'win32':
                #probably should also look in registry
                paths = path_string.split(';')
            else:    
                paths = path_string.split(':')
        return paths    

    def build_search_order(self):
        """ Returns a list of paths that are searched for catalogs.  
        
            Values specified in the catalog constructor are searched first,
            then values found in the PYTHONCOMPILED environment variable.
            The directory returned by default_dir() is always returned at
            the end of the list.
            
            There is a 'magic' path name called 'MODULE' that is replaced
            by the directory defined by set_module_directory().  If the
            module directory hasn't been set, 'MODULE' is ignored.
        """
        
        paths = self.user_path_list + self.get_environ_path()
        search_order = []
        for path in paths:
            if path == 'MODULE':
                if self.module_dir:
                    search_order.append(self.module_dir)
            else:
                search_order.append(path)
        search_order.append(default_dir())
        return search_order

    def get_catalog_files(self):
        """ Returns catalog file list in correct search order.
          
            Some of the catalog files may not currently exists.
            However, all will be valid locations for a catalog
            to be created (if you have write permission).
        """
        files = map(catalog_path,self.build_search_order())
        files = filter(lambda x: x is not None,files)
        return files

    def get_existing_files(self):
        """ Returns all existing catalog file list in correct search order.
        """
        files = self.get_catalog_files()
        # open every stinking file to check if it exists.
        # This is because anydbm doesn't provide a consistent naming 
        # convention across platforms for its files 
        existing_files = []
        for file in files:
            if get_catalog(os.path.dirname(file),'r') is not None:
                existing_files.append(file)
        # This is the non-portable (and much faster) old code
        #existing_files = filter(os.path.exists,files)
        return existing_files

    def get_writable_file(self,existing_only=0):
        """ Return the name of the first writable catalog file.
        
            Its parent directory must also be writable.  This is so that
            compiled modules can be written to the same directory.
        """
        # note: both file and its parent directory must be writeable
        if existing_only:
            files = self.get_existing_files()
        else:
            files = self.get_catalog_files()
        # filter for (file exists and is writable) OR directory is writable
        def file_test(x):
            from os import access, F_OK, W_OK
            return (access(x,F_OK) and access(x,W_OK) or
                    access(os.path.dirname(x),W_OK))
        writable = filter(file_test,files)
        if writable:
            file = writable[0]
        else:
            file = None
        return file
        
    def get_writable_dir(self):
        """ Return the parent directory of first writable catalog file.
        
            The returned directory has write access.
        """
        return os.path.dirname(self.get_writable_file())
        
    def unique_module_name(self,code,module_dir=None):
        """ Return full path to unique file name that in writable location.
        
            The directory for the file is the first writable directory in 
            the catalog search path.  The unique file name is derived from
            the code fragment.  If, module_dir is specified, it is used
            to replace 'MODULE' in the search path.
        """
        if module_dir is not None:
            self.set_module_directory(module_dir)
        try:
            d = self.get_writable_dir()
        finally:
            if module_dir is not None:
                self.clear_module_directory()
        return unique_file(d,code)

    def path_key(self,code):
        """ Return key for path information for functions associated with code.
        """
        return '__path__' + code
        
    def configure_path(self,cat,code):
        """ Add the python path for the given code to the sys.path
        
            unconfigure_path() should be called as soon as possible after
            imports associated with code are finished so that sys.path 
            is restored to normal.
        """
        try:
            paths = cat[self.path_key(code)]
            self.paths_added = len(paths)
            sys.path = paths + sys.path
        except:
            self.paths_added = 0            
                    
    def unconfigure_path(self):
        """ Restores sys.path to normal after calls to configure_path()
        
            Remove the previously added paths from sys.path
        """
        sys.path = sys.path[self.paths_added:]
        self.paths_added = 0

    def get_cataloged_functions(self,code):
        """ Load all functions associated with code from catalog search path.
        
            Sometimes there can be trouble loading a function listed in a
            catalog file because the actual module that holds the function 
            has been moved or deleted.  When this happens, that catalog file
            is "repaired", meaning the entire entry for this function is 
            removed from the file.  This only affects the catalog file that
            has problems -- not the others in the search path.
            
            The "repair" behavior may not be needed, but I'll keep it for now.
        """
        mode = 'r'
        cat = None
        function_list = []
        for path in self.build_search_order():
            cat = get_catalog(path,mode)
            if cat is not None and cat.has_key(code):
                # set up the python path so that modules for this
                # function can be loaded.
                self.configure_path(cat,code)
                try:                    
                    function_list += cat[code]
                except: #SystemError and ImportError so far seen                        
                    # problems loading a function from the catalog.  Try to
                    # repair the cause.
                    cat.close()
                    self.repair_catalog(path,code)
                self.unconfigure_path()             
        return function_list


    def repair_catalog(self,catalog_path,code):
        """ Remove entry for code from catalog_path
        
            Occasionally catalog entries could get corrupted. An example
            would be when a module that had functions in the catalog was
            deleted or moved on the disk.  The best current repair method is 
            just to trash the entire catalog entry for this piece of code.  
            This may loose function entries that are valid, but thats life.
            
            catalog_path must be writable for repair.  If it isn't, the
            function exists with a warning.            
        """
        writable_cat = None
        if not os.path.exists(catalog_path):
            return
        try:
            writable_cat = get_catalog(catalog_path,'w')
        except:
            print 'warning: unable to repair catalog entry\n %s\n in\n %s' % \
                  (code,catalog_path)
            return          
        if writable_cat.has_key(code):
            print 'repairing catalog by removing key'
            del writable_cat[code]
        
        # it is possible that the path key doesn't exist (if the function registered
        # was a built-in function), so we have to check if the path exists before
        # arbitrarily deleting it.
        path_key = self.path_key(code)       
        if writable_cat.has_key(path_key):
            del writable_cat[path_key]   
            
    def get_functions_fast(self,code):
        """ Return list of functions for code from the cache.
        
            Return an empty list if the code entry is not found.
        """
        return self.cache.get(code,[])
                
    def get_functions(self,code,module_dir=None):
        """ Return the list of functions associated with this code fragment.
        
            The cache is first searched for the function.  If an entry
            in the cache is not found, then catalog files on disk are 
            searched for the entry.  This is slooooow, but only happens
            once per code object.  All the functions found in catalog files
            on a cache miss are loaded into the cache to speed up future calls.
            The search order is as follows:
            
                1. user specified path (from catalog initialization)
                2. directories from the PYTHONCOMPILED environment variable
                3. The temporary directory on your platform.

            The path specified by module_dir will replace the 'MODULE' 
            place holder in the catalog search path. See build_search_order()
            for more info on the search path. 
        """        
        # Fast!! try cache first.
        if self.cache.has_key(code):
            return self.cache[code]
        
        # 2. Slow!! read previously compiled functions from disk.
        try:
            self.set_module_directory(module_dir)
            function_list = self.get_cataloged_functions(code)
            # put function_list in cache to save future lookups.
            if function_list:
                self.cache[code] = function_list
            # return function_list, empty or otherwise.
        finally:
            self.clear_module_directory()
        return function_list

    def add_function(self,code,function,module_dir=None):
        """ Adds a function to the catalog.
        
            The function is added to the cache as well as the first
            writable file catalog found in the search path.  If no
            code entry exists in the cache, the on disk catalogs
            are loaded into the cache and function is added to the
            beginning of the function list.
            
            The path specified by module_dir will replace the 'MODULE' 
            place holder in the catalog search path. See build_search_order()
            for more info on the search path. 
        """    

        # 1. put it in the cache.
        if self.cache.has_key(code):
            if function not in self.cache[code]:
                self.cache[code].insert(0,function)
            else:
                # if it is in the cache, then it is also
                # been persisted 
                return
        else:           
            # Load functions and put this one up front
            self.cache[code] = self.get_functions(code)          
            self.fast_cache(code,function)
        # 2. Store the function entry to disk.    
        try:
            self.set_module_directory(module_dir)
            self.add_function_persistent(code,function)
        finally:
            self.clear_module_directory()
        
    def add_function_persistent(self,code,function):
        """ Store the code->function relationship to disk.
        
            Two pieces of information are needed for loading functions
            from disk -- the function pickle (which conveniently stores
            the module name, etc.) and the path to its module's directory.
            The latter is needed so that the function can be loaded no
            matter what the user's Python path is.
        """       
        # add function to data in first writable catalog
        mode = 'c' # create if doesn't exist, otherwise, use existing
        cat_dir = self.get_writable_dir()
        cat = get_catalog(cat_dir,mode)
        if cat is None:
            cat_dir = default_dir()
            cat = get_catalog(cat_dir,mode)
        if cat is None:
            cat_dir = default_dir()                            
            cat_file = catalog_path(cat_dir)
            print 'problems with default catalog -- removing'
            import glob
            files = glob.glob(cat_file+'*')
            for f in files:
                os.remove(f)
            cat = get_catalog(cat_dir,mode)
        if cat is None:
            raise ValueError, 'Failed to access a catalog for storing functions'    
        # Prabhu was getting some corrupt catalog errors.  I'll put a try/except
        # to protect against this, but should really try and track down the issue.
        function_list = [function]
        try:
            function_list = function_list + cat.get(code,[])
        except pickle.UnpicklingError:
            pass
        cat[code] = function_list
        # now add needed path information for loading function
        module = getmodule(function)
        try:
            # built in modules don't have the __file__ extension, so this
            # will fail.  Just pass in this case since path additions aren't
            # needed for built-in modules.
            mod_path,f=os.path.split(os.path.abspath(module.__file__))
            pkey = self.path_key(code)
            cat[pkey] = [mod_path] + cat.get(pkey,[])
        except:
            pass
	cat.close()

    def fast_cache(self,code,function):
        """ Move function to the front of the cache entry for code
        
            If future calls to the function have the same type signature,
            this will speed up access significantly because the first
            function call is correct.
            
            Note:  The cache added to the inline_tools module is significantly
                   faster than always calling get_functions, so this isn't
                   as necessary as it used to be.  Still, it's probably worth
                   doing.              
        """
        try:
            if self.cache[code][0] == function:
                return
        except: # KeyError, IndexError   
            pass
        try:
            self.cache[code].remove(function)
        except ValueError:
            pass
        # put new function at the beginning of the list to search.
        self.cache[code].insert(0,function)
