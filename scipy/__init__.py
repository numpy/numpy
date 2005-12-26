"""\
SciPy Core
==========

You can support the development of SciPy by purchasing documentation
at

  http://www.trelgol.com

It is being distributed for a fee for a limited time to try and raise
money for development.

Documentation is also available in the docstrings.

Available subpackages
---------------------
"""

import os, sys
NO_SCIPY_IMPORT = os.environ.get('NO_SCIPY_IMPORT',None)
SCIPY_IMPORT_VERBOSE = int(os.environ.get('SCIPY_IMPORT_VERBOSE','0'))

try:
    from __core_config__ import show as show_core_config
except ImportError:
    show_core_config = None

class PackageLoader:
    def __init__(self):
        """ Manages loading SciPy packages.
        """

        self.frame = frame = sys._getframe(1)
        self.parent_name = eval('__name__',frame.f_globals,frame.f_locals)
        self.parent_path = eval('__path__[0]',frame.f_globals,frame.f_locals)
        if not frame.f_locals.has_key('__all__'):
            exec('__all__ = []',frame.f_globals,frame.f_locals)
        self.parent_export_names = eval('__all__',frame.f_globals,frame.f_locals)

        self.info_modules = None
        self.imported_packages = []

    def _init_info_modules(self, packages=None):
        """Initialize info_modules = {<package_name>: <package info.py module>}.
        """
        import imp
        from glob import glob
        if packages is None:
            info_files = glob(os.path.join(self.parent_path,'*','info.py'))
            for info_file in glob(os.path.join(self.parent_path,'*','info.pyc')):
                if info_file[:-1] not in info_files:
                    info_files.append(info_file)
        else:
            info_files = []
            for package in packages:
                package = os.path.join(*package.split('.'))
                info_file = os.path.join(self.parent_path,package,'info.py')
                if not os.path.isfile(info_file): info_file += 'c'
                if os.path.isfile(info_file):
                    info_files.append(info_file)
                else:
                    if self.verbose:
                        print >> sys.stderr, 'Package',`package`,\
                              'does not have info.py file. Ignoring.'

        info_modules = self.info_modules
        for info_file in info_files:
            package_name = os.path.basename(os.path.dirname(info_file))
            if info_modules.has_key(package_name):
                continue
            fullname = self.parent_name +'.'+ package_name

            if info_file[-1]=='c':
                filedescriptor = ('.pyc','rb',2)
            else:
                filedescriptor = ('.py','U',1)

            try:
                info_module = imp.load_module(fullname+'.info',
                                              open(info_file,filedescriptor[1]),
                                              info_file,
                                              filedescriptor)
            except Exception,msg:
                print >> sys.stderr, msg
                info_module = None

            if info_module is None or getattr(info_module,'ignore',False):
                info_modules.pop(package_name,None)
            else:
                self._init_info_modules(getattr(info_module,'depends',[]))
                info_modules[package_name] = info_module

        return

    def _get_sorted_names(self):
        """ Return package names sorted in the order as they should be
        imported due to dependence relations between packages. 
        """

        depend_dict = {}
        for name,info_module in self.info_modules.items():
            depend_dict[name] = getattr(info_module,'depends',[])
        package_names = []

        for name in depend_dict.keys():
            if not depend_dict[name]:
                package_names.append(name)
                del depend_dict[name]

        while depend_dict:
            for name, lst in depend_dict.items():
                new_lst = [n for n in lst if depend_dict.has_key(n)]
                if not new_lst:
                    package_names.append(name)
                    del depend_dict[name]
                else:
                    depend_dict[name] = new_lst

        return package_names

    def __call__(self,*packages, **options):
        """Load one or more packages into scipy's top-level namespace.

    Usage:

       This function is intended to shorten the need to import many of scipy's
       submodules constantly with statements such as

       import scipy.linalg, scipy.fft, scipy.etc...

       Instead, you can say:

         import scipy
         scipy.pkgload('linalg','fft',...)

       or

         scipy.pkgload()

       to load all of them in one call.

       If a name which doesn't exist in scipy's namespace is
       given, an exception [[WHAT? ImportError, probably?]] is raised.
       [NotImplemented]

     Inputs:

       - the names (one or more strings) of all the scipy modules one wishes to
       load into the top-level namespace.

     Optional keyword inputs:

       - verbose - integer specifying verbosity level [default: 0].
       - force   - when True, force reloading loaded packages [default: False].

     If no input arguments are given, then all of scipy's subpackages are
     imported.


     Outputs:

       The function returns a tuple with all the names of the modules which
       were actually imported. [NotImplemented]

     """
        frame = self.frame
        self.info_modules = {}
        if options.get('force',False):
            self.imported_packages = []
        self.verbose = verbose = options.get('verbose',False)

        self._init_info_modules(packages or None)

        for package_name in self._get_sorted_names():
            if package_name in self.imported_packages:
                continue
            fullname = self.parent_name +'.'+ package_name
            info_module = self.info_modules[package_name]
            if verbose>1:
                print >> sys.stderr, 'Importing',package_name,'to',self.parent_name

            old_object = frame.f_locals.get(package_name,None)

            try:
                exec ('import '+package_name, frame.f_globals,frame.f_locals)
            except Exception,msg:
                print >> sys.stderr, 'Failed to import',package_name
                print >> sys.stderr, msg
                continue

            if verbose:
                new_object = frame.f_locals.get(package_name)
                if old_object is not None and old_object is not new_object:
                    print >> sys.stderr, 'Overwriting',package_name,'=',\
                          `old_object`,'with',`new_object`            

            self.imported_packages.append(package_name)
            self.parent_export_names.append(package_name)

            global_symbols = getattr(info_module,'global_symbols',[])
            for symbol in global_symbols:
                if verbose:
                    print >> sys.stderr, 'Importing',symbol,'of',package_name,\
                          'to',self.parent_name

                if symbol=='*':
                    symbols = eval('getattr(%s,"__all__",None)'\
                                   % (package_name),
                                   frame.f_globals,frame.f_locals)
                    if symbols is None:
                        symbols = eval('dir(%s)' % (package_name),
                                       frame.f_globals,frame.f_locals)
                        symbols = filter(lambda s:not s.startswith('_'),symbols)
                else:
                    symbols = [symbol]

                if verbose:
                    old_objects = {}
                    for s in symbols:
                        if frame.f_locals.has_key(s):
                            old_objects[s] = frame.f_locals[s]
                try:
                    exec ('from '+package_name+' import '+symbol,
                          frame.f_globals,frame.f_locals)
                except Exception,msg:
                    print >> sys.stderr, 'Failed to import',symbol,'from',package_name
                    print >> sys.stderr, msg
                    continue

                if verbose:
                    for s,old_object in old_objects.items():
                        new_object = frame.f_locals[s]
                        if new_object is not old_object:
                            print >> sys.stderr, 'Overwriting',s,'=',\
                                  `old_object`,'with',`new_object`            

                if symbol=='*':
                    self.parent_export_names.extend(symbols)
                else:
                    self.parent_export_names.append(symbol)

        return

pkgload = PackageLoader()

if show_core_config is None:
    print >> sys.stderr, 'Running from scipy core source directory.'
else:
    from core_version import version as __core_version__

    pkgload('test','base','corefft','corelinalg','random',verbose=SCIPY_IMPORT_VERBOSE)

    test = ScipyTest('scipy').test

__scipy_doc__ = """

SciPy: A scientific computing package for Python
================================================

Available subpackages
---------------------
"""

if NO_SCIPY_IMPORT is not None:
    print >> sys.stderr, 'Skip importing scipy packages (NO_SCIPY_IMPORT=%s)' % (NO_SCIPY_IMPORT)
    show_scipy_config = None
elif show_core_config is None:
    show_scipy_config = None
else:
    try:
        from __scipy_config__ import show as show_scipy_config
    except ImportError:
        show_scipy_config = None


if show_scipy_config is not None:
    from scipy_version import scipy_version as __scipy_version__
    __doc__ += __scipy_doc__
    pkgload(verbose=SCIPY_IMPORT_VERBOSE)
