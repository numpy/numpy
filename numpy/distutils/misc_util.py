import os
import re
import sys
import imp
import copy
import glob
import atexit
import tempfile

try:
    set
except NameError:
    from sets import Set as set

__all__ = ['Configuration', 'get_numpy_include_dirs', 'default_config_dict',
           'dict_append', 'appendpath', 'generate_config_py',
           'get_cmd', 'allpath', 'get_mathlibs',
           'terminal_has_colors', 'red_text', 'green_text', 'yellow_text',
           'blue_text', 'cyan_text', 'cyg2win32','mingw32','all_strings',
           'has_f_sources', 'has_cxx_sources', 'filter_sources',
           'get_dependencies', 'is_local_src_dir', 'get_ext_source_files',
           'get_script_files', 'get_lib_source_files', 'get_data_files',
           'dot_join', 'get_frame', 'minrelpath','njoin',
           'is_sequence', 'is_string', 'as_list', 'gpaths', 'get_language',
           'quote_args']

def quote_args(args):
    # don't used _nt_quote_args as it does not check if
    # args items already have quotes.
    args = list(args)
    for i in range(len(args)):
        a = args[i]
        if ' ' in a and a[0] not in '"\'':
            args[i] = '"%s"' % (a)
    return args

def allpath(name):
    "Convert a /-separated pathname to one using the OS's path separator."
    splitted = name.split('/')
    return os.path.join(*splitted)

def rel_path(path, parent_path):
    """Return path relative to parent_path.
    """
    pd = os.path.abspath(parent_path)
    apath = os.path.abspath(path)
    if len(apath)<len(pd):
        return path
    if apath==pd:
        return ''
    if pd == apath[:len(pd)]:
        assert apath[len(pd)] in [os.sep],`path,apath[len(pd)]`
        path = apath[len(pd)+1:]
    return path

def get_path_from_frame(frame, parent_path=None):
    """Return path of the module given a frame object from the call stack.

    Returned path is relative to parent_path when given,
    otherwise it is absolute path.
    """

    # First, try to find if the file name is in the frame.
    try:
        caller_file = eval('__file__', frame.f_globals, frame.f_locals)
        d = os.path.dirname(os.path.abspath(caller_file))
    except NameError:
        # __file__ is not defined, so let's try __name__. We try this second
        # because setuptools spoofs __name__ to be '__main__' even though
        # sys.modules['__main__'] might be something else, like easy_install(1).
        caller_name = eval('__name__', frame.f_globals, frame.f_locals)
        __import__(caller_name)
        mod = sys.modules[caller_name]
        if hasattr(mod, '__file__'):
            d = os.path.dirname(os.path.abspath(mod.__file__))
        else:
            # we're probably running setup.py as execfile("setup.py")
            # (likely we're building an egg)
            d = os.path.abspath('.')
            # hmm, should we use sys.argv[0] like in __builtin__ case?

    if parent_path is not None:
        d = rel_path(d, parent_path)

    return d or '.'

def njoin(*path):
    """Join two or more pathname components +
    - convert a /-separated pathname to one using the OS's path separator.
    - resolve `..` and `.` from path.

    Either passing n arguments as in njoin('a','b'), or a sequence
    of n names as in njoin(['a','b']) is handled, or a mixture of such arguments.
    """
    paths = []
    for p in path:
        if is_sequence(p):
            # njoin(['a', 'b'], 'c')
            paths.append(njoin(*p))
        else:
            assert is_string(p)
            paths.append(p)
    path = paths
    if not path:
        # njoin()
        joined = ''
    else:
        # njoin('a', 'b')
        joined = os.path.join(*path)
    if os.path.sep != '/':
        joined = joined.replace('/',os.path.sep)
    return minrelpath(joined)

def get_mathlibs(path=None):
    """Return the MATHLIB line from config.h
    """
    if path is None:
        path = get_numpy_include_dirs()[0]
    config_file = os.path.join(path,'config.h')
    fid = open(config_file)
    mathlibs = []
    s = '#define MATHLIB'
    for line in fid.readlines():
        if line.startswith(s):
            value = line[len(s):].strip()
            if value:
                mathlibs.extend(value.split(','))
    fid.close()
    return mathlibs

def minrelpath(path):
    """Resolve `..` and '.' from path.
    """
    if not is_string(path):
        return path
    if '.' not in path:
        return path
    l = path.split(os.sep)
    while l:
        try:
            i = l.index('.',1)
        except ValueError:
            break
        del l[i]
    j = 1
    while l:
        try:
            i = l.index('..',j)
        except ValueError:
            break
        if l[i-1]=='..':
            j += 1
        else:
            del l[i],l[i-1]
            j = 1
    if not l:
        return ''
    return os.sep.join(l)

def _fix_paths(paths,local_path,include_non_existing):
    assert is_sequence(paths), repr(type(paths))
    new_paths = []
    assert not is_string(paths),`paths`
    for n in paths:
        if is_string(n):
            if '*' in n or '?' in n:
                p = glob.glob(n)
                p2 = glob.glob(njoin(local_path,n))
                if p2:
                    new_paths.extend(p2)
                elif p:
                    new_paths.extend(p)
                else:
                    if include_non_existing:
                        new_paths.append(n)
                    print 'could not resolve pattern in %r: %r' \
                              % (local_path,n)
            else:
                n2 = njoin(local_path,n)
                if os.path.exists(n2):
                    new_paths.append(n2)
                else:
                    if os.path.exists(n):
                        new_paths.append(n)
                    elif include_non_existing:
                        new_paths.append(n)
                    if not os.path.exists(n):
                        print 'non-existing path in %r: %r' \
                              % (local_path,n)

        elif is_sequence(n):
            new_paths.extend(_fix_paths(n,local_path,include_non_existing))
        else:
            new_paths.append(n)
    return map(minrelpath,new_paths)

def gpaths(paths, local_path='', include_non_existing=True):
    """Apply glob to paths and prepend local_path if needed.
    """
    if is_string(paths):
        paths = (paths,)
    return _fix_paths(paths,local_path, include_non_existing)


_temporary_directory = None
def clean_up_temporary_directory():
    from numpy.distutils import log
    global _temporary_directory
    if not _temporary_directory:
        return
    log.debug('removing %s', _temporary_directory)
    try:
        os.rmdir(_temporary_directory)
    except OSError:
        pass
    _temporary_directory = None

def make_temp_file(suffix='', prefix='', text=True):
    global _temporary_directory
    if not _temporary_directory:
        _temporary_directory = tempfile.mkdtemp()
        atexit.register(clean_up_temporary_directory)
    fid, name = tempfile.mkstemp(suffix=suffix,
                                 prefix=prefix,
                                 dir=_temporary_directory,
                                 text=text)
    fo = os.fdopen(fid, 'w')
    return fo, name

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
    _colour_codes = dict(black=0, red=1, green=2, yellow=3,
                         blue=4, magenta=5, cyan=6, white=7)
    def colour_text(s, fg=None, bg=None, bold=False):
        seq = []
        if bold:
            seq.append('1')
        if fg:
            fgcode = 30 + _colour_codes.get(fg.lower(), 0)
            seq.append(str(fgcode))
        if bg:
            bgcode = 40 + _colour_codes.get(fg.lower(), 7)
            seq.append(str(bgcode))
        if seq:
            return '\x1b[%sm%s\x1b[0m' % (';'.join(seq), s)
        else:
            return s
else:
    def colour_text(s, fg=None, bg=None):
        return s

def red_text(s):
    return colour_text(s, 'red')
def green_text(s):
    return colour_text(s, 'green')
def yellow_text(s):
    return colour_text(s, 'yellow')
def cyan_text(s):
    return colour_text(s, 'cyan')
def blue_text(s):
    return colour_text(s, 'blue')

#########################

def cyg2win32(path):
    if sys.platform=='cygwin' and path.startswith('/cygdrive'):
        path = path[10] + ':' + os.path.normcase(path[11:])
    return path

def mingw32():
    """Return true when using mingw32 environment.
    """
    if sys.platform=='win32':
        if os.environ.get('OSTYPE','')=='msys':
            return True
        if os.environ.get('MSYSTEM','')=='MINGW32':
            return True
    return False

def msvc_runtime_library():
    "Return name of MSVC runtime library if Python was built with MSVC >= 7"
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        msc_ver = sys.version[msc_pos+6:msc_pos+10]
        lib = {'1300' : 'msvcr70',    # MSVC 7.0
               '1310' : 'msvcr71',    # MSVC 7.1
               '1400' : 'msvcr80',    # MSVC 8
              }.get(msc_ver, None)
    else:
        lib = None
    return lib

def msvc_on_amd64():
    if not (sys.platform=='win32' or os.name=='nt'):
        return
    from distutils.msvccompiler import get_build_architecture
    if get_build_architecture() != 'AMD64':
        return
    if os.environ.has_key('DISTUTILS_USE_SDK'):
        return
    # try to avoid _MSVCCompiler__root attribute error
    os.environ['DISTUTILS_USE_SDK']=1
    return

msvc_on_amd64()

#########################

#XXX need support for .C that is also C++
cxx_ext_match = re.compile(r'.*[.](cpp|cxx|cc)\Z',re.I).match
fortran_ext_match = re.compile(r'.*[.](f90|f95|f77|for|ftn|f)\Z',re.I).match
f90_ext_match = re.compile(r'.*[.](f90|f95)\Z',re.I).match
f90_module_name_match = re.compile(r'\s*module\s*(?P<name>[\w_]+)',re.I).match
def _get_f90_modules(source):
    """Return a list of Fortran f90 module names that
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

def is_string(s):
    return isinstance(s, str)

def all_strings(lst):
    """Return True if all items in lst are string objects. """
    for item in lst:
        if not is_string(item):
            return False
    return True

def is_sequence(seq):
    if is_string(seq):
        return False
    try:
        len(seq)
    except:
        return False
    return True

def is_glob_pattern(s):
    return is_string(s) and ('*' in s or '?' is s)

def as_list(seq):
    if is_sequence(seq):
        return list(seq)
    else:
        return [seq]

def get_language(sources):
    # not used in numpy/scipy packages, use build_ext.detect_language instead
    """Determine language value (c,f77,f90) from sources """
    language = None
    for source in sources:
        if isinstance(source, str):
            if f90_ext_match(source):
                language = 'f90'
                break
            elif fortran_ext_match(source):
                language = 'f77'
    return language

def has_f_sources(sources):
    """Return True if sources contains Fortran files """
    for source in sources:
        if fortran_ext_match(source):
            return True
    return False

def has_cxx_sources(sources):
    """Return True if sources contains C++ files """
    for source in sources:
        if cxx_ext_match(source):
            return True
    return False

def filter_sources(sources):
    """Return four lists of filenames containing
    C, C++, Fortran, and Fortran 90 module sources,
    respectively.
    """
    c_sources = []
    cxx_sources = []
    f_sources = []
    fmodule_sources = []
    for source in sources:
        if fortran_ext_match(source):
            modules = _get_f90_modules(source)
            if modules:
                fmodule_sources.append(source)
            else:
                f_sources.append(source)
        elif cxx_ext_match(source):
            cxx_sources.append(source)
        else:
            c_sources.append(source)
    return c_sources, cxx_sources, f_sources, fmodule_sources


def _get_headers(directory_list):
    # get *.h files from list of directories
    headers = []
    for d in directory_list:
        head = glob.glob(os.path.join(d,"*.h")) #XXX: *.hpp files??
        headers.extend(head)
    return headers

def _get_directories(list_of_sources):
    # get unique directories from list of sources.
    direcs = []
    for f in list_of_sources:
        d = os.path.split(f)
        if d[0] != '' and not d[0] in direcs:
            direcs.append(d[0])
    return direcs

def get_dependencies(sources):
    #XXX scan sources for include statements
    return _get_headers(_get_directories(sources))

def is_local_src_dir(directory):
    """Return true if directory is local directory.
    """
    if not is_string(directory):
        return False
    abs_dir = os.path.abspath(directory)
    c = os.path.commonprefix([os.getcwd(),abs_dir])
    new_dir = abs_dir[len(c):].split(os.sep)
    if new_dir and not new_dir[0]:
        new_dir = new_dir[1:]
    if new_dir and new_dir[0]=='build':
        return False
    new_dir = os.sep.join(new_dir)
    return os.path.isdir(new_dir)

def general_source_files(top_path):
    pruned_directories = {'CVS':1, '.svn':1, 'build':1}
    prune_file_pat = re.compile(r'(?:[~#]|\.py[co]|\.o)$')
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [ d for d in dirnames if d not in pruned_directories ]
        dirnames[:] = pruned
        for f in filenames:
            if not prune_file_pat.search(f):
                yield os.path.join(dirpath, f)

def general_source_directories_files(top_path):
    """Return a directory name relative to top_path and
    files contained.
    """
    pruned_directories = ['CVS','.svn','build']
    prune_file_pat = re.compile(r'(?:[~#]|\.py[co]|\.o)$')
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [ d for d in dirnames if d not in pruned_directories ]
        dirnames[:] = pruned
        for d in dirnames:
            dpath = os.path.join(dirpath, d)
            rpath = rel_path(dpath, top_path)
            files = []
            for f in os.listdir(dpath):
                fn = os.path.join(dpath,f)
                if os.path.isfile(fn) and not prune_file_pat.search(fn):
                    files.append(fn)
            yield rpath, files
    dpath = top_path
    rpath = rel_path(dpath, top_path)
    filenames = [os.path.join(dpath,f) for f in os.listdir(dpath) \
                 if not prune_file_pat.search(f)]
    files = [f for f in filenames if os.path.isfile(f)]
    yield rpath, files


def get_ext_source_files(ext):
    # Get sources and any include files in the same directory.
    filenames = []
    sources = filter(is_string, ext.sources)
    filenames.extend(sources)
    filenames.extend(get_dependencies(sources))
    for d in ext.depends:
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames

def get_script_files(scripts):
    scripts = filter(is_string, scripts)
    return scripts

def get_lib_source_files(lib):
    filenames = []
    sources = lib[1].get('sources',[])
    sources = filter(is_string, sources)
    filenames.extend(sources)
    filenames.extend(get_dependencies(sources))
    depends = lib[1].get('depends',[])
    for d in depends:
        if is_local_src_dir(d):
            filenames.extend(list(general_source_files(d)))
        elif os.path.isfile(d):
            filenames.append(d)
    return filenames

def get_data_files(data):
    if is_string(data):
        return [data]
    sources = data[1]
    filenames = []
    for s in sources:
        if callable(s):
            continue
        if is_local_src_dir(s):
            filenames.extend(list(general_source_files(s)))
        elif is_string(s):
            if os.path.isfile(s):
                filenames.append(s)
            else:
                print 'Not existing data file:',s
        else:
            raise TypeError,repr(s)
    return filenames

def dot_join(*args):
    return '.'.join([a for a in args if a])

def get_frame(level=0):
    """Return frame object from call stack with given level.
    """
    try:
        return sys._getframe(level+1)
    except AttributeError:
        frame = sys.exc_info()[2].tb_frame
        for _ in range(level+1):
            frame = frame.f_back
        return frame

######################

class Configuration(object):

    _list_keys = ['packages', 'ext_modules', 'data_files', 'include_dirs',
                  'libraries', 'headers', 'scripts', 'py_modules']
    _dict_keys = ['package_dir']
    _extra_keys = ['name', 'version']

    numpy_include_dirs = []

    def __init__(self,
                 package_name=None,
                 parent_name=None,
                 top_path=None,
                 package_path=None,
                 caller_level=1,
                 **attrs):
        """Construct configuration instance of a package.

        package_name -- name of the package
                        Ex.: 'distutils'
        parent_name  -- name of the parent package
                        Ex.: 'numpy'
        top_path     -- directory of the toplevel package
                        Ex.: the directory where the numpy package source sits
        package_path -- directory of package. Will be computed by magic from the
                        directory of the caller module if not specified
                        Ex.: the directory where numpy.distutils is
        caller_level -- frame level to caller namespace, internal parameter.
        """
        self.name = dot_join(parent_name, package_name)
        self.version = None

        caller_frame = get_frame(caller_level)
        self.local_path = get_path_from_frame(caller_frame, top_path)
        # local_path -- directory of a file (usually setup.py) that
        #               defines a configuration() function.
        # local_path -- directory of a file (usually setup.py) that
        #               defines a configuration() function.
        if top_path is None:
            top_path = self.local_path
            self.local_path = ''
        if package_path is None:
            package_path = self.local_path
        elif os.path.isdir(njoin(self.local_path,package_path)):
            package_path = njoin(self.local_path,package_path)
        if not os.path.isdir(package_path or '.'):
            raise ValueError("%r is not a directory" % (package_path,))
        self.top_path = top_path
        self.package_path = package_path
        # this is the relative path in the installed package
        self.path_in_package = os.path.join(*self.name.split('.'))

        self.list_keys = self._list_keys[:]
        self.dict_keys = self._dict_keys[:]

        for n in self.list_keys:
            v = copy.copy(attrs.get(n, []))
            setattr(self, n, as_list(v))

        for n in self.dict_keys:
            v = copy.copy(attrs.get(n, {}))
            setattr(self, n, v)

        known_keys = self.list_keys + self.dict_keys
        self.extra_keys = self._extra_keys[:]
        for n in attrs.keys():
            if n in known_keys:
                continue
            a = attrs[n]
            setattr(self,n,a)
            if isinstance(a, list):
                self.list_keys.append(n)
            elif isinstance(a, dict):
                self.dict_keys.append(n)
            else:
                self.extra_keys.append(n)

        if os.path.exists(njoin(package_path,'__init__.py')):
            self.packages.append(self.name)
            self.package_dir[self.name] = package_path

        self.options = dict(
            ignore_setup_xxx_py = False,
            assume_default_configuration = False,
            delegate_options_to_subpackages = False,
            quiet = False,
            )

        caller_instance = None
        for i in range(1,3):
            try:
                f = get_frame(i)
            except ValueError:
                break
            try:
                caller_instance = eval('self',f.f_globals,f.f_locals)
                break
            except NameError:
                pass
        if isinstance(caller_instance, self.__class__):
            if caller_instance.options['delegate_options_to_subpackages']:
                self.set_options(**caller_instance.options)

    def todict(self):
        """Return configuration distionary suitable for passing
        to distutils.core.setup() function.
        """
        self._optimize_data_files()
        d = {}
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for n in known_keys:
            a = getattr(self,n)
            if a:
                d[n] = a
        return d

    def info(self, message):
        if not self.options['quiet']:
            print message

    def warn(self, message):
        print>>sys.stderr, blue_text('Warning: %s' % (message,))

    def set_options(self, **options):
        """Configure Configuration instance.

        The following options are available:
        - ignore_setup_xxx_py
        - assume_default_configuration
        - delegate_options_to_subpackages
        - quiet
        """
        for key, value in options.items():
            if self.options.has_key(key):
                self.options[key] = value
            else:
                raise ValueError,'Unknown option: '+key

    def get_distribution(self):
        from numpy.distutils.core import get_distribution
        return get_distribution()

    def _wildcard_get_subpackage(self, subpackage_name,
                                 parent_name,
                                 caller_level = 1):
        l = subpackage_name.split('.')
        subpackage_path = njoin([self.local_path]+l)
        dirs = filter(os.path.isdir,glob.glob(subpackage_path))
        config_list = []
        for d in dirs:
            if not os.path.isfile(njoin(d,'__init__.py')):
                continue
            if 'build' in d.split(os.sep):
                continue
            n = '.'.join(d.split(os.sep)[-len(l):])
            c = self.get_subpackage(n,
                                    parent_name = parent_name,
                                    caller_level = caller_level+1)
            config_list.extend(c)
        return config_list

    def _get_configuration_from_setup_py(self, setup_py,
                                         subpackage_name,
                                         subpackage_path,
                                         parent_name,
                                         caller_level = 1):
        # In case setup_py imports local modules:
        sys.path.insert(0,os.path.dirname(setup_py))
        try:
            fo_setup_py = open(setup_py, 'U')
            setup_name = os.path.splitext(os.path.basename(setup_py))[0]
            n = dot_join(self.name,subpackage_name,setup_name)
            setup_module = imp.load_module('_'.join(n.split('.')),
                                           fo_setup_py,
                                           setup_py,
                                           ('.py', 'U', 1))
            fo_setup_py.close()
            if not hasattr(setup_module,'configuration'):
                if not self.options['assume_default_configuration']:
                    self.warn('Assuming default configuration '\
                              '(%s does not define configuration())'\
                              % (setup_module))
                config = Configuration(subpackage_name, parent_name,
                                       self.top_path, subpackage_path,
                                       caller_level = caller_level + 1)
            else:
                pn = dot_join(*([parent_name] + subpackage_name.split('.')[:-1]))
                args = (pn,)
                if setup_module.configuration.func_code.co_argcount > 1:
                    args = args + (self.top_path,)
                config = setup_module.configuration(*args)
            if config.name!=dot_join(parent_name,subpackage_name):
                self.warn('Subpackage %r configuration returned as %r' % \
                          (dot_join(parent_name,subpackage_name), config.name))
        finally:
            del sys.path[0]
        return config

    def get_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       parent_name=None,
                       caller_level = 1):
        """Return list of subpackage configurations.

        '*' in subpackage_name is handled as a wildcard.
        """
        if subpackage_name is None:
            if subpackage_path is None:
                raise ValueError(
                    "either subpackage_name or subpackage_path must be specified")
            subpackage_name = os.path.basename(subpackage_path)

        # handle wildcards
        l = subpackage_name.split('.')
        if subpackage_path is None and '*' in subpackage_name:
            return self._wildcard_get_subpackage(subpackage_name,
                                                 parent_name,
                                                 caller_level = caller_level+1)
        assert '*' not in subpackage_name,`subpackage_name, subpackage_path,parent_name`
        if subpackage_path is None:
            subpackage_path = njoin([self.local_path] + l)
        else:
            subpackage_path = njoin([subpackage_path] + l[:-1])
            subpackage_path = self.paths([subpackage_path])[0]
        setup_py = njoin(subpackage_path, 'setup.py')
        if not self.options['ignore_setup_xxx_py']:
            if not os.path.isfile(setup_py):
                setup_py = njoin(subpackage_path,
                                 'setup_%s.py' % (subpackage_name))
        if not os.path.isfile(setup_py):
            if not self.options['assume_default_configuration']:
                self.warn('Assuming default configuration '\
                          '(%s/{setup_%s,setup}.py was not found)' \
                          % (os.path.dirname(setup_py), subpackage_name))
            config = Configuration(subpackage_name, parent_name,
                                   self.top_path, subpackage_path,
                                   caller_level = caller_level+1)
        else:
            config = self._get_configuration_from_setup_py(
                setup_py,
                subpackage_name,
                subpackage_path,
                parent_name,
                caller_level = caller_level + 1)
        if config:
            return [config]
        else:
            return []

    def add_subpackage(self,subpackage_name,
                       subpackage_path=None,
                       standalone = False):
        """Add subpackage to configuration.
        """
        if standalone:
            parent_name = None
        else:
            parent_name = self.name
        config_list = self.get_subpackage(subpackage_name,subpackage_path,
                                          parent_name = parent_name,
                                          caller_level = 2)
        if not config_list:
            self.warn('No configuration returned, assuming unavailable.')
        for config in config_list:
            d = config
            if isinstance(config, Configuration):
                d = config.todict()
            assert isinstance(d,dict),`type(d)`

            self.info('Appending %s configuration to %s' \
                      % (d.get('name'), self.name))
            self.dict_append(**d)

        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add a subpackage '+ subpackage_name)

    def add_data_dir(self,data_path):
        """Recursively add files under data_path to data_files list.
        Argument can be either
        - 2-sequence (<datadir suffix>,<path to data directory>)
        - path to data directory where python datadir suffix defaults
          to package dir.

        Rules for installation paths:
          foo/bar -> (foo/bar, foo/bar) -> parent/foo/bar
          (gun, foo/bar) -> parent/gun
          foo/* -> (foo/a, foo/a), (foo/b, foo/b) -> parent/foo/a, parent/foo/b
          (gun, foo/*) -> (gun, foo/a), (gun, foo/b) -> gun
          (gun/*, foo/*) -> parent/gun/a, parent/gun/b
          /foo/bar -> (bar, /foo/bar) -> parent/bar
          (gun, /foo/bar) -> parent/gun
          (fun/*/gun/*, sun/foo/bar) -> parent/fun/foo/gun/bar
        """
        if is_sequence(data_path):
            d, data_path = data_path
        else:
            d = None
        if is_sequence(data_path):
            [self.add_data_dir((d,p)) for p in data_path]
            return
        if not is_string(data_path):
            raise TypeError("not a string: %r" % (data_path,))
        if d is None:
            if os.path.isabs(data_path):
                return self.add_data_dir((os.path.basename(data_path), data_path))
            return self.add_data_dir((data_path, data_path))
        paths = self.paths(data_path, include_non_existing=False)
        if is_glob_pattern(data_path):
            if is_glob_pattern(d):
                pattern_list = allpath(d).split(os.sep)
                pattern_list.reverse()
                # /a/*//b/ -> /a/*/b
                rl = range(len(pattern_list)-1); rl.reverse()
                for i in rl:
                    if not pattern_list[i]:
                        del pattern_list[i]
                #
                for path in paths:
                    if not os.path.isdir(path):
                        print 'Not a directory, skipping',path
                        continue
                    rpath = rel_path(path, self.local_path)
                    path_list = rpath.split(os.sep)
                    path_list.reverse()
                    target_list = []
                    i = 0
                    for s in pattern_list:
                        if is_glob_pattern(s):
                            if i>=len(path_list):
                                raise ValueError,'cannot fill pattern %r with %r' \
                                      % (d, path)
                            target_list.append(path_list[i])
                        else:
                            assert s==path_list[i],`s,path_list[i],data_path,d,path,rpath`
                            target_list.append(s)
                        i += 1
                    if path_list[i:]:
                        self.warn('mismatch of pattern_list=%s and path_list=%s'\
                                  % (pattern_list,path_list))
                    target_list.reverse()
                    self.add_data_dir((os.sep.join(target_list),path))
            else:
                for path in paths:
                    self.add_data_dir((d,path))
            return
        assert not is_glob_pattern(d),`d`

        dist = self.get_distribution()
        if dist is not None and dist.data_files is not None:
            data_files = dist.data_files
        else:
            data_files = self.data_files

        for path in paths:
            for d1,f in list(general_source_directories_files(path)):
                target_path = os.path.join(self.path_in_package,d,d1)
                data_files.append((target_path, f))

    def _optimize_data_files(self):
        data_dict = {}
        for p,files in self.data_files:
            if not data_dict.has_key(p):
                data_dict[p] = set()
            map(data_dict[p].add,files)
        self.data_files[:] = [(p,list(files)) for p,files in data_dict.items()]

    def add_data_files(self,*files):
        """Add data files to configuration data_files.
        Argument(s) can be either
        - 2-sequence (<datadir prefix>,<path to data file(s)>)
        - paths to data files where python datadir prefix defaults
          to package dir.

        Rules for installation paths:
          file.txt -> (., file.txt)-> parent/file.txt
          foo/file.txt -> (foo, foo/file.txt) -> parent/foo/file.txt
          /foo/bar/file.txt -> (., /foo/bar/file.txt) -> parent/file.txt
          *.txt -> parent/a.txt, parent/b.txt
          foo/*.txt -> parent/foo/a.txt, parent/foo/b.txt
          */*.txt -> (*, */*.txt) -> parent/c/a.txt, parent/d/b.txt
          (sun, file.txt) -> parent/sun/file.txt
          (sun, bar/file.txt) -> parent/sun/file.txt
          (sun, /foo/bar/file.txt) -> parent/sun/file.txt
          (sun, *.txt) -> parent/sun/a.txt, parent/sun/b.txt
          (sun, bar/*.txt) -> parent/sun/a.txt, parent/sun/b.txt
          (sun/*, */*.txt) -> parent/sun/c/a.txt, parent/d/b.txt
        """

        if len(files)>1:
            map(self.add_data_files, files)
            return
        assert len(files)==1
        if is_sequence(files[0]):
            d,files = files[0]
        else:
            d = None
        if is_string(files):
            filepat = files
        elif is_sequence(files):
            if len(files)==1:
                filepat = files[0]
            else:
                for f in files:
                    self.add_data_files((d,f))
                return
        else:
            raise TypeError,`type(files)`

        if d is None:
            if callable(filepat):
                d = ''
            elif os.path.isabs(filepat):
                d = ''
            else:
                d = os.path.dirname(filepat)
            self.add_data_files((d,files))
            return

        paths = self.paths(filepat, include_non_existing=False)
        if is_glob_pattern(filepat):
            if is_glob_pattern(d):
                pattern_list = d.split(os.sep)
                pattern_list.reverse()
                for path in paths:
                    path_list = path.split(os.sep)
                    path_list.reverse()
                    path_list.pop() # filename
                    target_list = []
                    i = 0
                    for s in pattern_list:
                        if is_glob_pattern(s):
                            target_list.append(path_list[i])
                            i += 1
                        else:
                            target_list.append(s)
                    target_list.reverse()
                    self.add_data_files((os.sep.join(target_list), path))
            else:
                self.add_data_files((d,paths))
            return
        assert not is_glob_pattern(d),`d,filepat`

        dist = self.get_distribution()
        if dist is not None and dist.data_files is not None:
            data_files = dist.data_files
        else:
            data_files = self.data_files

        data_files.append((os.path.join(self.path_in_package,d),paths))

    ### XXX Implement add_py_modules

    def add_include_dirs(self,*paths):
        """Add paths to configuration include directories.
        """
        include_dirs = self.paths(paths)
        dist = self.get_distribution()
        if dist is not None:
            dist.include_dirs.extend(include_dirs)
        else:
            self.include_dirs.extend(include_dirs)

    def add_numarray_include_dirs(self):
        import numpy.numarray.util as nnu
        self.add_include_dirs(*nnu.get_numarray_include_dirs())

    def add_headers(self,*files):
        """Add installable headers to configuration.
        Argument(s) can be either
        - 2-sequence (<includedir suffix>,<path to header file(s)>)
        - path(s) to header file(s) where python includedir suffix will default
          to package name.
        """
        headers = []
        for path in files:
            if is_string(path):
                [headers.append((self.name,p)) for p in self.paths(path)]
            else:
                if not isinstance(path, (tuple, list)) or len(path) != 2:
                    raise TypeError(repr(path))
                [headers.append((path[0],p)) for p in self.paths(path[1])]
        dist = self.get_distribution()
        if dist is not None:
            dist.headers.extend(headers)
        else:
            self.headers.extend(headers)

    def paths(self,*paths,**kws):
        """Apply glob to paths and prepend local_path if needed.
        """
        include_non_existing = kws.get('include_non_existing',True)
        return gpaths(paths,
                      local_path = self.local_path,
                      include_non_existing=include_non_existing)

    def _fix_paths_dict(self,kw):
        for k in kw.keys():
            v = kw[k]
            if k in ['sources','depends','include_dirs','library_dirs',
                     'module_dirs','extra_objects']:
                new_v = self.paths(v)
                kw[k] = new_v

    def add_extension(self,name,sources,**kw):
        """Add extension to configuration.

        Keywords:
          include_dirs, define_macros, undef_macros,
          library_dirs, libraries, runtime_library_dirs,
          extra_objects, extra_compile_args, extra_link_args,
          export_symbols, swig_opts, depends, language,
          f2py_options, module_dirs
          extra_info - dict or list of dict of keywords to be
                       appended to keywords.
        """
        ext_args = copy.copy(kw)
        ext_args['name'] = dot_join(self.name,name)
        ext_args['sources'] = sources

        if ext_args.has_key('extra_info'):
            extra_info = ext_args['extra_info']
            del ext_args['extra_info']
            if isinstance(extra_info, dict):
                extra_info = [extra_info]
            for info in extra_info:
                assert isinstance(info, dict), repr(info)
                dict_append(ext_args,**info)

        self._fix_paths_dict(ext_args)

        # Resolve out-of-tree dependencies
        libraries = ext_args.get('libraries',[])
        libnames = []
        ext_args['libraries'] = []
        for libname in libraries:
            if isinstance(libname,tuple):
                self._fix_paths_dict(libname[1])

            # Handle library names of the form libname@relative/path/to/library
            if '@' in libname:
                lname,lpath = libname.split('@',1)
                lpath = os.path.abspath(njoin(self.local_path,lpath))
                if os.path.isdir(lpath):
                    c = self.get_subpackage(None,lpath,
                                            caller_level = 2)
                    if isinstance(c,Configuration):
                        c = c.todict()
                    for l in [l[0] for l in c.get('libraries',[])]:
                        llname = l.split('__OF__',1)[0]
                        if llname == lname:
                            c.pop('name',None)
                            dict_append(ext_args,**c)
                            break
                    continue
            libnames.append(libname)

        ext_args['libraries'] = libnames + ext_args['libraries']

        from numpy.distutils.core import Extension
        ext = Extension(**ext_args)
        self.ext_modules.append(ext)

        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add an extension '+name)
        return ext

    def add_library(self,name,sources,**build_info):
        """Add library to configuration.

        Valid keywords for build_info:
          depends
          macros
          include_dirs
          extra_compiler_args
          f2py_options
          language
        """
        build_info = copy.copy(build_info)
        name = name #+ '__OF__' + self.name
        build_info['sources'] = sources

        self._fix_paths_dict(build_info)

        self.libraries.append((name,build_info))

        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add a library '+ name)

    def add_scripts(self,*files):
        """Add scripts to configuration.
        """
        scripts = self.paths(files)
        dist = self.get_distribution()
        if dist is not None:
            dist.scripts.extend(scripts)
        else:
            self.scripts.extend(scripts)

    def dict_append(self,**dict):
        for key in self.list_keys:
            a = getattr(self,key)
            a.extend(dict.get(key,[]))
        for key in self.dict_keys:
            a = getattr(self,key)
            a.update(dict.get(key,{}))
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for key in dict.keys():
            if key not in known_keys:
                a = getattr(self, key, None)
                if a and a==dict[key]: continue
                self.warn('Inheriting attribute %r=%r from %r' \
                          % (key,dict[key],dict.get('name','?')))
                setattr(self,key,dict[key])
                self.extra_keys.append(key)
            elif key in self.extra_keys:
                self.info('Ignoring attempt to set %r (from %r to %r)' \
                          % (key, getattr(self,key), dict[key]))
            elif key in known_keys:
                # key is already processed above
                pass
            else:
                raise ValueError, "Don't know about key=%r" % (key)

    def __str__(self):
        from pprint import pformat
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        s = '<'+5*'-' + '\n'
        s += 'Configuration of '+self.name+':\n'
        known_keys.sort()
        for k in known_keys:
            a = getattr(self,k,None)
            if a:
                s += '%s = %s\n' % (k,pformat(a))
        s += 5*'-' + '>'
        return s

    def get_config_cmd(self):
        cmd = get_cmd('config')
        cmd.ensure_finalized()
        cmd.dump_source = 0
        cmd.noisy = 0
        old_path = os.environ.get('PATH')
        if old_path:
            path = os.pathsep.join(['.',old_path])
            os.environ['PATH'] = path
        return cmd

    def get_build_temp_dir(self):
        cmd = get_cmd('build')
        cmd.ensure_finalized()
        return cmd.build_temp

    def have_f77c(self):
        """Check for availability of Fortran 77 compiler.
        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.
        """
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine,lang='f77')
        return flag

    def have_f90c(self):
        """Check for availability of Fortran 90 compiler.
        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.
        """
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine,lang='f90')
        return flag

    def append_to(self, extlib):
        """Append libraries, include_dirs to extension or library item.
        """
        if is_sequence(extlib):
            lib_name, build_info = extlib
            dict_append(build_info,
                        libraries=self.libraries,
                        include_dirs=self.include_dirs)
        else:
            from numpy.distutils.core import Extension
            assert isinstance(extlib,Extension), repr(extlib)
            extlib.libraries.extend(self.libraries)
            extlib.include_dirs.extend(self.include_dirs)

    def _get_svn_revision(self,path):
        """Return path's SVN revision number.
        """
        revision = None
        m = None
        try:
            sin, sout = os.popen4('svnversion')
            m = re.match(r'(?P<revision>\d+)', sout.read())
        except:
            pass
        if m:
            revision = int(m.group('revision'))
            return revision
        if sys.platform=='win32' and os.environ.get('SVN_ASP_DOT_NET_HACK',None):
            entries = njoin(path,'_svn','entries')
        else:
            entries = njoin(path,'.svn','entries')
        if os.path.isfile(entries):
            f = open(entries)
            fstr = f.read()
            f.close()
            if fstr[:5] == '<?xml':  # pre 1.4
                m = re.search(r'revision="(?P<revision>\d+)"',fstr)
                if m:
                    revision = int(m.group('revision'))
            else:  # non-xml entries file --- check to be sure that
                m = re.search(r'dir[\n\r]+(?P<revision>\d+)', fstr)
                if m:
                    revision = int(m.group('revision'))
        return revision

    def get_version(self, version_file=None, version_variable=None):
        """Try to get version string of a package.
        """
        version = getattr(self,'version',None)
        if version is not None:
            return version

        # Get version from version file.
        if version_file is None:
            files = ['__version__.py',
                     self.name.split('.')[-1]+'_version.py',
                     'version.py',
                     '__svn_version__.py']
        else:
            files = [version_file]
        if version_variable is None:
            version_vars = ['version',
                            '__version__',
                            self.name.split('.')[-1]+'_version']
        else:
            version_vars = [version_variable]
        for f in files:
            fn = njoin(self.local_path,f)
            if os.path.isfile(fn):
                info = (open(fn),fn,('.py','U',1))
                name = os.path.splitext(os.path.basename(fn))[0]
                n = dot_join(self.name,name)
                try:
                    version_module = imp.load_module('_'.join(n.split('.')),*info)
                except ImportError,msg:
                    self.warn(str(msg))
                    version_module = None
                if version_module is None:
                    continue

                for a in version_vars:
                    version = getattr(version_module,a,None)
                    if version is not None:
                        break
                if version is not None:
                    break

        if version is not None:
            self.version = version
            return version

        # Get version as SVN revision number
        revision = self._get_svn_revision(self.local_path)
        if revision is not None:
            version = str(revision)
            self.version = version

        return version

    def make_svn_version_py(self, delete=True):
        """Generate package __svn_version__.py file from SVN revision number,
        it will be removed after python exits but will be available
        when sdist, etc commands are executed.

        If __svn_version__.py existed before, nothing is done.
        """
        target = njoin(self.local_path,'__svn_version__.py')
        revision = self._get_svn_revision(self.local_path)
        if os.path.isfile(target) or revision is None:
            return
        else:
            def generate_svn_version_py():
                if not os.path.isfile(target):
                    version = str(revision)
                    self.info('Creating %s (version=%r)' % (target,version))
                    f = open(target,'w')
                    f.write('version = %r\n' % (version))
                    f.close()

                import atexit
                def rm_file(f=target,p=self.info):
                    if delete:
                        try: os.remove(f); p('removed '+f)
                        except OSError: pass
                        try: os.remove(f+'c'); p('removed '+f+'c')
                        except OSError: pass

                atexit.register(rm_file)

                return target

            self.add_data_files(('', generate_svn_version_py()))

    def make_config_py(self,name='__config__'):
        """Generate package __config__.py file containing system_info
        information used during building the package.
        """
        self.py_modules.append((self.name,name,generate_config_py))

    def get_info(self,*names):
        """Get resources information.
        """
        from system_info import get_info, dict_append
        info_dict = {}
        for a in names:
            dict_append(info_dict,**get_info(a))
        return info_dict


def get_cmd(cmdname, _cache={}):
    if not _cache.has_key(cmdname):
        import distutils.core
        dist = distutils.core._setup_distribution
        if dist is None:
            from distutils.errors import DistutilsInternalError
            raise DistutilsInternalError(
                  'setup distribution instance not initialized')
        cmd = dist.get_command_obj(cmdname)
        _cache[cmdname] = cmd
    return _cache[cmdname]

def get_numpy_include_dirs():
    # numpy_include_dirs are set by numpy/core/setup.py, otherwise []
    include_dirs = Configuration.numpy_include_dirs[:]
    if not include_dirs:
        import numpy
        include_dirs = [ numpy.get_include() ]
    # else running numpy/core/setup.py
    return include_dirs

#########################

def default_config_dict(name = None, parent_name = None, local_path=None):
    """Return a configuration dictionary for usage in
    configuration() function defined in file setup_<name>.py.
    """
    import warnings
    warnings.warn('Use Configuration(%r,%r,top_path=%r) instead of '\
                  'deprecated default_config_dict(%r,%r,%r)'
                  % (name, parent_name, local_path,
                     name, parent_name, local_path,
                     ))
    c = Configuration(name, parent_name, local_path)
    return c.todict()


def dict_append(d, **kws):
    for k, v in kws.items():
        if d.has_key(k):
            ov = d[k]
            if isinstance(ov,str):
                d[k] = v
            else:
                d[k].extend(v)
        else:
            d[k] = v

def appendpath(prefix, path):
    if os.path.sep != '/':
        prefix = prefix.replace('/', os.path.sep)
        path = path.replace('/', os.path.sep)
    drive = ''
    if os.path.isabs(path):
        drive = os.path.splitdrive(prefix)[0]
        absprefix = os.path.splitdrive(os.path.abspath(prefix))[1]
        pathdrive, path = os.path.splitdrive(path)
        d = os.path.commonprefix([absprefix, path])
        if os.path.join(absprefix[:len(d)], absprefix[len(d):]) != absprefix \
           or os.path.join(path[:len(d)], path[len(d):]) != path:
            # Handle invalid paths
            d = os.path.dirname(d)
        subpath = path[len(d):]
        if os.path.isabs(subpath):
            subpath = subpath[1:]
    else:
        subpath = path
    return os.path.normpath(njoin(drive + prefix, subpath))

def generate_config_py(target):
    """Generate config.py file containing system_info information
    used during building the package.

    Usage:
        config['py_modules'].append((packagename, '__config__',generate_config_py))
    """
    from numpy.distutils.system_info import system_info
    from distutils.dir_util import mkpath
    mkpath(os.path.dirname(target))
    f = open(target, 'w')
    f.write('# This file is generated by %s\n' % (os.path.abspath(sys.argv[0])))
    f.write('# It contains system_info results at the time of building this package.\n')
    f.write('__all__ = ["get_info","show"]\n\n')
    for k, i in system_info.saved_results.items():
        f.write('%s=%r\n' % (k, i))
    f.write(r'''
def get_info(name):
    g = globals()
    return g.get(name, g.get(name + "_info", {}))

def show():
    for name,info_dict in globals().items():
        if name[0] == "_" or type(info_dict) is not type({}): continue
        print name + ":"
        if not info_dict:
            print "  NOT AVAILABLE"
        for k,v in info_dict.items():
            v = str(v)
            if k == "sources" and len(v) > 200:
                v = v[:60] + " ...\n... " + v[-60:]
            print "    %s = %s" % (k,v)
        print
    ''')

    f.close()
    return target
