import re
import os
import sys
import new

from distutils.ccompiler import *
from distutils import ccompiler
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion

import log
from exec_command import exec_command
from misc_util import cyg2win32, is_sequence, mingw32
from distutils.spawn import _nt_quote_args

# hack to set compiler optimizing options. Needs to integrated with something.
import distutils.sysconfig
_old_init_posix = distutils.sysconfig._init_posix
def _new_init_posix():
    _old_init_posix()
    distutils.sysconfig._config_vars['OPT'] = '-Wall -g -O0'
#distutils.sysconfig._init_posix = _new_init_posix

# Using customized CCompiler.spawn.
def CCompiler_spawn(self, cmd, display=None):
    if display is None:
        display = cmd
        if is_sequence(display):
            display = ' '.join(list(display))
    log.info(display)
    if is_sequence(cmd) and os.name == 'nt':
        cmd = _nt_quote_args(list(cmd))
    s,o = exec_command(cmd)
    if s:
        if is_sequence(cmd):
            cmd = ' '.join(list(cmd))
        print o
        raise DistutilsExecError,\
              'Command "%s" failed with exit status %d' % (cmd, s)
CCompiler.spawn = new.instancemethod(CCompiler_spawn,None,CCompiler)

def CCompiler_object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
    if output_dir is None:
        output_dir = ''
    obj_names = []
    for src_name in source_filenames:
        base, ext = os.path.splitext(os.path.normpath(src_name))
        base = os.path.splitdrive(base)[1] # Chop off the drive
        base = base[os.path.isabs(base):]  # If abs, chop off leading /
        if base.startswith('..'):
            # Resolve starting relative path components, middle ones
            # (if any) have been handled by os.path.normpath above.
            i = base.rfind('..')+2
            d = base[:i]
            d = os.path.basename(os.path.abspath(d))
            base = d + base[i:]
        if ext not in self.src_extensions:
            raise UnknownFileError, \
                  "unknown file type '%s' (from '%s')" % (ext, src_name)
        if strip_dir:
            base = os.path.basename(base)
        obj_name = os.path.join(output_dir,base + self.obj_extension)
        obj_names.append(obj_name)
    return obj_names

CCompiler.object_filenames = new.instancemethod(CCompiler_object_filenames,
                                                None,CCompiler)

def CCompiler_compile(self, sources, output_dir=None, macros=None,
                      include_dirs=None, debug=0, extra_preargs=None,
                      extra_postargs=None, depends=None):
    # This method is effective only with Python >=2.3 distutils.
    # Any changes here should be applied also to fcompiler.compile
    # method to support pre Python 2.3 distutils.
    if not sources:
        return []
    from fcompiler import FCompiler
    if isinstance(self, FCompiler):
        display = []
        for fc in ['f77','f90','fix']:
            fcomp = getattr(self,'compiler_'+fc)
            if fcomp is None:
                continue
            display.append("Fortran %s compiler: %s" % (fc, ' '.join(fcomp)))
        display = '\n'.join(display)
    else:
        ccomp = self.compiler_so
        display = "C compiler: %s\n" % (' '.join(ccomp),)
    log.info(display)
    macros, objects, extra_postargs, pp_opts, build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    display = "compile options: '%s'" % (' '.join(cc_args))
    if extra_postargs:
        display += "\nextra options: '%s'" % (' '.join(extra_postargs))
    log.info(display)

    # build any sources in same order as they were originally specified
    #   especially important for fortran .f90 files using modules
    if isinstance(self, FCompiler):
        objects_to_build = build.keys()
        for obj in objects:
            if obj in objects_to_build:
                src, ext = build[obj]
                if self.compiler_type=='absoft':
                    obj = cyg2win32(obj)
                    src = cyg2win32(src)
                self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    else:
        for obj, (src, ext) in build.items():
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # Return *all* object filenames, not just the ones we just built.
    return objects

CCompiler.compile = new.instancemethod(CCompiler_compile,None,CCompiler)

def CCompiler_customize_cmd(self, cmd):
    """ Customize compiler using distutils command.
    """
    log.info('customize %s using %s' % (self.__class__.__name__,
                                        cmd.__class__.__name__))
    if getattr(cmd,'include_dirs',None) is not None:
        self.set_include_dirs(cmd.include_dirs)
    if getattr(cmd,'define',None) is not None:
        for (name,value) in cmd.define:
            self.define_macro(name, value)
    if getattr(cmd,'undef',None) is not None:
        for macro in cmd.undef:
            self.undefine_macro(macro)
    if getattr(cmd,'libraries',None) is not None:
        self.set_libraries(self.libraries + cmd.libraries)
    if getattr(cmd,'library_dirs',None) is not None:
        self.set_library_dirs(self.library_dirs + cmd.library_dirs)
    if getattr(cmd,'rpath',None) is not None:
        self.set_runtime_library_dirs(cmd.rpath)
    if getattr(cmd,'link_objects',None) is not None:
        self.set_link_objects(cmd.link_objects)
    return

CCompiler.customize_cmd = new.instancemethod(\
    CCompiler_customize_cmd,None,CCompiler)

def _compiler_to_string(compiler):
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
            props.append((key,repr(v)))
    lines = []
    format = '%-' + repr(mx+1) + 's = %s'
    for prop in props:
        lines.append(format % prop)
    return '\n'.join(lines)

def CCompiler_show_customization(self):
    if 0:
        for attrname in ['include_dirs','define','undef',
                         'libraries','library_dirs',
                         'rpath','link_objects']:
            attr = getattr(self,attrname,None)
            if not attr:
                continue
            log.info("compiler '%s' is set to %s" % (attrname,attr))
    try: self.get_version()
    except: pass
    if log._global_log.threshold<2:
        print '*'*80
        print self.__class__
        print _compiler_to_string(self)
        print '*'*80

CCompiler.show_customization = new.instancemethod(\
    CCompiler_show_customization,None,CCompiler)


def CCompiler_customize(self, dist, need_cxx=0):
    # See FCompiler.customize for suggested usage.
    log.info('customize %s' % (self.__class__.__name__))
    customize_compiler(self)
    if need_cxx:
        # In general, distutils uses -Wstrict-prototypes, but this option is
        # not valid for C++ code, only for C.  Remove it if it's there to
        # avoid a spurious warning on every compilation.  All the default
        # options used by distutils can be extracted with:
        
        # from distutils import sysconfig
        # sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS',
        # 'CCSHARED', 'LDSHARED', 'SO')
        try:
            self.compiler_so.remove('-Wstrict-prototypes')
        except ValueError:
            pass
        
        if hasattr(self,'compiler') and self.compiler[0].find('cc')>=0:
            if not self.compiler_cxx:
                if self.compiler[0][:3] == 'gcc':
                    a, b = 'gcc', 'g++'
                else:
                    a, b = 'cc', 'c++'
                self.compiler_cxx = [self.compiler[0].replace(a,b)]\
                                    + self.compiler[1:]
        else:
            if hasattr(self,'compiler'):
                 log.warn("#### %s #######" % (self.compiler,))
            log.warn('Missing compiler_cxx fix for '+self.__class__.__name__)
    return

CCompiler.customize = new.instancemethod(\
    CCompiler_customize,None,CCompiler)

def simple_version_match(pat=r'[-.\d]+', ignore=None, start=''):
    def matcher(self, version_string):
        pos = 0
        if start:
            m = re.match(start, version_string)
            if not m:
                return None
            pos = m.end()
        while 1:
            m = re.search(pat, version_string[pos:])
            if not m:
                return None
            if ignore and re.match(ignore, m.group(0)):
                pos = m.end()
                continue
            break
        return m.group(0)
    return matcher

def CCompiler_get_version(self, force=0, ok_status=[0]):
    """ Compiler version. Returns None if compiler is not available. """
    if not force and hasattr(self,'version'):
        return self.version
    try:
        version_cmd = self.version_cmd
    except AttributeError:
        return None
    cmd = ' '.join(version_cmd)
    try:
        matcher = self.version_match
    except AttributeError:
        try:
            pat = self.version_pattern
        except AttributeError:
            return None
        def matcher(version_string):
            m = re.match(pat, version_string)
            if not m:
                return None
            version = m.group('version')
            return version

    status, output = exec_command(cmd,use_tee=0)
    version = None
    if status in ok_status:
        version = matcher(output)
        if not version:
            log.warn("Couldn't match compiler version for %r" % (output,))
        else:
            version = LooseVersion(version)
    self.version = version
    return version

CCompiler.get_version = new.instancemethod(\
    CCompiler_get_version,None,CCompiler)

compiler_class['intel'] = ('intelccompiler','IntelCCompiler',
                           "Intel C Compiler for 32-bit applications")
compiler_class['intele'] = ('intelccompiler','IntelItaniumCCompiler',
                           "Intel C Itanium Compiler for Itanium-based applications")
ccompiler._default_compilers = ccompiler._default_compilers \
                               + (('linux.*','intel'),('linux.*','intele'))

if sys.platform == 'win32':
    compiler_class['mingw32'] = ('mingw32ccompiler', 'Mingw32CCompiler',
                                 "Mingw32 port of GNU C Compiler for Win32"\
                                 "(for MSC built Python)")
    if mingw32():
        # On windows platforms, we want to default to mingw32 (gcc)
        # because msvc can't build blitz stuff.
        log.info('Setting mingw32 as default compiler for nt.')
        ccompiler._default_compilers = (('nt', 'mingw32'),) \
                                       + ccompiler._default_compilers


_distutils_new_compiler = new_compiler
def new_compiler (plat=None,
                  compiler=None,
                  verbose=0,
                  dry_run=0,
                  force=0):
    # Try first C compilers from numpy.distutils.
    if plat is None:
        plat = os.name
    try:
        if compiler is None:
            compiler = get_default_compiler(plat)
        (module_name, class_name, long_description) = compiler_class[compiler]
    except KeyError:
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError, msg
    module_name = "numpy.distutils." + module_name
    try:
        __import__ (module_name)
    except ImportError, msg:
        log.info('%s in numpy.distutils; trying from distutils',
                 str(msg))
        module_name = module_name[6:]
        try:
            __import__(module_name)
        except ImportError, msg:
            raise DistutilsModuleError, \
                  "can't compile C/C++ code: unable to load module '%s'" % \
                  module_name
    try:
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except KeyError:
        raise DistutilsModuleError, \
              ("can't compile C/C++ code: unable to find class '%s' " +
               "in module '%s'") % (class_name, module_name)
    compiler = klass(None, dry_run, force)
    log.debug('new_compiler returns %s' % (klass))
    return compiler

ccompiler.new_compiler = new_compiler


_distutils_gen_lib_options = gen_lib_options
def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
    r = _distutils_gen_lib_options(compiler, library_dirs,
                                   runtime_library_dirs, libraries)
    lib_opts = []
    for i in r:
        if is_sequence(i):
            lib_opts.extend(list(i))
        else:
            lib_opts.append(i)
    return lib_opts
ccompiler.gen_lib_options = gen_lib_options


##Fix distutils.util.split_quoted:
import re,string
_wordchars_re = re.compile(r'[^\\\'\"%s ]*' % string.whitespace)
_squote_re = re.compile(r"'(?:[^'\\]|\\.)*'")
_dquote_re = re.compile(r'"(?:[^"\\]|\\.)*"')
_has_white_re = re.compile(r'\s')
def split_quoted(s):
    s = string.strip(s)
    words = []
    pos = 0

    while s:
        m = _wordchars_re.match(s, pos)
        end = m.end()
        if end == len(s):
            words.append(s[:end])
            break

        if s[end] in string.whitespace: # unescaped, unquoted whitespace: now
            words.append(s[:end])       # we definitely have a word delimiter
            s = string.lstrip(s[end:])
            pos = 0

        elif s[end] == '\\':            # preserve whatever is being escaped;
                                        # will become part of the current word
            s = s[:end] + s[end+1:]
            pos = end+1

        else:
            if s[end] == "'":           # slurp singly-quoted string
                m = _squote_re.match(s, end)
            elif s[end] == '"':         # slurp doubly-quoted string
                m = _dquote_re.match(s, end)
            else:
                raise RuntimeError, \
                      "this can't happen (bad char '%c')" % s[end]

            if m is None:
                raise ValueError, \
                      "bad string (mismatched %s quotes?)" % s[end]

            (beg, end) = m.span()
            if _has_white_re.search(s[beg+1:end-1]):
                s = s[:beg] + s[beg+1:end-1] + s[end:]
                pos = m.end() - 2
            else:
                # Keeping quotes when a quoted word does not contain
                # white-space. XXX: send a patch to distutils
                pos = m.end()

        if pos >= len(s):
            words.append(s)
            break

    return words
ccompiler.split_quoted = split_quoted
