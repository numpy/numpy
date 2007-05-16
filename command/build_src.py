""" Build swig, f2py, weave, sources.
"""

import os
import re
import sys

from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform

try:
    from Pyrex.Compiler import Main
    have_pyrex = True
except ImportError:
    have_pyrex = False

# this import can't be done here, as it uses numpy stuff only available
# after it's installed
#import numpy.f2py
from numpy.distutils import log
from numpy.distutils.misc_util import fortran_ext_match, \
     appendpath, is_string, is_sequence
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file

class build_src(build_ext.build_ext):

    description = "build sources from SWIG, F2PY files or a function"

    user_options = [
        ('build-src=', 'd', "directory to \"build\" sources to"),
        ('f2pyflags=', None, "additonal flags to f2py"),
        ('swigflags=', None, "additional flags to swig"),
        ('force', 'f', "forcibly build everything (ignore file timestamps)"),
        ('inplace', 'i',
         "ignore build-lib and put compiled extensions into the source " +
         "directory alongside your pure Python modules"),
        ]

    boolean_options = ['force','inplace']

    help_options = []

    def initialize_options(self):
        self.extensions = None
        self.package = None
        self.py_modules = None
        self.py_modules_dict = None
        self.build_src = None
        self.build_lib = None
        self.build_base = None
        self.force = None
        self.inplace = None
        self.package_dir = None
        self.f2pyflags = None
        self.swigflags = None

    def finalize_options(self):
        self.set_undefined_options('build',
                                   ('build_base', 'build_base'),
                                   ('build_lib', 'build_lib'),
                                   ('force', 'force'))
        if self.package is None:
            self.package = self.distribution.ext_package
        self.extensions = self.distribution.ext_modules
        self.libraries = self.distribution.libraries or []
        self.py_modules = self.distribution.py_modules or []
        self.data_files = self.distribution.data_files or []

        if self.build_src is None:
            plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
            self.build_src = os.path.join(self.build_base, 'src'+plat_specifier)
        if self.inplace is None:
            build_ext = self.get_finalized_command('build_ext')
            self.inplace = build_ext.inplace

        # py_modules_dict is used in build_py.find_package_modules
        self.py_modules_dict = {}

        if self.f2pyflags is None:
            self.f2pyflags = []
        else:
            self.f2pyflags = self.f2pyflags.split() # XXX spaces??

        if self.swigflags is None:
            self.swigflags = []
        else:
            self.swigflags = self.swigflags.split() # XXX spaces??

    def run(self):
        if not (self.extensions or self.libraries):
            return
        self.build_sources()

    def build_sources(self):

        if self.inplace:
            self.get_package_dir = \
                     self.get_finalized_command('build_py').get_package_dir

        self.build_py_modules_sources()

        for libname_info in self.libraries:
            self.build_library_sources(*libname_info)

        if self.extensions:
            self.check_extensions_list(self.extensions)

            for ext in self.extensions:
                self.build_extension_sources(ext)

        self.build_data_files_sources()

    def build_data_files_sources(self):
        if not self.data_files:
            return
        log.info('building data_files sources')
        from numpy.distutils.misc_util import get_data_files
        new_data_files = []
        for data in self.data_files:
            if isinstance(data,str):
                new_data_files.append(data)
            elif isinstance(data,tuple):
                d,files = data
                if self.inplace:
                    build_dir = self.get_package_dir('.'.join(d.split(os.sep)))
                else:
                    build_dir = os.path.join(self.build_src,d)
                funcs = filter(callable,files)
                files = filter(lambda f:not callable(f), files)
                for f in funcs:
                    if f.func_code.co_argcount==1:
                        s = f(build_dir)
                    else:
                        s = f()
                    if s is not None:
                        if isinstance(s,list):
                            files.extend(s)
                        elif isinstance(s,str):
                            files.append(s)
                        else:
                            raise TypeError(repr(s))
                filenames = get_data_files((d,files))
                new_data_files.append((d, filenames))
            else:
                raise
        self.data_files[:] = new_data_files

    def build_py_modules_sources(self):
        if not self.py_modules:
            return
        log.info('building py_modules sources')
        new_py_modules = []
        for source in self.py_modules:
            if is_sequence(source) and len(source)==3:
                package, module_base, source = source
                if self.inplace:
                    build_dir = self.get_package_dir(package)
                else:
                    build_dir = os.path.join(self.build_src,
                                             os.path.join(*package.split('.')))
                if callable(source):
                    target = os.path.join(build_dir, module_base + '.py')
                    source = source(target)
                if source is None:
                    continue
                modules = [(package, module_base, source)]
                if not self.py_modules_dict.has_key(package):
                    self.py_modules_dict[package] = []
                self.py_modules_dict[package] += modules
            else:
                new_py_modules.append(source)
        self.py_modules[:] = new_py_modules

    def build_library_sources(self, lib_name, build_info):
        sources = list(build_info.get('sources',[]))

        if not sources:
            return

        log.info('building library "%s" sources' % (lib_name))

        sources = self.generate_sources(sources, (lib_name, build_info))

        sources = self.template_sources(sources, (lib_name, build_info))

        sources, h_files = self.filter_h_files(sources)

        if h_files:
            print self.package,'- nothing done with h_files=',h_files

        #for f in h_files:
        #    self.distribution.headers.append((lib_name,f))

        build_info['sources'] = sources
        return

    def build_extension_sources(self, ext):

        sources = list(ext.sources)

        log.info('building extension "%s" sources' % (ext.name))

        fullname = self.get_ext_fullname(ext.name)

        modpath = fullname.split('.')
        package = '.'.join(modpath[0:-1])

        if self.inplace:
            self.ext_target_dir = self.get_package_dir(package)

        sources = self.generate_sources(sources, ext)

        sources = self.template_sources(sources, ext)

        sources = self.swig_sources(sources, ext)

        sources = self.f2py_sources(sources, ext)

        sources = self.pyrex_sources(sources, ext)

        sources, py_files = self.filter_py_files(sources)

        if not self.py_modules_dict.has_key(package):
            self.py_modules_dict[package] = []
        modules = []
        for f in py_files:
            module = os.path.splitext(os.path.basename(f))[0]
            modules.append((package, module, f))
        self.py_modules_dict[package] += modules

        sources, h_files = self.filter_h_files(sources)

        if h_files:
            print package,'- nothing done with h_files=',h_files
        #for f in h_files:
        #    self.distribution.headers.append((package,f))

        ext.sources = sources

    def generate_sources(self, sources, extension):
        new_sources = []
        func_sources = []
        for source in sources:
            if is_string(source):
                new_sources.append(source)
            else:
                func_sources.append(source)
        if not func_sources:
            return new_sources
        if self.inplace and not is_sequence(extension):
            build_dir = self.ext_target_dir
        else:
            if is_sequence(extension):
                name = extension[0]
            #    if not extension[1].has_key('include_dirs'):
            #        extension[1]['include_dirs'] = []
            #    incl_dirs = extension[1]['include_dirs']
            else:
                name = extension.name
            #    incl_dirs = extension.include_dirs
            #if self.build_src not in incl_dirs:
            #    incl_dirs.append(self.build_src)
            build_dir = os.path.join(*([self.build_src]\
                                       +name.split('.')[:-1]))
        self.mkpath(build_dir)
        for func in func_sources:
            source = func(extension, build_dir)
            if not source:
                continue
            if is_sequence(source):
                [log.info("  adding '%s' to sources." % (s,)) for s in source]
                new_sources.extend(source)
            else:
                log.info("  adding '%s' to sources." % (source,))
                new_sources.append(source)

        return new_sources

    def filter_py_files(self, sources):
        return self.filter_files(sources,['.py'])

    def filter_h_files(self, sources):
        return self.filter_files(sources,['.h','.hpp','.inc'])

    def filter_files(self, sources, exts = []):
        new_sources = []
        files = []
        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext in exts:
                files.append(source)
            else:
                new_sources.append(source)
        return new_sources, files

    def template_sources(self, sources, extension):
        new_sources = []
        if is_sequence(extension):
            depends = extension[1].get('depends')
            include_dirs = extension[1].get('include_dirs')
        else:
            depends = extension.depends
            include_dirs = extension.include_dirs
        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.src':  # Template file
                if self.inplace:
                    target_dir = os.path.dirname(base)
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                self.mkpath(target_dir)
                target_file = os.path.join(target_dir,os.path.basename(base))
                if (self.force or newer_group([source] + depends, target_file)):
                    if _f_pyf_ext_match(base):
                        log.info("from_template:> %s" % (target_file))
                        outstr = process_f_file(source)
                    else:
                        log.info("conv_template:> %s" % (target_file))
                        outstr = process_c_file(source)
                    fid = open(target_file,'w')
                    fid.write(outstr)
                    fid.close()
                if _header_ext_match(target_file):
                    d = os.path.dirname(target_file)
                    if d not in include_dirs:
                        log.info("  adding '%s' to include_dirs." % (d))
                        include_dirs.append(d)
                new_sources.append(target_file)
            else:
                new_sources.append(source)
        return new_sources

    def pyrex_sources(self, sources, extension):
        new_sources = []
        ext_name = extension.name.split('.')[-1]
        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.pyx':
                if self.inplace or not have_pyrex:
                    target_dir = os.path.dirname(base)
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                target_file = os.path.join(target_dir, ext_name + '.c')
                depends = [source] + extension.depends
                if (self.force or newer_group(depends, target_file, 'newer')):
                    if have_pyrex:
                        log.info("pyrexc:> %s" % (target_file))
                        self.mkpath(target_dir)
                        from Pyrex.Compiler import Main
                        options = Main.CompilationOptions(
                            defaults=Main.default_options,
                            output_file=target_file)
                        pyrex_result = Main.compile(source, options=options)
                        if pyrex_result.num_errors != 0:
                            raise RuntimeError("%d errors in Pyrex compile" %
                                               pyrex_result.num_errors)
                    elif os.path.isfile(target_file):
                        log.warn("Pyrex needed to compile %s but not available."\
                                 " Using old target %s"\
                                 % (source, target_file))
                    else:
                        raise SystemError,"Non-existing target %r. "\
                              "Perhaps you need to install Pyrex."\
                              % (target_file)
                new_sources.append(target_file)
            else:
                new_sources.append(source)
        return new_sources

    def f2py_sources(self, sources, extension):
        new_sources = []
        f2py_sources = []
        f_sources = []
        f2py_targets = {}
        target_dirs = []
        ext_name = extension.name.split('.')[-1]
        skip_f2py = 0

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.pyf': # F2PY interface file
                if self.inplace:
                    target_dir = os.path.dirname(base)
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                if os.path.isfile(source):
                    name = get_f2py_modulename(source)
                    if name != ext_name:
                        raise ValueError('mismatch of extension names: %s '
                                         'provides %r but expected %r' % (
                                          source, name, ext_name))
                    target_file = os.path.join(target_dir,name+'module.c')
                else:
                    log.debug('  source %s does not exist: skipping f2py\'ing.' \
                              % (source))
                    name = ext_name
                    skip_f2py = 1
                    target_file = os.path.join(target_dir,name+'module.c')
                    if not os.path.isfile(target_file):
                        log.debug('  target %s does not exist:\n   '\
                                  'Assuming %smodule.c was generated with '\
                                  '"build_src --inplace" command.' \
                                  % (target_file, name))
                        target_dir = os.path.dirname(base)
                        target_file = os.path.join(target_dir,name+'module.c')
                        if not os.path.isfile(target_file):
                            raise ValueError("%r missing" % (target_file,))
                        log.debug('   Yes! Using %s as up-to-date target.' \
                                  % (target_file))
                target_dirs.append(target_dir)
                f2py_sources.append(source)
                f2py_targets[source] = target_file
                new_sources.append(target_file)
            elif fortran_ext_match(ext):
                f_sources.append(source)
            else:
                new_sources.append(source)

        if not (f2py_sources or f_sources):
            return new_sources

        map(self.mkpath, target_dirs)

        f2py_options = extension.f2py_options + self.f2pyflags

        if self.distribution.libraries:
            for name,build_info in self.distribution.libraries:
                if name in extension.libraries:
                    f2py_options.extend(build_info.get('f2py_options',[]))

        log.info("f2py options: %s" % (f2py_options))

        if f2py_sources:
            if len(f2py_sources) != 1:
                raise ValueError(
                    'only one .pyf file is allowed per extension module but got'\
                    ' more: %r' % (f2py_sources,))
            source = f2py_sources[0]
            target_file = f2py_targets[source]
            target_dir = os.path.dirname(target_file) or '.'
            depends = [source] + extension.depends
            if (self.force or newer_group(depends, target_file,'newer')) \
                   and not skip_f2py:
                log.info("f2py: %s" % (source))
                import numpy.f2py
                numpy.f2py.run_main(f2py_options
                                    + ['--build-dir',target_dir,source])
            else:
                log.debug("  skipping '%s' f2py interface (up-to-date)" % (source))
        else:
            #XXX TODO: --inplace support for sdist command
            if is_sequence(extension):
                name = extension[0]
            else: name = extension.name
            target_dir = os.path.join(*([self.build_src]\
                                        +name.split('.')[:-1]))
            target_file = os.path.join(target_dir,ext_name + 'module.c')
            new_sources.append(target_file)
            depends = f_sources + extension.depends
            if (self.force or newer_group(depends, target_file, 'newer')) \
                   and not skip_f2py:
                log.info("f2py:> %s" % (target_file))
                self.mkpath(target_dir)
                import numpy.f2py
                numpy.f2py.run_main(f2py_options + ['--lower',
                                                '--build-dir',target_dir]+\
                                ['-m',ext_name]+f_sources)
            else:
                log.debug("  skipping f2py fortran files for '%s' (up-to-date)"\
                          % (target_file))

        if not os.path.isfile(target_file):
            raise ValueError("%r missing" % (target_file,))

        target_c = os.path.join(self.build_src,'fortranobject.c')
        target_h = os.path.join(self.build_src,'fortranobject.h')
        log.info("  adding '%s' to sources." % (target_c))
        new_sources.append(target_c)
        if self.build_src not in extension.include_dirs:
            log.info("  adding '%s' to include_dirs." \
                     % (self.build_src))
            extension.include_dirs.append(self.build_src)

        if not skip_f2py:
            import numpy.f2py
            d = os.path.dirname(numpy.f2py.__file__)
            source_c = os.path.join(d,'src','fortranobject.c')
            source_h = os.path.join(d,'src','fortranobject.h')
            if newer(source_c,target_c) or newer(source_h,target_h):
                self.mkpath(os.path.dirname(target_c))
                self.copy_file(source_c,target_c)
                self.copy_file(source_h,target_h)
        else:
            if not os.path.isfile(target_c):
                raise ValueError("%r missing" % (target_c,))
            if not os.path.isfile(target_h):
                raise ValueError("%r missing" % (target_h,))

        for name_ext in ['-f2pywrappers.f','-f2pywrappers2.f90']:
            filename = os.path.join(target_dir,ext_name + name_ext)
            if os.path.isfile(filename):
                log.info("  adding '%s' to sources." % (filename))
                f_sources.append(filename)

        return new_sources + f_sources

    def swig_sources(self, sources, extension):
        # Assuming SWIG 1.3.14 or later. See compatibility note in
        #   http://www.swig.org/Doc1.3/Python.html#Python_nn6

        new_sources = []
        swig_sources = []
        swig_targets = {}
        target_dirs = []
        py_files = []     # swig generated .py files
        target_ext = '.c'
        typ = None
        is_cpp = 0
        skip_swig = 0
        ext_name = extension.name.split('.')[-1]

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.i': # SWIG interface file
                if self.inplace:
                    target_dir = os.path.dirname(base)
                    py_target_dir = self.ext_target_dir
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                    py_target_dir = target_dir
                if os.path.isfile(source):
                    name = get_swig_modulename(source)
                    if name != ext_name[1:]:
                        raise ValueError(
                            'mismatch of extension names: %s provides %r'
                            ' but expected %r' % (source, name, ext_name[1:]))
                    if typ is None:
                        typ = get_swig_target(source)
                        is_cpp = typ=='c++'
                        if is_cpp:
                            target_ext = '.cpp'
                    else:
                        assert typ == get_swig_target(source), repr(typ)
                    target_file = os.path.join(target_dir,'%s_wrap%s' \
                                               % (name, target_ext))
                else:
                    log.debug('  source %s does not exist: skipping swig\'ing.' \
                             % (source))
                    name = ext_name[1:]
                    skip_swig = 1
                    target_file = _find_swig_target(target_dir, name)
                    if not os.path.isfile(target_file):
                        log.debug('  target %s does not exist:\n   '\
                                  'Assuming %s_wrap.{c,cpp} was generated with '\
                                  '"build_src --inplace" command.' \
                                 % (target_file, name))
                        target_dir = os.path.dirname(base)
                        target_file = _find_swig_target(target_dir, name)
                        if not os.path.isfile(target_file):
                            raise ValueError("%r missing" % (target_file,))
                        log.debug('   Yes! Using %s as up-to-date target.' \
                                  % (target_file))
                target_dirs.append(target_dir)
                new_sources.append(target_file)
                py_files.append(os.path.join(py_target_dir, name+'.py'))
                swig_sources.append(source)
                swig_targets[source] = new_sources[-1]
            else:
                new_sources.append(source)

        if not swig_sources:
            return new_sources

        if skip_swig:
            return new_sources + py_files

        map(self.mkpath, target_dirs)
        swig = self.find_swig()
        swig_cmd = [swig, "-python"]
        if is_cpp:
            swig_cmd.append('-c++')
        for d in extension.include_dirs:
            swig_cmd.append('-I'+d)
        for source in swig_sources:
            target = swig_targets[source]
            depends = [source] + extension.depends
            if self.force or newer_group(depends, target, 'newer'):
                log.info("%s: %s" % (os.path.basename(swig) \
                                     + (is_cpp and '++' or ''), source))
                self.spawn(swig_cmd + self.swigflags \
                           + ["-o", target, '-outdir', py_target_dir, source])
            else:
                log.debug("  skipping '%s' swig interface (up-to-date)" \
                         % (source))

        return new_sources + py_files

_f_pyf_ext_match = re.compile(r'.*[.](f90|f95|f77|for|ftn|f|pyf)\Z',re.I).match
_header_ext_match = re.compile(r'.*[.](inc|h|hpp)\Z',re.I).match

#### SWIG related auxiliary functions ####
_swig_module_name_match = re.compile(r'\s*%module\s*(.*\(\s*package\s*=\s*"(?P<package>[\w_]+)".*\)|)\s*(?P<name>[\w_]+)',
                                     re.I).match
_has_c_header = re.compile(r'-[*]-\s*c\s*-[*]-',re.I).search
_has_cpp_header = re.compile(r'-[*]-\s*c[+][+]\s*-[*]-',re.I).search

def get_swig_target(source):
    f = open(source,'r')
    result = 'c'
    line = f.readline()
    if _has_cpp_header(line):
        result = 'c++'
    if _has_c_header(line):
        result = 'c'
    f.close()
    return result

def get_swig_modulename(source):
    f = open(source,'r')
    f_readlines = getattr(f,'xreadlines',f.readlines)
    name = None
    for line in f_readlines():
        m = _swig_module_name_match(line)
        if m:
            name = m.group('name')
            break
    f.close()
    return name

def _find_swig_target(target_dir,name):
    for ext in ['.cpp','.c']:
        target = os.path.join(target_dir,'%s_wrap%s' % (name, ext))
        if os.path.isfile(target):
            break
    return target

#### F2PY related auxiliary functions ####

_f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                re.I).match
_f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'\
                                     '__user__[\w_]*)',re.I).match

def get_f2py_modulename(source):
    name = None
    f = open(source)
    f_readlines = getattr(f,'xreadlines',f.readlines)
    for line in f_readlines():
        m = _f2py_module_name_match(line)
        if m:
            if _f2py_user_module_name_match(line): # skip *__user__* names
                continue
            name = m.group('name')
            break
    f.close()
    return name

##########################################
