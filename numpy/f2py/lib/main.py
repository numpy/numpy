"""
Tools for building F2PY generated extension modules.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

import os
import re
import sys
import tempfile

try:
    from numpy import __version__ as numpy_version
except ImportError:
    numpy_version = 'N/A'

__all__ = ['main', 'compile']

__usage__ = """
F2PY G3 --- The third generation of Fortran to Python Interface Generator
=========================================================================

Description
-----------

f2py program generates a Python C/API file (<modulename>module.c) that
contains wrappers for given Fortran functions and data so that they
can be accessed from Python. With the -c option the corresponding
extension modules are built.

Options
-------

  --g3-numpy       Use numpy.f2py.lib tool, the 3rd generation of F2PY,
                   with NumPy support.
  --2d-numpy       Use numpy.f2py tool with NumPy support. [DEFAULT]
  --2d-numeric     Use f2py2e tool with Numeric support.
  --2d-numarray    Use f2py2e tool with Numarray support.

  -m <modulename>  Name of the module; f2py generates a Python/C API
                   file <modulename>module.c or extension module <modulename>.
                   For wrapping Fortran 90 modules, f2py will use Fortran
                   module names.
  --parse          Parse Fortran files and print result to stdout.


Options effective only with -h
------------------------------

  -h <filename>    Write signatures of the fortran routines to file <filename>
                   and exit. You can then edit <filename> and use it instead
                   of <fortran files> for generating extension module source.
                   If <filename> is stdout or stderr then the signatures are
                   printed to the corresponding stream.

  --overwrite-signature  Overwrite existing signature file.

Options effective only with -c
------------------------------

  -c               Compile fortran sources and build extension module.

  --build-dir <dirname>  All f2py generated files are created in <dirname>.
                   Default is tempfile.mktemp() and it will be removed after
                   f2py stops unless <dirname> is specified via --build-dir
                   option.

numpy.distutils options effective only with -c
----------------------------------------------

  --fcompiler=<name>      Specify Fortran compiler type by vendor



Extra options effective only with -c
------------------------------------

  -L/path/to/lib/ -l<libname>
  -D<name[=define]> -U<name>
  -I/path/to/include/
  <filename>.o <filename>.(so|dynlib|dll) <filename>.a

  Using the following macros may be required with non-gcc Fortran
  compilers:
    -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN
    -DUNDERSCORE_G77

  -DF2PY_DEBUG_PYOBJ_TOFROM  --- pyobj_(to|from)_<ctype> functions will
  print debugging messages to stderr.

"""

import re
import shutil
import parser.api
from parser.api import parse, PythonModule, EndStatement, Module, Subroutine, Function,\
     get_reader

def get_values(sys_argv, prefix='', suffix='', strip_prefix=False, strip_suffix=False):
    """
    Return a list of values with pattern
      <prefix><value><suffix>.
    The corresponding items will be removed from sys_argv.
    """
    match = re.compile(prefix + r'.*' + suffix + '\Z').match
    ret = [item for item in sys_argv if match(item)]
    [sys_argv.remove(item) for item in ret]
    if strip_prefix and prefix:
        i = len(prefix)
        ret = [item[i:] for item in ret]
    if strip_suffix and suffix:
        i = len(suffix)
        ret = [item[:-i] for item in ret]
    return ret

def get_option(sys_argv, option, default_return = None):
    """
    Return True if sys_argv has <option>.
    If <option> is not in sys_argv, return default_return.
    <option> (when present) will be removed from sys_argv.
    """
    try:
        i = sys_argv.index(option)
    except ValueError:
        return default_return
    del sys_argv[i]
    return True

def get_option_value(sys_argv, option, default_value = None, default_return = None):
    """
    Return <value> from
      sys_argv = [...,<option>,<value>,...]
    list.
    If <option> is the last element, return default_value.
    If <option> is not in sys_argv, return default_return.
    Both <option> and <value> (when present) will be removed from sys_argv.
    """
    try:
        i = sys_argv.index(option)
    except ValueError:
        return default_return
    if len(sys_argv)-1==i:
        del sys_argv[i]
        return default_value
    value = sys_argv[i+1]
    del sys_argv[i+1]
    del sys_argv[i]
    return value

def get_signature_output(sys_argv):
    return get_option_value(sys_argv,'-h','stdout')


def parse_files(sys_argv):
    flag = 'file'
    file_names = []
    only_names = []
    skip_names = []
    options = []
    for word in sys_argv:
        if word=='': pass
        elif word=='only:': flag = 'only'
        elif word=='skip:': flag = 'skip'
        elif word==':': flag = 'file'
        elif word.startswith('--'): options.append(word)
        else:
            {'file': file_names,'only': only_names, 'skip': skip_names}[flag].append(word)

    if options:
        sys.stderr.write('Unused options: %s\n' % (', '.join(options)))
    for filename in file_names:
        if not os.path.isfile(filename):
            sys.stderr.write('No or not a file %r. Skipping.\n' % (filename))
            continue
        sys.stderr.write('Parsing %r..\n' % (filename))
        reader = parser.api.get_reader(filename)
        print parser.api.Fortran2003.Program(reader)
    return

def dump_signature(sys_argv):
    """ Read Fortran files and dump the signatures to file or stdout.
    XXX: Not well tested.
    """
    signature_output = get_signature_output(sys_argv)

    # initialize output stream
    if signature_output in ['stdout','stderr']:
        output_stream = getattr(sys, signature_output)
        modulename = get_option_value(sys_argv,'-m','untitled','unknown')
    else:
        name,ext = os.path.splitext(signature_output)
        if ext != '.pyf':
            signature_output += '.pyf'
        if os.path.isfile(signature_output):
            overwrite = get_option(sys_argv, '--overwrite-signature', False)
            if not overwrite:
                print >> sys.stderr, 'Signature file %r exists. '\
                      'Use --overwrite-signature to overwrite.' % (signature_output)
                sys.exit()
        modulename = get_option_value(sys_argv,'-m',os.path.basename(name),
                                      os.path.basename(name))
        output_stream = open(signature_output,'w')

    flag = 'file'
    file_names = []
    only_names = []
    skip_names = []
    options = []
    for word in sys_argv:
        if word=='': pass
        elif word=='only:': flag = 'only'
        elif word=='skip:': flag = 'skip'
        elif word==':': flag = 'file'
        elif word.startswith('--'): options.append(word)
        else:
            {'file': file_names,'only': only_names,
             'skip': skip_names}[flag].append(word)

    if options:
        sys.stderr.write('Unused options: %s\n' % (', '.join(options)))

    output_stream.write('''!    -*- f90 -*-
! Note: the context of this file is case sensitive.
''')
    output_stream.write('PYTHON MODULE %s\n' % (modulename))
    output_stream.write('  INTERFACE\n\n')
    for filename in file_names:
        if not os.path.isfile(filename):
            sys.stderr.write('No or not a file %r. Skipping.\n' % (filename))
            continue
        sys.stderr.write('Parsing %r..\n' % (filename))
        block = parse(filename)
        if block is None:
            sys.exit(1)
        output_stream.write('! File: %s, source mode = %r\n' % (filename, block.reader.mode))
        if block.content and isinstance(block.content[0],PythonModule):
            for subblock in block.content[0].content[0].content:
                if isinstance(subblock, EndStatement):
                    break
                output_stream.write(subblock.topyf('    ')+'\n')
        else:
            output_stream.write(block.topyf('    ')+'\n')
    output_stream.write('  END INTERFACE\n')
    output_stream.write('END PYTHON MODULE %s\n' % (modulename))

    if signature_output not in ['stdout','stderr']:
        output_stream.close()
    return

def construct_extension_sources(modulename, parse_files, include_dirs, build_dir):
    """
    Construct wrapper sources.
    """
    from py_wrap import PythonWrapperModule

    f90_modules = []
    external_subprograms = []
    for filename in parse_files:
        if not os.path.isfile(filename):
            sys.stderr.write('No or not a file %r. Skipping.\n' % (filename))
            continue
        sys.stderr.write('Parsing %r..\n' % (filename))
        for block in parse(filename, include_dirs=include_dirs).content:
            if isinstance(block, Module):
                f90_modules.append(block)
            elif isinstance(block, (Subroutine, Function)):
                external_subprograms.append(block)
            else:
                sys.stderr.write("Unhandled structure: %r\n" % (block.__class__))

    module_infos = []

    for block in f90_modules:
        wrapper = PythonWrapperModule(block.name)
        wrapper.add(block)
        c_code = wrapper.c_code()
        f_code = '! -*- f90 -*-\n' + wrapper.fortran_code()
        c_fn = os.path.join(build_dir,'%smodule.c' % (block.name))
        f_fn = os.path.join(build_dir,'%s_f_wrappers_f2py.f90' % (block.name))
        f = open(c_fn,'w')
        f.write(c_code)
        f.close()
        f = open(f_fn,'w')
        f.write(f_code)
        f.close()
        #f_lib = '%s_f_wrappers_f2py' % (block.name)
        module_info = {'name':block.name, 'c_sources':[c_fn],
                       'f_sources':[f_fn], 'language':'f90'}
        module_infos.append(module_info)

    if external_subprograms:
        wrapper = PythonWrapperModule(modulename)
        for block in external_subprograms:
            wrapper.add(block)
        c_code = wrapper.c_code()
        f_code = wrapper.fortran_code()
        c_fn = os.path.join(build_dir,'%smodule.c' % (modulename))
        ext = '.f'
        language = 'f77'
        if wrapper.isf90:
            f_code = '! -*- f90 -*-\n' + f_code
            ext = '.f90'
            language = 'f90'
        f_fn = os.path.join(build_dir,'%s_f_wrappers_f2py%s' % (modulename, ext))
        f = open(c_fn,'w')
        f.write(c_code)
        f.close()
        f = open(f_fn,'w')
        f.write(f_code)
        f.close()
        module_info = {'name':modulename, 'c_sources':[c_fn],
                       'f_sources':[f_fn], 'language':language}
        module_infos.append(module_info)

    return module_infos

def build_extension(sys_argv, sources_only = False):
    """
    Build wrappers to Fortran 90 modules and external subprograms.
    """
    modulename = get_option_value(sys_argv,'-m','untitled','unspecified')

    if sources_only:
        build_dir = get_option_value(sys_argv,'--build-dir','.','')
    else:
        build_dir = get_option_value(sys_argv,'--build-dir','.',None)
    if build_dir is None:
        build_dir = tempfile.mktemp()
        clean_build_dir = True
    else:
        clean_build_dir = False
    if build_dir and not os.path.exists(build_dir): os.makedirs(build_dir)

    include_dirs = get_values(sys_argv,'-I',strip_prefix=True)
    library_dirs = get_values(sys_argv,'-L',strip_prefix=True)
    libraries = get_values(sys_argv,'-l',strip_prefix=True)
    _define_macros = get_values(sys_argv,'-D',strip_prefix=True)
    undef_macros = get_values(sys_argv,'-U',strip_prefix=True)
    extra_objects = get_values(sys_argv,'','[.](o|a|so|dll|dylib|sl)')

    define_macros = []
    for item in _define_macros:
        name_value = item.split('=',1)
        if len(name_value)==1:
            name_value.append(None)
        if len(name_value)==2:
            define_macros.append(tuple(name_value))
        else:
            print 'Invalid use of -D:',name_value

    pyf_files = get_values(sys_argv,'','[.]pyf')
    fortran_files = get_values(sys_argv,'','[.](f|f90|F90|F)')
    c_files = get_values(sys_argv,'','[.](c|cpp|C|CPP|c[+][+])')

    fc_flags = get_values(sys_argv,'--fcompiler=')

    options = get_values(sys_argv,'-')
    if options:
        sys.stderr.write('Unused options: %s\n' % (', '.join(options)))

    if pyf_files:
        parse_files = pyf_files
    else:
        parse_files = fortran_files + c_files

    module_infos = construct_extension_sources(modulename, parse_files, include_dirs, build_dir)

    if sources_only:
        return

    def configuration(parent_package='', top_path=None or ''):
        from numpy.distutils.misc_util import Configuration
        config = Configuration('',parent_package,top_path)
        flibname = modulename + '_fortran_f2py'
        if fortran_files:
            config.add_library(flibname,
                               sources = fortran_files)
            libraries.insert(0,flibname)

        for module_info in module_infos:
            name = module_info['name']
            c_sources = module_info['c_sources']
            f_sources = module_info['f_sources']
            language = module_info['language']
            if f_sources:
                f_lib = '%s_f_wrappers_f2py' % (name)
                config.add_library(f_lib, sources = f_sources)
                libs = [f_lib] + libraries
            else:
                libs = libraries
            config.add_extension(name,
                                 sources=c_sources + c_files,
                                 libraries = libs,
                                 define_macros = define_macros,
                                 undef_macros = undef_macros,
                                 include_dirs = include_dirs,
                                 extra_objects = extra_objects,
                                 language = language,
                                 )
        return config

    old_sys_argv = sys.argv[:]
    build_dir_ext_temp = os.path.join(build_dir,'ext_temp')
    build_dir_clib_temp = os.path.join(build_dir,'clib_temp')
    build_dir_clib_clib = os.path.join(build_dir,'clib_clib')
    new_sys_argv = [sys.argv[0]] + ['build_ext',
                                    '--build-temp',build_dir_ext_temp,
                                    '--build-lib',build_dir,
                                    'build_clib',
                                    '--build-temp',build_dir_clib_temp,
                                    '--build-clib',build_dir_clib_clib,
                                    ]
    temp_dirs = [build_dir_ext_temp, build_dir_clib_temp, build_dir_clib_clib]
        
    if fc_flags:
        new_sys_argv += ['config_fc'] + fc_flags
    sys.argv[:] = new_sys_argv

    sys.stderr.write('setup arguments: %r\n' % (' '.join(sys.argv)))

    from numpy.distutils.core import setup
    setup(configuration=configuration)

    sys.argv[:] = old_sys_argv

    if 1 or clean_build_dir:
        for d in temp_dirs:
            if os.path.exists(d):
                sys.stderr.write('Removing build directory %s\n'%(d))
                shutil.rmtree(d)
    return

def main(sys_argv = None):
    """ Main function of f2py script.
    """
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    if '--help-link' in sys_argv:
        sys_argv.remove('--help-link')
        from numpy.distutils.system_info import show_all
        show_all()
        return
    if '-c' in sys_argv:
        sys_argv.remove('-c')
        build_extension(sys_argv)
        return
    if '--parse' in sys_argv:
        sys_argv.remove('--parse')
        parse_files(sys_argv)
        return
    if '-h' in sys_argv:
        dump_signature(sys_argv)
        return
    if not sys_argv or '--help' in sys_argv:
        print >> sys.stdout, __usage__

    build_extension(sys_argv, sources_only = True)
    return

def compile(source,
            jobname = 'untitled',
            extra_args = [],
            source_ext = None,
            modulenames = None
            ):
    """
    Build extension module from processing source with f2py.

    jobname - the name of compile job. For non-module source
              this will be also the name of extension module.
    modulenames - the list of extension module names that
              the given compilation job should create.
    extra_args - a list of extra arguments for numpy style
              setup.py command line.
    source_ext - extension of the Fortran source file: .f90 or .f

    Extension modules are saved to current working directory.
    Returns a list of module objects according to modulenames
    input.
    """
    from nary import encode
    tempdir = tempfile.gettempdir()
    s = 'f2pyjob_%s_%s' % (jobname, encode(source))
    tmpdir = os.path.join(tempdir, s)
    if source_ext is None:
        reader = get_reader(source)
        source_ext = {'free90':'.f90','fix90':'.f90','fix77':'.f','pyf':'.pyf'}[reader.mode]
        
    if modulenames is None:
        modulenames = jobname,
    if os.path.isdir(tmpdir):    
        try:
            sys.path.insert(0, tmpdir)
            modules = []
            for modulename in modulenames:
                exec('import %s as m' % (modulename))
                modules.append(m)
            sys.path.pop(0)
            return modules
        except ImportError:
            pass
        finally:
            sys.path.pop(0)
    else:
        os.mkdir(tmpdir)

    fname = os.path.join(tmpdir,'%s_src%s' % (jobname, source_ext))

    f = open(fname,'w')
    f.write(source)
    f.close()

    sys_argv = []
    sys_argv.extend(['--build-dir',tmpdir])
    #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
    sys_argv.extend(['-m',jobname, fname])

    build_extension(sys_argv + extra_args)

    sys.path.insert(0, tmpdir)
    modules = []
    for modulename in modulenames:
        exec('import %s as m' % (modulename))
        modules.append(m)
    sys.path.pop(0)
    return modules

#EOF
