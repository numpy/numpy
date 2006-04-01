
from distutils.core import *
try:
    from setuptools import setup as old_setup
    # very old setuptools don't have this
    from setuptools.command import bdist_egg
    # easy_install imports math, it may be picked up from cwd
    from setuptools.command import develop, easy_install
    have_setuptools = 1
except ImportError:
    from distutils.core import setup as old_setup
    have_setuptools = 0

from numpy.distutils.extension import Extension
from numpy.distutils.command import config
from numpy.distutils.command import build
from numpy.distutils.command import build_py
from numpy.distutils.command import config_compiler
from numpy.distutils.command import build_ext
from numpy.distutils.command import build_clib
from numpy.distutils.command import build_src
from numpy.distutils.command import build_scripts
from numpy.distutils.command import sdist
from numpy.distutils.command import install_data
from numpy.distutils.command import install_headers
from numpy.distutils.command import install
from numpy.distutils.command import bdist_rpm
from numpy.distutils.misc_util import get_data_files, is_sequence, is_string

numpy_cmdclass = {'build':            build.build,
                  'build_src':        build_src.build_src,
                  'build_scripts':    build_scripts.build_scripts,
                  'config_fc':        config_compiler.config_fc,
                  'config':           config.config,
                  'build_ext':        build_ext.build_ext,
                  'build_py':         build_py.build_py,
                  'build_clib':       build_clib.build_clib,
                  'sdist':            sdist.sdist,
                  'install_data':     install_data.install_data,
                  'install_headers':  install_headers.install_headers,
                  'install':          install.install,
                  'bdist_rpm':        bdist_rpm.bdist_rpm,
                  }
if have_setuptools:
    from numpy.distutils.command import egg_info
    numpy_cmdclass['bdist_egg'] = bdist_egg.bdist_egg
    numpy_cmdclass['develop'] = develop.develop
    numpy_cmdclass['easy_install'] = easy_install.easy_install
    numpy_cmdclass['egg_info'] = egg_info.egg_info

def _dict_append(d, **kws):
    for k,v in kws.items():
        if not d.has_key(k):
            d[k] = v
            continue
        dv = d[k]
        if isinstance(dv, tuple):
            dv += tuple(v)
            continue
        if isinstance(dv, list):
            dv += list(v)
            continue
        if isinstance(dv, dict):
            _dict_append(dv, **v)
            continue
        if isinstance(dv, str):
            assert isinstance(v,str),`type(v)`
            d[k] = v
        raise TypeError,`type(dv)`
    return

def _command_line_ok(_cache=[]):
    """ Return True if command line does not contain any
    help or display requests.
    """
    if _cache:
        return _cache[0]
    ok = True
    display_opts = ['--'+n for n in Distribution.display_option_names]
    for o in Distribution.display_options:
        if o[1]:
            display_opts.append('-'+o[1])
    for arg in sys.argv:
        if arg.startswith('--help') or arg=='-h' or arg in display_opts:
            ok = False
            break
    _cache.append(ok)
    return ok

def setup(**attr):

    cmdclass = numpy_cmdclass.copy()

    new_attr = attr.copy()
    if new_attr.has_key('cmdclass'):
        cmdclass.update(new_attr['cmdclass'])
    new_attr['cmdclass'] = cmdclass

    if new_attr.has_key('configuration'):
        # To avoid calling configuration if there are any errors
        # or help request in command in the line.
        configuration = new_attr.pop('configuration')

        import distutils.core
        old_dist = distutils.core._setup_distribution
        old_stop = distutils.core._setup_stop_after
        distutils.core._setup_distribution = None
        distutils.core._setup_stop_after = "commandline"
        try:
            dist = setup(**new_attr)
            distutils.core._setup_distribution = old_dist
            distutils.core._setup_stop_after = old_stop
        except Exception,msg:
            distutils.core._setup_distribution = old_dist
            distutils.core._setup_stop_after = old_stop
            raise msg
        if dist.help or not _command_line_ok():
            # probably displayed help, skip running any commands
            return dist

        # create setup dictionary and append to new_attr
        config = configuration()
        if hasattr(config,'todict'): config = config.todict()
        _dict_append(new_attr, **config)

    # Move extension source libraries to libraries
    libraries = []
    for ext in new_attr.get('ext_modules',[]):
        new_libraries = []
        for item in ext.libraries:
            if is_sequence(item):
                lib_name, build_info = item
                _check_append_ext_library(libraries, item)
                new_libraries.append(lib_name)
            elif is_string(item):
                new_libraries.append(item)
            else:
                raise TypeError("invalid description of extension module "
                                "library %r" % (item,))
        ext.libraries = new_libraries
    if libraries:
        if not new_attr.has_key('libraries'):
            new_attr['libraries'] = []
        for item in libraries:
            _check_append_library(new_attr['libraries'], item)

    # sources in ext_modules or libraries may contain header files
    if (new_attr.has_key('ext_modules') or new_attr.has_key('libraries')) \
       and not new_attr.has_key('headers'):
        new_attr['headers'] = []

    return old_setup(**new_attr)

def _check_append_library(libraries, item):
    import warnings
    for libitem in libraries:
        if is_sequence(libitem):
            if is_sequence(item):
                if item[0]==libitem[0]:
                    if item[1] is libitem[1]:
                        return
                    warnings.warn("[0] libraries list contains %r with"
                                  " different build_info" % (item[0],))
                    break
            else:
                if item==libitem[0]:
                    warnings.warn("[1] libraries list contains %r with"
                                  " no build_info" % (item[0],))
                    break
        else:
            if is_sequence(item):
                if item[0]==libitem:
                    warnings.warn("[2] libraries list contains %r with"
                                  " no build_info" % (item[0],))
                    break
            else:
                if item==libitem:
                    return
    libraries.append(item)
    return

def _check_append_ext_library(libraries, (lib_name,build_info)):
    import warnings
    for item in libraries:
        if is_sequence(item):
            if item[0]==lib_name:
                if item[1] is build_info:
                    return
                warnings.warn("[3] libraries list contains %r with"
                              " different build_info" % (lib_name,))
                break
        elif item==lib_name:
            warnings.warn("[4] libraries list contains %r with"
                          " no build_info" % (lib_name,))
            break
    libraries.append((lib_name,build_info))
    return
