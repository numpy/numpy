#!/usr/bin/env python3
"""
Numpy build options can be modified with a site.cfg file.
See site.cfg.example for a template and more information.
"""

import os
from pathlib import Path
import sys
import subprocess
import textwrap
import warnings
import builtins
import re
import tempfile

from distutils.errors import CompileError

# Python supported version checks. Keep right after stdlib imports to ensure we
# get a sensible error for older Python versions
if sys.version_info[:2] < (3, 9):
    raise RuntimeError("Python version >= 3.9 required.")


import versioneer


# This is a bit hackish: we are setting a global variable so that the main
# numpy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__NUMPY_SETUP__ = True

# Needed for backwards code compatibility below and in some CI scripts.
# The version components are changed from ints to strings, but only VERSION
# seems to matter outside of this module and it was already a str.
FULLVERSION = versioneer.get_version()

# Capture the version string:
# 1.22.0.dev0+ ... -> ISRELEASED == False, VERSION == 1.22.0
# 1.22.0rc1+ ... -> ISRELEASED == False, VERSION == 1.22.0
# 1.22.0 ... -> ISRELEASED == True, VERSION == 1.22.0
# 1.22.0rc1 ... -> ISRELEASED == True, VERSION == 1.22.0
ISRELEASED = re.search(r'(dev|\+)', FULLVERSION) is None
_V_MATCH = re.match(r'(\d+)\.(\d+)\.(\d+)', FULLVERSION)
if _V_MATCH is None:
    raise RuntimeError(f'Cannot parse version {FULLVERSION}')
MAJOR, MINOR, MICRO = _V_MATCH.groups()
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)

# The first version not in the `Programming Language :: Python :: ...` classifiers below
if sys.version_info >= (3, 12):
    fmt = "NumPy {} may not yet support Python {}.{}."
    warnings.warn(
        fmt.format(VERSION, *sys.version_info[:2]),
        RuntimeWarning)
    del fmt

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# We need to import setuptools here in order for it to persist in sys.modules.
# Its presence/absence is used in subclassing setup in numpy/distutils/core.py.
# However, we need to run the distutils version of sdist, so import that first
# so that it is in sys.modules
import numpy.distutils.command.sdist
import setuptools
if int(setuptools.__version__.split('.')[0]) >= 60:
    # setuptools >= 60 switches to vendored distutils by default; this
    # may break the numpy build, so make sure the stdlib version is used
    try:
        setuptools_use_distutils = os.environ['SETUPTOOLS_USE_DISTUTILS']
    except KeyError:
        os.environ['SETUPTOOLS_USE_DISTUTILS'] = "stdlib"
    else:
        if setuptools_use_distutils != "stdlib":
            raise RuntimeError("setuptools versions >= '60.0.0' require "
                    "SETUPTOOLS_USE_DISTUTILS=stdlib in the environment")

# Initialize cmdclass from versioneer
from numpy.distutils.core import numpy_cmdclass
cmdclass = versioneer.get_cmdclass(numpy_cmdclass)

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Typing :: Typed
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('numpy')
    config.add_data_files(('numpy', 'LICENSE.txt'))
    config.add_data_files(('numpy', 'numpy/*.pxd'))

    config.get_version('numpy/version.py')  # sets config.version

    return config


def check_submodules():
    """ verify that the submodules are checked out and clean
        use `git submodule update --init`; on failure
    """
    if not os.path.exists('.git'):
        return
    with open('.gitmodules') as f:
        for line in f:
            if 'path' in line:
                p = line.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule {} missing'.format(p))

    proc = subprocess.Popen(['git', 'submodule', 'status'],
                            stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: {}'.format(line))


class concat_license_files():
    """Merge LICENSE.txt and LICENSES_bundled.txt for sdist creation

    Done this way to keep LICENSE.txt in repo as exact BSD 3-clause (see
    gh-13447).  This makes GitHub state correctly how NumPy is licensed.
    """
    def __init__(self):
        self.f1 = 'LICENSE.txt'
        self.f2 = 'LICENSES_bundled.txt'

    def __enter__(self):
        """Concatenate files and remove LICENSES_bundled.txt"""
        with open(self.f1) as f1:
            self.bsd_text = f1.read()

        with open(self.f1, 'a') as f1:
            with open(self.f2) as f2:
                self.bundled_text = f2.read()
                f1.write('\n\n')
                f1.write(self.bundled_text)

    def __exit__(self, exception_type, exception_value, traceback):
        """Restore content of both files"""
        with open(self.f1, 'w') as f:
            f.write(self.bsd_text)


# Need to inherit from versioneer version of sdist to get the encoded
# version information.
class sdist_checked(cmdclass['sdist']):
    """ check submodules on sdist to prevent incomplete tarballs """
    def run(self):
        check_submodules()
        with concat_license_files():
            super().run()


def get_build_overrides():
    """
    Custom build commands to add std flags if required to compilation
    """
    from numpy.distutils.command.build_clib import build_clib
    from numpy.distutils.command.build_ext import build_ext
    from numpy._utils import _pep440

    def try_compile(compiler, file, flags = [], verbose=False):
        bk_ver = getattr(compiler, 'verbose', False)
        compiler.verbose = verbose
        try:
            compiler.compile([file], extra_postargs=flags)
            return True, ''
        except CompileError as e:
            return False, str(e)
        finally:
            compiler.verbose = bk_ver

    def flags_is_required(compiler, is_cpp, flags, code):
        if is_cpp:
            compiler = compiler.cxx_compiler()
            suf = '.cpp'
        else:
            suf = '.c'
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_file = os.path.join(temp_dir, "test" + suf)
            with open(tmp_file, "w+") as f:
                f.write(code)
            # without specify any flags in case of the required
            # standard already supported by default, then there's
            # no need for passing the flags
            comp = try_compile(compiler, tmp_file)
            if not comp[0]:
                comp = try_compile(compiler, tmp_file, flags)
                if not comp[0]:
                    # rerun to verbose the error
                    try_compile(compiler, tmp_file, flags, True)
                    if is_cpp:
                        raise RuntimeError(
                            "Broken toolchain during testing C++ compiler. \n"
                            "A compiler with support for C++17 language "
                            "features is required.\n"
                            f"Triggered the following error: {comp[1]}."
                        )
                    else:
                        raise RuntimeError(
                            "Broken toolchain during testing C compiler. \n"
                            "A compiler with support for C99 language "
                            "features is required.\n"
                            f"Triggered the following error: {comp[1]}."
                        )
                return True
        return False

    def std_cxx_flags(cmd):
        compiler = cmd.compiler
        flags = getattr(compiler, '__np_cache_cpp_flags', None)
        if flags is not None:
            return flags
        flags = dict(
            msvc = ['/std:c++17']
        ).get(compiler.compiler_type, ['-std=c++17'])
        # These flags are used to compile any C++ source within Numpy.
        # They are chosen to have very few runtime dependencies.
        extra_flags = dict(
            # to update #def __cplusplus with enabled C++ version
            msvc = ['/Zc:__cplusplus']
        ).get(compiler.compiler_type, [
            # The following flag is used to avoid emit any extra code
            # from STL since extensions are build by C linker and
            # without C++ runtime dependencies.
            '-fno-threadsafe-statics',
            '-D__STDC_VERSION__=0',  # for compatibility with C headers
            '-fno-exceptions',  # no exception support
            '-fno-rtti'  # no runtime type information
        ])
        if not flags_is_required(compiler, True, flags, textwrap.dedent('''
            #include <type_traits>
            template<typename ...T>
            constexpr bool test_fold = (... && std::is_const_v<T>);
            int main()
            {
                if (test_fold<int, const int>) {
                    return 0;
                }
                else {
                    return -1;
                }
            }
        ''')):
            flags.clear()
        flags += extra_flags
        setattr(compiler, '__np_cache_cpp_flags', flags)
        return flags

    def std_c_flags(cmd):
        compiler = cmd.compiler
        flags = getattr(compiler, '__np_cache_c_flags', None)
        if flags is not None:
            return flags
        flags = dict(
            msvc = []
        ).get(compiler.compiler_type, ['-std=c99'])

        if not flags_is_required(compiler, False, flags, textwrap.dedent('''
            inline static int test_inline() { return 0; }
            int main(void)
            { return test_inline(); }
        ''')):
            flags.clear()

        setattr(compiler, '__np_cache_c_flags', flags)
        return flags

    class new_build_clib(build_clib):
        def build_a_library(self, build_info, lib_name, libraries):
            build_info['extra_cflags'] = std_c_flags(self)
            build_info['extra_cxxflags'] = std_cxx_flags(self)
            build_clib.build_a_library(self, build_info, lib_name, libraries)

    class new_build_ext(build_ext):
        def build_extension(self, ext):
            ext.extra_c_compile_args += std_c_flags(self)
            ext.extra_cxx_compile_args += std_cxx_flags(self)
            build_ext.build_extension(self, ext)
    return new_build_clib, new_build_ext

def generate_cython():
    # Check Cython version
    from numpy._utils import _pep440
    try:
        # try the cython in the installed python first (somewhat related to
        # scipy/scipy#2397)
        import Cython
        from Cython.Compiler.Version import version as cython_version
    except ImportError as e:
        # The `cython` command need not point to the version installed in the
        # Python running this script, so raise an error to avoid the chance of
        # using the wrong version of Cython.
        msg = 'Cython needs to be installed in Python as a module'
        raise OSError(msg) from e
    else:
        # Note: keep in sync with that in pyproject.toml
        # Update for Python 3.11
        required_version = '0.29.30'

        if _pep440.parse(cython_version) < _pep440.Version(required_version):
            cython_path = Cython.__file__
            msg = 'Building NumPy requires Cython >= {}, found {} at {}'
            msg = msg.format(required_version, cython_version, cython_path)
            raise RuntimeError(msg)

    # Process files
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    for d in ('random',):
        p = subprocess.call([sys.executable,
                             os.path.join(cwd, 'tools', 'cythonize.py'),
                             'numpy/{0}'.format(d)],
                            cwd=cwd)
        if p != 0:
            raise RuntimeError("Running cythonize failed!")


def parse_setuppy_commands():
    """Check the commands and respond appropriately.  Disable broken commands.

    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).
    """
    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes',
                     'version',]

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg', 'build_src',
                     'bdist_egg')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        print(textwrap.dedent("""
            Note: if you need reliable uninstall behavior, then install
            with pip instead of using `setup.py install`:

              - `pip install .`       (from a git repo or downloaded source
                                       release)
              - `pip install numpy`   (last NumPy release on PyPI)

            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent("""
            NumPy-specific help
            -------------------

            To install NumPy from here with reliable uninstall, we recommend
            that you use `pip install .`. To install the latest NumPy release
            from PyPI, use `pip install numpy`.

            For help with build/installation issues, please ask on the
            numpy-discussion mailing list.  If you are sure that you have run
            into a bug, please report it at https://github.com/numpy/numpy/issues.

            Setuptools commands help
            ------------------------
            """))
        return False

    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = dict(
        test="""
            `setup.py test` is not supported.  Use one of the following
            instead:

              - `python runtests.py`              (to build and test)
              - `python runtests.py --no-build`   (to test installed numpy)
              - `>>> numpy.test()`           (run tests for installed numpy
                                              from within an interpreter)
            """,
        upload="""
            `setup.py upload` is not supported, because it's insecure.
            Instead, build what you want to upload and upload those files
            with `twine upload -s <filenames>` instead.
            """,
        clean="""
            `setup.py clean` is not supported, use one of the following instead:

              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
        build_sphinx="""
            `setup.py build_sphinx` is not supported, use the
            Makefile under doc/""",
        flake8="`setup.py flake8` is not supported, use flake8 standalone",
        )
    bad_commands['nosetests'] = bad_commands['test']
    for command in ('upload_docs', 'easy_install', 'bdist', 'bdist_dumb',
                    'register', 'check', 'install_data', 'install_headers',
                    'install_lib', 'install_scripts', ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate', 'dist_info']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    raise RuntimeError("Unrecognized setuptools command: {}".format(args))


def get_docs_url():
    if 'dev' in VERSION:
        return "https://numpy.org/devdocs"
    else:
        # For releases, this URL ends up on PyPI.
        # By pinning the version, users looking at old PyPI releases can get
        # to the associated docs easily.
        return "https://numpy.org/doc/{}.{}".format(MAJOR, MINOR)


def setup_package():
    src_path = os.path.dirname(os.path.abspath(__file__))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # The f2py scripts that will be installed
    if sys.platform == 'win32':
        f2py_cmds = [
            'f2py = numpy.f2py.f2py2e:main',
            ]
    else:
        f2py_cmds = [
            'f2py = numpy.f2py.f2py2e:main',
            'f2py%s = numpy.f2py.f2py2e:main' % sys.version_info[:1],
            'f2py%s.%s = numpy.f2py.f2py2e:main' % sys.version_info[:2],
            ]

    cmdclass["sdist"] = sdist_checked
    metadata = dict(
        name='numpy',
        maintainer="NumPy Developers",
        maintainer_email="numpy-discussion@python.org",
        description="Fundamental package for array computing in Python",
        long_description=Path("README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://www.numpy.org",
        author="Travis E. Oliphant et al.",
        download_url="https://pypi.python.org/pypi/numpy",
        project_urls={
            "Bug Tracker": "https://github.com/numpy/numpy/issues",
            "Documentation": get_docs_url(),
            "Source Code": "https://github.com/numpy/numpy",
        },
        license='BSD-3-Clause',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        test_suite='pytest',
        version=versioneer.get_version(),
        cmdclass=cmdclass,
        python_requires='>=3.9',
        zip_safe=False,
        entry_points={
            'console_scripts': f2py_cmds,
            'array_api': ['numpy = numpy.array_api'],
            'pyinstaller40': ['hook-dirs = numpy:_pyinstaller_hooks_dir'],
        },
    )

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove('--force')
    else:
        # Raise errors for unsupported commands, improve help output, etc.
        run_build = parse_setuppy_commands()

    if run_build:
        # patches distutils, even though we don't use it
        #from setuptools import setup
        from numpy.distutils.core import setup

        if 'sdist' not in sys.argv:
            # Generate Cython sources, unless we're generating an sdist
            generate_cython()

        metadata['configuration'] = configuration
        # Customize extension building
        cmdclass['build_clib'], cmdclass['build_ext'] = get_build_overrides()
    else:
        #from numpy.distutils.core import setup
        from setuptools import setup
        # workaround for broken --no-build-isolation with newer setuptools,
        # see gh-21288
        metadata["packages"] = []

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return


if __name__ == '__main__':
    setup_package()
    # This may avoid problems where numpy is installed via ``*_requires`` by
    # setuptools, the global namespace isn't reset properly, and then numpy is
    # imported later (which will then fail to load numpy extension modules).
    # See gh-7956 for details
    del builtins.__NUMPY_SETUP__
