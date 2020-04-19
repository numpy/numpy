#!/usr/bin/env python
""" NumPy is the fundamental package for array computing with Python.

It provides:

- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities
- and much more

Besides its obvious scientific uses, NumPy can also be used as an efficient
multi-dimensional container of generic data. Arbitrary data-types can be
defined. This allows NumPy to seamlessly and speedily integrate with a wide
variety of databases.

All NumPy wheels distributed on PyPI are BSD licensed.

"""
from __future__ import division, print_function

DOCLINES = (__doc__ or '').split("\n")

import os
import sys
import subprocess
import textwrap
import sysconfig


if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

import builtins


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR               = 1
MINOR               = 18
MICRO               = 3
ISRELEASED          = True
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except (subprocess.SubprocessError, OSError):
        GIT_REVISION = "Unknown"

    if not GIT_REVISION:
        # this shouldn't happen but apparently can (see gh-8512)
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# This is a bit hackish: we are setting a global variable so that the main
# numpy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__NUMPY_SETUP__ = True


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('numpy/version.py'):
        # must be a source distribution, use existing version file
        try:
            from numpy.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "numpy/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='numpy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM NUMPY SETUP.PY
#
# To compare versions robustly, use `numpy.lib.NumpyVersion`
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('numpy')
    config.add_data_files(('numpy', 'LICENSE.txt'))

    config.get_version('numpy/version.py') # sets config.version

    return config


def check_submodules():
    """ verify that the submodules are checked out and clean
        use `git submodule update --init`; on failure
    """
    if not os.path.exists('.git'):
        return
    with open('.gitmodules') as f:
        for l in f:
            if 'path' in l:
                p = l.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule %s missing' % p)


    proc = subprocess.Popen(['git', 'submodule', 'status'],
                            stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: %s' % line)


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
        with open(self.f1, 'r') as f1:
            self.bsd_text = f1.read()

        with open(self.f1, 'a') as f1:
            with open(self.f2, 'r') as f2:
                self.bundled_text = f2.read()
                f1.write('\n\n')
                f1.write(self.bundled_text)

    def __exit__(self, exception_type, exception_value, traceback):
        """Restore content of both files"""
        with open(self.f1, 'w') as f:
            f.write(self.bsd_text)


from distutils.command.sdist import sdist
class sdist_checked(sdist):
    """ check submodules on sdist to prevent incomplete tarballs """
    def run(self):
        check_submodules()
        with concat_license_files():
            sdist.run(self)


def get_build_overrides():
    """
    Custom build commands to add `-std=c99` to compilation
    """
    from numpy.distutils.command.build_clib import build_clib
    from numpy.distutils.command.build_ext import build_ext

    def _is_using_gcc(obj):
        is_gcc = False
        if obj.compiler.compiler_type == 'unix':
            cc = sysconfig.get_config_var("CC")
            if not cc:
                cc = ""
            compiler_name = os.path.basename(cc)
            is_gcc = "gcc" in compiler_name
        return is_gcc

    class new_build_clib(build_clib):
        def build_a_library(self, build_info, lib_name, libraries):
            if _is_using_gcc(self):
                args = build_info.get('extra_compiler_args') or []
                args.append('-std=c99')
                build_info['extra_compiler_args'] = args
            build_clib.build_a_library(self, build_info, lib_name, libraries)

    class new_build_ext(build_ext):
        def build_extension(self, ext):
            if _is_using_gcc(self):
                if '-std=c99' not in ext.extra_compile_args:
                    ext.extra_compile_args.append('-std=c99')
            build_ext.build_extension(self, ext)
    return new_build_clib, new_build_ext


def generate_cython():
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
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg', 'build_src')

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
              - `pip install numpy`   (last NumPy release on PyPi)

            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent("""
            NumPy-specific help
            -------------------

            To install NumPy from here with reliable uninstall, we recommend
            that you use `pip install .`. To install the latest NumPy release
            from PyPi, use `pip install numpy`.

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
        upload_docs="`setup.py upload_docs` is not supported",
        easy_install="`setup.py easy_install` is not supported",
        clean="""
            `setup.py clean` is not supported, use one of the following instead:

              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
        check="`setup.py check` is not supported",
        register="`setup.py register` is not supported",
        bdist_dumb="`setup.py bdist_dumb` is not supported",
        bdist="`setup.py bdist` is not supported",
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
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    import warnings
    warnings.warn("Unrecognized setuptools command, proceeding with "
                  "generating Cython sources and expanding templates", stacklevel=2)
    return True


def setup_package():
    src_path = os.path.dirname(os.path.abspath(__file__))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # Rewrite the version file everytime
    write_version_py()

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

    cmdclass={"sdist": sdist_checked,
             }
    metadata = dict(
        name = 'numpy',
        maintainer = "NumPy Developers",
        maintainer_email = "numpy-discussion@python.org",
        description = DOCLINES[0],
        long_description = "\n".join(DOCLINES[2:]),
        url = "https://www.numpy.org",
        author = "Travis E. Oliphant et al.",
        download_url = "https://pypi.python.org/pypi/numpy",
        project_urls={
            "Bug Tracker": "https://github.com/numpy/numpy/issues",
            "Documentation": "https://docs.scipy.org/doc/numpy/",
            "Source Code": "https://github.com/numpy/numpy",
        },
        license = 'BSD',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        test_suite='nose.collector',
        cmdclass=cmdclass,
        python_requires='>=3.5',
        zip_safe=False,
        entry_points={
            'console_scripts': f2py_cmds
        },
    )

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove('--force')
    else:
        # Raise errors for unsupported commands, improve help output, etc.
        run_build = parse_setuppy_commands()

    from setuptools import setup
    if run_build:
        from numpy.distutils.core import setup
        cwd = os.path.abspath(os.path.dirname(__file__))
        if not 'sdist' in sys.argv:
            # Generate Cython sources, unless we're generating an sdist
            generate_cython()

        metadata['configuration'] = configuration
        # Customize extension building
        cmdclass['build_clib'], cmdclass['build_ext'] = get_build_overrides()
    else:
        # Version number is added to metadata inside configuration() if build
        # is run.
        metadata['version'] = get_version_info()[0]

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
