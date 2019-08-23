from __future__ import division, absolute_import, print_function
import os
import sys
import textwrap
import platform
try:
    from urllib.request import urlopen
    from urllib.error import HTTPError
except:
    #Python2
    from urllib2 import urlopen, HTTPError

from tempfile import mkstemp, gettempdir
import zipfile
import tarfile

OPENBLAS_V = 'v0.3.5'
OPENBLAS_LONG = 'v0.3.5-274-g6a8b4269'
BASE_LOC = ''
RACKSPACE = 'https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com'
ARCHITECTURES = ['', 'windows', 'darwin', 'arm', 'x86', 'ppc64']

IS_32BIT = sys.maxsize < 2**32
def get_arch():
    if platform.system() == 'Windows':
        ret = 'windows'
    elif platform.system() == 'Darwin':
        ret = 'darwin'
    # Python3 returns a named tuple, but Python2 does not, so we are stuck
    elif 'arm' in os.uname()[-1]:
        ret = 'arm';
    elif 'aarch64' in os.uname()[-1]:
        ret = 'arm';
    elif 'x86' in os.uname()[-1]:
        ret = 'x86'
    elif 'ppc64' in os.uname()[-1]:
        ret = 'ppc64'
    else:
        ret = ''
    assert ret in ARCHITECTURES
    return ret

def download_openblas(target, arch):
    filename = ''
    if arch == 'arm':
        # ARMv8 OpenBLAS built using script available here:
        # https://github.com/tylerjereddy/openblas-static-gcc/tree/master/ARMv8
        # build done on GCC compile farm machine named gcc115
        # tarball uploaded manually to an unshared Dropbox location
        filename = ('https://www.dropbox.com/s/pbqkxzlmih4cky1/'
                    'openblas-{}-armv8.tar.gz?dl=1'.format(OPENBLAS_V))
        typ = 'tar.gz'
    elif arch == 'ppc64':
        # build script for POWER8 OpenBLAS available here:
        # https://github.com/tylerjereddy/openblas-static-gcc/blob/master/power8
        # built on GCC compile farm machine named gcc112
        # manually uploaded tarball to an unshared Dropbox location
        filename = ('https://www.dropbox.com/s/zcwhk7c2zptwy0s/'
                    'openblas-{}-ppc64le-power8.tar.gz?dl=1'.format(OPENBLAS_V))
        typ = 'tar.gz'
    elif arch == 'darwin':
        filename = '{0}/openblas-{1}-macosx_10_9_x86_64-gf_1becaaa.tar.gz'.format(
                        RACKSPACE, OPENBLAS_LONG)
        typ = 'tar.gz'
    elif arch == 'windows':
        if IS_32BIT:
            suffix = 'win32-gcc_7_1_0.zip'
        else:
            suffix = 'win_amd64-gcc_7_1_0.zip'
        filename = '{0}/openblas-{1}-{2}'.format(RACKSPACE, OPENBLAS_LONG, suffix)
        typ = 'zip'
    elif arch == 'x86':
        if IS_32BIT:
            suffix = 'manylinux1_i686.tar.gz'
        else:
            suffix = 'manylinux1_x86_64.tar.gz'
        filename = '{0}/openblas-{1}-{2}'.format(RACKSPACE, OPENBLAS_LONG, suffix)
        typ = 'tar.gz'
    if not filename:
        return None
    try:
        with open(target, 'wb') as fid:
            fid.write(urlopen(filename).read())
    except HTTPError:
        print('Could not download "%s"' % filename)
        return None
    return typ

def setup_openblas(arch=get_arch()):
    '''
    Download and setup an openblas library for building. If successful,
    the configuration script will find it automatically.

    Returns
    -------
    msg : str
        path to extracted files on success, otherwise indicates what went wrong
        To determine success, do ``os.path.exists(msg)``
    '''
    _, tmp = mkstemp()
    if not arch:
        raise ValueError('unknown architecture')
    typ = download_openblas(tmp, arch)
    if not typ:
        return ''
    if arch == 'windows':
        if not typ == 'zip':
            return 'expecting to download zipfile on windows, not %s' % str(typ)
        return unpack_windows_zip(tmp)
    else:
        if not typ == 'tar.gz':
            return 'expecting to download tar.gz, not %s' % str(typ)
        return unpack_targz(tmp)

def unpack_windows_zip(fname):
    import sysconfig
    with zipfile.ZipFile(fname, 'r') as zf:
        # Get the openblas.a file, but not openblas.dll.a nor openblas.dev.a
        lib = [x for x in zf.namelist() if OPENBLAS_LONG in x and
                  x.endswith('a') and not x.endswith('dll.a') and
                  not x.endswith('dev.a')]
        if not lib:
            return 'could not find libopenblas_%s*.a ' \
                    'in downloaded zipfile' % OPENBLAS_LONG
        target = os.path.join(gettempdir(), 'openblas.a')
        with open(target, 'wb') as fid:
            fid.write(zf.read(lib[0]))
    return target

def unpack_targz(fname):
    target = os.path.join(gettempdir(), 'openblas')
    if not os.path.exists(target):
        os.mkdir(target)
    with tarfile.open(fname, 'r') as zf:
        # TODO: check that all the zf.getnames() files do not escape the
        # extract directory (no leading '../', '/')
        zf.extractall(target)
    return target

def make_init(dirname):
    '''
    Create a _distributor_init.py file for OpenBlas
    '''
    with open(os.path.join(dirname, '_distributor_init.py'), 'wt') as fid:
        fid.write(textwrap.dedent("""
            '''
            Helper to preload windows dlls to prevent dll not found errors.
            Once a DLL is preloaded, its namespace is made available to any
            subsequent DLL. This file originated in the numpy-wheels repo,
            and is created as part of the scripts that build the wheel.
            '''
            import os
            from ctypes import WinDLL
            import glob
            if os.name == 'nt':
                # convention for storing / loading the DLL from
                # numpy/.libs/, if present
                try:
                    basedir = os.path.dirname(__file__)
                except:
                    pass
                else:
                    libs_dir = os.path.abspath(os.path.join(basedir, '.libs'))
                    DLL_filenames = []
                    if os.path.isdir(libs_dir):
                        for filename in glob.glob(os.path.join(libs_dir,
                                                             '*openblas*dll')):
                            # NOTE: would it change behavior to load ALL
                            # DLLs at this path vs. the name restriction?
                            WinDLL(os.path.abspath(filename))
                            DLL_filenames.append(filename)
                if len(DLL_filenames) > 1:
                    import warnings
                    warnings.warn("loaded more than 1 DLL from .libs:\\n%s" %
                              "\\n".join(DLL_filenames),
                              stacklevel=1)
    """))

def test_setup(arches):
    '''
    Make sure all the downloadable files exist and can be opened
    '''
    for arch in arches:
        if arch == '':
            continue
        try:
            target = setup_openblas(arch)
        except:
            print('Could not setup %s' % arch)
            raise
        if not target:
            raise RuntimeError('Could not setup %s' % arch)
        print(target)

def test_version(expected_version):
    """
    Assert that expected OpenBLAS version is
    actually available via NumPy
    """
    import numpy
    import ctypes

    dll = ctypes.CDLL(numpy.core._multiarray_umath.__file__)
    get_config = dll.openblas_get_config
    get_config.restype=ctypes.c_char_p
    res = get_config()
    print('OpenBLAS get_config returned', str(res))
    check_str = b'OpenBLAS %s' % expected_version[0].encode()
    assert check_str in res

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Download and expand an OpenBLAS archive for this ' \
                    'architecture')
    parser.add_argument('--test', nargs='*', default=None,
        help='Test different architectures. "all", or any of %s' % ARCHITECTURES)
    parser.add_argument('--check_version', nargs=1, default=None,
        help='Check provided OpenBLAS version string against available OpenBLAS')
    args = parser.parse_args()
    if args.check_version is not None:
        test_version(args.check_version)
    elif args.test is None:
        print(setup_openblas())
    else:
        if len(args.test) == 0 or 'all' in args.test:
            test_setup(ARCHITECTURES)
        else:
            test_setup(args.test)
