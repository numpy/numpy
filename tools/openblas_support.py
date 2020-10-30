import glob
import hashlib
import os
import platform
import sys
import shutil
import tarfile
import textwrap
import zipfile

from tempfile import mkstemp, gettempdir
from urllib.request import urlopen, Request

OPENBLAS_V = '0.3.9'
# Temporary build of OpenBLAS to test a fix for dynamic detection of CPU
OPENBLAS_LONG = 'v0.3.7-527-g79fd006c'  # the 0.3.7 is misleading
BASE_LOC = 'https://anaconda.org/multibuild-wheels-staging/openblas-libs'
BASEURL = f'{BASE_LOC}/{OPENBLAS_LONG}/download'
ARCHITECTURES = ['', 'windows', 'darwin', 'aarch64', 'x86_64', 'i686', 'ppc64le', 's390x']
sha256_vals = {
"openblas-v0.3.7-527-g79fd006c-win_amd64-gcc_7_1_0.zip": "7249d68c02e6b6339e06edfeab1fecddf29ee1e67a3afaa77917c320c43de840",
"openblas64_-v0.3.7-527-g79fd006c-win_amd64-gcc_7_1_0.zip": "6488e0961a5926e47242f63b63b41cfdd661e6f1d267e8e313e397cde4775c17",
"openblas-v0.3.7-527-g79fd006c-win32-gcc_7_1_0.zip": "5fb0867ca70b1d0fdbf68dd387c0211f26903d74631420e4aabb49e94aa3930d",
"openblas-v0.3.7-527-g79fd006c-macosx_10_9_x86_64-gf_1becaaa.tar.gz": "69434bd626bbc495da9ce8c36b005d140c75e3c47f94e88c764a199e820f9259",
"openblas64_-v0.3.7-527-g79fd006c-macosx_10_9_x86_64-gf_1becaaa.tar.gz": "093f6d953e3fa76a86809be67bd1f0b27656671b5a55b233169cfaa43fd63e22",
"openblas-v0.3.7-527-g79fd006c-manylinux2014_aarch64.tar.gz": "42676c69dc48cd6e412251b39da6b955a5a0e00323ddd77f9137f7c259d35319",
"openblas64_-v0.3.7-527-g79fd006c-manylinux2014_aarch64.tar.gz": "5aec167af4052cf5e9e3e416c522d9794efabf03a2aea78b9bb3adc94f0b73d8",
"openblas-v0.3.7-527-g79fd006c-manylinux2010_x86_64.tar.gz": "fa67c6cc29d4cc5c70a147c80526243239a6f95fc3feadcf83a78176cd9c526b",
"openblas64_-v0.3.7-527-g79fd006c-manylinux2010_x86_64.tar.gz": "9ad34e89a5307dcf5823bf5c020580d0559a0c155fe85b44fc219752e61852b0",
"openblas-v0.3.7-527-g79fd006c-manylinux2010_i686.tar.gz": "0b8595d316c8b7be84ab1f1d5a0c89c1b35f7c987cdaf61d441bcba7ab4c7439",
"openblas-v0.3.7-527-g79fd006c-manylinux2014_ppc64le.tar.gz": "3e1c7d6472c34e7210e3605be4bac9ddd32f613d44297dc50cf2d067e720c4a9",
"openblas64_-v0.3.7-527-g79fd006c-manylinux2014_ppc64le.tar.gz": "a0885873298e21297a04be6cb7355a585df4fa4873e436b4c16c0a18fc9073ea",
"openblas-v0.3.7-527-g79fd006c-manylinux2014_s390x.tar.gz": "79b454320817574e20499d58f05259ed35213bea0158953992b910607b17f240",
"openblas64_-v0.3.7-527-g79fd006c-manylinux2014_s390x.tar.gz": "9fddbebf5301518fc4a5d2022a61886544a0566868c8c014359a1ee6b17f2814",
"openblas-v0.3.7-527-g79fd006c-manylinux1_i686.tar.gz": "24fb92684ec4676185fff5c9340f50c3db6075948bcef760e9c715a8974e4680",
"openblas-v0.3.7-527-g79fd006c-manylinux1_x86_64.tar.gz": "ebb8236b57a1b4075fd5cdc3e9246d2900c133a42482e5e714d1e67af5d00e62",
"openblas-v0.3.7-527-g79fd006c-manylinux1_i686.tar.gz": "24fb92684ec4676185fff5c9340f50c3db6075948bcef760e9c715a8974e4680",
"openblas-v0.3.7-527-g79fd006c-manylinux1_x86_64.tar.gz": "ebb8236b57a1b4075fd5cdc3e9246d2900c133a42482e5e714d1e67af5d00e62",
"openblas-v0.3.7-527-g79fd006c-manylinux1_i686.tar.gz": "24fb92684ec4676185fff5c9340f50c3db6075948bcef760e9c715a8974e4680",
"openblas-v0.3.7-527-g79fd006c-manylinux1_x86_64.tar.gz": "ebb8236b57a1b4075fd5cdc3e9246d2900c133a42482e5e714d1e67af5d00e62",
}


IS_32BIT = sys.maxsize < 2**32

def get_arch():
    if platform.system() == 'Windows':
        ret = 'windows'
    elif platform.system() == 'Darwin':
        ret = 'darwin'
    else:
        ret = platform.uname().machine
        # What do 32 bit machines report?
        # If they are a docker, they can report x86_64
        if 'x86' in ret and IS_32BIT:
            arch = 'i686'
    assert ret in ARCHITECTURES, f'invalid architecture {ret}'
    return ret

def get_ilp64():
    if os.environ.get("NPY_USE_BLAS_ILP64", "0") == "0":
        return None
    if IS_32BIT:
        raise RuntimeError("NPY_USE_BLAS_ILP64 set on 32-bit arch")
    return "64_"

def get_manylinux(arch):
    if arch in ('x86_64', 'i686'):
        default = '2010'
    else:
        default = '2014'
    ret = os.environ.get("MB_ML_VER", default)
    # XXX For PEP 600 this can be a glibc version
    assert ret in ('1', '2010', '2014'), f'invalid MB_ML_VER {ret}'
    return ret


def download_openblas(target, arch, ilp64):
    ml_ver = get_manylinux(arch)
    fnsuffix = {None: "", "64_": "64_"}[ilp64]
    filename = ''
    headers = {'User-Agent': ('Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 ; '
                              '(KHTML, like Gecko) Chrome/41.0.2228.0 '
                              'Safari/537.3')}
    if arch in ('aarch64', 'ppc64le', 's390x', 'x86_64', 'i686'):
        suffix = f'manylinux{ml_ver}_{arch}.tar.gz'
        filename = f'{BASEURL}/openblas{fnsuffix}-{OPENBLAS_LONG}-{suffix}'
        typ = 'tar.gz'
    elif arch == 'darwin':
        suffix = 'macosx_10_9_x86_64-gf_1becaaa.tar.gz'
        filename = f'{BASEURL}/openblas{fnsuffix}-{OPENBLAS_LONG}-{suffix}'
        typ = 'tar.gz'
    elif arch == 'windows':
        if IS_32BIT:
            suffix = 'win32-gcc_7_1_0.zip'
        else:
            suffix = 'win_amd64-gcc_7_1_0.zip'
        filename = f'{BASEURL}/openblas{fnsuffix}-{OPENBLAS_LONG}-{suffix}'
        typ = 'zip'
    if not filename:
        return None
    req = Request(url=filename, headers=headers)
    response = urlopen(req)
    length = response.getheader('content-length')
    if response.status != 200:
        print(f'Could not download "{filename}"', file=sys.stderr)
        return None
    print(f"Downloading {length} from {filename}", file=sys.stderr)
    data = response.read()
    # Verify hash
    key = os.path.basename(filename)
    sha256_returned = hashlib.sha256(data).hexdigest()
    if key not in sha256_vals:
        raise ValueError(
            f'key "{key}" with hash "{sha256_returned}" not in sha256_vals')
    sha256_expected = sha256_vals[key]
    if sha256_returned != sha256_expected:
        raise ValueError(f'sha256 hash mismatch for filename {filename}')
    print("Saving to file", file=sys.stderr)
    with open(target, 'wb') as fid:
        fid.write(data)
    return typ

def setup_openblas(arch=get_arch(), ilp64=get_ilp64()):
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
    typ = download_openblas(tmp, arch, ilp64)
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
        # Strip common prefix from paths when unpacking
        prefix = os.path.commonpath(zf.getnames())
        extract_tarfile_to(zf, target, prefix)
        return target

def extract_tarfile_to(tarfileobj, target_path, archive_path):
    """Extract TarFile contents under archive_path/ to target_path/"""

    target_path = os.path.abspath(target_path)

    def get_members():
        for member in tarfileobj.getmembers():
            if archive_path:
                norm_path = os.path.normpath(member.name)
                if norm_path.startswith(archive_path + os.path.sep):
                    member.name = norm_path[len(archive_path)+1:]
                else:
                    continue

            dst_path = os.path.abspath(os.path.join(target_path, member.name))
            if os.path.commonpath([target_path, dst_path]) != target_path:
                # Path not under target_path, probably contains ../
                continue

            yield member

    tarfileobj.extractall(target_path, members=get_members())

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
            import glob
            if os.name == 'nt':
                # convention for storing / loading the DLL from
                # numpy/.libs/, if present
                try:
                    from ctypes import WinDLL
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
    def items():
        for arch in arches:
            yield arch, None
            if arch not in ('i686'):
                yield arch, '64_'

    errs = []
    for arch, ilp64 in items():
        if arch == '':
            continue

        target = None
        try:
            try:
                target = setup_openblas(arch, ilp64)
            except Exception as e:
                print(f'Could not setup {arch}:')
                print(str(e))
                errs.append(e)
                continue
            if not target:
                raise RuntimeError(f'Could not setup {arch}')
            print(target)
            if arch == 'windows':
                if not target.endswith('.a'):
                    raise RuntimeError("Not .a extracted!")
            else:
                files = glob.glob(os.path.join(target, "lib", "*.a"))
                if not files:
                    raise RuntimeError("No lib/*.a unpacked!")
        finally:
            if target is not None:
                if os.path.isfile(target):
                    os.unlink(target)
                else:
                    shutil.rmtree(target)
    if errs:
        raise errs[0]


def test_version(expected_version, ilp64=get_ilp64()):
    """
    Assert that expected OpenBLAS version is
    actually available via NumPy
    """
    import numpy
    import ctypes

    dll = ctypes.CDLL(numpy.core._multiarray_umath.__file__)
    if ilp64 == "64_":
        get_config = dll.openblas_get_config64_
    else:
        get_config = dll.openblas_get_config
    get_config.restype=ctypes.c_char_p
    res = get_config()
    print('OpenBLAS get_config returned', str(res))
    if not expected_version:
        expected_version = OPENBLAS_V
    check_str = b'OpenBLAS %s' % expected_version.encode()
    print(check_str)
    assert check_str in res, '%s not found in %s' %(expected_version, res)
    if ilp64:
        assert b"USE64BITINT" in res
    else:
        assert b"USE64BITINT" not in res

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Download and expand an OpenBLAS archive for this ' \
                    'architecture')
    parser.add_argument('--test', nargs='*', default=None,
        help='Test different architectures. "all", or any of %s' % ARCHITECTURES)
    parser.add_argument('--check_version', nargs='?', default='',
        help='Check provided OpenBLAS version string against available OpenBLAS')
    args = parser.parse_args()
    if args.check_version != '':
        test_version(args.check_version)
    elif args.test is None:
        print(setup_openblas())
    else:
        if len(args.test) == 0 or 'all' in args.test:
            test_setup(ARCHITECTURES)
        else:
            test_setup(args.test)
