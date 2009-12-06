#!/usr/bin/env python3
# -*- python -*-
"""
%prog SUBMODULE...

Hack to pipe submodules of Numpy through 2to3 and build them in-place
one-by-one.

Example usage:

    python3 tools/py3tool.py testing distutils core

This will copy files to _py3k/numpy, add a dummy __init__.py and
version.py on the top level, and copy and 2to3 the files of the three
submodules.

When running py3tool again, only changed files are re-processed, which
makes the test-bugfix cycle faster.

"""
from optparse import OptionParser
import shutil
import os
import sys
import re
import subprocess

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
TEMP = os.path.normpath(os.path.join(BASE, '_py3k'))

SCRIPTS = [
    'generate_umath.py',
    'setup.py',
    'generate_numpy_api.py',
    'generate_ufunc_api.py',
]

def main():
    p = OptionParser(usage=__doc__.strip())
    p.add_option("--clean", "-c", action="store_true",
                 help="clean source directory")
    options, args = p.parse_args()

    if not args:
        p.error('no submodules given')
    else:
        dirs = ['numpy/%s' % x for x in map(os.path.basename, args)]

    # Prepare
    if not os.path.isdir(TEMP):
        os.makedirs(TEMP)

    # Set up dummy files (for building only submodules)
    dummy_files = {
        '__init__.py': 'from numpy.version import version as __version__',
        'version.py': 'version = "1.4.0.dev"'
    }

    for fn, content in dummy_files.items():
        fn = os.path.join(TEMP, 'numpy', fn)
        if not os.path.isfile(fn):
            try:
                os.makedirs(os.path.dirname(fn))
            except OSError:
                pass
            f = open(fn, 'wb+')
            f.write(content.encode('ascii'))
            f.close()

    # Environment
    pp = [os.path.abspath(TEMP)]
    def getenv():
        env = dict(os.environ)
        env.update({'PYTHONPATH': ':'.join(pp)})
        return env

    # Copy
    for d in dirs:
        src = os.path.join(BASE, d)
        dst = os.path.join(TEMP, d)

        # Run 2to3
        sync_2to3(dst=dst,
                  src=src,
                  patchfile=os.path.join(TEMP, os.path.basename(d) + '.patch'),
                  clean=options.clean)

        # Run setup.py, falling back to Pdb post-mortem on exceptions
        setup_py = os.path.join(dst, 'setup.py')
        if os.path.isfile(setup_py):
            code = """\
import pdb, sys, traceback
p = pdb.Pdb()
try:
    import __main__
    __main__.__dict__.update({
        "__name__": "__main__", "__file__": "setup.py",
        "__builtins__": __builtins__})
    fp = open("setup.py", "rb")
    try:
        exec(compile(fp.read(), "setup.py", 'exec'))
    finally:
        fp.close()
except SystemExit:
    raise
except:
    traceback.print_exc()
    t = sys.exc_info()[2]
    p.interaction(None, t)
"""
            ret = subprocess.call([sys.executable, '-c', code,
                                   'build_ext', '-i'],
                                  cwd=dst,
                                  env=getenv())
            if ret != 0:
                raise RuntimeError("Build failed.")

        # Run nosetests
        subprocess.call(['nosetests3', '-v', d], cwd=TEMP)

def walk_sync(dir1, dir2, _seen=None):
    if _seen is None:
        seen = {}
    else:
        seen = _seen

    if not dir1.endswith(os.path.sep):
        dir1 = dir1 + os.path.sep

    # Walk through stuff (which we haven't yet gone through) in dir1
    for root, dirs, files in os.walk(dir1):
        sub = root[len(dir1):]
        if sub in seen:
            dirs = [x for x in dirs if x not in seen[sub][0]]
            files = [x for x in files if x not in seen[sub][1]]
            seen[sub][0].extend(dirs)
            seen[sub][1].extend(files)
        else:
            seen[sub] = (dirs, files)
        if not dirs and not files:
            continue
        yield os.path.join(dir1, sub), os.path.join(dir2, sub), dirs, files

    if _seen is None:
        # Walk through stuff (which we haven't yet gone through) in dir2
        for root2, root1, dirs, files in walk_sync(dir2, dir1, _seen=seen):
            yield root1, root2, dirs, files

def sync_2to3(src, dst, patchfile=None, clean=False):
    to_convert = []

    for src_dir, dst_dir, dirs, files in walk_sync(src, dst):
        for fn in dirs + files:
            src_fn = os.path.join(src_dir, fn)
            dst_fn = os.path.join(dst_dir, fn)

            # remove non-existing
            if os.path.exists(dst_fn) and not os.path.exists(src_fn):
                if clean:
                    if os.path.isdir(dst_fn):
                        shutil.rmtree(dst_fn)
                    else:
                        os.unlink(dst_fn)
                continue

            # make directories
            if os.path.isdir(src_fn):
                if not os.path.isdir(dst_fn):
                    os.makedirs(dst_fn)
                continue

            dst_dir = os.path.dirname(dst_fn)
            if os.path.isfile(dst_fn) and not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)

            # don't replace up-to-date files
            try:
                if os.path.isfile(dst_fn) and \
                       os.stat(dst_fn).st_mtime >= os.stat(src_fn).st_mtime:
                    continue
            except OSError:
                pass

            # copy file
            shutil.copyfile(src_fn, dst_fn)

            # add .py files to 2to3 list
            if dst_fn.endswith('.py'):
                to_convert.append(dst_fn)

    # run 2to3
    scripts = []

    for fn in list(to_convert):
        if os.path.basename(fn) in SCRIPTS:
            scripts.append(fn)
            to_convert.remove(fn)

    if patchfile:
        p = open(patchfile, 'wb+')
    else:
        p = open(os.devnull, 'wb')

    if to_convert:
        subprocess.call(['2to3', '-w'] + to_convert, stdout=p)
    if scripts:
        subprocess.call(['2to3', '-w', '-x', 'import'] + scripts, stdout=p)
    p.close()

if __name__ == "__main__":
    main()
