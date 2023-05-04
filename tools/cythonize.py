#!/usr/bin/env python3
""" cythonize

Cythonize pyx files into C files as needed.

Usage: cythonize [root_dir]

Default [root_dir] is 'numpy'.

Checks pyx files to see if they have been changed relative to their
corresponding C files.  If they have, then runs cython on these files to
recreate the C files.

The script thinks that the pyx files have changed relative to the C files
by comparing hashes stored in a database file.

Simple script to invoke Cython (and Tempita) on all .pyx (.pyx.in)
files; while waiting for a proper build system. Uses file hashes to
figure out if rebuild is needed.

Originally written by Dag Sverre Seljebotn, and copied here from:

https://raw.github.com/dagss/private-scipy-refactor/cythonize/cythonize.py

Note: this script does not check any of the dependent C libraries; it only
operates on the Cython .pyx files.
"""

import os
import re
import sys
import hashlib
import subprocess

HASH_FILE = 'cythonize.dat'
DEFAULT_ROOT = 'numpy'
VENDOR = 'NumPy'

#
# Rules
#
def process_pyx(fromfile, tofile):
    flags = ['-3', '--fast-fail']
    if tofile.endswith('.cxx'):
        flags.append('--cplus')

    subprocess.check_call(
        [sys.executable, '-m', 'cython'] + flags + ["-o", tofile, fromfile])


def process_tempita_pyx(fromfile, tofile):
    import npy_tempita as tempita

    assert fromfile.endswith('.pyx.in')
    with open(fromfile) as f:
        tmpl = f.read()
    pyxcontent = tempita.sub(tmpl)
    pyxfile = fromfile[:-len('.pyx.in')] + '.pyx'
    with open(pyxfile, "w") as f:
        f.write(pyxcontent)
    process_pyx(pyxfile, tofile)


def process_tempita_pyd(fromfile, tofile):
    import npy_tempita as tempita

    assert fromfile.endswith('.pxd.in')
    assert tofile.endswith('.pxd')
    with open(fromfile) as f:
        tmpl = f.read()
    pyxcontent = tempita.sub(tmpl)
    with open(tofile, "w") as f:
        f.write(pyxcontent)

def process_tempita_pxi(fromfile, tofile):
    import npy_tempita as tempita

    assert fromfile.endswith('.pxi.in')
    assert tofile.endswith('.pxi')
    with open(fromfile) as f:
        tmpl = f.read()
    pyxcontent = tempita.sub(tmpl)
    with open(tofile, "w") as f:
        f.write(pyxcontent)

def process_tempita_pxd(fromfile, tofile):
    import npy_tempita as tempita

    assert fromfile.endswith('.pxd.in')
    assert tofile.endswith('.pxd')
    with open(fromfile) as f:
        tmpl = f.read()
    pyxcontent = tempita.sub(tmpl)
    with open(tofile, "w") as f:
        f.write(pyxcontent)

rules = {
    # fromext : function, toext
    '.pyx' : (process_pyx, '.c'),
    '.pyx.in' : (process_tempita_pyx, '.c'),
    '.pxi.in' : (process_tempita_pxi, '.pxi'),
    '.pxd.in' : (process_tempita_pxd, '.pxd'),
    '.pyd.in' : (process_tempita_pyd, '.pyd'),
    }
#
# Hash db
#
def load_hashes(filename):
    # Return { filename : (sha256 of input, sha256 of output) }
    if os.path.isfile(filename):
        hashes = {}
        with open(filename) as f:
            for line in f:
                filename, inhash, outhash = line.split()
                hashes[filename] = (inhash, outhash)
    else:
        hashes = {}
    return hashes

def save_hashes(hash_db, filename):
    with open(filename, 'w') as f:
        for key, value in sorted(hash_db.items()):
            f.write("%s %s %s\n" % (key, value[0], value[1]))

def sha256_of_file(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

#
# Main program
#

def normpath(path):
    path = path.replace(os.sep, '/')
    if path.startswith('./'):
        path = path[2:]
    return path

def get_hash(frompath, topath):
    from_hash = sha256_of_file(frompath)
    to_hash = sha256_of_file(topath) if os.path.exists(topath) else None
    return (from_hash, to_hash)

def process(path, fromfile, tofile, processor_function, hash_db):
    fullfrompath = os.path.join(path, fromfile)
    fulltopath = os.path.join(path, tofile)
    current_hash = get_hash(fullfrompath, fulltopath)
    if current_hash == hash_db.get(normpath(fullfrompath), None):
        print(f'{fullfrompath} has not changed')
        return

    orig_cwd = os.getcwd()
    try:
        os.chdir(path)
        print(f'Processing {fullfrompath}')
        processor_function(fromfile, tofile)
    finally:
        os.chdir(orig_cwd)
    # changed target file, recompute hash
    current_hash = get_hash(fullfrompath, fulltopath)
    # store hash in db
    hash_db[normpath(fullfrompath)] = current_hash


def find_process_files(root_dir):
    hash_db = load_hashes(HASH_FILE)
    files  = [x for x in os.listdir(root_dir) if not os.path.isdir(x)]
    # .pxi or .pxi.in files are most likely dependencies for
    # .pyx files, so we need to process them first
    files.sort(key=lambda name: (name.endswith('.pxi') or
                                 name.endswith('.pxi.in') or
                                 name.endswith('.pxd.in')),
               reverse=True)

    for filename in files:
        in_file = os.path.join(root_dir, filename + ".in")
        for fromext, value in rules.items():
            if filename.endswith(fromext):
                if not value:
                    break
                function, toext = value
                if toext == '.c':
                    with open(os.path.join(root_dir, filename), 'rb') as f:
                        data = f.read()
                        m = re.search(br"^\s*#\s*distutils:\s*language\s*=\s*c\+\+\s*$", data, re.I|re.M)
                        if m:
                            toext = ".cxx"
                fromfile = filename
                tofile = filename[:-len(fromext)] + toext
                process(root_dir, fromfile, tofile, function, hash_db)
                save_hashes(hash_db, HASH_FILE)
                break

def main():
    try:
        root_dir = sys.argv[1]
    except IndexError:
        root_dir = DEFAULT_ROOT
    find_process_files(root_dir)


if __name__ == '__main__':
    main()
