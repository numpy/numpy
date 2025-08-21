#!/usr/bin/env python2.7
# WARNING! This a Python 2 script. Read README.rst for rationale.
"""
Usage: make_lite.py <wrapped_routines_file> <lapack_dir>

Typical invocation:

    make_lite.py wrapped_routines /tmp/lapack-3.x.x

Requires the following to be on the path:
 * f2c
 * patch

"""
import os
import re
import shutil
import subprocess
import sys

import clapack_scrub
import fortran

try:
    from distutils.spawn import find_executable as which  # Python 2
except ImportError:
    from shutil import which  # Python 3

# Arguments to pass to f2c. You'll always want -A for ANSI C prototypes
# Others of interest: -a to not make variables static by default
#                     -C to check array subscripts
F2C_ARGS = ['-A', '-Nx800']

# The header to add to the top of the f2c_*.c file. Note that dlamch_() calls
# will be replaced by the macros below by clapack_scrub.scrub_source()
HEADER_BLURB = '''\
/*
 * NOTE: This is generated code. Look in numpy/linalg/lapack_lite for
 *       information on remaking this file.
 */
'''

HEADER = HEADER_BLURB + '''\
#include "f2c.h"

#ifdef HAVE_CONFIG
#include "config.h"
#else
extern doublereal dlamch_(char *);
#define EPSILON dlamch_("Epsilon")
#define SAFEMINIMUM dlamch_("Safe minimum")
#define PRECISION dlamch_("Precision")
#define BASE dlamch_("Base")
#endif

extern doublereal dlapy2_(doublereal *x, doublereal *y);

/*
f2c knows the exact rules for precedence, and so omits parentheses where not
strictly necessary. Since this is generated code, we don't really care if
it's readable, and we know what is written is correct. So don't warn about
them.
*/
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
'''

class FortranRoutine:
    """Wrapper for a Fortran routine in a file.
    """
    type = 'generic'

    def __init__(self, name=None, filename=None):
        self.filename = filename
        if name is None:
            root, ext = os.path.splitext(filename)
            name = root
        self.name = name
        self._dependencies = None

    def dependencies(self):
        if self._dependencies is None:
            deps = fortran.getDependencies(self.filename)
            self._dependencies = [d.lower() for d in deps]
        return self._dependencies

    def __repr__(self):
        return f"FortranRoutine({self.name!r}, filename={self.filename!r})"

class UnknownFortranRoutine(FortranRoutine):
    """Wrapper for a Fortran routine for which the corresponding file
    is not known.
    """
    type = 'unknown'

    def __init__(self, name):
        FortranRoutine.__init__(self, name=name, filename='<unknown>')

    def dependencies(self):
        return []

class FortranLibrary:
    """Container for a bunch of Fortran routines.
    """
    def __init__(self, src_dirs):
        self._src_dirs = src_dirs
        self.names_to_routines = {}

    def _findRoutine(self, rname):
        rname = rname.lower()
        for s in self._src_dirs:
            ffilename = os.path.join(s, rname + '.f')
            if os.path.exists(ffilename):
                return self._newFortranRoutine(rname, ffilename)
        return UnknownFortranRoutine(rname)

    def _newFortranRoutine(self, rname, filename):
        return FortranRoutine(rname, filename)

    def addIgnorableRoutine(self, rname):
        """Add a routine that we don't want to consider when looking at
        dependencies.
        """
        rname = rname.lower()
        routine = UnknownFortranRoutine(rname)
        self.names_to_routines[rname] = routine

    def addRoutine(self, rname):
        """Add a routine to the library.
        """
        self.getRoutine(rname)

    def getRoutine(self, rname):
        """Get a routine from the library. Will add if it's not found.
        """
        unique = []
        rname = rname.lower()
        routine = self.names_to_routines.get(rname, unique)
        if routine is unique:
            routine = self._findRoutine(rname)
            self.names_to_routines[rname] = routine
        return routine

    def allRoutineNames(self):
        """Return the names of all the routines.
        """
        return list(self.names_to_routines.keys())

    def allRoutines(self):
        """Return all the routines.
        """
        return list(self.names_to_routines.values())

    def resolveAllDependencies(self):
        """Try to add routines to the library to satisfy all the dependencies
        for each routine in the library.

        Returns a set of routine names that have the dependencies unresolved.
        """
        done_this = set()
        last_todo = set()
        while True:
            todo = set(self.allRoutineNames()) - done_this
            if todo == last_todo:
                break
            for rn in todo:
                r = self.getRoutine(rn)
                deps = r.dependencies()
                for d in deps:
                    self.addRoutine(d)
                done_this.add(rn)
            last_todo = todo
        return todo

class LapackLibrary(FortranLibrary):
    def _newFortranRoutine(self, rname, filename):
        routine = FortranLibrary._newFortranRoutine(self, rname, filename)
        if 'blas' in filename.lower():
            routine.type = 'blas'
        elif 'install' in filename.lower():
            routine.type = 'config'
        elif rname.startswith('z'):
            routine.type = 'z_lapack'
        elif rname.startswith('c'):
            routine.type = 'c_lapack'
        elif rname.startswith('s'):
            routine.type = 's_lapack'
        elif rname.startswith('d'):
            routine.type = 'd_lapack'
        else:
            routine.type = 'lapack'
        return routine

    def allRoutinesByType(self, typename):
        routines = sorted((r.name, r) for r in self.allRoutines() if r.type == typename)
        return [a[1] for a in routines]

def printRoutineNames(desc, routines):
    print(desc)
    for r in routines:
        print(f'\t{r.name}')

def getLapackRoutines(wrapped_routines, ignores, lapack_dir):
    blas_src_dir = os.path.join(lapack_dir, 'BLAS', 'SRC')
    if not os.path.exists(blas_src_dir):
        blas_src_dir = os.path.join(lapack_dir, 'blas', 'src')
    lapack_src_dir = os.path.join(lapack_dir, 'SRC')
    if not os.path.exists(lapack_src_dir):
        lapack_src_dir = os.path.join(lapack_dir, 'src')
    install_src_dir = os.path.join(lapack_dir, 'INSTALL')
    if not os.path.exists(install_src_dir):
        install_src_dir = os.path.join(lapack_dir, 'install')

    library = LapackLibrary([install_src_dir, blas_src_dir, lapack_src_dir])

    for r in ignores:
        library.addIgnorableRoutine(r)

    for w in wrapped_routines:
        library.addRoutine(w)

    library.resolveAllDependencies()

    return library

def getWrappedRoutineNames(wrapped_routines_file):
    routines = []
    ignores = []
    with open(wrapped_routines_file) as fo:
        for line in fo:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('IGNORE:'):
                line = line[7:].strip()
                ig = line.split()
                ignores.extend(ig)
            else:
                routines.append(line)
    return routines, ignores


types = {'blas', 'lapack', 'd_lapack', 's_lapack', 'z_lapack', 'c_lapack', 'config'}

def dumpRoutineNames(library, output_dir):
    for typename in {'unknown'} | types:
        routines = library.allRoutinesByType(typename)
        filename = os.path.join(output_dir, typename + '_routines.lst')
        with open(filename, 'w') as fo:
            for r in routines:
                deps = r.dependencies()
                fo.write(f"{r.name}: {' '.join(deps)}\n")

def concatenateRoutines(routines, output_file):
    with open(output_file, 'w') as output_fo:
        for r in routines:
            with open(r.filename) as fo:
                source = fo.read()
            output_fo.write(source)

class F2CError(Exception):
    pass

def runF2C(fortran_filename, output_dir):
    fortran_filename = fortran_filename.replace('\\', '/')
    try:
        subprocess.check_call(
            ["f2c"] + F2C_ARGS + ['-d', output_dir, fortran_filename]
        )
    except subprocess.CalledProcessError:
        raise F2CError

def scrubF2CSource(c_file):
    with open(c_file) as fo:
        source = fo.read()
    source = clapack_scrub.scrubSource(source, verbose=True)
    with open(c_file, 'w') as fo:
        fo.write(HEADER)
        fo.write(source)

def ensure_executable(name):
    try:
        which(name)
    except Exception:
        raise SystemExit(name + ' not found')

def create_name_header(output_dir):
    routine_re = re.compile(r'^      (subroutine|.* function)\s+(\w+)\(.*$',
                            re.I)
    extern_re = re.compile(r'^extern [a-z]+ ([a-z0-9_]+)\(.*$')

    # BLAS/LAPACK symbols
    symbols = {'xerbla'}
    for fn in os.listdir(output_dir):
        fn = os.path.join(output_dir, fn)

        if not fn.endswith('.f'):
            continue

        with open(fn) as f:
            for line in f:
                m = routine_re.match(line)
                if m:
                    symbols.add(m.group(2).lower())

    # f2c symbols
    f2c_symbols = set()
    with open('f2c.h') as f:
        for line in f:
            m = extern_re.match(line)
            if m:
                f2c_symbols.add(m.group(1))

    with open(os.path.join(output_dir, 'lapack_lite_names.h'), 'w') as f:
        f.write(HEADER_BLURB)
        f.write(
            "/*\n"
            " * This file renames all BLAS/LAPACK and f2c symbols to avoid\n"
            " * dynamic symbol name conflicts, in cases where e.g.\n"
            " * integer sizes do not match with 'standard' ABI.\n"
            " */\n")

        # Rename BLAS/LAPACK symbols
        for name in sorted(symbols):
            f.write(f"#define {name}_ BLAS_FUNC({name})\n")

        # Rename also symbols that f2c exports itself
        f.write("\n"
                "/* Symbols exported by f2c.c */\n")
        for name in sorted(f2c_symbols):
            f.write(f"#define {name} numpy_lapack_lite_{name}\n")

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return
    # Make sure that patch and f2c are found on path
    ensure_executable('f2c')
    ensure_executable('patch')

    wrapped_routines_file = sys.argv[1]
    lapack_src_dir = sys.argv[2]
    output_dir = os.path.join(os.path.dirname(__file__), 'build')

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    wrapped_routines, ignores = getWrappedRoutineNames(wrapped_routines_file)
    library = getLapackRoutines(wrapped_routines, ignores, lapack_src_dir)

    dumpRoutineNames(library, output_dir)

    for typename in types:
        fortran_file = os.path.join(output_dir, f'f2c_{typename}.f')
        c_file = fortran_file[:-2] + '.c'
        print(f'creating {c_file} ...')
        routines = library.allRoutinesByType(typename)
        concatenateRoutines(routines, fortran_file)

        # apply the patchpatch
        patch_file = os.path.basename(fortran_file) + '.patch'
        if os.path.exists(patch_file):
            subprocess.check_call(['patch', '-u', fortran_file, patch_file])
            print(f"Patched {fortran_file}")
        try:
            runF2C(fortran_file, output_dir)
        except F2CError:
            print(f'f2c failed on {fortran_file}')
            break
        scrubF2CSource(c_file)

        # patch any changes needed to the C file
        c_patch_file = os.path.basename(c_file) + '.patch'
        if os.path.exists(c_patch_file):
            subprocess.check_call(['patch', '-u', c_file, c_patch_file])

        print()

    create_name_header(output_dir)

    for fname in os.listdir(output_dir):
        if fname.endswith('.c') or fname == 'lapack_lite_names.h':
            print('Copying ' + fname)
            shutil.copy(
                os.path.join(output_dir, fname),
                os.path.abspath(os.path.dirname(__file__)),
            )


if __name__ == '__main__':
    main()
