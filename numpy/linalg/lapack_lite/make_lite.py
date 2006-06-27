#!/usr/bin/env python

import sys, os
import fortran
import clapack_scrub

try: set
except NameError:
    from sets import Set as set

# Arguments to pass to f2c. You'll always want -A for ANSI C prototypes
# Others of interest: -a to not make variables static by default
#                     -C to check array subscripts
F2C_ARGS = '-A'

# The header to add to the top of the *_lite.c file. Note that dlamch_() calls
# will be replaced by the macros below by clapack_scrub.scrub_source()
HEADER = '''\
/*
NOTE: This is generated code. Look in Misc/lapack_lite for information on
      remaking this file.
*/
#include "Numeric/f2c.h"

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
        return self.names_to_routines.keys()

    def allRoutines(self):
        """Return all the routines.
        """
        return self.names_to_routines.values()

    def resolveAllDependencies(self):
        """Try to add routines to the library to satisfy all the dependencies
        for each routine in the library.

        Returns a set of routine names that have the dependencies unresolved.
        """
        done_this = set()
        last_todo = set()
        while 1:
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
        if filename.find('BLAS') != -1:
            routine.type = 'blas'
        elif rname.startswith('z'):
            routine.type = 'zlapack'
        else:
            routine.type = 'dlapack'
        return routine

    def allRoutinesByType(self, typename):
        routines = [(r.name,r) for r in self.allRoutines() if r.type == typename]
        routines.sort()
        return [a[1] for a in routines]

def printRoutineNames(desc, routines):
    print desc
    for r in routines:
        print '\t%s' % r.name

def getLapackRoutines(wrapped_routines, ignores, lapack_dir):
    blas_src_dir = os.path.join(lapack_dir, 'BLAS', 'SRC')
    if not os.path.exists(blas_src_dir):
        blas_src_dir = os.path.join(lapack_dir, 'blas', 'src')
    lapack_src_dir = os.path.join(lapack_dir, 'SRC')
    if not os.path.exists(lapack_src_dir):
        lapack_src_dir = os.path.join(lapack_dir, 'src')
    library = LapackLibrary([blas_src_dir, lapack_src_dir])

    for r in ignores:
        library.addIgnorableRoutine(r)

    for w in wrapped_routines:
        library.addRoutine(w)

    library.resolveAllDependencies()

    return library

def getWrappedRoutineNames(wrapped_routines_file):
    fo = open(wrapped_routines_file)
    routines = []
    ignores = []
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

def dumpRoutineNames(library, output_dir):
    for typename in ['unknown', 'blas', 'dlapack', 'zlapack']:
        routines = library.allRoutinesByType(typename)
        filename = os.path.join(output_dir, typename + '_routines.lst')
        fo = open(filename, 'w')
        for r in routines:
            deps = r.dependencies()
            fo.write('%s: %s\n' % (r.name, ' '.join(deps)))
        fo.close()

def concatenateRoutines(routines, output_file):
    output_fo = open(output_file, 'w')
    for r in routines:
        fo = open(r.filename, 'r')
        source = fo.read()
        fo.close()
        output_fo.write(source)
    output_fo.close()

class F2CError(Exception):
    pass

def runF2C(fortran_filename, output_dir):
    # we're assuming no funny business that needs to be quoted for the shell
    cmd = "f2c %s -d %s %s" % (F2C_ARGS, output_dir, fortran_filename)
    rc = os.system(cmd)
    if rc != 0:
        raise F2CError

def scrubF2CSource(c_file):
    fo = open(c_file, 'r')
    source = fo.read()
    fo.close()
    source = clapack_scrub.scrubSource(source, verbose=True)
    fo = open(c_file, 'w')
    fo.write(HEADER)
    fo.write(source)
    fo.close()

def main():
    if len(sys.argv) != 4:
        print 'Usage: %s wrapped_routines_file lapack_dir output_dir' % \
              (sys.argv[0],)
        return
    wrapped_routines_file = sys.argv[1]
    lapack_src_dir = sys.argv[2]
    output_dir = sys.argv[3]

    wrapped_routines, ignores = getWrappedRoutineNames(wrapped_routines_file)
    library = getLapackRoutines(wrapped_routines, ignores, lapack_src_dir)

    dumpRoutineNames(library, output_dir)

    for typename in ['blas', 'dlapack', 'zlapack']:
        print 'creating %s_lite.c ...'  % typename
        routines = library.allRoutinesByType(typename)
        fortran_file = os.path.join(output_dir, typename+'_lite.f')
        c_file = fortran_file[:-2] + '.c'
        concatenateRoutines(routines, fortran_file)
        try:
            runF2C(fortran_file, output_dir)
        except F2CError:
            print 'f2c failed on %s' % fortran_file
            break
        scrubF2CSource(c_file)

if __name__ == '__main__':
    main()
