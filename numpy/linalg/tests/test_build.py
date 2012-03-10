from subprocess import call, PIPE, Popen
import sys
import re

import numpy as np
from numpy.linalg import lapack_lite
from numpy.testing import TestCase, dec

from numpy.compat import asbytes_nested

class FindDependenciesLdd(object):
    def __init__(self):
        self.cmd = ['ldd']

        try:
            p = Popen(self.cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
        except OSError:
            raise RuntimeError("command %s cannot be run" % self.cmd)

    def get_dependencies(self, file):
        p = Popen(self.cmd + [file], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if not (p.returncode == 0):
            raise RuntimeError("Failed to check dependencies for %s" % libfile)

        return stdout

    def grep_dependencies(self, file, deps):
        stdout = self.get_dependencies(file)

        rdeps = dict([(dep, re.compile(dep)) for dep in deps])
        founds = []
        for l in stdout.splitlines():
            for k, v in rdeps.items():
                if v.search(l):
                    founds.append(k)

        return founds

class TestF77Mismatch(TestCase):
    @dec.skipif(not(sys.platform[:5] == 'linux'),
                "Skipping fortran compiler mismatch on non Linux platform")
    def test_lapack(self):
        f = FindDependenciesLdd()
        deps = f.grep_dependencies(lapack_lite.__file__,
                                   asbytes_nested(['libg2c', 'libgfortran']))
        self.assertFalse(len(deps) > 1,
"""Both g77 and gfortran runtimes linked in lapack_lite ! This is likely to
cause random crashes and wrong results. See numpy INSTALL.txt for more
information.""")
