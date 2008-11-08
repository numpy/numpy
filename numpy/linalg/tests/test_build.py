from subprocess import call, PIPE, Popen
import sys
import re

import numpy as np
from numpy.testing import TestCase

class FindDependenciesLdd:
    def __init__(self):
        self.cmd = ['ldd']

        try:
            st = call(self.cmd, stdout=PIPE, stderr=PIPE)
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
