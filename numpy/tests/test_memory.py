from __future__ import division, absolute_import, print_function

import sys
import subprocess
import itertools


class TestMemory(object):
    def test_memory_leak(self):
        # see #10157
        output = subprocess.check_output([
            'valgrind',
            sys.executable,
            '-c', 'import numpy'], stderr=subprocess.STDOUT)

        leak_summary = '\n'.join(
            itertools.dropwhile(lambda x: not x.endswith('LEAK SUMMARY:'),
                                output.decode().splitlines()))

        if 'definitely lost: 0 bytes in' not in leak_summary:
            raise ValueError(leak_summary)
