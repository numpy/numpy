#!python
""" Detect bitness (32 or 64) of Mingw-w64 gcc build target on Windows.
"""

import re
from subprocess import run, PIPE


def main():
    res = run(['gcc', '-v'], check=True, text=True, capture_output=True)
    target = re.search(r'^Target: (.*)$', res.stderr, flags=re.M).groups()[0]
    if target.startswith('i686'):
        print('32')
    elif target.startswith('x86_64'):
        print('64')
    else:
        raise RuntimeError('Could not detect Mingw-w64 bitness')


if __name__ == "__main__":
    main()
