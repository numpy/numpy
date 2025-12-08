#!/usr/bin/env python3
import os
import sys
import tempfile


def run():
    _path = os.getcwd()
    os.chdir(tempfile.gettempdir())
    print('------')
    print(f'os.name={os.name!r}')
    print('------')
    print(f'sys.platform={sys.platform!r}')
    print('------')
    print('sys.version:')
    print(sys.version)
    print('------')
    print('sys.prefix:')
    print(sys.prefix)
    print('------')
    print(f"sys.path={':'.join(sys.path)!r}")
    print('------')

    try:
        import numpy
        has_numpy = 1
    except ImportError as e:
        print('Failed to import numpy:', e)
        has_numpy = 0

    try:
        from numpy.f2py import f2py2e
        has_f2py2e = 1
    except ImportError as e:
        print('Failed to import f2py2e:', e)
        has_f2py2e = 0

    if has_numpy:
        try:
            print(f'Found numpy version {numpy.__version__!r} in {numpy.__file__}')
        except Exception as msg:
            print('error:', msg)
            print('------')

    if has_f2py2e:
        try:
            print('Found f2py2e version %r in %s' %
                  (f2py2e.__version__.version, f2py2e.__file__))
        except Exception as msg:
            print('error:', msg)
            print('------')

    os.chdir(_path)


if __name__ == "__main__":
    run()
