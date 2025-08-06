from numpy.distutils.core import Extension

ext1 = Extension(name='scalar',
                 sources=['scalar.f'])
ext2 = Extension(name='fib2',
                 sources=['fib2.pyf', 'fib1.f'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='f2py_example',
          description="F2PY Users Guide examples",
          author="Pearu Peterson",
          author_email="pearu@cens.ioc.ee",
          ext_modules=[ext1, ext2]
          )
# End of setup_example.py
