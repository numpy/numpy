.. currentmodule:: numpy

.. py:function:: show_config()
   
   Print information about various resources (libraries, library directories,
   include directories, etc.) in the system on which NumPy was built.

   See Also
   --------
   get_include : Returns the include directory containing NumPy \\*.h header files.

   Notes
   -----
   Classes specifying the information to be printed are defined
   in the `numpy.distutils.system_info` module.

   Information may include:

   * ``language``: language used to write the libraries (mostly C or f77)
   * ``libraries``: names of libraries found in the system
   * ``library_dirs``: directories containing the libraries
   * ``include_dirs``: directories containing library header files
   * ``src_dirs``: directories containing library source files
   * ``define_macros``: preprocessor macros used by ``distutils.setup``

   Examples
   --------
   >>> np.show_config()
   blas_opt_info:
      language = c
      define_macros = [('HAVE_CBLAS', None)]
      libraries = ['openblas', 'openblas']
      library_dirs = ['/usr/local/lib']