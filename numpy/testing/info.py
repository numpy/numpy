"""
Numpy testing tools
===================

Numpy-style unit-testing
------------------------

  NumpyTest -- Numpy tests site manager
  NumpyTestCase -- unittest.TestCase with measure method
  IgnoreException -- raise when checking disabled feature, it'll be ignored
  set_package_path -- prepend package build directory to path
  set_local_path -- prepend local directory (to tests files) to path
  restore_path -- restore path after set_package_path

Utility functions
-----------------

  jiffies -- return 1/100ths of a second that the current process has used
  memusage -- virtual memory size in bytes of the running python [linux]
  rand -- array of random numbers from given shape
  assert_equal -- assert equality
  assert_almost_equal -- assert equality with decimal tolerance
  assert_approx_equal -- assert equality with significant digits tolerance
  assert_array_equal -- assert arrays equality
  assert_array_almost_equal -- assert arrays equality with decimal tolerance
  assert_array_less -- assert arrays less-ordering

"""

global_symbols = ['ScipyTest','NumpyTest']
