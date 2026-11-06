``out`` can be combined with any ``s`` in N-D FFTs
--------------------------------------------------
`numpy.fft.fftn`, `numpy.fft.ifftn`, `numpy.fft.rfftn` and their 2-D
counterparts now accept ``out`` together with an ``s`` that changes the shape
along any of the transformed axes; previously, this raised an error unless
only the last of the given axes changed. In addition, intermediate results
are now transformed in-place where possible, also if ``out`` is not given,
which avoids allocating a new array for every transformed axis.
