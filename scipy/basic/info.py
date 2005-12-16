"""\
Basic tools
===========

linalg - lite version of scipy.linalg
fft    - lite version of scipy.fftpack
helper - lite version of scipy.linalg.helper

"""

depends = ['scipy.base']
global_symbols = ['fft','ifft', 
                  'linalg','fftpack']
