# This module should be used for functions both in numpy and scipy if
#  you want to use the numpy version if available but the scipy version
#  otherwise.
#  Usage  --- from numpy.dual import fft, inv

__all__ = ['fft','ifft','fftn','ifftn','fft2','ifft2',
           'inv','svd','solve','det','eig','eigvals','lstsq',
           'pinv','cholesky','i0']

try:
    import scipy.linalg as linpkg
except ImportError:
    import numpy.linalg as linpkg

try:
    import scipy.fftpack as fftpkg
except ImportError:
    import numpy.dft as fftpkg

try:
    from scipy.special import i0
except ImportError:
    from numpy.lib import i0

fft = fftpkg.fft
ifft = fftpkg.ifft
fftn = fftpkg.fftn
ifftn = fftpkg.ifftn
fft2 = fftpkg.fft2
ifft2 = fftpkg.ifft2

inv = linpkg.inv
svd = linpkg.svd
solve = linpkg.solve
det = linpkg.det
eig = linpkg.eig
eigvals = linpkg.eigvals
lstsq = linpkg.lstsq
pinv = linpkg.pinv
cholesky = linpkg.cholesky

