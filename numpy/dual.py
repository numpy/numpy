# This module should be used for functions both in numpy and scipy if
#  you want to use the numpy version if available but the scipy version
#  otherwise.
#  Usage  --- from numpy.dual import fft, inv

__all__ = ['fft','ifft','fftn','ifftn','fft2','ifft2',
           'inv','svd','solve','det','eig','eigvals','lstsq',
           'pinv','cholesky','i0']

# First check to see that scipy is "new" scipy
# Perhaps we could check to see if the functions actually work in
#  the scipy that will be imported.


have_scipy = 0
try:
    import scipy
    if scipy.__version__ >= '0.4.4':
        have_scipy = 1
except ImportError:
    pass

if have_scipy:
    import scipy.linalg as linpkg
    import scipy.fftpack as fftpkg
    from scipy.special import i0
else:
    import numpy.linalg as linpkg
    import numpy.dft as fftpkg
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

