import Numeric


def fftshift(x,axes=None):
    """ Shift the result of an FFT operation.

        Return a shifted version of x (useful for obtaining centered spectra).
        This function swaps "half-spaces" for all axes listed (defaults to all)
    """
    ndim = len(x.shape)
    if axes == None:
        axes = range(ndim)
    y = x
    for k in axes:
        N = x.shape[k]
        p2 = int(Numeric.ceil(N/2.0))
        mylist = Numeric.concatenate((Numeric.arange(p2,N),Numeric.arange(p2)))
        y = Numeric.take(y,mylist,k)
    return y

def ifftshift(x,axes=None):
    """ Reverse the effect of fftshift.
    """
    ndim = len(x.shape)
    if axes == None:
        axes = range(ndim)
    y = x
    for k in axes:
        N = x.shape[k]
        p2 = int(Numeric.floor(N/2.0))
        mylist = Numeric.concatenate((Numeric.arange(p2,N),Numeric.arange(p2)))
        y = Numeric.take(y,mylist,k)
    return y

def fftfreq(N,sample=1.0):
    """ FFT sample frequencies
    
        Return the frequency bins in cycles/unit (with zero at the start) given a
        window length N and a sample spacing.
    """
    N = int(N)
    sample = float(sample)
    return Numeric.concatenate((Numeric.arange(0,(N-1)/2+1,1,'d'),Numeric.arange(-(N-1)/2,0,1,'d')))/N/sample

def cont_ft(gn,fr,delta=1.0,n=None):
    """ Compute the (scaled) DFT of gn at frequencies fr.

        If the gn are alias-free samples of a continuous time function then the
        correct value for the spacing, delta, will give the properly scaled,
        continuous Fourier spectrum.
    
        The DFT is obtained when delta=1.0
    """
    if n is None:
        n = Numeric.arange(len(gn))
    dT = delta
    trans_kernel = Numeric.exp(-2j*Numeric.pi*fr[:,Numeric.NewAxis]*dT*n)
    return dT*Numeric.dot(trans_kernel,gn)

#-----------------------------------------------------------------------------
# Test Routines
#-----------------------------------------------------------------------------

def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)

if __name__ == '__main__':
    print 'float epsilon:',float_epsilon
    print 'float tiny:',float_tiny
    print 'double epsilon:',double_epsilon
    print 'double tiny:',double_tiny
