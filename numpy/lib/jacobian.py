import numpy as np

def jacobian(func, x, epsilon=1e-6, axis=None):
    """
    Computes the numerical Jacobian matrix of a vector-valued function using
    finite differences.

    Parameters
    ----------
    func:    Callable. The vector-valued function.
     x:       ndarray. The point to compute the Jacobian
    epsilon: float. Used for finite differences approximation.
        
    Returns
    -------

    """ 

    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    n = len(x)
    m = len(f0)
    jacobian = np.zeros((m,n))

    for i in range(n):
        x_perturb = x.copy()
        x_perturb[i] += epsilon
        f_perturb = np.asarray(func(x_perturb), dtype=float)
        jacobian[:,i] = (f_perturb - f0) / epsilon

    return jacobian
