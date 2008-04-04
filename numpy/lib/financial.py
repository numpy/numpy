# Some simple financial calculations
from numpy import log, where
import numpy as np

__all__ = ['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate', 'irr', 'npv']

_when_to_num = {'end':0, 'begin':1,
                'e':0, 'b':1,
                0:0, 1:1,
                'beginning':1,
                'start':1,
                'finish':0}

eqstr = """

    Parameters
    ---------- 
    rate : 
        Rate of interest (per period)
    nper : 
        Number of compounding periods
    pmt : 
        Payment 
    pv :
        Present value
    fv :
        Future value
    when : 
        When payments are due ('begin' (1) or 'end' (0))
                                                                   
                   nper       / (1 + rate*when) \   /        nper   \  
   fv + pv*(1+rate)    + pmt*|-------------------|*| (1+rate)    - 1 | = 0
                              \     rate        /   \               /

           fv + pv + pmt * nper = 0  (when rate == 0)
"""

def fv(rate, nper, pmt, pv, when='end'):
    """future value computed by solving the equation

    %s
    """ % eqstr
    when = _when_to_num[when]
    temp = (1+rate)**nper
    fact = where(rate==0.0, nper, (1+rate*when)*(temp-1)/rate)
    return -(pv*temp + pmt*fact)

def pmt(rate, nper, pv, fv=0, when='end'):
    """Payment computed by solving the equation

    %s
    """ % eqstr
    when = _when_to_num[when]
    temp = (1+rate)**nper
    fact = where(rate==0.0, nper, (1+rate*when)*(temp-1)/rate) 
    return -(fv + pv*temp) / fact

def nper(rate, pmt, pv, fv=0, when='end'):
    """Number of periods found by solving the equation

    %s
    """ % eqstr
    when = _when_to_num[when]
    try:
        z = pmt*(1.0+rate*when)/rate
    except ZeroDivisionError:
        z = 0.0
    A = -(fv + pv)/(pmt+0.0)
    B = (log(fv-z) - log(pv-z))/log(1.0+rate)
    return where(rate==0.0, A, B) + 0.0

def ipmt(rate, per, nper, pv, fv=0.0, when='end'):
    raise NotImplementedError


def ppmt(rate, per, nper, pv, fv=0.0, when='end'):
    raise NotImplementedError

def pv(rate, nper, pmt, fv=0.0, when='end'):
    """Number of periods found by solving the equation

    %s
    """ % eqstr
    when = _when_to_num[when]
    temp = (1+rate)**nper
    fact = where(rate == 0.0, nper, (1+rate*when)*(temp-1)/rate)
    return -(fv + pmt*fact)/temp

# Computed with Sage
#  (y + (r + 1)^n*x + p*((r + 1)^n - 1)*(r*w + 1)/r)/(n*(r + 1)^(n - 1)*x - p*((r + 1)^n - 1)*(r*w + 1)/r^2 + n*p*(r + 1)^(n - 1)*(r*w + 1)/r + p*((r + 1)^n - 1)*w/r)    

def _g_div_gp(r, n, p, x, y, w):
    t1 = (r+1)**n
    t2 = (r+1)**(n-1)
    return (y + t1*x + p*(t1 - 1)*(r*w + 1)/r)/(n*t2*x - p*(t1 - 1)*(r*w + 1)/(r**2) + n*p*t2*(r*w + 1)/r + p*(t1 - 1)*w/r)

# Use Newton's iteration until the change is less than 1e-6 
#  for all values or a maximum of 100 iterations is reached.
#  Newton's rule is 
#  r_{n+1} = r_{n} - g(r_n)/g'(r_n) 
#     where
#  g(r) is the formula 
#  g'(r) is the derivative with respect to r.
def rate(nper, pmt, pv, fv, when='end', guess=0.10, tol=1e-6, maxiter=100):
    """Number of periods found by solving the equation

    %s
    """ % eqstr
    when = _when_to_num[when]
    rn = guess
    iter = 0
    close = False
    while (iter < maxiter) and not close:
        rnp1 = rn - _g_div_gp(rn, nper, pmt, pv, fv, when)
        diff = abs(rnp1-rn)
        close = np.all(diff<tol)
        iter += 1
        rn = rnp1
    if not close:
        # Return nan's in array of the same shape as rn
        return np.nan + rn
    else:
        return rn
    
def irr(values):
    """Internal Rate of Return

    This is the rate of return that gives a net present value of 0.0

    npv(irr(values), values) == 0.0
    """
    res = np.roots(values[::-1])
    # Find the root(s) between 0 and 1
    mask = (res.imag == 0) & (res.real > 0) & (res.real <= 1)
    res = res[mask].real
    if res.size == 0:
        return np.nan
    rate = 1.0/res - 1
    if rate.size == 1:
        rate = rate.item()
    return rate
    
def npv(rate, values):
    """Net Present Value

    sum ( values_k / (1+rate)**k, k = 1..n)
    """
    values = np.asarray(values)
    return (values / (1+rate)**np.arange(1,len(values)+1)).sum(axis=0)


