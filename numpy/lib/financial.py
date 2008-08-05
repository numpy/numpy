# Some simple financial calculations
#  patterned after spreadsheet computations.

# There is some complexity in each function
#  so that the functions behave like ufuncs with
#  broadcasting and being able to be called with scalars
#  or arrays (or other sequences).
import numpy as np

__all__ = ['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate',
           'irr', 'npv', 'mirr']

_when_to_num = {'end':0, 'begin':1,
                'e':0, 'b':1,
                0:0, 1:1,
                'beginning':1,
                'start':1,
                'finish':0}

eqstr = """

                  nper       / (1 + rate*when) \   /        nper   \
  fv + pv*(1+rate)    + pmt*|-------------------|*| (1+rate)    - 1 | = 0
                             \     rate        /   \               /

       fv + pv + pmt * nper = 0  (when rate == 0)

where (all can be scalars or sequences)

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

"""

def _convert_when(when):
    try:
        return _when_to_num[when]
    except KeyError:
        return [_when_to_num[x] for x in when]


def fv(rate, nper, pmt, pv, when='end'):
    """
    Compute the future value.

    Parameters
    ----------
    rate : array-like
        Rate of interest (per period)
    nper : array-like
        Number of compounding periods
    pmt : array-like
        Payment
    pv : array-like
        Present value
    when : array-like
        When payments are due ('begin' (1) or 'end' (0))

    Notes
    -----
    The future value is computed by solving the equation::

      fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) == 0

    or, when ``rate == 0``::

      fv + pv + pmt * nper == 0

    Examples
    --------
    What is the future value after 10 years of saving $100 now, with
    an additional monthly savings of $100.  Assume the interest rate is
    5% (annually) compounded monthly?

    >>> np.fv(0.05/12, 10*12, -100, -100)
    15692.928894335748

    By convention, the negative sign represents cash flow out (i.e. money not
    available today).  Thus, saving $100 a month at 5% annual interest leads
    to $15,692.93 available to spend in 10 years.

    """
    when = _convert_when(when)
    rate, nper, pmt, pv, when = map(np.asarray, [rate, nper, pmt, pv, when])
    temp = (1+rate)**nper
    miter = np.broadcast(rate, nper, pmt, pv, when)
    zer = np.zeros(miter.shape)
    fact = np.where(rate==zer, nper+zer, (1+rate*when)*(temp-1)/rate+zer)
    return -(pv*temp + pmt*fact)
fv.__doc__ += eqstr + """
Example
--------

What is the future value after 10 years of saving $100 now, with
  an additional monthly savings of $100.  Assume the interest rate is
  5% (annually) compounded monthly?

>>> np.fv(0.05/12, 10*12, -100, -100)
15692.928894335748

By convention, the negative sign represents cash flow out (i.e. money not
  available today).  Thus, saving $100 a month at 5% annual interest leads
  to $15,692.93 available to spend in 10 years.
"""

def pmt(rate, nper, pv, fv=0, when='end'):
    """
    Compute the payment.

    Parameters
    ----------
    rate : array-like
        Rate of interest (per period)
    nper : array-like
        Number of compounding periods
    pv : array-like
        Present value
    fv : array-like
        Future value
    when : array-like
        When payments are due ('begin' (1) or 'end' (0))

    Notes
    -----
    The payment ``pmt`` is computed by solving the equation::

      fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) == 0

    or, when ``rate == 0``::

      fv + pv + pmt * nper == 0

    Examples
    --------
    What would the monthly payment need to be to pay off a $200,000 loan in 15
    years at an annual interest rate of 7.5%?

    >>> np.pmt(0.075/12, 12*15, 200000)
    -1854.0247200054619

    In order to pay-off (i.e. have a future-value of 0) the $200,000 obtained
    today, a monthly payment of $1,854.02 would be required.

    """
    when = _convert_when(when)
    rate, nper, pv, fv, when = map(np.asarray, [rate, nper, pv, fv, when])
    temp = (1+rate)**nper
    miter = np.broadcast(rate, nper, pv, fv, when)
    zer = np.zeros(miter.shape)
    fact = np.where(rate==zer, nper+zer, (1+rate*when)*(temp-1)/rate+zer)
    return -(fv + pv*temp) / fact
pmt.__doc__ += eqstr + """
Examples
--------

What would the monthly payment need to be to pay off a $200,000 loan in 15
  years at an annual interest rate of 7.5%?

>>> np.pmt(0.075/12, 12*15, 200000)
-1854.0247200054619

In order to pay-off (i.e. have a future-value of 0) the $200,000 obtained
  today, a monthly payment of $1,854.02 would be required.
"""

def nper(rate, pmt, pv, fv=0, when='end'):
    """
    Compute the number of periods.

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    pmt : array_like
        Payment
    pv : array_like
        Present value
    fv : array_like
        Future value
    when : array_like
        When payments are due ('begin' (1) or 'end' (0))

    Notes
    -----
    The number of periods ``nper`` is computed by solving the equation::

      fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) == 0

    or, when ``rate == 0``::

      fv + pv + pmt * nper == 0

    Examples
    --------
    If you only had $150 to spend as payment, how long would it take to pay-off
    a loan of $8,000 at 7% annual interest?

    >>> np.nper(0.07/12, -150, 8000)
    64.073348770661852

    So, over 64 months would be required to pay off the loan.

    The same analysis could be done with several different interest rates and/or
    payments and/or total amounts to produce an entire table.

    >>> np.nper(*(np.ogrid[0.06/12:0.071/12:0.01/12, -200:-99:100, 6000:7001:1000]))
    array([[[ 32.58497782,  38.57048452],
            [ 71.51317802,  86.37179563]],
    <BLANKLINE>
           [[ 33.07413144,  39.26244268],
            [ 74.06368256,  90.22989997]]])

    """
    when = _convert_when(when)
    rate, pmt, pv, fv, when = map(np.asarray, [rate, pmt, pv, fv, when])
    try:
        z = pmt*(1.0+rate*when)/rate
    except ZeroDivisionError:
        z = 0.0
    A = -(fv + pv)/(pmt+0.0)
    B = np.log((-fv+z) / (pv+z))/np.log(1.0+rate)
    miter = np.broadcast(rate, pmt, pv, fv, when)
    zer = np.zeros(miter.shape)
    return np.where(rate==zer, A+zer, B+zer) + 0.0
nper.__doc__ += eqstr + """
Examples
--------

If you only had $150 to spend as payment, how long would it take to pay-off
  a loan of $8,000 at 7% annual interest?

>>> np.nper(0.07/12, -150, 8000)
64.073348770661852

So, over 64 months would be required to pay off the loan.

The same analysis could be done with several different interest rates and/or
    payments and/or total amounts to produce an entire table.

>>> np.nper(*(np.ogrid[0.06/12:0.071/12:0.01/12, -200:-99:100, 6000:7001:1000]))
array([[[ 32.58497782,  38.57048452],
        [ 71.51317802,  86.37179563]],

       [[ 33.07413144,  39.26244268],
        [ 74.06368256,  90.22989997]]])
"""

def ipmt(rate, per, nper, pv, fv=0.0, when='end'):
    """
    Not implemented.

    """
    total = pmt(rate, nper, pv, fv, when)
    # Now, compute the nth step in the amortization
    raise NotImplementedError

def ppmt(rate, per, nper, pv, fv=0.0, when='end'):
    total = pmt(rate, nper, pv, fv, when)
    return total - ipmt(rate, per, nper, pv, fv, when)

def pv(rate, nper, pmt, fv=0.0, when='end'):
    """
    Compute the present value.

    Parameters
    ----------
    rate : array-like
        Rate of interest (per period)
    nper : array-like
        Number of compounding periods
    pmt : array-like
        Payment
    fv : array-like
        Future value
    when : array-like
        When payments are due ('begin' (1) or 'end' (0))

    Notes
    -----
    The present value ``pv`` is computed by solving the equation::

     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) = 0

    or, when ``rate = 0``::

     fv + pv + pmt * nper = 0

    """
    when = _convert_when(when)
    rate, nper, pmt, fv, when = map(np.asarray, [rate, nper, pmt, fv, when])
    temp = (1+rate)**nper
    miter = np.broadcast(rate, nper, pmt, fv, when)
    zer = np.zeros(miter.shape)
    fact = np.where(rate == zer, nper+zer, (1+rate*when)*(temp-1)/rate+zer)
    return -(fv + pmt*fact)/temp
pv.__doc__ += eqstr

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
    """
    Compute the rate of interest per period.

    Parameters
    ----------
    nper : array_like
        Number of compounding periods
    pmt : array_like
        Payment
    pv : array_like
        Present value
    fv : array_like
        Future value
    when : array_like, optional
        When payments are due ('begin' (1) or 'end' (0))
    guess : float, optional
        Starting guess for solving the rate of interest
    tol : float, optional
        Required tolerance for the solution
    maxiter : int, optional
        Maximum iterations in finding the solution

    Notes
    -----
    The rate of interest ``rate`` is computed by solving the equation::

     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) = 0

    or, if ``rate = 0``::

     fv + pv + pmt * nper = 0

    """
    when = _convert_when(when)
    nper, pmt, pv, fv, when = map(np.asarray, [nper, pmt, pv, fv, when])
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
rate.__doc__ += eqstr

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

def mirr(values, finance_rate, reinvest_rate):
    """Modified internal rate of return

    Parameters
    ----------
    values:
        Cash flows (must contain at least one positive and one negative value)
        or nan is returned.
    finance_rate :
        Interest rate paid on the cash flows
    reinvest_rate :
        Interest rate received on the cash flows upon reinvestment
    """

    values = np.asarray(values)
    pos = values > 0
    neg = values < 0
    if not (pos.size > 0 and neg.size > 0):
        return np.nan

    n = pos.size + neg.size
    numer = -npv(reinvest_rate, values[pos])*((1+reinvest_rate)**n)
    denom = npv(finance_rate, values[neg])*(1+finance_rate)
    return (numer / denom)**(1.0/(n-1)) - 1
