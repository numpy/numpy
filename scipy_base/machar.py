#
# Machine arithmetics - determine the parameters of the
# floating-point arithmetic system
#
# Author: Pearu Peterson, September 2003
#

__all__ = ['MachAr','machar_double','machar_single']

from Numeric import array

class MachAr:
    """Diagnosing machine parameters.

    The following attributes are available:

    ibeta  - radix in which numbers are represented
    it     - number of base-ibeta digits in the floating point mantissa M
    machep - exponent of the smallest (most negative) power of ibeta that,
             added to 1.0,
             gives something different from 1.0
    eps    - floating-point number beta**machep (floating point precision)
    negep  - exponent of the smallest power of ibeta that, substracted
             from 1.0, gives something different from 1.0
    epsneg - floating-point number beta**negep
    iexp   - number of bits in the exponent (including its sign and bias)
    minexp - smallest (most negative) power of ibeta consistent with there
             being no leading zeros in the mantissa
    xmin   - floating point number beta**minexp (the smallest (in
             magnitude) usable floating value)
    maxexp - smallest (positive) power of ibeta that causes overflow
    xmax   - (1-epsneg)* beta**maxexp (the largest (in magnitude)
             usable floating value)
    irnd   - in range(6), information on what kind of rounding is done
             in addition, and on how underflow is handled
    ngrd   - number of 'guard digits' used when truncating the product
             of two mantissas to fit the representation

    epsilon - same as eps
    tiny    - same as xmin
    huge    - same as xmax
    precision   - int(-log10(eps))
    resolution  - 10**(-precision)

    Reference:
      Numerical Recipies.
    """
    def __init__(self,
                 float_conv=float,
                 int_conv=int,
                 float_to_float=float,
                 float_to_str = lambda v:'%24.16e' % v,
                 title = 'Python floating point number',
                 ):
        """
          float_conv - convert integer to float (array)
          int_conv   - convert float (array) to integer
          float_to_float - convert float array to float
          float_to_str - convert array float to str
          title        - description of used floating point numbers
        """
        one = float_conv(1)
        two = one + one
        zero = one - one

        # Determine ibeta and beta
        a = one
        while 1:
            a = a + a
            temp = a + one
            temp1 = temp - a
            if temp1 - one != zero:
                break
        b = one
        while 1:
            b = b + b
            temp = a + b
            itemp = int_conv(temp-a)
            if itemp != 0:
                break
        ibeta = itemp
        beta = float_conv(ibeta)

        # Determine it and irnd
        it = 0
        b = one
        while 1:
            it = it + 1
            b = b * beta
            temp = b + one
            temp1 = temp - b
            if temp1 - one != zero:
                break

        betah = beta / two
        a = one
        while 1:
            a = a + a
            temp = a + one
            temp1 = temp - a
            if temp1 - one != zero:
                break
        temp = a + betah
        irnd = 0
        if temp-a != zero:
            irnd = 1
        tempa = a + beta
        temp = tempa + betah
        if irnd==0 and temp-tempa != zero:
            irnd = 2

        # Determine negep and epsneg
        negep = it + 3
        betain = one / beta
        a = one
        for i in range(negep):
            a = a * betain
        b = a
        while 1:
            temp = one - a
            if temp-one != zero:
                break
            a = a * beta
            negep = negep - 1
        negep = -negep
        epsneg = a

        # Determine machep and eps
        machep = - it - 3
        a = b

        while 1:
            temp = one + a
            if temp-one != zero:
                break
            a = a * beta
            machep = machep + 1
        eps = a

        # Determine ngrd
        ngrd = 0
        temp = one + eps
        if irnd==0 and temp*one - one != zero:
            ngrd = 1

        # Determine iexp
        i = 0
        k = 1
        z = betain
        t = one + eps
        nxres = 0
        while 1:
            y = z
            z = y*y
            a = z*one # Check here for underflow
            temp = z*t
            if a+a == zero or abs(z)>=y:
                break
            temp1 = temp * betain
            if temp1*beta == z:
                break
            i = i + 1
            k = k + k
        if ibeta != 10:
            iexp = i + 1
            mx = k + k
        else:
            iexp = 2
            iz = ibeta
            while k >= iz:
                iz = iz * ibeta
                iexp = iexp + 1
            mx = iz + iz - 1

        # Determine minexp and xmin
        while 1:
            xmin = y
            y = y * betain
            a = y * one
            temp = y * t
            if a+a != zero and abs(y) < xmin:
                k = k + 1
                temp1 = temp * betain
                if temp1*beta == y and temp != y:
                    nxres = 3
                    xmin = y
                    break
            else:
                break
        minexp = -k

        # Determine maxexp, xmax
        if mx <= k + k - 3 and ibeta != 10:
            mx = mx + mx
            iexp = iexp + 1
        maxexp = mx + minexp
        irnd = irnd + nxres
        if irnd >= 2:
            maxexp = maxexp - 2
        i = maxexp + minexp
        if ibeta == 2 and not i:
            maxexp = maxexp - 1
        if i > 20:
            maxexp = maxexp - 1
        if a != y:
            maxexp = maxexp - 2
        xmax = one - epsneg
        if xmax*one != xmax:
            xmax = one - beta*epsneg
        xmax = xmax / (xmin*beta*beta*beta)
        i = maxexp + minexp + 3
        for j in range(i):
            if ibeta==2:
                xmax = xmax + xmax
            else:
                xmax = xmax * beta

        self.ibeta = ibeta
        self.it = it
        self.negep = negep
        self.epsneg = float_to_float(epsneg)
        self._str_epsneg = float_to_str(epsneg)
        self.machep = machep
        self.eps = float_to_float(eps)
        self._str_eps = float_to_str(eps)
        self.ngrd = ngrd
        self.iexp = iexp
        self.minexp = minexp
        self.xmin = float_to_float(xmin)
        self._str_xmin = float_to_str(xmin)
        self.maxexp = maxexp
        self.xmax = float_to_float(xmax)
        self._str_xmax = float_to_str(xmax)
        self.irnd = irnd

        self.title = title
        # Commonly used parameters
        self.epsilon = self.eps
        self.tiny = self.xmin
        self.huge = self.xmax

        import math
        self.precision = int(-math.log10(float_to_float(self.eps)))
        ten = two + two + two + two + two
        resolution = ten ** (-self.precision)
        self.resolution = float_to_float(resolution)

    def __str__(self):
        return '''\
Machine parameters for %(title)s
---------------------------------------------------------------------
ibeta=%(ibeta)s it=%(it)s iexp=%(iexp)s ngrd=%(ngrd)s irnd=%(irnd)s
machep=%(machep)s     eps=%(_str_eps)s (beta**machep == epsilon)
negep =%(negep)s  epsneg=%(_str_epsneg)s (beta**epsneg)
minexp=%(minexp)s   xmin=%(_str_xmin)s (beta**minexp == tiny)
maxexp=%(maxexp)s    xmax=%(_str_xmax)s ((1-epsneg)*beta**maxexp == huge)
---------------------------------------------------------------------
''' % self.__dict__

machar_double = MachAr(lambda v:array([v],'d'),
                       lambda v:v.astype('i')[0],
                       lambda v:array(v[0],'d'),
                       lambda v:'%24.16e' % array(v[0],'d'),
                       'Numeric double precision floating point number')

machar_single = MachAr(lambda v:array([v],'f'),
                       lambda v:v.astype('i')[0],
                       lambda v:array(v[0],'f'),  #
                       lambda v:'%15.7e' % array(v[0],'f'),
                       'Numeric single precision floating point number')

if __name__ == '__main__':
    print MachAr()
    print machar_double
    print machar_single
