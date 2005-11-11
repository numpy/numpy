""" Machine limits for Float32 and Float64.
"""

__all__ = ['float_epsilon','float_tiny','float_min',
           'float_max','float_precision','float_resolution',
           'double_epsilon','double_tiny','double_min','double_max',
           'double_precision','double_resolution']

from machar import machar_double, machar_single

float_epsilon = machar_single.epsilon
float_tiny = machar_single.tiny
float_max = machar_single.huge
float_min = -float_max
float_precision = machar_single.precision
float_resolution = machar_single.resolution

double_epsilon = machar_double.epsilon
double_tiny = machar_double.tiny
double_max = machar_double.huge
double_min = -double_max
double_precision = machar_double.precision
double_resolution = machar_double.resolution

if __name__ == '__main__':
    print 'float epsilon:',float_epsilon
    print 'float tiny:',float_tiny
    print 'double epsilon:',double_epsilon
    print 'double tiny:',double_tiny

