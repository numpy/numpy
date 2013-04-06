from __future__ import division, absolute_import, print_function

from scipy import weave
from numpy import rand, zeros_like

def example1(a):
    if not isinstance(a, list):
        raise ValueError("argument must be a list")
    code = r"""
           int i;
           py::tuple results(2);
           for (i=0; i<a.length(); i++) {
                  a[i] = i;
           }
           results[0] = 3.0;
           results[1] = 4.0;
           return_val = results;
           """
    return weave.inline(code,['a'])

def arr(a):
    if a.ndim != 2:
        raise ValueError("a must be 2-d")
    code = r"""
    int i,j;
    for(i=1;i<Na[0]-1;i++) {
        for(j=1;j<Na[1]-1;j++) {
            B2(i,j) = A2(i,j) + A2(i-1,j)*0.5 +
                      A2(i+1,j)*0.5 + A2(i,j-1)*0.5
                      + A2(i,j+1)*0.5
                      + A2(i-1,j-1)*0.25
                      + A2(i-1,j+1)*0.25
                      + A2(i+1,j-1)*0.25
                      + A2(i+1,j+1)*0.25;
        }
    }
    """
    b = zeros_like(a)
    weave.inline(code,['a','b'])
    return b

a = [None]*10
print(example1(a))
print(a)

a = rand(512,512)
b = arr(a)

h = [[0.25,0.5,0.25],[0.5,1,0.5],[0.25,0.5,0.25]]
import scipy.signal as ss
b2 = ss.convolve(h,a,'same')
