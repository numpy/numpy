from __future__ import division, absolute_import, print_function

from scipy import weave, zeros_like

def filter(a):
    if a.ndim != 2:
        raise ValueError("a must be 2-d")
    code = r"""
    int i,j;
    for(i=1;i<Na[0]-1;i++) {
        for(j=1;j<Na[1]-1;j++) {
            B2(i,j) = A2(i,j) + (A2(i-1,j) +
                      A2(i+1,j) + A2(i,j-1)
                      + A2(i,j+1))*0.5
                      + (A2(i-1,j-1)
                      + A2(i-1,j+1)
                      + A2(i+1,j-1)
                      + A2(i+1,j+1))*0.25;
        }
    }
    """
    b = zeros_like(a)
    weave.inline(code,['a','b'])
    return b
