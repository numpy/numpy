import numpy as np

#contribution no.1      08/07/2019
#matlab equivalent of bitget
def bitget(value,bit_no):                  # equivalent bitget in matlab!!
    binval=eval(np.binary_repr(value))de
    x = ((binval & (1<<bit_no))!=0)
    if x == True:
        return 1
    else:
        return 0
