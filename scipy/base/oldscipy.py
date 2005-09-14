
# Need this to change array type for low precision values
# Should really be done with method call and rtype
#  This is for backwards compatibility...
def sum(x, axis=0, rtype=None):  
    x = asarray(x)
    if x.dtypechar in ['b','h','B','H']:
        x = x.astype(intp)
    return _nx.sum(x,axis)
    
