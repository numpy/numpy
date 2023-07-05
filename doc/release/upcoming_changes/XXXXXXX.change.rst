Add mean keyword to var and std function
----------------------------------------

Often when the standard deviation is needed the mean is also needed; the same holds for the variance and the mean. With the current code the mean is then calculated twice, the change introduced here for the var and std functions allows for passing in a precalculated mean as an keyword argument. 

Typical usage would be:

    mean = np.mean(A,
                   out=None,
                   axis=axis,
                   keepdims=True)

    std = np.std(A,
                 out=None,
                 axis=axis,
                 keepdims=False,
                 mean=mean)
                 
    
    var = np.var(A,
                 out=None,
                 axis=axis,
                 keepdims=False,
                 mean=mean)

Note that the mean passed into the std or var function needs to be calculated with keepdims=True. 

