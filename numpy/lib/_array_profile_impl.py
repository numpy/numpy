"""
Standalone test for array_profile function.
"""

import numpy as np

def array_profile(arr, return_dict=False):
    """
    Displays a comprehensive profile of a NumPy array's properties.

    This function provides a detailed overview of an array's statistical and
    structural information, including shape, dimensions, size, data type,
    and various statistical measures.

    Parameters
    ----------
    arr : array_like
        Input array or object that can be converted to an array.
    return_dict : bool, optional
        If True, returns the profile information as a dictionary instead of
        printing it. Default is False.

    Returns
    -------
    dict or None
        If `return_dict` is True, returns a dictionary containing all the
        profile information. Otherwise, prints the information and returns None.
    """
    # Convert input to numpy array if it's not already
    arr = np.asarray(arr)
    
    # Calculate basic properties
    shape = arr.shape
    dim = arr.ndim
    size = arr.size
    d_type = arr.dtype
    has_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.number) else False
    
    # Calculate statistical properties (handling NaN values)
    if size > 0 and np.issubdtype(arr.dtype, np.number):
        maxv = np.nanmax(arr)
        minv = np.nanmin(arr)
        meanv = np.nanmean(arr)
        d_sum = np.nansum(arr)
        rangev = maxv - minv
        stdv = np.nanstd(arr)
        varv = np.nanvar(arr)
        n_unique = np.unique(arr[~np.isnan(arr)]).size if has_nan else np.unique(arr).size
    else:
        # Handle non-numeric or empty arrays
        maxv = minv = meanv = d_sum = rangev = stdv = varv = n_unique = "N/A"
    
    # Create a dictionary with all the information
    profile_dict = {
        "shape": shape,
        "dimensions": dim,
        "size": size,
        "max_value": maxv,
        "min_value": minv,
        "dtype": d_type,
        "mean": meanv,
        "total_sum": d_sum,
        "range": rangev,
        "std_dev": stdv,
        "variance": varv,
        "unique_values": n_unique,
        "has_nan": has_nan
    }
    
    # Return the dictionary if requested
    if return_dict:
        return profile_dict
    
    # Otherwise print the formatted output
    print(f"""
 
       Array Profile          
=============================
 Shape        : {shape}
 Dimensions   : {dim}
 Size         : {size}
 Max Value    : {maxv}
 Min Value    : {minv}
 Dtype        : {d_type}
 Mean         : {meanv if isinstance(meanv, str) else f"{meanv:.4f}"}
 Total Sum    : {d_sum}
 Range        : {rangev}
 Std Dev      : {stdv if isinstance(stdv, str) else f"{stdv:.4f}"}
 Variance     : {varv if isinstance(varv, str) else f"{varv:.4f}"}
 Unique Values: {n_unique}
 Has NaN      : {has_nan}
 
""")
    return None
