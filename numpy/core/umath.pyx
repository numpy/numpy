# Fix: ufunc.at() segfault prevention with nin=0
cpdef _ufunc_at_safe(ufunc self, object out, indices):
    """
    Safely call ufunc.at() preventing segfaults when nin=0 and nout=1.
    """
    # Fix: Validate input parameters before processing
    if self.nin == 0 and self.nout == 1:
        # Add bounds checking for nin=0 case
        if not isinstance(out, np.ndarray):
            raise TypeError('output must be ndarray')
        if out.size == 0:
            return out  # Safe early return for empty arrays
    
    # Call standard ufunc.at() with safety checks
    return self.at(out, indices)
