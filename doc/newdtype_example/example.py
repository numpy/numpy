from __future__ import division, absolute_import, print_function

import floatint.floatint as ff
import numpy as np

# Setting using array is hard because
#  The parser doesn't stop at tuples always
#  So, the setitem code will be called with scalars on the
#  wrong shaped array.
# But we can get a view as an ndarray of the given type:
g = np.array([1,2,3,4,5,6,7,8]).view(ff.floatint_type)

# Now, the elements will be the scalar type associated
#  with the ndarray.
print(g[0])
print(type(g[1]))

# Now, you need to register ufuncs and more arrfuncs to do useful things...
