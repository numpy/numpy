/*
  FNV-1a hash algorithm implementation
  Based on the implementation from:
  https://github.com/lcn2/fnv
*/

#ifndef NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_FNV_H_
#define NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_FNV_H_


/*
  Compute a size_t FNV-1a hash of the given data
  This will use 32-bit or 64-bit hash depending on the size of size_t

  Parameters:
  -----------
  buf - pointer to the data to be hashed
  len - length of the data in bytes

  Returns:
  -----------
  size_t hash value
*/
size_t npy_fnv1a(const void *buf, size_t len);

#endif  // NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_FNV_H_
