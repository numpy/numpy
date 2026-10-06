#ifndef NUMPY_CORE_INCLUDE_NUMPY_RANDOM_BITGEN_BULK_H_
#define NUMPY_CORE_INCLUDE_NUMPY_RANDOM_BITGEN_BULK_H_

#pragma once
#include <stddef.h>
#include <stdint.h>

#define BITGEN_BULK_CAPSULE_NAME "BitGeneratorBulkV1"
#define BITGEN_BULK_ABI_VERSION 1

#define BITGEN_BULK_UINT32 (1ULL << 0)
#define BITGEN_BULK_UINT64 (1ULL << 1)
#define BITGEN_BULK_DOUBLE (1ULL << 2)

typedef struct bitgen_bulk_v1 {
  uint32_t abi_version;
  size_t struct_size;
  uint64_t capabilities;
  void (*fill_uint32)(void *state, size_t count, uint32_t *out);
  void (*fill_uint64)(void *state, size_t count, uint64_t *out);
  void (*fill_double)(void *state, size_t count, double *out);
} bitgen_bulk_v1;

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_RANDOM_BITGEN_BULK_H_ */
