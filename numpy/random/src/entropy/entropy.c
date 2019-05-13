#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "entropy.h"
#ifdef _WIN32
/* Windows */
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>

#include <wincrypt.h>
#else
/* Unix */
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#endif

bool entropy_getbytes(void *dest, size_t size) {
#ifndef _WIN32

  int fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0)
    return false;
  ssize_t sz = read(fd, dest, size);
  if ((sz < 0) || ((size_t)sz < size))
    return false;
  return close(fd) == 0;

#else

  HCRYPTPROV hCryptProv;
  BOOL done;

  if (!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL,
                           CRYPT_VERIFYCONTEXT) ||
      !hCryptProv) {
    return true;
  }
  done = CryptGenRandom(hCryptProv, (DWORD)size, (unsigned char *)dest);
  CryptReleaseContext(hCryptProv, 0);
  if (!done) {
    return false;
  }

  return true;
#endif
}

/* Thomas Wang 32/64 bits integer hash function */
uint32_t entropy_hash_32(uint32_t key) {
  key += ~(key << 15);
  key ^= (key >> 10);
  key += (key << 3);
  key ^= (key >> 6);
  key += ~(key << 11);
  key ^= (key >> 16);
  return key;
}

uint64_t entropy_hash_64(uint64_t key) {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

uint32_t entropy_randombytes(void) {

#ifndef _WIN32
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return entropy_hash_32(getpid()) ^ entropy_hash_32(tv.tv_sec) ^
         entropy_hash_32(tv.tv_usec) ^ entropy_hash_32(clock());
#else
  uint32_t out = 0;
  int64_t counter;
  struct _timeb tv;
  _ftime_s(&tv);
  out = entropy_hash_32(GetCurrentProcessId()) ^
        entropy_hash_32((uint32_t)tv.time) ^ entropy_hash_32(tv.millitm) ^
        entropy_hash_32(clock());
  if (QueryPerformanceCounter((LARGE_INTEGER *)&counter) != 0)
    out ^= entropy_hash_32((uint32_t)(counter & 0xffffffff));
  return out;
#endif
}

bool entropy_fallback_getbytes(void *dest, size_t size) {
  int hashes = (int)size;
  uint32_t *hash = malloc(hashes * sizeof(uint32_t));
  int i;
  for (i = 0; i < hashes; i++) {
    hash[i] = entropy_randombytes();
  }
  memcpy(dest, (void *)hash, size);
  free(hash);
  return true;
}

void entropy_fill(void *dest, size_t size) {
  bool success;
  success = entropy_getbytes(dest, size);
  if (!success) {
    entropy_fallback_getbytes(dest, size);
  }
}
