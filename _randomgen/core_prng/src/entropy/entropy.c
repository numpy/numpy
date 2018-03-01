/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

/* This code provides a mechanism for getting external randomness for
 * seeding purposes.  Usually, it's just a wrapper around reading
 * /dev/random.
 *
 * Alas, because not every system provides /dev/random, we need a fallback.
 * We also need to try to test whether or not to use the fallback.
 */

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
#endif

#ifndef IS_UNIX
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) ||               \
                         (defined(__APPLE__) && defined(__MACH__)))
#define IS_UNIX 1
#else
#define IS_UNIX 0
#endif
#endif

// If HAVE_DEV_RANDOM is set, we use that value, otherwise we guess
#ifndef HAVE_DEV_RANDOM
#define HAVE_DEV_RANDOM IS_UNIX
#endif

#if HAVE_DEV_RANDOM
#include <fcntl.h>
#include <unistd.h>
#endif

#if HAVE_DEV_RANDOM
/* entropy_getbytes(dest, size):
 *     Use /dev/random to get some external entropy for seeding purposes.
 *
 * Note:
 *     If reading /dev/random fails (which ought to never happen), it returns
 *     false, otherwise it returns true.  If it fails, you could instead call
 *     fallback_entropy_getbytes which always succeeds.
 */

bool entropy_getbytes(void *dest, size_t size) {
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0)
    return false;
  ssize_t sz = read(fd, dest, size);
  if ((sz < 0) || ((size_t)sz < size))
    return false;
  return close(fd) == 0;
}
#endif

#ifdef _WIN32
bool entropy_getbytes(void *dest, size_t size) {
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
}
#endif

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
  // uint32_t hash[hashes];
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
