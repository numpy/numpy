#include <stdint.h>

static inline uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (state[0] += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}
