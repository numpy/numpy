#include <stdint.h>

typedef struct s_splitmix64_state {
  uint64_t state;
} splitmix64_state;

static inline uint64_t splitmix64_next(splitmix64_state *state) {
  uint64_t z = (state->state += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}
