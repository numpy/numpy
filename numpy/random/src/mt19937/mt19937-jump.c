#include "mt19937-jump.h"
#include "mt19937.h"

/* 32-bits function */
/* return the i-th coefficient of the polynomial pf */
unsigned long get_coef(unsigned long *pf, unsigned int deg) {
  if ((pf[deg >> 5] & (LSB << (deg & 0x1ful))) != 0)
    return (1);
  else
    return (0);
}

void copy_state(mt19937_state *target_state, mt19937_state *state) {
  int i;

  for (i = 0; i < _MT19937_N; i++)
    target_state->key[i] = state->key[i];

  target_state->pos = state->pos;
}

/* next state generating function */
void gen_next(mt19937_state *state) {
  int num;
  unsigned long y;
  static unsigned long mag02[2] = {0x0ul, MATRIX_A};

  num = state->pos;
  if (num < _MT19937_N - _MT19937_M) {
    y = (state->key[num] & UPPER_MASK) | (state->key[num + 1] & LOWER_MASK);
    state->key[num] = state->key[num + _MT19937_M] ^ (y >> 1) ^ mag02[y % 2];
    state->pos++;
  } else if (num < _MT19937_N - 1) {
    y = (state->key[num] & UPPER_MASK) | (state->key[num + 1] & LOWER_MASK);
    state->key[num] = state->key[num + (_MT19937_M - _MT19937_N)] ^ (y >> 1) ^ mag02[y % 2];
    state->pos++;
  } else if (num == _MT19937_N - 1) {
    y = (state->key[_MT19937_N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
    state->key[_MT19937_N - 1] = state->key[_MT19937_M - 1] ^ (y >> 1) ^ mag02[y % 2];
    state->pos = 0;
  }
}

void add_state(mt19937_state *state1, mt19937_state *state2) {
  int i, pt1 = state1->pos, pt2 = state2->pos;

  if (pt2 - pt1 >= 0) {
    for (i = 0; i < _MT19937_N - pt2; i++)
      state1->key[i + pt1] ^= state2->key[i + pt2];
    for (; i < _MT19937_N - pt1; i++)
      state1->key[i + pt1] ^= state2->key[i + (pt2 - _MT19937_N)];
    for (; i < _MT19937_N; i++)
      state1->key[i + (pt1 - _MT19937_N)] ^= state2->key[i + (pt2 - _MT19937_N)];
  } else {
    for (i = 0; i < _MT19937_N - pt1; i++)
      state1->key[i + pt1] ^= state2->key[i + pt2];
    for (; i < _MT19937_N - pt2; i++)
      state1->key[i + (pt1 - _MT19937_N)] ^= state2->key[i + pt2];
    for (; i < _MT19937_N; i++)
      state1->key[i + (pt1 - _MT19937_N)] ^= state2->key[i + (pt2 - _MT19937_N)];
  }
}

/* compute pf(ss) using standard Horner method */
void horner1(unsigned long *pf, mt19937_state *state) {
  int i = MEXP - 1;
  mt19937_state *temp;

  temp = (mt19937_state *)calloc(1, sizeof(mt19937_state));

  while (get_coef(pf, i) == 0)
    i--;

  if (i > 0) {
    copy_state(temp, state);
    gen_next(temp);
    i--;
    for (; i > 0; i--) {
      if (get_coef(pf, i) != 0)
        add_state(temp, state);
      else
        ;
      gen_next(temp);
    }
    if (get_coef(pf, 0) != 0)
      add_state(temp, state);
    else
      ;
  } else if (i == 0)
    copy_state(temp, state);
  else
    ;

  copy_state(state, temp);
  free(temp);
}

void mt19937_jump_state(mt19937_state *state) {
  unsigned long *pf;
  int i;

  pf = (unsigned long *)calloc(P_SIZE, sizeof(unsigned long));
  for (i = 0; i<P_SIZE; i++) {
    pf[i] = poly_coef[i];
  }

  if (state->pos >= _MT19937_N) {
    state->pos = 0;
  }

  horner1(pf, state);

  free(pf);
}
