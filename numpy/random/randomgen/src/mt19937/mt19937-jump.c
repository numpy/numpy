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

/* 32-bit function */
/* set the coefficient of the polynomial pf with v */
void set_coef(unsigned long *pf, unsigned int deg, unsigned long v) {
  if (v != 0)
    pf[deg >> 5] ^= (LSB << (deg & 0x1ful));
  else
    ;
}

void gray_code(unsigned long *h) {
  unsigned int i, j = 1, l = 1, term = LL;

  h[0] = 0;

  for (i = 1; i <= QQ; i++) {
    l = (l << 1);
    term = (term >> 1);
    for (; j < l; j++)
      h[j] = h[l - j - 1] ^ term;
  }
}

void copy_state(mt19937_state *target_state, mt19937_state *state) {
  int i;

  for (i = 0; i < N; i++)
    target_state->key[i] = state->key[i];

  target_state->pos = state->pos;
}

/* next state generating function */
void gen_next(mt19937_state *state) {
  int num;
  unsigned long y;
  static unsigned long mag02[2] = {0x0ul, MATRIX_A};

  num = state->pos;
  if (num < N - M) {
    y = (state->key[num] & UPPER_MASK) | (state->key[num + 1] & LOWER_MASK);
    state->key[num] = state->key[num + M] ^ (y >> 1) ^ mag02[y % 2];
    state->pos++;
  } else if (num < N - 1) {
    y = (state->key[num] & UPPER_MASK) | (state->key[num + 1] & LOWER_MASK);
    state->key[num] = state->key[num + (M - N)] ^ (y >> 1) ^ mag02[y % 2];
    state->pos++;
  } else if (num == N - 1) {
    y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
    state->key[N - 1] = state->key[M - 1] ^ (y >> 1) ^ mag02[y % 2];
    state->pos = 0;
  }
}

void add_state(mt19937_state *state1, mt19937_state *state2) {
  int i, pt1 = state1->pos, pt2 = state2->pos;

  if (pt2 - pt1 >= 0) {
    for (i = 0; i < N - pt2; i++)
      state1->key[i + pt1] ^= state2->key[i + pt2];
    for (; i < N - pt1; i++)
      state1->key[i + pt1] ^= state2->key[i + (pt2 - N)];
    for (; i < N; i++)
      state1->key[i + (pt1 - N)] ^= state2->key[i + (pt2 - N)];
  } else {
    for (i = 0; i < N - pt1; i++)
      state1->key[i + pt1] ^= state2->key[i + pt2];
    for (; i < N - pt2; i++)
      state1->key[i + (pt1 - N)] ^= state2->key[i + pt2];
    for (; i < N; i++)
      state1->key[i + (pt1 - N)] ^= state2->key[i + (pt2 - N)];
  }
}

/*
void gen_vec_h(mt19937_state *state, mt19937_state *vec_h,
               unsigned long *h) {
  int i;
  unsigned long k, g;
  mt19937_state v;

  gray_code(h);

  copy_state(&vec_h[0], state);

  for (i = 0; i < QQ; i++)
    gen_next(&vec_h[0]);

  for (i = 1; i < LL; i++) {
    copy_state(&v, state);
    g = h[i] ^ h[i - 1];
    for (k = 1; k < g; k = (k << 1))
      gen_next(&v);
    copy_state(&vec_h[h[i]], &vec_h[h[i - 1]]);
    add_state(&vec_h[h[i]], &v);
  }
}
*/

/* compute pf(ss) using Sliding window algorithm */
/*
void calc_state(unsigned long *pf, mt19937_state *state,
                mt19937_state *vec_h) {
  mt19937_state *temp1;
  int i = MEXP - 1, j, digit, skip = 0;

  temp1 = (mt19937_state *)calloc(1, sizeof(mt19937_state));

  while (get_coef(pf, i) == 0)
    i--;

  for (; i >= QQ; i--) {
    if (get_coef(pf, i) != 0) {
      for (j = 0; j < QQ + 1; j++)
        gen_next(temp1);
      digit = 0;
      for (j = 0; j < QQ; j++)
        digit = (digit << 1) ^ get_coef(pf, i - j - 1);
      add_state(temp1, &vec_h[digit]);
      i -= QQ;
    } else
      gen_next(temp1);
  }

  for (; i > -1; i--) {
    gen_next(temp1);
    if (get_coef(pf, i) == 1)
      add_state(temp1, state);
    else
      ;
  }

  copy_state(state, temp1);
  free(temp1);
}
*/

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

void mt19937_jump_state(mt19937_state *state, const char *jump_str) {
  unsigned long *pf;
  int i;

  pf = (unsigned long *)calloc(P_SIZE, sizeof(unsigned long));

  for (i = MEXP - 1; i > -1; i--) {
    if (jump_str[i] == '1')
      set_coef(pf, i, 1);
  }
  /* TODO: Should generate the next set and start from 0, but doesn't matter ??
   */
  if (state->pos >= N) {
    state->pos = 0;
  }

  horner1(pf, state);

  free(pf);
}
/*
void mt19937_jump(mt19937_state *state, const char *jump_str)
{
    unsigned long h[LL];
    mt19937_state vec_h[LL];
    unsigned long *pf;
    int i;

    pf = (unsigned long *)calloc(P_SIZE, sizeof(unsigned long));

    for (i = MEXP - 1; i > -1; i--)
    {
        if (jump_str[i] == '1')
            set_coef(pf, i, 1);
    }

    gen_vec_h(state, &vec_h, &h);
    calc_state(pf, state, &vec_h);

    free(pf);
}
*/