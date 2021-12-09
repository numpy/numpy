#if (__VEC__ < 10303) || (__ARCH__ < 13)
    #error VXE2 not supported
#endif

#include <vecintrin.h>
#include <stdio.h>

int main(void) {
  int val;
  vector signed short large = { 'a', 'b', 'c', 'a', 'g', 'h', 'g', 'o' };
  vector signed short search = { 'g', 'h', 'g', 'o' };
  vector unsigned char len = { 0 };
  vector unsigned char res = vec_search_string_cc (large, search,
						      len, &val);
  if (len[7] == 0 && res[7] != 0)
     __builtin_abort ();
 
  return val;
}
