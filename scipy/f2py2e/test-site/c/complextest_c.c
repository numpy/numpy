#include <stdio.h>

typedef struct {double r,i;} complex_double;
void foo_();
void f(complex_double *c) {
  (*c).r=2;
  (*c).i=3;
  return;
}
void f2(complex_double *c) {
  (*c).r=3;
  (*c).i=2;
  return;
}
static __complex__ double g() {
  __complex__ double c;
  __real__ c=3;
  __imag__ c=2;
  return c;
}

main (){
  printf("f,g(");
  foo_(f,f2);
  printf(")\n");
/*    return 1; */
}

/*
#include <complex.h>
void f2(__complex__ double *c) {
  __real__ *c=2;
  __imag__ *c=3;
  return;
}
__complex__ double g2() {
  __complex__ double c;
  __real__ c=3;
  __imag__ c=2;
  return c;
}
 */
