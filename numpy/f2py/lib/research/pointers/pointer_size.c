#include <stdio.h>

#define settypeinfo settypeinfo_

void settypeinfo(int*);

int main(int argc, char* argv[])
{
  int i;
  int a[512];
  for(i=0;i<512;i++) a[i] = 0;

  settypeinfo(a);  

  if (a[0] != 333331) {
    printf("FAILED, start flag is incorrect = %d\n", a[0]);
    return 1;
  }

  for (i = 0; i < 512; i++) {
    if (a[i] == 333332) {
      printf("SUCCESSFULLY found Fortran pointer length of %d bytes\n",
	     (i-1)*sizeof(int));
      return 0;
    }
  }

  printf("FAILED to find end flag\n");
  return 1;
}
