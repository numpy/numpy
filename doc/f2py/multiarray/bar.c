
#include <stdio.h>

void bar(int *a,int m,int n) {
  int i,j;
  printf("C:");
  printf("m=%d, n=%d\n",m,n);
  for (i=0;i<m;++i) {
    printf("Row %d:\n",i+1);
    for (j=0;j<n;++j)
      printf("a(i=%d,j=%d)=%d\n",i,j,a[n*i+j]);
  }
  if (m*n)
    a[0] = 7777;
}
