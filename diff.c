#include <stdio.h>
#include <stdlib.h>
int diff(int x, int y) {
    return abs(x - y);
}

int main(void) {
    int n1, n2;

    puts("二つの値を入力してください。");
    printf("整数1: ");
    scanf("%d", &n1);
    printf("\n整数2: ");
    scanf("%d", &n2);

    printf("\nそれらの絶対値= %d です。\n", diff(n1, n2));

    return 0;
}
