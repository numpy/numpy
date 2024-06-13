#include <stdio.h>

int diff(int x, int y) {
    return x - y;
}

int main(void) {
    int n1, n2;

    puts("二つの値を入力してください。");
    printf("整数1: ");
    scanf("%d", &n1);
    printf("\n整数2: ");
    scanf("%d", &n2);

    printf("\nそれらの差は %d です。\n", diff(n1, n2));

    return 0;
}
