/* File : example.c */

double My_variable = 3.0;

/* Compute factorial of n */
int  fact(int n) {
    if (n <= 1) return 1;
    else return n*fact(n-1);
}

/* Compute n mod m */
int my_mod(int n, int m) {
    return(n % m);
}
