#include <time.h>
#include <stdio.h>

void Ramp(double* result, int size, double start, double end)
{
    double step = (end-start)/(size-1);
    double val = start;
    int i;
    for (i = 0; i < size; i++)
    {
        *result++ = val;
        val += step;           
    }    
}

void main()
{
    double array[10000];
    int i;
    clock_t t1, t2;
    float seconds;
    t1 = clock();
    for (i = 0; i < 10000; i++)
        Ramp(array, 10000, 0.0, 1.0);
    t2 = clock();
    seconds = (float)(t2-t1)/CLOCKS_PER_SEC; 
    printf("c version (seconds): %f\n", seconds);
    printf("array[500]: %f\n", array[500]);
}