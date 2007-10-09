#include <stdio.h>

#ifdef WIN32
#define FOO_EXPORT __declspec(dllexport)
#else
#define FOO_EXPORT 
#endif

int FOO_EXPORT foo(void)
{
        printf("hello\n");
		return 0;
}
