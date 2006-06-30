
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

typedef void* obj_type;
typedef obj_type (*rat_create_func)(void);
typedef obj_type (*rat_add_func)(obj_type, obj_type);
typedef void (*rat_show_func)(obj_type);
typedef void (*rat_set_func)(obj_type, int*, int*);

static void** rational_funcs;

static void set_f90_funcs2(int* n,...) {
  int i;
  va_list ap;
  //printf("In set_f90_funcs n=%d\n",(*n));
  va_start(ap,n);
  rational_funcs = (void*)malloc((*n)*sizeof(void*));
  for (i=0;i<(*n);i++)
    rational_funcs[i] = va_arg(ap,void*);
  va_end(ap);
}

int main(void) {
  init_f90_funcs_(set_f90_funcs2);
  rat_create_func rat_create = rational_funcs[0];
  rat_show_func rat_show = rational_funcs[1];
  rat_set_func rat_set = rational_funcs[2];
  rat_add_func rat_add = rational_funcs[3];

  obj_type obj_ptr = NULL;
  obj_ptr = (*rat_create)();
  int n=2,d=3;
  (*rat_set)(obj_ptr, &n, &d);
  (*rat_show)(obj_ptr);
  (*rat_show)((*rat_add)(obj_ptr,obj_ptr));
}
