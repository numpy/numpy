
#define WRAP_MODULE_MODDATA wrap_module_moddata__
#define DATAGETSIZE wrapdatagetsize_

extern void WRAP_MODULE_MODDATA();
extern void DATAGETSIZE();

static struct {
  int *n;
  double *data;
  void (*foo)();
  void (*foo2)();
  void (*bar)();
} moddata;

static void setup_module_moddata(int *n,double *data,void (*foo)(),void (*foo2)(),void (*bar)()) {
  int s,i;
  moddata.n = n;
  moddata.data = data;
  moddata.foo = foo;
  moddata.foo2 = foo2;
  moddata.bar = bar;
  printf("\tn=%p\n",n);
  printf("\tdata=%p\n",data);
  printf("\tfoo=%p\n",foo);
  printf("\tfoo2=%p\n",foo2);
  printf("\tbar=%p\n",bar);
  i=1;
  DATAGETSIZE(&i,&s);
  printf("\ts=%d\n",s);
}

main () {
  WRAP_MODULE_MODDATA(setup_module_moddata);
  if (1) {
    *(moddata.n) = 3;
    (*(moddata.foo2))();
    (*(moddata.bar))();
  }
  if (1) {
    int n = 4;
    (*(moddata.foo))(&n);
    WRAP_MODULE_MODDATA(setup_module_moddata); /* foo allocated data, so moddata.data needs to be reset */
    (*(moddata.bar))();
    moddata.data[1] = 7;
    (*(moddata.bar))();
  }
}






